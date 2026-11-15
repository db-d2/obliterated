"""Score public checkpoints from the registry with every detector, against their declared base.

    uv run python scripts/detect_checkpoints.py --family qwen2.5 --params-max 2
    uv run python scripts/detect_checkpoints.py --label benign_finetune --out runs/detect/benign.jsonl

Per checkpoint: weight-diff spectrum (streamed from safetensors, so the model never has to fit
on the GPU twice), activation gap and behaviour shift (base loaded first, then the candidate,
never both at once). Prompts and filtering match scripts/sweep_frontier.py, so the scores of
benign fine-tunes are usable as thresholds for the sweep (--negatives there).

Candidates are scored under the *base* model's tokenizer and chat template unless
--own-tokenizer is given: the question is what the weights do, and a fine-tune's own default
system prompt would otherwise count as a behaviour shift.

Rows carry the registry label, so AUROC per detector (abliterated against benign fine-tunes) is
printed at the end when both are present. Reruns skip repos already in --out. Every changed
writer matrix's spectrum is saved under <out dir>/spectra/, so a new aggregate never needs the
weights again. --rescore-weights recomputes only the weight fields of the rows already in --out
(no model is loaded) and refreshes their labels from the registry. Weights are downloaded for
analysis only.

With --record a row also carries weight_refusal_alignment (and _med, weight_vocab_alignment, and
weight_refusal_alignment_any, the largest of them all): how much of the base's refusal direction
lies in the span of a changed matrix's top factors. It finds an abliteration under a dense
change that hides it from the energy shares.
--add-alignment computes it for rows scored before it existed, from their records alone.

When the disk cannot hold a whole set (thirty 8B fine-tunes are about 480 GB), score each
checkpoint and delete it:

    uv run python scripts/detect_checkpoints.py --family llama3.1 --evict --out runs/detect/llama31.jsonl
    uv run python scripts/detect_checkpoints.py --check --out runs/detect/llama31.jsonl

--evict removes a checkpoint from the Hugging Face cache once it has been scored AND what was
saved for it has been read back from disk and validated (obliterated.record.validate). A
checkpoint that failed, or whose record has any problem, stays on disk and is listed at the
end. Bases are never removed. The next checkpoint downloads while the current one is scored
(--prefetch), so two are on disk at once. Since the weights will not be there afterwards, --evict turns on
the two things that keep a reported number traceable:

--completions  answers to JailbreakBench, written where judge_checkpoints.py looks for them
               (run that with --no-generate). A reasoning model whose answers stop inside the
               reasoning is generated again with --think-tokens.
--record       under <out dir>/record/: the top singular values and residual-stream directions
               of every changed writer matrix, of the embedding and of the output head; the
               base's refusal directions; the per-prompt measurements each score is a mean of;
               and which revision of the checkpoint and its base was scored, file by file, with
               the settings, library versions and a copy of the model card. The prompts used
               go to <out dir>/prompts/.

--check validates the record of every row already in --out, without loading anything; with
--evict it then deletes the checkpoints that validated (for a run scored without --evict).
Checkpoints and bases are loaded from the exact snapshot that is diffed and recorded.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import shutil
import subprocess
import threading
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import ModelCard, constants, hf_hub_download
from safetensors.torch import load_file

from obliterated import completions, data, detect, direction, record, sweep
from obliterated.models import load, unload
from obliterated.refusal import last_logits
from obliterated.registry import bases, evict, fetch, load_registry

N_FACTORS = 8


def reference(base_repo: str, prompts, args) -> dict:
    """Everything the detectors need from the clean base, computed once per base."""
    h_tr, s_tr, h_val, s_val = prompts
    snapshot = fetch(base_repo)
    base = load(str(snapshot), device=args.device)
    bs = args.batch_size
    h_tr, s_tr = direction.filter_by_baseline(base, h_tr, s_tr, bs)
    h_val, s_val = direction.filter_by_baseline(base, h_val, s_val, bs)
    ref_dirs = detect.reference_directions(base, h_tr, s_tr, batch_size=bs)
    proj = detect.projections(base, ref_dirs, h_val, s_val, batch_size=bs)
    ref = {
        "tokenizer": base.tokenizer,
        "ref_dirs": ref_dirs,
        "proj": proj,
        "sep": detect.separation_of(*proj),
        "snapshot": record.snapshot_info(snapshot),
        "dtype": str(next(base.model.parameters()).dtype),
        "device": str(base.device),
        "logits": last_logits(base, h_val, bs),
        "refusal_ids": base.refusal_token_ids,
        "n_layers": base.n_layers,
        "h_val": h_val,
        "s_val": s_val,
    }
    answers = completions.path(args.out, base_repo)
    if args.completions and not answers.exists():
        completions.record(base, base_repo, completions.prompts(args.n_jbb), answers, "base",
                           args.max_new_tokens, bs, args.think_tokens)
    unload(base)
    if args.record:
        ref["prompts"] = record.save_prompts(args.out, base_repo, {
            "base_revision": ref["snapshot"]["revision"], "source": "arditi", "n_train": args.n_train, "n_val": args.n_val,
            "note": "after filtering: harmful prompts the base refuses, harmless ones it answers",
            "harmful_train": h_tr, "harmless_train": s_tr, "harmful_val": h_val, "harmless_val": s_val,
            "jailbreakbench": completions.prompts(args.n_jbb) if args.completions else None,
        })
    return ref


@functools.cache
def versions() -> dict:
    """The code and libraries behind a score. Taken once per process: a run lasts days, and the
    sources on disk when a later checkpoint finishes need not be the ones that were loaded."""
    root = Path(__file__).resolve().parents[1]

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    # The commit alone does not identify the code while changes are uncommitted, so the
    # sources that produce the scores are hashed as well.
    sources = sorted((root / "src" / "obliterated").glob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in sources)).hexdigest()
    return {"obliterated": git("rev-parse", "--short", "HEAD") + ("+uncommitted" if git("status", "--porcelain") else ""),
            "source_sha256": digest,
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("torch", "transformers", "huggingface_hub", "safetensors")}}


def card(repo: str, revision: str, out: Path) -> dict | None:
    """The model card as it stood at ``revision``, copied next to the record, and the fields a
    table of checkpoints cites: what it says it was trained on, from what, under which licence."""
    try:
        text = Path(hf_hub_download(repo, "README.md", revision=revision)).read_text()
    except Exception:  # noqa: BLE001 - a repo without a card is still scored
        return None
    copy = record.provenance_path(out, repo).with_suffix(".README.md")
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text)
    meta = ModelCard(text, ignore_metadata_errors=True).data.to_dict()
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(),
            **{k: meta.get(k) for k in ("base_model", "datasets", "language", "license", "pipeline_tag", "tags")}}


def provenance(c, snapshot: Path, ref: dict, args) -> dict:
    info = record.snapshot_info(snapshot)
    return {
        "repo": c.repo, "base": c.base, "label": c.label, "method": c.method, "notes": c.notes,
        "revision": info["revision"], "files": info["files"],
        "base_revision": ref["snapshot"]["revision"], "base_files": ref["snapshot"]["files"],
        "prompts": ref["prompts"],
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": ref["device"], "dtype": ref["dtype"],
        "settings": {k: getattr(args, k) for k in ("n_train", "n_val", "batch_size", "own_tokenizer", "skip_weights",
                                                   "completions", "n_jbb", "max_new_tokens", "think_tokens")},
        "versions": versions(),
        "card": card(c.repo, info["revision"], Path(args.out)),
    }


def prefetch(repo: str) -> threading.Thread:
    """Start downloading ``repo`` in the background, so the next checkpoint arrives while the
    current one is on the GPU. A failure here is left for the scoring pass to hit and report."""
    def run() -> None:
        try:
            fetch(repo)
        except Exception:  # noqa: BLE001, S110
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def changed(delta: torch.Tensor) -> bool:
    return bool(torch.isfinite(delta).all()) and bool(delta.abs().max() > 0)


def noting_factors(pairs, kept: dict):
    """Pass the writer deltas through, keeping each changed one's top singular values and
    residual-stream directions in ``kept``."""
    for name, delta in pairs:
        if changed(delta):
            kept[f"{name}.s"], u = detect.top_factors(delta, N_FACTORS)
            kept[f"{name}.u"] = u.half()
        yield name, delta


def weight_fields(c, n_layers: int, out: Path, kept: dict | None = None, cand_dir: Path | None = None) -> dict:
    """Weight-detector fields for one checkpoint, from the cached safetensors of it and its
    base. With ``kept``, the factors of every changed matrix are added to it, and so are those
    of the embedding and the output head, which the detector itself does not read."""
    base_dir, cand_dir = fetch(c.base), cand_dir or fetch(c.repo)
    pairs = detect.diff_pairs_from_safetensors(base_dir, cand_dir, range(n_layers))
    fields = detect.weight_scores(pairs if kept is None else noting_factors(pairs, kept), detect.mid_layers(n_layers))
    spectra = record.spectra_path(out, c.repo)
    spectra.parent.mkdir(parents=True, exist_ok=True)
    spectra.write_text(json.dumps(fields.pop("per_matrix")))
    if kept is not None:
        for name, delta in detect.vocab_pairs(base_dir, cand_dir):
            kept[f"{name}.fro"] = torch.tensor(detect.frobenius(delta))
            if changed(delta):
                kept[f"{name}.s"], u = detect.top_factors(delta, N_FACTORS)
                kept[f"{name}.u"] = u.half()
    return fields


def rescore_weights(out: Path) -> None:
    registry = {c.repo: c for c in load_registry()}
    rows = []
    for r in sweep.read_rows(out):
        c = registry.get(r["repo"])
        if c is None:
            print(f"dropping {r['repo']}: no longer in the registry")
            continue
        n_layers = json.loads((fetch(c.base) / "config.json").read_text())["num_hidden_layers"]
        r = {k: v for k, v in r.items() if not k.startswith("weight_")}
        r.update(label=c.label, family=c.family, params_b=c.params_b, **weight_fields(c, n_layers, out))
        print(f"{c.repo}: top1 {r['weight_top1_energy']:.3f} weighted {r['weight_top1_energy_w']:.3f} "
              f"top8w {r['weight_top8_energy_w']:.3f} active {r['weight_n_active']}/{r['weight_n_changed']}", flush=True)
        rows.append(r)
    tmp = out.with_suffix(".tmp")
    sweep.write_rows(rows, tmp)
    tmp.replace(out)


def add_alignment(out: Path) -> None:
    """Give every row in ``out`` that has a record its refusal-alignment scores. Only the
    record files are read, so this works after the weights are gone. A scoring run appends to
    the same file: if it changed while this ran, nothing is written."""
    before = out.read_text()
    rows = sweep.read_rows(out)
    done = 0
    for r in rows:
        saved = record.tensors_path(out, r["repo"])
        if saved.exists():
            r.update(record.alignment(load_file(saved)))
            done += 1
    if out.read_text() != before:
        raise SystemExit(f"{out} changed while this ran (is a scoring run writing to it?); nothing written")
    tmp = out.with_suffix(".tmp")
    sweep.write_rows(rows, tmp)
    tmp.replace(out)
    print(f"{done} of {len(rows)} rows have a record and now carry {', '.join(record.ALIGNMENT)}")


def summary(out: Path) -> None:
    scored = sweep.read_rows(out)
    pos = [r for r in scored if r["label"].startswith("abliterated")]
    neg = sweep.benign(scored)
    if not pos or not neg:
        return
    for det in sweep.DETECTORS + sweep.EXTRA_DETECTORS:
        vals = [(r[det], y) for rows, y in ((pos, 1), (neg, 0)) for r in rows if det in r and r[det] == r[det]]
        if len({y for _, y in vals}) == 2:
            print(f"{det}: AUROC {detect.auroc([v for v, _ in vals], [y for _, y in vals]):.3f} "
                  f"({sum(y for _, y in vals)} edited, {sum(1 - y for _, y in vals)} benign)")


def score(c, ref: dict, args) -> dict:
    bs = args.batch_size
    mid = detect.mid_layers(ref["n_layers"])
    snapshot = fetch(c.repo)
    cand = load(str(snapshot), device=args.device, tokenizer=None if args.own_tokenizer else ref["tokenizer"])
    if cand.n_layers != ref["n_layers"]:
        unload(cand)
        raise ValueError(f"{cand.n_layers} layers, base has {ref['n_layers']}")
    proj = detect.projections(cand, ref["ref_dirs"], ref["h_val"], ref["s_val"], batch_size=bs)
    cand_logits = last_logits(cand, ref["h_val"], bs)
    if not torch.isfinite(cand_logits).all():
        unload(cand)
        raise ValueError("non-finite logits: the uploaded weights are broken")
    budget = None
    if args.completions:
        budget = completions.record(cand, c.repo, completions.prompts(args.n_jbb), completions.path(args.out, c.repo),
                                    "own" if args.own_tokenizer else "base", args.max_new_tokens, bs, args.think_tokens)
    unload(cand)

    per_prompt = detect.behaviour_per_prompt(ref["logits"], cand_logits, ref["refusal_ids"])
    shift = detect.behaviour_summary(per_prompt)
    row = {
        "repo": c.repo,
        "base": c.base,
        "label": c.label,
        "family": c.family,
        "params_b": c.params_b,
        "tokenizer": "own" if args.own_tokenizer else "base",
        "n_harmful": len(ref["h_val"]),
        "bypass_rate": shift["bypass_rate"],
        "baseline_bypass_rate": shift["baseline_bypass_rate"],
        "activation_gap": detect.activation_gap(ref["sep"], detect.separation_of(*proj), mid),
        "behaviour_kl_harmful": shift["kl_harmful"],
        "behaviour_refusal_drop": shift["refusal_score_drop"],
    }
    if budget is not None:
        row["completions_max_new_tokens"] = budget
    kept: dict | None = {} if args.record else None
    if not args.skip_weights:
        row.update(weight_fields(c, ref["n_layers"], Path(args.out), kept, snapshot))
    if kept is not None:
        kept["reference.directions"] = ref["ref_dirs"].half()
        if not args.skip_weights:
            row.update(record.alignment(kept))
        kept.update({f"behaviour.{k}": v for k, v in per_prompt.items()})
        kept["activation.harmful"], kept["activation.harmless"] = proj
        kept["activation.harmful_base"], kept["activation.harmless_base"] = ref["proj"]
        record.write(args.out, c.repo, kept, provenance(c, snapshot, ref, args))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--repo", nargs="+", default=None, help="score only these registry rows")
    ap.add_argument("--base", default=None, help="only checkpoints of this base")
    ap.add_argument("--params-max", type=float, default=None, help="skip checkpoints above this size (billions)")
    ap.add_argument("--n-train", type=int, default=128, help="prompts per class for the reference directions")
    ap.add_argument("--n-val", type=int, default=64, help="prompts per class the detectors are scored on")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--own-tokenizer", action="store_true", help="use each candidate's own tokenizer and chat template")
    ap.add_argument("--redo", action="store_true", help="rescore repos already present in --out")
    ap.add_argument("--rescore-weights", action="store_true",
                    help="recompute the weight fields and labels of the rows in --out, then exit")
    ap.add_argument("--add-alignment", action="store_true",
                    help="add the refusal-alignment scores to the rows in --out from their saved records, then exit")
    ap.add_argument("--completions", action="store_true",
                    help="also write each checkpoint's answers to JailbreakBench, for judge_checkpoints.py")
    ap.add_argument("--n-jbb", type=int, default=100, help="JailbreakBench prompts for --completions")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="answer length for --completions")
    ap.add_argument("--think-tokens", type=int, default=3072,
                    help="budget for a second pass when more than a fifth of the answers stop inside the reasoning")
    ap.add_argument("--record", action="store_true",
                    help="save weight-delta factors, per-prompt measurements and provenance under <out dir>/record/")
    ap.add_argument("--evict", action="store_true",
                    help="delete each checkpoint from the Hugging Face cache once it is scored and its record "
                         "validates; implies --completions and --record. Bases are kept")
    ap.add_argument("--prefetch", action="store_true",
                    help="download the next checkpoint while the current one is scored (on with --evict)")
    ap.add_argument("--check", action="store_true", help="validate the record of every row in --out, then exit")
    ap.add_argument("--min-free-gb", type=float, default=60.0,
                    help="with --evict, stop before a download if the cache disk has less free than this")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/detect/checkpoints.jsonl")
    args = ap.parse_args()
    if args.evict:
        args.completions = args.record = args.prefetch = True
    if args.record:
        versions()  # now, before hours of scoring during which the sources may be edited
    if args.check:  # with --evict: delete the checkpoints whose record validates, and only those
        keep = bases(load_registry())
        bad = 0
        for r in sweep.read_rows(args.out):
            problems = record.validate(r, args.out, completions.prompts(args.n_jbb), weights="weight_n_changed" in r)
            bad += bool(problems)
            print(f"{'PROBLEMS' if problems else 'ok':<9}{r['repo']}" + "".join(f"\n    - {p}" for p in problems), flush=True)
            if args.evict and not problems and r["repo"] not in keep and (freed := evict(r["repo"], keep)):
                print(f"         deleted from the cache: {freed / 1e9:.1f} GB freed", flush=True)
        raise SystemExit(1 if bad else 0)
    if args.rescore_weights:
        rescore_weights(Path(args.out))
        summary(Path(args.out))
        return
    if args.add_alignment:
        add_alignment(Path(args.out))
        summary(Path(args.out))
        return

    registry = load_registry()
    keep = bases(registry)
    rows = [
        c for c in registry
        if (args.family is None or c.family == args.family)
        and (args.label is None or c.label == args.label)
        and (args.repo is None or c.repo in args.repo)
        and (args.base is None or c.base == args.base)
        and (args.params_max is None or (c.params_b or 0) <= args.params_max)
    ]
    if not rows:
        raise SystemExit("no registry rows match")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.redo or not out.exists() else {r["repo"] for r in sweep.read_rows(out)}
    rows = sorted((c for c in rows if c.repo not in done), key=lambda c: c.base)  # one reference pass per base
    prompts = data.contrast_sets("arditi", args.n_train, args.n_val)

    refs: dict[str, dict] = {}
    failed = []
    unvalidated = []
    arriving: threading.Thread | None = None
    for i, c in enumerate(rows):
        print(f"== [{i + 1}/{len(rows)}] {c.repo} (base {c.base}, {c.label})", flush=True)
        free = shutil.disk_usage(constants.HF_HUB_CACHE).free / 1e9
        if args.evict and free < args.min_free_gb:
            print(f"stopping: {free:.0f} GB free on the cache disk, below --min-free-gb {args.min_free_gb:.0f}")
            break
        if arriving is not None:
            arriving.join()  # this checkpoint's own download, started during the previous one
        arriving = prefetch(rows[i + 1].repo) if args.prefetch and i + 1 < len(rows) else None
        problems: list[str] | None = None
        try:
            if c.base not in refs:
                refs.clear()  # bases are visited in order; keep one in memory
                refs[c.base] = reference(c.base, prompts, args)
            row = score(c, refs[c.base], args)
        except Exception as e:  # noqa: BLE001 - one broken upload should not end a long batch
            traceback.print_exc()
            failed.append((c.repo, f"{type(e).__name__}: {e}"))
            problems = [f"scoring failed: {type(e).__name__}: {e}"]
        else:
            print(json.dumps(row), flush=True)
            sweep.write_rows([row], out)
            if args.record:  # judge what is on disk, not what is in memory
                problems = record.validate(sweep.read_rows(out)[-1], out, completions.prompts(args.n_jbb),
                                           weights=not args.skip_weights)
                print(f"record of {c.repo}: " + ("validated" if not problems else "".join(f"\n  - {p}" for p in problems)), flush=True)
        if not args.evict:
            continue
        if problems:  # nothing is deleted until what replaces it has been checked
            unvalidated.append(c.repo)
            print(f"kept on disk: {c.repo} did not validate", flush=True)
        elif c.repo in keep:
            print(f"kept on disk: {c.repo} is the base of another row", flush=True)
        else:
            print(f"deleted {c.repo} from the cache: {evict(c.repo, keep) / 1e9:.1f} GB freed", flush=True)

    for repo, err in failed:
        print(f"FAILED {repo}: {err}")
    if unvalidated:
        print(f"{len(unvalidated)} checkpoints are still on disk because they failed or their record did not validate:")
        print("".join(f"  {repo}\n" for repo in unvalidated), end="")
    if out.exists():
        summary(out)


if __name__ == "__main__":
    main()
