"""A working reader-side edit, hidden: what still sees it, and what hiding it costs.

    uv run python scripts/padded_edits.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --direction runs/unlisted/sweeps/searched_answers_from_remainder_advbench \
        --readers runs/detect/readers/qwen25.jsonl --scratch /tmp/padded --shared 2,4,8,16,32 --decoys 8,16,32,64

Two reading-side scores need no direction named beforehand: the share of a matrix's change
that lies in one direction, and whether the changed matrices share an input direction
(obliterated.record.shared_scores). An editor who knows them has two ways to hide an edit
along --direction (a sweep folder holding one direction, as scripts/unlisted_directions.py
writes them), and this makes both at full strength in every layer:

- --shared k: the edit made along k directions, the given one and k - 1 random ones outside
  the base's refusal span, the same in every matrix (obliterated.unlisted.padded)
- --decoys p: the edit, and in every matrix p directions of that matrix's own, each changed
  --decoy-gain times as much as the edit changes it (obliterated.edits.blind_readers_on_disk)

Each is made from the base's files, read against the base by the code that reads a checkpoint
(obliterated.record.read_readers, keeping the first of --kept factors per matrix as the
records do), and then run: the model is loaded once and, for each variant, runs from the very
matrices that were just read (obliterated.edits.readers_from). So a row of
<--out>/<direction>/<model>/padded.jsonl holds what a detector sees and what the edit does:
the reader_* scores, first-token bypass and KL on harmless prompts as the sweeps measure them,
and the share of answers that open with a refusal phrase. For each further value of --kept
the shared-direction scores are added as they would be had a record kept that many factors.

With --answers n, every variant and the unedited base also answer n JailbreakBench prompts in
full, and each --judge labels them once the model is dropped (<--out>/judged.jsonl; the
answers under <--out>/completions/ are model output on harmful prompts and stay local).

The base is read at the revision, and scored against the directions, that the reader pass
used for its fine-tunes in --readers, as scripts/reader_positives.py does. Nothing is
downloaded; the edited matrices are removed as soon as they are measured; a second run makes
only what is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from safetensors import safe_open

from obliterated import completions, data, direction, record, sweep, unlisted
from obliterated.edits import EditSpec, blind_readers_on_disk, readers_from
from obliterated.generate import generate
from obliterated.judge import Judge
from obliterated.models import load, unload
from obliterated.refusal import kl_div, last_logits, refusal_rate, refusal_score_from_logits
from obliterated.registry import snapshot_folder

BEHAVIOUR = ("bypass_rate", "kl_harmless", "phrase_refusal_rate")


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the base")
    ap.add_argument("--direction", required=True, help="sweep folder holding the one direction of a working reader-side edit")
    ap.add_argument("--readers", required=True, help="results file of reader_weights.py holding fine-tunes of the base")
    ap.add_argument("--span", default="runs/detect/span", help="folder of base span files (scripts/refusal_span.py)")
    ap.add_argument("--scratch", required=True, help="where the edited reading matrices are written while they are measured")
    ap.add_argument("--out", default="runs/padded")
    ap.add_argument("--shared", type=ints, default=[], help="edits of this many directions, the same in every matrix")
    ap.add_argument("--decoys", type=ints, default=[], help="edits with this many decoys in each matrix")
    ap.add_argument("--decoy-gain", type=float, default=1.5, help="a decoy's change over the edit's, in each matrix")
    ap.add_argument("--kept", type=ints, default=[8], help="factors a record keeps per matrix: the first as the records do, the rest for the shared-direction scores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=32, help="tokens generated for the refusal-phrase check")
    ap.add_argument("--answers", type=int, default=0, help="JailbreakBench prompts to answer in full for the judges")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--judge", nargs="*", default=[], help="judges for the full answers")
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    versions = record.code_versions(Path(__file__))
    bs, slug, source = args.batch_size, args.model.replace("/", "__"), Path(args.direction)
    out, scratch, judged = Path(args.out) / source.name / slug, Path(args.scratch), Path(args.out) / "judged.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    found = sorted((source / slug).glob("subspace_*.json"))
    spec = EditSpec.load(found[0]) if len(found) == 1 else None
    if spec is None or spec.side != "readers" or spec.rank != 1:
        raise SystemExit(f"{source / slug} does not hold one direction of a reader-side edit")
    readers = Path(args.readers)
    repos = [r["repo"] for r in sweep.read_rows(readers) if r["base"] == args.model and r["repo"] != args.model]
    if not repos:
        raise SystemExit(f"{readers} has no fine-tune of {args.model}: there are no directions to score against")
    with safe_open(record.tensors_path(readers, repos[0]), "pt") as f:
        directions = f.get_tensor("reference.directions")
    pinned = json.loads(record.provenance_path(readers, repos[0]).read_text())
    base_dir = snapshot_folder(args.model, pinned["base_revision"])
    if not base_dir.exists() or record._weights(record.snapshot_info(base_dir)["files"]) != record._weights(pinned["base_files"]):
        raise SystemExit(f"{args.model} at {pinned['base_revision']} is not on disk with the weight files its fine-tunes were read against")
    try:
        span = record.checked_span(args.span, args.model, pinned["base_revision"], pinned["base_files"], directions)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    rows = unlisted.padded(spec.vector.reshape(-1), span, max([*args.shared, 1]) - 1, args.seed)
    variants = [("plain", 1, 0), *((f"shared_k{k}", k, 0) for k in args.shared), *((f"decoys_p{p}", 1, p) for p in args.decoys)]
    jobs = {args.model: {"repo": args.model, "base": args.model, "label": "base", "bypass_rate": 0.0}}
    jobs |= {f"{args.model}#{source.name}+{name}": {"repo": f"{args.model}#{source.name}+{name}", "base": args.model, "label": "reader_edit",
                                                    "direction": source.name, "variant": name} for name, _, _ in variants}
    table = out / "padded.jsonl"
    done = {r["name"]: r for r in sweep.read_rows(table)} if table.exists() else {}
    wanted = [name for name in jobs if args.answers and not completions.path(judged, name).exists()]
    todo = [v for v in variants if v[0] not in done or f"{args.model}#{source.name}+{v[0]}" in wanted]
    span_path = record.span_path(args.span, args.model).resolve()
    (out / "padded.json").write_text(json.dumps({
        "model": args.model, "base_revision": pinned["base_revision"], "base_files": pinned["base_files"], "directions_from": repos[0],
        "direction": str(found[0]), "direction_sha256": hashlib.sha256(found[0].read_bytes()).hexdigest(),
        "span": {"path": str(span_path), "sha256": hashlib.sha256(span_path.read_bytes()).hexdigest()},
        "decoy_gain": args.decoy_gain, "kept": args.kept, "seed": args.seed, "alpha": 1.0, "band": "all",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions}, indent=1))

    if todo or wanted:
        lm = load(str(base_dir), device=args.device)
        lm.name = args.model
        _, _, h_val, s_val = data.contrast_sets("arditi", args.n_train, args.n_val)
        h_val, s_val = direction.filter_by_baseline(lm, h_val, s_val, bs)
        ids, base_s = lm.refusal_token_ids, last_logits(lm, s_val, bs)
        prompts = completions.prompts(args.answers) if args.answers else []
        if args.model in wanted:
            completions.record(lm, args.model, prompts, completions.path(judged, args.model), "base", args.max_new_tokens, bs)
        print(f"{args.model}, {source.name}: {len(variants)} variants, {len(todo)} to make; {len(h_val)} held-out harmful prompts", flush=True)
        for name, k, p in todo:
            started, edit, job = time.time(), scratch / f"{slug}_{name}", f"{args.model}#{source.name}+{name}"
            try:
                blind_readers_on_disk(base_dir, edit, rows[:k], decoys=p, decoy_gain=args.decoy_gain, seed=args.seed, avoid=span["basis"])
                row = done.get(name)
                if row is None:
                    read = record.read_readers(base_dir, edit, directions, args.kept[0], span=span)
                    if read["missing"]:
                        raise SystemExit(f"{name} could not be read: {read['missing'][:3]}")
                    more = {f"{key}_k{n}": value for n in args.kept[1:]
                            for key, value in record.shared_scores(record.read_readers(base_dir, edit, directions, n)["kept"]).items()
                            if key != "reader_shared_n"}
                with readers_from(lm, edit):
                    if row is None:
                        h = refusal_score_from_logits(last_logits(lm, h_val, bs), ids)
                        does = {"bypass_rate": float((h < 0).float().mean()), "kl_harmless": float(kl_div(base_s, last_logits(lm, s_val, bs)).mean()),
                                "phrase_refusal_rate": refusal_rate(generate(lm, h_val, args.gen_tokens, bs))}
                    if job in wanted:
                        completions.record(lm, job, prompts, completions.path(judged, job), "base", args.max_new_tokens, bs)
            finally:
                shutil.rmtree(edit, ignore_errors=True)
            if row is None:
                row = {"name": name, "model": args.model, "direction": source.name, "k": k, "decoys": p, "decoy_gain": args.decoy_gain if p else None,
                       "alpha": 1.0, "band": "all", "n_harmful": len(h_val), **does, **read["fields"], **more}
                sweep.write_rows([row], table)
                done[name] = row
            print(f"   {name:<12} bypass {row['bypass_rate']:.2f}, KL {row['kl_harmless']:.3f}, opens with a refusal phrase {row['phrase_refusal_rate']:.2f} | "
                  f"energy in one direction {row['reader_top1_energy']:.3f}, shared direction {row['reader_shared_direction']:.3f}, "
                  f"subspace {row['reader_shared_subspace']:.3f}; {time.time() - started:.0f}s", flush=True)
        unload(lm)
        if scratch.exists() and not any(scratch.iterdir()):
            scratch.rmdir()

    for job in jobs.values():  # what each variant does, beside what the judges will say of it
        if job["label"] == "reader_edit":
            job.update({key: done[job["variant"]][key] for key in BEHAVIOUR})
    for name in args.judge if args.answers else []:
        judge = Judge(name, device=args.device)
        for job, row in completions.judge_all(jobs, judge, judged, args.judge_batch_size):
            if row is None:
                continue
            verdict = (f"helpfulness score {row[f'{judge.kind}_score']:.3f}" if getattr(judge, "graded", False) else
                       f"judged refusal {row.get('wildguard_refusal', float('nan')):.2f}, harmful {row[f'{judge.kind}_harmful']:.2f}")
            print(f"{job.get('variant', '(no edit)'):<14} first-token bypass {job['bypass_rate']:.2f} | substring refusal {row['substring_refusal']:.2f} | {verdict}", flush=True)


if __name__ == "__main__":
    main()
