"""A reader-side edit, hidden: what still sees it, and what hiding it costs.

    uv run python scripts/padded_edits.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --direction runs/unlisted/sweeps/<direction> \
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

A decoy along any direction is the same to a score that reads the weights, but not to the
model. With --least-used n, each matrix's decoys are taken from the n directions the stream has
least in where that matrix reads it, the edit's direction and the base's refusal span left out
(obliterated.stream): measured on the training prompts, each followed by --stream-tokens of the
base's own answer. <--out>/<direction>/<model>/stream.json says how much of the stream those
directions hold, there and on the held-out prompts, and how much the edit's own direction does.
With --skip-least-used s as well, the n directions come after the s the stream has least in:
directions it uses more, at a cost to the model that grows with them.

A detector that has the base's activations can rank the parts of a change by what they do to
the stream, and a decoy that is large in the weights and does little then ranks low. With
--by-effect, --decoy-gain is a decoy's size as the stream meets it: each decoy does to the
stream, under its second moment and under its variance, no less than that many times what the
edit does (obliterated.edits.blind_readers_on_disk). Those variants are named apart (<name>_e).

With --detector-stream, a folder holding a detector's own measurement of the base's stream
(scripts/weighted_readers.py --measure, on prompts other than the ones decoys are chosen on),
every row also has the shared-direction score with each change weighed by that measurement, at
the sizes that pass keeps (obliterated.record.weighted_shared_scores). A row that was made
without it is not made again.

With --gain-floor f, every variant is made without touching the columns of a matrix behind a
gain under f times its norm's median (obliterated.edits.blind_readers_on_disk), and is named
apart (<name>_f<f>), so that it stands beside the same variant made without.

Each is made from the base's files, read against the base by the code that reads a checkpoint
(obliterated.record.read_readers, keeping the first of --kept factors per matrix as the
records do), and then run: the model is loaded once and, for each variant, runs from the very
matrices that were just read (obliterated.edits.readers_from). So a row of
<--out>/<direction>/<model>/padded.jsonl holds what a detector sees and what the edit does:
the reader_* scores, first-token bypass and KL on harmless prompts as the sweeps measure them,
and the share of answers that open with a refusal phrase. For each further value of --kept
the shared-direction scores are added as they would be had a record kept that many factors.

With --divergence-tokens t, every variant is also measured against the base over whole
answers: the base's own greedy answers to the held-out harmless prompts, t tokens of each, are
fed back to the variant, and <--out>/<direction>/<model>/divergence.jsonl gets a row with how
far it stands from the base over all their tokens (obliterated.divergence) beside the
first-token KL of the same pass. A variant whose row an earlier run made is made again for
this and its row is left as it was.

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

import torch
from safetensors import safe_open

from obliterated import completions, data, direction, divergence, record, stream, sweep, unlisted
from obliterated.edits import EditSpec, blind_readers_on_disk, readers_from
from obliterated.generate import generate
from obliterated.judge import Judge
from obliterated.models import load, unload
from obliterated.refusal import kl_div, last_logits, refusal_rate, refusal_score_from_logits
from obliterated.registry import snapshot_folder

BEHAVIOUR = ("bypass_rate", "kl_harmless", "phrase_refusal_rate")


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def with_answers(lm, prompts: list[str], tokens: int, bs: int) -> list[str]:
    """Each prompt as the model is handed it, followed by the first tokens of the model's own answer."""
    answers = generate(lm, prompts, tokens, bs) if tokens else [""] * len(prompts)
    return [lm.format(p) + a for p, a in zip(prompts, answers)]


def shares(rows: dict, seen: dict, held: dict) -> dict:
    """How much of the stream lies in the directions given for each norm, where they were chosen and on held-out text."""
    by_norm = {norm: [stream.share(seen[norm], r), stream.share(held[norm], r)] for norm, r in rows.items()}
    return {"share": sum(a for a, _ in by_norm.values()) / len(by_norm), "share_heldout": sum(b for _, b in by_norm.values()) / len(by_norm),
            "by_norm": by_norm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the base")
    ap.add_argument("--direction", required=True, help="sweep folder holding the one direction of the reader-side edit to hide")
    ap.add_argument("--readers", required=True, help="results file of reader_weights.py holding fine-tunes of the base")
    ap.add_argument("--span", default="runs/detect/span", help="folder of base span files (scripts/refusal_span.py)")
    ap.add_argument("--scratch", required=True, help="where the edited reading matrices are written while they are measured")
    ap.add_argument("--out", default="runs/padded")
    ap.add_argument("--shared", type=ints, default=[], help="edits of this many directions, the same in every matrix")
    ap.add_argument("--decoys", type=ints, default=[], help="edits with this many decoys in each matrix")
    ap.add_argument("--decoy-gain", type=lambda s: [float(x) for x in s.split(",") if x], default=[1.5],
                    help="a decoy's change over the edit's, in each matrix; several give a variant each")
    ap.add_argument("--least-used", type=ints, default=[0],
                    help="take a matrix's decoys from this many of the directions the stream has least in where it reads; 0 for any direction; several give a variant each")
    ap.add_argument("--skip-least-used", type=int, default=0,
                    help="take the --least-used directions after this many of the least used; the variants are named apart")
    ap.add_argument("--by-effect", action="store_true",
                    help="--decoy-gain is what a decoy does to the stream over what the edit does, not its size in the weights; the variants are named apart")
    ap.add_argument("--stream-tokens", type=int, default=64, help="tokens of the base's own answer counted after each prompt where the stream is measured")
    ap.add_argument("--gain-floor", type=float, default=0.0,
                    help="leave a matrix's columns alone where the norm's gain is under this many times its median; the variants are named apart")
    ap.add_argument("--kept", type=ints, default=[8], help="factors a record keeps per matrix: the first as the records do, the rest for the shared-direction scores")
    ap.add_argument("--detector-stream", default=None,
                    help="folder of a detector's measurements of the stream (scripts/weighted_readers.py): the scores weighed by it are added to every row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=32, help="tokens generated for the refusal-phrase check")
    ap.add_argument("--divergence-tokens", type=int, default=0,
                    help="measure every variant against the base over this many tokens of the base's own answer to each held-out harmless prompt")
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
    if any(0 < n < p for n in args.least_used for p in args.decoys):
        raise SystemExit("--least-used: fewer directions than there are decoys to take from them")
    if args.skip_least_used and not any(args.least_used):
        raise SystemExit("--skip-least-used: there are no --least-used directions to skip in")
    if args.detector_stream and not stream.path(args.detector_stream, args.model).exists():
        raise SystemExit(f"--detector-stream: {args.detector_stream} holds no measurement of {args.model}")
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
    gains, skip = args.decoy_gain, args.skip_least_used
    said = len(gains) > 1 or gains != ap.get_default("decoy_gain")  # the one usual strength goes unnamed
    ranked = lambda n: f"{n}_s{skip}" if skip else str(n)  # which of the ranking's directions, as stream.json names them
    variants = [("plain", 1, 0, None, 0), *((f"shared_k{k}", k, 0, None, 0) for k in args.shared),
                *((f"decoys_p{p}" + (f"_g{g}" if said else "") + (f"_n{ranked(n)}" if n else "") + ("_e" if args.by_effect else ""), 1, p, g, n)
                  for n in args.least_used for g in gains for p in args.decoys)]
    if args.gain_floor:
        variants = [(f"{name}_f{args.gain_floor}", *rest) for name, *rest in variants]
    jobs = {args.model: {"repo": args.model, "base": args.model, "label": "base", "bypass_rate": 0.0}}
    jobs |= {f"{args.model}#{source.name}+{name}": {"repo": f"{args.model}#{source.name}+{name}", "base": args.model, "label": "reader_edit",
                                                    "direction": source.name, "variant": name} for name, *_ in variants}
    table, far_table = out / "padded.jsonl", out / "divergence.jsonl"
    done = {r["name"]: r for r in sweep.read_rows(table)} if table.exists() else {}
    far_done = {r["name"] for r in sweep.read_rows(far_table)} if far_table.exists() else set()
    wanted = [name for name in jobs if args.answers and not completions.path(judged, name).exists()]
    todo = [v for v in variants if v[0] not in done or f"{args.model}#{source.name}+{v[0]}" in wanted
            or args.divergence_tokens and v[0] not in far_done]
    span_path = record.span_path(args.span, args.model).resolve()
    (out / "padded.json").write_text(json.dumps({
        "model": args.model, "base_revision": pinned["base_revision"], "base_files": pinned["base_files"], "directions_from": repos[0],
        "direction": str(found[0]), "direction_sha256": hashlib.sha256(found[0].read_bytes()).hexdigest(),
        "span": {"path": str(span_path), "sha256": hashlib.sha256(span_path.read_bytes()).hexdigest()},
        "decoy_gains": gains, "least_used": args.least_used, "skip_least_used": skip, "by_effect": args.by_effect, "stream_tokens": args.stream_tokens, "gain_floor": args.gain_floor, "kept": args.kept, "seed": args.seed, "alpha": 1.0, "band": "all",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions}, indent=1))

    if todo or wanted:
        lm = load(str(base_dir), device=args.device)
        lm.name = args.model
        h_train, s_train, h_val, s_val = data.contrast_sets("arditi", args.n_train, args.n_val)
        h_val, s_val = direction.filter_by_baseline(lm, h_val, s_val, bs)
        ids, base_s = lm.refusal_token_ids, last_logits(lm, s_val, bs)
        base_far = divergence.base_answers(lm, s_val, args.divergence_tokens, bs) if args.divergence_tokens else None
        prompts = completions.prompts(args.answers) if args.answers else []
        if args.model in wanted:
            completions.record(lm, args.model, prompts, completions.path(judged, args.model), "base", args.max_new_tokens, bs)
        print(f"{args.model}, {source.name}: {len(variants)} variants, {len(todo)} to make; {len(h_val)} held-out harmful prompts", flush=True)
        among, used, weigh = {}, {}, None
        if any(n or p and args.by_effect for _, _, p, _, n in todo):
            texts = with_answers(lm, h_train + s_train, args.stream_tokens, bs)
            if args.by_effect:  # the second moment and the variance about the mean, for every norm
                means, seen, n_seen = stream.measure(lm, texts, bs)
                weigh = {norm: [c, c - torch.outer(means[norm], means[norm])] for norm, c in seen.items()}
            else:
                seen, n_seen = stream.moments(lm, texts, bs)
            held, n_held = stream.moments(lm, with_answers(lm, h_val + s_val, args.stream_tokens, bs), bs)
            clear = torch.cat([rows[:1], span["basis"].float()])  # what no decoy may touch
            for n in sorted({n for *_, n in todo if n}):
                among[n] = {norm: stream.least_used(c, n, avoid=clear, skip=skip) for norm, c in seen.items()}
                used[ranked(n)] = shares(among[n], seen, held)
                print(f"   the {n} directions the stream has least in{f' after the first {skip}' if skip else ''} hold {used[ranked(n)]['share']:.5f} of it "
                      f"({used[ranked(n)]['share_heldout']:.5f} on held-out prompts); any {n} directions hold {n / lm.d_model:.5f}", flush=True)
            made = {"texts": len(h_train) + len(s_train), "tokens": n_seen, "heldout_texts": len(h_val) + len(s_val), "heldout_tokens": n_held,
                    "answer_tokens": args.stream_tokens, "d_model": lm.d_model, "direction": shares(dict.fromkeys(seen, rows[:1]), seen, held)}
            before = json.loads((out / "stream.json").read_text()) if (out / "stream.json").exists() else {}
            same = all(before.get(key) == value for key, value in made.items() if key != "direction")
            (out / "stream.json").write_text(json.dumps({**made, "least_used": {**(before["least_used"] if same else {}), **used},
                                                         "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions}, indent=1))
            del seen, held
        detector = None  # what a detector weighs a change by, worked out when the first row needs it
        for name, k, p, gain, n in todo:
            started, edit, job = time.time(), scratch / f"{slug}_{name}", f"{args.model}#{source.name}+{name}"
            try:
                blind_readers_on_disk(base_dir, edit, rows[:k], decoys=p, decoy_gain=gain or 1.0, seed=args.seed, avoid=span["basis"], among=among.get(n),
                                      gain_floor=args.gain_floor, weigh=weigh if p else None)
                row = done.get(name)
                if row is None:
                    read = record.read_readers(base_dir, edit, directions, args.kept[0], span=span)
                    if read["missing"]:
                        raise SystemExit(f"{name} could not be read: {read['missing'][:3]}")
                    more = {f"{key}_k{n}": value for n in args.kept[1:]
                            for key, value in record.shared_scores(record.read_readers(base_dir, edit, directions, n)["kept"]).items()
                            if key != "reader_shared_n"}
                    if args.detector_stream:
                        if detector is None:
                            detector = stream.roots(stream.load(stream.path(args.detector_stream, args.model))[1])
                        more |= record.weighted_shared_scores(record.weighted_readers(base_dir, edit, detector))
                with readers_from(lm, edit):
                    if row is None:
                        h = refusal_score_from_logits(last_logits(lm, h_val, bs), ids)
                        does = {"bypass_rate": float((h < 0).float().mean()), "kl_harmless": float(kl_div(base_s, last_logits(lm, s_val, bs)).mean()),
                                "phrase_refusal_rate": refusal_rate(generate(lm, h_val, args.gen_tokens, bs))}
                    if job in wanted:
                        completions.record(lm, job, prompts, completions.path(judged, job), "base", args.max_new_tokens, bs)
                    if base_far is not None and name not in far_done:
                        # the first-token KL is measured again here when the row is an earlier run's: it says the edit was made again as it was
                        first = does["kl_harmless"] if row is None else float(kl_div(base_s, last_logits(lm, s_val, bs)).mean())
                        sweep.write_rows([{"name": name, "model": args.model, "direction": source.name, "max_new_tokens": args.divergence_tokens,
                                           "kl_harmless": first, **divergence.measure(lm, base_far, bs)}], far_table)
                        far_done.add(name)
            finally:
                shutil.rmtree(edit, ignore_errors=True)
            if row is None:
                row = {"name": name, "model": args.model, "direction": source.name, "k": k, "decoys": p, "decoy_gain": gain,
                       "decoys_among": n or None, "decoys_skip": skip if n and skip else None, "decoys_by_effect": True if p and args.by_effect else None,
                       "gain_floor": args.gain_floor or None, "stream_share": used[ranked(n)]["share"] if n else None,
                       "stream_share_heldout": used[ranked(n)]["share_heldout"] if n else None,
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
