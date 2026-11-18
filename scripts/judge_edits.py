"""Does a reader-side edit stop refusal? Full answers from the edited base, judged like a checkpoint's.

    uv run python scripts/judge_edits.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps runs/unlisted/sweeps/listed runs/unlisted/sweeps/searched_answers_free \
        --out runs/unlisted/judged.jsonl --judge qylu4156/strongreject-15k-v1

A sweep point is called working when the first generated token is no longer a refusal token.
That is a fair shortcut for a direction taken from the difference in means, and no test at all
of a direction that was searched for, which can be fitted to that token. This does for an edit
what judge_checkpoints.py does for an uploaded checkpoint. Each folder in --sweeps holds one
direction, as scripts/unlisted_directions.py writes them. The base is edited in memory along
it at --alpha in --band, answers JailbreakBench in full and is restored; the unedited base
answers the same prompts as the reference. Then the judge labels every answer. Run it again
with another --judge to add that judge's columns: answers that are on disk are not made again,
and the model is then not loaded at all.

The base is loaded from the snapshot its span file pins, as the sweeps in those folders were.
Generation finishes and the model is dropped before the judge is loaded. Answers are written
under <out dir>/completions/ (model output on harmful prompts: local only, gitignored); rows
go to --out, one per edit, named <model>#<direction>@<alpha>-<band>. Nothing is downloaded and
no weights are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from obliterated import completions, record, sweep
from obliterated.edits import EditSpec, temporarily_blinded
from obliterated.judge import Judge
from obliterated.models import load, unload
from obliterated.registry import snapshot_folder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the base")
    ap.add_argument("--sweeps", nargs="+", required=True, help="sweep folders holding one direction each for the base")
    ap.add_argument("--span", default="runs/detect/span", help="folder of base span files: names the base's snapshot")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--band", default="all", choices=["all", "mid"])
    ap.add_argument("--n", type=int, default=100, help="JailbreakBench prompts")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--judge", default="qylu4156/strongreject-15k-v1")
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--skip-judge", action="store_true", help="generate only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/unlisted/judged.jsonl")
    args = ap.parse_args()
    out, slug = Path(args.out), args.model.replace("/", "__")

    jobs = {args.model: {"repo": args.model, "base": args.model, "label": "base", "bypass_rate": 0.0}}
    directions = {}
    for folder in map(Path, args.sweeps):  # all of them before anything is generated
        found = sorted((folder / slug).glob("subspace_*.json"))
        spec = EditSpec.load(found[0]) if len(found) == 1 else None
        if spec is None or spec.side != "readers" or spec.rank != 1:
            raise SystemExit(f"{folder / slug} does not hold one direction of a reader-side edit")
        at = [r for r in sweep.read_rows(folder / slug / "frontier.jsonl") if (r["alpha"], r["band"], r["k"]) == (args.alpha, args.band, 1)]
        if len(at) != 1:
            raise SystemExit(f"{folder / slug} has {len(at)} points at strength {args.alpha} in band {args.band}")
        name = f"{args.model}#{folder.name}@{args.alpha}-{args.band}"
        jobs[name] = {"repo": name, "base": args.model, "label": "reader_edit", "direction": folder.name, "alpha": args.alpha,
                      "band": args.band, "bypass_rate": at[0]["bypass_rate"], "kl_harmless": at[0]["kl_harmless"]}
        directions[name] = spec.vector

    prompts = completions.prompts(args.n)
    todo = [name for name in jobs if not completions.path(out, name).exists()]
    if todo:
        lm = load(str(snapshot_folder(args.model, record.load_span(args.span, args.model)[1]["base_revision"])), device=args.device)
        lm.name = args.model
        layers = sweep.layer_bands(lm.n_layers)[args.band]
        for i, name in enumerate(todo):
            print(f"== [{i + 1}/{len(todo)}] generate {name}", flush=True)
            if name == args.model:
                completions.record(lm, name, prompts, completions.path(out, name), "base", args.max_new_tokens, args.batch_size)
                continue
            with temporarily_blinded(lm, directions[name], args.alpha, layers):
                completions.record(lm, name, prompts, completions.path(out, name), "base", args.max_new_tokens, args.batch_size)
        unload(lm)
    if args.skip_judge:
        return

    judge = Judge(args.judge, device=args.device)
    for job, row in completions.judge_all(jobs, judge, out, args.judge_batch_size):
        verdict = (f"helpfulness score {row[f'{judge.kind}_score']:.3f}" if getattr(judge, "graded", False) else
                   f"judged refusal {row.get('wildguard_refusal', float('nan')):.2f}, harmful {row[f'{judge.kind}_harmful']:.2f}")
        print(f"{job['label']:<12} {job['repo'][len(args.model):] or '(no edit)':<56} first-token bypass {job['bypass_rate']:.2f} | "
              f"substring refusal {row['substring_refusal']:.2f} | {verdict}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
