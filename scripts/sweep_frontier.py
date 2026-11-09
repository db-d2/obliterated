"""Direction A: sweep the attack family and record behaviour + detector scores per point.

    uv run python scripts/sweep_frontier.py --model Qwen/Qwen2.5-1.5B-Instruct
    uv run python scripts/sweep_frontier.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --edit runs/Qwen__Qwen2.5-1.5B-Instruct/edit.json --ks 1,2,4 --mode weights

Writes runs/<model>/frontier.jsonl (one row per attack point, appended across invocations)
and runs/<model>/frontier.png. If --edit is omitted the source layer is chosen by the Arditi
procedure on the training split first. Pass --negatives runs/detect/benign.jsonl (from
scripts/detect_checkpoints.py) to draw detector thresholds at 5% false-positive rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from obliterated import data, detect, direction, sweep
from obliterated.edits import EditSpec
from obliterated.models import load


def parse_list(s: str, cast):
    return [cast(x) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--edit", default=None, help="EditSpec from reproduce_refusal_direction.py (gives the source layer)")
    ap.add_argument("--layer", type=int, default=None, help="source layer if no --edit")
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--bands", default="all,mid,source,source±2")
    ap.add_argument("--mode", default="hooks", choices=["hooks", "weights"])
    ap.add_argument("--method", default="pca", choices=["pca", "iterative"])
    ap.add_argument("--source", default="arditi", choices=["arditi", "advbench", "harmbench"])
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--negatives", default=None, help="JSONL of detector rows for benign fine-tunes")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    out = Path(args.out) / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    lm = load(args.model, device=args.device)
    bs = args.batch_size

    h_tr, s_tr, h_val, s_val = data.contrast_sets(args.source, args.n_train, args.n_val)
    h_tr, s_tr = direction.filter_by_baseline(lm, h_tr, s_tr, bs)
    h_val, s_val = direction.filter_by_baseline(lm, h_val, s_val, bs)

    if args.edit:
        spec = EditSpec.load(args.edit)
        layer, pos = spec.source_layer, spec.source_pos if spec.source_pos is not None else args.pos
    elif args.layer is not None:
        layer, pos = args.layer, args.pos
    else:
        cands = direction.candidate_directions(lm, h_tr, s_tr, bs)
        sel = direction.select_direction(lm, cands, h_val, s_val, batch_size=bs)
        layer, pos = sel.layer, sel.pos
    print(f"source layer {layer}, pos {pos}")

    ks = parse_list(args.ks, int)
    sub = direction.refusal_subspace(lm, h_tr, s_tr, layer, pos, max(ks), args.method, bs)
    EditSpec(direction=sub.tolist(), source_layer=layer, source_pos=pos, model=args.model, method=args.method).save(
        out / f"subspace_{args.method}_k{max(ks)}.json"
    )

    all_bands = sweep.layer_bands(lm.n_layers, layer)
    bands = {b: all_bands[b] for b in parse_list(args.bands, str) if b in all_bands}
    ref_dirs = detect.reference_directions(lm, h_tr, s_tr, pos, bs)
    rows = sweep.run_sweep(
        lm, sub, h_val, s_val, parse_list(args.alphas, float), ks, bands, args.mode, ref_dirs, batch_size=bs,
        tag=f"{args.method}",
    )
    sweep.write_rows(rows, out / "frontier.jsonl")

    thresholds = None
    if args.negatives:
        neg = sweep.read_rows(args.negatives)
        thresholds = {}
        for det in sweep.DETECTORS:
            vals = sorted(r[det] for r in neg if det in r)
            if vals:
                thresholds[det] = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
                print(f"{det}: AUROC {sweep.frontier_auroc(rows, neg, det):.3f}")
    sweep.plot_frontier(sweep.read_rows(out / "frontier.jsonl"), out / "frontier.png", thresholds)
    for r in rows:
        print(
            f"a={r['alpha']:<5} k={r['k']} {r['band']:<9} bypass {r['bypass_rate']:.2f} "
            f"KL {r['kl_harmless']:.3f} | energy {r['weight_top1_energy']:.2f} "
            f"gap {r['activation_gap']:.2f} klH {r['behaviour_kl_harmful']:.2f}"
        )


if __name__ == "__main__":
    main()
