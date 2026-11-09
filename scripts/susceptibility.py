"""Direction B: predictors on the clean model and the measured worst-case drop, for one model.

    uv run python scripts/susceptibility.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --edit runs/Qwen__Qwen2.5-1.5B-Instruct/edit.json

Writes runs/<model>/susceptibility.json. Run it on every model in configs/models.yaml (and on
defended variants: ReFAT, extended-refusal, fail-closed, safety-pretrained checkpoints), then
correlate the predictors with ``susceptibility`` across models with scripts/collect.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from obliterated import data, direction, susceptibility
from obliterated.edits import EditSpec
from obliterated.models import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--edit", default=None, help="EditSpec with source_layer/source_pos")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--kl-budget", type=float, default=0.1)
    ap.add_argument("--method", default="pca", choices=["pca", "iterative"])
    ap.add_argument("--source", default="arditi", choices=["arditi", "advbench", "harmbench"])
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
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
        layer = spec.source_layer
        pos = spec.source_pos if spec.source_pos is not None else args.pos
    elif args.layer is not None:
        layer, pos = args.layer, args.pos
    else:
        cands = direction.candidate_directions(lm, h_tr, s_tr, bs)
        sel = direction.select_direction(lm, cands, h_val, s_val, batch_size=bs)
        layer, pos = sel.layer, sel.pos

    rep = susceptibility.report(
        lm, h_tr, s_tr, h_val, s_val, layer, pos, args.k_max, args.kl_budget, args.method, bs
    )
    rep["n_train"] = [len(h_tr), len(s_tr)]
    rep["n_val"] = [len(h_val), len(s_val)]
    (out / "susceptibility.json").write_text(json.dumps(rep, indent=2))
    g = rep["geometry"]
    print(
        f"{args.model}: layer {layer} | stable rank {g['stable_rank']:.2f}, dim90 {g['dim90']}, "
        f"probe margin {g['probe_margin']:.3f} | layers above half peak "
        f"{rep['layers']['n_layers_above_half_peak']} | susceptibility {rep['susceptibility']:.3f}"
    )
    for r in rep["drop_by_rank"]:
        print(f"  k={r['k']}: drop {r['refusal_score_drop']:.2f} bypass {r['bypass_rate']:.2f} KL {r['kl_harmless']:.3f}")


if __name__ == "__main__":
    main()
