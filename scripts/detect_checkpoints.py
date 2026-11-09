"""Score public checkpoints from the registry with every detector, against their declared base.

    uv run python scripts/detect_checkpoints.py --family llama3.2 --n 64
    uv run python scripts/detect_checkpoints.py --label benign_finetune --out runs/detect/benign.jsonl

Per checkpoint: weight-diff energy (streamed from safetensors, so the model never has to fit
on the GPU twice), activation gap and behaviour shift (base loaded first, then the candidate,
never both at once). Rows carry the registry label, so AUROC per detector is printed at the end
when both edited and benign rows are present. Weights are downloaded for analysis only.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from obliterated import data, detect, direction, sweep
from obliterated.models import load
from obliterated.refusal import last_logits
from obliterated.registry import load_registry


def free(lm) -> None:
    del lm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--repo", default=None, help="score one registry row")
    ap.add_argument("--n", type=int, default=64, help="harmful and harmless prompts for the activation tests")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/detect/checkpoints.jsonl")
    args = ap.parse_args()

    rows = [
        c for c in load_registry()
        if (args.family is None or c.family == args.family)
        and (args.label is None or c.label == args.label)
        and (args.repo is None or c.repo == args.repo)
    ]
    if not rows:
        raise SystemExit("no registry rows match")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    h_tr, s_tr, h_val, s_val = data.contrast_sets("arditi", args.n, args.n)

    for c in rows:
        print(f"== {c.repo} (base {c.base}, {c.label})")
        base = load(c.base, device=args.device)
        h_f, s_f = direction.filter_by_baseline(base, h_tr, s_tr, args.batch_size)
        ref_dirs = detect.reference_directions(base, h_f, s_f, batch_size=args.batch_size)
        sep_ref = detect.separation(base, ref_dirs, h_val, s_val, batch_size=args.batch_size)
        base_logits = last_logits(base, h_val, args.batch_size)
        ids = base.refusal_token_ids
        n_layers = base.n_layers
        free(base)

        cand = load(c.repo, device=args.device)
        sep = detect.separation(cand, ref_dirs, h_val, s_val, batch_size=args.batch_size)
        cand_logits = last_logits(cand, h_val, args.batch_size)
        free(cand)

        mid = detect.mid_layers(n_layers)
        row = {
            "repo": c.repo,
            "base": c.base,
            "label": c.label,
            "family": c.family,
            "activation_gap": detect.activation_gap(sep_ref, sep, mid),
        }
        shift = detect.behaviour_shift(base_logits, cand_logits, ids)
        row["behaviour_kl_harmful"] = shift["kl_harmful"]
        row["behaviour_refusal_drop"] = shift["refusal_score_drop"]
        if not args.skip_weights:
            bdir = snapshot_download(c.base, allow_patterns=["*.safetensors", "*.json"])
            cdir = snapshot_download(c.repo, allow_patterns=["*.safetensors", "*.json"])
            energy = detect.weight_recovery_energy(detect.diff_pairs_from_safetensors(bdir, cdir, mid))
            row["weight_top1_energy"] = energy["top1_energy"]
            row["weight_stable_rank"] = energy["stable_rank"]
            row["weight_n_matrices"] = energy["n_matrices"]
        print(json.dumps(row))
        sweep.write_rows([row], args.out)

    scored = sweep.read_rows(args.out)
    labels = [0 if r["label"] in ("benign_finetune", "base") else 1 for r in scored]
    if 0 < sum(labels) < len(labels):
        for det in sweep.DETECTORS:
            vals = [(r[det], y) for r, y in zip(scored, labels) if det in r]
            if vals:
                print(f"{det}: AUROC {detect.auroc([v for v, _ in vals], [y for _, y in vals]):.3f} (n={len(vals)})")


if __name__ == "__main__":
    main()
