"""Reproduce Arditi et al. (NeurIPS 2024) on one model, end to end.

    uv run python scripts/reproduce_refusal_direction.py --model Qwen/Qwen2.5-1.5B-Instruct

Writes to runs/<model>/:
    edit.json            the selected direction as an EditSpec (a few KB; no weights are saved)
    scores.json          bypass / induce / KL for every (position, layer) candidate
    scores.png           heatmap of the bypass score
    completions.jsonl    generations per condition (local only; gitignored)
    summary.json         substring refusal rates per condition, plus a hooks-vs-weights check
Then run scripts/judge.py on completions.jsonl for judge-based refusal rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from obliterated import data, direction, hooks
from obliterated.edits import EditSpec, orthogonalize
from obliterated.generate import generate
from obliterated.models import load
from obliterated.refusal import last_logits, refusal_rate


def plot(scores: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = sorted({r["pos"] for r in scores})
    layers = sorted({r["layer"] for r in scores})
    grid = torch.full((len(positions), len(layers)), float("nan"))
    for r in scores:
        grid[positions.index(r["pos"]), r["layer"]] = r["bypass"]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(positions) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(positions)), [str(p) for p in positions])
    ax.set_xlabel("source layer")
    ax.set_ylabel("token position")
    ax.set_title("Refusal score on harmful prompts with the candidate ablated (lower = bypassed)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--source", default="arditi", choices=["arditi", "advbench", "harmbench"])
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=100, help="prompts per eval set")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--kl-threshold", type=float, default=0.1)
    ap.add_argument("--induce-threshold", type=float, default=0.0)
    ap.add_argument("--skip-filter", action="store_true", help="keep prompts the model mis-handles at baseline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    out = Path(args.out) / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    lm = load(args.model, device=args.device)
    bs = args.batch_size
    print(f"loaded {args.model}: {lm.n_layers} layers, d_model={lm.d_model}, eoi_len={lm.eoi_len}")

    # 1. Contrast sets, filtered to prompts the model already refuses / already answers.
    h_tr, s_tr, h_val, s_val = data.contrast_sets(args.source, args.n_train, args.n_val)
    if not args.skip_filter:
        h_tr, s_tr = direction.filter_by_baseline(lm, h_tr, s_tr, bs)
        h_val, s_val = direction.filter_by_baseline(lm, h_val, s_val, bs)
    print(f"after baseline filter: train {len(h_tr)}/{len(s_tr)}, val {len(h_val)}/{len(s_val)}")

    # 2. Candidates and selection.
    cands = direction.candidate_directions(lm, h_tr, s_tr, bs)
    sel = direction.select_direction(
        lm, cands, h_val, s_val, args.kl_threshold, args.induce_threshold, batch_size=bs
    )
    print(f"selected layer {sel.layer}, position {sel.pos}")
    spec = EditSpec(
        direction=sel.direction.tolist(),
        source_layer=sel.layer,
        source_pos=sel.pos,
        model=args.model,
    )
    spec.save(out / "edit.json")
    (out / "scores.json").write_text(json.dumps(sel.scores))
    plot(sel.scores, out / "scores.png")

    # 3. Held-out generations under each condition.
    evals = data.eval_sets(n_harmless=args.n_eval)
    for name in evals:
        evals[name] = data.sample(evals[name], args.n_eval)
    rows, summary = [], {"model": args.model, "layer": sel.layer, "pos": sel.pos}

    def run(condition: str, sets: list[str]) -> None:
        for name in sets:
            comps = generate(lm, evals[name], args.max_new_tokens, bs)
            summary[f"{condition}/{name}/substring_refusal"] = refusal_rate(comps)
            rows.extend(
                dict(condition=condition, dataset=name, prompt=p, completion=c)
                for p, c in zip(evals[name], comps)
            )
            print(f"{condition:>12} {name:>15}: refusal {summary[f'{condition}/{name}/substring_refusal']:.2f}")

    all_sets = list(evals)
    run("baseline", all_sets)
    with hooks.ablate(lm, sel.direction):
        run("ablation", all_sets)
        probe = evals["harmless"][:8]
        hook_logits = last_logits(lm, probe, bs)
    with hooks.actadd(lm, sel.direction, sel.layer):
        run("actadd", ["harmless"])

    # Weight orthogonalization last, since it edits the model in place.
    orthogonalize(lm, sel.direction)
    summary["hooks_vs_weights_max_abs_logit_diff"] = (
        (last_logits(lm, probe, bs) - hook_logits).abs().max().item()
    )
    run("orthogonalized", all_sets)

    with open(out / "completions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
