"""Refusal-direction extraction and selection, following Arditi et al. (NeurIPS 2024).

1. Difference in mean residual activations between harmful and harmless instructions, at each
   layer and at each post-instruction token position, gives candidate directions.
2. Each candidate is scored on validation prompts by
     - bypass:  mean refusal score on harmful prompts with the direction ablated everywhere,
     - induce:  mean refusal score on harmless prompts with the direction added at its layer,
     - KL:      KL divergence on harmless prompts under ablation.
3. Discard candidates from the last 20% of layers, with KL above 0.1, or that do not induce
   refusal; pick the one with the lowest bypass score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from . import hooks
from .models import LM
from .refusal import kl_div, last_logits, refusal_score_from_logits, refusal_scores


@torch.no_grad()
def mean_activations(lm: LM, instructions: list[str], positions: int, batch_size: int = 16):
    """Mean residual-stream input at each layer, last ``positions`` tokens: (positions, layers, d)."""
    store: list[list[torch.Tensor]] = []
    total = torch.zeros(positions, lm.n_layers, lm.d_model, dtype=torch.float64)
    with hooks.cache_residual(lm, positions, store):
        for i in range(0, len(instructions), batch_size):
            enc = lm.tokenize(instructions[i : i + batch_size])
            lm.model(**enc)
            for layer in range(lm.n_layers):
                total[:, layer] += store[layer].pop().double().sum(0)
    return (total / len(instructions)).float()


def candidate_directions(lm: LM, harmful: list[str], harmless: list[str], batch_size: int = 16):
    """Difference-in-means candidates, shape (positions, layers, d_model). Position index p
    corresponds to token offset p - positions from the end of the prompt."""
    pos = lm.eoi_len
    return mean_activations(lm, harmful, pos, batch_size) - mean_activations(lm, harmless, pos, batch_size)


def filter_by_baseline(lm: LM, harmful: list[str], harmless: list[str], batch_size: int = 16):
    """Keep harmful prompts the model already refuses and harmless ones it already answers."""
    h = refusal_scores(lm, harmful, batch_size)
    s = refusal_scores(lm, harmless, batch_size)
    return [p for p, v in zip(harmful, h) if v > 0], [p for p, v in zip(harmless, s) if v < 0]


@dataclass
class Selection:
    direction: torch.Tensor  # raw (unnormalized) difference in means
    pos: int  # negative token offset from the end of the prompt
    layer: int
    scores: list[dict]  # one row per candidate
    baseline_harmful: float
    baseline_harmless: float


@torch.no_grad()
def select_direction(
    lm: LM,
    candidates: torch.Tensor,
    harmful_val: list[str],
    harmless_val: list[str],
    kl_threshold: float = 0.1,
    induce_threshold: float = 0.0,
    prune_layer_fraction: float = 0.2,
    batch_size: int = 16,
) -> Selection:
    n_pos, n_layer, _ = candidates.shape
    ids = lm.refusal_token_ids
    base_harmless_logits = last_logits(lm, harmless_val, batch_size)
    base_h = refusal_scores(lm, harmful_val, batch_size).mean().item()
    base_s = refusal_score_from_logits(base_harmless_logits, ids).mean().item()

    rows = []
    max_layer = int(n_layer * (1 - prune_layer_fraction))
    for p in range(n_pos):
        for layer in tqdm(range(n_layer), desc=f"pos {p - n_pos}", leave=False):
            d = candidates[p, layer]
            with hooks.ablate(lm, d):
                bypass = refusal_scores(lm, harmful_val, batch_size).mean().item()
                kl = kl_div(base_harmless_logits, last_logits(lm, harmless_val, batch_size)).mean().item()
            with hooks.actadd(lm, d, layer):
                induce = refusal_scores(lm, harmless_val, batch_size).mean().item()
            rows.append(dict(pos=p - n_pos, layer=layer, bypass=bypass, induce=induce, kl=kl))

    def keep(r):
        vals = (r["bypass"], r["induce"], r["kl"])
        return (
            not any(math.isnan(v) for v in vals)
            and r["layer"] < max_layer
            and r["kl"] <= kl_threshold
            and r["induce"] >= induce_threshold
        )

    kept = [r for r in rows if keep(r)]
    if not kept:
        raise RuntimeError("no candidate passed the filters; inspect Selection.scores or loosen thresholds")
    best = min(kept, key=lambda r: r["bypass"])
    return Selection(
        direction=candidates[best["pos"] + n_pos, best["layer"]].clone(),
        pos=best["pos"],
        layer=best["layer"],
        scores=rows,
        baseline_harmful=base_h,
        baseline_harmless=base_s,
    )
