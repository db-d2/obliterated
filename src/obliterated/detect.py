"""Direction A: has this model had a refusal subspace removed, and how confidently can we tell?

Three detector families, each a primitive that produces one score per candidate model, so the
same code scores our own parameterized edits (the sweep) and public checkpoints (the registry):

1. Weight-diff spectrum. Orthogonalization changes every residual writer by a rank-k update,
   so the base-to-candidate difference of ``o_proj``/``down_proj`` has one (or k) dominant
   singular value. ``weight_recovery_energy`` is the share of the difference's energy in its top
   singular value, averaged over mid-stack matrices (Hurtado 2026, after Zhong and Raghunathan's
   WeightWatch). Needs the clean base weights.
2. Activation gap. Along the base model's per-layer refusal directions, the harmful-minus-
   harmless separation shrinks in an edited model. ``separation`` measures it; the gap is one
   minus the candidate/reference ratio. Needs the base model (or its directions) and prompts.
3. Behaviour shift. KL and refusal-score drop on harmful prompts, candidate vs base. Needs only
   next-token logits, so it is the one test that could run against an API.

Plus ``auroc`` to turn scores and labels into the number the frontier plot reports.
Reference-free detection (no trusted base) is an open problem and deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

from . import direction
from .edits import EditSpec
from .models import LM
from .refusal import kl_div, refusal_score_from_logits
from .susceptibility import spectrum_summary

Tensor = torch.Tensor


# ---- 1. weight-diff spectrum ---------------------------------------------------------------------


def mid_layers(n_layers: int, lo: float = 0.25, hi: float = 0.75) -> list[int]:
    return list(range(int(n_layers * lo), max(int(n_layers * hi), int(n_layers * lo) + 1)))


def _as_matrix(delta: Tensor) -> Tensor:
    """(d_model, d_in) view of a writer's delta; fused experts (n, d_model, d_in) are laid side
    by side as (d_model, n * d_in) so a per-expert rank-k edit stays rank-k."""
    if delta.dim() == 3:
        return delta.movedim(1, 0).reshape(delta.shape[1], -1)
    return delta


def diff_pairs_from_models(base: LM, cand: LM, layers: Iterable[int]) -> Iterator[tuple[str, Tensor]]:
    """(name, cand - base) for every residual writer in ``layers``. Both models must share an
    architecture; weights are compared in float32 on the candidate's device."""
    for i in layers:
        for j, (wb, wc) in enumerate(zip(base.write_params(i), cand.write_params(i))):
            tb = wb if isinstance(wb, nn.Parameter) else wb.weight
            tc = wc if isinstance(wc, nn.Parameter) else wc.weight
            yield f"layer{i}.w{j}", _as_matrix(tc.detach().float() - tb.detach().float().to(tc.device))


def diff_pairs_from_edit(lm: LM, spec: EditSpec, layers: Iterable[int]) -> Iterator[tuple[str, Tensor]]:
    """The weight difference an ``EditSpec`` *would* produce, computed analytically from the
    clean model: delta W = -alpha B^T B W per writer. Lets the sweep score the weight-mode
    version of an edit without copying or modifying the model."""
    from .hooks import orthonormal

    basis = orthonormal(spec.vector)
    edited = set(range(lm.n_layers) if spec.layers is None else spec.layers)
    for i in layers:
        for j, w in enumerate(lm.write_params(i)):
            t = (w if isinstance(w, nn.Parameter) else w.weight).detach().float()
            if i not in edited:
                yield f"layer{i}.w{j}", torch.zeros_like(_as_matrix(t))
                continue
            b = basis.to(t.device)
            yield f"layer{i}.w{j}", _as_matrix(-spec.alpha * b.T @ (b @ t))


def diff_pairs_from_safetensors(
    base_dir: str | Path,
    cand_dir: str | Path,
    layers: Iterable[int],
    suffixes: tuple[str, ...] = ("self_attn.o_proj.weight", "mlp.down_proj.weight"),
) -> Iterator[tuple[str, Tensor]]:
    """Stream (name, cand - base) tensor by tensor from two local safetensors snapshots (as
    ``huggingface_hub.snapshot_download`` leaves them), never loading a full model. This is how
    to audit checkpoints larger than the GPU, including fused-expert MoE if ``suffixes`` names
    their down projections."""
    from safetensors import safe_open

    def index(d: Path) -> dict[str, Path]:
        out = {}
        for f in sorted(Path(d).glob("*.safetensors")):
            with safe_open(f, "pt") as sf:
                for k in sf.keys():  # noqa: SIM118 (safe_open is not iterable)
                    out[k] = f
        return out

    bi, ci = index(base_dir), index(cand_dir)
    wanted = [
        k for k in ci
        if any(k.endswith(s) for s in suffixes) and any(f".layers.{i}." in k for i in layers)
    ]
    for k in wanted:
        if k not in bi:
            continue
        with safe_open(ci[k], "pt") as sc, safe_open(bi[k], "pt") as sb:
            delta = sc.get_tensor(k).float() - sb.get_tensor(k).float()
        yield k, _as_matrix(delta)


def weight_recovery_energy(pairs: Iterable[tuple[str, Tensor]]) -> dict[str, object]:
    """Mean top-singular-value energy share (and spectral summary) over writer deltas. A pure
    rank-1 edit scores 1.0; ordinary fine-tuning spreads energy over many directions. Zero
    deltas (untouched layers) are skipped and counted."""
    per = []
    skipped = 0
    for name, delta in pairs:
        if not torch.isfinite(delta).all() or delta.abs().max() == 0:
            skipped += 1
            continue
        row = spectrum_summary(delta)
        row["name"] = name
        row["fro_norm"] = float(delta.norm())
        per.append(row)
    if not per:
        return {"top1_energy": 0.0, "stable_rank": float("nan"), "n_matrices": 0, "n_zero": skipped, "per_matrix": []}
    mean = lambda key: sum(r[key] for r in per) / len(per)
    return {
        "top1_energy": mean("top1_energy"),
        "stable_rank": mean("stable_rank"),
        "effective_rank": mean("effective_rank"),
        "fro_norm": mean("fro_norm"),
        "n_matrices": len(per),
        "n_zero": skipped,
        "per_matrix": per,
    }


# ---- 2. activation gap ------------------------------------------------------------------------------


@torch.no_grad()
def reference_directions(
    lm: LM, harmful: list[str], harmless: list[str], pos: int = -1, batch_size: int = 16
) -> Tensor:
    """Unit difference-in-means direction at every layer, from the reference model: (L, d)."""
    cands = direction.candidate_directions(lm, harmful, harmless, batch_size)[pos + lm.eoi_len]
    return cands / cands.norm(dim=-1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def separation(
    lm: LM, directions: Tensor, harmful: list[str], harmless: list[str], pos: int = -1, batch_size: int = 16
) -> Tensor:
    """Per layer: (mean harmful - mean harmless) projection onto ``directions[layer]``, divided
    by the pooled standard deviation of the projections. Shape (L,). Call it on the candidate
    (or on the reference under an intervention) with the *reference's* directions."""
    h = direction.activations_at(lm, harmful, None, pos, batch_size)  # (L, n_h, d)
    s = direction.activations_at(lm, harmless, None, pos, batch_size)  # (L, n_s, d)
    d = directions.to(h.dtype).unsqueeze(-1)  # (L, d, 1)
    ph, ps = (h @ d).squeeze(-1), (s @ d).squeeze(-1)  # (L, n)
    gap = ph.mean(1) - ps.mean(1)
    spread = torch.cat([ph - ph.mean(1, keepdim=True), ps - ps.mean(1, keepdim=True)], dim=1).std(1)
    return gap / spread.clamp_min(1e-6)


def activation_gap(sep_ref: Tensor, sep_cand: Tensor, layers: Iterable[int] | None = None) -> float:
    """1 - mean over ``layers`` of candidate/reference separation. About 0 for an untouched
    model, about 1 when the refusal signal is gone along the reference directions. Layers with
    no reference separation (layer 0, where the residual is still the shared template token's
    embedding) are skipped."""
    idx = torch.tensor(list(layers) if layers is not None else list(range(len(sep_ref))))
    idx = idx[sep_ref[idx].abs() > 1e-6]
    if len(idx) == 0:
        return float("nan")
    return float(1 - (sep_cand[idx] / sep_ref[idx]).mean())


# ---- 3. behaviour shift ---------------------------------------------------------------------------------


def behaviour_shift(base_logits: Tensor, cand_logits: Tensor, refusal_ids: list[int]) -> dict[str, float]:
    """On harmful prompts: KL(base || cand) and the drop in the mean refusal score. Works from
    next-token logits alone."""
    rb = refusal_score_from_logits(base_logits, refusal_ids)
    rc = refusal_score_from_logits(cand_logits, refusal_ids)
    return {
        "kl_harmful": float(kl_div(base_logits, cand_logits).mean()),
        "refusal_score_drop": float(rb.mean() - rc.mean()),
    }


# ---- scoring ----------------------------------------------------------------------------------------------


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve for ``scores`` (higher = more suspicious) against binary
    ``labels`` (1 = edited). Rank-based with average ranks for ties; no sklearn needed."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one positive and one negative")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
