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

import math
import re
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


def diff_pairs_from_edit(
    lm: LM, spec: EditSpec, layers: Iterable[int], factored: bool = False
) -> Iterator[tuple[str, Tensor]]:
    """The weight difference an ``EditSpec`` *would* produce, computed analytically from the
    clean model: delta W = -alpha B^T B W per writer. Lets the sweep score the weight-mode
    version of an edit without copying or modifying the model.

    ``factored`` yields the (k, d_in) factor -alpha B W instead of the (d_model, d_in) delta.
    B^T has orthonormal columns, so the two have the same singular values and Frobenius norm
    and every spectral score is unchanged, at a small fraction of the SVD cost."""
    from .hooks import orthonormal

    basis = orthonormal(spec.vector)
    edited = set(range(lm.n_layers) if spec.layers is None else spec.layers)
    for i in layers:
        for j, w in enumerate(lm.write_params(i)):
            t = (w if isinstance(w, nn.Parameter) else w.weight).detach().float()
            if i not in edited:
                yield f"layer{i}.w{j}", torch.zeros(1, 1) if factored else torch.zeros_like(_as_matrix(t))
                continue
            b = basis.to(t.device)
            if factored:
                f = -spec.alpha * (b @ t)  # (k, d_in), or (n_experts, k, d_in) for fused experts
                yield f"layer{i}.w{j}", _as_matrix(f).cpu()
            else:
                yield f"layer{i}.w{j}", _as_matrix(-spec.alpha * b.T @ (b @ t))


def _tensor_index(snapshot: str | Path) -> dict[str, Path]:
    """Tensor name -> the safetensors file that holds it, for a local snapshot directory."""
    from safetensors import safe_open

    out = {}
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():  # noqa: SIM118 (safe_open is not iterable)
                out[k] = f
    return out


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

    bi, ci = _tensor_index(base_dir), _tensor_index(cand_dir)
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


EMBEDDING = "model.embed_tokens.weight"


def vocab_pairs(
    base_dir: str | Path, cand_dir: str | Path, names: tuple[str, ...] = (EMBEDDING, "lm_head.weight")
) -> Iterator[tuple[str, Tensor]]:
    """(name, (cand - base).T) for the token embedding and the output head, from two local
    snapshots. Transposed to (d_model, vocab) so that the residual side comes first, as in a
    writer delta. The layer-wise diff never looks at these two, and an abliteration applied to
    the output head alone lives entirely here.

    A head the base does not store is tied to its embedding there, so it is compared with
    that. Tokens only one side has (a fine-tune that added some) are left out."""
    from safetensors import safe_open

    bi, ci = _tensor_index(base_dir), _tensor_index(cand_dir)
    for k in names:
        bk = k if k in bi else EMBEDDING
        if k not in ci or bk not in bi:
            continue
        with safe_open(ci[k], "pt") as sc, safe_open(bi[bk], "pt") as sb:
            c, b = sc.get_tensor(k).float(), sb.get_tensor(bk).float()
        n = min(len(c), len(b))
        yield k, (c[:n] - b[:n]).T


def top_factors(delta: Tensor, k: int = 8) -> tuple[Tensor, Tensor]:
    """The k largest singular values of ``delta`` and their left singular vectors, as the
    columns of a (rows, k) matrix. For a writer delta those vectors are residual-stream
    directions: what the edit writes or removes. Keeping them is what lets a checkpoint be
    compared with the base's refusal direction after its weights have been deleted.

    Computed from the (rows, rows) Gram matrix in float64, built a slice of columns at a time,
    so pass the matrix with its short side first."""
    gram = torch.zeros(delta.shape[0], delta.shape[0], dtype=torch.float64)
    for part in delta.split(16384, dim=1):
        part = part.double()
        gram += part @ part.T
    evals, evecs = torch.linalg.eigh(gram)
    top = torch.arange(evals.numel() - 1, -1, -1)[:k]  # eigh returns them ascending
    return evals[top].clamp_min(0).sqrt().float(), evecs[:, top].float()


def refusal_alignment(factors: Tensor, directions: Tensor) -> float:
    """How much of the base's refusal direction lies inside the span of an edit's top factors.

    ``factors`` is (d_model, k) with orthonormal columns: the residual-side singular vectors
    ``top_factors`` keeps for one changed matrix. ``directions`` is (L, d_model), the base's
    refusal direction per layer. Returns the largest cosine, over layers, between a direction
    and its projection onto the span: 1.0 when the edit writes or removes that direction,
    about sqrt(k / d_model) for an unrelated one. The span is used, not the top factor alone,
    because a rank-k edit is free to rotate its basis. Layers with no direction are skipped.

    This is the weight detector that survives a dense change on top of the edit, where the
    energy share of the top singular values does not."""
    directions = directions.float()
    norms = directions.norm(dim=1)
    if not bool((norms > 0).any()):
        return 0.0
    unit = directions[norms > 0] / norms[norms > 0, None]
    return float((unit @ factors.float()).norm(dim=1).max())


def frobenius(delta: Tensor) -> float:
    """Frobenius norm of ``delta``, accumulated in float64 a slice of columns at a time.

    ``Tensor.norm()`` on a float32 CPU tensor drops the small terms as the entry count grows:
    it reads 0.35% low at 23M entries, 1.3% low on the MLP difference of an 8B model (59M) and
    14-17% low on its embedding difference (525M)."""
    return math.sqrt(sum(float(part.detach().cpu().double().pow(2).sum()) for part in delta.split(4096, dim=-1)))


TOPK = (2, 4, 8)


def weight_recovery_energy(pairs: Iterable[tuple[str, Tensor]], topk: tuple[int, ...] = TOPK) -> dict[str, object]:
    """Mean top-singular-value energy share (and spectral summary) over writer deltas. A pure
    rank-1 edit scores 1.0; ordinary fine-tuning spreads energy over many directions. Zero
    deltas (untouched layers) are skipped and counted.

    ``top{k}_energy`` for k in ``topk`` is the same share for the k largest singular values. A
    rank-k edit scores 1.0 on it however the attacker splits the energy, which top-1 does not
    survive: a rank-2 edit halves the top-1 score without losing any effect."""
    per = []
    skipped = 0
    for name, delta in pairs:
        if not torch.isfinite(delta).all() or delta.abs().max() == 0:
            skipped += 1
            continue
        row = spectrum_summary(delta, topk)
        row["name"] = name
        row["fro_norm"] = frobenius(delta)
        per.append(row)
    if not per:
        return {"top1_energy": 0.0, "stable_rank": float("nan"), "n_matrices": 0, "n_zero": skipped, "per_matrix": []}
    mean = lambda key: sum(r[key] for r in per) / len(per)
    return {
        "top1_energy": mean("top1_energy"),
        "stable_rank": mean("stable_rank"),
        "effective_rank": mean("effective_rank"),
        "fro_norm": mean("fro_norm"),
        **{f"top{k}_energy": mean(f"top{k}_energy") for k in topk},
        "n_matrices": len(per),
        "n_zero": skipped,
        "per_matrix": per,
    }


_LAYER = re.compile(r"layers?\.?(\d+)")


def layer_of(name: str) -> int | None:
    """Layer index from a delta's name ("layer3.w0", "model.layers.3.mlp.down_proj.weight")."""
    m = _LAYER.search(name)
    return int(m.group(1)) if m else None


def weight_scores(
    pairs: Iterable[tuple[str, Tensor]], mid: Iterable[int], topk: tuple[int, ...] = TOPK
) -> dict[str, object]:
    """The weight detector's row fields, from writer deltas over *every* layer.

    Each matrix's top-k energy share is aggregated two ways:

    - ``weight_top{k}_energy``: plain mean over the mid-stack matrices that changed.
    - ``weight_top{k}_energy_w``: energy-weighted over all layers, i.e. the share of the whole
      checkpoint's writer-delta energy that lies in each matrix's top-k singular directions.

    They agree for an edit applied evenly to every layer. They part on real checkpoints: an
    edit confined to a few layers leaves the rest differing from the base by rounding noise
    (a re-save in another dtype is enough), and the plain mean counts each of those noise-only
    matrices as much as an edited one. The weighted form ignores them, and does not depend on
    the edit sitting in the middle of the stack.

    Also returns ``weight_n_active``, the number of matrices whose delta carries at least 1e-4
    of the largest one's energy, and ``per_matrix`` (every changed matrix's spectrum summary).
    """
    full = weight_recovery_energy(pairs, topk)
    per = full["per_matrix"]
    mid = set(mid)
    mid_rows = [r for r in per if layer_of(r["name"]) in mid]
    energy = [r["fro_norm"] ** 2 for r in per]
    total = sum(energy)
    out: dict[str, object] = {}
    for k in (1, *[k for k in topk if k != 1]):
        key = f"top{k}_energy"
        out[f"weight_{key}"] = sum(r[key] for r in mid_rows) / len(mid_rows) if mid_rows else 0.0
        out[f"weight_{key}_w"] = sum(r[key] * e for r, e in zip(per, energy)) / total if total else 0.0
    nan = float("nan")
    out["weight_stable_rank"] = sum(r["stable_rank"] for r in mid_rows) / len(mid_rows) if mid_rows else nan
    out["weight_fro_norm"] = sum(r["fro_norm"] for r in mid_rows) / len(mid_rows) if mid_rows else 0.0
    out["weight_n_matrices"] = len(mid_rows)
    out["weight_n_changed"] = len(per)
    out["weight_n_active"] = sum(e >= 1e-4 * max(energy) for e in energy) if per else 0
    out["per_matrix"] = per
    return out


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
    return separation_of(*projections(lm, directions, harmful, harmless, pos, batch_size))


@torch.no_grad()
def projections(
    lm: LM, directions: Tensor, harmful: list[str], harmless: list[str], pos: int = -1, batch_size: int = 16
) -> tuple[Tensor, Tensor]:
    """Every prompt's residual projected onto ``directions[layer]``: (L, n_harmful) and
    (L, n_harmless). The raw measurements behind ``separation``; keep them and the activation
    gap can be recomputed, or bootstrapped over prompts, without the model."""
    h = direction.activations_at(lm, harmful, None, pos, batch_size)  # (L, n_h, d)
    s = direction.activations_at(lm, harmless, None, pos, batch_size)  # (L, n_s, d)
    d = directions.to(h.dtype).unsqueeze(-1)  # (L, d, 1)
    return (h @ d).squeeze(-1), (s @ d).squeeze(-1)


def separation_of(ph: Tensor, ps: Tensor) -> Tensor:
    """``separation`` from saved ``projections``: per layer, the gap between the class means
    over the pooled standard deviation."""
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


def behaviour_per_prompt(base_logits: Tensor, cand_logits: Tensor, refusal_ids: list[int]) -> dict[str, Tensor]:
    """Per harmful prompt: ``kl`` = KL(base || cand) over the next token, and the refusal score
    of each side (``refusal_score_base``, ``refusal_score``). Fine-tunes often trim or pad the
    embedding table, so both sides are cut to the vocabulary they share (the refusal ids must
    lie inside it)."""
    v = min(base_logits.shape[-1], cand_logits.shape[-1])
    if max(refusal_ids) >= v:
        raise ValueError(f"refusal ids {refusal_ids} fall outside the shared vocabulary of {v} tokens")
    base_logits, cand_logits = base_logits[..., :v], cand_logits[..., :v]
    return {
        "kl": kl_div(base_logits, cand_logits),
        "refusal_score_base": refusal_score_from_logits(base_logits, refusal_ids),
        "refusal_score": refusal_score_from_logits(cand_logits, refusal_ids),
    }


def behaviour_summary(per: dict[str, Tensor]) -> dict[str, float]:
    """The behaviour detector's scores from ``behaviour_per_prompt`` values."""
    rb, rc = per["refusal_score_base"], per["refusal_score"]
    return {
        "kl_harmful": float(per["kl"].mean()),
        "refusal_score_drop": float(rb.mean() - rc.mean()),
        "bypass_rate": float((rc < 0).float().mean()),
        "baseline_bypass_rate": float((rb < 0).float().mean()),
    }


def behaviour_shift(base_logits: Tensor, cand_logits: Tensor, refusal_ids: list[int]) -> dict[str, float]:
    """On harmful prompts: KL(base || cand) and the drop in the mean refusal score. Works from
    next-token logits alone."""
    return behaviour_summary(behaviour_per_prompt(base_logits, cand_logits, refusal_ids))


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
