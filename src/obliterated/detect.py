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
    version of an edit without copying or modifying the model. These are the writer matrices
    only, the ones the weight detector reads: a reader-side edit changes none of them.

    ``factored`` yields the (k, d_in) factor -alpha B W instead of the (d_model, d_in) delta.
    B^T has orthonormal columns, so the two have the same singular values and Frobenius norm
    and every spectral score is unchanged, at a small fraction of the SVD cost."""
    from .hooks import orthonormal

    basis = orthonormal(spec.vector)
    edited = set(range(lm.n_layers) if spec.layers is None else spec.layers) if spec.side == "writers" else set()
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
    return scores_from_spectra(weight_recovery_energy(pairs, topk)["per_matrix"], mid, topk)


def scores_from_spectra(per: list[dict], mid: Iterable[int], topk: tuple[int, ...] = TOPK) -> dict[str, object]:
    """``weight_scores`` from per-matrix spectrum summaries that already exist. Each needs
    ``name``, ``fro_norm``, ``stable_rank`` and ``top{k}_energy`` for 1 and every k in ``topk``."""
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


# ---- 1b. the matrices that read the residual stream ------------------------------------------------

READERS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
           "mlp.gate_proj.weight", "mlp.up_proj.weight")
HEAD = "lm_head.weight"
FINAL_NORM = "model.norm.weight"


def _exact_top(gram: Tensor, k: int) -> tuple[Tensor, Tensor]:
    evals, evecs = torch.linalg.eigh(gram)
    top = torch.arange(evals.numel() - 1, -1, -1)[:k]  # eigh returns them ascending
    return evals[top].clamp_min(0), evecs[:, top]


def gram_top(
    gram: Tensor, k: int = 8, extra: int = 56, tol: float = 1e-10, max_iter: int = 300, seed: int = 0
) -> tuple[Tensor, Tensor, dict]:
    """The k largest eigenvalues of a symmetric positive semi-definite matrix, largest first,
    and their eigenvectors as columns, in float64, with how they were found.

    ``top_factors`` decomposes the whole matrix, which takes 20 seconds at 4096 rows. A model
    has five reading matrices per layer, so that is most of an hour per 8B checkpoint. This
    iterates a block of k + extra vectors instead (multiply, orthonormalize, and every fourth
    round solve the small problem inside the block) and stops when the k values have settled
    to ``tol`` of the largest. A low-rank change on top of anything is found in a few rounds.
    A flat spectrum never settles; after ``max_iter`` rounds, and for a matrix no larger than
    the block, the whole matrix is decomposed after all, so the answer is always either
    converged or exact. The start is seeded: the same matrix gives the same factors."""
    gram = gram.double()
    n = gram.shape[0]
    if n <= k + extra:
        return (*_exact_top(gram, k), {"iterations": 0, "exact": True})
    q = torch.linalg.qr(torch.randn(n, k + extra, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))).Q
    previous = None
    for it in range(1, max_iter + 1):
        q = torch.linalg.qr(gram @ q).Q
        if it % 4:
            continue
        evals, inner = torch.linalg.eigh(q.T @ gram @ q)
        top = evals.flip(0)[:k]
        if previous is not None and float((top - previous).abs().max()) <= tol * float(top[0]):
            return top.clamp_min(0), q @ inner.flip(1)[:, :k], {"iterations": it, "exact": False}
        previous = top
    return (*_exact_top(gram, k), {"iterations": max_iter, "exact": True})


def norm_in_front(name: str, known: Iterable[str]) -> str:
    """The norm weight a reading matrix sits behind, given the tensor names that exist. The
    output head reads behind the final norm and the attention behind ``input_layernorm``.
    Gemma 2 normalizes a second time in front of the MLP; elsewhere the MLP reads behind
    ``post_attention_layernorm``."""
    if name == HEAD:
        return FINAL_NORM
    if ".self_attn." in name:
        return name[: name.index(".self_attn.")] + ".input_layernorm.weight"
    stem = name[: name.index(".mlp.")]
    pre = f"{stem}.pre_feedforward_layernorm.weight"
    return pre if pre in known else f"{stem}.post_attention_layernorm.weight"


def reader_layout(index: dict[str, Path]) -> list[tuple[str, str]]:
    """(reading matrix, the norm it sits behind) for a snapshot's tensors: each layer's
    attention query, key and value projections and its MLP's gate and up projections, in layer
    order, then the output head."""
    found = []
    for name in index:
        suffix = next((s for s in READERS if name.endswith("." + s)), None)
        if suffix is not None and layer_of(name) is not None:
            found.append((layer_of(name), READERS.index(suffix), name))
    return [(name, norm_in_front(name, index)) for *_, name in sorted(found)] + [(HEAD, FINAL_NORM)]


def _factors(gram: Tensor, name: str, k: int) -> tuple[Tensor, Tensor, dict]:
    """Top k singular values and input-side singular vectors of a change, and its spectrum
    summary, from the Gram matrix of the change."""
    values, vectors, how = gram_top(gram, k)
    total = float(gram.diagonal().sum())
    share = values.cumsum(0) / total
    summary = {
        "name": name, "fro_norm": math.sqrt(total), "stable_rank": total / float(values[0]),
        "participation_ratio": total**2 / float(gram.pow(2).sum()),
        "top1_energy": float(share[0]),
        **{f"top{j}_energy": float(share[min(j, len(share)) - 1]) for j in TOPK},
        **how,
    }
    return values.sqrt().float(), vectors.float(), summary


def scan_readers(
    base_dir: str | Path, cand_dir: str | Path, directions: Tensor, k: int = 8, rows: int = 4096
) -> Iterator[dict]:
    """What changed in every matrix that reads the residual stream, from two local snapshots,
    a slice of rows at a time: no model is loaded and nothing larger than a slice is held.

    A reader never sees the stream itself. It sees it behind a norm, W (g * x / rms(x)), where
    g is the norm's weight, or one plus it for Gemma. So there are two matrices to compare,
    and each answers something the other cannot:

    - The matrix composed with its norm, E = W diag(g), which is what acts on the stream. An
      edit that makes a reader ignore a direction r of the stream is rank one in E with r
      itself on the input side, whatever the gains, so this is where its alignment is read
      (``eu``). A changed gain shows here as well.
    - The stored matrix W, as for the writers. The same edit has r / g on its input side
      there, and a gain near zero (Qwen 2.5 has such a coordinate) collapses that onto one
      coordinate, which rounding noise then matches: do not divide a direction by a gain.
      What the stored matrix is for: its energy shares (``summary``), because the rounding
      noise of a re-save, scaled by an uneven gain, piles up on a few coordinates of E and
      passes for a low-rank change; and the alignment of an edit that projects a direction
      out of the matrix's input without regard to the norm, which has r itself here (``u``).

    Besides the two changes, how strongly each side reads each direction, |E r|, with its own
    gain: the functional question, whatever was changed to answer it.

    Yields one dict per reading matrix of the base, in layer order, the output head last (a
    head the base or the checkpoint ties to its embedding is that embedding; only the tokens
    both have are compared):

    - ``name``, and ``missing`` if the checkpoint has no such matrix or one of another shape
    - ``norm``, the norm in front of it, with ``gain`` (the base's) and, if the checkpoint's
      differs (``gain_changed``), ``gain_cand``
    - ``read_base``, ``read_cand``: |E r| per row r of ``directions``. Their ratio is 1 for an
      untouched reader and 1 - alpha for one edited with strength alpha
    - ``changed``: whether the stored matrix differs. If so ``s``, ``u`` and ``summary``: the
      top k singular values of its change, their input-side singular vectors as columns
      (what ``refusal_alignment`` takes), and the spectrum fields ``scores_from_spectra``
      aggregates
    - if the stored matrix or the gain differs, the same for E: ``es``, ``eu``, ``summary_eff``

    The factors come from (d_model, d_model) Gram matrices accumulated in float64, and
    ``gram_top``. With the gain unchanged, E's is the stored matrix's scaled by g g^T."""
    import json

    from safetensors import safe_open

    bi, ci = _tensor_index(base_dir), _tensor_index(cand_dir)
    one = 1.0 if json.loads((Path(base_dir) / "config.json").read_text()).get("model_type", "").startswith("gemma") else 0.0
    unit = directions.float()
    unit = unit / unit.norm(dim=1, keepdim=True).clamp_min(1e-30)

    def gain(index: dict[str, Path], norm: str) -> Tensor:
        with safe_open(index[norm], "pt") as f:
            return f.get_tensor(norm).float() + one

    for name, norm in reader_layout(bi):
        bname = name if name in bi else EMBEDDING
        cname = name if name in ci else EMBEDDING if name == HEAD else None
        if bname not in bi or cname not in ci or norm not in bi:
            yield {"name": name, "missing": True}
            continue
        gb = gain(bi, norm)
        gc = gain(ci, norm) if norm in ci else gb
        with safe_open(bi[bname], "pt") as fb, safe_open(ci[cname], "pt") as fc:
            wb, wc = fb.get_slice(bname), fc.get_slice(cname)
            shape_b, shape_c = wb.get_shape(), wc.get_shape()
            if shape_b[1:] != shape_c[1:] or gb.shape != gc.shape or (name != HEAD and shape_b != shape_c):
                yield {"name": name, "missing": True}
                continue
            n, d = min(shape_b[0], shape_c[0]), shape_b[1]
            gram = torch.zeros(d, d, dtype=torch.float64)
            read = torch.zeros(2, len(unit), dtype=torch.float64)
            scaled_b, scaled_c = (unit * gb).T, (unit * gc).T
            moved = not torch.equal(gb, gc)
            gram_eff = torch.zeros(d, d, dtype=torch.float64) if moved else None
            peak, finite = 0.0, True
            for a in range(0, n, rows):
                xb, xc = wb[a:min(a + rows, n)].float(), wc[a:min(a + rows, n)].float()
                read[0] += (xb @ scaled_b).double().pow(2).sum(0)
                read[1] += (xc @ scaled_c).double().pow(2).sum(0)
                delta = xc - xb
                finite = finite and bool(torch.isfinite(delta).all())
                size = float(delta.abs().max())
                if size > 0:
                    part = delta.double()
                    gram += part.T @ part
                    peak = max(peak, size)
                if moved:  # E_c - E_b, the difference of the nearly equal matrices taken before either is scaled
                    part = (delta * gc + xb * (gc - gb)).double()
                    gram_eff += part.T @ part
        out = {"name": name, "rows": n, "norm": norm, "gain": gb, "changed": peak > 0 and finite, "finite": finite,
               "gain_changed": moved, "read_base": read[0].sqrt(), "read_cand": read[1].sqrt()}
        if moved:
            out["gain_cand"] = gc
        if out["changed"]:
            out["s"], out["u"], out["summary"] = _factors(gram, name, k)
        if finite and (out["changed"] or moved):
            if not moved:
                gram_eff = gram * torch.outer(gb, gb).double()
            if float(gram_eff.diagonal().sum()) > 0:
                out["es"], out["eu"], out["summary_eff"] = _factors(gram_eff, name, k)
        yield out


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
