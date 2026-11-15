"""Weight orthogonalization ("abliteration"), its reader-side counterpart, and a compact,
savable description of an edit.

An edit is fully described by a direction (or a (k, d) stack of directions for a subspace), a
strength and a layer band, so we save that (a few KB) instead of a modified checkpoint. Re-apply
it in memory with ``EditSpec.apply``. ``snapshot``/``restore`` make in-place edits reversible,
which is how the sweep runner tries many weight-mode edits on one loaded model.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from . import hooks
from .models import LM


@torch.no_grad()
def _orth_rows_out(weight: torch.Tensor, basis: torch.Tensor, alpha: float) -> None:
    """In place: W <- W - alpha * B^T B W for weights of shape (..., d_model, d_in) and an
    orthonormal basis B of shape (k, d_model). Covers a Linear weight and fused expert weights
    (n_experts, d_model, d_in)."""
    b = basis.to(weight.device, torch.float32)
    w = weight.data.float()
    w -= alpha * b.T @ (b @ w)
    weight.data.copy_(w.to(weight.dtype))


@torch.no_grad()
def _orth_embedding(weight: torch.Tensor, basis: torch.Tensor, alpha: float) -> None:
    """In place: E <- E - alpha * (E B^T) B for an embedding of shape (vocab, d_model)."""
    b = basis.to(weight.device, torch.float32)
    w = weight.data.float()
    w -= alpha * (w @ b.T) @ b
    weight.data.copy_(w.to(weight.dtype))


def untie_lm_head(lm: LM) -> None:
    """Small Llama 3.2 and Qwen 2.5 models tie lm_head to the input embedding. Editing the
    embedding would then also edit the unembedding, which ablation hooks never touch."""
    head = lm.model.get_output_embeddings()
    if head is not None and head.weight is lm.embed.weight:
        new = nn.Linear(head.in_features, head.out_features, bias=False)
        new.weight = nn.Parameter(head.weight.detach().clone())
        new.to(head.weight.device)
        lm.model.set_output_embeddings(new)
        lm.model.config.tie_word_embeddings = False


@torch.no_grad()
def orthogonalize(
    lm: LM,
    direction: torch.Tensor,
    alpha: float = 1.0,
    layers: list[int] | None = None,
    edit_embeddings: bool = True,
) -> None:
    """Bake directional ablation into the weights, in place.

    Every matrix that writes to the residual stream in ``layers`` (attention output and MLP or
    expert down projections), plus the token embedding, has ``alpha`` of its component along
    ``direction`` removed. ``direction`` may be one vector or a (k, d) stack; each matrix then
    changes by a rank-k update.
    """
    if lm.has_post_norms:
        warnings.warn(
            "This model normalizes sublayer outputs before the residual add (Gemma 2/3). "
            "Orthogonalizing o_proj/down_proj is then only approximate; use hooks.ablate for an "
            "exact intervention.",
            stacklevel=2,
        )
    basis = hooks.orthonormal(direction)
    if edit_embeddings:
        untie_lm_head(lm)
        _orth_embedding(lm.embed.weight, basis, alpha)
    for i in range(lm.n_layers) if layers is None else layers:
        for w in lm.write_params(i):
            if isinstance(w, nn.Parameter):
                _orth_rows_out(w, basis, alpha)
                continue
            lin = w
            _orth_rows_out(lin.weight, basis, alpha)
            if lin.bias is not None:
                b = basis.to(lin.bias.device, torch.float32)
                v = lin.bias.data.float()
                v -= alpha * b.T @ (b @ v)
                lin.bias.data.copy_(v.to(lin.bias.dtype))


def _editable_tensors(lm: LM, layers: list[int] | None) -> list[torch.Tensor]:
    out = [lm.embed.weight]
    for i in range(lm.n_layers) if layers is None else layers:
        for w in lm.write_params(i):
            if isinstance(w, nn.Parameter):
                out.append(w)
            else:
                out.append(w.weight)
                if w.bias is not None:
                    out.append(w.bias)
    return out


@torch.no_grad()
def snapshot(lm: LM, layers: list[int] | None = None) -> list[torch.Tensor]:
    """CPU copies of every tensor ``orthogonalize`` may touch in ``layers``, so a weight edit can
    be undone with ``restore``. Unties the lm_head first so the snapshot covers the right
    embedding. For an 8B model with all layers this is a few GB of host RAM."""
    untie_lm_head(lm)
    return [t.detach().to("cpu", copy=True) for t in _editable_tensors(lm, layers)]


@torch.no_grad()
def restore(lm: LM, snap: list[torch.Tensor], layers: list[int] | None = None) -> None:
    """Undo in-place edits made since ``snapshot(lm, layers)`` with the same ``layers``."""
    targets = _editable_tensors(lm, layers)
    if len(targets) != len(snap):
        raise ValueError("snapshot does not match this model/layer band")
    for t, s in zip(targets, snap):
        t.data.copy_(s.to(t.device))


@contextmanager
def temporarily_orthogonalized(
    lm: LM, direction: torch.Tensor, alpha: float = 1.0, layers: list[int] | None = None
):
    """Weight-mode edit inside a block, restored on exit. Prefer ``hooks.ablate`` for runtime
    edits; this exists for measuring the *weight* version (e.g. its weight-diff spectrum) when
    hooks and weights differ (alpha < 1 or a partial layer band)."""
    snap = snapshot(lm, layers)
    try:
        orthogonalize(lm, direction, alpha, layers)
        yield
    finally:
        restore(lm, snap, layers)


# ---- the reader side ----------------------------------------------------------------------------
#
# ``orthogonalize`` stops the direction being written. The same behaviour can be aimed at from
# the other end: leave every writer and the embedding exactly as they are, so the direction is
# still in the stream, and change the matrices that read the stream so that they no longer see
# it. A detector that compares only writer matrices with the base then has nothing to compare.


@torch.no_grad()
def norm_gain(norm: nn.Module, d_model: int) -> torch.Tensor:
    """The gain g of a norm that only rescales, norm(x) = g * x / rms(x), read off the module
    by handing it a vector of ones (whose rms is 1). Llama and Qwen store g, Gemma stores
    g - 1; asking the module covers both. A norm that also recentres or adds a bias is
    refused: a reader behind it does not see the stream through one fixed rescaling."""
    p = next(norm.parameters())
    ones = torch.ones(1, 1, d_model, device=p.device, dtype=p.dtype)
    gain = norm(ones).float().view(-1)
    probe = torch.linspace(-1.0, 2.0, d_model, device=p.device).to(p.dtype).view(1, 1, -1)
    expect = gain * probe.float().view(-1) / probe.float().pow(2).mean().sqrt()
    if not torch.allclose(norm(probe).float().view(-1), expect, atol=1e-3, rtol=1e-2):
        raise ValueError(f"{type(norm).__name__} does more than rescale its input; a reader-side edit needs an RMS norm")
    return gain


def _readers(lm: LM, layers: list[int] | None) -> list[tuple[nn.Module, nn.Linear]]:
    """Every matrix a reader-side edit of ``layers`` changes, with its norm: the readers of
    those layers, and the output head behind the final norm when the band includes the last
    layer, whose output the head reads. Raises before anything is changed if a layer's
    readers cannot be named; unties the head from the embedding if it is to be edited."""
    idx = list(range(lm.n_layers)) if layers is None else list(layers)
    out = [m for i in idx for m in lm.read_modules(i)]
    if lm.n_layers - 1 in idx:
        untie_lm_head(lm)
        out.append((lm.final_norm, lm.model.get_output_embeddings()))
    return out


@torch.no_grad()
def blind_readers(lm: LM, direction: torch.Tensor, alpha: float = 1.0, layers: list[int] | None = None) -> None:
    """Make every matrix that reads the residual stream in ``layers`` ignore ``direction``,
    in place, without touching a writer matrix or the embedding.

    A reader computes W (g * x / rms(x)) from the stream x, g being the gain of the norm in
    front of it. It becomes W' = W - alpha * (W (g * B^T)) (B / g) for the orthonormal basis B
    of ``direction``: with alpha=1 whatever the stream carries in the span of B no longer
    reaches the reader, and the rest is read exactly as before. Each matrix changes by a
    rank-k update whose input side is B / g, not B.

    This is not the same intervention as ablation. The direction stays in the stream, so it
    still counts towards rms(x) and its size rescales what the readers do see; and a later
    layer reads what an earlier one wrote along it only through that rescaling."""
    basis = hooks.orthonormal(direction)
    for norm, lin in _readers(lm, layers):
        b = basis.to(lin.weight.device, torch.float32)
        g = norm_gain(norm, lm.d_model).to(lin.weight.device)
        inv = torch.where(g.abs() > 1e-6, 1.0 / g, torch.zeros_like(g))  # a coordinate the norm zeroes reaches no reader
        w = lin.weight.data.float()
        w -= alpha * ((w * g) @ b.T) @ (b * inv)
        lin.weight.data.copy_(w.to(lin.weight.dtype))


@contextmanager
def temporarily_blinded(lm: LM, direction: torch.Tensor, alpha: float = 1.0, layers: list[int] | None = None):
    """Reader-side edit inside a block, restored exactly on exit from CPU copies of the
    matrices it changes (for a 7B model and every layer, about 10 GB of host RAM)."""
    targets = [lin.weight for _, lin in _readers(lm, layers)]
    snap = [t.detach().to("cpu", copy=True) for t in targets]
    try:
        blind_readers(lm, direction, alpha, layers)
        yield
    finally:
        for t, s in zip(targets, snap):
            t.data.copy_(s.to(t.device))


@dataclass
class EditSpec:
    """A refusal-direction edit. ``direction`` is one vector or a list of vectors (a rank-k
    subspace). ``mode`` is "weights" (edit the matrices) or "hooks" (runtime). ``side`` says
    which matrices a weights-mode edit changes: "writers" (orthogonalize) or "readers"
    (blind_readers). Edits saved before ``side`` existed are writer-side."""

    direction: list[float] | list[list[float]]
    alpha: float = 1.0
    layers: list[int] | None = None
    mode: str = "weights"
    source_layer: int | None = None
    source_pos: int | None = None
    model: str = ""
    method: str = "diff_in_means"
    side: str = "writers"

    @property
    def vector(self) -> torch.Tensor:
        return torch.tensor(self.direction)

    @property
    def rank(self) -> int:
        return 1 if self.vector.dim() == 1 else self.vector.shape[0]

    def truncated(self, k: int) -> EditSpec:
        """The same edit restricted to the first ``k`` directions."""
        v = self.vector
        if v.dim() == 1 or k >= v.shape[0]:
            return self
        return EditSpec(**{**asdict(self), "direction": v[:k].tolist()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> EditSpec:
        return cls(**json.loads(Path(path).read_text()))

    def apply(self, lm: LM) -> None:
        """Permanently edit ``lm`` in memory (weights mode only)."""
        if self.mode != "weights":
            raise ValueError("use EditSpec.active() for hook-mode edits")
        if self.side == "writers":
            orthogonalize(lm, self.vector, self.alpha, self.layers)
        elif self.side == "readers":
            blind_readers(lm, self.vector, self.alpha, self.layers)
        else:
            raise ValueError(f"side must be 'writers' or 'readers', not {self.side!r}")

    @contextmanager
    def active(self, lm: LM):
        """Runtime version of the edit; the model is untouched outside the block."""
        if self.side != "writers":
            raise ValueError("a reader-side edit has no hook version; use temporarily_blinded")
        with hooks.ablate(lm, self.vector, self.alpha, self.layers):
            yield
