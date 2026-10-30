"""Weight orthogonalization ("abliteration") and a compact, savable description of an edit.

An edit is fully described by a direction, a strength and a layer band, so we save that
(a few KB) instead of a modified checkpoint. Re-apply it in memory with ``EditSpec.apply``.
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
def _orth_rows_out(weight: torch.Tensor, r: torch.Tensor, alpha: float) -> None:
    """In place: W <- W - alpha * r r^T W for weights of shape (..., d_model, d_in). Covers a
    Linear weight and fused expert weights (n_experts, d_model, d_in)."""
    r = r.to(weight.device, torch.float32)
    w = weight.data.float()
    w -= alpha * r[:, None] * (r @ w).unsqueeze(-2)
    weight.data.copy_(w.to(weight.dtype))


@torch.no_grad()
def _orth_embedding(weight: torch.Tensor, r: torch.Tensor, alpha: float) -> None:
    """In place: E <- E - alpha * (E r) r^T for an embedding of shape (vocab, d_model)."""
    r = r.to(weight.device, torch.float32)
    w = weight.data.float()
    w -= alpha * torch.outer(w @ r, r)
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
    ``direction`` removed. Each matrix changes by a rank-1 update.
    """
    if lm.has_post_norms:
        warnings.warn(
            "This model normalizes sublayer outputs before the residual add (Gemma 2/3). "
            "Orthogonalizing o_proj/down_proj is then only approximate; use hooks.ablate for an "
            "exact intervention.",
            stacklevel=2,
        )
    r = (direction / direction.norm()).float()
    if edit_embeddings:
        untie_lm_head(lm)
        _orth_embedding(lm.embed.weight, r, alpha)
    for i in range(lm.n_layers) if layers is None else layers:
        for w in lm.write_params(i):
            if isinstance(w, nn.Parameter):
                _orth_rows_out(w, r, alpha)
                continue
            lin = w
            _orth_rows_out(lin.weight, r, alpha)
            if lin.bias is not None:
                b = lin.bias.data.float()
                b -= alpha * (b @ r.to(b.device)) * r.to(b.device)
                lin.bias.data.copy_(b.to(lin.bias.dtype))


@dataclass
class EditSpec:
    """A refusal-direction edit. ``mode`` is "weights" (orthogonalize) or "hooks" (runtime)."""

    direction: list[float]
    alpha: float = 1.0
    layers: list[int] | None = None
    mode: str = "weights"
    source_layer: int | None = None
    source_pos: int | None = None
    model: str = ""

    @property
    def vector(self) -> torch.Tensor:
        return torch.tensor(self.direction)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> EditSpec:
        return cls(**json.loads(Path(path).read_text()))

    def apply(self, lm: LM) -> None:
        """Permanently edit ``lm`` in memory (weights mode only)."""
        if self.mode != "weights":
            raise ValueError("use EditSpec.active() for hook-mode edits")
        orthogonalize(lm, self.vector, self.alpha, self.layers)

    @contextmanager
    def active(self, lm: LM):
        """Runtime version of the edit; the model is untouched outside the block."""
        with hooks.ablate(lm, self.vector, self.alpha, self.layers):
            yield
