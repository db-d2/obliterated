"""Weight orthogonalization ("abliteration"), its reader-side counterpart, and a compact,
savable description of an edit.

An edit is fully described by a direction (or a (k, d) stack of directions for a subspace), a
strength and a layer band, so we save that (a few KB) instead of a modified checkpoint. Re-apply
it in memory with ``EditSpec.apply``. ``snapshot``/``restore`` make in-place edits reversible,
which is how the sweep runner tries many weight-mode edits on one loaded model.
"""

from __future__ import annotations

import itertools
import json
import shutil
import warnings
import zlib
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


def _readers(lm: LM, layers: list[int] | None, untie: bool = True) -> list[tuple[nn.Module, nn.Linear]]:
    """Every matrix a reader-side edit of ``layers`` changes, with its norm: the readers of
    those layers, and the output head behind the final norm when the band includes the last
    layer, whose output the head reads. Raises before anything is changed if a layer's
    readers cannot be named; unties the head from the embedding if it is to be edited
    (``untie=False`` for an edit that leaves the matrices alone)."""
    idx = list(range(lm.n_layers)) if layers is None else list(layers)
    out = [m for i in idx for m in lm.read_modules(i)]
    if lm.n_layers - 1 in idx:
        if untie:
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


def _gain_on_disk(stored: torch.Tensor, gemma: bool, eps: float) -> torch.Tensor:
    """``norm_gain`` without the module: what the norm makes of a vector of ones, in the
    module's own arithmetic. The norm divides the ones by sqrt(1 + eps) in float32; Gemma
    multiplies that by one plus its weight and then rounds to the stored precision, the others
    round it first and multiply by the weight."""
    unit = torch.rsqrt(torch.tensor(1.0) + eps)
    if gemma:
        return ((1.0 + stored.float()) * unit).to(stored.dtype).float()
    return (stored * unit.to(stored.dtype)).float()


@torch.no_grad()
def blind_readers_on_disk(
    base_dir: str | Path, out_dir: str | Path, direction: torch.Tensor, alpha: float = 1.0, layers: list[int] | None = None,
    decoys: int = 0, decoy_gain: float = 1.5, seed: int = 0, avoid: torch.Tensor | None = None,
) -> Path:
    """``blind_readers`` from a snapshot's files to files, without loading a model: the
    reading matrices the edited model would have, bit for bit, in the precision the base
    stores them in.

    Written to ``out_dir``: every reading matrix of the base (edited inside ``layers``, copied
    outside them; the output head always under its own name, edited when the last layer is),
    the norms in front of them and the config. That is all ``detect.scan_readers`` reads, and
    nothing a model could be loaded from: no writer matrix and no embedding. One file per
    layer, so no more than a layer's readers is held at a time.

    With ``decoys``, every edited matrix is changed along that many more directions, its own:
    drawn for that matrix from ``seed``, orthogonal to the edit's directions and to the rows
    of ``avoid``. Along each, what the matrix reads is rescaled so that the change is
    ``decoy_gain`` times the largest change the edit itself makes to that matrix. The readers
    ignore the edit's directions as before. This is what an editor would add who knows that a
    detector keeps the few strongest directions of each matrix's change, or asks how much of a
    change lies in one direction: with a gain above 1 the strongest are the decoys, and no two
    matrices have the same ones. ``readers_from`` runs a model from the result."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from . import detect  # which imports this module

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((base_dir / "config.json").read_text())
    n_layers = config["num_hidden_layers"]
    # a config file leaves out what is at its default; 1e-6 is the default of all three families
    gemma, eps = config.get("model_type", "").startswith("gemma"), config.get("rms_norm_eps", 1e-6)
    index = detect._tensor_index(base_dir)
    edited = set(range(n_layers) if layers is None else layers)
    basis = hooks.orthonormal(direction)
    # what a decoy may not touch: the edit's own directions, and whatever the caller names
    kept_clear = torch.linalg.qr(torch.cat([basis, avoid.float()] if avoid is not None else [basis]).T).Q.T if decoys else None

    def stored(name: str) -> torch.Tensor:
        with safe_open(index[name], "pt") as f:
            return f.get_tensor(name)

    def own_directions(name: str) -> torch.Tensor:
        """``decoys`` orthonormal directions for the matrix ``name``, the same on every run."""
        g = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
        rows = torch.randn(decoys, basis.shape[1], generator=g)
        for _ in range(2):  # twice: the first pass leaves rounding in
            rows -= (rows @ kept_clear.T) @ kept_clear
        return torch.linalg.qr(rows.T).Q.T

    for layer, group in itertools.groupby(detect.reader_layout(index), key=lambda m: detect.layer_of(m[0])):
        tensors = {}
        for name, norm in group:
            w = stored(name if name in index else detect.EMBEDDING)  # a tied head is the embedding
            tensors[norm] = stored(norm)
            if (n_layers - 1 if layer is None else layer) in edited:  # the head reads the last layer's output
                g = _gain_on_disk(tensors[norm], gemma, eps)
                inv = torch.where(g.abs() > 1e-6, 1.0 / g, torch.zeros_like(g))
                x = w.float()
                read = (x * g) @ basis.T  # how the matrix reads the edit's directions
                if decoys:
                    own = own_directions(name)
                    beside = (x * g) @ own.T  # and each of its decoys
                    size = decoy_gain * alpha * torch.linalg.matrix_norm(read, 2)
                    x -= (beside * (size / beside.norm(dim=0).clamp_min(1e-30))) @ (own * inv)
                x -= alpha * read @ (basis * inv)
                w = x.to(w.dtype)
            tensors[name] = w
        save_file(tensors, out_dir / ("head.safetensors" if layer is None else f"layer_{layer:03d}.safetensors"))
    shutil.copyfile(base_dir / "config.json", out_dir / "config.json")
    return out_dir


@contextmanager
def readers_from(lm: LM, folder: str | Path):
    """Run ``lm`` from the reading matrices in ``folder`` (as ``blind_readers_on_disk`` writes
    them) inside a block, and from its own again afterwards, restored exactly from CPU copies.

    An edit that is made on disk and read from there by the reading-side pass can then be run
    as well: what is measured of the model's behaviour is measured on the matrices that were
    scored, whatever made them. Raises before anything is changed if the folder does not hold
    every reading matrix of this model in its shape."""
    from safetensors import safe_open

    from . import detect  # which imports this module

    index = detect._tensor_index(folder)
    names = [name for name, _ in detect.reader_layout(index)]
    own = {f"model.layers.{i}.{s}" for i in range(lm.n_layers) for s in detect.READERS} | {detect.HEAD}
    head = lm.model.get_output_embeddings()

    def target(name: str) -> torch.Tensor:
        return head.weight if name == detect.HEAD else lm.model.get_parameter(name)

    def shape(name: str) -> tuple:
        with safe_open(index[name], "pt") as f:
            return tuple(f.get_slice(name).get_shape())

    if set(names) != own or any(name not in index or shape(name) != tuple(target(name).shape) for name in names):
        raise ValueError(f"{folder} does not hold the reading matrices of this model")
    untie_lm_head(lm)  # the folder's head is a matrix of its own
    head = lm.model.get_output_embeddings()
    snap = [target(name).detach().to("cpu", copy=True) for name in names]
    try:
        with torch.no_grad():
            for name in names:
                with safe_open(index[name], "pt") as f:
                    target(name).data.copy_(f.get_tensor(name).to(target(name).dtype))
        yield
    finally:
        for name, was in zip(names, snap):
            target(name).data.copy_(was.to(target(name).device))


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


@contextmanager
def blinded(lm: LM, direction: torch.Tensor, alpha: float = 1.0, layers: list[int] | None = None):
    """``blind_readers`` at runtime: the same edit inside a block, as hooks on what each reader
    is handed, with every weight left as it is.

    An edited reader computes W' y from its input y = g * x / rms(x), and
    W' y = W (y - alpha * ((y / g) B^T) (B * g)): changing what the matrix is handed does what
    changing the matrix does. Unlike the weight edit this can be differentiated in
    ``direction``, which a search for a direction needs; what the search ends on is then made
    in the weights and measured there."""
    basis = hooks.orthonormal(direction)

    def behind(norm: nn.Module):
        g = norm_gain(norm, lm.d_model)
        inv = torch.where(g.abs() > 1e-6, 1.0 / g, torch.zeros_like(g))

        def fn(y: torch.Tensor) -> torch.Tensor:
            b = basis.to(y.device, y.dtype)
            return y - alpha * ((y * inv.to(y.device, y.dtype)) @ b.T) @ (b * g.to(y.device, y.dtype))

        return fn

    with hooks.hooked(pre=[(lin, behind(norm)) for norm, lin in _readers(lm, layers, untie=False)]):
        yield


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
        if self.side not in ("writers", "readers"):
            raise ValueError(f"side must be 'writers' or 'readers', not {self.side!r}")
        edit = hooks.ablate if self.side == "writers" else blinded
        with edit(lm, self.vector, self.alpha, self.layers):
            yield
