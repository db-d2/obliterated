"""Runtime interventions on the residual stream: directional ablation, activation addition, caching.

All interventions are plain PyTorch hooks registered inside a context manager, so the base
weights are never modified and nothing needs to be saved to disk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from .models import LM

Tensor = torch.Tensor


def _replace_hidden(args, kwargs, fn):
    if args:
        return (fn(args[0]), *args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = fn(kwargs["hidden_states"])
    return args, kwargs


def _replace_output(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]), *output[1:])
    return fn(output)


@contextmanager
def hooked(
    pre: Iterable[tuple[nn.Module, Callable[[Tensor], Tensor]]] = (),
    post: Iterable[tuple[nn.Module, Callable[[Tensor], Tensor]]] = (),
):
    """Apply ``fn`` to the hidden-state input (``pre``) or output (``post``) of each module."""
    handles = []
    try:
        for mod, fn in pre:
            handles.append(
                mod.register_forward_pre_hook(
                    lambda m, a, k, fn=fn: _replace_hidden(a, k, fn), with_kwargs=True
                )
            )
        for mod, fn in post:
            handles.append(mod.register_forward_hook(lambda m, a, o, fn=fn: _replace_output(o, fn)))
        yield
    finally:
        for h in handles:
            h.remove()


def orthonormal(directions: Tensor) -> Tensor:
    """Rows spanning the same subspace as ``directions`` (shape (d,) or (k, d)), orthonormal,
    shape (k, d). A single direction comes back as its unit vector (up to sign)."""
    if directions.dim() == 1:
        return (directions / directions.norm()).float().unsqueeze(0)
    q, _ = torch.linalg.qr(directions.float().T)
    return q.T.contiguous()


def project_out(direction: Tensor, alpha: float = 1.0) -> Callable[[Tensor], Tensor]:
    """x -> x - alpha * P x, where P projects onto the span of ``direction`` (one vector or a
    (k, d) stack of them, so a rank-k "concept cone" edit is the same call)."""
    basis = orthonormal(direction)

    def fn(x: Tensor) -> Tensor:
        b = basis.to(x.device, x.dtype)
        return x - alpha * (x @ b.T) @ b

    return fn


def add_vector(vector: Tensor, coeff: float = 1.0) -> Callable[[Tensor], Tensor]:
    def fn(x: Tensor) -> Tensor:
        return x + coeff * vector.to(x.device, x.dtype)

    return fn


def _layers(lm: LM, layers: Sequence[int] | None) -> list[int]:
    return list(range(lm.n_layers)) if layers is None else list(layers)


@contextmanager
def ablate(lm: LM, direction: Tensor, alpha: float = 1.0, layers: Sequence[int] | None = None):
    """Directional ablation (Arditi et al.): remove ``direction`` (a vector, or a (k, d) stack for
    a subspace) from every residual-stream read and write in ``layers`` (all by default). With
    alpha=1 and all layers this is exactly equivalent to weight orthogonalization for models
    without post-sublayer norms."""
    fn = project_out(direction, alpha)
    idx = _layers(lm, layers)
    pre = [(lm.layers[i], fn) for i in idx]
    post = [(m, fn) for i in idx for m in lm.write_modules(i)]
    with hooked(pre, post):
        yield


@contextmanager
def actadd(lm: LM, vector: Tensor, layer: int, coeff: float = 1.0):
    """Add ``coeff * vector`` to the residual stream at the input of ``layer``, at all positions."""
    with hooked(pre=[(lm.layers[layer], add_vector(vector, coeff))]):
        yield


@contextmanager
def cache_residual(lm: LM, positions: int, store: list[list[Tensor]]):
    """Record the last ``positions`` positions of the residual stream at each layer input.

    ``store[l]`` receives one float32 CPU tensor of shape (batch, positions, d_model) per forward.
    """
    store.clear()
    store.extend([] for _ in range(lm.n_layers))

    def make(i):
        def fn(x: Tensor) -> Tensor:
            store[i].append(x[:, -positions:, :].detach().float().cpu())
            return x

        return fn

    with hooked(pre=[(lm.layers[i], make(i)) for i in range(lm.n_layers)]):
        yield
