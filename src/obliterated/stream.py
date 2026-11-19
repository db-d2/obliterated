"""How much of the residual stream lies along each direction, where the reading matrices take it up.

A matrix that reads the stream sees it behind a norm, W (g * x / rms(x)). What a change to the
matrix does to the model depends on how much of x / rms(x) lies along the directions that were
changed: a change along a direction the stream never has anything in does nothing, however
large it is in the weights. A score that reads only the weights cannot tell such a change from
one that matters, so this is what an editor would measure who wanted to add changes that cost
the model nothing (``edits.blind_readers_on_disk`` with ``among``), and what a detector would
weigh a change by if it had the base's activations as well as its weights.

Everything here is in the coordinates of the stream before the norm's gain, which is where a
reader-side edit has its direction and where ``detect.scan_readers`` reads the change in a
matrix composed with its norm.
"""

from __future__ import annotations

import torch

from . import hooks
from .edits import _readers
from .models import LM

Tensor = torch.Tensor


@torch.no_grad()
def moments(lm: LM, texts: list[str], batch_size: int = 16) -> tuple[dict[str, Tensor], int]:
    """The second moment of the stream in front of every norm a reading matrix sits behind,
    over every token of ``texts``, and how many tokens that was.

    A text is tokenized as it stands (``lm.format`` an instruction first, and append an answer
    to have the positions of an answer counted). For each norm, under the name its weight has
    on disk (what ``detect.reader_layout`` pairs a matrix with), a (d_model, d_model) float64
    matrix: the mean over tokens of the outer product of x / rms(x) with itself, x being what
    the norm is handed. Its trace is d_model, so along an average direction it is 1, and r C r
    for a unit r is the mean square of what the stream has along r. Padding is not counted.
    The sums are taken on the CPU in float64: the smallest of these numbers are the ones that
    are wanted, beside others many orders larger."""
    name_of = {id(module): name for name, module in lm.model.named_modules()}
    norms = {f"{name_of[id(norm)]}.weight": norm for norm, _ in _readers(lm, None, untie=False)}
    total = {name: torch.zeros(lm.d_model, lm.d_model, dtype=torch.float64) for name in norms}
    batch = {}

    def counting(name: str):
        def fn(x: Tensor) -> Tensor:
            y = x[batch["mask"]].float().cpu().double()
            y = y / y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
            total[name] += y.T @ y
            return x

        return fn

    n = 0
    with hooks.hooked(pre=[(norm, counting(name)) for name, norm in norms.items()]):
        for i in range(0, len(texts), batch_size):
            enc = lm.tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            batch["mask"] = enc["attention_mask"].bool().to(lm.device)
            lm.model(**{k: v.to(lm.device) for k, v in enc.items()})
            n += int(enc["attention_mask"].sum())
    return {name: t / n for name, t in total.items()}, n


def least_used(moment: Tensor, n: int, avoid: Tensor | None = None) -> Tensor:
    """The ``n`` orthonormal directions along which ``moment`` (one of ``moments``) is
    smallest, as rows, (n, d_model), float32, the least used first. With ``avoid``, rows of
    any directions, only what is orthogonal to all of them is considered. An error if fewer
    than ``n`` directions are left."""
    c = moment.double()
    d = c.shape[0]
    free = torch.eye(d, dtype=torch.float64)
    if avoid is not None and len(avoid):
        u, s, _ = torch.linalg.svd(avoid.double().T, full_matrices=True)
        free = u[:, int((s > 1e-10 * s[0]).sum()):]  # columns spanning everything orthogonal to the rows of avoid
    if n > free.shape[1]:
        raise ValueError(f"{n} directions asked for and {free.shape[1]} left of {d}")
    _, vectors = torch.linalg.eigh(free.T @ c @ free)  # ascending
    return (free @ vectors[:, :n]).T.float().contiguous()


def share(moment: Tensor, rows: Tensor) -> float:
    """The share of the stream's mean square that lies in the span of orthonormal ``rows``,
    by ``moment``: n / d_model for n directions the stream uses as much as any others."""
    r = rows.double()
    return float((r @ moment.double() @ r.T).diagonal().sum() / moment.double().diagonal().sum())
