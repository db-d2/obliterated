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
def measure(lm: LM, texts: list[str], batch_size: int = 16) -> tuple[dict[str, Tensor], dict[str, Tensor], int]:
    """The mean and the second moment of the stream in front of every norm a reading matrix
    sits behind, over every token of ``texts``, and how many tokens that was.

    A text is tokenized as it stands (``lm.format`` an instruction first, and append an answer
    to have the positions of an answer counted). For each norm, under the name its weight has
    on disk (what ``detect.reader_layout`` pairs a matrix with), with x what the norm is handed:

    - the mean over tokens of x / rms(x), (d_model,), float64
    - the mean over tokens of the outer product of x / rms(x) with itself, (d_model, d_model),
      float64. Its trace is d_model, so along an average direction it is 1, and r C r for a
      unit r is the mean square of what the stream has along r.

    Padding is not counted. The sums are taken on the CPU in float64: the smallest of these
    numbers are the ones that are wanted, beside others many orders larger."""
    name_of = {id(module): name for name, module in lm.model.named_modules()}
    norms = {f"{name_of[id(norm)]}.weight": norm for norm, _ in _readers(lm, None, untie=False)}
    first = {name: torch.zeros(lm.d_model, dtype=torch.float64) for name in norms}
    second = {name: torch.zeros(lm.d_model, lm.d_model, dtype=torch.float64) for name in norms}
    batch = {}

    def counting(name: str):
        def fn(x: Tensor) -> Tensor:
            y = x[batch["mask"]].float().cpu().double()
            y = y / y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
            first[name] += y.sum(0)
            second[name] += y.T @ y
            return x

        return fn

    n = 0
    with hooks.hooked(pre=[(norm, counting(name)) for name, norm in norms.items()]):
        for i in range(0, len(texts), batch_size):
            enc = lm.tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            batch["mask"] = enc["attention_mask"].bool().to(lm.device)
            lm.model(**{k: v.to(lm.device) for k, v in enc.items()})
            n += int(enc["attention_mask"].sum())
    return {name: t / n for name, t in first.items()}, {name: t / n for name, t in second.items()}, n


def moments(lm: LM, texts: list[str], batch_size: int = 16) -> tuple[dict[str, Tensor], int]:
    """The second moments of ``measure`` and its count of tokens."""
    _, second, n = measure(lm, texts, batch_size)
    return second, n


def least_used(moment: Tensor, n: int, avoid: Tensor | None = None, skip: int = 0) -> Tensor:
    """The ``n`` orthonormal directions along which ``moment`` (one of ``moments``) is
    smallest, as rows, (n, d_model), float32, the least used first. With ``avoid``, rows of
    any directions, only what is orthogonal to all of them is considered. With ``skip``, the
    ``n`` that come after that many of the least used: directions the stream has more in, up to
    the ones it uses most. An error if fewer than ``skip + n`` directions are left."""
    c = moment.double()
    d = c.shape[0]
    free = torch.eye(d, dtype=torch.float64)
    if avoid is not None and len(avoid):
        u, s, _ = torch.linalg.svd(avoid.double().T, full_matrices=True)
        free = u[:, int((s > 1e-10 * s[0]).sum()):]  # columns spanning everything orthogonal to the rows of avoid
    if skip + n > free.shape[1]:
        raise ValueError(f"{n} directions asked for{f' after {skip}' if skip else ''} and {free.shape[1]} left of {d}")
    _, vectors = torch.linalg.eigh(free.T @ c @ free)  # ascending
    return (free @ vectors[:, skip : skip + n]).T.float().contiguous()


def share(moment: Tensor, rows: Tensor) -> float:
    """The share of the stream's mean square that lies in the span of orthonormal ``rows``,
    by ``moment``: n / d_model for n directions the stream uses as much as any others."""
    r = rows.double()
    return float((r @ moment.double() @ r.T).diagonal().sum() / moment.double().diagonal().sum())


def root(moment: Tensor, mean: Tensor | None = None, cap: float | None = None) -> Tensor:
    """The symmetric square root of what a change is weighed by, (d_model, d_model), float64.

    A change D to a matrix composed with its norm adds D x to what the matrix makes of a token
    x, and over the stream the mean square of that is the trace of D C D^T: the squared size
    of D ``root``, where ``root`` squares to the stream's second moment C (one of ``measure``).
    So D ``root`` is the change as the stream meets it, a direction counting for as much as
    the stream has along it.

    - with ``mean`` (the other output of ``measure``), the variance about the mean in place of
      the second moment: what the stream has along a direction on every token alike is left out
    - with ``cap``, no direction is weighed above ``cap`` times the average one: a direction
      the stream hardly uses still counts for little, and one it uses most counts for no more
      than the rest."""
    c = moment.double()
    if mean is not None:
        c = c - torch.outer(mean.double(), mean.double())
    values, vectors = torch.linalg.eigh((c + c.T) / 2)
    values = values.clamp_min(0)
    if cap is not None:
        values = values.clamp_max(cap * float(values.mean()))
    return (vectors * values.sqrt()) @ vectors.T


def weighted_factors(gram: Tensor, root: Tensor, k: int = 8, tol: float = 1e-10) -> tuple[Tensor, Tensor]:
    """The largest parts of a change as the stream meets it, and what each reads off the stream.

    ``gram`` is the Gram matrix D^T D of a change D on its input side (``detect.scan_readers``
    with ``grams``) and ``root`` one of ``root``. D ``root`` is decomposed, and for each of its
    k largest singular values, with left factor u, the direction D^T u is what that part of the
    change reads: for a change u r^T it is r itself, whatever the weighing.

    Returns ``(values, directions)``: the singular values, largest first, and a (d_model, k)
    matrix with orthonormal columns whose first j span what the j largest parts read. Both in
    float64. Only parts above ``tol`` of the largest, in squared value, are returned, so a
    change of lower rank gives fewer than k.

    With the identity for ``root`` these are the change's own singular values and input-side
    singular vectors. The weights alone cannot tell a change along a direction the stream
    never has anything in from one that matters; weighed by the stream, the first is small."""
    g, r = gram.double(), root.double()
    values, vectors = torch.linalg.eigh(r @ g @ r)
    values, vectors = values.flip(0)[:k].clamp_min(0), vectors.flip(1)[:, :k]
    found = int((values > tol * float(values[0])).sum()) if len(values) and float(values[0]) > 0 else 0
    values, vectors = values[:found], vectors[:, :found]
    reads = g @ (r @ vectors)  # D^T (D root v), a column for each part: D^T u up to its length
    reads = reads / reads.norm(dim=0, keepdim=True).clamp_min(1e-300)
    return values.sqrt(), torch.linalg.qr(reads).Q
