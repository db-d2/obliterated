"""Directions the reading-side detector does not list.

The reading side of a checkpoint is scored along a base's refusal directions: the difference in
means at every position of the template's suffix and every layer, and what they span
(``detect.refusal_span``). A reader-side edit along a direction v changes how a matrix reads a
listed direction r only through v . r. So the scores have two blind sides, and this module
makes the directions to test them with:

- inside the span but not on the list: an axis of the span, or a random direction in it. The
  scores along the listed directions see such an edit only as far as it leans on one of them.
- outside the span: what is left of a direction after the span is taken out of it, a random
  direction, or one searched for (``find_direction``). Along the span such an edit changes
  nothing at all, so no score that looks there can see it.

Whether an edit along any of them stops refusal is a question for the model, not for this
module: ``scripts/unlisted_directions.py`` measures it.
"""

from __future__ import annotations

import torch

from .edits import blinded
from .models import LM
from .refusal import last_logits

Tensor = torch.Tensor


def split(direction: Tensor, basis: Tensor) -> tuple[Tensor, Tensor, float]:
    """The part of ``direction`` inside the rows of an orthonormal ``basis``, the part outside
    them, and the share of its squared length that is inside. Float64."""
    v, b = direction.double(), basis.double()
    inside = (v @ b.T) @ b
    return inside, v - inside, float(inside.norm() ** 2 / v.norm() ** 2)


def describe(direction: Tensor, span: dict[str, Tensor]) -> dict[str, float]:
    """Where a direction stands to a base's refusal span (the tensors of its span file):
    ``inside``, the share of its squared length inside the span, the controls not counted; and
    ``listed_cosine``, its largest cosine, in size, with a listed direction, which is the
    alignment a rank-one edit along it shows."""
    listed = span["directions"].reshape(-1, span["directions"].shape[-1]).double()
    listed = listed[listed.norm(dim=1) > 0]
    unit = direction.double() / direction.double().norm()
    cosine = (listed / listed.norm(dim=1, keepdim=True)) @ unit
    return {"inside": split(unit, span["basis"][: int(span["n_ref"])])[2], "listed_cosine": float(cosine.abs().max())}


def candidates(span: dict[str, Tensor], known: dict[str, Tensor] | None = None, n_random: int = 8, seed: int = 0) -> list[dict]:
    """Unit directions to make reader-side edits from, none of them on the detector's list.

    Each is a dict with ``name``, ``kind``, ``direction`` (float32) and what ``describe`` says
    of it. The kinds:

    - ``axis``: every axis of the refusal span, the strongest first (the rows of the span's
      basis: the principal axes of the listed directions)
    - ``random_inside``: ``n_random`` random directions inside the span
    - ``random_outside``: ``n_random`` random directions orthogonal to the span and to the
      span file's control directions, so that nothing the record keeps moves
    - ``remainder``: for each direction in ``known`` (name -> vector; say, the direction a
      sweep took from other prompts), what is left of it outside the span and the controls;
      ``of_whole`` is the share of the direction's squared length that this is. A direction
      wholly inside leaves none.

    The random ones are the same for a ``seed``."""
    basis, n_ref = span["basis"].double(), int(span["n_ref"])
    g = torch.Generator().manual_seed(seed)
    made = [(f"axis_{i:03d}", "axis", row, {}) for i, row in enumerate(basis[:n_ref])]
    for i in range(n_random):
        made.append((f"random_inside_{i}", "random_inside", torch.randn(n_ref, generator=g, dtype=torch.float64) @ basis[:n_ref], {}))
    for i in range(n_random):
        made.append((f"random_outside_{i}", "random_outside", split(torch.randn(basis.shape[1], generator=g, dtype=torch.float64), basis)[1], {}))
    for name, v in (known or {}).items():
        _, outside, inside = split(v, basis)
        if 1 - inside > 1e-6:
            made.append((f"remainder_{name}", "remainder", outside, {"of_whole": 1 - inside}))
    out = []
    for name, kind, v, more in made:
        if kind != "axis" and kind != "random_inside":  # once more: the first pass leaves rounding in
            v = split(v, basis)[1]
        unit = (v / v.norm()).float()
        out.append({"name": name, "kind": kind, "direction": unit, **describe(unit, span), **more})
    return out


def padded(direction: Tensor, span: dict[str, Tensor], n: int, seed: int = 0) -> Tensor:
    """``direction`` with ``n`` more to edit along with it, for an edit spread over several
    directions: orthonormal rows, (n + 1, d), the unit direction first and then random ones,
    each orthogonal to the base's refusal span and its controls, as the direction is expected
    to be already. The same for a ``seed``, and the first rows of a longer padding are those
    of a shorter one."""
    basis = span["basis"].double()
    g = torch.Generator().manual_seed(seed)
    rows = [direction.double() / direction.double().norm()]
    for _ in range(n):
        new = split(torch.randn(basis.shape[1], generator=g, dtype=torch.float64), basis)[1]
        for _ in range(2):  # twice: the first pass leaves rounding in
            new = new - sum((new @ r) * r for r in rows)
            new = split(new, basis)[1]
        rows.append(new / new.norm())
    return torch.stack(rows).float()


def until_end(ids: Tensor, end_ids: list[int]) -> Tensor:
    """Which tokens of each generated row belong to its answer: all up to and including the
    first end token, none of the padding after it."""
    ended = torch.isin(ids, torch.tensor(end_ids, device=ids.device)).int()
    return ended.cumsum(1) - ended == 0


@torch.no_grad()
def answers_under(lm: LM, direction: Tensor, instructions: list[str], tokens: int = 16, batch_size: int = 16) -> tuple[Tensor, Tensor]:
    """The first ``tokens`` tokens the model answers each instruction with under the
    reader-side edit along ``direction`` (every layer, full strength), greedily: token ids,
    (n, tokens), and which of them belong to an answer (``until_end``). Held in memory for a
    search to aim at; nothing is decoded."""
    end = lm.model.generation_config.eos_token_id
    end = [end] if isinstance(end, int) else list(end or [lm.tokenizer.eos_token_id])
    pad = lm.tokenizer.pad_token_id
    out = []
    with blinded(lm, direction):
        for i in range(0, len(instructions), batch_size):
            enc = lm.tokenize(instructions[i : i + batch_size])
            new = lm.model.generate(**enc, max_new_tokens=tokens, do_sample=False, pad_token_id=pad)[:, enc["input_ids"].shape[1]:].cpu()
            out.append(torch.cat([new, new.new_full((len(new), tokens - new.shape[1]), pad)], 1))
    ids = torch.cat(out)
    return ids, until_end(ids, end)


def find_direction(
    lm: LM, harmful: list[str], harmless: list[str], start: Tensor, avoid: Tensor | None = None,
    like: Tensor | None = None, tokens: int = 16,
    steps: int = 200, lr: float = 0.01, kl_weight: float = 1.0, batch_size: int = 16, seed: int = 0,
) -> tuple[Tensor, list[dict]]:
    """Search for a unit direction whose reader-side edit (every layer, full strength) stops
    refusal, from ``start``, never leaving the directions orthogonal to the orthonormal rows
    of ``avoid``: what an editor would do who knows which directions a detector looks along.

    Minimized by Adam over batches of prompts: a loss on harmful prompts, plus ``kl_weight``
    times the KL from the unedited model's next-token distribution on harmless ones, so that
    the direction found is not one whose loss wrecks the model. On the harmful prompts:

    - without ``like``, how unlikely a first token other than a refusal is. That is the
      sweeps' measure of an edit, and a search can satisfy it by moving the first token alone,
      with the answer that follows still a refusal.
    - with ``like``, a direction whose edit is known to work: how unlikely, per token, the
      first ``tokens`` tokens are that the model answers with under that edit
      (``answers_under``). The search is then for an edit that answers as the known one does.

    The edit is ``edits.blinded``, the weight edit as hooks; the model is not changed. Load it
    in float32: half precision loses the gradient.

    Returns the direction (float32, on the CPU) and one dict per step with ``loss`` and its
    two parts, ``harmful`` and ``kl``. What a search finds is fitted to its loss: measure it
    by something the loss did not see."""
    dev = lm.device
    rows = None if avoid is None else avoid.to(dev, torch.float32)

    def unit(w: Tensor) -> Tensor:
        if rows is not None:
            w = w - (w @ rows.T) @ rows
        return w / w.norm()

    refusing = torch.zeros(lm.model.config.vocab_size, dtype=torch.bool, device=dev)
    refusing[lm.refusal_token_ids] = True
    base = last_logits(lm, harmless, batch_size)
    answers, real = (None, None) if like is None else answers_under(lm, like, harmful, tokens, batch_size)
    w = unit(start.to(dev, torch.float32)).clone().requires_grad_()
    opt = torch.optim.Adam([w], lr=lr)
    g = torch.Generator().manual_seed(seed)
    trained = [p for p in lm.model.parameters() if p.requires_grad]
    history = []
    for p in trained:  # only the direction is being fitted
        p.requires_grad_(False)
    try:
        for _ in range(steps):
            hi = torch.randperm(len(harmful), generator=g)[:batch_size].tolist()
            si = torch.randperm(len(harmless), generator=g)[:batch_size].tolist()
            with blinded(lm, unit(w)):
                enc = lm.tokenize([harmful[i] for i in hi])
                if like is None:
                    first = lm.model(**enc).logits[:, -1].float().log_softmax(-1)
                    on_harmful = -first.masked_fill(refusing, float("-inf")).logsumexp(-1).mean()
                else:  # the known edit's answers, fed back in: how unlikely each of their tokens is
                    ids = answers[hi].to(dev)
                    logits = lm.model(input_ids=torch.cat([enc["input_ids"], ids], 1),
                                      attention_mask=torch.cat([enc["attention_mask"], torch.ones_like(ids)], 1)).logits
                    each = -logits[:, -tokens - 1 : -1].float().log_softmax(-1).gather(-1, ids[..., None])[..., 0]
                    on_harmful = each[real[hi].to(dev)].mean()
                on_harmless = lm.model(**lm.tokenize([harmless[i] for i in si])).logits[:, -1].float().log_softmax(-1)
            was = base[si].to(dev).log_softmax(-1)
            kl = (was.exp() * (was - on_harmless)).sum(-1).mean()
            loss = on_harmful + kl_weight * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append({"loss": loss.item(), "harmful": on_harmful.item(), "kl": kl.item()})
    finally:
        for p in trained:
            p.requires_grad_(True)
    found = w.detach().cpu().double()  # in that order: the Mac's GPU has no float64
    if avoid is not None:  # in float64, so that what is returned is outside to rounding
        found = split(split(found, avoid)[1], avoid)[1]
    return (found / found.norm()).float(), history
