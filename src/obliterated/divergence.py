"""How far a model moves from its base over whole answers to harmless prompts.

The sweeps report what an edit costs as the KL divergence between the base's and the edited
model's next-token distributions at the first generated token (``refusal.kl_div`` on
``refusal.last_logits``): one token for each prompt. Young (arXiv 2611.17846) reports that a
first-token KL did not track the change in capability benchmarks over the edited models
studied there, and that a divergence taken over whole answers did. This module measures the
second kind. The base answers each harmless prompt greedily, its answers are fed back to the
edited model token by token (teacher forcing), and the two models are compared at every token.

    base = divergence.base_answers(lm, harmless)  # before the edit
    with temporarily_blinded(lm, direction):
        row = divergence.measure(lm, base)  # answer_logprob_drop, answer_kl, answer_first_kl
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from .models import LM


@dataclass
class Answers:
    """The base's greedy answers to ``instructions`` and what the base gave every token of
    them: ``logprobs`` holds one row per answer token, the answers end to end, each row the
    base's log-probabilities over the vocabulary at that position (float32 on CPU). That is
    4 bytes x tokens x vocab: 100 answers of 64 tokens over a vocabulary of 150,000 take
    3.8 GB."""

    instructions: list[str]
    tokens: list[list[int]]
    logprobs: torch.Tensor


@torch.no_grad()
def greedy(lm: LM, instructions: list[str], max_new_tokens: int = 64, batch_size: int = 16) -> list[list[int]]:
    """The token ids of the model's greedy answer to each instruction. An answer that ends
    within ``max_new_tokens`` keeps the token that ends it, since stopping there is part of
    the answer, and nothing after it."""
    ends = lm.model.generation_config.eos_token_id
    ends = set() if ends is None else {ends} if isinstance(ends, int) else set(ends)
    out = []
    for i in range(0, len(instructions), batch_size):
        enc = lm.tokenize(instructions[i : i + batch_size])
        ids = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
        for row in ids[:, enc["input_ids"].shape[1] :].tolist():
            stop = next((t for t, token in enumerate(row) if token in ends), len(row) - 1)
            out.append(row[: stop + 1])
    return out


@torch.no_grad()
def teacher_forced(lm: LM, instructions: list[str], answers: list[list[int]], batch_size: int = 16) -> Iterator[torch.Tensor]:
    """For each answer in turn, what the model gives the whole vocabulary at every position of
    it when the answer is fed back token by token after its instruction: log-probabilities,
    float32 on CPU, shape (tokens in the answer, vocab). Row t is the model's next-token
    distribution once the first t tokens of the answer are in place, so row 0 is its
    distribution at the first generated token."""
    pad = lm.tokenizer.pad_token_id
    for i in range(0, len(instructions), batch_size):
        batch = answers[i : i + batch_size]
        rows = [lm.tokenizer(lm.format(p), add_special_tokens=False).input_ids + a for p, a in zip(instructions[i : i + batch_size], batch)]
        width = max(map(len, rows))
        ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=lm.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=lm.device)
        logits = lm.model(input_ids=ids, attention_mask=mask).logits
        for row, answer in zip(logits, batch):
            # the token at position t is predicted from position t - 1, and rows are padded on the left
            at = row[width - len(answer) - 1 : width - 1].float().cpu()
            yield torch.log_softmax(at.double(), -1).float()


def base_answers(lm: LM, instructions: list[str], max_new_tokens: int = 64, batch_size: int = 16) -> Answers:
    """What an edit of ``lm`` is later measured against. Call it before the edit is made."""
    tokens = greedy(lm, instructions, max_new_tokens, batch_size)
    return Answers(instructions, tokens, torch.cat(list(teacher_forced(lm, instructions, tokens, batch_size))))


def per_token(lm: LM, base: Answers, batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """How far ``lm`` stands from the base at every token of the base's answers, in the order
    of ``base.logprobs``. Two numbers per token, both in nats and both 0 for the base itself:

    - the drop in the log-probability of the token the base chose there, log p_base(y) -
      log p_lm(y). It can be negative, where ``lm`` is surer of that token than the base was
    - the KL divergence of the base's whole next-token distribution from that of ``lm``,
      KL(base || lm), which is never negative. At the first token of an answer it is what
      ``refusal.kl_div`` gives for ``refusal.last_logits``

    ``lm`` is run on the base's answers and not on its own, so both models are compared on the
    same text. The comparison is made answer by answer and only the base's distributions are
    held in memory.

    A fine-tune whose vocabulary is longer or shorter than the base's is compared over the
    tokens both have, each side's distribution brought to sum to 1 over them, as
    ``detect.behaviour_per_prompt`` does at the first token. The tokens of the base's answers
    have to be among them."""
    chosen = torch.tensor([token for answer in base.tokens for token in answer])
    if int(chosen.max()) >= lm.embed.num_embeddings:
        raise ValueError(f"the base's answers use token {int(chosen.max())}, outside the vocabulary of {lm.embed.num_embeddings} tokens this model has")
    drops, kls, at = [], [], 0
    for logq in teacher_forced(lm, base.instructions, base.tokens, batch_size):
        logp, logq = base.logprobs[at : at + len(logq)].double(), logq.double()
        if logq.shape[-1] != logp.shape[-1]:
            shared = min(logp.shape[-1], logq.shape[-1])
            logp, logq = torch.log_softmax(logp[:, :shared], -1), torch.log_softmax(logq[:, :shared], -1)
        drops.append((logp - logq).gather(1, chosen[at : at + len(logq), None])[:, 0])
        kls.append((logp.exp() * (logp - logq)).sum(-1))
        at += len(logq)
    return torch.cat(drops).float(), torch.cat(kls).float()


def summary(base: Answers, drop: torch.Tensor, kl: torch.Tensor) -> dict:
    """What ``per_token`` comes to, as columns of a results row. The two whole-answer numbers
    are means over every token of every answer, so a long answer counts for more than a short
    one: ``answer_logprob_drop`` is the measure of Young (arXiv 2611.17846), nats per token,
    and ``answer_kl`` is the KL divergence averaged the same way. ``answer_first_kl`` is the
    KL at the first token alone, averaged over the prompts, which is what the sweeps report as
    ``kl_harmless``."""
    starts = torch.tensor([0, *torch.tensor([len(a) for a in base.tokens]).cumsum(0)[:-1].tolist()])
    return {
        "answer_logprob_drop": float(drop.mean()),
        "answer_kl": float(kl.mean()),
        "answer_first_kl": float(kl[starts].mean()),
        "answer_prompts": len(base.tokens),
        "answer_tokens": len(kl),
    }


def measure(lm: LM, base: Answers, batch_size: int = 16) -> dict:
    """The columns of ``summary`` for ``lm`` as it now stands: call it while an edit is in place."""
    return summary(base, *per_token(lm, base, batch_size))
