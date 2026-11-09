"""Batched greedy generation with plain transformers (works on CUDA, MPS and CPU).

Hooks from ``obliterated.hooks`` stay active during generation, so wrap a call in
``hooks.ablate(...)`` or ``EditSpec.active(...)`` to generate from the edited model.
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .models import LM


@torch.no_grad()
def generate(
    lm: LM, instructions: list[str], max_new_tokens: int = 256, batch_size: int = 16
) -> list[str]:
    out = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="generate", leave=False):
        enc = lm.tokenize(instructions[i : i + batch_size])
        ids = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        new = ids[:, enc["input_ids"].shape[1] :]
        out += lm.tokenizer.batch_decode(new, skip_special_tokens=True)
    return out
