"""Tiny randomly initialized models and a local word-level tokenizer, so tests run on CPU offline."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2MoeConfig,
)

from obliterated.models import wrap

WORDS = "<pad> <bos> <eos> <|user|> <|assistant|> I As Sure tell me how to bake a cake build the thing write poem".split()
TEMPLATE = (
    "<bos> {% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="session")
def tokenizer():
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"
    )
    t.chat_template = TEMPLATE
    return t


COMMON = dict(
    vocab_size=len(WORDS),
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)

CONFIGS = {
    "llama_tied": lambda: LlamaConfig(**COMMON, tie_word_embeddings=True),
    "llama_untied": lambda: LlamaConfig(**COMMON, tie_word_embeddings=False),
    "qwen2_bias": lambda: Qwen2Config(**COMMON, tie_word_embeddings=True),
    "qwen2_moe": lambda: Qwen2MoeConfig(
        **COMMON,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        decoder_sparse_step=1,
    ),
    "gemma2": lambda: Gemma2Config(**COMMON, head_dim=8),
}


def make_lm(kind: str, tokenizer):
    torch.manual_seed(0)
    cfg = CONFIGS[kind]()
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    return wrap(model, tokenizer, name=kind)
