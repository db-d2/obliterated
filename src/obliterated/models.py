"""Model loading, chat formatting and architecture-agnostic access to residual-stream modules.

Supports the decoder families used in this project through the standard Hugging Face layout
(``model.model.layers[i]`` with ``self_attn.o_proj`` and an MLP whose output projection is
``down_proj``): Llama 3.x, Qwen 2.5/3, Gemma 2/3 (text), OLMo 2, SmolLM2, Mistral, and
DeepSeek-V3-style MoE such as Moonlight (every expert's ``down_proj`` is a residual writer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .device import best_device, default_dtype

_PLACEHOLDER = "<<<INSTRUCTION>>>"


@dataclass
class LM:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    name: str = ""
    system_prompt: str | None = None
    refusal_strings: tuple[str, ...] = ("I", "As")
    _eoi_len: int | None = field(default=None, repr=False)

    # ---- architecture access -------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def embed(self) -> nn.Embedding:
        return self.model.model.embed_tokens

    @property
    def has_post_norms(self) -> bool:
        """Gemma 2/3 normalize each sublayer's output before adding it to the residual stream."""
        return hasattr(self.layers[0], "post_feedforward_layernorm")

    def write_modules(self, layer: int) -> list[nn.Module]:
        """Modules whose outputs are added directly to the residual stream in this layer."""
        block = self.layers[layer]
        if self.has_post_norms:
            return [block.post_attention_layernorm, block.post_feedforward_layernorm]
        return [block.self_attn, block.mlp]

    def write_params(self, layer: int) -> list[nn.Linear | nn.Parameter]:
        """Everything that writes into the residual stream in this layer: the attention output
        projection and every MLP, expert and shared-expert down projection. Entries are either
        ``nn.Linear`` modules or fused expert weights of shape (n_experts, d_model, d_in), the
        layout transformers 5 uses for MoE blocks."""
        block = self.layers[layer]
        out: list[nn.Linear | nn.Parameter] = [block.self_attn.o_proj]
        for name, mod in block.mlp.named_modules():
            down = getattr(mod, "down_proj", None)
            if isinstance(down, nn.Parameter):
                if down.shape[-2] != self.d_model:
                    raise ValueError(f"unexpected fused down_proj layout {tuple(down.shape)} in {name}")
                out.append(down)
            elif isinstance(down, nn.Linear):
                out.append(down)
        if len(out) == 1:
            raise ValueError(f"no down_proj found in layer {layer}; add support for this architecture")
        return out

    # ---- prompting ------------------------------------------------------------------------

    def format(self, instruction: str) -> str:
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": instruction})
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @property
    def eoi_len(self) -> int:
        """Number of template tokens after the instruction (the "post-instruction" positions)."""
        if self._eoi_len is None:
            suffix = self.format(_PLACEHOLDER).split(_PLACEHOLDER)[-1]
            self._eoi_len = len(self.tokenizer(suffix, add_special_tokens=False).input_ids)
        return self._eoi_len

    @property
    def refusal_token_ids(self) -> list[int]:
        ids = {self.tokenizer(s, add_special_tokens=False).input_ids[0] for s in self.refusal_strings}
        return sorted(ids)

    def tokenize(self, instructions: list[str]) -> dict[str, torch.Tensor]:
        texts = [self.format(i) for i in instructions]
        # The chat template already contains BOS where the model expects it.
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}


def load(
    name: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> LM:
    dev = best_device(device)
    dtype = dtype or default_dtype(dev)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs).to(dev).eval()
    return wrap(model, tok, name=name, system_prompt=system_prompt)


def wrap(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs) -> LM:
    tokenizer.padding_side = "left"  # keeps post-instruction positions aligned at the end
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LM(model=model, tokenizer=tokenizer, **kwargs)
