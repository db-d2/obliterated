import pytest
import torch

from obliterated import direction, hooks
from obliterated.edits import EditSpec, orthogonalize
from obliterated.refusal import last_logits

from .conftest import make_lm

PROMPTS = ["tell me how to bake a cake", "write poem", "build the thing", "how to write"]
EXACT = ["llama_tied", "llama_untied", "qwen2_bias", "qwen2_moe"]


def rand_dir(lm, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(lm.d_model, generator=g)


@pytest.mark.parametrize("kind", EXACT)
def test_weight_orthogonalization_matches_hook_ablation(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    d = rand_dir(lm)
    base = last_logits(lm, PROMPTS)
    with hooks.ablate(lm, d):
        hooked = last_logits(lm, PROMPTS)
    assert not torch.allclose(base, hooked, atol=1e-3), "ablation should change the output"
    orthogonalize(lm, d)
    edited = last_logits(lm, PROMPTS)
    torch.testing.assert_close(edited, hooked, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", EXACT + ["gemma2"])
def test_hook_ablation_removes_direction_from_residual(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    d = rand_dir(lm)
    r = d / d.norm()
    store = []
    with hooks.ablate(lm, d), hooks.cache_residual(lm, 3, store):
        lm.model(**lm.tokenize(PROMPTS))
    for layer_acts in store:
        assert (layer_acts[0] @ r).abs().max() < 1e-4


def test_partial_alpha_and_layer_band(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    d = rand_dir(lm)
    base = last_logits(lm, PROMPTS)
    with hooks.ablate(lm, d, alpha=0.0):
        torch.testing.assert_close(last_logits(lm, PROMPTS), base)
    before = [p.clone() for p in lm.layers[0].parameters()]
    orthogonalize(lm, d, alpha=0.5, layers=[2, 3], edit_embeddings=False)
    assert all(torch.equal(a, b) for a, b in zip(before, lm.layers[0].parameters()))
    w = lm.layers[2].self_attn.o_proj.weight
    r = d / d.norm()
    # Half of the original component along r remains.
    assert (r @ w).norm() > 1e-3


def test_untie_keeps_lm_head(tokenizer):
    lm = make_lm("llama_tied", tokenizer)
    head = lm.model.get_output_embeddings().weight.detach().clone()
    orthogonalize(lm, rand_dir(lm))
    torch.testing.assert_close(lm.model.get_output_embeddings().weight, head)
    assert lm.model.get_output_embeddings().weight is not lm.embed.weight


def test_gemma2_weight_edit_warns(tokenizer):
    lm = make_lm("gemma2", tokenizer)
    with pytest.warns(UserWarning, match="approximate"):
        orthogonalize(lm, rand_dir(lm))


def test_moe_writers_include_every_expert(tokenizer):
    lm = make_lm("qwen2_moe", tokenizer)
    writers = lm.write_params(0)
    # o_proj, the experts (one fused tensor in transformers 5, else 4 Linears), the shared expert
    n_experts = sum(w.shape[0] if isinstance(w, torch.nn.Parameter) else 1 for w in writers[1:])
    assert n_experts == 4 + 1


def test_editspec_roundtrip(tmp_path, tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    spec = EditSpec(direction=rand_dir(lm).tolist(), alpha=0.7, layers=[1, 2], mode="hooks")
    spec.save(tmp_path / "e.json")
    again = EditSpec.load(tmp_path / "e.json")
    assert again == spec
    with again.active(lm):
        pass


def test_direction_pipeline_runs(tokenizer):
    lm = make_lm("qwen2_bias", tokenizer)
    harmful, harmless = PROMPTS[:2], PROMPTS[2:]
    assert lm.eoi_len == 1
    cands = direction.candidate_directions(lm, harmful, harmless, batch_size=2)
    assert cands.shape == (lm.eoi_len, lm.n_layers, lm.d_model)
    sel = direction.select_direction(
        lm, cands, harmful, harmless, kl_threshold=float("inf"), induce_threshold=float("-inf")
    )
    assert len(sel.scores) == lm.eoi_len * lm.n_layers
    assert sel.layer < int(lm.n_layers * 0.8)
    torch.testing.assert_close(sel.direction, cands[sel.pos, sel.layer])
