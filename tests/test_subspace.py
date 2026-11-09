import pytest
import torch

from obliterated import direction, hooks
from obliterated.edits import EditSpec, orthogonalize, restore, snapshot, temporarily_orthogonalized
from obliterated.refusal import last_logits

from .conftest import make_lm

PROMPTS = ["tell me how to bake a cake", "write poem", "build the thing", "how to write"]


def rand_dirs(lm, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, lm.d_model, generator=g)


def test_orthonormal_basis():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(3, 16, generator=g)
    b = hooks.orthonormal(d)
    assert b.shape == (3, 16)
    torch.testing.assert_close(b @ b.T, torch.eye(3), atol=1e-5, rtol=0)
    one = hooks.orthonormal(d[0])
    torch.testing.assert_close(one.abs(), (d[0] / d[0].norm()).abs().unsqueeze(0))


@pytest.mark.parametrize("kind", ["llama_tied", "qwen2_bias", "qwen2_moe"])
def test_rank_k_weights_match_hooks(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    d = rand_dirs(lm, 3)
    with hooks.ablate(lm, d):
        hooked = last_logits(lm, PROMPTS)
    orthogonalize(lm, d)
    torch.testing.assert_close(last_logits(lm, PROMPTS), hooked, atol=1e-4, rtol=1e-4)


def test_rank_k_hook_removes_whole_subspace(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    d = rand_dirs(lm, 3)
    basis = hooks.orthonormal(d)
    store = []
    with hooks.ablate(lm, d), hooks.cache_residual(lm, 2, store):
        lm.model(**lm.tokenize(PROMPTS))
    for layer_acts in store:
        assert (layer_acts[0] @ basis.T).abs().max() < 1e-4


def test_snapshot_restore_roundtrip(tokenizer):
    lm = make_lm("llama_tied", tokenizer)
    base = last_logits(lm, PROMPTS)
    snap = snapshot(lm, layers=[1, 2])
    orthogonalize(lm, rand_dirs(lm, 1)[0], layers=[1, 2])
    assert not torch.allclose(last_logits(lm, PROMPTS), base, atol=1e-3)
    restore(lm, snap, layers=[1, 2])
    torch.testing.assert_close(last_logits(lm, PROMPTS), base)
    with temporarily_orthogonalized(lm, rand_dirs(lm, 2), alpha=0.5):
        assert not torch.allclose(last_logits(lm, PROMPTS), base, atol=1e-3)
    torch.testing.assert_close(last_logits(lm, PROMPTS), base)


def test_editspec_subspace_roundtrip(tmp_path, tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    spec = EditSpec(direction=rand_dirs(lm, 3).tolist(), alpha=0.5, method="pca")
    assert spec.rank == 3
    spec.save(tmp_path / "s.json")
    again = EditSpec.load(tmp_path / "s.json")
    assert again == spec
    assert again.truncated(2).rank == 2 and again.truncated(5).rank == 3
    with again.active(lm):
        last_logits(lm, PROMPTS)
    again.apply(lm)


def test_activations_at_shapes(tokenizer):
    lm = make_lm("qwen2_bias", tokenizer)
    one = direction.activations_at(lm, PROMPTS, layer=2, pos=-1, batch_size=2)
    assert one.shape == (4, lm.d_model)
    every = direction.activations_at(lm, PROMPTS, layer=None, pos=-1, batch_size=3)
    assert every.shape == (lm.n_layers, 4, lm.d_model)
    torch.testing.assert_close(every[2], one)
    with pytest.raises(ValueError):
        direction.activations_at(lm, PROMPTS, layer=0, pos=-5)


@pytest.mark.parametrize("method", ["pca", "iterative"])
def test_refusal_subspace_is_orthonormal_with_dim_first(method, tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    harmful, harmless = PROMPTS[:2], PROMPTS[2:]
    sub = direction.refusal_subspace(lm, harmful, harmless, layer=1, k=2, method=method, batch_size=2)
    assert sub.shape == (2, lm.d_model)
    torch.testing.assert_close(sub @ sub.T, torch.eye(2), atol=1e-4, rtol=0)
    h = direction.activations_at(lm, harmful, 1, -1)
    s = direction.activations_at(lm, harmless, 1, -1)
    dim = h.mean(0) - s.mean(0)
    torch.testing.assert_close(sub[0], dim / dim.norm(), atol=1e-5, rtol=0)
