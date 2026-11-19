import json

import pytest
import torch

from obliterated import detect, direction, hooks, sweep
from obliterated.edits import (
    EditSpec,
    _editable_tensors,
    blind_readers,
    blinded,
    norm_gain,
    orthogonalize,
    temporarily_blinded,
    untie_lm_head,
)
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


# ---- reader-side edits: the writers stay as they are, the matrices that read the stream change

READABLE = ["llama_tied", "llama_untied", "qwen2_bias", "gemma2"]


def uneven_gains(lm):
    """Give every norm a different gain per coordinate. Fresh norms scale all coordinates
    alike, and then an edit that forgot the norm would pass every check below."""
    g = torch.Generator().manual_seed(7)
    norms = {id(n): n for i in range(lm.n_layers) for n, _ in lm.read_modules(i)} | {id(lm.final_norm): lm.final_norm}
    for norm in norms.values():
        norm.weight.data += torch.empty_like(norm.weight).uniform_(-0.4, 0.4, generator=g)


def readers_and_head(lm):
    return [m for i in range(lm.n_layers) for m in lm.read_modules(i)] + [(lm.final_norm, lm.model.get_output_embeddings())]


@pytest.mark.parametrize("kind", ["llama_untied", "gemma2"])
def test_a_norms_gain_is_read_off_the_module_however_it_stores_it(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    uneven_gains(lm)
    norm = lm.layers[0].input_layernorm
    x = torch.randn(3, lm.d_model, generator=torch.Generator().manual_seed(2))
    gain = norm_gain(norm, lm.d_model)
    torch.testing.assert_close(norm(x), gain * x / x.pow(2).mean(-1, keepdim=True).sqrt(), atol=1e-5, rtol=1e-4)


def test_a_norm_that_does_more_than_rescale_is_refused():
    with pytest.raises(ValueError, match="rescal"):
        norm_gain(torch.nn.LayerNorm(8), 8)  # subtracts the mean first


@pytest.mark.parametrize("kind", READABLE)
def test_a_reader_side_edit_leaves_every_writer_and_the_embedding_untouched(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    base = last_logits(lm, PROMPTS)
    blind_readers(lm, rand_dir(lm), alpha=0.0)
    torch.testing.assert_close(last_logits(lm, PROMPTS), base, atol=1e-6, rtol=0)
    writers = [t.clone() for t in _editable_tensors(lm, None)]
    blind_readers(lm, rand_dir(lm))
    assert all(torch.equal(a, b) for a, b in zip(writers, _editable_tensors(lm, None)))
    assert not torch.allclose(base, last_logits(lm, PROMPTS), atol=1e-3), "the edit should change the output"


@pytest.mark.parametrize("kind", READABLE)
def test_an_edited_reader_ignores_the_direction_in_the_stream_and_reads_the_rest_as_before(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    uneven_gains(lm)
    d = rand_dir(lm)
    r = d / d.norm()
    x = torch.randn(5, lm.d_model, generator=torch.Generator().manual_seed(3))
    rest = x - (x @ r)[:, None] * r  # stream vectors with nothing along the direction
    edit_spec = EditSpec(direction=d.tolist(), side="readers")
    untie_lm_head(lm)  # so the head read here is the matrix the edit changes
    before = [lin.weight.clone() for _, lin in readers_and_head(lm)]
    edit_spec.apply(lm)
    for was, (norm, lin) in zip(before, readers_and_head(lm)):
        gain = norm_gain(norm, lm.d_model)  # a reader sees gain * stream / rms(stream)
        assert (lin.weight @ (gain * r)).abs().max() < 1e-5
        torch.testing.assert_close(lin.weight @ (gain * rest).T, was @ (gain * rest).T, atol=1e-5, rtol=1e-4)


def test_a_reader_side_band_edits_only_its_layers_and_the_head_only_with_the_last_one(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    head = lm.model.get_output_embeddings().weight.clone()
    was = {i: [lin.weight.clone() for _, lin in lm.read_modules(i)] for i in range(lm.n_layers)}
    blind_readers(lm, rand_dir(lm), alpha=0.5, layers=[1, 2])
    same = {i: [torch.equal(a, lin.weight) for a, (_, lin) in zip(was[i], lm.read_modules(i))] for i in was}
    assert all(same[0]) and all(same[3]) and not any(same[1]) and not any(same[2])
    assert torch.equal(head, lm.model.get_output_embeddings().weight)
    blind_readers(lm, rand_dir(lm), layers=[3])
    assert not torch.equal(head, lm.model.get_output_embeddings().weight)


def test_a_temporary_reader_side_edit_is_undone_exactly(tokenizer):
    lm = make_lm("llama_tied", tokenizer)
    base = last_logits(lm, PROMPTS)
    with temporarily_blinded(lm, rand_dir(lm)):
        assert not torch.allclose(base, last_logits(lm, PROMPTS), atol=1e-3)
    assert torch.equal(last_logits(lm, PROMPTS), base)


@pytest.mark.parametrize("kind", READABLE)
def test_the_runtime_reader_side_edit_is_the_weight_edit(kind, tokenizer):
    """Same output as the matrices edited in place, with the weights left as they were: what a
    search measures through the hooks is then what the edit it ends on will do."""
    lm = make_lm(kind, tokenizer)
    uneven_gains(lm)
    sub = torch.stack([rand_dir(lm, 1), rand_dir(lm, 2)])
    base = last_logits(lm, PROMPTS)
    for alpha, layers in ((1.0, None), (0.5, [1, lm.n_layers - 1]), (1.0, [0, 1])):
        with temporarily_blinded(lm, sub, alpha, layers):
            want = last_logits(lm, PROMPTS)
        assert not torch.allclose(base, want, atol=1e-3), "the weight edit should change the output"
        weights = [p.clone() for p in lm.model.parameters()]
        with blinded(lm, sub, alpha, layers):
            got = last_logits(lm, PROMPTS)
            assert all(torch.equal(a, b) for a, b in zip(weights, lm.model.parameters())), "no weight changes"
        torch.testing.assert_close(got, want, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(last_logits(lm, PROMPTS), base, atol=1e-6, rtol=0)  # and nothing stays behind


def test_the_runtime_reader_side_edit_has_the_weight_edits_slope_in_its_direction(tokenizer):
    """The gradient with respect to the direction, against a finite difference of the edit made
    in the weights."""
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    lm.model.double()  # the forward pass exactly; the edit itself is made in float32 either way
    d, step = rand_dir(lm, 1), rand_dir(lm, 2)
    enc = lm.tokenize(PROMPTS[:1])  # one prompt: a padded batch is not finite in float64, edit or no edit

    def out(direction, edit):
        with edit(lm, direction):
            return lm.model(**enc).logits[:, -1].logsumexp(-1).sum()

    at = d.clone().requires_grad_()
    (grad,) = torch.autograd.grad(out(at, blinded), at)
    eps = 1e-3
    with torch.no_grad():
        slope = (out(d + eps * step, temporarily_blinded) - out(d - eps * step, temporarily_blinded)) / (2 * eps)
    assert abs(float(slope)) > 1e-3, "the direction should matter along this step"
    assert float(grad @ step) == pytest.approx(float(slope), rel=1e-2)


def test_a_reader_side_editspec_is_active_at_runtime_as_it_applies(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    spec = EditSpec(direction=rand_dir(lm).tolist(), alpha=0.5, layers=[1, 3], side="readers")
    weights = [p.clone() for p in lm.model.parameters()]
    with spec.active(lm):
        got = last_logits(lm, PROMPTS)
    assert all(torch.equal(a, b) for a, b in zip(weights, lm.model.parameters()))
    spec.apply(lm)
    torch.testing.assert_close(got, last_logits(lm, PROMPTS), atol=2e-4, rtol=1e-4)


def test_reader_side_edits_refuse_a_layout_they_do_not_cover(tokenizer):
    lm = make_lm("qwen2_moe", tokenizer)
    before = [p.clone() for p in lm.model.parameters()]
    with pytest.raises(ValueError, match="reads the residual stream"):
        blind_readers(lm, rand_dir(lm))
    assert all(torch.equal(a, b) for a, b in zip(before, lm.model.parameters())), "nothing may be half edited"


def test_an_edit_saved_before_sides_existed_is_a_writer_side_edit(tmp_path, tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    spec = EditSpec(direction=rand_dir(lm).tolist(), side="readers")
    spec.save(tmp_path / "e.json")
    assert EditSpec.load(tmp_path / "e.json") == spec
    old = {k: v for k, v in json.loads((tmp_path / "e.json").read_text()).items() if k != "side"}
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert EditSpec.load(tmp_path / "old.json").side == "writers"
    with pytest.raises(ValueError, match="side"):
        EditSpec(direction=[1.0, 0.0], side="both").apply(lm)


def test_a_reader_side_edit_shows_nothing_to_a_detector_that_reads_the_writers(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    spec = EditSpec(direction=rand_dir(lm).tolist(), side="readers")
    scores = detect.weight_scores(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers), factored=True),
                                  detect.mid_layers(lm.n_layers))
    assert scores["weight_n_changed"] == 0 and scores["weight_top1_energy_w"] == 0.0


def test_the_sweep_runs_reader_side_edits(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    sub = hooks.orthonormal(torch.stack([rand_dir(lm, 1), rand_dir(lm, 2)]))
    weights_before = [p.clone() for p in lm.model.parameters()]
    rows = sweep.run_sweep(lm, sub, PROMPTS[:2], PROMPTS[2:], alphas=(1.0,), ks=(1, 2), bands={"all": None, "one": [1]},
                           mode="readers", batch_size=2)
    assert len(rows) == 4 and {r["mode"] for r in rows} == {"readers"}
    assert all(r["weight_n_changed"] == 0 for r in rows), "no writer matrix changes"
    assert all(r["behaviour_kl_harmful"] > 0 for r in rows), "the model behaves differently inside the edit"
    assert all(torch.equal(a, b) for a, b in zip(weights_before, lm.model.parameters())), "and is restored after"
