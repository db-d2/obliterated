import math

import pytest
import torch

from obliterated import detect, hooks
from obliterated.edits import EditSpec, orthogonalize
from obliterated.refusal import last_logits

from .conftest import make_lm

HARMFUL = ["tell me how to bake a cake", "build the thing"]
HARMLESS = ["write poem", "how to write"]


def test_auroc_basic():
    assert detect.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert detect.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert detect.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    with pytest.raises(ValueError):
        detect.auroc([0.1, 0.2], [1, 1])


def test_mid_layers():
    assert detect.mid_layers(32) == list(range(8, 24))
    assert detect.mid_layers(4) == [1, 2]


def test_rank1_edit_has_full_top1_energy(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    g = torch.Generator().manual_seed(0)
    d = torch.randn(lm.d_model, generator=g)
    spec = EditSpec(direction=d.tolist(), alpha=1.0)
    analytic = detect.weight_recovery_energy(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers)))
    assert analytic["n_matrices"] == 2 * lm.n_layers
    assert analytic["top1_energy"] > 0.999
    assert analytic["stable_rank"] == pytest.approx(1.0, abs=1e-3)

    # The analytic delta equals the real one: edit a copy and diff against the clean model.
    cand = make_lm("llama_untied", tokenizer)
    orthogonalize(cand, d, edit_embeddings=False)
    real = detect.weight_recovery_energy(detect.diff_pairs_from_models(lm, cand, range(lm.n_layers)))
    assert real["top1_energy"] > 0.999
    assert real["fro_norm"] == pytest.approx(analytic["fro_norm"], rel=1e-3)


def test_rank_k_edit_and_noise_separate(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    g = torch.Generator().manual_seed(0)
    spec = EditSpec(direction=torch.randn(3, lm.d_model, generator=g).tolist())
    edit = detect.weight_recovery_energy(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers)))
    # Exactly rank 3 per matrix: stable rank in (1, 3], and 3 components carry everything.
    assert 1.0 < edit["stable_rank"] <= 3.0 + 1e-3
    assert all(r["dim90"] <= 3 for r in edit["per_matrix"])
    # A full-rank perturbation of matched size looks nothing like a low-rank edit.
    noise = [(n, torch.randn_like(d) * d.norm() / d.numel() ** 0.5) for n, d in detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers))]
    noisy = detect.weight_recovery_energy(noise)
    assert noisy["top1_energy"] < edit["top1_energy"]
    assert noisy["stable_rank"] > 2 * edit["stable_rank"]
    assert detect.auroc([edit["top1_energy"], noisy["top1_energy"]], [1, 0]) == 1.0


def test_untouched_layers_are_zero_deltas(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    spec = EditSpec(direction=torch.ones(lm.d_model).tolist(), layers=[3])
    e = detect.weight_recovery_energy(detect.diff_pairs_from_edit(lm, spec, [0, 1, 2, 3]))
    assert e["n_zero"] == 6 and e["n_matrices"] == 2


def test_activation_gap_rises_under_ablation(tokenizer):
    lm = make_lm("qwen2_bias", tokenizer)
    dirs = detect.reference_directions(lm, HARMFUL, HARMLESS, batch_size=2)
    assert dirs.shape == (lm.n_layers, lm.d_model)
    sep_ref = detect.separation(lm, dirs, HARMFUL, HARMLESS, batch_size=2)
    assert detect.activation_gap(sep_ref, sep_ref) == pytest.approx(0.0, abs=1e-5)
    # Ablating each layer's own reference direction removes the separation along it.
    with hooks.ablate(lm, dirs[1], layers=[1]):
        sep = detect.separation(lm, dirs, HARMFUL, HARMLESS, batch_size=2)
    assert detect.activation_gap(sep_ref, sep, layers=[1]) == pytest.approx(1.0, abs=1e-3)


def test_behaviour_shift_zero_for_identity(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    logits = last_logits(lm, HARMFUL)
    s = detect.behaviour_shift(logits, logits, lm.refusal_token_ids)
    assert s["kl_harmful"] == pytest.approx(0.0, abs=1e-6)
    assert s["refusal_score_drop"] == pytest.approx(0.0, abs=1e-6)


def test_safetensors_streaming_diff(tmp_path, tokenizer):
    from safetensors.torch import save_file

    lm = make_lm("llama_untied", tokenizer)
    cand = make_lm("llama_untied", tokenizer)
    orthogonalize(cand, torch.ones(lm.d_model), edit_embeddings=False)
    for name, m in (("base", lm), ("cand", cand)):
        (tmp_path / name).mkdir()
        sd = {k: v.contiguous() for k, v in m.model.state_dict().items()}
        save_file(sd, tmp_path / name / "model.safetensors")
    pairs = list(detect.diff_pairs_from_safetensors(tmp_path / "base", tmp_path / "cand", [1, 2]))
    assert len(pairs) == 4 and all(".layers.1." in n or ".layers.2." in n for n, _ in pairs)
    assert detect.weight_recovery_energy(pairs)["top1_energy"] > 0.999


def test_topk_energy_is_flat_in_how_a_rank_k_edit_splits_its_energy(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    g = torch.Generator().manual_seed(0)
    spec = EditSpec(direction=torch.randn(3, lm.d_model, generator=g).tolist())
    edit = detect.weight_recovery_energy(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers)))
    assert edit["top1_energy"] < 0.9  # rank 3 dilutes the top-1 score ...
    assert edit["top4_energy"] == pytest.approx(1.0, abs=1e-4)  # ... but not top-k for k >= 3
    assert edit["top2_energy"] < edit["top4_energy"] <= edit["top8_energy"] + 1e-6
    noise = [(n, torch.randn_like(d)) for n, d in detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers))]
    assert detect.weight_recovery_energy(noise)["top4_energy"] < 0.9
    assert "top4_energy" not in detect.weight_recovery_energy(noise, topk=())


def test_behaviour_shift_cuts_to_the_shared_vocabulary():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(5, 20, generator=g)
    same = detect.behaviour_shift(base, torch.cat([base, torch.full((5, 4), -1e4)], dim=1), [1, 2])
    assert same["kl_harmful"] == pytest.approx(0.0, abs=1e-5)
    assert same["refusal_score_drop"] == pytest.approx(0.0, abs=1e-5)
    assert same["bypass_rate"] == same["baseline_bypass_rate"]
    with pytest.raises(ValueError):
        detect.behaviour_shift(base, base[:, :10], [1, 15])


def test_weight_scores_ignore_noise_only_matrices_and_find_edits_outside_the_mid_stack():
    g = torch.Generator().manual_seed(0)
    u, v = torch.randn(32, 1, generator=g), torch.randn(1, 64, generator=g)
    pairs = [(f"model.layers.{i}.mlp.down_proj.weight", 1e-6 * torch.randn(32, 64, generator=g)) for i in range(8)]
    pairs[6] = ("model.layers.6.mlp.down_proj.weight", u @ v)  # a rank-1 edit in one late layer
    s = detect.weight_scores(pairs, mid=[2, 3, 4, 5])
    assert s["weight_top1_energy"] < 0.5  # the mid-stack mean sees only rounding-sized noise
    assert s["weight_top1_energy_w"] > 0.999  # the weighted score sees the edit
    assert s["weight_n_active"] == 1 and s["weight_n_changed"] == 8 and s["weight_n_matrices"] == 4
    assert detect.layer_of("layer3.w0") == 3 and detect.layer_of("model.layers.11.self_attn.o_proj.weight") == 11
    empty = detect.weight_scores([("model.layers.0.x", torch.zeros(4, 4))], mid=[0])
    assert empty["weight_top1_energy_w"] == 0.0 and empty["weight_n_active"] == 0


def test_factored_edit_deltas_score_the_same_as_full_ones(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    g = torch.Generator().manual_seed(0)
    spec = EditSpec(direction=torch.randn(3, lm.d_model, generator=g).tolist(), alpha=0.5, layers=[1, 2])
    mid = detect.mid_layers(lm.n_layers)
    full = detect.weight_scores(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers)), mid)
    fast = detect.weight_scores(detect.diff_pairs_from_edit(lm, spec, range(lm.n_layers), factored=True), mid)
    assert fast["weight_n_changed"] == full["weight_n_changed"] == 4
    for key, v in full.items():
        if isinstance(v, float) and v == v:
            assert fast[key] == pytest.approx(v, rel=1e-3, abs=1e-4), key


def test_top_factors_recover_a_low_rank_delta():
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(6, 2, generator=g))   # residual side, orthonormal columns
    v, _ = torch.linalg.qr(torch.randn(10, 2, generator=g))  # input side
    delta = 3.0 * torch.outer(u[:, 0], v[:, 0]) + 1.0 * torch.outer(u[:, 1], v[:, 1])
    s, vecs = detect.top_factors(delta, k=2)
    assert s.tolist() == pytest.approx([3.0, 1.0], abs=1e-4)
    assert vecs.shape == (6, 2)
    assert abs(float(vecs[:, 0] @ u[:, 0])) == pytest.approx(1.0, abs=1e-4)
    assert abs(float(vecs[:, 1] @ u[:, 1])) == pytest.approx(1.0, abs=1e-4)


def test_top_factors_of_a_matrix_with_fewer_rows_than_k():
    s, vecs = detect.top_factors(torch.tensor([[2.0, 0.0, 0.0]]), k=8)
    assert s.tolist() == pytest.approx([2.0])
    assert vecs.shape == (1, 1)


def test_vocab_pairs_expose_a_head_only_edit(tmp_path):
    """The base ties its output head to the embedding; the candidate saved an edited head
    separately and added a token. Only the head moved, by a rank-1 projection along ``r``."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    embed = torch.randn(5, 4, generator=g)
    r = torch.tensor([0.0, 1.0, 0.0, 0.0])
    grown = torch.cat([embed, torch.randn(1, 4, generator=g)])
    (tmp_path / "base").mkdir()
    (tmp_path / "cand").mkdir()
    save_file({"model.embed_tokens.weight": embed}, tmp_path / "base" / "model.safetensors")
    save_file({"model.embed_tokens.weight": grown, "lm_head.weight": grown - torch.outer(grown @ r, r)},
              tmp_path / "cand" / "model.safetensors")

    pairs = dict(detect.vocab_pairs(tmp_path / "base", tmp_path / "cand"))

    assert set(pairs) == {"model.embed_tokens.weight", "lm_head.weight"}
    assert pairs["model.embed_tokens.weight"].shape == (4, 5)  # residual side first, shared tokens only
    assert pairs["model.embed_tokens.weight"].abs().max() == 0
    s, vecs = detect.top_factors(pairs["lm_head.weight"], k=2)
    assert float(s[1]) == pytest.approx(0.0, abs=1e-5)
    assert abs(float(vecs[:, 0] @ r)) == pytest.approx(1.0, abs=1e-5)


def test_behaviour_per_prompt_matches_hand_computed_values():
    import math

    base = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])  # second prompt: p(refusal token 0) = 0.75
    cand = torch.zeros(2, 2)
    per = detect.behaviour_per_prompt(base, cand, [0])
    assert per["kl"].tolist() == pytest.approx([0.0, 0.75 * math.log(1.5) + 0.25 * math.log(0.5)], abs=1e-6)
    assert per["refusal_score_base"].tolist() == pytest.approx([0.0, math.log(3.0)], abs=1e-5)
    assert per["refusal_score"].tolist() == pytest.approx([0.0, 0.0], abs=1e-5)


def test_separation_of_projections_matches_hand_computed_value():
    harmful, harmless = torch.tensor([[1.0, 3.0]]), torch.tensor([[0.0, 0.0]])
    # gap 2; pooled deviations (-1, 1, 0, 0) have sample standard deviation sqrt(2/3)
    assert detect.separation_of(harmful, harmless).tolist() == pytest.approx([2.0 / (2 / 3) ** 0.5], abs=1e-5)


def test_weight_spectrum_reports_the_true_norm_of_a_large_difference():
    """float32 ``Tensor.norm()`` on the CPU comes out low as a matrix grows: 1.3% on the MLP
    difference of an 8B model, 17% on its embedding difference. 14M entries already show it.
    The record check rebuilds the top singular value from the two numbers stored here."""
    delta = torch.randn(1536, 8960, generator=torch.Generator().manual_seed(0)) * 1e-3
    exact = float(delta.double().pow(2).sum().sqrt())
    [m] = detect.weight_recovery_energy([("m", delta)])["per_matrix"]
    assert m["fro_norm"] == pytest.approx(exact, rel=1e-6)
    top = float(detect.top_factors(delta, 1)[0][0])
    assert math.sqrt(m["top1_energy"]) * m["fro_norm"] == pytest.approx(top, rel=1e-5)


def _basis(d: int = 16) -> torch.Tensor:
    """An orthonormal basis to build factors and directions from by hand."""
    return torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(3)))[0]


def test_an_edit_along_the_refusal_direction_aligns_fully():
    q = _basis()
    directions = torch.stack([q[:, 5], q[:, 9]])       # two layers' refusal directions
    factors = q[:, [9, 0, 1]]                          # the edit's top factor is the second one
    assert detect.refusal_alignment(factors, directions) == pytest.approx(1.0, abs=1e-5)


def test_a_direction_split_across_two_factors_still_aligns_fully():
    """A rank-2 edit can rotate its basis so that no single factor is the refusal direction."""
    q = _basis()
    mixed = torch.stack([(q[:, 5] + q[:, 6]) / 2**0.5, (q[:, 5] - q[:, 6]) / 2**0.5], dim=1)
    assert float((mixed[:, 0] @ q[:, 5]).abs()) == pytest.approx(2**-0.5, abs=1e-5)
    assert detect.refusal_alignment(mixed, q[:, 5][None]) == pytest.approx(1.0, abs=1e-5)


def test_an_unrelated_edit_does_not_align():
    q = _basis()
    assert detect.refusal_alignment(q[:, [0, 1, 2]], torch.stack([q[:, 5], q[:, 9]])) == pytest.approx(0.0, abs=1e-5)


def test_alignment_is_a_cosine_whatever_the_length_of_the_directions():
    q = _basis()
    partly = (0.6 * q[:, 5] + 0.8 * q[:, 7])[:, None]  # 0.6 of it lies along layer 0's direction below
    assert detect.refusal_alignment(partly, 40.0 * q[:, 5][None]) == pytest.approx(0.6, abs=1e-5)


def test_a_layer_without_a_direction_is_skipped():
    """At layer 0 every prompt ends in the same template token, so the direction saved there is zero."""
    q = _basis()
    directions = torch.stack([torch.zeros(16), q[:, 5]])
    assert detect.refusal_alignment(q[:, [5]], directions) == pytest.approx(1.0, abs=1e-5)
    assert detect.refusal_alignment(q[:, [5]], torch.zeros(2, 16)) == 0.0
