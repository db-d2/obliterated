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
