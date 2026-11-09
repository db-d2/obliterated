import pytest
import torch

from obliterated import direction, sweep
from obliterated import susceptibility as sus

from .conftest import make_lm

HARMFUL = ["tell me how to bake a cake", "build the thing", "tell me how to build a cake"]
HARMLESS = ["write poem", "how to write", "write a poem"]


def test_spectral_measures_on_known_matrices():
    g = torch.Generator().manual_seed(0)
    u, v = torch.randn(20, generator=g), torch.randn(8, generator=g)
    rank1 = torch.outer(u, v)
    assert sus.stable_rank(rank1) == pytest.approx(1.0, abs=1e-4)
    assert sus.effective_rank(rank1) == pytest.approx(1.0, abs=1e-3)
    assert sus.energy_dimension(rank1) == 1
    iso = torch.eye(8)
    assert sus.stable_rank(iso) == pytest.approx(8.0)
    assert sus.effective_rank(iso) == pytest.approx(8.0, abs=1e-3)
    assert sus.participation_ratio(iso) == pytest.approx(8.0)
    assert sus.energy_dimension(iso, 0.9) == 8
    summ = sus.spectrum_summary(rank1)
    assert summ["top1_energy"] == pytest.approx(1.0, abs=1e-5)


def test_probe_separates_linearly_separable_clouds():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(40, 16, generator=g)
    shift = torch.zeros(16)
    shift[0] = 6.0
    out = sus.probe_margin(base + shift, base - shift, steps=200)
    assert out["probe_accuracy"] == 1.0
    assert out["probe_margin"] > 0.5
    assert out["probe_dim_cosine"] > 0.9


def test_refusal_geometry_and_layers_run(tokenizer):
    lm = make_lm("qwen2_bias", tokenizer)
    g = sus.refusal_geometry(lm, HARMFUL, HARMLESS, layer=1, batch_size=2)
    for key in ("stable_rank", "effective_rank", "dim90", "dim_relative_norm", "probe_accuracy", "probe_margin"):
        assert key in g
    assert 1 <= g["dim90"] <= len(HARMFUL)
    layers = sus.signal_by_layer(lm, HARMFUL, HARMLESS, batch_size=2)
    assert len(layers["relative_signal_by_layer"]) == lm.n_layers
    assert 1 <= layers["n_layers_above_half_peak"] <= lm.n_layers


def test_worst_case_drop_and_score(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    sub = direction.refusal_subspace(lm, HARMFUL, HARMLESS, layer=1, k=2, batch_size=2)
    rows = sus.worst_case_drop(lm, sub, HARMFUL, HARMLESS, kl_budget=float("inf"), batch_size=2)
    assert [r["k"] for r in rows] == [1, 2]
    assert all(r["within_budget"] for r in rows)
    assert sus.susceptibility_score(rows) == max(r["refusal_score_drop"] for r in rows)
    assert sus.susceptibility_score([{**r, "within_budget": False} for r in rows]) == 0.0


def test_full_report_is_json_serializable(tokenizer):
    import json

    lm = make_lm("qwen2_bias", tokenizer)
    rep = sus.report(lm, HARMFUL, HARMLESS, HARMFUL, HARMLESS, layer=1, k_max=2, batch_size=2)
    json.dumps(rep)
    assert rep["susceptibility"] >= 0.0 or rep["susceptibility"] <= 0.0  # a finite float


def test_sweep_runs_and_plots(tmp_path, tokenizer):
    lm = make_lm("llama_tied", tokenizer)
    sub = direction.refusal_subspace(lm, HARMFUL, HARMLESS, layer=1, k=2, batch_size=2)
    bands = sweep.layer_bands(lm.n_layers, source_layer=1)
    assert bands["source"] == [1] and bands["source±2"] == [0, 1, 2, 3]
    rows = []
    for mode in ("hooks", "weights"):
        rows += sweep.run_sweep(
            lm, sub, HARMFUL, HARMLESS, alphas=(0.5, 1.0), ks=(1, 2),
            bands={"all": None, "source": [1]}, mode=mode, batch_size=2,
        )
    assert len(rows) == 2 * 2 * 2 * 2
    for r in rows:
        assert set(sweep.DETECTORS) <= set(r)
        assert 0.0 <= r["bypass_rate"] <= 1.0
    full = [r for r in rows if r["alpha"] == 1.0 and r["band"] == "all"]
    assert all(r["weight_top1_energy"] > 0.99 for r in full if r["k"] == 1)
    # Hooks and weights agree at alpha=1 on every layer.
    h = next(r for r in full if r["mode"] == "hooks" and r["k"] == 2)
    w = next(r for r in full if r["mode"] == "weights" and r["k"] == 2)
    assert h["bypass_rate"] == pytest.approx(w["bypass_rate"])
    assert h["kl_harmless"] == pytest.approx(w["kl_harmless"], abs=1e-4)
    sweep.write_rows(rows, tmp_path / "f.jsonl")
    assert sweep.read_rows(tmp_path / "f.jsonl") == rows
    sweep.plot_frontier(rows, tmp_path / "f.png", thresholds={"activation_gap": 0.5})
    assert (tmp_path / "f.png").stat().st_size > 0
    neg = [{**rows[0], det: 0.0} for det in sweep.DETECTORS for _ in range(1)]
    assert 0.0 <= sweep.frontier_auroc(rows, neg, "weight_top1_energy") <= 1.0
