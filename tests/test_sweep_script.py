"""scripts/sweep_frontier.py: a sweep of the attack family, one row per point."""

from obliterated import sweep

from .conftest import load_script, make_lm
from .test_susceptibility import HARMFUL, HARMLESS


def run(tmp_path, tokenizer, monkeypatch, *extra: str) -> list[dict]:
    script = load_script("sweep_frontier")
    monkeypatch.setattr(script, "load", lambda name, device=None: make_lm("llama_untied", tokenizer))
    monkeypatch.setattr(script.data, "contrast_sets", lambda source, n_train, n_val: (HARMFUL, HARMLESS, HARMFUL, HARMLESS))
    monkeypatch.setattr(script.direction, "filter_by_baseline", lambda lm, harmful, harmless, bs: (harmful, harmless))
    monkeypatch.setattr("sys.argv", ["sweep_frontier.py", "--model", "org/tiny", "--layer", "1", "--ks", "1,2", "--alphas", "1.0", "--bands", "all",
                                     "--mode", "readers", "--batch-size", "2", "--out", str(tmp_path), *extra])
    script.main()
    return sweep.read_rows(tmp_path / "org__tiny" / "frontier.jsonl")


def test_a_sweep_has_no_whole_answer_columns_unless_asked(tmp_path, tokenizer, monkeypatch):
    rows = run(tmp_path, tokenizer, monkeypatch)
    assert len(rows) == 2 and not any(k.startswith("answer_") for r in rows for k in r)


def test_a_sweep_asked_for_divergence_tokens_measures_every_point_over_whole_answers(tmp_path, tokenizer, monkeypatch):
    rows = run(tmp_path, tokenizer, monkeypatch, "--divergence-tokens", "5")
    assert len(rows) == 2
    for r in rows:
        assert r["answer_prompts"] == len(HARMLESS) and r["answer_tokens"] == 5 * len(HARMLESS)
        assert r["answer_kl"] > 0 and abs(r["answer_first_kl"] - r["kl_harmless"]) < 1e-6
