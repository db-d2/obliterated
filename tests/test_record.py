"""The check that decides whether a scored checkpoint's weights may be deleted."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from obliterated import completions, detect, record, sweep

REPO, BASE = "someone/tune", "lab/base"
REV = "0123456789abcdef0123456789abcdef01234567"
PROMPTS = ["first request", "second request", "third request", "fourth request"]
MATRIX = "model.layers.1.mlp.down_proj.weight"


def _parts(scale: float = 1.0) -> dict:
    """What one scoring pass produces, with behaviour numbers small enough to check by hand."""
    delta = torch.randn(6, 10, generator=torch.Generator().manual_seed(0)) * scale
    fields = detect.weight_scores([(MATRIX, delta)], [1])
    spectra = fields.pop("per_matrix")
    s, u = detect.top_factors(delta, 8)
    wave = torch.tensor([[1.0, 3.0, 1.0, 3.0]] * 4)  # 4 layers, 4 prompts
    row = {
        "repo": REPO, "base": BASE, "label": "benign_finetune", "n_harmful": 4,
        "bypass_rate": 0.75, "baseline_bypass_rate": 0.0,
        "behaviour_kl_harmful": 1.0, "behaviour_refusal_drop": 3.0,
        "activation_gap": 0.5,  # the candidate's separation is half the base's on every layer
        "completions_max_new_tokens": 512, **fields,
    }
    tensors = {
        f"{MATRIX}.s": s, f"{MATRIX}.u": u.half(),
        "reference.directions": torch.eye(4, 6).half(),
        "behaviour.kl": torch.tensor([0.5, 1.5, 1.0, 1.0]),
        "behaviour.refusal_score_base": torch.tensor([2.0, 1.0, 3.0, 2.0]),
        "behaviour.refusal_score": torch.tensor([-1.0, 1.0, -2.0, -2.0]),
        "activation.harmful_base": wave, "activation.harmless_base": torch.zeros(4, 4),
        "activation.harmful": wave - 1.0, "activation.harmless": torch.zeros(4, 4),
    }
    def answer(repo: str) -> list[dict]:
        return [{"repo": repo, "prompt": p, "completion": "I can't help with that."} for p in PROMPTS]

    return {"row": row, "spectra": spectra, "tensors": tensors, "answers": answer(REPO), "base_answers": answer(BASE),
            "provenance": {"repo": REPO, "revision": REV, "base": BASE, "base_revision": REV[::-1]}}


def _problems(tmp_path, parts: dict) -> list[str]:
    out = tmp_path / "scores.jsonl"
    parts["provenance"].setdefault("prompts", record.save_prompts(out, BASE, {"harmful_val": ["a prompt"]}))
    record.write(out, REPO, parts["tensors"], parts["provenance"])
    record.spectra_path(out, REPO).parent.mkdir(parents=True, exist_ok=True)
    record.spectra_path(out, REPO).write_text(json.dumps(parts["spectra"]))
    for repo, rows in ((REPO, parts["answers"]), (BASE, parts["base_answers"])):
        if rows is not None:
            completions.save(rows, completions.path(out, repo))
    return record.validate(parts["row"], out, PROMPTS)


def _flagged(problems: list[str], word: str) -> bool:
    return any(word in p for p in problems)


def test_a_complete_record_has_no_problems(tmp_path):
    assert _problems(tmp_path, _parts()) == []


def test_missing_answers_are_a_problem(tmp_path):
    parts = _parts()
    parts["answers"] = parts["answers"][:3]
    assert _flagged(_problems(tmp_path, parts), "completions")


def test_answers_to_other_prompts_are_a_problem(tmp_path):
    parts = _parts()
    parts["answers"][0]["prompt"] = "a different request"
    assert _flagged(_problems(tmp_path, parts), "completions")


def test_a_base_without_answers_is_a_problem(tmp_path):
    parts = _parts()
    parts["base_answers"] = None
    assert _flagged(_problems(tmp_path, parts), BASE)


def test_answers_cut_off_inside_the_reasoning_are_a_problem(tmp_path):
    parts = _parts()
    for a in parts["answers"]:
        a["completion"] = "<think>still working through it"
    assert _flagged(_problems(tmp_path, parts), "reasoning")


def test_empty_answers_are_a_problem(tmp_path):
    parts = _parts()
    for a in parts["answers"]:
        a["completion"] = ""
    assert _flagged(_problems(tmp_path, parts), "empty")


def test_a_matrix_without_its_saved_directions_is_a_problem(tmp_path):
    parts = _parts()
    del parts["tensors"][f"{MATRIX}.u"]
    assert _flagged(_problems(tmp_path, parts), MATRIX)


def test_saved_singular_values_that_disagree_with_the_spectrum_are_a_problem(tmp_path):
    parts = _parts()
    parts["tensors"][f"{MATRIX}.s"] = parts["tensors"][f"{MATRIX}.s"] * 2
    assert _flagged(_problems(tmp_path, parts), MATRIX)


def test_a_stored_norm_half_a_percent_low_is_a_problem(tmp_path):
    """The size of the float32 undercount on an 8B fine-tune that a 1% tolerance let through."""
    parts = _parts()
    parts["spectra"][0]["fro_norm"] *= 0.994
    assert _flagged(_problems(tmp_path, parts), MATRIX)


def test_a_small_difference_is_held_to_the_same_relative_standard(tmp_path):
    """A light fine-tune moves a matrix by far less than 1; its numbers must still agree."""
    assert _problems(tmp_path, _parts(scale=1e-3)) == []
    parts = _parts(scale=1e-3)
    parts["spectra"][0]["fro_norm"] *= 0.9
    assert _flagged(_problems(tmp_path / "off", parts), MATRIX)


def _with_embedding(parts: dict, norm_factor: float) -> dict:
    """Add an embedding difference whose stored norm is ``norm_factor`` times the true one."""
    delta = torch.randn(6, 40, generator=torch.Generator().manual_seed(1))
    s, u = detect.top_factors(delta, 8)
    parts["tensors"] |= {f"{detect.EMBEDDING}.s": s, f"{detect.EMBEDDING}.u": u.half(),
                         f"{detect.EMBEDDING}.fro": torch.tensor(detect.frobenius(delta) * norm_factor)}
    return parts


def test_an_embedding_norm_consistent_with_its_singular_values_is_fine(tmp_path):
    assert _problems(tmp_path, _with_embedding(_parts(), 1.0)) == []


def test_an_embedding_norm_smaller_than_its_own_singular_values_is_a_problem(tmp_path):
    """No matrix has less energy than its top singular values carry. The float32 undercount
    left an abliterated embedding's stored norm 17% under the truth."""
    assert _flagged(_problems(tmp_path, _with_embedding(_parts(), 0.83)), detect.EMBEDDING)


def test_a_non_finite_saved_tensor_is_a_problem(tmp_path):
    parts = _parts()
    parts["tensors"]["reference.directions"] = torch.full((4, 6), float("nan")).half()
    assert _flagged(_problems(tmp_path, parts), "reference.directions")


def test_a_score_the_saved_measurements_do_not_reproduce_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"]["behaviour_kl_harmful"] = 1.2  # the per-prompt values average 1.0
    assert _flagged(_problems(tmp_path, parts), "behaviour_kl_harmful")


def test_an_activation_gap_the_saved_projections_do_not_reproduce_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"]["activation_gap"] = 0.9
    assert _flagged(_problems(tmp_path, parts), "activation_gap")


def test_a_non_finite_score_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"]["weight_top1_energy_w"] = float("nan")
    assert _flagged(_problems(tmp_path, parts), "weight_top1_energy_w")


def test_a_checkpoint_identical_to_its_base_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"]["weight_n_changed"], parts["spectra"] = 0, []
    assert _flagged(_problems(tmp_path, parts), "no writer matrix")


def test_an_unpinned_revision_is_a_problem(tmp_path):
    parts = _parts()
    parts["provenance"]["revision"] = "main"
    assert _flagged(_problems(tmp_path, parts), "revision")


def test_nothing_saved_at_all_is_reported_not_raised(tmp_path):
    problems = record.validate(_parts()["row"], tmp_path / "scores.jsonl", PROMPTS)
    assert _flagged(problems, "record") and _flagged(problems, "completions")


def test_snapshot_info_pins_the_revision_and_the_files(tmp_path):
    blobs = tmp_path / "models--a--b" / "blobs"
    snap = tmp_path / "models--a--b" / "snapshots" / REV
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "abc").write_bytes(b"0123456789")
    (snap / "model.safetensors").symlink_to("../../blobs/abc")
    assert record.snapshot_info(snap) == {"revision": REV, "files": [{"name": "model.safetensors", "bytes": 10, "blob": "abc"}]}


def test_a_record_whose_prompt_set_is_gone_is_a_problem(tmp_path):
    parts = _parts()
    parts["provenance"]["prompts"] = "lab__base__00000000.json"  # named, never written
    assert _flagged(_problems(tmp_path, parts), "prompt set")


def test_prompt_sets_are_kept_side_by_side_not_overwritten(tmp_path):
    out = tmp_path / "scores.jsonl"
    first = record.save_prompts(out, BASE, {"harmful_val": ["x"]})
    second = record.save_prompts(out, BASE, {"harmful_val": ["y"]})
    assert first != second
    assert record.save_prompts(out, BASE, {"harmful_val": ["x"]}) == first
    assert json.loads((tmp_path / "prompts" / first).read_text())["harmful_val"] == ["x"]
    assert json.loads((tmp_path / "prompts" / second).read_text())["harmful_val"] == ["y"]


def test_a_null_control_is_expected_to_match_its_base(tmp_path):
    parts = _parts()
    parts["row"]["label"] = "base"  # an unmodified mirror: no matrix should differ
    parts["row"]["weight_n_changed"], parts["spectra"] = 0, []
    assert _problems(tmp_path, parts) == []


def test_a_null_control_scored_by_the_real_code_validates(tmp_path):
    """Nothing changed, so the stable rank of the change is undefined, not wrong."""
    parts = _parts()
    fields = detect.weight_scores([], [1])
    parts["spectra"] = fields.pop("per_matrix")
    parts["row"] |= {"label": "base", **fields}
    assert math.isnan(parts["row"]["weight_stable_rank"])
    del parts["tensors"][f"{MATRIX}.s"], parts["tensors"][f"{MATRIX}.u"]
    assert _problems(tmp_path, parts) == []


def test_a_fine_tune_that_left_the_middle_layers_alone_validates(tmp_path):
    """The stable rank is a mean over the mid-stack matrices that changed; here there are none."""
    parts = _parts()
    delta = torch.randn(6, 10, generator=torch.Generator().manual_seed(0))
    fields = detect.weight_scores([(MATRIX, delta)], [7])  # layer 1 changed; the mid stack is elsewhere
    parts["spectra"] = fields.pop("per_matrix")
    parts["row"] |= fields
    assert math.isnan(parts["row"]["weight_stable_rank"]) and parts["row"]["weight_n_changed"] == 1
    assert _problems(tmp_path, parts) == []


def test_an_undefined_stable_rank_with_middle_layers_changed_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"]["weight_stable_rank"] = float("nan")
    assert _flagged(_problems(tmp_path, parts), "weight_stable_rank")


def test_any_other_undefined_weight_score_is_a_problem_even_for_a_null_control(tmp_path):
    parts = _parts()
    fields = detect.weight_scores([], [1])
    parts["spectra"] = fields.pop("per_matrix")
    parts["row"] |= {"label": "base", **fields, "weight_top1_energy_w": float("nan")}
    assert _flagged(_problems(tmp_path, parts), "weight_top1_energy_w")


def _aligned() -> dict:
    """Record tensors with hand-made factors: three writer matrices and the embedding, against
    four layers whose refusal directions are the first four axes of a 6-dimensional stream."""
    e = torch.eye(6)
    factors = {
        "model.layers.0.mlp.down_proj.weight": e[:, [4, 5]],                         # unrelated: 0
        "model.layers.1.self_attn.o_proj.weight": e[:, [1]],                         # layer 1's direction: 1
        "model.layers.2.mlp.down_proj.weight": (0.6 * e[:, 2] + 0.8 * e[:, 5])[:, None],  # 0.6 of layer 2's
        detect.EMBEDDING: e[:, [3]],                                                 # the embedding: 1
    }
    kept = {"reference.directions": torch.eye(4, 6).half()}
    for name, u in factors.items():
        kept |= {f"{name}.u": u.half(), f"{name}.s": torch.ones(u.shape[1])}
    return kept


def test_alignment_summarises_the_writer_matrices_and_reports_the_vocabulary_apart():
    scores = record.alignment(_aligned())
    assert scores["weight_refusal_alignment"] == pytest.approx(1.0, abs=1e-3)
    assert scores["weight_refusal_alignment_med"] == pytest.approx(0.6, abs=1e-3)
    assert scores["weight_vocab_alignment"] == pytest.approx(1.0, abs=1e-3)


def test_a_writer_edit_does_not_count_as_a_vocabulary_edit_or_the_reverse():
    kept = _aligned()
    del kept[f"{detect.EMBEDDING}.u"], kept[f"{detect.EMBEDDING}.s"]
    assert record.alignment(kept)["weight_vocab_alignment"] == 0.0
    only_vocab = {k: v for k, v in _aligned().items() if not k.startswith("model.layers.")}
    scores = record.alignment(only_vocab)
    assert (scores["weight_refusal_alignment"], scores["weight_refusal_alignment_med"]) == (0.0, 0.0)
    assert scores["weight_vocab_alignment"] == pytest.approx(1.0, abs=1e-3)


def test_a_checkpoint_that_changed_nothing_aligns_with_nothing():
    assert record.alignment({"reference.directions": torch.eye(4, 6).half()}) == {
        "weight_refusal_alignment": 0.0, "weight_refusal_alignment_med": 0.0, "weight_vocab_alignment": 0.0}


def test_a_row_whose_alignment_the_record_reproduces_validates(tmp_path):
    parts = _parts()
    parts["row"] |= record.alignment(parts["tensors"])
    assert _problems(tmp_path, parts) == []


def test_an_alignment_the_record_does_not_reproduce_is_a_problem(tmp_path):
    parts = _parts()
    parts["row"] |= record.alignment(parts["tensors"]) | {"weight_refusal_alignment_med": 0.123}
    assert _flagged(_problems(tmp_path, parts), "weight_refusal_alignment_med")


def _script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_code_version_is_taken_once_per_process(monkeypatch):
    """A run that lasts days must record the code it loaded, not whatever is on disk when a
    later checkpoint finishes."""
    script = _script("detect_checkpoints")
    first = script.versions()
    monkeypatch.setattr(script.hashlib, "sha256", lambda *_: pytest.fail("the sources were hashed a second time"))
    assert script.versions() == first


def _scored(tmp_path) -> Path:
    """A results file with one row that has a record and one, scored earlier, that has none."""
    out = tmp_path / "scores.jsonl"
    record.write(out, REPO, _aligned(), {"repo": REPO})
    sweep.write_rows([{"repo": REPO, "label": "abliterated"}, {"repo": "someone/older", "label": "benign_finetune"}], out)
    return out


def test_alignment_is_added_to_the_rows_that_have_a_record(tmp_path):
    out = _scored(tmp_path)
    _script("detect_checkpoints").add_alignment(out)
    with_record, without = sweep.read_rows(out)
    assert with_record["weight_refusal_alignment_med"] == pytest.approx(0.6, abs=1e-3)
    assert without == {"repo": "someone/older", "label": "benign_finetune"}


def test_a_results_file_that_grew_meanwhile_is_not_overwritten(tmp_path, monkeypatch):
    """A scoring run appends to the same file; rewriting it from a stale read would drop a row."""
    out = _scored(tmp_path)
    script = _script("detect_checkpoints")
    real = record.alignment

    def a_live_run_appends(kept):
        sweep.write_rows([{"repo": "someone/new", "label": "benign_finetune"}], out)
        return real(kept)

    monkeypatch.setattr(script.record, "alignment", a_live_run_appends)
    with pytest.raises(SystemExit):
        script.add_alignment(out)
    rows = sweep.read_rows(out)
    assert [r["repo"] for r in rows] == [REPO, "someone/older", "someone/new"]
    assert "weight_refusal_alignment" not in rows[0]
