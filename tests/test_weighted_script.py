"""scripts/weighted_readers.py: checkpoints the reader pass has read, scored again with each change weighed by the base's stream."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import record, stream, sweep

from .conftest import load_script
from .test_edits import PROMPTS
from .test_readers import BASE, BASE_REV, REPO, REV, _edit_json, _flagged, pair, scored  # noqa: F401  (scored is a fixture)


@pytest.fixture
def passed(scored, tokenizer, tmp_path, monkeypatch):  # noqa: F811
    """The checkpoint of ``scored`` (a reader-side edit, read by the reader pass), the base as
    a model, and the folders this pass reads and writes."""
    script = load_script("weighted_readers")
    sweep.write_rows([{"repo": BASE, "base": BASE, "label": "base"}], scored["out"])  # a base's own row is not a checkpoint to score
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: scored["folders"][repo])
    monkeypatch.setattr(script, "load", lambda *a, **k: pytest.fail("no model is loaded unless the stream is to be measured"))
    monkeypatch.setattr(script, "fetch", lambda *a, **k: pytest.fail("nothing is downloaded unless it is asked for"))
    monkeypatch.setattr(script, "evict", lambda *a, **k: pytest.fail("nothing is deleted unless it is asked for"))
    monkeypatch.setattr(script, "load_registry", list)
    monkeypatch.setattr(script, "bases", lambda registry: {BASE})
    base, _ = pair("llama_untied", tokenizer)
    return {"script": script, "rows": scored["out"], "out": tmp_path / "weighted" / "scores.jsonl", "stream": tmp_path / "stream", "base": base,
            "folders": scored["folders"]}


def measured(s: dict) -> None:
    """The base's stream, as the detector would have measured it beforehand."""
    means, moments, _ = stream.measure(s["base"], [s["base"].format(p) for p in PROMPTS])
    s["stream"].mkdir()
    stream.save(stream.path(s["stream"], BASE), means, moments)


def run(s: dict, monkeypatch, *extra: str) -> list[dict]:
    monkeypatch.setattr("sys.argv", ["weighted_readers.py", "--rows", str(s["rows"]), "--out", str(s["out"]), "--stream", str(s["stream"]), *extra])
    s["script"].main()
    return sweep.read_rows(s["out"]) if s["out"].exists() else []


def test_a_checkpoint_gets_a_row_of_scores_and_a_record_of_what_was_kept(passed, monkeypatch):
    measured(passed)
    (row,) = run(passed, monkeypatch)
    assert (row["repo"], row["base"], row["label"]) == (REPO, BASE, "abliterated")
    assert row["reader_weighted_shared_k8"] == pytest.approx(1.0, abs=1e-3)  # every reading matrix ignores the one direction
    assert row["reader_weighted_shared_k128"] == pytest.approx(1.0, abs=1e-3)
    assert row["reader_weighted_shared_n"] == 5 * passed["base"].n_layers
    kept = load_file(record.tensors_path(passed["out"], REPO))
    assert sum(key.endswith(".wu") for key in kept) == row["reader_weighted_shared_n"]
    assert {key.rsplit(".", 1)[1] for key in kept} == {"wu", "ws", "wt"}
    made = json.loads(record.provenance_path(passed["out"], REPO).read_text())
    assert made["base_revision"] == BASE_REV and made["n_kept"] == 128 and made["weighed_by"] == "second moment"
    assert made["parts"] == ["wu", "ws", "wt"]
    assert made["files"] == record.snapshot_info(passed["folders"][REPO])["files"]  # what was read, not what was expected
    assert made["stream"]["sha256"] and "versions" in made
    assert run(passed, monkeypatch) == [row]  # a second run reads nothing again


def test_a_row_its_record_does_not_reproduce_fails_the_check(passed, monkeypatch):
    measured(passed)
    (row,) = run(passed, monkeypatch)
    monkeypatch.setattr(passed["script"].record, "weighted_readers", lambda *a, **k: pytest.fail("the check reads no weights"))
    with pytest.raises(SystemExit) as ok:
        run(passed, monkeypatch, "--check")
    assert ok.value.code == 0
    passed["out"].write_text(json.dumps({**row, "reader_weighted_shared_k8": 0.5}) + "\n")
    with pytest.raises(SystemExit) as bad:
        run(passed, monkeypatch, "--check")
    assert bad.value.code == 1


def test_a_base_whose_stream_was_not_measured_is_left_out_unless_measuring_is_asked_for(passed, monkeypatch, capsys):
    script = passed["script"]
    assert run(passed, monkeypatch) == []
    assert "not measured" in capsys.readouterr().out
    monkeypatch.setattr(script, "load", lambda name, device=None: passed["base"])
    monkeypatch.setattr(script, "unload", lambda lm: None)
    monkeypatch.setattr(script.data, "arditi_split", lambda kind, split: {"harmful": PROMPTS[:2], "harmless": PROMPTS[2:]}[kind] if split == "test" else pytest.fail(split))
    real_generate, real_measure, answered, texts = script.generate, script.stream.measure, [], []
    monkeypatch.setattr(script, "generate", lambda lm, prompts, tokens, bs: answered.append((list(prompts), tokens, real_generate(lm, prompts, tokens, bs))) or answered[-1][2])
    monkeypatch.setattr(script.stream, "measure", lambda lm, t, bs: texts.append(list(t)) or real_measure(lm, t, bs))
    (row,) = run(passed, monkeypatch, "--measure", "--tokens", "3", "--batch-size", "2")
    assert row["reader_weighted_shared_k8"] == pytest.approx(1.0, abs=1e-3)
    lm = passed["base"]
    ((asked, tokens, answers),) = answered
    assert (asked, tokens) == (PROMPTS, 3)
    assert texts == [[lm.format(p) + a for p, a in zip(PROMPTS, answers)]]  # each prompt as the base is handed it, and the start of its own answer
    means, moments = stream.load(stream.path(passed["stream"], BASE))
    assert set(means) == set(moments) and len(moments) == 2 * lm.n_layers + 1  # two norms a layer and the last
    made = json.loads(stream.path(passed["stream"], BASE).with_suffix(".json").read_text())
    assert (made["base"], made["base_revision"], made["prompts"], made["harmful"], made["harmless"], made["answer_tokens"]) == (BASE, BASE_REV, "arditi test split", 2, 2, 3)
    assert made["tokens"] == sum(len(lm.tokenizer(t, add_special_tokens=False).input_ids) for t in texts[0]) and made["d_model"] == lm.d_model


def test_a_checkpoint_that_is_not_on_disk_as_it_was_read_is_left_out_and_nothing_is_fetched(passed, monkeypatch, capsys):
    measured(passed)
    info = record.provenance_path(passed["rows"], REPO)
    was = json.loads(info.read_text())
    info.write_text(json.dumps({**was, "files": [{**f, "bytes": f["bytes"] + 1} for f in was["files"]]}))
    assert run(passed, monkeypatch) == []
    assert "left out" in capsys.readouterr().out  # and the fixture fails the test if anything is fetched


# ---- what has to hold of a record before the weights it was made from can go


def problems(s: dict, row: dict | None = None) -> list[str]:
    return record.validate_weighted(row or sweep.read_rows(s["out"])[-1], s["out"], s["rows"])


def change_record(s: dict, change) -> None:
    path = record.tensors_path(s["out"], REPO)
    kept = load_file(path)
    change(kept)
    save_file(kept, path)


def test_what_the_pass_keeps_of_a_checkpoint_validates(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    assert problems(passed) == []


def test_a_score_the_record_does_not_give_is_a_problem(passed, monkeypatch):
    measured(passed)
    (row,) = run(passed, monkeypatch)
    assert _flagged(problems(passed, {**row, "reader_weighted_shared_k8": 0.5}), "reader_weighted_shared_k8")
    assert _flagged(problems(passed, {**row, "reader_weighted_shared_k128": float("nan")}), "not finite")


def test_weights_other_than_the_ones_the_reader_pass_read_are_a_problem(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    _edit_json(record.provenance_path(passed["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert _flagged(problems(passed), "not the files")


def test_a_revision_other_than_the_one_the_reader_pass_read_is_a_problem(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    _edit_json(record.provenance_path(passed["rows"], REPO), lambda b: b.update(base_revision=REV))
    assert _flagged(problems(passed), "base_revision")
    _edit_json(record.provenance_path(passed["out"], REPO), lambda b: b.update(revision="main"))
    assert _flagged(problems(passed), "not a pinned commit")


def test_a_matrix_that_changed_and_is_not_in_the_record_is_a_problem(passed, monkeypatch):
    """Which matrices changed is the reader pass's to say: it read the same weights."""
    measured(passed)
    run(passed, monkeypatch)
    change_record(passed, lambda kept: kept.pop("model.layers.1.self_attn.q_proj.weight.wu"))
    assert _flagged(problems(passed), "model.layers.1.self_attn.q_proj.weight")


def test_directions_that_are_not_orthonormal_columns_are_a_problem(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    name = "model.layers.0.mlp.up_proj.weight"
    change_record(passed, lambda kept: kept.update({f"{name}.wu": 2 * kept[f"{name}.wu"]}))
    assert _flagged(problems(passed), name)


def test_sizes_that_do_not_go_with_the_directions_are_a_problem(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    name = "model.layers.0.mlp.up_proj.weight"
    change_record(passed, lambda kept: kept.update({f"{name}.ws": kept[f"{name}.ws"][:-1]}))
    assert _flagged(problems(passed), name)
    run(passed, monkeypatch, "--redo")
    assert problems(passed) == []
    change_record(passed, lambda kept: kept.update({f"{name}.wt": kept[f"{name}.wt"] / 4}))
    assert _flagged(problems(passed), "whole")


def test_a_measurement_other_than_the_one_the_record_was_made_with_is_a_problem(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    path = stream.path(passed["stream"], BASE)
    means, moments = stream.load(path)
    stream.save(path, means, {norm: 2 * c for norm, c in moments.items()})
    assert _flagged(problems(passed), "not the file the record was made with")
    path.unlink()
    assert _flagged(problems(passed), "not on disk")


def test_no_record_at_all_is_reported_not_raised(passed, tmp_path):
    assert problems({**passed, "out": tmp_path / "elsewhere" / "scores.jsonl"}, {"repo": REPO})


def test_the_changes_own_factors_are_kept_and_checked_against_the_reader_pass_when_asked_for(passed, monkeypatch):
    measured(passed)
    assert len(run(passed, monkeypatch, "--unweighed")) == 1
    kept = load_file(record.tensors_path(passed["out"], REPO))
    assert {key.rsplit(".", 1)[1] for key in kept} == {"wu", "ws", "wt", "es", "eu"}
    assert json.loads(record.provenance_path(passed["out"], REPO).read_text())["parts"] == ["wu", "ws", "wt", "es", "eu"]
    assert problems(passed) == []
    name = "model.layers.2.self_attn.v_proj.weight"
    change_record(passed, lambda kept: kept.update({f"{name}.es": 1.5 * kept[f"{name}.es"]}))
    assert _flagged(problems(passed), "the reader pass found")
    change_record(passed, lambda kept: kept.pop(f"{name}.eu"))
    assert _flagged(problems(passed), f"{name}.eu")


def test_the_check_is_the_validation_and_reads_no_weights(passed, monkeypatch):
    measured(passed)
    run(passed, monkeypatch)
    monkeypatch.setattr(passed["script"].record, "weighted_readers", lambda *a, **k: pytest.fail("the check reads no weights"))
    _edit_json(record.provenance_path(passed["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    with pytest.raises(SystemExit) as bad:
        run(passed, monkeypatch, "--check")
    assert bad.value.code == 1


# ---- checkpoints that are not on disk: fetched when asked, deleted when asked and validated


def away(s: dict, monkeypatch, tmp_path, comes: bool = True) -> tuple[list, list]:
    """The checkpoint is not on disk until it is fetched. Returns what is fetched and what is deleted."""
    script, fetched, deleted = s["script"], [], []
    monkeypatch.setattr(script, "snapshot_folder",
                        lambda repo, revision: s["folders"][repo] if repo == BASE or (fetched and comes) else tmp_path / "nowhere" / revision)
    monkeypatch.setattr(script, "fetch", lambda repo, revision=None: fetched.append((repo, revision)) or s["folders"][repo])
    monkeypatch.setattr(script, "evict", lambda repo, keep: deleted.append((repo, set(keep))) or 1)
    return fetched, deleted


def mine(s: dict) -> list[str]:
    path = s["out"].with_suffix(".downloaded.json")
    return json.loads(path.read_text()) if path.exists() else []


def test_a_checkpoint_that_is_not_on_disk_is_fetched_at_the_revision_that_was_read(passed, monkeypatch, tmp_path):
    measured(passed)
    fetched, deleted = away(passed, monkeypatch, tmp_path)
    (row,) = run(passed, monkeypatch, "--fetch")
    assert fetched == [(REPO, REV)] and deleted == [] and mine(passed) == [REPO]
    assert row["reader_weighted_shared_k8"] == pytest.approx(1.0, abs=1e-3) and problems(passed) == []


def test_a_checkpoint_the_pass_fetched_is_deleted_once_what_it_kept_validates(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    fetched, deleted = away(passed, monkeypatch, tmp_path)
    assert len(run(passed, monkeypatch, "--evict")) == 1
    assert fetched == [(REPO, REV)] and deleted == [(REPO, {BASE})]
    assert "validated" in capsys.readouterr().out


def test_a_checkpoint_that_was_on_disk_already_is_left_there(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    monkeypatch.setattr(passed["script"], "evict", lambda *a, **k: pytest.fail("it was there before this pass"))
    assert len(run(passed, monkeypatch, "--evict")) == 1 and mine(passed) == []
    assert "was there before this pass" in capsys.readouterr().out


def test_a_checkpoint_whose_record_has_a_problem_stays_on_disk(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    fetched, deleted = away(passed, monkeypatch, tmp_path)
    _edit_json(record.provenance_path(passed["rows"], REPO), lambda b: b.update(revision="main"))
    assert len(run(passed, monkeypatch, "--evict")) == 1
    assert fetched == [(REPO, "main")] and deleted == []
    assert "did not validate" in capsys.readouterr().out


def test_a_download_that_was_interrupted_is_still_the_passes_own(passed, monkeypatch, tmp_path):
    """The first run notes the download before it starts; a second finds the files on disk and
    must still take them for its own."""
    measured(passed)
    deleted = []
    monkeypatch.setattr(passed["script"], "evict", lambda repo, keep: deleted.append(repo) or 1)
    passed["out"].parent.mkdir(parents=True)
    passed["out"].with_suffix(".downloaded.json").write_text(json.dumps([REPO]))
    assert len(run(passed, monkeypatch, "--evict")) == 1 and deleted == [REPO]


def test_a_download_that_fails_is_reported_and_the_pass_goes_on(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    away(passed, monkeypatch, tmp_path)

    def gone(repo, revision=None):
        raise OSError("the repository is not there any more")

    monkeypatch.setattr(passed["script"], "fetch", gone)
    assert run(passed, monkeypatch, "--evict") == []
    assert "FAILED someone/tune" in capsys.readouterr().out


def test_what_comes_down_has_to_be_the_files_that_were_read_before(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    fetched, deleted = away(passed, monkeypatch, tmp_path)
    _edit_json(record.provenance_path(passed["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert run(passed, monkeypatch, "--evict") == []
    assert len(fetched) == 1 and deleted == []
    assert "not the files the reader pass read" in capsys.readouterr().out


def test_a_base_is_never_fetched(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    script = passed["script"]
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: passed["folders"][repo] if repo == REPO else tmp_path / "nowhere")
    assert run(passed, monkeypatch, "--fetch") == []  # the fixture fails the test if anything is fetched
    assert "left out" in capsys.readouterr().out


def test_the_pass_stops_before_a_download_when_the_disk_is_nearly_full(passed, monkeypatch, tmp_path, capsys):
    measured(passed)
    fetched, _ = away(passed, monkeypatch, tmp_path)
    assert run(passed, monkeypatch, "--fetch", "--min-free-gb", "1e9") == [] and fetched == []
    assert "stopping" in capsys.readouterr().out


# ---- thresholds, and the sizes used together


def rows_file(tmp_path, rows: list[dict]):
    path = tmp_path / "weighted.jsonl"
    sweep.write_rows([{"base": BASE, **r} for r in rows], path)
    return path


def scores(k8: float, k128: float) -> dict:
    return {"reader_weighted_shared_k8": k8, "reader_weighted_shared_k128": k128, "reader_weighted_shared_n": 20}


def test_the_report_flags_what_is_above_the_largest_benign_score_at_either_size(tmp_path):
    script = load_script("weighted_readers")
    path = rows_file(tmp_path, [
        {"repo": "a/tune1", "label": "benign_finetune", **scores(0.30, 0.50)},
        {"repo": "a/tune2", "label": "benign_finetune", **scores(0.40, 0.45)},
        {"repo": "a/tune3", "label": "benign_finetune", **scores(0.20, 0.60)},
        {"repo": "a/complies", "label": "benign_finetune", **scores(0.90, 0.90)},  # a fine-tune the judge finds no longer refusing sets nothing
        {"repo": "b/few_large", "label": "abliterated", **scores(0.35, 0.70)},
        {"repo": "b/many_small", "label": "abliterated", **scores(0.45, 0.55)},
        {"repo": "b/neither", "label": "other_edit", **scores(0.40, 0.60)},  # at both thresholds, above neither
    ])
    judged = tmp_path / "judged.jsonl"
    sweep.write_rows([{"repo": BASE, "base": BASE, "strongreject_score": 0.0}]
                     + [{"repo": f"a/tune{i}", "base": BASE, "strongreject_score": 0.1} for i in (1, 2, 3)]
                     + [{"repo": "a/complies", "base": BASE, "strongreject_score": 0.6}], judged)
    got = script.report([path], judged, margin=0.3, fpr=0.0)
    assert got["negatives"] == 3
    assert got["thresholds"] == {"reader_weighted_shared_k8": 0.40, "reader_weighted_shared_k128": 0.60}
    assert got["flagged"] == {"a/complies": ["reader_weighted_shared_k8", "reader_weighted_shared_k128"],
                              "b/few_large": ["reader_weighted_shared_k128"], "b/many_small": ["reader_weighted_shared_k8"], "b/neither": []}
    assert got["held_out"] == {"a/tune1": [], "a/tune2": ["reader_weighted_shared_k8"], "a/tune3": ["reader_weighted_shared_k128"]}


def test_without_a_judges_rows_every_benign_fine_tune_sets_the_thresholds(tmp_path):
    script = load_script("weighted_readers")
    path = rows_file(tmp_path, [{"repo": "a/tune1", "label": "benign_finetune", **scores(0.30, 0.50)},
                                {"repo": "a/tune2", "label": "benign_finetune", **scores(0.90, 0.90)},
                                {"repo": "b/edit", "label": "abliterated", **scores(0.95, 0.20)}])
    got = script.report([path], None, margin=0.3, fpr=0.0)
    assert got["negatives"] == 2 and got["thresholds"]["reader_weighted_shared_k8"] == 0.90
    assert got["flagged"] == {"b/edit": ["reader_weighted_shared_k8"]}
    assert torch.isfinite(torch.tensor(list(got["thresholds"].values()))).all()
