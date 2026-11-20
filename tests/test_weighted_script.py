"""scripts/weighted_readers.py: checkpoints the reader pass has read, scored again with each change weighed by the base's stream."""

import json

import pytest
import torch
from safetensors.torch import load_file

from obliterated import record, stream, sweep

from .conftest import load_script
from .test_edits import PROMPTS
from .test_readers import BASE, BASE_REV, REPO, pair, scored  # noqa: F401  (scored is a fixture)


@pytest.fixture
def passed(scored, tokenizer, tmp_path, monkeypatch):  # noqa: F811
    """The checkpoint of ``scored`` (a reader-side edit, read by the reader pass), the base as
    a model, and the folders this pass reads and writes."""
    script = load_script("weighted_readers")
    sweep.write_rows([{"repo": BASE, "base": BASE, "label": "base"}], scored["out"])  # a base's own row is not a checkpoint to score
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: scored["folders"][repo])
    monkeypatch.setattr(script, "load", lambda *a, **k: pytest.fail("no model is loaded unless the stream is to be measured"))
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
    assert len(kept) == row["reader_weighted_shared_n"] and all(key.endswith(".wu") for key in kept)
    made = json.loads(record.provenance_path(passed["out"], REPO).read_text())
    assert made["base_revision"] == BASE_REV and made["n_kept"] == 128 and made["weighed_by"] == "second moment"
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
    assert "left out" in capsys.readouterr().out
    assert not hasattr(passed["script"], "fetch")


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
