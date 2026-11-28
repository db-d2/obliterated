"""scripts/answer_divergence.py: how far a checkpoint the main pass scored stands from its base
over whole answers to harmless prompts."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import divergence, record, sweep
from obliterated.edits import blind_readers
from obliterated.refusal import kl_div, last_logits

from .conftest import load_script
from .test_edits import PROMPTS
from .test_readers import BASE, BASE_REV, REPO, REV, _edit_json, _flagged, pair, snapshot

HARMFUL, HARMLESS = PROMPTS[:2], PROMPTS[2:]


@pytest.fixture
def scored(tmp_path, tokenizer, monkeypatch):
    """A base and a checkpoint of it on disk (the base with its input weights edited), scored
    by the main pass: its row, the record that pins what was scored, and the prompts it used."""
    script = load_script("answer_divergence")
    base, cand = pair("llama_untied", tokenizer)
    blind_readers(cand, torch.linalg.qr(torch.randn(base.d_model, 8, generator=torch.Generator().manual_seed(1))).Q.T)
    folders = {BASE: snapshot(tmp_path / "hub" / "base", base, BASE_REV), REPO: snapshot(tmp_path / "hub" / "cand", cand)}
    tokenizer.save_pretrained(folders[BASE])
    rows = tmp_path / "main" / "scores.jsonl"
    rows.parent.mkdir()
    per_prompt = kl_div(last_logits(base, HARMFUL, 2), last_logits(cand, HARMFUL, 2))
    harmful = float(per_prompt.mean())
    sweep.write_rows([{"repo": BASE, "base": BASE, "label": "base"},  # a base's own row is not a checkpoint to measure
                      {"repo": REPO, "base": BASE, "label": "abliterated", "family": "llama", "params_b": 0.001, "behaviour_kl_harmful": harmful}], rows)
    prompts = record.save_prompts(rows, BASE, {"base_revision": BASE_REV, "harmful_val": HARMFUL, "harmless_val": HARMLESS})
    record.write(rows, REPO, {"behaviour.kl": per_prompt}, {
        "repo": REPO, "base": BASE, "revision": REV, "files": record.snapshot_info(folders[REPO])["files"],
        "base_revision": BASE_REV, "base_files": record.snapshot_info(folders[BASE])["files"], "prompts": prompts})
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: folders[repo])
    monkeypatch.setattr(script, "fetch", lambda *a, **k: pytest.fail("nothing is downloaded unless it is asked for"))
    monkeypatch.setattr(script, "evict", lambda *a, **k: pytest.fail("nothing is deleted unless it is asked for"))
    monkeypatch.setattr(script, "load_registry", list)
    monkeypatch.setattr(script, "bases", lambda registry: {BASE})
    return {"script": script, "rows": rows, "out": tmp_path / "answers" / "scores.jsonl", "folders": folders, "base": base, "cand": cand,
            "harmful": harmful}


def run(s: dict, monkeypatch, *extra: str) -> list[dict]:
    monkeypatch.setattr("sys.argv", ["answer_divergence.py", "--rows", str(s["rows"]), "--out", str(s["out"]), "--tokens", "5",
                                     "--batch-size", "2", "--device", "cpu", *extra])
    s["script"].main()
    return sweep.read_rows(s["out"]) if s["out"].exists() else []


def test_a_checkpoint_gets_a_row_of_how_far_it_stands_from_its_base_over_whole_answers(scored, monkeypatch):
    (row,) = run(scored, monkeypatch)
    assert (row["repo"], row["base"], row["label"], row["revision"], row["max_new_tokens"]) == (REPO, BASE, "abliterated", REV, 5)
    base = divergence.base_answers(scored["base"], HARMLESS, 5, 2)
    want = divergence.measure(scored["cand"], base, 2)
    assert want["answer_kl"] > 1e-3 and want["answer_tokens"] == 5 * len(HARMLESS)
    assert {k: row[k] for k in want} == pytest.approx(want, rel=1e-4)


def test_the_row_has_the_first_token_kl_on_the_main_passes_harmful_prompts_measured_again(scored, monkeypatch):
    (row,) = run(scored, monkeypatch)
    assert scored["harmful"] > 1e-3 and row["behaviour_kl_harmful"] == pytest.approx(scored["harmful"], rel=1e-4)


def test_what_is_kept_is_the_two_numbers_at_every_token_and_what_was_read(scored, monkeypatch):
    run(scored, monkeypatch)
    kept = load_file(record.tensors_path(scored["out"], REPO))
    base = divergence.base_answers(scored["base"], HARMLESS, 5, 2)
    drop, kl = divergence.per_token(scored["cand"], base, 2)
    assert torch.allclose(kept["answer.drop"], drop, atol=1e-5) and torch.allclose(kept["answer.kl"], kl, atol=1e-5)
    assert kept["answer.lengths"].tolist() == [len(a) for a in base.tokens]
    made = json.loads(record.provenance_path(scored["out"], REPO).read_text())
    assert (made["revision"], made["base_revision"], made["max_new_tokens"], made["tokenizer"]) == (REV, BASE_REV, 5, "base")
    assert made["files"] == record.snapshot_info(scored["folders"][REPO])["files"]  # what was read, not what was expected
    assert "versions" in made and made["prompts"] == json.loads(record.provenance_path(scored["rows"], REPO).read_text())["prompts"]
    answers = json.loads((scored["out"].parent / "prompts" / made["answers"]).read_text())
    assert answers["harmless_val"] == HARMLESS and answers["answer_tokens"] == base.tokens and answers["max_new_tokens"] == 5


def test_a_second_run_measures_nothing_again(scored, monkeypatch):
    rows = run(scored, monkeypatch)
    monkeypatch.setattr(scored["script"], "load", lambda *a, **k: pytest.fail("everything is measured already"))
    assert run(scored, monkeypatch) == rows


# ---- what the pass keeps of a checkpoint, and the check that it is all there


def problems(s: dict, row: dict | None = None) -> list[str]:
    return record.validate_answers(row or sweep.read_rows(s["out"])[-1], s["out"], s["rows"])


def test_what_the_pass_keeps_of_a_checkpoint_validates(scored, monkeypatch):
    run(scored, monkeypatch)
    assert problems(scored) == []


def change_record(s: dict, change, of: str = "out") -> None:
    path = record.tensors_path(s[of], REPO)
    kept = load_file(path)
    change(kept)
    save_file(kept, path)


def test_a_number_the_record_does_not_give_back_is_a_problem(scored, monkeypatch):
    (row,) = run(scored, monkeypatch)
    for key in ("answer_kl", "answer_logprob_drop", "answer_first_kl"):
        assert _flagged(problems(scored, {**row, key: row[key] * 1.01}), key)
    assert _flagged(problems(scored, {**row, "answer_tokens": row["answer_tokens"] + 1}), "answer_tokens")
    change_record(scored, lambda kept: kept.update({"answer.kl": kept["answer.kl"] * 1.01}))
    assert _flagged(problems(scored), "answer_kl")


def test_a_number_that_is_not_finite_is_a_problem(scored, monkeypatch):
    run(scored, monkeypatch)
    change_record(scored, lambda kept: kept["answer.drop"].__setitem__(0, float("nan")))
    assert _flagged(problems(scored), "not finite")


def test_answers_other_than_the_ones_the_numbers_were_taken_over_are_a_problem(scored, monkeypatch):
    run(scored, monkeypatch)
    made = json.loads(record.provenance_path(scored["out"], REPO).read_text())
    answers = scored["out"].parent / "prompts" / made["answers"]
    _edit_json(answers, lambda b: b["answer_tokens"][0].pop())
    assert _flagged(problems(scored), "answers")
    answers.unlink()
    assert _flagged(problems(scored), "answers")


def test_a_revision_other_than_the_one_the_main_pass_scored_is_a_problem(scored, monkeypatch):
    run(scored, monkeypatch)
    _edit_json(record.provenance_path(scored["rows"], REPO), lambda b: b.update(revision=BASE_REV))
    assert _flagged(problems(scored), "revision")


def test_weights_other_than_the_ones_the_main_pass_scored_are_a_problem(scored, monkeypatch):
    run(scored, monkeypatch)
    _edit_json(record.provenance_path(scored["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert _flagged(problems(scored), "not the files")
    _edit_json(record.provenance_path(scored["out"], REPO), lambda b: b.update(base_revision=REV))
    assert _flagged(problems(scored), "base_revision")


def test_a_first_token_kl_on_harmful_prompts_that_is_not_the_main_passes_is_a_problem(scored, monkeypatch):
    """It is measured again so that a checkpoint loaded some other way than the main pass
    loaded it does not go unnoticed."""
    run(scored, monkeypatch)
    change_record(scored, lambda kept: kept.update({"behaviour.kl": kept["behaviour.kl"] + 0.001}), of="rows")
    assert problems(scored) == []  # within what two runs of one model differ by
    change_record(scored, lambda kept: kept.update({"behaviour.kl": kept["behaviour.kl"] + 0.01}), of="rows")
    assert _flagged(problems(scored), "harmful prompts")


def test_no_record_at_all_is_reported_not_raised(scored, tmp_path):
    assert _flagged(record.validate_answers({"repo": REPO}, scored["out"], scored["rows"]), "no record")


def test_the_pass_validates_each_row_as_it_is_written(scored, monkeypatch, capsys):
    run(scored, monkeypatch)
    assert "validated" in capsys.readouterr().out


def test_the_check_is_the_validation_and_loads_nothing(scored, monkeypatch, capsys):
    run(scored, monkeypatch)
    monkeypatch.setattr(scored["script"], "load", lambda *a, **k: pytest.fail("the check reads what was kept, not the weights"))
    with pytest.raises(SystemExit) as ok:
        run(scored, monkeypatch, "--check")
    assert ok.value.code == 0 and "ok" in capsys.readouterr().out
    _edit_json(record.provenance_path(scored["rows"], REPO), lambda b: b.update(revision=BASE_REV))
    with pytest.raises(SystemExit) as bad:
        run(scored, monkeypatch, "--check")
    assert bad.value.code == 1 and "PROBLEMS" in capsys.readouterr().out


def test_a_checkpoint_that_is_not_on_disk_as_it_was_scored_is_left_out_and_nothing_is_fetched(scored, monkeypatch, capsys):
    _edit_json(record.provenance_path(scored["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert run(scored, monkeypatch) == []  # the fixture fails the test if anything is fetched or loaded for it
    assert "left out" in capsys.readouterr().out


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


def test_a_checkpoint_that_is_not_on_disk_is_fetched_at_the_revision_that_was_scored(scored, monkeypatch, tmp_path):
    fetched, deleted = away(scored, monkeypatch, tmp_path)
    (row,) = run(scored, monkeypatch, "--fetch")
    assert fetched == [(REPO, REV)] and deleted == [] and mine(scored) == [REPO]
    assert row["answer_kl"] > 1e-3 and problems(scored) == []


def test_a_checkpoint_the_pass_fetched_is_deleted_once_what_it_kept_validates(scored, monkeypatch, tmp_path, capsys):
    fetched, deleted = away(scored, monkeypatch, tmp_path)
    assert len(run(scored, monkeypatch, "--evict")) == 1
    assert fetched == [(REPO, REV)] and deleted == [(REPO, {BASE})]
    assert "validated" in capsys.readouterr().out


def test_a_checkpoint_that_was_on_disk_already_is_left_there(scored, monkeypatch, capsys):
    assert len(run(scored, monkeypatch, "--evict")) == 1 and mine(scored) == []  # the fixture fails the test if anything is deleted
    assert "was there before this pass" in capsys.readouterr().out


def test_a_checkpoint_whose_record_has_a_problem_stays_on_disk(scored, monkeypatch, tmp_path, capsys):
    fetched, deleted = away(scored, monkeypatch, tmp_path)
    change_record(scored, lambda kept: kept.update({"behaviour.kl": kept["behaviour.kl"] + 0.01}), of="rows")
    assert len(run(scored, monkeypatch, "--evict")) == 1
    assert fetched == [(REPO, REV)] and deleted == []
    assert "did not validate" in capsys.readouterr().out


def test_a_download_that_was_interrupted_is_still_the_passes_own(scored, monkeypatch):
    """The first run notes the download before it starts; a second finds the files on disk and
    must still take them for its own."""
    deleted = []
    monkeypatch.setattr(scored["script"], "evict", lambda repo, keep: deleted.append(repo) or 1)
    scored["out"].parent.mkdir(parents=True)
    scored["out"].with_suffix(".downloaded.json").write_text(json.dumps([REPO]))
    assert len(run(scored, monkeypatch, "--evict")) == 1 and deleted == [REPO]


def test_a_download_that_fails_is_reported_and_the_pass_goes_on(scored, monkeypatch, tmp_path, capsys):
    away(scored, monkeypatch, tmp_path)

    def gone(repo, revision=None):
        raise OSError("the repository is not there any more")

    monkeypatch.setattr(scored["script"], "fetch", gone)
    assert run(scored, monkeypatch, "--evict") == []
    assert "FAILED someone/tune" in capsys.readouterr().out


def test_what_comes_down_has_to_be_the_files_that_were_scored_before(scored, monkeypatch, tmp_path, capsys):
    fetched, deleted = away(scored, monkeypatch, tmp_path)
    _edit_json(record.provenance_path(scored["rows"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert run(scored, monkeypatch, "--evict") == []
    assert len(fetched) == 1 and deleted == []
    assert "not the files the main pass scored" in capsys.readouterr().out


def test_a_base_is_never_fetched(scored, monkeypatch, tmp_path, capsys):
    script = scored["script"]
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: scored["folders"][repo] if repo == REPO else tmp_path / "nowhere")
    assert run(scored, monkeypatch, "--fetch") == []  # the fixture fails the test if anything is fetched
    assert "left out" in capsys.readouterr().out


def test_the_pass_stops_before_a_download_when_the_disk_is_nearly_full(scored, monkeypatch, tmp_path, capsys):
    fetched, _ = away(scored, monkeypatch, tmp_path)
    assert run(scored, monkeypatch, "--fetch", "--min-free-gb", "1e9") == [] and fetched == []
    assert "stopping" in capsys.readouterr().out
