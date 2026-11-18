"""scripts/padded_edits.py: a working edit spread over more directions or hidden among decoys,
scored from the matrices on disk and run from the same matrices."""

import pytest
import torch

from obliterated import completions, record, sweep, unlisted
from obliterated.edits import EditSpec
from obliterated.models import load

from .conftest import load_script
from .test_edits import PROMPTS, rand_dir
from .test_judge_edits import Grader
from .test_readers import BASE, BASE_REV, REPO, pair, snapshot

SLUG = BASE.replace("/", "__")


@pytest.fixture
def setup(tmp_path, tokenizer, monkeypatch):
    """A base on disk with its span file, the record of one of its fine-tunes (which holds its
    directions and pins its files), and a sweep folder holding one direction outside the span."""
    script = load_script("padded_edits")
    base, _ = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "hub", base, BASE_REV)
    tokenizer.save_pretrained(folder)
    directions = torch.randn(2, base.n_layers, base.d_model, generator=torch.Generator().manual_seed(4))
    span, files = record.make_span(folder, directions, n_controls=2), record.snapshot_info(folder)["files"]
    record.write_span(tmp_path / "span", BASE, span, {"model": BASE, "base_revision": BASE_REV, "base_files": files})
    readers = tmp_path / "readers" / "set.jsonl"
    readers.parent.mkdir()
    sweep.write_rows([{"repo": REPO, "base": BASE, "label": "benign_finetune"}], readers)
    record.write(readers, REPO, {"reference.directions": directions[-1].half()},
                 {"repo": REPO, "base": BASE, "base_revision": BASE_REV, "base_files": files})
    v = unlisted.split(rand_dir(base, 11), span["basis"])[1].float()
    found = tmp_path / "sweeps" / "found"
    (found / SLUG).mkdir(parents=True)
    EditSpec(direction=[v.tolist()], model=BASE, method="found", side="readers").save(found / SLUG / "subspace_found.json")
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: folder)
    monkeypatch.setattr(script.data, "contrast_sets", lambda source, n_train, n_val: (PROMPTS[:2], PROMPTS[2:], PROMPTS[:2], PROMPTS[2:]))
    monkeypatch.setattr(script.direction, "filter_by_baseline", lambda lm, harmful, harmless, bs: (harmful, harmless))
    monkeypatch.setattr(script.completions, "prompts", lambda n: PROMPTS[:n])
    monkeypatch.setattr(script, "Judge", Grader)
    return {"script": script, "folder": folder, "v": v, "found": found, "span": tmp_path / "span", "readers": readers,
            "scratch": tmp_path / "scratch", "out": tmp_path / "padded", "rows": tmp_path / "padded" / "found" / SLUG / "padded.jsonl"}


def run(s: dict, monkeypatch, *extra: str) -> dict[str, dict]:
    monkeypatch.setattr("sys.argv", ["padded_edits.py", "--model", BASE, "--direction", str(s["found"]), "--readers", str(s["readers"]),
                                     "--span", str(s["span"]), "--scratch", str(s["scratch"]), "--out", str(s["out"]),
                                     "--shared", "2", "--decoys", "3", "--kept", "2,6", "--gen-tokens", "4", "--batch-size", "2",
                                     "--device", "cpu", *extra])
    s["script"].main()
    rows = sweep.read_rows(s["rows"])
    assert len({r["name"] for r in rows}) == len(rows)
    return {r["name"]: r for r in rows}


def test_each_way_of_hiding_the_edit_gets_a_row_with_what_it_does_and_what_reads_of_it(setup, monkeypatch):
    rows = run(setup, monkeypatch)
    assert list(rows) == ["plain", "shared_k2", "decoys_p3"]
    assert [(r["k"], r["decoys"]) for r in rows.values()] == [(1, 0), (2, 0), (1, 3)]
    for r in rows.values():
        assert {"bypass_rate", "kl_harmless", "phrase_refusal_rate", "reader_top1_energy", "reader_shared_direction_k6"} <= set(r)
        assert r["reader_span_read_drop"] < 1e-3  # the direction is outside the span, and so is everything added to it
    assert not setup["scratch"].exists() or not any(setup["scratch"].iterdir())


def test_the_plain_edit_behaves_as_the_sweeps_measure_it(setup, monkeypatch):
    plain = run(setup, monkeypatch)["plain"]
    lm = load(str(setup["folder"]), device="cpu")
    (want,) = sweep.run_sweep(lm, setup["v"][None], PROMPTS[:2], PROMPTS[2:], (1.0,), (1,), {"all": None}, "readers", batch_size=2)
    assert plain["bypass_rate"] == want["bypass_rate"]
    assert plain["kl_harmless"] == pytest.approx(want["kl_harmless"], rel=1e-4) and plain["kl_harmless"] > 0


def test_decoys_hide_the_shared_direction_from_a_record_that_keeps_fewer_factors_and_not_from_one_that_keeps_more(setup, monkeypatch):
    rows = run(setup, monkeypatch)
    plain, hidden = rows["plain"], rows["decoys_p3"]
    assert plain["reader_shared_direction"] == pytest.approx(1.0, abs=1e-3) and plain["reader_top1_energy"] > 0.99
    assert hidden["reader_shared_direction"] < 0.6 and hidden["reader_top1_energy"] < 0.5  # two factors kept, three decoys
    assert hidden["reader_shared_subspace_k6"] == pytest.approx(1.0, abs=1e-2)  # six kept: the edit's direction is among them in every matrix
    two = rows["shared_k2"]  # two directions, the same in every matrix
    assert two["reader_shared_subspace"] == pytest.approx(1.0, abs=1e-2) and two["reader_top1_energy"] < 0.99 < two["reader_top2_energy"]


def test_whole_answers_are_judged_for_every_variant_and_the_base_and_a_second_run_adds_nothing(setup, monkeypatch):
    run(setup, monkeypatch, "--answers", "3", "--max-new-tokens", "6", "--judge", "some/grader")
    judged = setup["out"] / "judged.jsonl"
    rated = {r["repo"]: r for r in sweep.read_rows(judged)}
    assert list(rated) == [BASE, f"{BASE}#found+plain", f"{BASE}#found+shared_k2", f"{BASE}#found+decoys_p3"]
    assert all(r["n"] == 3 and "strongreject_score" in r for r in rated.values())
    assert all(completions.path(judged, name).exists() for name in rated)
    monkeypatch.setattr(setup["script"], "blind_readers_on_disk", lambda *a, **k: pytest.fail("every variant has its row and its answers"))
    assert len(run(setup, monkeypatch, "--answers", "3", "--max-new-tokens", "6", "--judge", "some/grader")) == 3
