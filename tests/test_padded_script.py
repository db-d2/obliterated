"""scripts/padded_edits.py: an edit spread over more directions or hidden among decoys,
scored from the matrices on disk and run from the same matrices."""

import json
from pathlib import Path

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


def test_two_decoy_strengths_give_a_variant_each(setup, monkeypatch):
    rows = run(setup, monkeypatch, "--decoy-gain", "1.1,2.0")
    assert list(rows) == ["plain", "shared_k2", "decoys_p3_g1.1", "decoys_p3_g2.0"]
    weak, strong = rows["decoys_p3_g1.1"], rows["decoys_p3_g2.0"]
    assert (weak["decoy_gain"], strong["decoy_gain"], rows["plain"]["decoy_gain"]) == (1.1, 2.0, None)
    assert rows["plain"]["reader_fro_norm"] < weak["reader_fro_norm"] < strong["reader_fro_norm"]  # stronger decoys, a larger change


# ---- decoys along the directions the stream uses least


def test_decoys_from_the_directions_the_stream_uses_least_get_rows_of_their_own(setup, monkeypatch):
    rows = run(setup, monkeypatch, "--least-used", "0,8", "--stream-tokens", "3")
    assert list(rows) == ["plain", "shared_k2", "decoys_p3", "decoys_p3_n8"]
    anywhere, least = rows["decoys_p3"], rows["decoys_p3_n8"]
    assert (rows["plain"]["decoys_among"], anywhere["decoys_among"], least["decoys_among"]) == (None, None, 8)
    assert 0 <= least["stream_share"] < 0.5 * 8 / 32  # eight of 32 dimensions, with far less of the stream than eight hold on average
    assert least["stream_share_heldout"] >= 0 and anywhere["stream_share"] is None
    assert least["decoys"] == 3 and least["reader_span_read_drop"] < 1e-3
    made = json.loads((setup["rows"].parent / "stream.json").read_text())
    assert made["tokens"] > 0 and made["heldout_tokens"] > 0 and made["answer_tokens"] == 3 and made["d_model"] == 32
    assert set(made["least_used"]) == {"8"} and made["least_used"]["8"]["share"] == least["stream_share"]
    assert len(made["least_used"]["8"]["by_norm"]) == len(made["direction"]["by_norm"]) == 2 * 4 + 1  # two norms a layer and the last
    assert not setup["scratch"].exists() or not any(setup["scratch"].iterdir())


def test_the_least_used_directions_are_clear_of_the_edit_and_of_the_span(setup, monkeypatch):
    real, given = setup["script"].blind_readers_on_disk, {}

    def spy(base_dir, out_dir, *args, **kwargs):
        given[Path(out_dir).name] = kwargs.get("among")
        return real(base_dir, out_dir, *args, **kwargs)

    monkeypatch.setattr(setup["script"], "blind_readers_on_disk", spy)
    run(setup, monkeypatch, "--least-used", "0,8", "--stream-tokens", "3")
    assert given[f"{SLUG}_plain"] is None and given[f"{SLUG}_decoys_p3"] is None
    span, v = record.load_span(setup["span"], BASE)[0]["basis"].float(), setup["v"] / setup["v"].norm()
    assert len(given[f"{SLUG}_decoys_p3_n8"]) == 2 * 4 + 1
    for rows in given[f"{SLUG}_decoys_p3_n8"].values():
        assert rows.shape == (8, 32)
        assert float((rows @ v).abs().max()) < 1e-5 and float((rows @ span.T).abs().max()) < 1e-5


def test_the_stream_is_measured_on_the_training_prompts_with_the_bases_own_answers_and_checked_on_the_held_out_ones(setup, monkeypatch):
    script = setup["script"]
    monkeypatch.setattr(script.data, "contrast_sets", lambda source, n_train, n_val: (PROMPTS[:1], PROMPTS[1:2], PROMPTS[2:3], PROMPTS[3:]))
    real_generate, real_moments, answered, measured = script.generate, script.stream.moments, [], []

    def generate(lm, prompts, tokens, bs):
        out = real_generate(lm, prompts, tokens, bs)
        answered.append((list(prompts), tokens, out))
        return out

    def moments(lm, texts, bs):
        measured.append(list(texts))
        return real_moments(lm, texts, bs)

    monkeypatch.setattr(script, "generate", generate)
    monkeypatch.setattr(script.stream, "moments", moments)
    run(setup, monkeypatch, "--least-used", "8", "--stream-tokens", "3")
    lm = load(str(setup["folder"]), device="cpu")
    for texts, prompts in zip(measured, (PROMPTS[:2], PROMPTS[2:])):  # first what the directions are chosen on, then what they are checked on
        (answers,) = [out for asked, tokens, out in answered if asked == prompts and tokens == 3]
        assert texts == [lm.format(p) + a for p, a in zip(prompts, answers)]
    assert len(measured) == 2


def test_fewer_least_used_directions_than_decoys_are_refused_before_a_model_is_loaded(setup, monkeypatch):
    monkeypatch.setattr(setup["script"], "load", lambda *a, **k: pytest.fail("nothing is to be made"))
    with pytest.raises(SystemExit, match="least-used"):
        run(setup, monkeypatch, "--least-used", "2")


def test_a_second_run_with_another_number_of_directions_keeps_what_the_first_wrote_of_the_stream(setup, monkeypatch):
    run(setup, monkeypatch, "--least-used", "8", "--stream-tokens", "3")
    rows = run(setup, monkeypatch, "--least-used", "8,12", "--stream-tokens", "3")
    assert list(rows) == ["plain", "shared_k2", "decoys_p3_n8", "decoys_p3_n12"]
    made = json.loads((setup["rows"].parent / "stream.json").read_text())
    assert set(made["least_used"]) == {"8", "12"}
    assert made["least_used"]["8"]["share"] <= made["least_used"]["12"]["share"]  # four more directions, each used no less


def test_a_gain_floor_gives_every_variant_a_row_of_its_own_made_with_it(setup, monkeypatch):
    real, floors = setup["script"].blind_readers_on_disk, []

    def spy(*args, **kwargs):
        floors.append(kwargs.get("gain_floor"))
        return real(*args, **kwargs)

    monkeypatch.setattr(setup["script"], "blind_readers_on_disk", spy)
    rows = run(setup, monkeypatch, "--gain-floor", "0.1")
    assert list(rows) == ["plain_f0.1", "shared_k2_f0.1", "decoys_p3_f0.1"]
    assert all(r["gain_floor"] == 0.1 for r in rows.values()) and floors == [0.1] * 3
    rows = run(setup, monkeypatch)  # and without one, the edits as they were, beside them
    assert list(rows)[3:] == ["plain", "shared_k2", "decoys_p3"] and rows["plain"]["gain_floor"] is None and floors[3:] == [0.0] * 3
