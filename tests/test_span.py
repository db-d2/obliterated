"""What the reading matrices do on the base's refusal span, kept so that the weights can go."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import detect, record, sweep
from obliterated.edits import blind_readers

from .conftest import load_script
from .test_detect import HARMFUL, HARMLESS
from .test_edits import READABLE
from .test_readers import BASE, BASE_REV, REPO, REV, pair, snapshot

Q1 = "model.layers.1.self_attn.q_proj.weight"


def positions(lm, n_pos: int = 3) -> torch.Tensor:
    """A direction per template position and layer, (positions, layers, d); layer 0 has none,
    as in a real model, where its residual is the same token's embedding for every prompt."""
    d = torch.randn(n_pos, lm.n_layers, lm.d_model, generator=torch.Generator().manual_seed(21))
    d[:, 0] = 0
    return d


def units(directions: torch.Tensor) -> torch.Tensor:
    flat = directions.reshape(-1, directions.shape[-1])
    return torch.nn.functional.normalize(flat[flat.norm(dim=1) > 0], dim=1)


# ---- the span


def test_the_span_holds_every_direction_and_its_controls_hold_none(tokenizer):
    base, _ = pair("llama_untied", tokenizer)
    dirs = positions(base)
    basis, n_ref = detect.refusal_span(dirs, n_controls=4)
    assert n_ref == 3 * (base.n_layers - 1) and basis.shape == (n_ref + 4, base.d_model)
    torch.testing.assert_close(basis @ basis.T, torch.eye(len(basis)), atol=1e-5, rtol=0)
    torch.testing.assert_close((units(dirs) @ basis[:n_ref].T).norm(dim=1), torch.ones(n_ref), atol=1e-5, rtol=0)
    assert float((units(dirs) @ basis[n_ref:].T).abs().max()) < 1e-5


def test_a_direction_that_repeats_does_not_widen_the_span(tokenizer):
    base, _ = pair("llama_untied", tokenizer)
    dirs = positions(base)
    dirs[1] = 2.0 * dirs[0]  # the same directions at another position, at another length
    _, n_ref = detect.refusal_span(dirs, n_controls=2)
    assert n_ref == 2 * (base.n_layers - 1)


def test_the_controls_are_the_same_on_every_run(tokenizer):
    base, _ = pair("llama_untied", tokenizer)
    dirs = positions(base)
    first, _ = detect.refusal_span(dirs, n_controls=4)
    again, _ = detect.refusal_span(dirs, n_controls=4)
    other, n_ref = detect.refusal_span(dirs, n_controls=4, seed=1)
    assert torch.equal(first, again)
    assert torch.equal(first[:n_ref], other[:n_ref]) and not torch.allclose(first[n_ref:], other[n_ref:])


# ---- what the scan keeps on it


@pytest.mark.parametrize("kind", READABLE)
def test_the_gram_on_the_span_gives_back_how_strongly_any_direction_in_it_is_read(kind, tokenizer, tmp_path):
    """The scan measures the read of the directions it is given. The gram has to give the same
    for a direction it was not given, as long as that direction is inside the span: with an
    edit, with a changed gain, and behind Gemma's norms."""
    base, cand = pair(kind, tokenizer)
    basis, _ = detect.refusal_span(positions(base), n_controls=4)
    blind_readers(cand, basis[1] + 0.5 * basis[4], alpha=0.5, layers=[1, 3])
    cand.read_modules(2)[0][0].weight.data[::2] *= 1.3  # a gain alone
    cand.read_modules(3)[0][0].weight.data[1::2] *= 0.8  # and a gain in front of matrices that changed as well
    mix = torch.nn.functional.normalize(torch.randn(6, len(basis), generator=torch.Generator().manual_seed(22)), dim=1)
    found = list(detect.scan_readers(snapshot(tmp_path / "base", base, BASE_REV), snapshot(tmp_path / "cand", cand),
                                     mix @ basis, span=basis))
    assert len(found) == 5 * base.n_layers + 1
    c = mix.double()
    for f in found:
        before = torch.einsum("im,mn,in->i", c, f["span_base"], c).sqrt()
        after = torch.einsum("im,mn,in->i", c, f["span_base"] + f["span_change"], c).clamp_min(0).sqrt()
        torch.testing.assert_close(before, f["read_base"], rtol=1e-4, atol=0, msg=f["name"])
        torch.testing.assert_close(after, f["read_cand"], rtol=1e-4, atol=1e-6, msg=f["name"])
    touched = {f["name"] for f in found if f["changed"] or f["gain_changed"]}
    assert 0 < len(touched) < len(found)
    assert all(bool(f["span_change"].any()) == (f["name"] in touched) for f in found)  # exactly zero where nothing moved


def test_a_change_far_smaller_than_the_matrix_is_not_lost_in_the_difference(tokenizer, tmp_path):
    """A light fine-tune moves a matrix in its sixth digit. Subtracting two grams would leave
    rounding noise; the change has to be accumulated as a change."""
    base, cand = pair("llama_untied", tokenizer)
    basis, _ = detect.refusal_span(positions(base), n_controls=4)
    lin, gain = cand.layers[1].self_attn.q_proj, cand.layers[1].input_layernorm.weight.data
    before = lin.weight.data.clone()
    lin.weight.data += 1e-6 * torch.randn(lin.weight.shape, generator=torch.Generator().manual_seed(23))
    by = {f["name"]: f for f in detect.scan_readers(snapshot(tmp_path / "base", base, BASE_REV), snapshot(tmp_path / "cand", cand),
                                                    basis[:1], span=basis)}
    e_base, e_cand = (before * gain).double() @ basis.T.double(), (lin.weight.data * gain).double() @ basis.T.double()
    want = e_cand.T @ e_cand - e_base.T @ e_base
    assert float(want.abs().max()) < 1e-4 * float((e_base.T @ e_base).abs().max())  # the case the test is about
    torch.testing.assert_close(by[Q1]["span_change"], want, rtol=2e-3, atol=1e-3 * float(want.abs().max()))


# ---- the scores it gives


def spanned_pair(tokenizer, tmp_path, **edit):
    """A base with its span on file, and a checkpoint that stops reading the direction of the
    first position at layer 2: not one of the last position's directions."""
    base, cand = pair("llama_untied", tokenizer)
    dirs = positions(base)
    base_dir = snapshot(tmp_path / "hub" / "base", base, BASE_REV)
    span = record.make_span(base_dir, dirs, n_controls=4)
    blind_readers(cand, dirs[0, 2], **edit)
    return base, base_dir, snapshot(tmp_path / "hub" / "cand", cand), dirs, span


def test_an_edit_taken_at_another_position_shows_on_the_span_and_barely_at_the_last(tokenizer, tmp_path):
    _, base_dir, cand_dir, dirs, span = spanned_pair(tokenizer, tmp_path, alpha=0.5)
    fields = record.read_readers(base_dir, cand_dir, dirs[-1], span=span)["fields"]
    assert set(record.SPAN_SCORES) <= set(fields)
    assert fields["reader_span_read_drop"] == pytest.approx(0.5, abs=1e-3)
    assert fields["reader_span_read_drop_med"] == pytest.approx(0.5, abs=1e-3)  # every layer was edited
    assert fields["reader_span_head_read_drop"] == pytest.approx(0.5, abs=1e-3)
    assert fields["reader_span_read_drop_by_position"][0] == pytest.approx(0.5, abs=1e-3)
    assert fields["reader_read_drop"] < 0.3  # the score that looks at the last position only
    assert fields["reader_span_last_read_drop"] == pytest.approx(fields["reader_read_drop"], abs=1e-4)  # two routes, one number
    assert fields["reader_span_read_drop_by_position"][-1] == fields["reader_span_last_read_drop"]
    assert fields["reader_span_control_drop"] < 1e-3  # directions outside the span are read as before
    assert fields["reader_span_alignment"] == pytest.approx(1.0, abs=1e-3)
    assert fields["reader_span_alignment_by_position"][0] == pytest.approx(1.0, abs=1e-3)
    assert fields["reader_span_alignment_by_position"][-1] == pytest.approx(fields["reader_refusal_alignment"], abs=1e-4)
    assert fields["reader_span_alignment_by_position"][-1] < 0.9


def test_the_median_and_the_head_are_kept_apart_from_the_largest(tokenizer, tmp_path):
    _, base_dir, cand_dir, dirs, span = spanned_pair(tokenizer, tmp_path, alpha=0.5, layers=[1])
    read = record.read_readers(base_dir, cand_dir, dirs[-1], span=span)
    fields = read["fields"]
    assert fields["reader_span_read_drop"] == pytest.approx(0.5, abs=1e-3)
    assert fields["reader_span_read_drop_med"] == pytest.approx(0.0, abs=1e-6)  # one layer of four
    assert fields["reader_span_head_read_drop"] == pytest.approx(0.0, abs=1e-6)  # the head goes with the last layer
    kept = sorted(k.removesuffix(".span") for k in read["kept"] if k.endswith(".span"))
    assert kept == sorted(n for n in read["matrices"] if ".layers.1." in n)  # only what changed keeps a gram


def test_a_direction_inside_the_span_that_was_never_listed_can_be_scored_afterwards(tokenizer, tmp_path):
    """The point of keeping grams and not a list of reads: an edit along a mix of two listed
    directions, read back from the record and the span file alone."""
    base, cand = pair("llama_untied", tokenizer)
    dirs = positions(base)
    base_dir = snapshot(tmp_path / "base", base, BASE_REV)
    span = record.make_span(base_dir, dirs, n_controls=4)
    mixed = torch.nn.functional.normalize(units(dirs)[0] + units(dirs)[5], dim=0)
    blind_readers(cand, mixed, alpha=0.75, layers=[1])
    kept = record.read_readers(base_dir, snapshot(tmp_path / "cand", cand), dirs[-1], span=span)["kept"]
    ratio = record.span_read(kept, span, Q1, mixed[None])
    assert float(ratio[0]) == pytest.approx(0.25, abs=1e-3)
    assert float(record.span_read(kept, span, "model.layers.2.self_attn.q_proj.weight", mixed[None])[0]) == 1.0


def test_a_head_with_fewer_tokens_than_the_base_is_scored_over_the_tokens_it_has(tokenizer, tmp_path):
    """A fine-tune that trimmed its vocabulary: the head's change is over the tokens both
    have, so the base's side has to be over those too, not the whole head of the span file."""
    base, cand = pair("llama_untied", tokenizer)
    dirs = positions(base)
    base_dir = snapshot(tmp_path / "base", base, BASE_REV)
    span = record.make_span(base_dir, dirs, n_controls=4)
    blind_readers(cand, dirs[0, 2], alpha=0.5)
    cand.model.resize_token_embeddings(cand.model.config.vocab_size - 5)
    read = record.read_readers(base_dir, snapshot(tmp_path / "cand", cand), dirs[-1], span=span)
    assert [k for k in read["kept"] if k.endswith(".span_base")] == [f"{detect.HEAD}.span_base"]
    assert read["fields"]["reader_span_head_read_drop"] == pytest.approx(0.5, abs=1e-3)
    assert float(record.span_read(read["kept"], span, detect.HEAD, dirs[0, 2][None])[0]) == pytest.approx(0.5, abs=1e-3)


def test_a_base_whose_readers_are_not_the_ones_the_span_was_made_from_is_refused(tokenizer, tmp_path):
    _, base_dir, cand_dir, dirs, span = spanned_pair(tokenizer, tmp_path)
    span[f"gram.{Q1}"] = span[f"gram.{Q1}"] * 1.01
    with pytest.raises(ValueError, match=Q1):
        record.read_readers(base_dir, cand_dir, dirs[-1], span=span)


# ---- the pass with a span: what it writes, what it checks


@pytest.fixture
def spanned(tmp_path, tokenizer, monkeypatch):
    """As ``scored`` in test_readers.py, with the base's span on file: the checkpoint's edit
    is taken at the first position, the main pass kept the last position's directions."""
    script = load_script("reader_weights")
    _, base_dir, cand_dir, dirs, span = spanned_pair(tokenizer, tmp_path, alpha=0.5)
    folders = {BASE: base_dir, REPO: cand_dir}
    monkeypatch.setattr(script, "locate", lambda repo, revision, files: folders[repo])
    main_out, out, spans = tmp_path / "main" / "scores.jsonl", tmp_path / "readers" / "scores.jsonl", tmp_path / "span"
    main_row = {"repo": REPO, "base": BASE, "label": "abliterated", "family": "llama", "params_b": 0.001}
    record.write(main_out, REPO, {"reference.directions": dirs[-1].half()}, {
        "repo": REPO, "base": BASE, "revision": REV, "files": record.snapshot_info(cand_dir)["files"],
        "base_revision": BASE_REV, "base_files": record.snapshot_info(base_dir)["files"]})
    record.write_span(spans, BASE, span, {"model": BASE, "base_revision": BASE_REV, "base_files": record.snapshot_info(base_dir)["files"]})
    return {"script": script, "out": out, "main_out": main_out, "main_row": main_row, "spans": spans, "folders": folders, "dirs": dirs}


def score(s: dict) -> dict:
    row = s["script"].score(s["main_row"], s["main_out"], s["out"], s["spans"])
    sweep.write_rows([row], s["out"])
    return row


def problems(s: dict, row: dict) -> list[str]:
    return record.validate_readers(row, s["out"], s["main_out"])


def flagged(found: list[str], word: str) -> bool:
    return any(word in p for p in found)


def test_a_pass_with_a_span_leaves_a_record_that_validates(spanned):
    row = score(spanned)
    assert problems(spanned, row) == []
    assert row["reader_span_read_drop"] == pytest.approx(0.5, abs=1e-3) and row["reader_read_drop"] < 0.25
    made = json.loads(record.provenance_path(spanned["out"], REPO).read_text())
    assert made["span"]["path"] == str(record.span_path(spanned["spans"], BASE).resolve()) and len(made["span"]["sha256"]) == 64


def test_a_pass_without_a_span_still_validates_and_has_no_span_scores(spanned):
    row = spanned["script"].score(spanned["main_row"], spanned["main_out"], spanned["out"])
    sweep.write_rows([row], spanned["out"])
    assert problems(spanned, row) == [] and not any(k.startswith("reader_span_") for k in row)


def test_a_gram_the_record_lost_is_a_problem(spanned):
    row = score(spanned)
    path = record.tensors_path(spanned["out"], REPO)
    kept = load_file(path)
    del kept[f"{Q1}.span"]
    save_file(kept, path)
    assert flagged(problems(spanned, row), Q1)  # no score moves: every other matrix carries the same edit


def test_a_span_score_the_record_does_not_reproduce_is_a_problem(spanned):
    row = score(spanned)
    assert flagged(problems(spanned, {**row, "reader_span_read_drop": 0.9}), "reader_span_read_drop")
    by = [*row["reader_span_read_drop_by_position"]]
    by[1] += 0.1
    assert flagged(problems(spanned, {**row, "reader_span_read_drop_by_position": by}), "reader_span_read_drop_by_position")


def test_a_span_file_changed_since_the_pass_is_a_problem(spanned):
    row = score(spanned)
    path = record.span_path(spanned["spans"], BASE)
    span = load_file(path)
    span["basis"] = span["basis"].flip(0)
    save_file(span, path)
    assert flagged(problems(spanned, row), "not the file the record was made with")


def test_a_span_file_that_is_gone_is_a_problem(spanned):
    row = score(spanned)
    record.span_path(spanned["spans"], BASE).unlink()
    assert flagged(problems(spanned, row), "not on disk")


def test_a_last_position_that_reads_differently_the_two_ways_is_a_problem(spanned):
    """The read drop at the last position is measured directly and again through the Gram
    matrices. A record in which the two disagree is wrong somewhere, even if each part of it
    reproduces its own score."""
    row = score(spanned)
    path = record.tensors_path(spanned["out"], REPO)
    kept = load_file(path)
    kept[f"{Q1}.read"][1] = 0.0  # as if this matrix no longer read the last position's directions at all
    save_file(kept, path)
    found = problems(spanned, {**row, **record.reader_scores(kept)})
    assert found and all("last position" in p for p in found)


@pytest.mark.parametrize("what", ["revision", "files"])
def test_a_span_made_from_other_base_weights_is_refused(spanned, what):
    path = record.span_path(spanned["spans"], BASE).with_suffix(".json")
    made = json.loads(path.read_text())
    if what == "revision":
        made["base_revision"] = REV
    else:  # the same revision, another file under the same name
        made["base_files"] = [{**f, "bytes": f["bytes"] + 1} for f in made["base_files"]]
    path.write_text(json.dumps(made))
    with pytest.raises(ValueError, match="other weights"):
        score(spanned)


def test_a_span_whose_last_position_is_not_what_the_main_pass_kept_is_refused(spanned):
    record.write(spanned["main_out"], REPO, {"reference.directions": spanned["dirs"][0].half()},
                 json.loads(record.provenance_path(spanned["main_out"], REPO).read_text()))
    with pytest.raises(ValueError, match="last position"):
        score(spanned)


def test_a_base_with_no_span_on_file_is_refused(spanned):
    record.span_path(spanned["spans"], BASE).unlink()
    with pytest.raises(FileNotFoundError, match="span"):
        score(spanned)


def test_the_pass_deletes_a_checkpoint_scored_with_a_span_once_it_validates(spanned, monkeypatch):
    script, deleted = spanned["script"], []
    sweep.write_rows([spanned["main_row"]], spanned["main_out"])
    monkeypatch.setattr(script, "cached", lambda repo, revision, files: False)
    monkeypatch.setattr(script, "evict", lambda repo, keep: deleted.append(repo) or 1)
    monkeypatch.setattr(script, "bases", lambda registry: {BASE})
    monkeypatch.setattr("sys.argv", ["reader_weights.py", "--rows", str(spanned["main_out"]), "--out", str(spanned["out"]),
                                     "--span", str(spanned["spans"]), "--evict"])
    script.main()
    (row,) = sweep.read_rows(spanned["out"])
    assert deleted == [REPO] and row["reader_span_read_drop"] == pytest.approx(0.5, abs=1e-3)


# ---- making the span of a base


def test_the_span_of_a_base_is_made_by_the_recipe_of_the_main_pass(tmp_path, tokenizer, monkeypatch):
    """The directions at the last position are the ones the main pass kept; the script checks
    that against a record before it writes anything."""
    script = load_script("refusal_span")
    base, _ = pair("llama_untied", tokenizer)
    base._eoi_len = 2  # as if the template's suffix were two tokens: two positions to take directions at
    base_dir = snapshot(tmp_path / "hub" / "base", base, BASE_REV)
    main_out = tmp_path / "main" / "scores.jsonl"
    reference = detect.reference_directions(base, HARMFUL, HARMLESS, batch_size=2)
    record.write(main_out, REPO, {"reference.directions": reference.half()}, {
        "repo": REPO, "base": BASE, "revision": REV, "base_revision": BASE_REV, "base_files": record.snapshot_info(base_dir)["files"]})
    sweep.write_rows([{"repo": REPO, "base": BASE, "label": "benign_finetune"}], main_out)
    monkeypatch.setattr(script, "load", lambda path, device=None: base)
    monkeypatch.setattr(script, "unload", lambda lm: None)
    monkeypatch.setattr(script, "locate", lambda repo, revision, files: base_dir)
    monkeypatch.setattr(script.data, "contrast_sets", lambda source, n_train, n_val: (HARMFUL, HARMLESS, HARMFUL, HARMLESS))
    monkeypatch.setattr(script.direction, "filter_by_baseline", lambda lm, h, s, bs: (h, s))
    monkeypatch.setattr("sys.argv", ["refusal_span.py", "--rows", str(main_out), "--out", str(tmp_path / "span"), "--controls", "4",
                                     "--batch-size", "2"])
    script.main()
    span, meta = record.load_span(tmp_path / "span", BASE)
    assert span["directions"].shape == (2, base.n_layers, base.d_model)
    torch.testing.assert_close(torch.nn.functional.normalize(span["directions"][-1], dim=1), reference, atol=1e-5, rtol=0)
    assert len(span["basis"]) == int(span["n_ref"]) + 4
    assert sum(k.startswith("gram.") for k in span) == 5 * base.n_layers + 1
    assert meta["base_revision"] == BASE_REV and meta["base_files"] == record.snapshot_info(base_dir)["files"]
    assert meta["positions"] == [-2, -1] and meta["versions"]["source_sha256"]

    record.write(main_out, REPO, {"reference.directions": reference.flip(1).half()},
                 json.loads(record.provenance_path(main_out, REPO).read_text()))
    monkeypatch.setattr("sys.argv", ["refusal_span.py", "--rows", str(main_out), "--out", str(tmp_path / "other"), "--controls", "4",
                                     "--batch-size", "2"])
    with pytest.raises(SystemExit, match="last position"):
        script.main()
    assert not (tmp_path / "other").exists() or not any((tmp_path / "other").iterdir())
