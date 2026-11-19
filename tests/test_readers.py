"""The weights-only pass over the matrices that read the residual stream."""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import detect, record, registry, sweep
from obliterated.edits import blind_readers, orthogonalize

from .conftest import load_script, make_lm
from .test_edits import READABLE, uneven_gains

REPO, BASE = "someone/tune", "lab/base"
REV, BASE_REV = "0123456789abcdef0123456789abcdef01234567", "76543210fedcba9876543210fedcba9876543210"


def snapshot(root: Path, lm, revision: str = REV) -> Path:
    """A model's weights the way the Hub cache holds them: a folder named by the revision, a
    tied output head not stored."""
    folder = root / "snapshots" / revision
    folder.mkdir(parents=True)
    state = {k: v.clone() for k, v in lm.model.state_dict().items()}
    if lm.model.get_output_embeddings().weight is lm.model.get_input_embeddings().weight:
        del state["lm_head.weight"]
    save_file(state, folder / "model.safetensors")
    lm.model.config.to_json_file(folder / "config.json")
    return folder


def pair(kind: str, tokenizer):
    """A base and an identical copy to change, with norms that scale each coordinate differently."""
    base, cand = make_lm(kind, tokenizer), make_lm(kind, tokenizer)
    uneven_gains(base)
    uneven_gains(cand)
    return base, cand


def direction(lm) -> torch.Tensor:
    r = torch.randn(lm.d_model, generator=torch.Generator().manual_seed(3))
    return r / r.norm()


def scan(tmp_path, base, cand, directions) -> list[dict]:
    return list(detect.scan_readers(snapshot(tmp_path / "base", base, BASE_REV), snapshot(tmp_path / "cand", cand), directions))


# ---- the iteration that replaces a full decomposition


def test_the_iterative_top_factors_agree_with_the_exact_ones():
    g = torch.Generator().manual_seed(0)
    low = torch.randn(300, 3, generator=g, dtype=torch.float64) @ torch.randn(3, 500, generator=g, dtype=torch.float64)
    m = low + 0.05 * torch.randn(300, 500, generator=g, dtype=torch.float64)
    gram = m @ m.T
    values, vectors, info = detect.gram_top(gram, 8)
    exact_values, exact_vectors = torch.linalg.eigh(gram)
    torch.testing.assert_close(values, exact_values.flip(0)[:8], rtol=1e-8, atol=0)
    inside = (exact_vectors[:, -3:].T @ vectors[:, :3]).norm()  # the three dominant directions span the same space
    assert float(inside) == pytest.approx(math.sqrt(3), abs=1e-6)
    torch.testing.assert_close(vectors.T @ vectors, torch.eye(8, dtype=torch.float64), atol=1e-9, rtol=0)
    assert not info["exact"]


def test_a_flat_spectrum_still_gets_its_largest_values_right():
    m = torch.randn(300, 400, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    gram = m @ m.T
    values, _, _ = detect.gram_top(gram, 8)
    torch.testing.assert_close(values, torch.linalg.eigvalsh(gram).flip(0)[:8], rtol=1e-6, atol=0)


def test_a_matrix_no_larger_than_the_block_is_decomposed_exactly():
    m = torch.randn(5, 9, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    values, vectors, info = detect.gram_top(m @ m.T, 8)
    assert info["exact"] and values.shape == (5,) and vectors.shape == (5, 5)
    torch.testing.assert_close(values.sqrt(), torch.linalg.svdvals(m), rtol=1e-9, atol=0)


# ---- what the pass sees in a pair of snapshots


@pytest.mark.parametrize("kind", READABLE)
def test_a_reader_side_edit_shows_in_every_reading_matrix(kind, tokenizer, tmp_path):
    """The gains are uneven, and Gemma stores them differently and puts a norm of its own in
    front of the MLP: a pass that got either wrong would still find the readers reading."""
    base, cand = pair(kind, tokenizer)
    r = direction(base)
    blind_readers(cand, r)
    directions = r.repeat(base.n_layers + 1, 1)
    found = scan(tmp_path, base, cand, directions)
    assert [f["name"] for f in found][-1] == detect.HEAD and len(found) == 5 * base.n_layers + 1
    for f in found:
        assert f["changed"] and not f["gain_changed"], f["name"]
        assert f["summary"]["top1_energy"] == pytest.approx(1.0, abs=1e-6), f["name"]
        assert f["summary_eff"]["top1_energy"] == pytest.approx(1.0, abs=1e-6), f["name"]
        assert detect.refusal_alignment(f["eu"], directions) == pytest.approx(1.0, abs=1e-4), f["name"]
        assert detect.refusal_alignment(f["u"], directions) < 0.999, f["name"]  # in the stored matrix it is r / g
        assert float(f["read_cand"].max()) < 1e-4 * float(f["read_base"].min()), f["name"]


def test_a_projection_that_ignores_the_norm_aligns_with_the_direction_itself(tokenizer, tmp_path):
    """The textbook edit on the input side, W (I - r r^T): it does not make the reader ignore
    r, because the reader sees r scaled by the gain, but its change is still rank one and its
    input side is r exactly."""
    base, cand = pair("llama_untied", tokenizer)
    r = direction(base)
    for i in range(cand.n_layers):
        for _, lin in cand.read_modules(i):
            lin.weight.data -= torch.outer(lin.weight.data @ r, r)
    directions = r.repeat(base.n_layers + 1, 1)
    found = [f for f in scan(tmp_path, base, cand, directions) if f["name"] != detect.HEAD]
    for f in found:
        assert f["summary"]["top1_energy"] == pytest.approx(1.0, abs=1e-6), f["name"]
        assert detect.refusal_alignment(f["u"], directions) == pytest.approx(1.0, abs=1e-4), f["name"]
        assert detect.refusal_alignment(f["eu"], directions) < 0.999, f["name"]
        assert float(f["read_cand"][0]) > 0.05 * float(f["read_base"][0]), f["name"]


def test_rounding_noise_stays_flat_in_the_stored_matrix_however_uneven_the_gains(tokenizer, tmp_path):
    """A re-save in another precision changes every entry a little. Scaled by a norm whose
    gain is large on a few coordinates, that noise piles up on them and passes for a low-rank
    change; the energy shares are therefore taken from the stored matrix."""
    base, cand = make_lm("llama_untied", tokenizer), make_lm("llama_untied", tokenizer)
    r = torch.zeros(base.d_model)
    r[:2] = 1.0
    for lm in (base, cand):
        for i in range(lm.n_layers):
            for norm, _ in lm.read_modules(i):
                norm.weight.data[:2] = 200.0
    g = torch.Generator().manual_seed(8)
    for i in range(cand.n_layers):
        for _, lin in cand.read_modules(i):
            lin.weight.data += 1e-5 * torch.randn(lin.weight.shape, generator=g)
    directions = r.repeat(base.n_layers + 1, 1)
    found = [f for f in scan(tmp_path, base, cand, directions) if f["name"] != detect.HEAD]
    assert max(f["summary"]["top2_energy"] for f in found) < 0.6
    assert min(f["summary_eff"]["top2_energy"] for f in found) > 0.99


def test_a_partial_reader_side_edit_reads_that_much_less_of_the_direction(tokenizer, tmp_path):
    base, cand = pair("llama_untied", tokenizer)
    r = direction(base)
    blind_readers(cand, r, alpha=0.25, layers=[1, 2])
    by = {f["name"]: f for f in scan(tmp_path, base, cand, r.repeat(base.n_layers + 1, 1))}
    inside, outside = by["model.layers.1.mlp.up_proj.weight"], by["model.layers.0.mlp.up_proj.weight"]
    assert float(inside["read_cand"][0] / inside["read_base"][0]) == pytest.approx(0.75, abs=1e-5)
    assert not outside["changed"] and "u" not in outside
    torch.testing.assert_close(outside["read_cand"], outside["read_base"])
    assert not by[detect.HEAD]["changed"]  # the head is edited only with the last layer


def test_dense_noise_on_the_readers_neither_aligns_nor_stops_them_reading(tokenizer, tmp_path):
    base, cand = pair("llama_untied", tokenizer)
    g = torch.Generator().manual_seed(5)
    for i in range(cand.n_layers):
        for _, lin in cand.read_modules(i):
            lin.weight.data += 1e-3 * torch.randn(lin.weight.shape, generator=g)
    r = direction(base)
    directions = r.repeat(base.n_layers + 1, 1)
    found = [f for f in scan(tmp_path, base, cand, directions) if f["name"] != detect.HEAD]
    assert all(f["changed"] for f in found)
    assert max(f["summary"]["top1_energy"] for f in found) < 0.5
    assert max(detect.refusal_alignment(f["u"], directions) for f in found) < 0.95
    ratios = torch.stack([f["read_cand"] / f["read_base"] for f in found])
    assert float((ratios - 1).abs().max()) < 0.05


def test_a_writer_side_edit_changes_no_reading_matrix(tokenizer, tmp_path):
    base, cand = pair("llama_untied", tokenizer)
    orthogonalize(cand, direction(base), edit_embeddings=False)
    found = scan(tmp_path, base, cand, direction(base).repeat(base.n_layers + 1, 1))
    assert len(found) == 5 * base.n_layers + 1
    assert not any(f["changed"] for f in found) and not any("summary" in f for f in found)


def test_a_changed_norm_gain_changes_what_the_matrices_behind_it_read(tokenizer, tmp_path):
    base, cand = pair("llama_untied", tokenizer)
    cand.layers[1].input_layernorm.weight.data[::2] *= 1.5
    found = scan(tmp_path, base, cand, direction(base).repeat(base.n_layers + 1, 1))
    behind = {f"model.layers.1.self_attn.{n}_proj.weight" for n in "qkv"}
    assert {f["name"] for f in found if f["gain_changed"]} == behind
    assert not any(f["changed"] for f in found)  # no stored matrix differs
    for f in found:
        same = torch.allclose(f["read_cand"], f["read_base"], rtol=1e-6)
        assert same != (f["name"] in behind), f["name"]
        assert ("gain_cand" in f) == (f["name"] in behind)
        assert ("eu" in f) == (f["name"] in behind) and "u" not in f  # it changed what they do, not what is stored


def acting(folder: Path) -> dict[str, torch.Tensor]:
    """Every reading matrix of an untied Llama snapshot composed with its norm, in float64."""
    got = load_file(folder / "model.safetensors")
    return {name: got[name].double() * got[norm].double() for name, norm in detect.reader_layout(detect._tensor_index(folder))}


def test_the_gram_matrix_of_the_change_in_what_acts_on_the_stream_is_handed_over_when_asked_for(tokenizer, tmp_path):
    """Whether the stored matrix changed, the gain in front of it, or both."""
    base, cand = pair("llama_untied", tokenizer)
    blind_readers(cand, direction(base), layers=[0, 2])
    cand.layers[1].input_layernorm.weight.data[::2] *= 1.5
    cand.layers[2].input_layernorm.weight.data[::2] *= 1.5
    b, c = snapshot(tmp_path / "base", base, BASE_REV), snapshot(tmp_path / "cand", cand)
    directions = direction(base).repeat(base.n_layers + 1, 1)
    found = {f["name"]: f for f in detect.scan_readers(b, c, directions, grams=True)}
    before, after = acting(b), acting(c)
    kinds = set()
    for name, f in found.items():
        change = after[name] - before[name]
        if not f["changed"] and not f["gain_changed"]:
            assert "gram_eff" not in f and float(change.abs().max()) == 0, name
            continue
        kinds.add((f["changed"], f["gain_changed"]))
        assert f["gram_eff"].dtype == torch.float64
        torch.testing.assert_close(f["gram_eff"], change.T @ change, atol=1e-6 * float(change.norm() ** 2), rtol=0)
    assert kinds == {(True, False), (False, True), (True, True)}
    assert not any("gram_eff" in f for f in detect.scan_readers(b, c, directions))  # only when asked for


def test_no_factors_are_worked_out_when_none_are_asked_for(tokenizer, tmp_path, monkeypatch):
    base, cand = pair("llama_untied", tokenizer)
    blind_readers(cand, direction(base))
    b, c = snapshot(tmp_path / "base", base, BASE_REV), snapshot(tmp_path / "cand", cand)
    directions = direction(base).repeat(base.n_layers + 1, 1)
    with_factors = list(detect.scan_readers(b, c, directions, grams=True))

    def decomposed(*a, **k):
        raise AssertionError("a Gram matrix was decomposed")

    monkeypatch.setattr(detect, "gram_top", decomposed)
    found = list(detect.scan_readers(b, c, directions, k=0, grams=True))
    assert len(found) == len(with_factors)
    for f, w in zip(found, with_factors):
        assert f["changed"] and not {"s", "u", "summary", "es", "eu", "summary_eff"} & f.keys(), f["name"]
        assert torch.equal(f["gram_eff"], w["gram_eff"]) and torch.equal(f["read_cand"], w["read_cand"])


def test_the_head_is_compared_over_the_tokens_both_sides_have(tmp_path):
    """A base that ties its head to the embedding does not store one; a fine-tune that added
    tokens stores a longer one."""
    g = torch.Generator().manual_seed(4)
    embed = torch.randn(10, 6, generator=g)
    grown = torch.cat([embed, torch.randn(2, 6, generator=g)])
    r = torch.nn.functional.normalize(torch.randn(6, generator=g), dim=0)
    gain = torch.linspace(0.5, 2.0, 6)
    head = grown - torch.outer(grown @ (gain * r), r / gain)  # blind to r behind the final norm
    for name, state in (("base", {"model.embed_tokens.weight": embed}),
                        ("cand", {"model.embed_tokens.weight": grown, "lm_head.weight": head})):
        (tmp_path / name).mkdir()
        save_file({**state, "model.norm.weight": gain}, tmp_path / name / "model.safetensors")
        (tmp_path / name / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (found,) = detect.scan_readers(tmp_path / "base", tmp_path / "cand", r[None])
    assert found["name"] == detect.HEAD and found["changed"] and found["rows"] == 10
    assert detect.refusal_alignment(found["eu"], r[None]) == pytest.approx(1.0, abs=1e-4)
    assert float(found["read_cand"][0]) < 1e-5 * float(found["read_base"][0])


def test_a_reading_matrix_the_checkpoint_lacks_is_reported_not_skipped(tokenizer, tmp_path):
    base, cand = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "cand", cand)
    state = load_file(folder / "model.safetensors")
    del state["model.layers.2.self_attn.k_proj.weight"]
    save_file(state, folder / "model.safetensors")
    found = list(detect.scan_readers(snapshot(tmp_path / "base", base, BASE_REV), folder, direction(base)[None]))
    assert [f["name"] for f in found if f.get("missing")] == ["model.layers.2.self_attn.k_proj.weight"]


def test_spectra_aggregate_the_same_way_whichever_pass_made_them():
    g = torch.Generator().manual_seed(6)
    pairs = [(f"model.layers.{i}.mlp.down_proj.weight", torch.randn(6, 10, generator=g) * (i + 1)) for i in range(4)]
    fields = detect.weight_scores(pairs, [1, 2])
    again = detect.scores_from_spectra(fields["per_matrix"], [1, 2])
    assert {k: v for k, v in fields.items() if k != "per_matrix"} == {k: v for k, v in again.items() if k != "per_matrix"}


# ---- the scores a record gives back


def _kept() -> dict:
    """Three layer readers and a head, six coordinates, the direction along the first.
    Layer 0's query changed along the direction as the matrix acts on the stream (``eu``);
    layer 1's up projection changed along it in the stored matrix only (``u``); layer 2's did
    not change; the head changed along something else."""
    eye = torch.eye(6)
    return {
        "reference.directions": eye[:1].repeat(3, 1),
        "model.layers.0.self_attn.q_proj.weight.eu": eye[:, :2], "model.layers.0.self_attn.q_proj.weight.u": eye[:, 2:4],
        "model.layers.0.self_attn.q_proj.weight.read": torch.tensor([[2.0] * 3, [0.5] * 3]),
        "model.layers.1.mlp.up_proj.weight.eu": eye[:, 2:4], "model.layers.1.mlp.up_proj.weight.u": eye[:, :2],
        "model.layers.1.mlp.up_proj.weight.read": torch.tensor([[2.0] * 3, [1.9, 2.0, 2.2]]),
        "model.layers.2.mlp.up_proj.weight.read": torch.tensor([[1.0] * 3, [1.0] * 3]),
        "lm_head.weight.eu": eye[:, 4:], "lm_head.weight.u": eye[:, 4:], "lm_head.weight.read": torch.tensor([[4.0] * 3, [3.0] * 3]),
    }


def test_reader_scores_summarise_the_layers_and_report_the_head_apart():
    scores = record.reader_scores(_kept())
    assert scores["reader_refusal_alignment"] == pytest.approx(1.0)
    assert scores["reader_refusal_alignment_med"] == pytest.approx(0.5)  # of the two that changed
    assert scores["reader_refusal_alignment_plain"] == pytest.approx(1.0)
    assert scores["reader_head_alignment"] == pytest.approx(0.0) and scores["reader_head_alignment_plain"] == pytest.approx(0.0)
    assert scores["reader_refusal_alignment_any"] == pytest.approx(1.0)
    assert scores["reader_read_drop"] == pytest.approx(0.75)
    assert scores["reader_read_drop_med"] == pytest.approx(0.05)  # of all three, changed or not
    assert scores["reader_head_read_drop"] == pytest.approx(0.25)


def test_the_two_alignments_are_kept_apart():
    kept = _kept()
    del kept["model.layers.1.mlp.up_proj.weight.u"]  # nothing is left that changed along the direction as stored
    scores = record.reader_scores(kept)
    assert scores["reader_refusal_alignment"] == pytest.approx(1.0)
    assert scores["reader_refusal_alignment_plain"] == pytest.approx(0.0)


def test_a_checkpoint_whose_readers_did_not_change_scores_nothing():
    kept = {k: v for k, v in _kept().items() if k.endswith(".read") and "layers.2" in k or k.startswith("reference")}
    assert set(record.reader_scores(kept).values()) == {0.0}


# ---- the pass itself: what it writes, what it checks, what it deletes


@pytest.fixture
def scored(tmp_path, tokenizer, monkeypatch):
    """A checkpoint with a reader-side edit, already scored by the main pass (which kept its
    revision, its files and the base's directions), and the reader pass run over it."""
    script = load_script("reader_weights")
    base, cand = pair("llama_untied", tokenizer)
    r = direction(base)
    blind_readers(cand, r)
    folders = {BASE: snapshot(tmp_path / "hub" / "base", base, BASE_REV), REPO: snapshot(tmp_path / "hub" / "cand", cand)}
    monkeypatch.setattr(script, "locate", lambda repo, revision, files: folders[repo])
    main_out, out = tmp_path / "main" / "scores.jsonl", tmp_path / "readers" / "scores.jsonl"
    main_row = {"repo": REPO, "base": BASE, "label": "abliterated", "family": "llama", "params_b": 0.001}
    record.write(main_out, REPO, {"reference.directions": r.repeat(base.n_layers + 1, 1).half()}, {
        "repo": REPO, "base": BASE, "revision": REV, "files": record.snapshot_info(folders[REPO])["files"],
        "base_revision": BASE_REV, "base_files": record.snapshot_info(folders[BASE])["files"]})
    row = script.score(main_row, main_out, out)
    sweep.write_rows([row], out)
    return {"script": script, "row": row, "out": out, "main_out": main_out, "folders": folders, "main_row": main_row}


def _reader_problems(s: dict, row: dict | None = None) -> list[str]:
    return record.validate_readers(row or s["row"], s["out"], s["main_out"])


def _flagged(problems: list[str], word: str) -> bool:
    return any(word in p for p in problems)


def _edit_json(path: Path, change) -> None:
    body = json.loads(path.read_text())
    change(body)
    path.write_text(json.dumps(body))


def test_a_reader_pass_leaves_a_record_that_validates(scored):
    row = scored["row"]
    assert _reader_problems(scored) == []
    assert row["reader_refusal_alignment"] == pytest.approx(1.0, abs=1e-3)
    assert row["reader_read_drop"] == pytest.approx(1.0, abs=1e-3) and row["reader_head_read_drop"] == pytest.approx(1.0, abs=1e-3)
    assert row["reader_top1_energy_w"] == pytest.approx(1.0, abs=1e-5) and row["reader_eff_top1_energy_w"] == pytest.approx(1.0, abs=1e-5)
    assert row["reader_refusal_alignment_plain"] < 0.999
    assert row["reader_n_changed"] == 20 and row["revision"] == REV


def test_weights_other_than_the_ones_scored_are_a_problem(scored):
    _edit_json(record.provenance_path(scored["main_out"], REPO), lambda b: b["files"][-1].update(bytes=b["files"][-1]["bytes"] + 1))
    assert _flagged(_reader_problems(scored), "not the files")


def test_a_base_other_than_the_one_scored_against_is_a_problem(scored):
    _edit_json(record.provenance_path(scored["main_out"], REPO), lambda b: b.update(base_revision=REV))
    assert _flagged(_reader_problems(scored), "base_revision")


@pytest.mark.parametrize("kind", ["u", "eu"])
def test_a_reading_matrix_without_its_saved_directions_is_a_problem(scored, kind):
    path = record.tensors_path(scored["out"], REPO)
    kept = load_file(path)
    del kept[f"model.layers.1.self_attn.v_proj.weight.{kind}"]
    save_file(kept, path)
    assert _flagged(_reader_problems(scored), "model.layers.1.self_attn.v_proj.weight")


def test_a_reading_matrix_that_was_never_read_is_a_problem(scored):
    path = record.tensors_path(scored["out"], REPO)
    kept = load_file(path)
    del kept["model.layers.3.mlp.gate_proj.weight.read"]
    save_file(kept, path)
    assert _flagged(_reader_problems(scored), "model.layers.3.mlp.gate_proj.weight")


def test_a_checkpoint_missing_a_reading_matrix_is_a_problem(scored):
    _edit_json(record.provenance_path(scored["out"], REPO), lambda b: b.update(missing=["model.layers.0.mlp.up_proj.weight"]))
    assert _flagged(_reader_problems(scored), "missing")


def test_a_norm_whose_gain_was_not_kept_is_a_problem(scored):
    path = record.tensors_path(scored["out"], REPO)
    kept = load_file(path)
    del kept["gain.model.layers.2.input_layernorm.weight"]
    save_file(kept, path)
    assert _flagged(_reader_problems(scored), "model.layers.2.input_layernorm.weight")


def test_a_reader_score_the_record_does_not_reproduce_is_a_problem(scored):
    assert _flagged(_reader_problems(scored, {**scored["row"], "reader_read_drop": 0.5}), "reader_read_drop")


def test_saved_reader_singular_values_that_disagree_with_the_spectrum_are_a_problem(scored):
    path = record.spectra_path(scored["out"], REPO)
    _edit_json(path, lambda spectra: spectra[0]["effective"].update(fro_norm=spectra[0]["effective"]["fro_norm"] * 1.01))
    assert _flagged(_reader_problems(scored), "disagrees")


def test_a_non_finite_reader_score_is_a_problem(scored):
    assert _flagged(_reader_problems(scored, {**scored["row"], "reader_top8_energy_w": float("nan")}), "reader_top8_energy_w")


def test_no_reader_record_at_all_is_reported_not_raised(scored, tmp_path):
    assert _reader_problems({**scored, "out": tmp_path / "elsewhere" / "scores.jsonl"})


def _run(s: dict, monkeypatch, cached: set[str], *extra: str) -> list[str]:
    """The script's main over the one scored checkpoint; returns what it deleted."""
    script, deleted = s["script"], []
    sweep.write_rows([s["main_row"]], s["main_out"])
    monkeypatch.setattr(script, "cached", lambda repo, revision, files: repo in cached)
    monkeypatch.setattr(script, "evict", lambda repo, keep: deleted.append(repo) or 1)
    monkeypatch.setattr(script, "bases", lambda registry: {BASE})
    monkeypatch.setattr("sys.argv", ["reader_weights.py", "--rows", str(s["main_out"]), "--out", str(s["out"]), "--redo", *extra])
    script.main()
    return deleted


def test_a_checkpoint_the_pass_downloaded_is_deleted_once_its_record_validates(scored, monkeypatch):
    assert _run(scored, monkeypatch, set(), "--evict") == [REPO]


def test_a_checkpoint_that_was_already_on_disk_is_left_there(scored, monkeypatch):
    assert _run(scored, monkeypatch, {REPO}, "--evict") == []


def test_nothing_is_deleted_without_being_asked(scored, monkeypatch):
    assert _run(scored, monkeypatch, set()) == []


def test_a_checkpoint_whose_reader_record_has_a_problem_stays_on_disk(scored, monkeypatch):
    _edit_json(record.provenance_path(scored["main_out"], REPO), lambda b: b.update(revision=BASE_REV))
    assert _run(scored, monkeypatch, set(), "--evict") == []


def test_a_download_that_was_interrupted_is_still_the_passes_own(scored, monkeypatch):
    """The first run notes the download before it starts; a second run finds the files on disk
    and must still treat them as its own."""
    (scored["out"].parent / "downloaded.json").write_text(json.dumps([REPO]))
    assert _run(scored, monkeypatch, {REPO}, "--evict") == [REPO]


def test_fetch_asks_for_the_revision_it_is_given(monkeypatch, tmp_path):
    import huggingface_hub

    asked = {}

    class Api:
        def model_info(self, repo, revision=None):
            asked["info"] = revision
            return type("Info", (), {"siblings": [type("File", (), {"rfilename": "model.safetensors"})()]})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, revision=None, allow_patterns=None: asked.update(download=revision) or tmp_path)
    assert registry.fetch(REPO, revision=REV) == tmp_path and asked == {"info": REV, "download": REV}
    registry.fetch(REPO)
    assert asked == {"info": None, "download": None}
