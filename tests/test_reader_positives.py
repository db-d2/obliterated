"""Reader-side edits made again from the base's files and read by the reading-side pass."""

import json

import pytest
import torch
from safetensors.torch import load_file

from obliterated import detect, record, sweep
from obliterated.edits import EditSpec, blind_readers, blind_readers_on_disk

from .conftest import load_script
from .test_edits import READABLE
from .test_readers import BASE, BASE_REV, REPO, pair, snapshot

WRITERS = ("o_proj", "down_proj", "embed_tokens")


def subspace(lm, k: int = 2) -> torch.Tensor:
    return torch.randn(k, lm.d_model, generator=torch.Generator().manual_seed(11))


def saved(folder) -> dict[str, torch.Tensor]:
    return {k: v for f in sorted(folder.glob("*.safetensors")) for k, v in load_file(f).items()}


# ---- the edit, from files to files


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", READABLE)
def test_the_edit_made_on_disk_is_the_one_made_in_memory(kind, dtype, tokenizer, tmp_path):
    """Bit for bit, in the precision the weights are stored in: the positives are then the
    edits whose behaviour the sweep measured, not something close to them."""
    base, cand = pair(kind, tokenizer)
    base.model.to(dtype)
    cand.model.to(dtype)
    sub, layers = subspace(base), [1, base.n_layers - 1]
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    blind_readers(cand, sub, alpha=0.5, layers=layers)
    out = blind_readers_on_disk(folder, tmp_path / "edit", sub, alpha=0.5, layers=layers)
    got, want, before = saved(out), cand.model.state_dict(), base.model.state_dict()
    layout = detect.reader_layout(detect._tensor_index(folder))
    assert len(layout) == 5 * base.n_layers + 1
    for name, norm in layout:
        assert got[name].dtype == dtype, name
        assert torch.equal(got[name], want[name]), name
        assert torch.equal(got[norm], before[norm]), norm
    assert not torch.equal(got["model.layers.1.self_attn.q_proj.weight"], before["model.layers.1.self_attn.q_proj.weight"])
    assert torch.equal(got["model.layers.0.self_attn.q_proj.weight"], before["model.layers.0.self_attn.q_proj.weight"])
    assert not torch.equal(got[detect.HEAD], before[detect.HEAD])  # the band holds the last layer
    assert not any(w in name for name in got for w in WRITERS)  # nothing a model could be loaded from


def test_the_head_is_left_alone_unless_the_last_layer_is_edited(tokenizer, tmp_path):
    base, _ = pair("llama_tied", tokenizer)
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    out = blind_readers_on_disk(folder, tmp_path / "edit", subspace(base), layers=[0, 1])
    assert torch.equal(saved(out)[detect.HEAD], base.model.state_dict()[detect.HEAD])


def test_the_reading_side_pass_finds_every_matrix_of_an_edit_made_on_disk(tokenizer, tmp_path):
    base, _ = pair("gemma2", tokenizer)
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    sub = subspace(base, 1)
    out = blind_readers_on_disk(folder, tmp_path / "edit", sub, alpha=0.25, layers=[2])
    found = list(detect.scan_readers(folder, out, sub.repeat(base.n_layers, 1)))
    assert len(found) == 5 * base.n_layers + 1 and not any(f.get("missing") for f in found)
    assert [f["name"] for f in found if f["changed"]] == [n for n, _ in detect.reader_layout(detect._tensor_index(folder))
                                                         if ".layers.2." in n]
    for f in found:
        ratio = float(f["read_cand"][0] / f["read_base"][0])
        assert ratio == pytest.approx(0.75 if f["changed"] else 1.0, abs=1e-5), f["name"]


# ---- the sweep's points, scored


@pytest.fixture
def swept(tmp_path, tokenizer, monkeypatch):
    """A base whose fine-tunes the reader pass has scored (so its directions and its files are
    on record), and a sweep over it with two reader-side points and one writer-side point."""
    script = load_script("reader_positives")
    base, _ = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "hub" / "base", base, BASE_REV)
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: folder)
    sub = torch.linalg.qr(subspace(base).T)[0].T
    directions = sub[0].repeat(base.n_layers, 1)
    readers, sweeps = tmp_path / "readers" / "set.jsonl", tmp_path / "sweeps"
    readers.parent.mkdir()
    (sweeps / BASE.replace("/", "__")).mkdir(parents=True)
    sweep.write_rows([{"repo": REPO, "base": BASE, "label": "benign_finetune"},
                      {"repo": "someone/other", "base": BASE, "label": "benign_finetune"}], readers)
    for repo in (REPO, "someone/other"):
        record.write(readers, repo, {"reference.directions": directions.half()},
                     {"repo": repo, "base": BASE, "base_revision": BASE_REV, "base_files": record.snapshot_info(folder)["files"]})
    point = {"tag": "pca", "model": BASE, "kl_harmless": 0.1}
    sweep.write_rows([
        {**point, "mode": "readers", "alpha": 1.0, "k": 1, "band": "all", "n_layers_edited": base.n_layers, "bypass_rate": 1.0},
        {**point, "mode": "readers", "alpha": 0.25, "k": 2, "band": "source", "n_layers_edited": 1, "bypass_rate": 0.5},
        {**point, "mode": "weights", "alpha": 1.0, "k": 1, "band": "all", "n_layers_edited": base.n_layers, "bypass_rate": 1.0},
    ], sweeps / BASE.replace("/", "__") / "frontier.jsonl")
    EditSpec(direction=sub.tolist(), source_layer=1, model=BASE, method="pca").save(
        sweeps / BASE.replace("/", "__") / "subspace_pca_k2.json")
    return {"script": script, "sweeps": sweeps, "readers": readers, "scratch": tmp_path / "scratch", "n_layers": base.n_layers,
            "out": sweeps / BASE.replace("/", "__") / "reader_scores.jsonl", "folder": folder, "subspace": sub}


def run(s: dict, monkeypatch, *extra: str) -> list[dict]:
    monkeypatch.setattr("sys.argv", ["reader_positives.py", "--model", BASE, "--sweeps", str(s["sweeps"]),
                                     "--readers", str(s["readers"]), "--scratch", str(s["scratch"]), *extra])
    s["script"].main()
    return sweep.read_rows(s["out"])


def test_every_reader_side_point_of_the_sweep_gets_the_scores_a_checkpoint_gets(swept, monkeypatch):
    rows = run(swept, monkeypatch)
    assert [(r["alpha"], r["k"], r["band"]) for r in rows] == [(1.0, 1, "all"), (0.25, 2, "source")]
    full, part = rows
    n = swept["n_layers"]
    assert set(record.READER_SCORES) <= set(full)
    assert (full["reader_n_changed"], full["reader_head_changed"]) == (5 * n, True)
    assert full["reader_top1_energy"] == pytest.approx(1.0, abs=1e-6)  # one direction of the two
    assert part["reader_top1_energy"] < 0.999 and part["reader_top2_energy"] == pytest.approx(1.0, abs=1e-6)
    assert full["reader_refusal_alignment"] == pytest.approx(1.0, abs=1e-3)
    assert full["reader_read_drop"] == pytest.approx(1.0, abs=1e-3)
    assert full["reader_read_drop_med"] == pytest.approx(1.0, abs=1e-3)
    assert full["reader_head_read_drop"] == pytest.approx(1.0, abs=1e-3)
    assert (part["reader_n_changed"], part["reader_head_changed"]) == (5, False)
    assert part["reader_refusal_alignment"] == pytest.approx(1.0, abs=1e-3)
    assert part["reader_read_drop"] == pytest.approx(0.25, abs=1e-3)
    assert part["reader_read_drop_med"] == pytest.approx(0.0, abs=1e-6)  # one layer of four
    assert (part["bypass_rate"], part["kl_harmless"], part["model"], part["mode"]) == (0.5, 0.1, BASE, "readers")


def test_the_edited_matrices_are_gone_once_they_are_scored(swept, monkeypatch):
    run(swept, monkeypatch)
    assert not swept["scratch"].exists() or not any(swept["scratch"].iterdir())


def test_what_the_scores_were_made_from_is_written_down(swept, monkeypatch):
    run(swept, monkeypatch)
    made = json.loads(swept["out"].with_suffix(".json").read_text())
    assert made["base_revision"] == BASE_REV and made["directions_from"] == REPO
    assert made["base_files"] == record.snapshot_info(swept["folder"])["files"]
    assert made["versions"]["source_sha256"] and len(made["subspace_sha256"]) == 64


def test_a_second_run_scores_only_what_is_missing(swept, monkeypatch):
    first = run(swept, monkeypatch)
    monkeypatch.setattr(swept["script"].record, "read_readers", lambda *a, **k: pytest.fail("scored a point twice"))
    assert run(swept, monkeypatch) == first


def test_a_band_the_sweep_did_not_edit_is_refused(swept, monkeypatch):
    """The band's layers are worked out again here; if that no longer gives what the sweep
    edited, the scores would belong to another edit."""
    path = swept["sweeps"] / BASE.replace("/", "__") / "frontier.jsonl"
    rows = sweep.read_rows(path)
    rows[1]["n_layers_edited"] = 3
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="source"):
        run(swept, monkeypatch)


def test_a_base_other_than_the_one_the_fine_tunes_were_read_against_is_refused(swept, monkeypatch):
    (swept["folder"] / "model.safetensors").write_bytes(b"not the weights")
    with pytest.raises(SystemExit, match="lab/base"):
        run(swept, monkeypatch)


def test_records_of_one_base_that_disagree_about_its_directions_are_refused(swept, monkeypatch):
    kept = load_file(record.tensors_path(swept["readers"], "someone/other"))
    record.write(swept["readers"], "someone/other", {"reference.directions": kept["reference.directions"].flip(1)},
                 json.loads(record.provenance_path(swept["readers"], "someone/other").read_text()))
    with pytest.raises(SystemExit, match="directions"):
        run(swept, monkeypatch)


# ---- the same, on the base's refusal span


def span_folder(s: dict, tmp_path):
    """The base's span file: the directions the records hold at the last position, and the
    sweep's second direction at a position before it."""
    sub, n = s["subspace"], s["n_layers"]
    dirs = torch.stack([sub[1].repeat(n, 1), sub[0].repeat(n, 1)])
    span = record.make_span(s["folder"], dirs, n_controls=4)
    record.write_span(tmp_path / "span", BASE, span, {"model": BASE, "base_revision": BASE_REV,
                                                      "base_files": record.snapshot_info(s["folder"])["files"]})
    return tmp_path / "span"


def test_with_a_span_every_point_is_scored_at_every_position(swept, monkeypatch, tmp_path):
    full, part = run(swept, monkeypatch, "--span", str(span_folder(swept, tmp_path)))
    assert set(record.SPAN_SCORES) <= set(full)
    assert full["reader_span_read_drop_by_position"] == [pytest.approx(0.0, abs=1e-3), pytest.approx(1.0, abs=1e-3)]  # one direction
    assert part["reader_span_read_drop_by_position"] == [pytest.approx(0.25, abs=1e-3), pytest.approx(0.25, abs=1e-3)]  # both
    assert part["reader_span_last_read_drop"] == pytest.approx(part["reader_read_drop"], abs=1e-4)
    made = json.loads(swept["out"].with_suffix(".json").read_text())
    assert made["span"]["path"] == str(record.span_path(tmp_path / "span", BASE).resolve()) and len(made["span"]["sha256"]) == 64


def test_without_a_span_the_rows_are_as_before(swept, monkeypatch):
    rows = run(swept, monkeypatch)
    assert not any(k.startswith("reader_span_") for r in rows for k in r)
    assert "span" not in json.loads(swept["out"].with_suffix(".json").read_text())


def test_a_span_that_is_missing_or_about_another_base_is_refused(swept, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="span"):
        run(swept, monkeypatch, "--span", str(tmp_path / "nowhere"))
    spans = span_folder(swept, tmp_path)
    path = record.span_path(spans, BASE).with_suffix(".json")
    path.write_text(json.dumps({**json.loads(path.read_text()), "base_revision": "0" * 40}))
    with pytest.raises(SystemExit, match="other weights"):
        run(swept, monkeypatch, "--span", str(spans))
    assert not swept["out"].exists()
