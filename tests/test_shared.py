"""Whether the reading matrices a checkpoint changed were all changed along one input direction."""

import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import detect, record
from obliterated.edits import blind_readers_on_disk

from .test_readers import BASE_REV, pair, scored, snapshot  # noqa: F401  (scored is a fixture)


def e(*index: int, d: int = 8) -> torch.Tensor:
    """Factors as a record keeps them: unit axes as columns, (d, k), the strongest first."""
    return torch.eye(d)[list(index)].T.contiguous()


def kept(*factors: torch.Tensor, head: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    out = {f"model.layers.{i}.self_attn.q_proj.weight.eu": u for i, u in enumerate(factors)}
    return out | ({"lm_head.weight.eu": head} if head is not None else {})


def test_matrices_changed_along_one_direction_share_it():
    got = record.shared_scores(kept(e(0, 1), -e(0, 2), e(0, 3), -e(0, 4)))  # a factor's sign means nothing
    assert got["reader_shared_direction"] == pytest.approx(1.0)
    assert got["reader_shared_subspace"] == pytest.approx(1.0)
    assert got["reader_shared_n"] == 4


def test_matrices_changed_along_unrelated_directions_share_nothing():
    got = record.shared_scores(kept(e(0, 1), e(2, 3), e(4, 5)))
    assert got["reader_shared_direction"] == pytest.approx(0.0, abs=1e-9)
    assert got["reader_shared_subspace"] == pytest.approx(0.0, abs=1e-9)


def test_a_direction_every_matrix_holds_counts_even_where_it_is_not_the_strongest():
    got = record.shared_scores(kept(e(0, 1), e(1, 0), e(2, 0)))
    assert got["reader_shared_direction"] == pytest.approx(0.0, abs=1e-9)  # the strongest factors are three different axes
    assert got["reader_shared_subspace"] == pytest.approx(1.0)  # and the first axis is among the factors of all three


def test_two_matrices_score_the_cosine_between_their_strongest_factors():
    tilted = torch.tensor([math.cos(math.pi / 3), math.sin(math.pi / 3), 0, 0, 0, 0, 0, 0])[:, None]
    assert record.shared_scores(kept(e(0), tilted))["reader_shared_direction"] == pytest.approx(0.5)


def test_one_changed_matrix_shares_nothing_and_the_head_is_not_counted():
    got = record.shared_scores(kept(e(0, 1), head=e(0, 1)))
    assert got == {"reader_shared_direction": 0.0, "reader_shared_subspace": 0.0, "reader_shared_n": 1}
    assert record.shared_scores({})["reader_shared_n"] == 0


def test_factors_kept_in_half_precision_score_the_same():
    full = kept(e(0, 1), e(0, 2), e(3, 4))
    half = {k: v.half() for k, v in full.items()}
    assert record.shared_scores(half) == pytest.approx(record.shared_scores(full), abs=1e-3)


# ---- through the reading-side pass


def noisy(folder, out, scale: float, seed: int = 0):
    """A copy of a folder of reading matrices with every matrix changed a little, each its own
    way, as a fine-tune changes them: dense noise of ``scale`` times the matrix's own size."""
    out.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    for f in sorted(folder.glob("*.safetensors")):
        tensors = load_file(f)
        for name in tensors:
            if name.endswith("_proj.weight"):
                w = tensors[name]
                tensors[name] = w + scale * w.norm() / math.sqrt(w.numel()) * torch.randn(w.shape, generator=g)
        save_file(tensors, out / f.name)
    (out / "config.json").write_text((folder / "config.json").read_text())
    return out


def read(base_dir, cand_dir, base) -> dict:
    directions = torch.randn(base.n_layers, base.d_model, generator=torch.Generator().manual_seed(4))
    return record.read_readers(base_dir, cand_dir, directions)["fields"]


def test_a_reader_side_edit_shares_its_direction_and_matrices_changed_each_their_own_way_do_not(tokenizer, tmp_path):
    base, _ = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    v = torch.randn(base.d_model, generator=torch.Generator().manual_seed(11))
    edit = blind_readers_on_disk(folder, tmp_path / "edit", v)
    readers = blind_readers_on_disk(folder, tmp_path / "readers", v, alpha=0.0)  # the base's reading matrices, untouched
    edited, tuned = read(folder, edit, base), read(folder, noisy(readers, tmp_path / "tuned", 0.05), base)
    assert edited["reader_shared_direction"] == pytest.approx(1.0, abs=1e-3)
    assert edited["reader_shared_n"] == tuned["reader_shared_n"] == 5 * base.n_layers
    assert tuned["reader_shared_direction"] < 0.25  # 0.12; eight factors of 32 dimensions overlap too much to say the same of them


def test_noise_on_top_of_an_edit_lowers_its_energy_share_and_not_its_shared_direction(tokenizer, tmp_path):
    """What the score is for: the share of a change's energy in one direction falls as soon as
    anything else is added to the matrix, while the strongest factor of every matrix is still
    the edit's."""
    base, _ = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    v = torch.randn(base.d_model, generator=torch.Generator().manual_seed(11))
    edit = blind_readers_on_disk(folder, tmp_path / "edit", v)
    plain, padded = read(folder, edit, base), read(folder, noisy(edit, tmp_path / "padded", 0.12), base)
    assert plain["reader_top1_energy"] > 0.99 and padded["reader_top1_energy"] < 0.8
    assert padded["reader_shared_direction"] > 0.9


def test_a_shared_score_the_record_does_not_reproduce_is_a_problem(scored):  # noqa: F811
    row = scored["row"]
    assert row["reader_shared_direction"] == pytest.approx(1.0, abs=1e-3)
    assert record.validate_readers(row, scored["out"], scored["main_out"]) == []
    found = record.validate_readers({**row, "reader_shared_direction": 0.5}, scored["out"], scored["main_out"])
    assert any("reader_shared_direction" in p for p in found)
    old = {k: v for k, v in row.items() if not k.startswith("reader_shared_")}  # a row from before the score existed
    assert record.validate_readers(old, scored["out"], scored["main_out"]) == []
    assert detect.layer_of("lm_head.weight") is None
