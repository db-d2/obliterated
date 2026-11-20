"""The shared-direction score with each change weighed by the stream, kept at two sizes and used together."""

import pytest
import torch

from obliterated import record, stream, sweep
from obliterated.edits import blind_readers_on_disk

from .conftest import make_lm
from .test_edits import PROMPTS, rand_dir, uneven_gains
from .test_padded import layout, made  # noqa: F401  (made is a fixture)
from .test_readers import BASE_REV, snapshot
from .test_shared import e, noisy


def alike(made) -> dict[str, torch.Tensor]:  # noqa: F811
    """For every norm, a weighing that counts every direction the same."""
    return {norm: torch.eye(made["base"].d_model, dtype=torch.float64) for norm in set(layout(made).values())}


def mostly_along_the_edit(made) -> dict[str, torch.Tensor]:  # noqa: F811
    """For every norm, the root of a stream that has four hundred times as much along the edit's direction as along any other."""
    moment = 0.01 * torch.eye(made["base"].d_model, dtype=torch.float64) + 4.0 * torch.outer(made["unit"], made["unit"])
    return stream.roots({norm: moment for norm in set(layout(made).values())})


def weighted(made, cand, roots, k: int) -> float:  # noqa: F811
    kept = record.weighted_readers(made["folder"], cand, roots, k=k)
    return record.weighted_shared_scores(kept, sizes=(k,))[f"reader_weighted_shared_k{k}"]


def unweighted(made, cand, k: int) -> float:  # noqa: F811
    directions = made["v"].repeat(made["base"].n_layers, 1)
    return record.read_readers(made["folder"], cand, directions, k=k)["fields"]["reader_shared_subspace"]


# ---- the score, from what is kept


def kept(*factors: torch.Tensor, head: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    out = {f"model.layers.{i}.self_attn.q_proj.weight.wu": u for i, u in enumerate(factors)}
    return out | ({"lm_head.weight.wu": head} if head is not None else {})


def test_the_two_sizes_that_are_used_together_are_eight_and_all_that_is_kept():
    assert record.WEIGHTED_SIZES == (8, 128)
    assert record.WEIGHTED_SCORES == ("reader_weighted_shared_k8", "reader_weighted_shared_k128")


def test_each_size_scores_the_first_that_many_directions_of_every_matrix():
    """The first axis is the third direction of each matrix: two kept of each share nothing, three share it."""
    got = record.weighted_shared_scores(kept(e(1, 2, 0), e(3, 4, 0), e(5, 6, 0)), sizes=(2, 3))
    assert got["reader_weighted_shared_k2"] == pytest.approx(0.0, abs=1e-9)
    assert got["reader_weighted_shared_k3"] == pytest.approx(1.0)
    assert got["reader_weighted_shared_n"] == 3


def test_a_matrix_with_fewer_directions_than_a_size_gives_all_it_has():
    got = record.weighted_shared_scores(kept(e(0), e(0, 1), e(0, 2, 3)))
    assert got == {"reader_weighted_shared_k8": pytest.approx(1.0), "reader_weighted_shared_k128": pytest.approx(1.0), "reader_weighted_shared_n": 3}


def test_only_what_was_weighed_is_scored_and_the_head_is_not_counted():
    mixed = kept(e(0, 1), head=e(0, 1)) | {"model.layers.1.self_attn.q_proj.weight.eu": e(0, 1)}  # the second matrix kept unweighed
    assert record.weighted_shared_scores(mixed, sizes=(2,)) == {"reader_weighted_shared_k2": 0.0, "reader_weighted_shared_n": 1}
    assert record.weighted_shared_scores({})["reader_weighted_shared_n"] == 0


# ---- from two folders of weights and a measurement of the stream


def test_an_edit_among_larger_decoys_is_found_when_the_stream_has_little_along_the_decoys(made):  # noqa: F811
    """Four decoys in each matrix, each 1.5 times the edit in the weights. A detector that keeps
    two directions of each matrix keeps two decoys. Weighed by a stream that has most along the
    edit's direction, the edit is the largest part of every matrix's change. What that part
    reads is not the edit's direction alone: a decoy is the matrix's own output along another
    direction, which is not orthogonal to its output along the edit's, so some of each decoy
    comes with it."""
    plain, hidden = made["paths"]
    roots = mostly_along_the_edit(made)
    assert unweighted(made, plain, 2) == pytest.approx(1.0, abs=1e-3)
    assert weighted(made, plain, roots, 2) == pytest.approx(1.0, abs=1e-3)
    assert unweighted(made, hidden, 2) < 0.3  # 0.14
    assert weighted(made, hidden, roots, 2) > 0.7  # 0.83
    first = [abs(float(u[:, 0].double() @ made["unit"])) for u in record.weighted_readers(made["folder"], hidden, roots, k=1).values()]
    assert min(first) > 0.6 and sorted(first)[len(first) // 2] > 0.85  # 0.71 and 0.91
    assert weighted(made, hidden, roots, 5) == pytest.approx(1.0, abs=1e-3)  # with all five parts kept the direction is in every matrix


def test_with_every_direction_weighed_alike_the_score_is_the_unweighted_one(made):  # noqa: F811
    plain, hidden = made["paths"]
    tuned = noisy(made["tmp"] / "untouched", made["tmp"] / "tuned", 0.05)
    for cand in (plain, hidden, tuned):
        assert weighted(made, cand, alike(made), 2) == pytest.approx(unweighted(made, cand, 2), abs=2e-3)


def test_the_first_directions_kept_are_the_ones_a_smaller_record_would_hold(made):  # noqa: F811
    tuned = noisy(made["tmp"] / "untouched", made["tmp"] / "tuned", 0.05)
    roots = mostly_along_the_edit(made)
    both = record.weighted_shared_scores(record.weighted_readers(made["folder"], tuned, roots, k=6), sizes=(2, 6))
    assert both["reader_weighted_shared_k2"] == pytest.approx(weighted(made, tuned, roots, 2), abs=2e-3)
    assert both["reader_weighted_shared_k6"] == pytest.approx(weighted(made, tuned, roots, 6), abs=2e-3)
    assert both["reader_weighted_shared_k2"] < both["reader_weighted_shared_k6"]


def test_what_is_kept_is_each_changed_layer_matrix_in_half_precision_and_not_the_head(made):  # noqa: F811
    hidden = made["paths"][1]
    names = [name for name in layout(made) if name != "lm_head.weight"]
    got = record.weighted_readers(made["folder"], hidden, alike(made), k=3)
    assert set(got) == {f"{name}.wu" for name in names}  # the edit changes the head as well
    for u in got.values():
        assert u.dtype == torch.float16 and u.shape == (made["base"].d_model, 3)
    band = blind_readers_on_disk(made["folder"], made["tmp"] / "band", made["v"], layers=[1])
    assert set(record.weighted_readers(made["folder"], band, alike(made))) == {f"{name}.wu" for name in names if ".layers.1." in name}


def test_weights_that_did_not_change_keep_nothing_and_score_nothing(made):  # noqa: F811
    got = record.weighted_readers(made["folder"], made["tmp"] / "untouched", alike(made))
    assert got == {}
    assert record.weighted_shared_scores(got) == {"reader_weighted_shared_k8": 0.0, "reader_weighted_shared_k128": 0.0, "reader_weighted_shared_n": 0}


def test_a_norm_the_stream_was_not_measured_in_front_of_is_refused(made):  # noqa: F811
    roots = alike(made)
    del roots["model.layers.2.post_attention_layernorm.weight"]
    with pytest.raises(ValueError, match="model.layers.2.post_attention_layernorm.weight"):
        record.weighted_readers(made["folder"], made["paths"][0], roots)


def test_the_stream_as_it_is_measured_and_saved_weighs_the_matrices_behind_each_norm(tokenizer, tmp_path):
    """Measured on the model, saved, read back, and used on its files. With one direction kept
    of each matrix, the weights alone give a decoy; the stream has next to nothing along the
    directions that measurement says are used least, and the edit comes first."""
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    folder, v = snapshot(tmp_path / "base", lm, BASE_REV), rand_dir(lm, 11)
    means, moments, _ = stream.measure(lm, [lm.format(p) for p in PROMPTS[1:3]])  # eleven tokens: most of the 32 dimensions are left unused
    stream.save(tmp_path / "stream.safetensors", means, moments)
    read_means, read_moments = stream.load(tmp_path / "stream.safetensors")
    assert set(read_means) == set(read_moments) == set(moments)
    for norm in moments:
        assert read_moments[norm].dtype == torch.float64
        assert torch.equal(read_moments[norm], moments[norm]) and torch.equal(read_means[norm], means[norm])
    among = {norm: stream.least_used(c, 8, avoid=v[None]) for norm, c in moments.items()}
    hidden = blind_readers_on_disk(folder, tmp_path / "hidden", v, decoys=4, decoy_gain=1.5, seed=3, among=among)
    score = lambda roots: record.weighted_shared_scores(record.weighted_readers(folder, hidden, roots, k=1), sizes=(1,))
    alike = {norm: torch.eye(lm.d_model, dtype=torch.float64) for norm in moments}
    by_the_weights, by_the_stream = score(alike), score(stream.roots(read_moments))
    assert by_the_weights["reader_weighted_shared_k1"] < 0.3 and by_the_stream["reader_weighted_shared_k1"] > 0.7  # 0.13 and 0.78
    assert by_the_stream["reader_weighted_shared_n"] == 5 * lm.n_layers


def test_a_bases_measurement_has_a_file_of_its_own_in_a_folder():
    assert stream.path("runs/detect/stream", "lab/base").as_posix() == "runs/detect/stream/lab__base.safetensors"


def test_roots_about_the_mean_leave_out_what_the_stream_has_on_every_token_alike():
    a = torch.randn(40, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64) + 3.0
    moments, means = {"n": a.T @ a / 40}, {"n": a.mean(0)}
    torch.testing.assert_close(stream.roots(moments)["n"], stream.root(moments["n"]))
    torch.testing.assert_close(stream.roots(moments, means)["n"], stream.root(moments["n"], mean=means["n"]))


# ---- several scores used together


def test_scores_used_together_flag_what_is_above_the_threshold_of_any_of_them():
    thresholds = {"a": 0.4, "b": 0.6}
    assert sweep.flagged_by({"a": 0.5, "b": 0.1}, thresholds) == ["a"]
    assert sweep.flagged_by({"a": 0.1, "b": 0.7}, thresholds) == ["b"]
    assert sweep.flagged_by({"a": 0.5, "b": 0.7}, thresholds) == ["a", "b"]
    assert sweep.flagged_by({"a": 0.4, "b": 0.6}, thresholds) == []  # at a threshold is not above it
    assert sweep.flagged_by({"b": float("nan")}, thresholds) == []  # a score that is missing or not a number flags nothing


def test_each_negative_is_held_out_against_thresholds_from_the_others():
    negatives = [{"a": 0.1, "b": 0.5}, {"a": 0.3, "b": 0.2}, {"a": 0.2, "b": 0.1}]
    assert sweep.held_out(negatives, ("a", "b")) == [["b"], ["a"], []]  # two scores, each with a largest negative of its own
    one_tops_both = [{"a": 0.9, "b": 0.9}, {"a": 0.3, "b": 0.2}, {"a": 0.2, "b": 0.1}]
    assert sweep.held_out(one_tops_both, ("a", "b")) == [["a", "b"], [], []]
    assert sweep.held_out(negatives, ("a",)) == [[], ["a"], []]
