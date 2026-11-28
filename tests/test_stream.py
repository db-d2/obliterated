"""What the residual stream carries in front of each norm, and the directions it uses least."""

import pytest
import torch

from obliterated import detect, stream

from .conftest import make_lm
from .test_edits import PROMPTS, READABLE, uneven_gains
from .test_readers import BASE_REV, snapshot


def texts(lm, prompts=PROMPTS) -> list[str]:
    return [lm.format(p) for p in prompts]


def at_unit_rms(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


@pytest.mark.parametrize("kind", READABLE)
def test_every_norm_a_reading_matrix_sits_behind_has_a_moment_under_its_name_on_disk(kind, tokenizer, tmp_path):
    lm = make_lm(kind, tokenizer)
    got, n = stream.moments(lm, texts(lm), batch_size=2)
    layout = detect.reader_layout(detect._tensor_index(snapshot(tmp_path / "base", lm, BASE_REV)))
    assert set(got) == {norm for _, norm in layout}
    assert n == sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts(lm))
    for c in got.values():
        assert c.dtype == torch.float64 and c.shape == (lm.d_model, lm.d_model)
        assert float(c.diagonal().sum()) == pytest.approx(lm.d_model, rel=1e-5)  # every token counted at rms 1


def test_the_moment_in_front_of_a_layer_is_the_mean_outer_product_of_what_enters_it_before_any_gain(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    got, n = stream.moments(lm, texts(lm)[:1])  # one text: nothing is padded
    hidden = lm.model(**lm.tokenize(PROMPTS[:1]), output_hidden_states=True).hidden_states
    for i in range(lm.n_layers):
        x = at_unit_rms(hidden[i][0])
        assert len(x) == n
        torch.testing.assert_close(got[f"model.layers.{i}.input_layernorm.weight"], x.T @ x / n, atol=1e-5, rtol=1e-4)


def test_padding_is_not_counted(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    long, short = texts(lm)[:2]
    (a, na), (b, nb) = stream.moments(lm, [long]), stream.moments(lm, [short])
    both, n = stream.moments(lm, [long, short], batch_size=2)
    assert n == na + nb and na != nb
    for name in both:
        torch.testing.assert_close(both[name], (a[name] * na + b[name] * nb) / n, atol=1e-5, rtol=1e-4)


def test_measuring_the_stream_leaves_the_model_as_it_was(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    before = lm.model(**lm.tokenize(PROMPTS)).logits
    stream.moments(lm, texts(lm), batch_size=2)
    assert torch.equal(lm.model(**lm.tokenize(PROMPTS)).logits, before)


# ---- the directions a moment says are used least


def test_the_least_used_directions_of_a_diagonal_moment_are_its_smallest_coordinates():
    moment = torch.diag(torch.tensor([5.0, 0.1, 3.0, 0.2, 4.0, 0.3]))
    rows = stream.least_used(moment, 2)
    assert rows.shape == (2, 6)
    torch.testing.assert_close(rows.pow(2).sum(0), torch.tensor([0.0, 1, 0, 1, 0, 0]), atol=1e-6, rtol=0)
    assert stream.share(moment, rows) == pytest.approx(0.3 / 12.6, rel=1e-5)
    assert stream.share(moment, torch.eye(6)) == pytest.approx(1.0)


def test_least_used_directions_keep_clear_of_what_is_to_be_avoided_and_nothing_beside_them_is_used_less():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(40, 12, generator=g, dtype=torch.float64)
    moment = a.T @ a / 40
    avoid = torch.linalg.qr(torch.randn(12, 3, generator=g, dtype=torch.float64)).Q.T
    rows = stream.least_used(moment, 4, avoid=avoid.float()).double()
    torch.testing.assert_close(rows @ rows.T, torch.eye(4, dtype=torch.float64), atol=1e-5, rtol=0)
    assert float((rows @ avoid.T).abs().max()) < 1e-5
    # the moment on everything orthogonal to the three, with the three themselves pushed far up
    clear = torch.eye(12, dtype=torch.float64) - avoid.T @ avoid
    smallest = torch.linalg.eigvalsh(clear @ moment @ clear + 1e6 * avoid.T @ avoid)[:4]
    assert float((rows @ moment @ rows.T).diagonal().sum()) == pytest.approx(float(smallest.sum()), rel=1e-4)


def test_what_is_to_be_avoided_need_not_be_orthonormal():
    moment = torch.diag(torch.tensor([5.0, 0.1, 3.0, 0.2, 4.0, 0.3]))
    avoid = torch.tensor([[0.0, 2, 0, 0, 0, 0], [0.0, 4, 0, 0, 0, 0], [0.0, 1, 0, 1, 0, 0]])  # two coordinates, said three times
    rows = stream.least_used(moment, 2, avoid=avoid)
    torch.testing.assert_close(rows.pow(2).sum(0), torch.tensor([0.0, 0, 1, 0, 0, 1]), atol=1e-6, rtol=0)


def test_more_directions_than_are_left_are_refused():
    with pytest.raises(ValueError, match="directions"):
        stream.least_used(torch.eye(6), 5, avoid=torch.eye(6)[:2])
    assert stream.least_used(torch.eye(6), 4, avoid=torch.eye(6)[:2]).shape == (4, 6)


def test_past_the_least_used_come_the_ones_used_next_least():
    moment = torch.diag(torch.tensor([5.0, 0.1, 3.0, 0.2, 4.0, 0.3]))
    rows = stream.least_used(moment, 2, skip=2)
    torch.testing.assert_close(rows.abs(), torch.tensor([[0.0, 0, 0, 0, 0, 1], [0.0, 0, 1, 0, 0, 0]]), atol=1e-6, rtol=0)  # the less used of the two first
    # what is to be avoided is left out before anything is counted
    rows = stream.least_used(moment, 1, avoid=torch.eye(6)[1:2], skip=1)
    torch.testing.assert_close(rows.abs(), torch.tensor([[0.0, 0, 0, 0, 0, 1]]), atol=1e-6, rtol=0)


def test_directions_that_are_skipped_count_against_what_is_left():
    with pytest.raises(ValueError, match="directions"):
        stream.least_used(torch.eye(6), 3, avoid=torch.eye(6)[:2], skip=2)
    assert stream.least_used(torch.eye(6), 2, avoid=torch.eye(6)[:2], skip=2).shape == (2, 6)


# ---- the stream's mean, and what a change is weighed by


def test_the_mean_in_front_of_a_layer_is_the_mean_of_what_enters_it_at_unit_size(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    means, second, n = stream.measure(lm, texts(lm)[:1])
    again, m = stream.moments(lm, texts(lm)[:1])
    with torch.no_grad():
        hidden = lm.model(**lm.tokenize(PROMPTS[:1]), output_hidden_states=True).hidden_states
    assert n == m and set(means) == set(second) == set(again)
    for i in range(lm.n_layers):
        name = f"model.layers.{i}.input_layernorm.weight"
        assert means[name].dtype == torch.float64
        torch.testing.assert_close(means[name], at_unit_rms(hidden[i][0]).mean(0), atol=1e-5, rtol=1e-4)
        assert torch.equal(second[name], again[name])


def some_moment(seed: int = 0, d: int = 12, shift: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forty points in d dimensions, their second moment and their mean."""
    a = torch.randn(40, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) + shift
    return a, a.T @ a / 40, a.mean(0)


def test_the_root_of_a_moment_squares_to_it():
    _, moment, _ = some_moment()
    r = stream.root(moment.float())
    assert r.dtype == torch.float64
    torch.testing.assert_close(r, r.T, atol=1e-12, rtol=0)
    torch.testing.assert_close(r @ r, moment, atol=1e-6, rtol=0)


def test_with_the_mean_the_root_is_of_the_variance_about_it():
    a, moment, mean = some_moment(shift=3.0)  # a stream that sits far from zero
    r = stream.root(moment, mean=mean)
    about = a - mean
    torch.testing.assert_close(r @ r, about.T @ about / 40, atol=1e-9, rtol=0)


def test_a_cap_weighs_no_direction_above_that_many_times_the_average_one():
    moment = torch.diag(torch.tensor([5.0, 0.1, 3.0, 0.2, 4.0, 0.3], dtype=torch.float64))  # 2.1 on average
    r = stream.root(moment, cap=1.0)
    torch.testing.assert_close(r @ r, torch.diag(torch.tensor([2.1, 0.1, 2.1, 0.2, 2.1, 0.3], dtype=torch.float64)), atol=1e-9, rtol=0)
    half = stream.root(moment, cap=0.5)
    torch.testing.assert_close(half @ half, torch.diag(torch.tensor([1.05, 0.1, 1.05, 0.2, 1.05, 0.3], dtype=torch.float64)), atol=1e-9, rtol=0)


# ---- a change's factors, weighed by the stream


def a_change(seed: int = 1, rows: int = 30, d: int = 12, rank: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if rank is None:
        return torch.randn(rows, d, generator=g, dtype=torch.float64)
    return torch.randn(rows, rank, generator=g, dtype=torch.float64) @ torch.randn(rank, d, generator=g, dtype=torch.float64)


def test_with_every_direction_weighed_alike_the_factors_are_the_changes_own():
    m = a_change()
    values, directions = stream.weighted_factors(m.T @ m, torch.eye(12, dtype=torch.float64), k=4)
    _, s, vh = torch.linalg.svd(m)
    torch.testing.assert_close(values, s[:4], rtol=1e-8, atol=0)
    for j in range(4):
        assert abs(float(directions[:, j] @ vh[j])) == pytest.approx(1.0, abs=1e-8)


def test_a_change_along_a_direction_the_stream_uses_comes_before_larger_ones_along_directions_it_does_not():
    """An edit along one direction and four decoys, each 1.5 times as large in the weights. The
    stream has a hundred times less along a decoy, so to the model the edit is the largest."""
    moment = torch.diag(torch.tensor([3.0, 2.0, 1.5, 1.0] + [0.01] * 8, dtype=torch.float64))
    g = torch.Generator().manual_seed(2)
    b = torch.zeros(12, dtype=torch.float64)
    b[1] = 1.0
    decoys = torch.zeros(4, 12, dtype=torch.float64)
    decoys[:, 4:] = torch.linalg.qr(torch.randn(8, 4, generator=g, dtype=torch.float64)).Q.T
    left = torch.linalg.qr(torch.randn(30, 5, generator=g, dtype=torch.float64)).Q
    m = torch.outer(left[:, 0], b) + 1.5 * left[:, 1:] @ decoys
    gram = m.T @ m
    strongest = torch.linalg.svd(m).Vh[0]
    assert abs(float(strongest @ b)) < 1e-8  # by the weights alone, a decoy
    values, directions = stream.weighted_factors(gram, stream.root(moment), k=3)
    assert abs(float(directions[:, 0] @ b)) == pytest.approx(1.0, abs=1e-8)
    assert float(values[0]) == pytest.approx(2.0**0.5, rel=1e-8)  # what the stream has along b, 2.0 in mean square
    torch.testing.assert_close(values[1:], torch.full((2,), 0.15, dtype=torch.float64), rtol=1e-8, atol=0)  # 1.5 * sqrt(0.01)
    assert float((directions[:, 1:].T @ b).abs().max()) < 1e-8


def test_the_first_directions_span_what_the_largest_weighed_parts_of_a_change_read():
    m, (_, moment, _) = a_change(3), some_moment(4)
    r = stream.root(moment)
    values, directions = stream.weighted_factors(m.T @ m, r, k=5)
    torch.testing.assert_close(directions.T @ directions, torch.eye(5, dtype=torch.float64), atol=1e-9, rtol=0)
    u, s, _ = torch.linalg.svd(m @ r, full_matrices=False)  # the change as the stream meets it
    torch.testing.assert_close(values, s[:5], rtol=1e-8, atol=0)
    reads = m.T @ u[:, :5]  # what each of its largest parts reads off the stream, a column each
    for j in range(1, 6):
        inside = directions[:, :j] @ (directions[:, :j].T @ reads[:, :j])
        torch.testing.assert_close(inside, reads[:, :j], atol=1e-8 * float(reads.norm()), rtol=0)


def test_the_squared_values_add_up_to_the_mean_square_of_what_the_change_does_to_the_stream():
    m, (a, moment, _) = a_change(5), some_moment(6)
    values, _ = stream.weighted_factors(m.T @ m, stream.root(moment), k=12)
    assert float(values.pow(2).sum()) == pytest.approx(float((a @ m.T).pow(2).sum() / 40), rel=1e-9)


def test_a_change_of_lower_rank_than_asked_for_gives_only_that_many_directions():
    m, (_, moment, _) = a_change(7, rank=2), some_moment(8)
    values, directions = stream.weighted_factors(m.T @ m, stream.root(moment), k=5)
    assert values.shape == (2,) and directions.shape == (12, 2)
    row_space = torch.linalg.svd(m, full_matrices=False).Vh[:2]
    torch.testing.assert_close(row_space.T @ (row_space @ directions), directions, atol=1e-8, rtol=0)
