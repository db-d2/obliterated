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
