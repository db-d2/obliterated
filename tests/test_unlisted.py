"""Directions the reading-side detector does not list: made from a base's refusal span, or searched for."""

import math

import pytest
import torch

from obliterated import detect, unlisted
from obliterated.edits import blind_readers, temporarily_blinded
from obliterated.refusal import kl_div, last_logits

from .conftest import make_lm
from .test_edits import PROMPTS, rand_dir, uneven_gains


def e(i: int, d: int = 6) -> torch.Tensor:
    return torch.eye(d)[i]


def span_of(directions: torch.Tensor, n_controls: int = 4) -> dict[str, torch.Tensor]:
    """What a base's span file holds of its directions, (positions, layers, d_model)."""
    basis, n_ref = detect.refusal_span(directions, n_controls)
    return {"directions": directions, "basis": basis, "n_ref": torch.tensor([n_ref])}


def outside_of(v: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """What is left of ``v`` after the best fit by ``rows``, worked out without a basis."""
    fit = torch.linalg.lstsq(rows.double().T, v.double()[:, None]).solution[:, 0]
    return v.double() - rows.double().T @ fit


def test_split_gives_the_parts_inside_and_outside_a_basis():
    inside, outside, share = unlisted.split(torch.tensor([3.0, 0.0, 4.0, 0.0, 0.0, 0.0]), torch.stack([e(0), e(1)]))
    assert inside.tolist() == [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert outside.tolist() == [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    assert share == pytest.approx(9 / 25)


def test_a_direction_is_described_by_its_share_inside_the_span_and_its_nearest_listed_direction():
    span = span_of(torch.stack([e(0), 2 * e(0), e(1)])[None])  # one position, three layers
    about = unlisted.describe(e(0) + e(2), span)
    assert about["inside"] == pytest.approx(0.5)
    assert about["listed_cosine"] == pytest.approx(1 / math.sqrt(2))
    assert unlisted.describe(-3 * e(1), span) == {"inside": pytest.approx(1.0), "listed_cosine": pytest.approx(1.0)}
    assert unlisted.describe(e(4), span) == {"inside": pytest.approx(0.0, abs=1e-6), "listed_cosine": pytest.approx(0.0, abs=1e-6)}


def test_the_controls_are_not_part_of_the_span_a_direction_is_described_against():
    span = span_of(torch.stack([e(0), e(1)])[None], n_controls=3)
    control = span["basis"][-1]
    assert unlisted.describe(control, span)["inside"] == pytest.approx(0.0, abs=1e-6)


def test_candidates_inside_the_span_combine_listed_directions_and_those_outside_meet_none():
    g = torch.Generator().manual_seed(3)
    directions = torch.randn(2, 3, 24, generator=g)
    listed, extra = directions.reshape(-1, 24), torch.randn(24, generator=g)
    span = span_of(directions, n_controls=4)
    everything = torch.cat([listed, span["basis"][int(span["n_ref"]):]])  # the listed directions and the controls
    found = unlisted.candidates(span, {"inside": listed[2] - listed[4], "mixed": listed[0] + extra}, n_random=3, seed=5)
    by_kind = {}
    for c in found:
        by_kind.setdefault(c["kind"], []).append(c)
        assert float(c["direction"].norm()) == pytest.approx(1.0, abs=1e-5), c["name"]
    assert len({c["name"] for c in found}) == len(found)
    assert {k: len(v) for k, v in by_kind.items()} == {"axis": 6, "random_inside": 3, "random_outside": 3, "remainder": 1}
    for c in by_kind["axis"] + by_kind["random_inside"]:
        assert float(outside_of(c["direction"], listed).norm()) < 1e-4, c["name"]
        assert c["inside"] == pytest.approx(1.0, abs=1e-5)
    for c in by_kind["random_outside"] + by_kind["remainder"]:
        assert float((everything.double() @ c["direction"].double()).abs().max()) < 1e-4, c["name"]
        assert c["inside"] == pytest.approx(0.0, abs=1e-5) and c["listed_cosine"] == pytest.approx(0.0, abs=1e-5)
    (rest,) = by_kind["remainder"]  # a direction wholly inside the span leaves nothing to make one from
    want = outside_of(listed[0] + extra, everything)
    assert rest["name"] == "remainder_mixed"
    torch.testing.assert_close(rest["direction"].double(), want / want.norm(), atol=1e-5, rtol=0)
    assert rest["of_whole"] == pytest.approx(float(want.norm() ** 2 / (listed[0] + extra).norm() ** 2), rel=1e-4)


def test_random_candidates_are_the_same_for_a_seed_and_differ_between_seeds():
    span = span_of(torch.randn(2, 3, 24, generator=torch.Generator().manual_seed(3)))
    pick = lambda seed: torch.stack([c["direction"] for c in unlisted.candidates(span, n_random=2, seed=seed) if c["kind"].startswith("random")])
    assert torch.equal(pick(1), pick(1))
    assert not torch.allclose(pick(1), pick(2), atol=1e-3)


def test_a_direction_is_padded_with_others_outside_the_span_and_stays_first():
    directions = torch.randn(2, 3, 24, generator=torch.Generator().manual_seed(3))
    span = span_of(directions, n_controls=4)
    v = outside_of(torch.randn(24, generator=torch.Generator().manual_seed(8)), span["basis"]).float()
    rows = unlisted.padded(v, span, 5, seed=2)
    assert rows.shape == (6, 24)
    torch.testing.assert_close(rows @ rows.T, torch.eye(6), atol=1e-5, rtol=0)
    torch.testing.assert_close(rows[0], v / v.norm(), atol=1e-5, rtol=0)
    assert float((span["basis"] @ rows.T).abs().max()) < 1e-5  # none of them leans on the span or its controls
    assert torch.equal(rows, unlisted.padded(v, span, 5, seed=2)) and not torch.allclose(rows[1:], unlisted.padded(v, span, 5, seed=3)[1:], atol=1e-3)
    assert torch.equal(unlisted.padded(v, span, 2, seed=2), rows[:3])  # more padding keeps what was there


# ---- searching for a direction


def objective(lm, direction, harmful, harmless, base, kl_weight):
    """What a search minimizes, measured on the edit made in the weights: how unlikely a first
    token other than a refusal is on the harmful prompts, plus the KL on the harmless ones."""
    with temporarily_blinded(lm, direction):
        h, s = last_logits(lm, harmful), last_logits(lm, harmless)
    refusing = h.double().softmax(-1)[:, lm.refusal_token_ids].sum(-1)
    return float(-(1 - refusing).log().mean() + kl_weight * kl_div(base, s).mean())


def searchable(tokenizer):
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    avoid = torch.linalg.qr(torch.randn(lm.d_model, 5, generator=torch.Generator().manual_seed(9))).Q.T.contiguous()
    return lm, avoid, PROMPTS[:2], PROMPTS[2:]


def test_a_search_ends_on_a_weight_edit_that_refuses_less_and_lies_outside_what_it_was_told_to_avoid(tokenizer):
    lm, avoid, harmful, harmless = searchable(tokenizer)
    base, start = last_logits(lm, harmless), rand_dir(lm, 4)
    weights = [p.clone() for p in lm.model.parameters()]
    found, history = unlisted.find_direction(lm, harmful, harmless, start, avoid, steps=40, lr=0.02, kl_weight=0.5, batch_size=2)
    assert float(found.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((avoid @ found).abs().max()) < 1e-5
    began = outside_of(start, avoid).float()
    assert objective(lm, found, harmful, harmless, base, 0.5) < 0.9 * objective(lm, began, harmful, harmless, base, 0.5)
    assert len(history) == 40 and all(math.isfinite(h["loss"]) for h in history)
    assert all(torch.equal(a, b) and b.requires_grad for a, b in zip(weights, lm.model.parameters())), "the model is as it was"
    torch.testing.assert_close(last_logits(lm, harmless), base, atol=1e-6, rtol=0)


def test_a_search_without_anything_to_avoid_is_free_to_go_anywhere(tokenizer):
    lm, avoid, harmful, harmless = searchable(tokenizer)
    found, _ = unlisted.find_direction(lm, harmful, harmless, avoid[0], None, steps=5, lr=0.02, batch_size=2)
    assert float(found.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((avoid @ found).abs().max()) > 0.5  # it began on a direction the other search may not touch


def test_a_search_repeats_for_a_seed(tokenizer):
    lm, avoid, _, _ = searchable(tokenizer)
    run = lambda seed: unlisted.find_direction(lm, PROMPTS, PROMPTS, rand_dir(lm, 4), avoid, steps=6, lr=0.02, batch_size=2, seed=seed)[0]
    assert torch.equal(run(0), run(0))
    assert not torch.allclose(run(0), run(1), atol=1e-4)


# ---- searching for a direction whose edit answers as a known one does


def test_an_answer_counts_up_to_its_first_end_token():
    ids = torch.tensor([[5, 2, 0, 0], [5, 6, 7, 8], [9, 5, 9, 2]])
    assert unlisted.until_end(ids, [2, 9]).tolist() == [[True, True, False, False], [True, True, True, True], [True, False, False, False]]


def generated(lm, prompts, tokens):
    enc = lm.tokenize(prompts)
    ids = lm.model.generate(**enc, max_new_tokens=tokens, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id)
    return enc, ids[:, enc["input_ids"].shape[1]:]


def test_the_answers_under_an_edit_are_what_the_edited_weights_generate(tokenizer):
    lm, other = make_lm("llama_untied", tokenizer), make_lm("llama_untied", tokenizer)
    for m in (lm, other):
        uneven_gains(m)
    d = rand_dir(lm, 6)
    ids, real = unlisted.answers_under(lm, d, PROMPTS, tokens=6, batch_size=3)
    blind_readers(other, d)
    _, want = generated(other, PROMPTS, 6)
    assert ids.shape == real.shape == (len(PROMPTS), 6)
    assert torch.equal(ids[:, : want.shape[1]], want)
    assert not torch.equal(want, generated(lm, PROMPTS, 6)[1][:, : want.shape[1]]), "the edit should change what is generated"


def answer_loss(lm, direction, prompts, ids, real):
    """How unlikely the edit made in the weights finds the given answers, per token."""
    enc = lm.tokenize(prompts)
    with temporarily_blinded(lm, direction), torch.no_grad():
        logits = lm.model(input_ids=torch.cat([enc["input_ids"], ids], 1),
                          attention_mask=torch.cat([enc["attention_mask"], torch.ones_like(ids)], 1)).logits
    n = ids.shape[1]
    each = -logits[:, -n - 1 : -1].double().log_softmax(-1).gather(-1, ids[..., None])[..., 0]
    return float(each[real].mean())


def test_a_search_towards_a_known_edit_ends_on_a_weight_edit_that_answers_more_like_it(tokenizer):
    lm, _, harmful, harmless = searchable(tokenizer)
    like, start = rand_dir(lm, 6), rand_dir(lm, 4)
    rest = torch.randn(lm.d_model, 4, generator=torch.Generator().manual_seed(9))
    avoid = torch.linalg.qr(torch.cat([like[:, None], rest], 1)).Q.T.contiguous()  # the known direction itself is out of bounds
    ids, real = unlisted.answers_under(lm, like, harmful, tokens=6)
    found, history = unlisted.find_direction(lm, harmful, harmless, start, avoid, like=like, tokens=6, steps=40, lr=0.02,
                                             kl_weight=0.0, batch_size=2)
    assert float((avoid @ found).abs().max()) < 1e-5 and float(found.norm()) == pytest.approx(1.0, abs=1e-5)
    began = outside_of(start, avoid).float()
    assert answer_loss(lm, found, harmful, ids, real) < 0.99 * answer_loss(lm, began, harmful, ids, real)  # 2.75 against 2.81
    assert history[0]["harmful"] == pytest.approx(answer_loss(lm, began, harmful, ids, real), rel=1e-3)  # the search measures the same thing


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Mac's GPU, which has no float64")
def test_a_search_runs_on_the_macs_gpu(tokenizer):
    lm, avoid, harmful, harmless = searchable(tokenizer)
    lm.model.to("mps")
    found, history = unlisted.find_direction(lm, harmful, harmless, rand_dir(lm, 4), avoid, steps=3, lr=0.02, batch_size=2)
    assert found.device.type == "cpu" and float(found.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((avoid @ found).abs().max()) < 1e-5 and len(history) == 3
