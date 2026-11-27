"""How far a model moves from its base over the base's whole answers to harmless prompts."""

from __future__ import annotations

import math

import pytest
import torch

from obliterated import divergence, sweep
from obliterated.edits import temporarily_blinded
from obliterated.refusal import kl_div, last_logits

from .conftest import make_lm

PROMPTS = ["tell me how to bake a cake", "write a poem", "build the thing", "tell me how"]


@pytest.fixture()
def lm(tokenizer):
    return make_lm("llama_untied", tokenizer)


def logits_after(lm, instruction: str, so_far: list[int]) -> torch.Tensor:
    """Next-token logits once ``so_far`` has been answered, from this one prompt run alone."""
    ids = lm.tokenize([instruction])["input_ids"]
    ids = torch.cat([ids, torch.tensor([so_far], dtype=ids.dtype)], dim=1)
    with torch.no_grad():
        return lm.model(input_ids=ids).logits[0, -1].double()


def alone(lm, base) -> list[torch.Tensor]:
    """The model's log-probabilities at every token of the base's answers, one prompt at a time."""
    return [torch.log_softmax(logits_after(lm, p, a[:t]), -1) for p, a in zip(base.instructions, base.tokens) for t in range(len(a))]


def edited(lm):
    """An edit to measure: what reads the residual stream no longer reads eight fixed directions."""
    v = torch.randn(lm.d_model, 8, generator=torch.Generator().manual_seed(1))
    return temporarily_blinded(lm, torch.linalg.qr(v).Q.T)


def test_greedy_answers_are_the_tokens_the_model_picks_one_at_a_time(lm):
    answers = divergence.greedy(lm, PROMPTS, max_new_tokens=5, batch_size=2)
    assert [len(a) for a in answers] == [5, 5, 5, 5]
    for prompt, answer in zip(PROMPTS, answers):
        for t, token in enumerate(answer):
            assert token == int(logits_after(lm, prompt, answer[:t]).argmax())


def test_a_greedy_answer_ends_with_the_token_that_ends_it(lm):
    unbounded = divergence.greedy(lm, PROMPTS[:2], max_new_tokens=5)
    stop = unbounded[0][1]
    lm.model.generation_config.eos_token_id = [stop, 0]
    answers = divergence.greedy(lm, PROMPTS[:2], max_new_tokens=5)
    for full, answer in zip(unbounded, answers):
        assert answer == (full[: full.index(stop) + 1] if stop in full else full)
    assert len(answers[0]) == 2 and len(answers[1]) == 5  # one answer ends early, the other does not


def test_teacher_forcing_gives_every_answer_token_the_log_probabilities_the_model_gives_alone(lm):
    answers = [[7, 8, 9], [10], [11, 12, 13, 14, 15], [6, 7]]
    got = list(divergence.teacher_forced(lm, PROMPTS, answers, batch_size=3))
    assert [tuple(g.shape) for g in got] == [(len(a), lm.model.config.vocab_size) for a in answers]
    for prompt, answer, logp in zip(PROMPTS, answers, got):
        for t in range(len(answer)):
            alone = torch.log_softmax(logits_after(lm, prompt, answer[:t]), -1)
            assert torch.allclose(logp[t].double(), alone, atol=1e-5)


def test_the_base_gives_its_own_answers_and_what_it_gave_every_token_of_them(lm):
    lm.model.generation_config.eos_token_id = 11  # so that the answers differ in length
    base = divergence.base_answers(lm, PROMPTS, max_new_tokens=5, batch_size=3)
    assert base.instructions == PROMPTS
    assert base.tokens == divergence.greedy(lm, PROMPTS, max_new_tokens=5)
    assert len({len(a) for a in base.tokens}) > 1
    flat = [token for answer in base.tokens for token in answer]
    assert tuple(base.logprobs.shape) == (len(flat), lm.model.config.vocab_size)
    assert base.logprobs.argmax(-1).tolist() == flat  # a greedy answer is the likeliest token every time
    assert torch.equal(base.logprobs, torch.cat(list(divergence.teacher_forced(lm, PROMPTS, base.tokens, batch_size=3))))


def test_a_model_that_was_not_edited_has_not_moved_at_any_token(lm):
    base = divergence.base_answers(lm, PROMPTS, max_new_tokens=5, batch_size=3)
    drop, kl = divergence.per_token(lm, base, batch_size=3)
    n = sum(map(len, base.tokens))
    assert tuple(drop.shape) == tuple(kl.shape) == (n,)
    assert torch.equal(drop, torch.zeros(n)) and torch.equal(kl, torch.zeros(n))


def test_every_token_gets_its_drop_in_log_probability_and_the_kl_between_the_two_models(lm):
    lm.model.generation_config.eos_token_id = 11  # answers of different lengths
    base = divergence.base_answers(lm, PROMPTS, max_new_tokens=5, batch_size=3)
    flat = [token for answer in base.tokens for token in answer]
    before = alone(lm, base)
    with edited(lm):
        drop, kl = divergence.per_token(lm, base, batch_size=2)
        after = alone(lm, base)
    assert torch.allclose(drop.double(), torch.stack([b[y] - a[y] for b, a, y in zip(before, after, flat)]), atol=1e-5)
    assert torch.allclose(kl.double(), torch.stack([(b.exp() * (b - a)).sum() for b, a in zip(before, after)]), atol=1e-5)
    assert kl.min() > 5e-4 and drop.abs().max() > 1e-2  # the edit moves the model far more than the tolerance


def test_the_summary_averages_over_tokens_and_keeps_the_first_token_of_each_answer_apart():
    base = divergence.Answers(["one", "two"], [[3], [4, 5, 6]], torch.zeros(4, 7))
    row = divergence.summary(base, drop=torch.tensor([4.0, 0.0, 0.0, 0.0]), kl=torch.tensor([0.8, 0.2, 0.4, 0.2]))
    assert row == {
        "answer_logprob_drop": pytest.approx(1.0),  # 4 / 4 tokens, where the mean of the two answers' means would be 2
        "answer_kl": pytest.approx(0.4),
        "answer_first_kl": pytest.approx(0.5),  # tokens 0 and 1 open the two answers
        "answer_prompts": 2,
        "answer_tokens": 4,
    }


def test_measured_on_an_edit_the_first_token_kl_is_the_kl_the_sweeps_report(lm):
    base = divergence.base_answers(lm, PROMPTS, max_new_tokens=5, batch_size=3)
    before = last_logits(lm, PROMPTS)
    with edited(lm):
        row = divergence.measure(lm, base, batch_size=3)
        after = last_logits(lm, PROMPTS)
    assert row["answer_first_kl"] == pytest.approx(float(kl_div(before, after).mean()), abs=1e-6)
    assert row["answer_first_kl"] > 5e-4
    assert (row["answer_prompts"], row["answer_tokens"]) == (4, 20)


@pytest.mark.parametrize("kind", ["qwen2_bias", "gemma2", "qwen2_moe"])
def test_padding_a_batch_does_not_change_what_a_model_gives_an_answer(kind, tokenizer):
    lm = make_lm(kind, tokenizer)
    answers = [[7, 8, 9], [10], [11, 12, 13, 14, 15], [6, 7]]
    one_at_a_time = list(divergence.teacher_forced(lm, PROMPTS, answers, batch_size=1))
    for alone_, batched in zip(one_at_a_time, divergence.teacher_forced(lm, PROMPTS, answers, batch_size=4)):
        assert torch.allclose(alone_, batched, atol=1e-5)


# ---- a fine-tune whose vocabulary is not the base's size


def test_a_model_with_more_tokens_than_the_base_is_compared_over_the_tokens_both_have(lm):
    base = divergence.base_answers(lm, PROMPTS, 4, 2)
    with edited(lm):
        want = divergence.per_token(lm, base, 2)
    lm.model.resize_token_embeddings(lm.model.config.vocab_size + 3)
    drop, kl = divergence.per_token(lm, base, 2)
    assert drop.abs().max() < 1e-5 and kl.abs().max() < 1e-5  # the added tokens aside, it is the base
    with edited(lm):
        got = divergence.per_token(lm, base, 2)
    assert want[1].min() > 5e-4
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-5)


def test_a_model_with_fewer_tokens_than_the_base_is_compared_over_the_tokens_it_has(lm):
    prompts, answers = ["tell me how", "build the thing"], [[7, 8, 9], [10, 11]]  # nothing here uses the last token of the vocabulary
    base = divergence.Answers(prompts, answers, torch.cat(list(divergence.teacher_forced(lm, prompts, answers, 2))))
    v = lm.model.config.vocab_size - 1
    lm.model.resize_token_embeddings(v)
    with edited(lm):
        drop, kl = divergence.per_token(lm, base, 2)
        logq = torch.cat(list(divergence.teacher_forced(lm, prompts, answers, 2))).double()
    logp = torch.log_softmax(base.logprobs[:, :v].double(), -1)  # the base over the tokens both have
    chosen = torch.tensor([t for a in answers for t in a])
    assert logq.shape[1] == v and kl.min() > 5e-4
    assert torch.allclose(drop.double(), (logp - logq).gather(1, chosen[:, None])[:, 0], atol=1e-5)
    assert torch.allclose(kl.double(), (logp.exp() * (logp - logq)).sum(-1), atol=1e-5)


def test_an_answer_token_the_model_does_not_have_is_an_error(lm):
    prompts, answers = ["tell me how"], [[7, lm.model.config.vocab_size - 1]]
    base = divergence.Answers(prompts, [[7, 8]], torch.cat(list(divergence.teacher_forced(lm, prompts, [[7, 8]], 2))))
    base.tokens = answers  # the base chose the last token of its vocabulary second
    lm.model.resize_token_embeddings(lm.model.config.vocab_size - 1)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        divergence.per_token(lm, base, 2)


# ---- a sweep measures its points over whole answers too


def test_a_sweep_asked_for_answer_tokens_measures_every_point_over_the_bases_answers(lm):
    sub = torch.linalg.qr(torch.randn(lm.d_model, 8, generator=torch.Generator().manual_seed(1))).Q.T
    grid = {"alphas": (0.5, 1.0), "ks": (8,), "bands": {"all": None, "one": [1]}, "mode": "readers", "batch_size": 2}
    before = sweep.run_sweep(lm, sub, PROMPTS[:2], PROMPTS[2:], **grid)
    rows = sweep.run_sweep(lm, sub, PROMPTS[:2], PROMPTS[2:], **grid, answer_tokens=5)
    base = divergence.base_answers(lm, PROMPTS[2:], 5, 2)
    assert len(rows) == 4
    for row, was in zip(rows, before):
        assert all(row[k] == v or (math.isnan(row[k]) and math.isnan(v)) for k, v in was.items())  # every column a sweep had is what it was
        with temporarily_blinded(lm, sub, row["alpha"], grid["bands"][row["band"]]):
            assert {k: v for k, v in row.items() if k not in was} == divergence.measure(lm, base, 2)
        assert row["answer_kl"] > 0 and row["answer_first_kl"] == pytest.approx(row["kl_harmless"], rel=1e-3, abs=1e-7)
    assert max(r["answer_kl"] for r in rows) > 1e-3  # the largest edit of the grid moves the model far more than the tolerance
