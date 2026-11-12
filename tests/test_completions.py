import pytest

from obliterated import completions, sweep
from obliterated.judge import Verdict


@pytest.mark.parametrize(
    ("completion", "want"),
    [
        ("Sure, here it is.", "Sure, here it is."),
        ("<think>plan the reply</think>\n\nI can't help with that.", "I can't help with that."),
        ("<|begin_of_thought|>plan<|end_of_thought|><|begin_of_solution|>Step one<|end_of_solution|>", "Step one"),
        ("<think>first</think>draft<think>second</think>final", "final"),
    ],
)
def test_answer_is_what_follows_the_reasoning(completion, want):
    assert completions.answer(completion) == want


@pytest.mark.parametrize(
    ("completion", "want"),
    [
        ("<think>still reasoning when the budget ran out", True),
        ("<|begin_of_thought|>still reasoning", True),
        ("<think>done</think>answer", False),
        ("no reasoning at all", False),
    ],
)
def test_cut_off_means_reasoning_opened_and_never_closed(completion, want):
    assert completions.cut_off(completion) is want


def _reasoner(budgets: list[int]):
    """Stands in for a reasoning model: it only reaches its answer with 3072 tokens."""
    def gen(max_new_tokens: int) -> list[str]:
        budgets.append(max_new_tokens)
        return ["<think>a</think>b"] * 4 if max_new_tokens >= 3072 else ["<think>a"] * 3 + ["<think>a</think>b"]
    return gen


def test_think_budget_regenerates_when_reasoning_was_cut_off():
    budgets: list[int] = []
    comps, used = completions.with_think_budget(_reasoner(budgets), 512, think_tokens=3072)
    assert (comps, used) == (["<think>a</think>b"] * 4, 3072)
    assert budgets == [512, 3072]


def test_think_budget_is_not_spent_on_a_model_that_finished():
    budgets: list[int] = []
    comps, used = completions.with_think_budget(lambda n: budgets.append(n) or ["plain answer"] * 4, 512, think_tokens=3072)
    assert (comps, used) == (["plain answer"] * 4, 512)
    assert budgets == [512]


def test_think_budget_tolerates_a_few_cut_off_answers():
    one_in_ten = ["<think>a"] + ["<think>a</think>b"] * 9
    assert completions.with_think_budget(lambda n: one_in_ten, 512, think_tokens=3072) == (one_in_ten, 512)


def test_no_think_budget_means_one_pass():
    cut = ["<think>a"] * 4
    assert completions.with_think_budget(lambda n: cut, 512, think_tokens=None) == (cut, 512)


class _Judge:
    """Stands in for an 8B safety classifier. It calls every answer harmful, and can be made to
    stop the run on its nth checkpoint, the way a killed process would."""

    kind = "llamaguard"

    def __init__(self, stops_on: int | None = None):
        self.seen: list[list[str]] = []
        self.stops_on = stops_on

    def __call__(self, prompts, responses, batch_size=8):
        if len(self.seen) + 1 == self.stops_on:
            raise KeyboardInterrupt
        self.seen.append(list(responses))
        return [Verdict(None, True, "unsafe") for _ in prompts]


def _job(repo: str, base: str = "a/base", label: str = "benign_finetune") -> dict:
    return {"repo": repo, "base": base, "label": label, "bypass_rate": 0.5}


def _answers(out, repo: str, texts: list[str], **verdicts) -> None:
    completions.save(
        [{"repo": repo, "dataset": "jailbreakbench", "tokenizer": "base", "max_new_tokens": 512,
          "prompt": f"prompt {i}", "completion": t} | {k: v[i] for k, v in verdicts.items()}
         for i, t in enumerate(texts)],
        completions.path(out, repo),
    )


def test_a_judging_run_that_is_stopped_keeps_the_rows_it_finished(tmp_path):
    out = tmp_path / "judged.jsonl"
    jobs = {r: _job(r) for r in ("a/base", "a/tune")}
    for repo in jobs:
        _answers(out, repo, ["Sure, step one", "Sure, step two"])
    with pytest.raises(KeyboardInterrupt):
        list(completions.judge_all(jobs, _Judge(stops_on=2), out))
    rows = sweep.read_rows(out)
    assert [(r["repo"], r["llamaguard_harmful"]) for r in rows] == [("a/base", 1.0)]


def test_a_rerun_judges_only_what_is_missing_and_still_reports_every_row(tmp_path):
    out = tmp_path / "judged.jsonl"
    jobs = {r: _job(r) for r in ("a/base", "a/tune")}
    _answers(out, "a/base", ["I can't help with that."], llamaguard_refusal=[None], llamaguard_harmful=[False])
    _answers(out, "a/tune", ["Sure, step one"])
    judge = _Judge()
    done = list(completions.judge_all(jobs, judge, out))
    assert judge.seen == [["Sure, step one"]]
    assert [(job["repo"], row["llamaguard_harmful"]) for job, row in done] == [("a/base", 0.0), ("a/tune", 1.0)]
    assert [r["repo"] for r in sweep.read_rows(out)] == ["a/base", "a/tune"]


def test_only_the_answer_after_the_reasoning_is_judged(tmp_path):
    out = tmp_path / "judged.jsonl"
    _answers(out, "a/tune", ["<think>they want steps</think>I can't help with that."])
    judge = _Judge()
    list(completions.judge_all({"a/tune": _job("a/tune")}, judge, out))
    assert judge.seen == [["I can't help with that."]]


def test_a_second_judge_adds_the_rate_both_agree_on_and_keeps_the_first_judges_columns(tmp_path):
    out = tmp_path / "judged.jsonl"
    _answers(out, "a/tune", ["Sure, step one", "Voici la traduction"],
             wildguard_refusal=[False, False], wildguard_harmful=[True, False])
    completions.save([_job("a/tune") | {"wildguard_harmful": 0.5, "wildguard_refusal": 0.0}], out)
    [(_, row)] = completions.judge_all({"a/tune": _job("a/tune")}, _Judge(), out)
    assert (row["wildguard_harmful"], row["llamaguard_harmful"], row["both_harmful"]) == (0.5, 1.0, 0.5)
    assert row["wildguard_refusal"] == 0.0
    assert sweep.read_rows(out) == [row]


def test_a_checkpoint_with_no_saved_answers_is_reported_and_not_judged(tmp_path):
    out = tmp_path / "judged.jsonl"
    judge = _Judge()
    assert list(completions.judge_all({"a/gone": _job("a/gone")}, judge, out)) == [(_job("a/gone"), None)]
    assert judge.seen == []
