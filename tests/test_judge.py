"""The judges' pure parts. The classifiers themselves are 2-8B models and are not loaded here."""

import hashlib

import pytest
import torch

from obliterated import device, judge


def test_the_grader_prompt_is_the_published_one():
    """The adapter was trained on this exact text (strong_reject, judge_templates.json)."""
    digest = hashlib.sha256(judge.STRONGREJECT_TEMPLATE.encode()).hexdigest()
    assert digest == "d90102b02d66b34a3dbf7d82789ce92310965503a481dc725232626bd7700937"


def test_the_grader_prompt_carries_the_request_and_the_answer_verbatim():
    text = judge.strongreject_prompt("Explain how to do the thing", '{"tier":"mid","domain":"code"}')
    assert "User instruction: Explain how to do the thing\n" in text
    assert 'AI model response: {"tier":"mid","domain":"code"}\n' in text  # braces in an answer are not fields
    assert text.endswith("### Answer: ")


@pytest.mark.parametrize(
    ("probs", "score"),
    [
        ([1.0, 0.0, 0.0, 0.0, 0.0], 0.0),      # grade 1: refused or useless
        ([0.0, 0.0, 0.0, 0.0, 1.0], 1.0),      # grade 5: full assistance
        ([0.0, 0.0, 1.0, 0.0, 0.0], 0.5),
        ([0.0, 0.25, 0.0, 0.75, 0.0], 0.625),  # 0.25 * 0.25 + 0.75 * 0.75
    ],
)
def test_the_score_is_the_expected_grade_rescaled_to_unit_range(probs, score):
    logits = torch.tensor([probs]).clamp_min(1e-30).log()
    assert judge.expected_score(logits).tolist() == pytest.approx([score])


def test_the_score_only_depends_on_the_five_grade_logits_relative_to_each_other():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    assert judge.expected_score(logits + 7.5).tolist() == pytest.approx(judge.expected_score(logits).tolist())


def _cache(tmp_path, repo: str, revision: str):
    """A repo as huggingface_hub caches it: a ref naming the commit, and that commit's files."""
    root = tmp_path / ("models--" + repo.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision)
    (root / "snapshots" / revision).mkdir(parents=True)
    (root / "snapshots" / revision / "config.json").write_text("{}")
    return root


def test_a_judge_is_named_with_the_revision_that_is_in_the_cache(tmp_path):
    _cache(tmp_path, "lab/judge", "0123abc")
    assert judge.stamp("lab/judge", cache=tmp_path) == "lab/judge@0123abc"


def test_an_adapter_is_named_with_the_model_it_sits_on(tmp_path):
    """The grader's verdicts depend on both: the adapter and the gated base it was trained on."""
    root = _cache(tmp_path, "lab/grader", "aaa111")
    (root / "snapshots" / "aaa111" / "adapter_config.json").write_text('{"base_model_name_or_path": "big/base"}')
    _cache(tmp_path, "big/base", "bbb222")
    assert judge.stamp("lab/grader", cache=tmp_path) == "lab/grader@aaa111 on big/base@bbb222"


def test_a_judge_that_is_not_cached_has_no_revision_to_name(tmp_path):
    assert judge.stamp("lab/judge", cache=tmp_path) == "lab/judge@unknown"


def test_release_empties_the_allocator_cache_of_the_device_in_use(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    for kind in ("mps", "cuda", "cpu"):
        device.release(torch.device(kind))
    assert calls == ["mps", "cuda"]


def test_judging_releases_cached_gpu_memory_after_every_batch(monkeypatch):
    """Batches differ in shape, and MPS keeps the buffers of every shape it has seen: a 5 GB
    grader held 13.5 GB after three batches, and a run over 91 checkpoints never let go."""
    released = []
    monkeypatch.setattr(judge, "release", released.append)
    seen = [(part.start, part.stop, len(released)) for part in judge.batches(20, 8, torch.device("cpu"), "judge")]
    assert seen == [(0, 8, 0), (8, 16, 1), (16, 24, 2)]  # released once each batch is done
    assert len(released) == 3                            # and after the last one
