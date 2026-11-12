"""Full answers to held-out harmful prompts, kept on disk so they can be judged later.

The files outlive the weights: a checkpoint that is scored and then deleted is still judged
from what it wrote here. They hold model output on harmful prompts, so they stay local
(``runs/`` is gitignored).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from . import data
from .generate import generate
from .models import LM
from .refusal import refusal_rate
from .sweep import read_rows

THOUGHT_OPEN = ("<think>", "<|begin_of_thought|>")
THOUGHT_CLOSE = ("</think>", "<|end_of_thought|>")
WRAPPERS = ("<|begin_of_solution|>", "<|end_of_solution|>")


def answer(completion: str) -> str:
    """The part of a completion a user is meant to read: what follows the reasoning, if any."""
    for tag in THOUGHT_CLOSE:
        if tag in completion:
            completion = completion.rsplit(tag, 1)[-1]
            for w in WRAPPERS:
                completion = completion.replace(w, "")
            return completion.strip()
    return completion


def cut_off(completion: str) -> bool:
    """Reasoning was opened and never closed, so there is no answer to judge."""
    return any(t in completion for t in THOUGHT_OPEN) and not any(t in completion for t in THOUGHT_CLOSE)


def with_think_budget(
    gen: Callable[[int], list[str]], max_new_tokens: int, think_tokens: int | None = None, tolerance: float = 0.2
) -> tuple[list[str], int]:
    """Completions from ``gen(max_new_tokens)`` and the budget they were made with. If more
    than ``tolerance`` of them stop inside their reasoning, they are made again with
    ``think_tokens``: a reasoning model needs the room, and nothing else should pay for it."""
    comps = gen(max_new_tokens)
    if think_tokens and think_tokens > max_new_tokens and sum(map(cut_off, comps)) > tolerance * len(comps):
        return gen(think_tokens), think_tokens
    return comps, max_new_tokens


def prompts(n: int = 100) -> list[str]:
    return data.sample(data.jailbreakbench(), n)


def path(out: str | Path, repo: str) -> Path:
    """Where the completions of ``repo`` live, next to the results file ``out``."""
    return Path(out).parent / "completions" / (repo.replace("/", "__") + ".jsonl")


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def record(
    lm: LM,
    repo: str,
    instructions: list[str],
    path: Path,
    template: str,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    think_tokens: int | None = None,
) -> int:
    """Generate ``repo``'s answers with the loaded ``lm`` and write them to ``path``. Returns
    the token budget used. The longer reasoning pass runs at half the batch size."""
    comps, used = with_think_budget(
        lambda n: generate(lm, instructions, n, batch_size if n == max_new_tokens else max(1, batch_size // 2)),
        max_new_tokens,
        think_tokens,
    )
    save([{"repo": repo, "dataset": "jailbreakbench", "tokenizer": template, "max_new_tokens": used,
           "prompt": p, "completion": c} for p, c in zip(instructions, comps)], path)
    return used


def rates(comps: list[dict], tag: str) -> dict:
    """What one checkpoint's judged answers come to: the columns of its results row that the
    judge ``tag`` is responsible for, and the rate both judges agree on once both have run."""
    n = len(comps)
    row = {
        "dataset": "jailbreakbench", "n": n, "tokenizer": comps[0]["tokenizer"],
        "max_new_tokens": comps[0]["max_new_tokens"],
        "substring_refusal": refusal_rate([answer(c["completion"]) for c in comps]),
        "think_unfinished": sum(cut_off(c["completion"]) for c in comps) / n,
        f"{tag}_harmful": sum(bool(c[f"{tag}_harmful"]) for c in comps) / n,
        f"{tag}_unparsed": sum(c[f"{tag}_harmful"] is None for c in comps) / n,
    }
    if tag == "wildguard":
        row["wildguard_refusal"] = sum(bool(c["wildguard_refusal"]) for c in comps) / n
    if all("wildguard_harmful" in c and "llamaguard_harmful" in c for c in comps):
        row["both_harmful"] = sum(bool(c["wildguard_harmful"]) and bool(c["llamaguard_harmful"]) for c in comps) / n
    return row


def judge_all(
    jobs: dict[str, dict], judge, out: str | Path, batch_size: int = 8
) -> Iterator[tuple[dict, dict | None]]:
    """Have ``judge`` label the saved answers of every job, yielding ``(job, row)`` as each
    checkpoint finishes. The row is its line of the results file ``out``, merged over what an
    earlier judge wrote there, or ``None`` for a checkpoint with no saved answers.

    Verdicts go into the answer files and rows into ``out`` checkpoint by checkpoint, so a run
    that is stopped keeps what it finished, and a rerun judges only what is missing."""
    out = Path(out)
    tag = judge.kind
    merged = {r["repo"]: r for r in read_rows(out)} if out.exists() else {}
    for job in jobs.values():
        file = path(out, job["repo"])
        if not file.exists():
            yield job, None
            continue
        comps = read_rows(file)
        if any(f"{tag}_harmful" not in c for c in comps):
            verdicts = judge([c["prompt"] for c in comps], [answer(c["completion"]) for c in comps], batch_size)
            for c, v in zip(comps, verdicts):
                c[f"{tag}_refusal"], c[f"{tag}_harmful"] = v.refusal, v.harmful_response
            save(comps, file)
        row = merged[job["repo"]] = merged.get(job["repo"], {}) | job | rates(comps, tag)
        save(list(merged.values()), out)
        yield job, row
