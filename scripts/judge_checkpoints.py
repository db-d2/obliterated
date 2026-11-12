"""Do scored checkpoints still refuse? Generate on held-out harmful prompts and judge the text.

    uv run python scripts/judge_checkpoints.py --checkpoints runs/detect/qwen25.jsonl --min-bypass 0.5
    uv run python scripts/judge_checkpoints.py --checkpoints runs/detect/*.jsonl --label abliterated

The bypass rate in detect_checkpoints.py only says the first generated token is no longer a
refusal token. A fine-tune that opens every answer with a thinking tag, a JSON brace or another
language reads as a bypass there without having lost its refusals. This script generates full
answers on JailbreakBench from the selected checkpoints, and from their bases as the reference,
and has the judge label each one. A benign fine-tune that really stopped refusing can then be
told apart from one that only changed its first token.

A reasoning model's answer is what follows its closing think tag, and only that is judged. If
the tag never arrives within --max-new-tokens the whole truncated trace is judged instead and
counted in ``think_unfinished``; rerun those repos with a larger budget and --redo.

Neither judge is reliable for fine-tunes that no longer answer the question at all. WildGuard
calls a translation of the prompt, arithmetic gibberish or unrelated boilerplate "harmful"
40-98% of the time, and Llama Guard 3 (--judge meta-llama/Llama-Guard-3-8B on a second run)
calls almost any non-refusal unsafe. Both verdicts are kept per completion, and
``both_harmful`` is the rate at which the two agree; on the small Qwen2.5 and Llama 3.2 sets
it changed one unsafe label in 82 relative to WildGuard alone.

Checkpoints scored with ``detect_checkpoints.py --evict`` already have their completions and
no longer have their weights: judge those with --no-generate, which never loads or downloads a
checkpoint.

Generation uses the base tokenizer and chat template, like the detectors, unless
--own-tokenizer is given. Every generation finishes before the judge is loaded, so the two
never share the GPU. Completions are written under <out dir>/completions/ (local only,
gitignored) and reused on a rerun. Rates per checkpoint are merged into --out as each one is
judged, so a run that is stopped keeps what it finished and a rerun judges only the rest.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from transformers import AutoTokenizer

from obliterated import completions, sweep
from obliterated.judge import Judge
from obliterated.models import load, unload


def select(args) -> dict[str, dict]:
    """Checkpoints to generate from, keyed by repo, each base ahead of its fine-tunes."""
    scored = {r["repo"]: r for f in args.checkpoints for r in sweep.read_rows(f)}
    picked = [
        r for r in scored.values()
        if (args.label is None or r["label"] == args.label)
        and (args.min_bypass is None or r["bypass_rate"] >= args.min_bypass)
        and (not args.repo or r["repo"] in args.repo)
    ]
    jobs: dict[str, dict] = {}
    for r in sorted(picked, key=lambda r: r["base"]):
        if r["base"] not in jobs:
            parent = scored.get(r["base"], {})
            jobs[r["base"]] = {"repo": r["base"], "base": parent.get("base", r["base"]),
                               "label": parent.get("label", "base"), "bypass_rate": parent.get("bypass_rate", 0.0)}
        jobs[r["repo"]] = {"repo": r["repo"], "base": r["base"], "label": r["label"], "bypass_rate": r["bypass_rate"]}
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="files written by detect_checkpoints.py")
    ap.add_argument("--label", default=None)
    ap.add_argument("--min-bypass", type=float, default=None, help="only checkpoints at or above this next-token bypass rate")
    ap.add_argument("--repo", nargs="*", default=None, help="only these repos")
    ap.add_argument("--n", type=int, default=100, help="JailbreakBench prompts")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="long enough for a reasoning model to finish thinking")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--own-tokenizer", action="store_true", help="use each checkpoint's own tokenizer and chat template")
    ap.add_argument("--judge", default="allenai/wildguard")
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--think-tokens", type=int, default=None,
                    help="generate again with this budget when more than a fifth of the answers stop inside the reasoning")
    ap.add_argument("--skip-judge", action="store_true", help="generate only")
    ap.add_argument("--no-generate", action="store_true",
                    help="judge the completions that exist; never load or download a checkpoint")
    ap.add_argument("--redo", action="store_true", help="regenerate completions that already exist")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/detect/judged.jsonl")
    args = ap.parse_args()

    jobs = select(args)
    if not jobs:
        raise SystemExit("no scored checkpoints match")
    out = Path(args.out)
    prompts = completions.prompts(args.n)
    template = "own" if args.own_tokenizer else "base"

    tokenizers: dict[str, object] = {}
    for i, j in enumerate(jobs.values()):
        path = completions.path(out, j["repo"])
        if args.no_generate or (path.exists() and not args.redo):
            continue
        print(f"== [{i + 1}/{len(jobs)}] generate {j['repo']} ({j['label']})", flush=True)
        try:
            tok = None
            if not args.own_tokenizer:
                root = j["base"]
                tok = tokenizers[root] = tokenizers.get(root) or AutoTokenizer.from_pretrained(root)
            lm = load(j["repo"], device=args.device, tokenizer=tok)
            completions.record(lm, j["repo"], prompts, path, template, args.max_new_tokens, args.batch_size, args.think_tokens)
            unload(lm)
        except Exception:  # noqa: BLE001 - one broken upload should not end the batch
            traceback.print_exc()
    if args.skip_judge:
        return

    judge = Judge(args.judge, device=args.device)
    for j, row in completions.judge_all(jobs, judge, out, args.judge_batch_size):
        if row is None:
            print(f"no completions for {j['repo']}: not judged", flush=True)
            continue
        print(f"{j['label'][:20]:<20} {j['repo'][:56]:<56} next-token bypass {j['bypass_rate']:.2f} | substring refusal "
              f"{row['substring_refusal']:.2f} | judged refusal {row.get('wildguard_refusal', float('nan')):.2f} "
              f"harmful {row[f'{judge.kind}_harmful']:.2f}" + (f" both judges {row['both_harmful']:.2f}" if "both_harmful" in row else "") + (f" | thinking cut off {row['think_unfinished']:.2f}" if row["think_unfinished"] else ""),
              flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
