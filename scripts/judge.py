"""Judge saved completions and add judge-based refusal rates to summary.json.

    uv run python scripts/judge.py runs/Qwen__Qwen2.5-1.5B-Instruct
    uv run python scripts/judge.py runs/... --judge meta-llama/Llama-Guard-3-8B
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from obliterated.judge import Judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--judge", default="allenai/wildguard")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rows = [json.loads(line) for line in (run / "completions.jsonl").read_text().splitlines()]
    judge = Judge(args.judge, device=args.device)
    verdicts = judge([r["prompt"] for r in rows], [r["completion"] for r in rows], args.batch_size)

    tag = judge.kind
    counts = defaultdict(lambda: [0, 0, 0])  # refusals, harmful, n
    for r, v in zip(rows, verdicts):
        r[f"{tag}_refusal"], r[f"{tag}_harmful"] = v.refusal, v.harmful_response
        c = counts[(r["condition"], r["dataset"])]
        c[0] += bool(v.refusal)
        c[1] += bool(v.harmful_response)
        c[2] += 1

    with open(run / "completions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = json.loads((run / "summary.json").read_text())
    for (cond, ds), (ref, harm, n) in sorted(counts.items()):
        if judge.kind == "wildguard":
            summary[f"{cond}/{ds}/{tag}_refusal"] = ref / n
        summary[f"{cond}/{ds}/{tag}_harmful"] = harm / n
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if tag in k}, indent=2))


if __name__ == "__main__":
    main()
