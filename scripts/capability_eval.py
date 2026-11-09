"""MMLU / GSM8K / WMDP-Cyber through lm-evaluation-harness, on the clean or the edited model.

    uv run --extra eval python scripts/capability_eval.py --model Qwen/Qwen2.5-1.5B-Instruct
    uv run --extra eval python scripts/capability_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --edit runs/Qwen__Qwen2.5-1.5B-Instruct/edit.json --limit 200

The edit is applied in memory and handed to lm-eval as a model object, so no edited
checkpoint is ever written to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from obliterated.edits import EditSpec
from obliterated.models import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--edit", default=None, help="EditSpec json; omit for the clean model")
    ap.add_argument("--alpha", type=float, default=None, help="override the edit's strength")
    ap.add_argument("--config", default="configs/lm_eval.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    cfg = yaml.safe_load(Path(args.config).read_text())
    lm = load(args.model, device=args.device)
    tag = "clean"
    if args.edit:
        spec = EditSpec.load(args.edit)
        if args.alpha is not None:
            spec.alpha = args.alpha
        spec.mode = "weights"
        spec.apply(lm)
        tag = f"edited_a{spec.alpha:g}"

    hflm = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer, batch_size=cfg.get("batch_size", "auto"))
    results = {}
    for task in cfg["tasks"]:
        r = lm_eval.simple_evaluate(
            model=hflm,
            tasks=[task["name"]],
            num_fewshot=task.get("num_fewshot"),
            limit=args.limit or cfg.get("limit"),
        )
        results[task["name"]] = r["results"]
        print(task["name"], json.dumps(r["results"].get(task["name"], {}), indent=1))

    out = Path(args.out) / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"capability_{tag}.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
