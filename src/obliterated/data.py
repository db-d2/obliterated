"""Loaders for the public prompt sets used to extract and evaluate refusal directions.

Every set is downloaded once from its official public source and cached under ``data/raw``.
Only instructions are returned (lists of strings); nothing here prints prompt text, so logs stay
free of benchmark content.

Contrast sets (direction extraction):
  - ``arditi``: the exact harmful/harmless splits from Arditi et al. (NeurIPS 2024), built from
    AdvBench, MaliciousInstruct, TDC2023 and HarmBench (harmful) and Alpaca (harmless).
  - ``advbench``, ``harmbench`` vs ``alpaca``: the raw sources, for custom splits.

Held-out evaluation:
  - ``jailbreakbench`` (100 harmful behaviors, plus 100 benign ones) and ``strongreject``.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import urllib.request
import warnings
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

_ARDITI = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{name}.json"
SOURCES = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    "jailbreakbench": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
    "jailbreakbench_benign": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv",
}


def _fetch(name: str, url: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(url).suffix or ".txt"
    path = DATA_DIR / f"{name}{suffix}"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8")


def _csv_column(text: str, column: str, where: dict[str, str] | None = None) -> list[str]:
    rows = csv.DictReader(io.StringIO(text))
    out = []
    for row in rows:
        if where and any(row.get(k) != v for k, v in where.items()):
            continue
        out.append(row[column].strip())
    return out


def arditi_split(kind: str, split: str) -> list[str]:
    """kind in {harmful, harmless}; split in {train, val, test}."""
    name = f"{kind}_{split}"
    data = json.loads(_fetch(f"arditi_{name}", _ARDITI.format(name=name)))
    return [d["instruction"] for d in data]


def advbench() -> list[str]:
    return _csv_column(_fetch("advbench", SOURCES["advbench"]), "goal")


def harmbench(functional_category: str = "standard") -> list[str]:
    """Standard behaviors only by default (contextual and copyright ones need extra context)."""
    text = _fetch("harmbench", SOURCES["harmbench"])
    return _csv_column(text, "Behavior", where={"FunctionalCategory": functional_category})


def alpaca(no_input_only: bool = True) -> list[str]:
    data = json.loads(_fetch("alpaca", SOURCES["alpaca"]))
    return [d["instruction"] for d in data if not (no_input_only and d["input"].strip())]


def strongreject() -> list[str]:
    return _csv_column(_fetch("strongreject", SOURCES["strongreject"]), "forbidden_prompt")


def jailbreakbench(benign: bool = False) -> list[str]:
    key = "jailbreakbench_benign" if benign else "jailbreakbench"
    return _csv_column(_fetch(key, SOURCES[key]), "Goal")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def drop_overlap(prompts: list[str], *exclude: list[str]) -> list[str]:
    """Remove prompts that also appear (after normalization) in any of the excluded sets."""
    banned = {_norm(p) for ex in exclude for p in ex}
    return [p for p in prompts if _norm(p) not in banned]


def sample(prompts: list[str], n: int | None, seed: int = 0) -> list[str]:
    if n is None or n >= len(prompts):
        return list(prompts)
    return random.Random(seed).sample(prompts, n)


def contrast_sets(source: str = "arditi", n_train: int = 128, n_val: int = 32, seed: int = 0):
    """Return (harmful_train, harmless_train, harmful_val, harmless_val)."""
    if source == "arditi":
        return (
            sample(arditi_split("harmful", "train"), n_train, seed),
            sample(arditi_split("harmless", "train"), n_train, seed),
            sample(arditi_split("harmful", "val"), n_val, seed),
            sample(arditi_split("harmless", "val"), n_val, seed),
        )
    harmful = {"advbench": advbench, "harmbench": harmbench}[source]()
    harmful = drop_overlap(harmful, jailbreakbench(), strongreject())
    harmless = alpaca()
    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    return (
        harmful[:n_train],
        harmless[:n_train],
        harmful[n_train : n_train + n_val],
        harmless[n_train : n_train + n_val],
    )


def eval_sets(n_harmless: int = 100, seed: int = 0) -> dict[str, list[str]]:
    """Held-out sets. Harmless prompts come from Arditi's Alpaca test split. A set whose source
    cannot be reached (JailbreakBench is hosted on Hugging Face) is skipped with a warning."""
    out = {}
    for name, fn in (("jailbreakbench", jailbreakbench), ("strongreject", strongreject)):
        try:
            out[name] = fn()
        except OSError as e:
            warnings.warn(f"skipping {name}: {e}", stacklevel=2)
    out["harmless"] = sample(arditi_split("harmless", "test"), n_harmless, seed)
    return out


def fetch_all() -> dict[str, int]:
    """Download everything and return the size of each set."""
    sizes = {}
    for kind in ("harmful", "harmless"):
        for split in ("train", "val", "test"):
            sizes[f"arditi_{kind}_{split}"] = len(arditi_split(kind, split))
    sizes["advbench"] = len(advbench())
    sizes["harmbench_standard"] = len(harmbench())
    sizes["alpaca_no_input"] = len(alpaca())
    sizes["strongreject"] = len(strongreject())
    for benign in (False, True):
        key = "jailbreakbench_benign" if benign else "jailbreakbench"
        try:
            sizes[key] = len(jailbreakbench(benign))
        except OSError as e:  # HF can be blocked on some networks
            sizes[key] = f"unavailable: {e}"
    return sizes
