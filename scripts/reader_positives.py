"""Score a sweep's reader-side edits the way reader_weights.py scores an uploaded checkpoint.

    uv run python scripts/reader_positives.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --sweeps runs/reader_side --readers runs/detect/readers/qwen25.jsonl --scratch /tmp/reader_edits

sweep_frontier.py --mode readers measures what a reader-side edit does to behaviour, with the
edit made in memory. reader_weights.py reads the reading matrices of uploaded checkpoints.
This script joins the two. Every reader-side point of <--sweeps>/<model>/frontier.jsonl is
made again from the base's files (obliterated.edits.blind_readers_on_disk: the matrices the
sweep had in memory, in the precision the base stores them), read against the base by the
code that reads a checkpoint (obliterated.record.read_readers) and given the same reader_*
fields, in <--sweeps>/<model>/reader_scores.jsonl. The sweep's points can then be set against
thresholds taken from the benign fine-tunes in --readers.

The base is read at the revision, and scored against the refusal directions, that the reader
pass used for this base's fine-tunes in --readers. The script refuses to run if the base's
weight files on disk are not those, if the records disagree about the directions, or if a
band's layers no longer come out as the sweep edited them. Nothing is downloaded and no model
is loaded. Only reading matrices are written to --scratch, one point at a time, and removed
as soon as they are scored: nothing a model could be loaded from, and nothing kept. What the
scores were made from goes in reader_scores.json. A second run scores only what is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open

from obliterated import record, sweep
from obliterated.edits import EditSpec, blind_readers_on_disk
from obliterated.registry import snapshot_folder

N_FACTORS = 8  # as reader_weights.py
CARRIED = ("model", "tag", "mode", "alpha", "k", "band", "n_layers_edited", "bypass_rate", "kl_harmless")


def reference(readers: Path, model: str) -> tuple[str, torch.Tensor, dict]:
    """What the reader pass used for the fine-tunes of ``model`` in ``readers``: one of them
    by name, the base's refusal directions and that fine-tune's record."""
    repos = [r["repo"] for r in sweep.read_rows(readers) if r["base"] == model and r["repo"] != model]
    if not repos:
        raise SystemExit(f"{readers} has no fine-tune of {model}: there are no directions to score against")
    kept = []
    for repo in repos:
        with safe_open(record.tensors_path(readers, repo), "pt") as f:
            kept.append(f.get_tensor("reference.directions"))
    if not all(torch.equal(k, kept[0]) for k in kept):
        raise SystemExit(f"the records beside {readers} disagree about the refusal directions of {model}")
    return repos[0], kept[0], json.loads(record.provenance_path(readers, repos[0]).read_text())


def base_folder(model: str, made: dict) -> Path:
    """The base's snapshot, if it is on disk with the weight files the reader pass read."""
    folder = snapshot_folder(model, made["base_revision"])
    if not folder.exists() or record._weights(record.snapshot_info(folder)["files"]) != record._weights(made["base_files"]):
        raise SystemExit(f"{model} at {made['base_revision']} is not on disk with the weight files the fine-tunes were read against")
    return folder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the swept base")
    ap.add_argument("--sweeps", default="runs/reader_side", help="folder of sweep_frontier.py --mode readers outputs")
    ap.add_argument("--readers", required=True, help="results file of reader_weights.py holding fine-tunes of the base")
    ap.add_argument("--scratch", required=True, help="where the edited reading matrices are written while they are read")
    ap.add_argument("--subspace", default=None, help="the sweep's subspace file, if its folder holds more than one")
    args = ap.parse_args()
    versions = record.code_versions(Path(__file__))  # now: the sources may be edited while this runs

    folder = Path(args.sweeps) / args.model.replace("/", "__")
    out, scratch = folder / "reader_scores.jsonl", Path(args.scratch)
    points = [r for r in sweep.read_rows(folder / "frontier.jsonl") if r["mode"] == "readers"]
    found = [Path(args.subspace)] if args.subspace else sorted(folder.glob("subspace_*.json"))
    if len(found) != 1:
        raise SystemExit(f"{len(found)} subspace files in {folder}: name the sweep's with --subspace")
    spec = EditSpec.load(found[0])
    subspace = spec.vector.reshape(-1, spec.vector.shape[-1])

    repo, directions, made = reference(Path(args.readers), args.model)
    base_dir = base_folder(args.model, made)
    n_layers = json.loads((base_dir / "config.json").read_text())["num_hidden_layers"]
    bands = sweep.layer_bands(n_layers, spec.source_layer)
    for p in points:  # all of them before any is scored
        if p["band"] not in bands or p["k"] > len(subspace):
            raise SystemExit(f"cannot make the edit again: band {p['band']!r}, k={p['k']}")
        layers = bands[p["band"]]
        if p["n_layers_edited"] != (n_layers if layers is None else len(layers)):
            raise SystemExit(f"band {p['band']!r} is {n_layers if layers is None else len(layers)} layers here "
                             f"and was {p['n_layers_edited']} in the sweep")

    key = lambda r: (r["alpha"], r["k"], r["band"])
    done = {key(r) for r in sweep.read_rows(out)} if out.exists() else set()
    out.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "base_revision": made["base_revision"], "base_files": made["base_files"],
        "directions_from": repo, "readers": str(args.readers), "sweep": str(folder / "frontier.jsonl"),
        "subspace": str(found[0]), "subspace_sha256": hashlib.sha256(found[0].read_bytes()).hexdigest(),
        "n_factors": N_FACTORS, "scored_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions,
    }, indent=1))
    todo = [p for p in points if key(p) not in done]
    print(f"{args.model}: {len(points)} reader-side points, {len(todo)} to score; directions as for {repo}", flush=True)
    for i, p in enumerate(todo):
        started, edit = time.time(), scratch / f"{args.model.replace('/', '__')}_{i}"
        try:
            blind_readers_on_disk(base_dir, edit, subspace[: p["k"]], p["alpha"], bands[p["band"]])
            read = record.read_readers(base_dir, edit, directions, N_FACTORS)
        finally:
            shutil.rmtree(edit, ignore_errors=True)
        if read["missing"]:
            raise SystemExit(f"the edit a={p['alpha']} k={p['k']} {p['band']} could not be read: {read['missing'][:3]}")
        row = {**{k: p[k] for k in CARRIED if k in p}, **read["fields"]}
        sweep.write_rows([row], out)
        print(f"   [{i + 1}/{len(todo)}] a={p['alpha']:<5} k={p['k']} {p['band']:<9} bypass {p['bypass_rate']:.2f} | "
              f"{row['reader_n_changed']} layer readers changed; alignment {row['reader_refusal_alignment']:.3f}, "
              f"read drop {row['reader_read_drop']:.3f} (median {row['reader_read_drop_med']:.3f}), "
              f"head {row['reader_head_read_drop']:.3f}; {time.time() - started:.0f}s", flush=True)
    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()


if __name__ == "__main__":
    main()
