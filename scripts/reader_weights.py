"""Score the matrices that read the residual stream, for checkpoints already scored by
detect_checkpoints.py.

    uv run python scripts/reader_weights.py --rows runs/detect/qwen25.jsonl --out runs/detect/readers/qwen25.jsonl
    uv run python scripts/reader_weights.py --rows runs/detect/llama31.jsonl --out runs/detect/readers/llama31.jsonl --evict
    uv run python scripts/reader_weights.py --check --rows runs/detect/llama31.jsonl --out runs/detect/readers/llama31.jsonl

The weight detectors of detect_checkpoints.py compare the matrices that write into the
residual stream. An edit can leave all of those alone and change the matrices that read the
stream instead (obliterated.edits.blind_readers), so this pass reads the other half of the
model: each layer's attention query, key and value projections and its MLP's gate and up
projections, and the output head. See obliterated.detect.scan_readers for what is measured.

It is a weights-only pass. No model is loaded and nothing is run: tensors are streamed from
the safetensors files a slice of rows at a time, on the CPU, so it can run beside a job that
has a model in memory. It takes each checkpoint's pinned revision, and the base's refusal
directions, from the record the main pass left under <--rows dir>/record/, and reads exactly
that revision; record.validate_readers then checks that the weight files are the same ones,
by name, size and content hash.

Per checkpoint it writes, beside --out: one row of scores (reader_*), spectra/<repo>.json (the
spectrum summaries of every changed reading matrix) and record/<repo>.safetensors with its
.json: the top singular values and input-side directions of each changed matrix (of the stored
matrix, and of the matrix composed with the norm in front of it), how strongly
every reading matrix of the base and of the checkpoint reads each refusal direction, the gain
of every norm in front of one (and the checkpoint's, where it differs), the directions
themselves, and what was read, at which revision, with which code.

--evict deletes a checkpoint from the Hugging Face cache once its reader record has been read
back and validated, but only if this pass downloaded it: a checkpoint that was on disk already
is someone else's to delete. The downloads this pass starts are listed in
<out dir>/downloaded.json before they begin, so an interrupted one is still recognised. Bases
are never deleted. The next checkpoint downloads while the current one is read (--prefetch).
--check validates every row in --out again without reading any weights.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import load_file

from obliterated import detect, record, sweep
from obliterated.registry import bases, evict, fetch, load_registry

N_FACTORS = 8


@functools.cache
def versions() -> dict:
    """Taken once per process: the sources on disk at the end of a long run need not be the
    ones that were loaded."""
    return record.code_versions(Path(__file__))


def snapshot_folder(repo: str, revision: str) -> Path:
    return Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo, repo_type="model") / "snapshots" / revision


def cached(repo: str, revision: str, files: list[dict]) -> bool:
    """Whether every weight file the main pass recorded for ``repo`` at ``revision`` is in the
    Hugging Face cache, at its recorded size. A download that stopped half way is not."""
    folder = snapshot_folder(repo, revision)
    weights = [f for f in files if f["name"].endswith(".safetensors")]
    return bool(weights) and all((folder / f["name"]).exists() and (folder / f["name"]).stat().st_size == f["bytes"]
                                 for f in weights)


def locate(repo: str, revision: str, files: list[dict]) -> Path:
    """The snapshot of ``repo`` at ``revision``: the cached one if it is complete (no network,
    so a repo since removed from the Hub can still be read), otherwise downloaded."""
    return snapshot_folder(repo, revision) if cached(repo, revision, files) else fetch(repo, revision=revision)


def pinned(main_out: Path, repo: str) -> dict:
    """What the main pass recorded about the weights it scored for ``repo``."""
    path = record.provenance_path(main_out, repo)
    if not path.exists():
        raise FileNotFoundError(f"{repo} has no record beside {main_out}: there is no pinned revision to read")
    return json.loads(path.read_text())


def score(main_row: dict, main_out: str | Path, out: str | Path) -> dict:
    """Read one checkpoint's reading matrices against its base's, write its spectra and record
    beside ``out`` and return its row."""
    main_out, out = Path(main_out), Path(out)
    repo, base = main_row["repo"], main_row["base"]
    main = pinned(main_out, repo)
    base_dir = locate(base, main["base_revision"], main["base_files"])
    cand_dir = locate(repo, main["revision"], main["files"])
    n_layers = json.loads((base_dir / "config.json").read_text())["num_hidden_layers"]
    kept = {"reference.directions": load_file(record.tensors_path(main_out, repo))["reference.directions"]}
    spectra, matrices, missing, gains = [], [], [], []
    for m in detect.scan_readers(base_dir, cand_dir, kept["reference.directions"].float(), N_FACTORS):
        if m.get("missing") or not m["finite"]:
            missing.append(m["name"])
            continue
        matrices.append(m["name"])
        kept[f"{m['name']}.read"] = torch.stack([m["read_base"], m["read_cand"]]).float()
        kept.setdefault(f"gain.{m['norm']}", m["gain"])
        if m["gain_changed"]:
            gains.append(m["name"])
            kept[f"gain_cand.{m['norm']}"] = m["gain_cand"]
        if m["changed"]:
            kept[f"{m['name']}.s"], kept[f"{m['name']}.u"] = m["s"], m["u"].half()
        if "eu" in m:
            kept[f"{m['name']}.es"], kept[f"{m['name']}.eu"] = m["es"], m["eu"].half()
            spectra.append({"name": m["name"], "stored": m.get("summary"), "effective": m["summary_eff"]})
    mid = detect.mid_layers(n_layers)
    in_layers = [s for s in spectra if detect.layer_of(s["name"]) is not None]
    fields = detect.scores_from_spectra([s["stored"] for s in in_layers if s["stored"]], mid)
    fields.pop("per_matrix")
    effective = detect.scores_from_spectra([s["effective"] for s in in_layers], mid)
    info, base_info = record.snapshot_info(cand_dir), record.snapshot_info(base_dir)
    row = {
        "repo": repo, "base": base, "label": main_row.get("label"), "family": main_row.get("family"),
        "params_b": main_row.get("params_b"), "revision": info["revision"],
        **{k.replace("weight_", "reader_", 1): v for k, v in fields.items()},
        **{k.replace("weight_", "reader_eff_", 1): v for k, v in effective.items() if k.endswith("_energy_w")},
        **record.reader_scores(kept),  # from what is kept, at the precision it is kept in
        "reader_n_gain_changed": len(gains),
        "reader_head_changed": any(s["name"] == detect.HEAD and bool(s["stored"]) for s in spectra),
    }
    path = record.spectra_path(out, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spectra))
    record.write(out, repo, kept, {
        "repo": repo, "base": base, "label": main_row.get("label"),
        "revision": info["revision"], "files": info["files"],
        "base_revision": base_info["revision"], "base_files": base_info["files"],
        "main_record": str(record.tensors_path(main_out, repo)),
        "n_layers": n_layers, "n_factors": N_FACTORS, "matrices": matrices, "missing": missing, "gain_changed": gains,
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions(),
    })
    return row


def prefetch(repo: str, revision: str, files: list[dict]) -> threading.Thread:
    """Start downloading ``repo`` in the background. A failure here is left for the pass
    itself to hit and report."""
    def run() -> None:
        try:
            locate(repo, revision, files)
        except Exception:  # noqa: BLE001, S110
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="results file of detect_checkpoints.py; its record/ pins what to read")
    ap.add_argument("--out", required=True, help="results file of this pass; its spectra/ and record/ go beside it")
    ap.add_argument("--repo", nargs="+", default=None, help="only these checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many checkpoints")
    ap.add_argument("--cached-only", action="store_true", help="skip checkpoints whose weights are not on disk: download nothing")
    ap.add_argument("--redo", action="store_true", help="read checkpoints again that are already in --out")
    ap.add_argument("--evict", action="store_true",
                    help="delete each checkpoint this pass downloaded once its reader record validates; implies --prefetch")
    ap.add_argument("--prefetch", action="store_true", help="download the next checkpoint while the current one is read")
    ap.add_argument("--check", action="store_true", help="validate the reader record of every row in --out, then exit")
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="stop before a download if the cache disk has less free")
    args = ap.parse_args()
    main_out, out = Path(args.rows), Path(args.out)
    if args.check:
        bad = 0
        for r in sweep.read_rows(out):
            problems = record.validate_readers(r, out, main_out)
            bad += bool(problems)
            print(f"{'PROBLEMS' if problems else 'ok':<9}{r['repo']}" + "".join(f"\n    - {p}" for p in problems), flush=True)
        raise SystemExit(1 if bad else 0)
    if args.evict:
        args.prefetch = True
    versions()  # now, before hours of reading during which the sources may be edited

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.redo or not out.exists() else {r["repo"] for r in sweep.read_rows(out)}
    todo = [r for r in sweep.read_rows(main_out)
            if r["repo"] != r["base"] and r["repo"] not in done and (args.repo is None or r["repo"] in args.repo)]
    pins = {}
    for r in todo:
        try:
            pins[r["repo"]] = pinned(main_out, r["repo"])
        except FileNotFoundError as e:
            print(f"skipping {r['repo']}: {e}", flush=True)
    todo = [r for r in todo if r["repo"] in pins]
    if args.cached_only:
        todo = [r for r in todo if cached(r["repo"], pins[r["repo"]]["revision"], pins[r["repo"]]["files"])]
    todo = sorted(todo, key=lambda r: r["base"])[: args.limit]
    if not todo:
        raise SystemExit("nothing to read")

    keep = bases(load_registry())
    mine_path = out.parent / "downloaded.json"
    mine = set(json.loads(mine_path.read_text())) if mine_path.exists() else set()

    def claim(r: dict) -> None:
        """Note, before its download can start, that this pass is the one fetching ``r``."""
        pin = pins[r["repo"]]
        if r["repo"] not in mine and not cached(r["repo"], pin["revision"], pin["files"]):
            mine.add(r["repo"])
            mine_path.write_text(json.dumps(sorted(mine), indent=1))

    failed, kept_back = [], []
    arriving: threading.Thread | None = None
    for i, r in enumerate(todo):
        repo = r["repo"]
        print(f"== [{i + 1}/{len(todo)}] {repo} (base {r['base']}, {r.get('label')})", flush=True)
        free = shutil.disk_usage(constants.HF_HUB_CACHE).free / 1e9
        if free < args.min_free_gb:
            print(f"stopping: {free:.0f} GB free on the cache disk, below --min-free-gb {args.min_free_gb:.0f}")
            break
        if arriving is not None:
            arriving.join()  # this checkpoint's own download, started during the previous one
        claim(r)
        arriving = None
        if args.prefetch and i + 1 < len(todo):
            nxt = todo[i + 1]
            claim(nxt)
            arriving = prefetch(nxt["repo"], pins[nxt["repo"]]["revision"], pins[nxt["repo"]]["files"])
        started = time.time()
        try:
            row = score(r, main_out, out)
        except Exception as e:  # noqa: BLE001 - one broken or vanished upload should not end a long batch
            traceback.print_exc()
            failed.append((repo, f"{type(e).__name__}: {e}"))
            problems = [f"reading failed: {type(e).__name__}: {e}"]
        else:
            sweep.write_rows([row], out)
            problems = record.validate_readers(sweep.read_rows(out)[-1], out, main_out)  # what is on disk, not in memory
            print(f"   {time.time() - started:.0f}s; {row['reader_n_changed']} layer readers changed, "
                  f"head {'changed' if row['reader_head_changed'] else 'unchanged'}; "
                  f"top-8 energy {row['reader_top8_energy_w']:.3f}, alignment {row['reader_refusal_alignment_any']:.3f}, "
                  f"read drop {row['reader_read_drop']:.3f}", flush=True)
            print(f"   reader record of {repo}: " + ("validated" if not problems else "".join(f"\n     - {p}" for p in problems)), flush=True)
        if problems:
            kept_back.append(repo)
        if not args.evict:
            continue
        if problems:  # nothing is deleted until what replaces it has been checked
            print(f"   kept on disk: {repo} did not validate", flush=True)
        elif repo in keep:
            print(f"   kept on disk: {repo} is the base of another row", flush=True)
        elif repo not in mine:
            print(f"   kept on disk: {repo} was there before this pass", flush=True)
        else:
            print(f"   deleted {repo} from the cache: {evict(repo, keep) / 1e9:.1f} GB freed", flush=True)

    for repo, err in failed:
        print(f"FAILED {repo}: {err}")
    if kept_back:
        print(f"{len(kept_back)} checkpoints failed or have a reader record that did not validate:")
        print("".join(f"  {repo}\n" for repo in kept_back), end="")


if __name__ == "__main__":
    main()
