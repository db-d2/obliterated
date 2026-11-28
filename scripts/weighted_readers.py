"""Whether the reading matrices a checkpoint changed share an input direction, with each
change weighed by its base's stream, for checkpoints already read by reader_weights.py.

    uv run python scripts/weighted_readers.py --rows runs/detect/readers_span/qwen25.jsonl --out runs/detect/weighted/qwen25.jsonl
    uv run python scripts/weighted_readers.py --rows runs/detect/readers_span/qwen25.jsonl --out runs/detect/weighted/qwen25.jsonl --measure
    uv run python scripts/weighted_readers.py --check --rows runs/detect/readers_span/qwen25.jsonl --out runs/detect/weighted/qwen25.jsonl
    uv run python scripts/weighted_readers.py --rows runs/detect/readers_span/llama31.jsonl --out runs/detect/weighted/llama31.jsonl --evict --unweighed
    uv run python scripts/weighted_readers.py --report runs/detect/weighted/qwen25.jsonl

The shared-direction score of the reader pass (obliterated.record.shared_scores) reads the
strongest factors of each matrix's change, strongest in the weights. Changes that are large in
the weights and lie along directions the stream has next to nothing in cost the model nothing
and take an edit's place among those factors. This pass orders the parts of each change by what
they do to the stream instead (obliterated.record.weighted_readers), which needs one thing the
weights do not give: how much the base's stream has along each direction where each matrix
reads it.

That is the detector's own measurement, one file per base under --stream. With --measure, a
base that has none is loaded and measured (obliterated.stream.measure) on --n harmful and --n
harmless prompts of Arditi's test split, each followed by --tokens tokens of the base's own
answer: prompts other than the train and val splits that directions are extracted from and
edits are checked on. Nothing a model wrote is kept, only sums over its activations. Without
--measure no model is loaded, and the fine-tunes of a base that has no file are left out.

Scoring is a weights-only pass like the reader pass: tensors are streamed from the safetensors
files on the CPU. It reads the revision that pass pinned (<--rows dir>/record/) and only what
is on disk with the files that pass read. Unless it is asked to, it downloads nothing and
deletes nothing. Per checkpoint it writes one row to --out and record/<repo>.safetensors with
its .json beside it: the first 128 directions of every changed reading matrix, in half
precision, the size of each of those parts and of the whole weighed change, and what was read,
with which measurement of the stream and which code. The row has the score with the first 8
of those directions and with all 128 (reader_weighted_shared_k8, _k128).

--fetch downloads a checkpoint that is not on disk, at the pinned revision; what comes down has
to be the files the reader pass read. --evict (which fetches) deletes a checkpoint from the
Hugging Face cache once what was kept of it has been read back and validated
(obliterated.record.validate_weighted), and only if this pass downloaded it: the downloads it
starts are listed in <--out>.downloaded.json before they begin, so an interrupted one is still
recognised. Bases are never fetched and never deleted. The next checkpoint downloads while the
current one is read. --unweighed also keeps the 128 largest factors of each change as it is in
the weights, which the reader pass kept 8 of: once the weights are gone they cannot be had
again, and their sizes are checked against that pass's. --check validates every row of --out
again, without reading any weights.

The two sizes are for use together. --report sets a threshold for each from the benign
fine-tunes in the files it is given (with --judged, those the grader still finds refusing: a
judged score under --margin above their base's) and lists every other checkpoint with the
sizes at which it is above the threshold: it is flagged if there is any. Two thresholds give a
benign fine-tune two to be above, so the report also holds each benign fine-tune out in turn
and counts how many are above a threshold that the others set.
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

from huggingface_hub import constants

from obliterated import data, record, stream, sweep
from obliterated.generate import generate
from obliterated.models import load, unload
from obliterated.registry import bases, evict, fetch, load_registry, snapshot_folder

N_KEPT = max(record.WEIGHTED_SIZES)
PARTS = ("wu", "ws", "wt")  # of each changed matrix: what its weighed change reads, the size of each part, and of the whole
OWN = ("es", "eu")  # with --unweighed: the sizes and directions of the change as it is in the weights


class LeftOut(Exception):
    """Why a checkpoint is not scored by this run."""


@functools.cache
def versions() -> dict:
    """Taken once per process: the sources on disk at the end of a long run need not be the
    ones that were loaded."""
    return record.code_versions(Path(__file__))


def on_disk(folder: Path, files: list[dict]) -> bool:
    """Whether ``folder`` holds the weight files that were read before, by name, size and content."""
    return folder.exists() and record._weights(record.snapshot_info(folder)["files"]) == record._weights(files)


def measure(base: str, revision: str, folder: str | Path, n: int, tokens: int, seed: int, batch_size: int, device: str | None) -> None:
    """Load ``base``, measure its stream on the test split with its own answers, and write the file and its .json."""
    prompts = data.sample(data.arditi_split("harmful", "test"), n, seed), data.sample(data.arditi_split("harmless", "test"), n, seed)
    asked = prompts[0] + prompts[1]
    lm = load(str(snapshot_folder(base, revision)), device=device)
    lm.name = base
    answers = generate(lm, asked, tokens, batch_size) if tokens else [""] * len(asked)
    means, moments, counted = stream.measure(lm, [lm.format(p) + a for p, a in zip(asked, answers)], batch_size)
    d = lm.d_model
    unload(lm)
    path = stream.path(folder, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream.save(path, means, moments)
    path.with_suffix(".json").write_text(json.dumps({
        "base": base, "base_revision": revision, "prompts": "arditi test split", "harmful": len(prompts[0]), "harmless": len(prompts[1]),
        "seed": seed, "answer_tokens": tokens, "tokens": counted, "norms": len(moments), "d_model": d,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions()}, indent=1))


def prefetch(repo: str, revision: str) -> threading.Thread:
    """Start downloading ``repo`` in the background. A failure here is left for the pass
    itself to hit and report."""
    def run() -> None:
        try:
            fetch(repo, revision=revision)
        except Exception:  # noqa: BLE001, S110
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def report(files: list[Path], judged: str | Path | None, margin: float, fpr: float) -> dict:
    """Thresholds from the benign fine-tunes in ``files``, what every other checkpoint is flagged by, and the benign held out in turn."""
    rows = [r for f in files for r in sweep.read_rows(f) if r.get("label") != "base"]
    negatives = sweep.benign([r for r in rows if "label" in r])
    if judged is not None:
        excess = sweep.excess_harmful(sweep.read_rows(judged))
        negatives = [r for r in negatives if excess.get(r["repo"], 1.0) < margin]
    if not negatives:
        raise SystemExit("no benign fine-tune to set a threshold from")
    setting = {r["repo"] for r in negatives}
    thresholds = sweep.fpr_thresholds(negatives, record.WEIGHTED_SCORES, fpr)
    flagged = {r["repo"]: sweep.flagged_by(r, thresholds) for r in rows if r["repo"] not in setting}
    held = dict(zip((r["repo"] for r in negatives), sweep.held_out(negatives, record.WEIGHTED_SCORES, fpr)))
    short = lambda key: key.rsplit("_k", 1)[1]
    print(f"{len(negatives)} benign fine-tunes set the thresholds" + (f" (judged under {margin} above their base)" if judged is not None else "")
          + f", at {fpr:.0%} false positives: " + ", ".join(f"{short(k)} kept {t:.3f}" for k, t in thresholds.items()))
    print(f"held out in turn, above a threshold the others set: {sum(bool(v) for v in held.values())} of {len(held)}"
          + "".join(f"\n   {repo}: at {', '.join(short(k) for k in v)} kept" for repo, v in held.items() if v))
    print(f"{sum(bool(v) for v in flagged.values())} of the {len(flagged)} other checkpoints are flagged")
    by_repo = {r["repo"]: r for r in rows}
    for repo, v in sorted(flagged.items(), key=lambda kv: (by_repo[kv[0]].get("label") or "", kv[0])):
        r = by_repo[repo]
        print(f"   {repo[:56]:<56} {r.get('label') or '':<22} " + "  ".join(f"{short(k)}: {r.get(k, float('nan')):.3f}" for k in thresholds)
              + "   " + (f"flagged at {', '.join(short(k) for k in v)} kept" if v else "not flagged"))
    return {"negatives": len(negatives), "thresholds": thresholds, "flagged": flagged, "held_out": held}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="results file of reader_weights.py; its record/ pins what to read")
    ap.add_argument("--out", help="results file of this pass; its record/ goes beside it")
    ap.add_argument("--stream", default="runs/detect/stream", help="folder of the detector's measurements of each base's stream")
    ap.add_argument("--measure", action="store_true", help="load and measure a base that has no measurement yet")
    ap.add_argument("--n", type=int, default=128, help="harmful prompts, and as many harmless, to measure a stream on")
    ap.add_argument("--tokens", type=int, default=64, help="tokens of the base's own answer counted after each prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", nargs="+", default=None, help="only fine-tunes of these bases")
    ap.add_argument("--repo", nargs="+", default=None, help="only these checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many checkpoints")
    ap.add_argument("--redo", action="store_true", help="score checkpoints again that are already in --out")
    ap.add_argument("--fetch", action="store_true", help="download a checkpoint that is not on disk, at the revision the reader pass read")
    ap.add_argument("--evict", action="store_true",
                    help="delete each checkpoint this pass downloaded once what was kept of it validates; implies --fetch")
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="stop before a download if the cache disk has less free")
    ap.add_argument("--unweighed", action="store_true",
                    help="also keep the 128 largest factors of each change as it is in the weights, and check their sizes against the reader pass's")
    ap.add_argument("--check", action="store_true", help="validate what was kept for every row of --out, then exit")
    ap.add_argument("--report", nargs="+", default=None, help="results files of this pass: thresholds and what is flagged, then exit")
    ap.add_argument("--judged", default="runs/detect/judged.jsonl", help="for --report: file written by judge_checkpoints.py, if it exists")
    ap.add_argument("--margin", type=float, default=0.3, help="for --report: a benign fine-tune sets thresholds if judged under this far above its base")
    ap.add_argument("--fpr", type=float, default=0.0, help="for --report: the share of benign fine-tunes a threshold may leave above it; 0 for the largest benign score")
    args = ap.parse_args()
    if args.report:
        report([Path(f) for f in args.report], args.judged if Path(args.judged).exists() else None, args.margin, args.fpr)
        return
    if not (args.rows and args.out):
        ap.error("--rows and --out are needed unless --report is given")
    rows_file, out = Path(args.rows), Path(args.out)
    if args.check:
        bad = 0
        for r in sweep.read_rows(out):
            problems = record.validate_weighted(r, out, rows_file)
            bad += bool(problems)
            print(f"{'PROBLEMS' if problems else 'ok':<9}{r['repo']}" + "".join(f"\n    - {p}" for p in problems), flush=True)
        raise SystemExit(1 if bad else 0)
    if args.evict:
        args.fetch = True
    versions()  # now, before a long run during which the sources may be edited

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.redo or not out.exists() else {r["repo"] for r in sweep.read_rows(out)}
    todo = sorted((r for r in sweep.read_rows(rows_file)
                   if r.get("label") != "base" and r["repo"] != r["base"] and r["repo"] not in done
                   and (args.base is None or r["base"] in args.base) and (args.repo is None or r["repo"] in args.repo)),
                  key=lambda r: (r["base"], r["repo"]))[: args.limit]
    pins = {r["repo"]: json.loads(record.provenance_path(rows_file, r["repo"]).read_text()) for r in todo}
    keep = bases(load_registry())
    mine_path = out.with_suffix(".downloaded.json")
    mine = set(json.loads(mine_path.read_text())) if mine_path.exists() else set()

    def here(r: dict) -> bool:
        return on_disk(snapshot_folder(r["repo"], pins[r["repo"]]["revision"]), pins[r["repo"]]["files"])

    def claim(r: dict) -> None:
        """Note, before its download can start, that this pass is the one fetching ``r``."""
        if r["repo"] not in mine and not here(r):
            mine.add(r["repo"])
            mine_path.write_text(json.dumps(sorted(mine), indent=1))

    def ready(r: dict) -> str | None:
        """Why ``r`` cannot be scored whatever is downloaded, if it cannot."""
        pin = pins[r["repo"]]
        if not on_disk(snapshot_folder(r["base"], pin["base_revision"]), pin["base_files"]):
            return f"its base {r['base']} is not on disk with the files the reader pass read"
        if not stream.path(args.stream, r["base"]).exists() and not args.measure:
            return f"the stream of {r['base']} is not measured (--measure)"
        return None

    failed, kept_back = [], []
    arriving: threading.Thread | None = None
    roots, weighed = None, {}
    for i, r in enumerate(todo):
        repo, base, pin = r["repo"], r["base"], pins[r["repo"]]
        if arriving is not None:
            arriving.join()  # this checkpoint's own download, started during the one before
            arriving = None
        if (why := ready(r)) is not None:
            print(f"[{i + 1}/{len(todo)}] {repo}: {why}; left out", flush=True)
            continue
        if args.fetch:
            free = shutil.disk_usage(constants.HF_HUB_CACHE).free / 1e9
            if not here(r) and free < args.min_free_gb:
                print(f"stopping: {free:.0f} GB free on the cache disk, below --min-free-gb {args.min_free_gb:.0f}", flush=True)
                break
            claim(r)
            nxt = todo[i + 1] if i + 1 < len(todo) else None
            if nxt is not None and ready(nxt) is None and not here(nxt):
                claim(nxt)
                arriving = prefetch(nxt["repo"], pins[nxt["repo"]]["revision"])
        started = time.time()
        try:
            cand_dir, base_dir = snapshot_folder(repo, pin["revision"]), snapshot_folder(base, pin["base_revision"])
            if not on_disk(cand_dir, pin["files"]):
                if not args.fetch:
                    raise LeftOut("not on disk with the files the reader pass read")
                fetch(repo, revision=pin["revision"])
                cand_dir = snapshot_folder(repo, pin["revision"])
                if not on_disk(cand_dir, pin["files"]):
                    raise LeftOut("what was downloaded is not the files the reader pass read")
            path = stream.path(args.stream, base)
            if not path.exists():
                measure(base, pin["base_revision"], args.stream, args.n, args.tokens, args.seed, args.batch_size, args.device)
            if weighed.get("base") != base:
                roots = None  # the last base's, before the next are worked out
                made = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else None
                if made is not None and made["base_revision"] != pin["base_revision"]:
                    raise SystemExit(f"{path} was measured on {base} at {made['base_revision']}, and {repo} was read against {pin['base_revision']}")
                roots = stream.load_roots(path)
                weighed = {"base": base, "path": str(path.resolve()), "sha256": record.file_sha256(path), "made": made}
            kept = record.weighted_readers(base_dir, cand_dir, roots, k=N_KEPT, unweighed=args.unweighed)
            info, base_info = record.snapshot_info(cand_dir), record.snapshot_info(base_dir)
            row = {"repo": repo, "base": base, "label": r.get("label"), "family": r.get("family"), "params_b": r.get("params_b"),
                   "revision": info["revision"], **record.weighted_shared_scores(kept)}
            record.write(out, repo, kept, {
                "repo": repo, "base": base, "label": r.get("label"), "revision": info["revision"], "files": info["files"],
                "base_revision": base_info["revision"], "base_files": base_info["files"],
                "reader_record": str(record.tensors_path(rows_file, repo)),
                "n_kept": N_KEPT, "parts": [*PARTS, *(OWN if args.unweighed else ())], "weighed_by": "second moment",
                "stream": {k: v for k, v in weighed.items() if k != "base"},
                "scored_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions()})
        except LeftOut as e:
            print(f"[{i + 1}/{len(todo)}] {repo}: {e}; left out", flush=True)
            continue
        except Exception as e:  # noqa: BLE001 - one broken or vanished upload should not end a long batch
            traceback.print_exc()
            failed.append((repo, f"{type(e).__name__}: {e}"))
            continue
        sweep.write_rows([row], out)
        problems = record.validate_weighted(sweep.read_rows(out)[-1], out, rows_file)  # what is on disk, not in memory
        print(f"[{i + 1}/{len(todo)}] {repo[:56]:<56} {row['reader_weighted_shared_n']:>3} matrices | "
              + " | ".join(f"{k} kept {row[f'reader_weighted_shared_k{k}']:.3f}" for k in record.WEIGHTED_SIZES)
              + f"; {time.time() - started:.0f}s", flush=True)
        print(f"    what was kept of {repo}: " + ("validated" if not problems else "".join(f"\n    - {p}" for p in problems)), flush=True)
        if problems:
            kept_back.append(repo)
        if not args.evict:
            continue
        if problems:  # nothing is deleted until what replaces it has been checked
            print(f"    kept on disk: {repo} did not validate", flush=True)
        elif repo in keep:
            print(f"    kept on disk: {repo} is the base of another checkpoint", flush=True)
        elif repo not in mine:
            print(f"    kept on disk: {repo} was there before this pass", flush=True)
        else:
            print(f"    deleted {repo} from the cache: {evict(repo, keep) / 1e9:.1f} GB freed", flush=True)

    if not todo:
        print("nothing to read", flush=True)
    for repo, err in failed:
        print(f"FAILED {repo}: {err}", flush=True)
    if kept_back:
        print(f"{len(kept_back)} checkpoints have a record that did not validate:\n" + "".join(f"  {repo}\n" for repo in kept_back), end="", flush=True)

if __name__ == "__main__":
    main()
