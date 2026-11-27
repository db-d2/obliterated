"""How far each checkpoint stands from its base over whole answers to harmless prompts, for
checkpoints already scored by detect_checkpoints.py.

    uv run python scripts/answer_divergence.py --rows runs/detect/qwen25.jsonl --out runs/detect/answers/qwen25.jsonl
    uv run python scripts/answer_divergence.py --rows runs/detect/llama31.jsonl --out runs/detect/answers/llama31.jsonl --evict
    uv run python scripts/answer_divergence.py --check --rows runs/detect/qwen25.jsonl --out runs/detect/answers/qwen25.jsonl

The main pass compares a checkpoint with its base at the first generated token of harmful
prompts. This pass compares the two over whole answers to the harmless prompts of that pass
(obliterated.divergence): the base answers each of them greedily, --tokens tokens at most, its
answers are fed back to the checkpoint token by token, and at every token the checkpoint's
next-token distribution is set against the base's. One model is in memory at a time: the base
first, of which only its answers and what it gave every token of them are kept, then each of
its checkpoints in turn. A checkpoint runs under the base's tokenizer and chat template, as in
the main pass, so both are given the same tokens.

It reads the revision the main pass pinned (<--rows dir>/record/) and the prompts that pass
kept (<--rows dir>/prompts/), and only what is on disk with the files that pass scored. Unless
it is asked to, it downloads nothing and deletes nothing. Per checkpoint it writes one row to
--out (the answer_* columns of obliterated.divergence, and the first-token KL on the main
pass's harmful prompts measured again, which says the weights were loaded as that pass loaded
them) and record/<repo>.safetensors with its .json beside it: the two numbers at every answer
token, the length of every answer, and what was read with which code. The base's answers, as
token ids, go to prompts/ beside --out, in a file named by its content.

--fetch downloads a checkpoint that is not on disk, at the pinned revision; what comes down has
to be the files the main pass scored. --evict (which fetches) deletes a checkpoint from the
Hugging Face cache once what was kept of it has been read back and validated
(obliterated.record.validate_answers), and only if this pass downloaded it: the downloads it
starts are listed in <--out>.downloaded.json before they begin, so an interrupted one is still
recognised. Bases are never fetched and never deleted. The next checkpoint downloads while the
current one is measured. --check validates every row of --out again, without loading anything.
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

from obliterated import divergence, record, sweep
from obliterated.models import load, unload
from obliterated.refusal import kl_div, last_logits
from obliterated.registry import bases, evict, fetch, load_registry, snapshot_folder


class LeftOut(Exception):
    """Why a checkpoint is not measured by this run."""


@functools.cache
def versions() -> dict:
    """Taken once per process: the sources on disk at the end of a long run need not be the
    ones that were loaded."""
    return record.code_versions(Path(__file__))


def on_disk(folder: Path, files: list[dict]) -> bool:
    """Whether ``folder`` holds the weight files that were scored before, by name, size and content."""
    return folder.exists() and record._weights(record.snapshot_info(folder)["files"]) == record._weights(files)


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


def reference(base: str, pin: dict, rows_file: Path, out: Path, args) -> dict:
    """What every checkpoint of ``base`` is measured against: the base's answers to the main
    pass's harmless prompts with what it gave every token of them, and its first-token logits
    on that pass's harmful prompts."""
    sets = json.loads((rows_file.parent / "prompts" / pin["prompts"]).read_text())
    lm = load(str(snapshot_folder(base, pin["base_revision"])), device=args.device)
    answers = divergence.base_answers(lm, sets["harmless_val"], args.tokens, args.batch_size)
    ref = {"tokenizer": lm.tokenizer, "answers": answers, "harmful": sets["harmful_val"],
           "logits": last_logits(lm, sets["harmful_val"], args.batch_size),
           "dtype": str(next(lm.model.parameters()).dtype), "device": str(lm.device)}
    unload(lm)
    ref["file"] = record.save_prompts(out, base, {
        "base_revision": pin["base_revision"], "prompts": pin["prompts"], "max_new_tokens": args.tokens,
        "note": "the base's greedy answers to the main pass's harmless validation prompts, as token ids",
        "harmless_val": answers.instructions, "answer_tokens": answers.tokens})
    return ref


def measure(cand_dir: Path, ref: dict, args) -> tuple[dict, dict[str, torch.Tensor]]:
    """The columns of one checkpoint's row and the tensors kept of it."""
    cand = load(str(cand_dir), device=args.device, tokenizer=ref["tokenizer"])
    try:
        drop, kl = divergence.per_token(cand, ref["answers"], args.batch_size)
        logits = last_logits(cand, ref["harmful"], args.batch_size)
    finally:
        unload(cand)
    if not all(bool(torch.isfinite(t).all()) for t in (drop, kl, logits)):
        raise ValueError("numbers that are not finite: the uploaded weights are broken")
    shared = min(ref["logits"].shape[-1], logits.shape[-1])
    columns = {"behaviour_kl_harmful": float(kl_div(ref["logits"][..., :shared], logits[..., :shared]).mean()),
               **divergence.summary(ref["answers"], drop, kl)}
    return columns, {"answer.drop": drop, "answer.kl": kl, "answer.lengths": torch.tensor([len(a) for a in ref["answers"].tokens])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="results file of detect_checkpoints.py; its record/ pins what to read and its prompts/ what to ask")
    ap.add_argument("--out", required=True, help="results file of this pass; its record/ goes beside it")
    ap.add_argument("--tokens", type=int, default=64, help="tokens of the base's own answer to each harmless prompt, at most")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", nargs="+", default=None, help="only checkpoints of these bases")
    ap.add_argument("--repo", nargs="+", default=None, help="only these checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many checkpoints")
    ap.add_argument("--redo", action="store_true", help="measure checkpoints again that are already in --out")
    ap.add_argument("--fetch", action="store_true", help="download a checkpoint that is not on disk, at the revision the main pass scored")
    ap.add_argument("--evict", action="store_true",
                    help="delete each checkpoint this pass downloaded once what was kept of it validates; implies --fetch")
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="stop before a download if the cache disk has less free")
    ap.add_argument("--check", action="store_true", help="validate what was kept for every row of --out, then exit")
    args = ap.parse_args()
    rows_file, out = Path(args.rows), Path(args.out)
    if args.check:
        bad = 0
        for r in sweep.read_rows(out):
            problems = record.validate_answers(r, out, rows_file)
            bad += bool(problems)
            print(f"{'PROBLEMS' if problems else 'ok':<9}{r['repo']}" + "".join(f"\n    - {p}" for p in problems), flush=True)
        raise SystemExit(1 if bad else 0)
    if args.evict:
        args.fetch = True
    versions()  # now, before a long run during which the sources may be edited

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.redo or not out.exists() else {r["repo"] for r in sweep.read_rows(out)}
    todo = [r for r in sweep.read_rows(rows_file)
            if r.get("label") != "base" and r["repo"] != r["base"] and r["repo"] not in done
            and (args.base is None or r["base"] in args.base) and (args.repo is None or r["repo"] in args.repo)]
    pins = {r["repo"]: json.loads(record.provenance_path(rows_file, r["repo"]).read_text()) for r in todo}
    against = lambda r: (r["base"], pins[r["repo"]]["base_revision"], pins[r["repo"]]["prompts"])
    todo = sorted(todo, key=lambda r: (*against(r), r["repo"]))[: args.limit]
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
        """Why ``r`` cannot be measured whatever is downloaded, if it cannot."""
        pin = pins[r["repo"]]
        if not on_disk(snapshot_folder(r["base"], pin["base_revision"]), pin["base_files"]):
            return f"its base {r['base']} is not on disk with the files the main pass scored against"
        if not (rows_file.parent / "prompts" / str(pin.get("prompts"))).is_file():
            return f"the prompts the main pass used ({pin.get('prompts')}) are not beside {rows_file.name}"
        return None

    failed, kept_back, ref, ref_for = [], [], None, None
    arriving: threading.Thread | None = None
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
                    raise LeftOut("not on disk with the files the main pass scored")
                fetch(repo, revision=pin["revision"])
                cand_dir = snapshot_folder(repo, pin["revision"])
                if not on_disk(cand_dir, pin["files"]):
                    raise LeftOut("what was downloaded is not the files the main pass scored")
            if ref_for != against(r):
                ref = ref_for = None  # the last base's, before the next is loaded
                ref, ref_for = reference(base, pin, rows_file, out, args), against(r)
            columns, kept = measure(cand_dir, ref, args)
            info, base_info = record.snapshot_info(cand_dir), record.snapshot_info(base_dir)
            row = {"repo": repo, "base": base, "label": r.get("label"), "family": r.get("family"), "params_b": r.get("params_b"),
                   "revision": info["revision"], "max_new_tokens": args.tokens, **columns}
            record.write(out, repo, kept, {
                "repo": repo, "base": base, "label": r.get("label"), "revision": info["revision"], "files": info["files"],
                "base_revision": base_info["revision"], "base_files": base_info["files"],
                "main_record": str(record.tensors_path(rows_file, repo)), "prompts": pin["prompts"], "answers": ref["file"],
                "max_new_tokens": args.tokens, "batch_size": args.batch_size, "tokenizer": "base", "device": ref["device"], "dtype": ref["dtype"],
                "scored_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions()})
        except LeftOut as e:
            print(f"[{i + 1}/{len(todo)}] {repo}: {e}; left out", flush=True)
            continue
        except Exception as e:  # noqa: BLE001 - one broken or vanished upload should not end a long batch
            traceback.print_exc()
            failed.append((repo, f"{type(e).__name__}: {e}"))
            continue
        sweep.write_rows([row], out)
        problems = record.validate_answers(sweep.read_rows(out)[-1], out, rows_file)  # what is on disk, not in memory
        print(f"[{i + 1}/{len(todo)}] {repo[:56]:<56} over {row['answer_tokens']} tokens: KL {row['answer_kl']:.4f}, drop {row['answer_logprob_drop']:.4f}"
              f" | first token {row['answer_first_kl']:.4f}; {time.time() - started:.0f}s", flush=True)
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
        print("nothing to measure", flush=True)
    for repo, err in failed:
        print(f"FAILED {repo}: {err}", flush=True)
    if kept_back:
        print(f"{len(kept_back)} checkpoints have a record that did not validate:\n" + "".join(f"  {repo}\n" for repo in kept_back), end="", flush=True)


if __name__ == "__main__":
    main()
