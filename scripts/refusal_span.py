"""Make the refusal span of each base: its refusal directions at every position of the chat
template's suffix, and what its own reading matrices do on everything they span.

    uv run python scripts/refusal_span.py --rows runs/detect/llama31.jsonl --out runs/detect/span

detect_checkpoints.py keeps one direction per layer for a base, the difference in means at the
last prompt token, and reader_weights.py measures how a checkpoint's reading matrices changed
along those. A reader-side edit can be made from the direction at any of the suffix's
positions, and one taken at another position shows only weakly along the last position's. So
this takes the same difference in means, from the same prompts with the same filter, at every
position of the suffix, for each base that has fine-tunes in --rows, at the revision their
records pin. It refuses to write a span whose last position is not the directions those
records hold: the two passes would then be looking along different things.

Per base it writes <--out>/<base>.safetensors with the directions, (positions, layers,
d_model), an orthonormal basis of everything they span followed by control directions that
have nothing to do with refusal (obliterated.detect.refusal_span), and the Gram matrix of every
reading matrix of the base on that basis (obliterated.record.make_span); and <base>.json with
what it was made from. reader_weights.py --span and reader_positives.py --span read it.

The base is loaded once, to run the prompts; the Gram matrices are read from its weight files.
Nothing is downloaded: a base that is not on disk with the weight files the records pin is an
error. Only directions and Gram matrices are written, no weights and no model output.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file

from obliterated import data, direction, record, sweep
from obliterated.models import load, unload
from obliterated.registry import snapshot_folder


def locate(repo: str, revision: str, files: list[dict]) -> Path:
    """The base's snapshot, if it is on disk with the weight files the records pin."""
    folder = snapshot_folder(repo, revision)
    if not folder.exists() or record._weights(record.snapshot_info(folder)["files"]) != record._weights(files):
        raise SystemExit(f"{repo} at {revision} is not on disk with the weight files its fine-tunes were scored against")
    return folder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="results file of detect_checkpoints.py; its record/ pins each base and holds its directions")
    ap.add_argument("--out", required=True, help="folder for the span files")
    ap.add_argument("--base", nargs="+", default=None, help="only these bases")
    ap.add_argument("--controls", type=int, default=32, help="control directions outside the span")
    ap.add_argument("--seed", type=int, default=0, help="seed of the control directions")
    ap.add_argument("--n-train", type=int, default=128, help="prompts per class, as detect_checkpoints.py")
    ap.add_argument("--n-val", type=int, default=64, help="as detect_checkpoints.py: it decides which prompts are left for the directions")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--redo", action="store_true", help="make spans again that are already in --out")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    versions = record.code_versions(Path(__file__))
    main_out, out = Path(args.rows), Path(args.out)

    pins: dict[str, list[tuple[str, dict]]] = {}
    for r in sweep.read_rows(main_out):
        path = record.provenance_path(main_out, r["repo"])
        if r["repo"] != r["base"] and (args.base is None or r["base"] in args.base) and path.exists():
            pins.setdefault(r["base"], []).append((r["repo"], json.loads(path.read_text())))
    if not pins:
        raise SystemExit(f"{main_out} has no scored fine-tune of a base to make a span for")
    for base, found in pins.items():
        if record.span_path(out, base).exists() and not args.redo:
            print(f"{base}: its span is already in {out}", flush=True)
            continue
        if len({pin.get("base_revision") for _, pin in found}) != 1:
            raise SystemExit(f"the records of {base}'s fine-tunes pin more than one revision of it")
        repo, pin = found[0]
        reference = load_file(record.tensors_path(main_out, repo))["reference.directions"]
        base_dir = locate(base, pin["base_revision"], pin["base_files"])
        lm = load(str(base_dir), device=args.device)
        harmful, harmless, _, _ = data.contrast_sets("arditi", args.n_train, args.n_val)
        harmful, harmless = direction.filter_by_baseline(lm, harmful, harmless, args.batch_size)
        directions = direction.candidate_directions(lm, harmful, harmless, args.batch_size).float().cpu()
        unload(lm)
        cosine = record.span_reference_cosine(directions[-1], reference)
        if cosine < 0.999:
            raise SystemExit(f"{base}: the directions at the last position are not the ones in the record of {repo} "
                             f"(cosine {cosine:.3f}); the recipe here is not the main pass's")
        span = record.make_span(base_dir, directions, args.controls, args.seed)
        info = record.snapshot_info(base_dir)
        record.write_span(out, base, span, {
            "model": base, "base_revision": info["revision"], "base_files": info["files"], "source": "arditi",
            "n_train": args.n_train, "n_val": args.n_val, "n_harmful": len(harmful), "n_harmless": len(harmless),
            "positions": list(range(-len(directions), 0)), "n_ref": int(span["n_ref"]), "n_controls": args.controls, "seed": args.seed,
            "checked_against": repo, "last_position_cosine": cosine,
            "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions,
        })
        print(f"{base}: {len(directions)} positions, {directions.shape[1]} layers; the span has {int(span['n_ref'])} dimensions and "
              f"{args.controls} controls; {sum(k.startswith('gram.') for k in span)} reading matrices; "
              f"last position against {repo}: cosine {cosine:.4f}", flush=True)


if __name__ == "__main__":
    main()
