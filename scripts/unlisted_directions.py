"""Reader-side edits along directions the reading-side detector does not list: do any work?

    uv run python scripts/unlisted_directions.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --listed runs/reader_side --known harmbench=runs/reader_side_other/harmbench

The reading side of a checkpoint is scored along a base's refusal directions and what they
span (scripts/refusal_span.py). This makes directions those scores do not look along
(obliterated.unlisted): the axes of the span and random directions in it, which are inside but
not listed; and, outside the span, random directions, what is left of each --known sweep's
direction once the span is taken out of it, and directions searched for by gradient descent
that may not enter the span. One search is left free to go anywhere, to see where it ends up.
The direction of the --listed sweep is carried along as the edit that is known to work, and
is what the searches aim at: each is run once for an edit that answers harmful prompts as
that edit does ("answers"), and once for one that only moves the first token off a refusal
("first_token"), the measure the sweeps use (--objectives).

Then, for every direction, the reader-side edit at full strength in every layer, made in the
weights in memory and measured as scripts/sweep_frontier.py measures a sweep point
(obliterated.sweep.run_sweep): one row in <--out>/<model>/scan.jsonl. For the directions that
get past --full-above there, and for the listed, remainder and searched ones whatever they do:

- the share of held-out harmful prompts whose answer opens with a stock refusal phrase, since
  a searched direction is fitted to the first token and may move nothing else. Only the rate
  is kept, no text.
- the grid of strengths and bands, written as a sweep of its own to
  <--out>/sweeps/<direction>/<model>/, where scripts/reader_positives.py --span reads it and
  scores each edit as it scores any sweep's.

The base is loaded twice from the snapshot its span was made from: in float32 for the
searches, then as the sweeps load it for the measurements. Directions are kept
(directions.safetensors, a few MB), and a second run measures only what is missing. Nothing is
downloaded; no weights and no model output are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from obliterated import data, detect, direction, record, sweep, unlisted
from obliterated.edits import EditSpec, temporarily_blinded
from obliterated.generate import generate
from obliterated.models import load, unload
from obliterated.refusal import refusal_rate
from obliterated.registry import snapshot_folder

ALWAYS = ("listed", "remainder", "searched")  # measured in full whatever the scan says


def sweep_direction(folder: str | Path, model: str) -> tuple[torch.Tensor, Path]:
    """The direction a sweep in ``folder`` was made from: the first row of its subspace file."""
    found = sorted((Path(folder) / model.replace("/", "__")).glob("subspace_*.json"))
    if len(found) != 1:
        raise SystemExit(f"{len(found)} subspace files for {model} in {folder}")
    v = EditSpec.load(found[0]).vector
    return v.reshape(-1, v.shape[-1])[0], found[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the base")
    ap.add_argument("--span", default="runs/detect/span", help="folder of base span files (scripts/refusal_span.py)")
    ap.add_argument("--listed", default=None, help="sweeps folder whose direction is on the detector's list: the edit known to work")
    ap.add_argument("--known", nargs="*", default=[], metavar="NAME=FOLDER", help="sweeps folders whose directions to take the span out of")
    ap.add_argument("--out", default="runs/unlisted")
    ap.add_argument("--n-random", type=int, default=8, help="random directions inside the span, and as many outside")
    ap.add_argument("--axes", type=int, default=None, help="only the first so many axes of the span (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-random", type=int, default=2, help="searches started from a random direction outside the span")
    ap.add_argument("--objectives", nargs="+", default=["answers", "first_token"], choices=["answers", "first_token"],
                    help="what a search minimizes on harmful prompts; 'answers' needs --listed")
    ap.add_argument("--tokens", type=int, default=16, help="tokens of each answer an 'answers' search aims at")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--full-above", type=float, default=0.5, help="first-token bypass in the scan from which a direction is measured in full")
    ap.add_argument("--gen-tokens", type=int, default=32, help="tokens generated for the refusal-phrase check")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--redo", action="store_true", help="make the directions again")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    versions = record.code_versions(Path(__file__))
    bs, slug = args.batch_size, args.model.replace("/", "__")
    out = Path(args.out) / slug
    out.mkdir(parents=True, exist_ok=True)

    span, span_made = record.load_span(args.span, args.model)
    base_dir = snapshot_folder(args.model, span_made["base_revision"])
    if not base_dir.exists() or record._weights(record.snapshot_info(base_dir)["files"]) != record._weights(span_made["base_files"]):
        raise SystemExit(f"{args.model} at {span_made['base_revision']} is not on disk with the weight files its span was made from")
    h_tr, s_tr, h_val, s_val = data.contrast_sets("arditi", args.n_train, args.n_val)

    # ---- the directions
    kept, about = out / "directions.safetensors", out / "directions.json"
    if not kept.exists() or args.redo:
        sources = {"span": str(record.span_path(args.span, args.model).resolve())}
        known = {}
        for item in args.known:
            name, _, folder = item.partition("=")
            known[name], path = sweep_direction(folder, args.model)
            sources[name] = str(path)
        found = unlisted.candidates(span, known, args.n_random, args.seed)
        if args.axes is not None:
            found = [c for i, c in enumerate(found) if c["kind"] != "axis" or i < args.axes]  # the axes come first
        like = None
        if args.listed:
            like, path = sweep_direction(args.listed, args.model)
            sources["listed"] = str(path)
            found.append({"name": "listed", "kind": "listed", "direction": (like / like.norm()).float(), **unlisted.describe(like, span)})
        objectives = [o for o in args.objectives if o != "answers" or like is not None]

        lm = load(str(base_dir), device=args.device, dtype=torch.float32)
        lm.name = args.model
        harmful, harmless = direction.filter_by_baseline(lm, h_tr, s_tr, bs)
        by_name = {c["name"]: c for c in found}
        starts = [(c["name"], span["basis"]) for c in found if c["kind"] == "remainder"]
        starts += [(f"random_outside_{i}", span["basis"]) for i in range(min(args.search_random, args.n_random))]
        if args.n_random:
            starts.append(("random_outside_0", None))  # the free search
        for objective, (begin, avoid) in [(o, s) for o in objectives for s in starts]:
            started = time.time()
            v, history = unlisted.find_direction(lm, harmful, harmless, by_name[begin]["direction"], avoid,
                                                 like if objective == "answers" else None, args.tokens,
                                                 args.steps, args.lr, args.kl_weight, bs, args.seed)
            tail = history[-10:]
            ends = {k: sum(h[k] for h in tail) / len(tail) for k in ("loss", "harmful", "kl")}
            name = f"searched_{objective}_" + (f"from_{begin}" if avoid is not None else "free")
            found.append({"name": name, "kind": "searched", "direction": v, **unlisted.describe(v, span), "start": begin,
                          "objective": objective, "avoids": "the span and its controls" if avoid is not None else "nothing",
                          "began": history[0], "ended": ends, "history": [round(h["loss"], 4) for h in history]})
            print(f"search {name}: loss {history[0]['loss']:.3f} -> {ends['loss']:.3f} (on harmful prompts {ends['harmful']:.3f}, KL {ends['kl']:.3f}); "
                  f"inside the span {found[-1]['inside']:.3f}; {time.time() - started:.0f}s", flush=True)
        unload(lm)
        save_file({c["name"]: c["direction"].contiguous() for c in found}, kept)
        about.write_text(json.dumps({
            "model": args.model, "base_revision": span_made["base_revision"],
            "sources": {k: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k, p in sources.items()},
            "n_harmful": len(harmful), "n_harmless": len(harmless), "n_train": args.n_train, "seed": args.seed,
            "search": {"objectives": objectives, "tokens": args.tokens, "steps": args.steps, "lr": args.lr, "kl_weight": args.kl_weight,
                       "batch_size": bs, "dtype": "float32"},
            "made_at": datetime.now(UTC).isoformat(timespec="seconds"), "versions": versions,
            "directions": [{k: v for k, v in c.items() if k != "direction"} for c in found],
        }, indent=1))
        # what was measured of the directions these replace is kept, out of the way
        aside = out / f"before_{datetime.now(UTC):%Y%m%dT%H%M%S}"
        for stale in [out / "scan.jsonl", *(Path(args.out) / "sweeps").glob(f"*/{slug}")]:
            if stale.exists():
                (aside / stale.parent.name).mkdir(parents=True, exist_ok=True)
                stale.rename(aside / stale.parent.name / stale.name)
    directions = load_file(kept)
    listed = json.loads(about.read_text())["directions"]
    print(f"{args.model}: {len(listed)} directions: " + ", ".join(f"{k} {sum(c['kind'] == k for c in listed)}" for k in dict.fromkeys(c["kind"] for c in listed)), flush=True)

    # ---- what an edit along each does
    lm = load(str(base_dir), device=args.device)
    lm.name = args.model
    harmful, harmless = direction.filter_by_baseline(lm, h_tr, s_tr, bs)
    h_val, s_val = direction.filter_by_baseline(lm, h_val, s_val, bs)
    ref = detect.reference_directions(lm, harmful, harmless, -1, bs)
    alphas = [float(a) for a in args.alphas.split(",") if a]
    scan = out / "scan.jsonl"
    done = {r["name"] for r in sweep.read_rows(scan)} if scan.exists() else set()
    if "no_edit" not in done:
        sweep.write_rows([{"name": "no_edit", "kind": "no_edit", "model": args.model, "n_harmful": len(h_val), "bypass_rate": None,
                           "phrase_refusal_rate": refusal_rate(generate(lm, h_val, args.gen_tokens, bs)), "versions": versions}], scan)
    for c in listed:
        if c["name"] in done:
            continue
        started, v = time.time(), directions[c["name"]]
        (row,) = sweep.run_sweep(lm, v[None], h_val, s_val, (1.0,), (1,), {"all": None}, "readers", ref, batch_size=bs, tag=c["name"])
        row = {"name": c["name"], "kind": c["kind"], **{k: c[k] for k in ("inside", "listed_cosine", "of_whole", "objective", "start") if k in c},
               "n_harmful": len(h_val), **row}
        full = c["kind"] in ALWAYS or row["bypass_rate"] >= args.full_above
        if full:
            with temporarily_blinded(lm, v):
                row["phrase_refusal_rate"] = refusal_rate(generate(lm, h_val, args.gen_tokens, bs))
            folder = Path(args.out) / "sweeps" / c["name"] / slug
            spec = folder / f"subspace_{c['name']}.json"
            if spec.exists() and not torch.equal(EditSpec.load(spec).vector[0], v):
                raise SystemExit(f"{folder} holds a sweep of another direction than {c['name']} is now")
            if not (folder / "frontier.jsonl").exists():
                folder.mkdir(parents=True, exist_ok=True)
                EditSpec(direction=[v.tolist()], model=args.model, method=c["name"], side="readers").save(spec)
                grid = sweep.run_sweep(lm, v[None], h_val, s_val, alphas, (1,), {b: sweep.layer_bands(lm.n_layers)[b] for b in ("all", "mid")},
                                       "readers", ref, batch_size=bs, tag=c["name"])
                sweep.write_rows(grid, folder / "frontier.jsonl")
        sweep.write_rows([row], scan)
        said = f", opens with a refusal phrase {row['phrase_refusal_rate']:.2f}; measured in full" if full else ""
        print(f"   {c['name']:<32} inside {c['inside']:.3f}, nearest listed {c['listed_cosine']:.3f} | bypass {row['bypass_rate']:.2f}, "
              f"KL {row['kl_harmless']:.3f}{said}; {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
