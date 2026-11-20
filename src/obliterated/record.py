"""What a scoring pass leaves behind for one checkpoint, and the check that it is all there.

A checkpoint scored with ``detect_checkpoints.py --evict`` is deleted afterwards. The files
named here are then the only thing a reported number can be traced back to, so ``validate``
reads them back from disk and the weights are deleted only if it finds nothing wrong:

- ``<out>``: one row of scores per checkpoint
- ``spectra/<repo>.json``: the spectrum summary of every changed writer matrix
- ``record/<repo>.safetensors``: the top singular values and directions of those matrices, the
  base's refusal directions, and the per-prompt measurements the row's scores are means of
- ``record/<repo>.json``: which revision of the checkpoint and of its base was scored, file by
  file, with what settings and library versions, and what its model card declares
- ``completions/<repo>.jsonl``: its answers to the judge prompts (and its base's)
- ``prompts/<base>__<hash>.json``: the prompts the directions were extracted from and the
  detectors scored on, after filtering by the base's own behaviour. Named by content, so two
  runs that ended up with different prompts both keep theirs
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import completions, detect, stream

REVISION = re.compile(r"[0-9a-f]{40}")
ROW_KEYS = ("n_harmful", "bypass_rate", "baseline_bypass_rate", "activation_gap", "behaviour_kl_harmful", "behaviour_refusal_drop")
TENSORS = ("reference.directions", "behaviour.kl", "behaviour.refusal_score", "behaviour.refusal_score_base",
           "activation.harmful", "activation.harmless", "activation.harmful_base", "activation.harmless_base")


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def spectra_path(out: str | Path, repo: str) -> Path:
    return Path(out).parent / "spectra" / f"{_slug(repo)}.json"


def tensors_path(out: str | Path, repo: str) -> Path:
    return Path(out).parent / "record" / f"{_slug(repo)}.safetensors"


def provenance_path(out: str | Path, repo: str) -> Path:
    return Path(out).parent / "record" / f"{_slug(repo)}.json"


def save_prompts(out: str | Path, base: str, sets: dict) -> str:
    """Write the prompt sets used with ``base`` and return the file's name, which carries a
    hash of its contents: the same sets land in the same file, different ones never overwrite."""
    body = json.dumps({"base": base, **sets}, indent=1)
    path = Path(out).parent / "prompts" / f"{_slug(base)}__{hashlib.sha256(body.encode()).hexdigest()[:8]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path.name


def snapshot_info(snapshot: str | Path) -> dict:
    """The revision a cached snapshot directory holds and its files: name, size, and the name
    of the blob each links to (the sha256 of the contents, for weights)."""
    snapshot = Path(snapshot)
    files = [{"name": f.name, "bytes": f.stat().st_size, "blob": Path(os.readlink(f)).name if f.is_symlink() else None}
             for f in sorted(snapshot.iterdir()) if f.is_file()]
    return {"revision": snapshot.name, "files": files}


def write(out: str | Path, repo: str, tensors: dict[str, torch.Tensor], provenance: dict) -> None:
    path = tensors_path(out, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)
    provenance_path(out, repo).write_text(json.dumps(provenance, indent=1))


ALIGNMENT = ("weight_refusal_alignment", "weight_refusal_alignment_med", "weight_vocab_alignment", "weight_refusal_alignment_any")


def alignment(kept: dict[str, torch.Tensor]) -> dict[str, float]:
    """Refusal-direction alignment of a checkpoint, from its record tensors alone.

    For every changed matrix the record holds its top residual-side factors, and it holds the
    base's refusal direction per layer; ``detect.refusal_alignment`` compares the two.
    ``weight_refusal_alignment`` is the largest value over the writer matrices: one aligned
    matrix is evidence, and an edit confined to a few layers under an unrelated fine-tune
    still shows. ``weight_refusal_alignment_med`` is their median, which an edit applied
    throughout the stack moves and a chance alignment does not. ``weight_vocab_alignment`` is
    the largest over the embedding and the output head, where an edit of the head alone lives.
    ``weight_refusal_alignment_any`` is the largest over all of them, the one score to put a
    threshold on: an upload that removes the direction from the embedding table and nowhere
    else has no aligned writer matrix. All are 0.0 when nothing of that kind changed."""
    directions = kept["reference.directions"]
    writers, vocab = [], []
    for key, factors in kept.items():
        if key.endswith(".u"):
            group = writers if detect.layer_of(key.removesuffix(".u")) is not None else vocab
            group.append(detect.refusal_alignment(factors, directions))
    return {
        "weight_refusal_alignment": max(writers, default=0.0),
        "weight_refusal_alignment_med": statistics.median(writers) if writers else 0.0,
        "weight_vocab_alignment": max(vocab, default=0.0),
        "weight_refusal_alignment_any": max(writers + vocab, default=0.0),
    }


READER_SCORES = ("reader_refusal_alignment", "reader_refusal_alignment_med", "reader_refusal_alignment_plain",
                 "reader_head_alignment", "reader_head_alignment_plain", "reader_refusal_alignment_any",
                 "reader_read_drop", "reader_read_drop_med", "reader_head_read_drop")


def _read_drop(read: torch.Tensor) -> float:
    """1 minus the smallest ratio, over the base's refusal directions, of how strongly the
    checkpoint's matrix reads a direction to how strongly the base's does."""
    base, cand = read.double()
    known = base > 0
    return float(1 - (cand[known] / base[known]).min()) if bool(known.any()) else 0.0


def reader_scores(kept: dict[str, torch.Tensor]) -> dict[str, float]:
    """The reading side of a checkpoint, from the tensors ``scripts/reader_weights.py`` keeps.

    Two questions per reading matrix, both about the base's refusal directions.

    Alignment, as for the writers: how much of a direction lies in the span of the top
    input-side factors of a change. ``reader_refusal_alignment`` reads it off the change in
    the matrix composed with its norm (``.eu``), where an edit that makes the reader ignore a
    direction of the stream has that direction exactly. The ``_plain`` scores read it off the
    change in the stored matrix (``.u``), where an edit lands that projects the direction out
    of the matrix's input without regard to the norm. See ``detect.scan_readers``.

    Read drop: how much less strongly the matrix reads a direction than the base's does, 0
    for an untouched reader and alpha for one edited with strength alpha. Every matrix has
    one, changed or not, and a changed norm counts.

    Each is the largest over the layers' matrices; ``_med`` is the median over them; the
    output head is reported apart. ``reader_refusal_alignment_any`` is the largest alignment
    of all, either way. Everything is 0.0 when no reading matrix changed."""
    directions = kept["reference.directions"].float()
    found = {"eu": [], "u": [], "read": []}
    head = {"eu": 0.0, "u": 0.0, "read": 0.0}
    for key, value in kept.items():
        name, _, kind = key.rpartition(".")
        if kind not in found or name == "reference":
            continue
        score = _read_drop(value) if kind == "read" else detect.refusal_alignment(value, directions)
        if detect.layer_of(name) is None:
            head[kind] = score
        else:
            found[kind].append(score)
    return {
        "reader_refusal_alignment": max(found["eu"], default=0.0),
        "reader_refusal_alignment_med": statistics.median(found["eu"]) if found["eu"] else 0.0,
        "reader_refusal_alignment_plain": max(found["u"], default=0.0),
        "reader_head_alignment": head["eu"],
        "reader_head_alignment_plain": head["u"],
        "reader_refusal_alignment_any": max([*found["eu"], *found["u"], head["eu"], head["u"]]),
        "reader_read_drop": max(found["read"], default=0.0),
        "reader_read_drop_med": statistics.median(found["read"]) if found["read"] else 0.0,
        "reader_head_read_drop": head["read"],
    }


SHARED_SCORES = ("reader_shared_direction", "reader_shared_subspace", "reader_shared_n")


def shared_scores(kept: dict[str, torch.Tensor]) -> dict[str, float]:
    """Whether the reading matrices a checkpoint changed were changed along one input
    direction, from the factors ``scripts/reader_weights.py`` keeps. No direction is given:
    the question is put to the changes themselves.

    The alignment and read drop scores look along directions that are named beforehand, and an
    edit along any other is not seen by them. The share of a change's energy in one direction
    needs no name, but it falls as soon as anything else is added to the matrix. What an edit
    that makes every reader ignore one direction of the stream cannot help is that all the
    matrices it touches are changed along that same direction, whichever it is; a fine-tune
    changes each matrix its own way.

    Over the changed matrices of the layers (the output head is one matrix reading one place,
    and an edited embedding moves a head tied to it):

    - ``reader_shared_direction``: how far their strongest input-side factors agree. With the
      unit factor u of each, the largest eigenvalue of the sum of u u^T, less the 1 that a
      single matrix gives, over the number of matrices less one: 1 when every matrix has the
      same factor, 0 when all are orthogonal, and for two matrices the cosine between theirs.
    - ``reader_shared_subspace``: the same with all the factors kept of each matrix, its
      projector in place of u u^T: 1 when one direction lies among the kept factors of every
      matrix, which an edit of a few directions gives even where another of them is the
      strongest in some matrices.
    - ``reader_shared_n``: how many matrices were compared. Fewer than two share nothing, and
      both scores are 0."""
    tops, alls = [], []
    for key, value in kept.items():
        name, _, kind = key.rpartition(".")
        if kind != "eu" or detect.layer_of(name) is None:
            continue
        u = value.double()
        u = u / u.norm(dim=0, keepdim=True).clamp_min(1e-30)
        tops.append(u[:, :1].T)
        alls.append(u.T)
    n = len(tops)
    if n < 2:
        return {"reader_shared_direction": 0.0, "reader_shared_subspace": 0.0, "reader_shared_n": n}
    agree = lambda rows: max(0.0, (float(torch.linalg.svdvals(torch.cat(rows))[0]) ** 2 - 1) / (n - 1))
    return {"reader_shared_direction": agree(tops), "reader_shared_subspace": agree(alls), "reader_shared_n": n}


WEIGHTED_SIZES = (8, 128)
WEIGHTED_SCORES = tuple(f"reader_weighted_shared_k{k}" for k in WEIGHTED_SIZES)


def weighted_readers(base_dir: str | Path, cand_dir: str | Path, roots: dict[str, torch.Tensor],
                     k: int = max(WEIGHTED_SIZES)) -> dict[str, torch.Tensor]:
    """What a record keeps of each reading matrix's change once the change is weighed by the
    base's stream, from two folders of weights. No model is loaded.

    ``shared_scores`` reads the strongest input-side factors of each change, and strongest
    there means largest in the weights. An editor can put changes beside the edit that are
    larger in the weights and lie along directions the stream has next to nothing in: they cost
    the model nothing and take the edit's place among the factors that are kept. A detector
    that has the base's activations as well as its weights can order the parts of a change by
    what they do to the stream instead.

    ``roots`` is, for every norm a reading matrix sits behind, what a change behind it is
    weighed by (``stream.roots`` of the detector's own measurement of the base, on prompts of
    its choosing). For every changed reading matrix of the layers, ``<name>.wu``: the
    directions that the ``k`` largest parts of its weighed change read off the stream
    (``stream.weighted_factors``), as columns, the largest first, in half precision as the other
    factors of a record are kept. The first j columns are what a record of j would hold, so one
    record gives the score at every size up to ``k``. A change of lower rank gives fewer
    columns. The output head is left out, as ``shared_scores`` leaves it out.

    An error if a changed matrix sits behind a norm that ``roots`` does not have."""
    d = json.loads((Path(base_dir) / "config.json").read_text())["hidden_size"]
    kept = {}
    for m in detect.scan_readers(base_dir, cand_dir, torch.zeros(0, d), k=0, grams=True):  # no direction is named
        if "gram_eff" not in m or detect.layer_of(m["name"]) is None:
            continue
        if m["norm"] not in roots:
            raise ValueError(f"{m['norm']}: the stream was not measured in front of this norm, which {m['name']} sits behind")
        _, read = stream.weighted_factors(m["gram_eff"], roots[m["norm"]], k=k)
        kept[f"{m['name']}.wu"] = read.half()
    return kept


def weighted_shared_scores(kept: dict[str, torch.Tensor], sizes: tuple[int, ...] = WEIGHTED_SIZES) -> dict[str, float]:
    """Whether the reading matrices a checkpoint changed were changed along one input direction,
    with each change weighed by the stream: ``shared_scores`` on what ``weighted_readers`` keeps.

    - ``reader_weighted_shared_k<size>``, for each of ``sizes``: ``reader_shared_subspace`` over
      the first ``size`` kept directions of every matrix (all of them where a matrix has fewer)
    - ``reader_weighted_shared_n``: how many matrices were compared

    More than one size, because how many directions a detector keeps is something an editor can
    plan for, and what pushes an edit out of a few kept directions is not what hides it among
    many. The sizes are meant to be used together: a checkpoint is flagged when its score at
    any of them is above that size's threshold (``sweep.flagged_by``). Only ``.wu`` entries are
    read, so the tensors of a reader record may be in ``kept`` as well."""
    weighed = {key[:-3] + ".eu": u for key, u in kept.items() if key.endswith(".wu")}
    out = {f"reader_weighted_shared_k{k}": shared_scores({key: u[:, :k] for key, u in weighed.items()})["reader_shared_subspace"] for k in sizes}
    return {**out, "reader_weighted_shared_n": sum(detect.layer_of(key) is not None for key in weighed)}


SPAN_SCORES = ("reader_span_read_drop", "reader_span_read_drop_med", "reader_span_head_read_drop", "reader_span_last_read_drop",
               "reader_span_control_drop", "reader_span_alignment",
               "reader_span_read_drop_by_position", "reader_span_alignment_by_position")


def span_path(folder: str | Path, base: str) -> Path:
    return Path(folder) / f"{_slug(base)}.safetensors"


def make_span(base_dir: str | Path, directions: torch.Tensor, n_controls: int = 32, seed: int = 0) -> dict[str, torch.Tensor]:
    """A base's refusal span and what its own reading matrices do on it.

    ``directions`` is (positions, layers, d_model): the base's difference-in-means direction at
    every position of the template's suffix, the last position last. Returns the tensors of
    the base's span file: ``directions`` as given, ``basis`` and ``n_ref``
    (``detect.refusal_span``), and for every reading matrix ``gram.<name>``, the Gram matrix of
    the matrix composed with its norm, on the basis. With a checkpoint's change to that Gram
    (``read_readers``), how strongly the checkpoint reads any direction inside the span can be
    worked out after its weights are gone (``span_read``). Reads weight files only."""
    basis, n_ref = detect.refusal_span(directions, n_controls, seed)
    span = {"directions": directions.float(), "basis": basis, "n_ref": torch.tensor([n_ref])}
    for m in detect.scan_readers(base_dir, base_dir, basis[:1], k=1, span=basis):
        if m.get("missing"):
            raise ValueError(f"{m['name']} could not be read from {base_dir}")
        span[f"gram.{m['name']}"] = m["span_base"]
    return span


def write_span(folder: str | Path, base: str, span: dict[str, torch.Tensor], made: dict) -> Path:
    path = span_path(folder, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in span.items()}, path)
    path.with_suffix(".json").write_text(json.dumps(made, indent=1))
    return path


def load_span(folder: str | Path, base: str) -> tuple[dict[str, torch.Tensor], dict]:
    path = span_path(folder, base)
    if not path.exists() or not path.with_suffix(".json").exists():
        raise FileNotFoundError(f"no refusal span for {base} in {folder}: make it with scripts/refusal_span.py")
    return load_file(path), json.loads(path.with_suffix(".json").read_text())


def span_reference_cosine(last: torch.Tensor, reference: torch.Tensor) -> float:
    """The smallest cosine, over the layers that have a direction, between a span's directions
    at the last position and the directions a main record kept. 0 if they are not comparable."""
    last, reference = last.float(), reference.float()
    if last.shape != reference.shape:
        return 0.0
    live = (last.norm(dim=1) > 0) | (reference.norm(dim=1) > 0)
    cosine = torch.nn.functional.cosine_similarity(last[live], reference[live], dim=1)
    return float(cosine.min()) if bool(live.any()) else 0.0


def checked_span(folder: str | Path, base: str, base_revision: str, base_files: list[dict], reference: torch.Tensor) -> dict:
    """The span file of ``base``, if it describes the weights and the directions a pass is about
    to score against. The reading side's own scores are measured against ``reference``, the
    last position's directions, so a span made from another revision of the base, or by
    another recipe, would put two different detectors in one row."""
    span, made = load_span(folder, base)
    if made.get("base_revision") != base_revision or _weights(made.get("base_files")) != _weights(base_files):
        raise ValueError(f"the refusal span of {base} in {folder} was made from other weights than the ones scored against")
    cosine = span_reference_cosine(span["directions"][-1], reference)
    if cosine < 0.999:
        raise ValueError(f"the refusal span of {base}: its last position is not the directions the main pass kept (cosine {cosine:.3f})")
    return span


def _ratio(c: torch.Tensor, base: torch.Tensor, change: torch.Tensor | None) -> torch.Tensor:
    """|E_cand r| / |E_base r| for directions given by their coefficients on the span's basis.
    Exactly 1 where nothing changed, and where the base does not read the direction at all."""
    before = torch.einsum("im,mn,in->i", c, base.double(), c)
    if change is None:
        return torch.ones(len(c), dtype=torch.float64)
    after = before + torch.einsum("im,mn,in->i", c, change.double(), c)
    return torch.where(before > 0, (after.clamp_min(0) / before.clamp_min(1e-300)).sqrt(), torch.ones_like(before))


def span_read(kept: dict[str, torch.Tensor], span: dict[str, torch.Tensor], name: str, directions: torch.Tensor) -> torch.Tensor:
    """How strongly the checkpoint's reading matrix ``name`` reads each row of ``directions``,
    over how strongly the base's does, from the record and the span file alone. Only the part
    of a direction inside the span counts: for one outside it this says nothing."""
    unit = torch.nn.functional.normalize(directions.double(), dim=1)
    return _ratio(unit @ span["basis"].double().T, kept.get(f"{name}.span_base", span[f"gram.{name}"]), kept.get(f"{name}.span"))


def span_scores(kept: dict[str, torch.Tensor], span: dict[str, torch.Tensor]) -> dict:
    """The reading side of a checkpoint against the base's directions at every position of the
    template's suffix, from the record and the base's span file.

    ``reader_scores`` looks along the last position's directions, the ones the main pass
    kept. A reader-side edit made from a direction taken at another position stops the
    readers reading that one, and shows only weakly along the last position's. Here:

    - ``reader_span_read_drop``: the read drop of ``reader_scores``, over the directions of
      every position; ``_med`` and ``_head`` as there; ``_by_position`` per position, the last
      position last; ``reader_span_last_read_drop`` is that last entry, which must agree with
      ``reader_read_drop`` although it is computed another way (from the Gram matrices)
    - ``reader_span_control_drop``: the same along the span's control directions, which have
      nothing to do with refusal: what a drop of that size means for this checkpoint
    - ``reader_span_alignment``: the alignment of ``reader_scores`` over every position, and
      ``_by_position``"""
    directions, basis = span["directions"].double(), span["basis"].double()
    n_pos, n_layers, d = directions.shape
    flat = directions.reshape(-1, d)
    unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30)
    on_basis, controls = unit @ basis.T, torch.eye(len(basis), dtype=torch.float64)[int(span["n_ref"]):]
    drops, control, aligned, head = [], [], [], 0.0
    for key, gram in span.items():
        if not key.startswith("gram."):
            continue
        name = key.removeprefix("gram.")
        base, change = kept.get(f"{name}.span_base", gram), kept.get(f"{name}.span")
        drop = 1 - _ratio(on_basis, base, change).view(n_pos, n_layers).min(dim=1).values
        if detect.layer_of(name) is None:
            head = float(drop.max())
            continue
        drops.append(drop)
        control.append(float(1 - _ratio(controls, base, change).min()) if len(controls) else 0.0)
        if f"{name}.eu" in kept:
            aligned.append((unit.float() @ kept[f"{name}.eu"].float()).norm(dim=1).view(n_pos, n_layers).max(dim=1).values)
    by_position = torch.stack(drops).max(dim=0).values if drops else torch.zeros(n_pos)
    alignment = torch.stack(aligned).max(dim=0).values if aligned else torch.zeros(n_pos)
    each = [float(x.max()) for x in drops]
    return {
        "reader_span_read_drop": max(each, default=0.0),
        "reader_span_read_drop_med": statistics.median(each) if each else 0.0,
        "reader_span_head_read_drop": head,
        "reader_span_last_read_drop": float(by_position[-1]),
        "reader_span_control_drop": max(control, default=0.0),
        "reader_span_alignment": float(alignment.max()),
        "reader_span_read_drop_by_position": [float(x) for x in by_position],
        "reader_span_alignment_by_position": [float(x) for x in alignment],
    }


def read_readers(base_dir: str | Path, cand_dir: str | Path, directions: torch.Tensor, k: int = 8,
                 span: dict[str, torch.Tensor] | None = None) -> dict:
    """The reading matrices of the weights in ``cand_dir`` against those in ``base_dir``
    (``detect.scan_readers``), as the reader pass keeps and reports them, whether the
    candidate is an uploaded checkpoint or an edit of the base made here:

    - ``kept``: the tensors of the record, ``directions`` among them as given
    - ``spectra``: per changed matrix, the spectrum summary of the stored matrix's change and
      of the change in the matrix composed with its norm
    - ``matrices`` read, those ``missing`` (absent, of another shape, or not finite) and those
      whose ``gain_changed``, by name, and ``n_layers``
    - ``fields``: the reader_* fields of the candidate's row

    With ``span``, the tensors of the base's span file (``make_span``), every reading matrix
    that changed also keeps ``<name>.span``, the change of its Gram matrix on the span, and the
    fields gain ``span_scores``. The base's own Gram matrices are the span file's; it is an
    error if the base read here gives others. A head compared over fewer tokens than the base
    has (a checkpoint that trimmed its vocabulary) keeps its own, ``<name>.span_base``."""
    n_layers = json.loads((Path(base_dir) / "config.json").read_text())["num_hidden_layers"]
    kept = {"reference.directions": directions}
    spectra, matrices, missing, gains = [], [], [], []
    for m in detect.scan_readers(base_dir, cand_dir, directions.float(), k, span=None if span is None else span["basis"]):
        if m.get("missing") or not m["finite"]:
            missing.append(m["name"])
            continue
        matrices.append(m["name"])
        if span is not None:
            if m["rows"] != m["rows_base"]:
                kept[f"{m['name']}.span_base"] = m["span_base"]
            else:
                want = span.get(f"gram.{m['name']}")
                if want is None or float((m["span_base"] - want.double()).abs().max()) > 1e-6 * float(want.abs().max()):
                    raise ValueError(f"{m['name']}: the base's reading matrix is not the one the span file was made from")
            if m["changed"] or m["gain_changed"]:
                kept[f"{m['name']}.span"] = m["span_change"].float()
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
    stored = detect.scores_from_spectra([s["stored"] for s in in_layers if s["stored"]], mid)
    stored.pop("per_matrix")
    effective = detect.scores_from_spectra([s["effective"] for s in in_layers], mid)
    fields = {
        **{key.replace("weight_", "reader_", 1): v for key, v in stored.items()},
        **{key.replace("weight_", "reader_eff_", 1): v for key, v in effective.items() if key.endswith("_energy_w")},
        **reader_scores(kept),  # from what is kept, at the precision it is kept in
        **shared_scores(kept),
        "reader_n_gain_changed": len(gains),
        "reader_head_changed": any(s["name"] == detect.HEAD and bool(s["stored"]) for s in spectra),
        **(span_scores(kept, span) if span is not None else {}),
    }
    return {"kept": kept, "spectra": spectra, "matrices": matrices, "missing": missing, "gain_changed": gains,
            "n_layers": n_layers, "fields": fields}


def code_versions(script: Path) -> dict:
    """The code and libraries behind a score: the commit, whether anything was uncommitted, a
    hash of the sources that produce the scores (the commit alone does not identify them while
    changes are uncommitted) and the library versions."""
    import platform
    import subprocess
    from importlib.metadata import version

    root = script.resolve().parents[1]

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    sources = sorted((root / "src" / "obliterated").glob("*.py")) + [script.resolve()]
    return {"obliterated": git("rev-parse", "--short", "HEAD") + ("+uncommitted" if git("status", "--porcelain") else ""),
            "source_sha256": hashlib.sha256(b"".join(f.read_bytes() for f in sources)).hexdigest(),
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("torch", "transformers", "huggingface_hub", "safetensors")}}


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def _same(a: float, b: float, tol: float = 1e-3) -> bool:
    """Agreement to a relative tolerance whatever the size: a light fine-tune moves a matrix
    by far less than 1, where ``_close`` would accept anything."""
    return abs(a - b) <= tol * abs(b)


def _answers(out: Path, repo: str, prompts: list[str], whose: str) -> list[str]:
    path = completions.path(out, repo)
    if not path.exists():
        return [f"no completions for {whose} {repo}"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r.get("prompt") for r in rows] != prompts:
        return [f"completions of {whose} {repo}: {len(rows)} rows that do not match the {len(prompts)} prompts"]
    texts = [r.get("completion") for r in rows]
    if not all(isinstance(t, str) for t in texts):
        return [f"completions of {whose} {repo}: a completion is not text"]
    problems = []
    empty = sum(not t.strip() for t in texts)
    if empty > len(texts) / 2:
        problems.append(f"completions of {whose} {repo}: {empty} of {len(texts)} are empty")
    cut = sum(map(completions.cut_off, texts))
    if cut > 0.2 * len(texts):
        problems.append(f"completions of {whose} {repo}: {cut} of {len(texts)} stop inside the reasoning")
    return problems


def _weights(files: list[dict] | None) -> list[tuple]:
    return sorted((f["name"], f["bytes"], f["blob"]) for f in files or [] if f["name"].endswith(".safetensors"))


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def validate_readers(row: dict, out: str | Path, main_out: str | Path) -> list[str]:
    """Everything wrong with what the reading-side pass saved for ``row``'s checkpoint; empty
    means the weights it downloaded for that are no longer needed. Reads only from disk.

    ``main_out`` is the results file of the pass that scored the checkpoint's behaviour. The
    two passes have to describe the same weights, so the revision and the weight files read
    here (name, size and content hash) must be the ones recorded there, for the checkpoint and
    for its base. Beyond that: every reading matrix of the model was read, each changed one
    has its factors, the gain of every norm in front of one was kept, the saved singular
    values agree with the saved spectra, and the record reproduces the row's scores."""
    out, repo = Path(out), row["repo"]
    undefined = {"reader_stable_rank"} if row.get("reader_n_matrices") == 0 else set()
    problems = [f"{k} is not finite" for k, v in row.items()
                if k.startswith("reader_") and isinstance(v, float) and not math.isfinite(v) and k not in undefined]
    info, main = _json(provenance_path(out, repo)), _json(provenance_path(main_out, repo)) or {}
    if info is None or not tensors_path(out, repo).exists():
        return [*problems, f"no reader record for {repo} beside {out.name}"]
    for key in ("revision", "base_revision"):
        if not REVISION.fullmatch(str(info.get(key))):
            problems.append(f"provenance: {key} is not a pinned commit ({info.get(key)!r})")
        elif info[key] != main.get(key):
            problems.append(f"{key}: the weights read are at {info[key][:12]}, the ones scored were at {str(main.get(key))[:12]}")
    for key, whose in (("files", "checkpoint"), ("base_files", "base")):
        if not _weights(info.get(key)) or _weights(info.get(key)) != _weights(main.get(key)):
            problems.append(f"the {whose}'s weights are not the files that were scored")
    if info.get("missing"):
        problems.append(f"missing or unreadable in the checkpoint: {', '.join(info['missing'])}")
    matrices, n_layers = info.get("matrices") or [], info.get("n_layers")
    if not isinstance(n_layers, int) or len(matrices) + len(info.get("missing") or []) != 5 * n_layers + 1:
        problems.append(f"{len(matrices)} reading matrices were read; a model of {n_layers} layers has {5 * (n_layers or 0) + 1}")

    kept = load_file(tensors_path(out, repo))
    problems += [f"{k} in the record is not finite" for k, v in kept.items() if not torch.isfinite(v.float()).all()]
    if "reference.directions" not in kept:
        return [*problems, "reference.directions is not in the record"]
    problems += [f"{name}: how strongly it reads the directions is not in the record" for name in matrices if f"{name}.read" not in kept]
    norms = {k.removeprefix("gain.") for k in kept if k.startswith("gain.")}
    lost = sorted({norm for name in matrices if (norm := detect.norm_in_front(name, norms)) not in norms})
    if lost:
        return [*problems, *(f"{norm}: the gain of this norm is not in the record" for norm in lost)]
    spectra = _json(spectra_path(out, repo))
    in_layers = None if spectra is None else sum(detect.layer_of(m["name"]) is not None and bool(m.get("stored")) for m in spectra)
    if in_layers != row.get("reader_n_changed"):
        problems.append(f"spectra: {in_layers} changed layer matrices saved, the row counts {row.get('reader_n_changed')}")
    for m in spectra or []:
        for which, values, vectors in (("stored", "s", "u"), ("effective", "es", "eu")):
            saved, s, u = m.get(which), kept.get(f"{m['name']}.{values}"), kept.get(f"{m['name']}.{vectors}")
            if not saved:
                continue
            if s is None or u is None:
                problems.append(f"{m['name']}: the singular values or directions of its {which} change are not in the record")
            elif not _same(float(s[0]), math.sqrt(saved["top1_energy"]) * saved["fro_norm"]):
                problems.append(f"{m['name']}: saved top singular value {float(s[0]):.5g} of its {which} change disagrees "
                                f"with the spectrum ({math.sqrt(saved['top1_energy']) * saved['fro_norm']:.5g})")
            elif float(s.double().pow(2).sum().sqrt()) > saved["fro_norm"] * (1 + 1e-3):
                problems.append(f"{m['name']}: the norm {saved['fro_norm']:.5g} of its {which} change is below what its "
                                "saved singular values carry")
    for key, value in reader_scores(kept).items():
        if not isinstance(row.get(key), (int, float)) or not _close(row[key], value):
            problems.append(f"{key}: the row says {row.get(key)}, the record gives {value:.5f}")
    if any(k.startswith("reader_shared_") for k in row):  # rows made before the score existed have none
        for key, value in shared_scores(kept).items():
            if not isinstance(row.get(key), (int, float)) or not _close(row[key], value):
                problems.append(f"{key}: the row says {row.get(key)}, the record gives {value:.5f}")
    if any(k.startswith("reader_span_") for k in row):
        problems += _span_problems(row, info.get("span") or {}, kept, matrices)
    return problems


def _span_problems(row: dict, made: dict, kept: dict[str, torch.Tensor], matrices: list[str]) -> list[str]:
    """What is wrong with the part of a reader record that rests on the base's span file: the
    file must be the one the record was made with, every matrix that changed must have kept
    its change on the span, and the two together must give back the row's span scores."""
    path = Path(made.get("path") or "")
    if not made or not path.is_file():
        return [f"the refusal span the record was made with is not on disk ({made.get('path')})"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != made.get("sha256"):
        return [f"the refusal span at {path} is not the file the record was made with"]
    span = load_file(path)
    problems = [f"{name}: the refusal span has no Gram matrix for it" for name in matrices if f"gram.{name}" not in span]
    problems += [f"{name}: it changed, and its change on the refusal span is not in the record"
                 for name in sorted(k.removesuffix(".es") for k in kept if k.endswith(".es")) if f"{name}.span" not in kept]
    if problems:
        return problems
    for key, value in span_scores(kept, span).items():
        got = row.get(key)
        if isinstance(value, list):
            if not isinstance(got, list) or len(got) != len(value) or not all(_close(g, v) for g, v in zip(got, value)):
                problems.append(f"{key}: the row says {got}, the record gives {[round(v, 5) for v in value]}")
        elif not isinstance(got, (int, float)) or not _close(got, value):
            problems.append(f"{key}: the row says {got}, the record gives {value:.5f}")
    if isinstance(row.get("reader_read_drop"), (int, float)) and not _close(row.get("reader_span_last_read_drop", math.nan),
                                                                              row["reader_read_drop"], tol=5e-3):
        problems.append(f"the read drop at the last position is {row.get('reader_span_last_read_drop')} from the span "
                        f"and {row['reader_read_drop']} read directly")
    return problems


def validate(row: dict, out: str | Path, prompts: list[str], weights: bool = True) -> list[str]:
    """Everything wrong with what was saved for ``row``'s checkpoint; empty means its weights
    are no longer needed. Reads only from disk. Beyond presence and finiteness it checks that
    the saved per-prompt measurements reproduce the row's scores, that the saved singular
    values agree with the separately computed spectra to one part in a thousand, and that no
    stored norm is smaller than the singular values saved beside it."""
    out, repo = Path(out), row["repo"]
    problems = [f"{k} is missing or not finite" for k in ROW_KEYS
                if not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])]
    # The stable rank is a mean over the mid-stack matrices that changed: undefined when none
    # did, as for a null control or a fine-tune that only touched the outer layers.
    undefined = {"weight_stable_rank"} if row.get("weight_n_matrices") == 0 else set()
    problems += [f"{k} is not finite" for k, v in row.items()
                 if k.startswith("weight_") and isinstance(v, float) and not math.isfinite(v) and k not in undefined]
    problems += _answers(out, repo, prompts, "checkpoint") + _answers(out, row["base"], prompts, "base")

    info = json.loads(provenance_path(out, repo).read_text()) if provenance_path(out, repo).exists() else {}
    for key in ("revision", "base_revision"):
        if not REVISION.fullmatch(str(info.get(key))):
            problems.append(f"provenance: {key} is not a pinned commit ({info.get(key)!r})")
    if not info.get("prompts") or not (out.parent / "prompts" / info["prompts"]).exists():
        problems.append(f"provenance: the prompt set {info.get('prompts')!r} is not on disk")

    if not tensors_path(out, repo).exists():
        return [*problems, f"no record file {tensors_path(out, repo).name}"]
    kept = load_file(tensors_path(out, repo))
    problems += [f"{k} is not in the record" for k in TENSORS if k not in kept]
    problems += [f"{k} in the record is not finite" for k, v in kept.items() if not torch.isfinite(v.float()).all()]
    if problems:
        return problems

    saved = detect.behaviour_summary({k.removeprefix("behaviour."): v for k, v in kept.items() if k.startswith("behaviour.")})
    mid = detect.mid_layers(len(kept["activation.harmful"]))
    saved["activation_gap"] = detect.activation_gap(
        detect.separation_of(kept["activation.harmful_base"], kept["activation.harmless_base"]),
        detect.separation_of(kept["activation.harmful"], kept["activation.harmless"]), mid)
    for key, name in (("kl_harmful", "behaviour_kl_harmful"), ("refusal_score_drop", "behaviour_refusal_drop"),
                      ("bypass_rate", "bypass_rate"), ("baseline_bypass_rate", "baseline_bypass_rate"),
                      ("activation_gap", "activation_gap")):
        if not _close(saved[key], row[name]):
            problems.append(f"{name}: the row says {row[name]:.5f}, the saved measurements give {saved[key]:.5f}")
    if len(kept["behaviour.kl"]) != row["n_harmful"]:
        problems.append(f"n_harmful: the row says {row['n_harmful']}, the record holds {len(kept['behaviour.kl'])} prompts")

    if weights:
        spectra = json.loads(spectra_path(out, repo).read_text()) if spectra_path(out, repo).exists() else None
        if spectra is None or len(spectra) != row.get("weight_n_changed"):
            problems.append(f"spectra: {None if spectra is None else len(spectra)} matrices saved, the row counts {row.get('weight_n_changed')}")
        elif not spectra and row.get("label") != "base":  # a null control is meant to match
            problems.append("no writer matrix differs from the base: an unmodified copy, or the wrong files")
        for m in spectra or []:
            s, u = kept.get(f"{m['name']}.s"), kept.get(f"{m['name']}.u")
            if s is None or u is None:
                problems.append(f"{m['name']}: its singular values or directions are not in the record")
            elif not _same(float(s[0]), math.sqrt(m["top1_energy"]) * m["fro_norm"]):
                problems.append(f"{m['name']}: saved top singular value {float(s[0]):.5g} disagrees with the "
                                f"spectrum ({math.sqrt(m['top1_energy']) * m['fro_norm']:.5g})")
        for key, value in alignment(kept).items():
            if key in row and not _close(row[key], value):
                problems.append(f"{key}: the row says {row[key]:.5f}, the saved factors give {value:.5f}")
        for key in kept:  # the embedding and the output head: a norm, and the factors beside it
            name = key.removesuffix(".fro")
            if name != key and f"{name}.s" in kept:
                norm, carried = float(kept[key]), float(kept[f"{name}.s"].double().pow(2).sum().sqrt())
                if carried > norm * (1 + 1e-3):
                    problems.append(f"{name}: stored norm {norm:.5g} is below the {carried:.5g} its saved singular values carry")
    return problems
