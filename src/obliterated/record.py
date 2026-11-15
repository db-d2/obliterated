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

from . import completions, detect

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
