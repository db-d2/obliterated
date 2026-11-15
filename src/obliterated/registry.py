"""Registry of public checkpoints: abliterated models, their bases, and benign fine-tune controls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "registry" / "checkpoints.yaml"


@dataclass
class Checkpoint:
    repo: str
    base: str
    label: str  # abliterated | abliterated+finetune | benign_finetune | other_edit | unlearned | base
    family: str = ""
    params_b: float | None = None
    method: str = ""
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def load_registry(path: Path = REGISTRY) -> list[Checkpoint]:
    data = yaml.safe_load(path.read_text())
    return [Checkpoint(**row) for row in data["checkpoints"]]


def fetch(repo: str, revision: str | None = None) -> Path:
    """Download a checkpoint's top-level weights, config and tokenizer files and return the
    snapshot directory. Subfolders (duplicate exports, training checkpoints), GGUF files and
    LoRA adapters are skipped, so the safetensors in the result are exactly one model.

    ``revision`` pins a commit. A pass that goes back to a checkpoint scored earlier has to
    read the weights that were scored then, not whatever the repo holds now."""
    from huggingface_hub import HfApi, snapshot_download

    keep = (".safetensors", ".json", ".txt", ".jinja", ".model")
    files = [s.rfilename for s in HfApi().model_info(repo, revision=revision).siblings]
    want = [f for f in files if "/" not in f and f.endswith(keep) and not f.startswith("adapter")]
    return Path(snapshot_download(repo, revision=revision, allow_patterns=want))


def bases(checkpoints: Iterable[Checkpoint]) -> set[str]:
    """Every repo another row is diffed against. These have to stay on disk."""
    return {c.base for c in checkpoints}


def evict(repo: str, protect: Iterable[str], cache: Path | None = None) -> int:
    """Delete one checkpoint from the Hugging Face cache and return the bytes freed (0 if it
    was not cached). For batches that score a checkpoint and move on because the disk cannot
    hold the whole set. Refuses any repo in ``protect``: pass ``bases(load_registry())``.

    Only this repo's own folder is removed. Its large files may instead sit in the cache-wide
    shared store (huggingface_hub >= 2), where a byte-identical copy in another repo points at
    the same file; those are handed to the library's own sweep, which deletes a shared file
    only when no repo references it any more. Deleting by revision hash is avoided on purpose:
    a duplicated repo carries the same commit hash as its source."""
    import shutil

    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    if repo in set(protect):
        raise ValueError(f"{repo} is protected: other checkpoints are diffed against it")
    cache = Path(cache or constants.HF_HUB_CACHE)
    root = cache / repo_folder_name(repo_id=repo, repo_type="model")
    if not root.is_dir():
        return 0
    try:
        from huggingface_hub.utils import _shared_blobs as store
    except ImportError:  # an older huggingface_hub: every file is inside the repo folder
        store = None
    blobs = list((root / "blobs").iterdir()) if store and (root / "blobs").is_dir() else []
    shared = {target for blob in blobs if (target := store.shared_blob_target(blob, cache))}
    freed = sum(f.lstat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())
    shutil.rmtree(root)
    return freed + sum(store.sweep_shared_blob(target, cache_dir=cache) for target in shared)


def verify(checkpoints: list[Checkpoint]) -> list[tuple[str, str]]:
    """Check each repo and its base exist on the Hub. Returns (repo, status) pairs."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    results = []
    for c in checkpoints:
        for repo in (c.repo, c.base):
            try:
                api.model_info(repo)
                status = "ok"
            except GatedRepoError:
                status = "gated (exists; accept the license on HF)"
            except RepositoryNotFoundError:
                status = "NOT FOUND"
            except Exception as e:  # network errors etc.
                status = f"error: {type(e).__name__}"
            results.append((repo, status))
    return list(dict.fromkeys(results))
