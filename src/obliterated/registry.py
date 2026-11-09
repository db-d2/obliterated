"""Registry of public checkpoints: abliterated models, their bases, and benign fine-tune controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "registry" / "checkpoints.yaml"


@dataclass
class Checkpoint:
    repo: str
    base: str
    label: str  # abliterated | abliterated+finetune | benign_finetune | base
    family: str = ""
    params_b: float | None = None
    method: str = ""
    verified: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def load_registry(path: Path = REGISTRY) -> list[Checkpoint]:
    data = yaml.safe_load(path.read_text())
    return [Checkpoint(**row) for row in data["checkpoints"]]


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
