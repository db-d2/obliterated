"""Check that every checkpoint and base in registry/checkpoints.yaml exists on the Hub."""

from obliterated.registry import load_registry, verify

if __name__ == "__main__":
    for repo, status in verify(load_registry()):
        print(f"{status:>45}  {repo}")
