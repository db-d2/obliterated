"""Pick a device and dtype that work on the PC (CUDA), the Mac (MPS) or a CPU-only box."""

from __future__ import annotations

import torch


def best_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        # bf16 works on recent macOS, but fp16 is faster and better supported by MPS kernels.
        return torch.float16
    return torch.float32


def describe() -> str:
    d = best_device()
    parts = [f"device={d}", f"torch={torch.__version__}"]
    if d.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        parts.append(f"gpu={props.name} ({props.total_memory / 2**30:.1f} GiB)")
        parts.append(f"cuda={torch.version.cuda}")
    return " ".join(parts)


if __name__ == "__main__":
    print(describe())
