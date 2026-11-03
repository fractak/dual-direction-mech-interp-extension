"""Runtime provenance without credentials or machine environment dumps."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_metadata():
    packages = {}
    for name in ("numpy", "torch", "transformers", "accelerate", "bitsandbytes", "peft", "trl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    backend = None
    if packages["torch"] is not None:
        import torch

        backend = {
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    return {
        "torch_backend": backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def adapter_manifest(path):
    if path is None:
        return None
    path = Path(path)
    files = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and (p.name == "adapter_config.json" or p.name.startswith("adapter_model"))
    )
    if not (path / "adapter_config.json").is_file() or not any(
        p.name in ("adapter_model.safetensors", "adapter_model.bin") for p in files
    ):
        raise ValueError("Adapter must contain config and adapter_model weights")
    return {"files": {str(p.relative_to(path)): sha256_file(p) for p in files}}
