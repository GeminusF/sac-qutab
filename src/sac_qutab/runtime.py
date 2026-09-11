from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


_LOGICAL_ROOT_ENV = "SAC_QUTAB_LOGICAL_ROOT"
_PHYSICAL_ROOT_ENV = "SAC_QUTAB_PHYSICAL_ROOT"


def _relocation_roots() -> tuple[Path, Path] | None:
    logical_text = os.environ.get(_LOGICAL_ROOT_ENV)
    physical_text = os.environ.get(_PHYSICAL_ROOT_ENV)
    if not logical_text and not physical_text:
        return None
    if not logical_text or not physical_text:
        raise RuntimeError(f"{_LOGICAL_ROOT_ENV} and {_PHYSICAL_ROOT_ENV} must be set together")
    logical, physical = Path(logical_text), Path(physical_text)
    if not logical_text.startswith("/") or not physical.is_absolute():
        raise RuntimeError("logical and physical relocation roots must be absolute")
    return logical, physical


def runtime_path(path: str | Path) -> Path:
    """Map a frozen logical root path to its writable physical location."""
    candidate = Path(path)
    roots = _relocation_roots()
    if roots is None:
        return candidate
    logical, physical = roots
    try:
        relative = candidate.relative_to(logical)
    except ValueError:
        return candidate
    return physical / relative


def logical_path(path: str | Path) -> Path:
    """Map a physical path back to its frozen form for identity hashing."""
    candidate = Path(path)
    roots = _relocation_roots()
    if roots is None:
        return candidate
    logical, physical = roots
    try:
        relative = candidate.relative_to(physical)
    except ValueError:
        return candidate
    return logical / relative


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with runtime_path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = runtime_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_bytes(path, body.encode("utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with runtime_path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
                rows.append(value)
    return rows


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed; exact resume is impossible")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scikit-learn", "scipy", "wandb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source: dict[str, Any] = {"declared_version": os.environ.get("SAC_QUTAB_CODE_VERSION")}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5,
        )
        source["git_commit"] = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        source["git_commit"] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "path_relocation": {
            "logical_root": os.environ.get(_LOGICAL_ROOT_ENV),
            "physical_root": os.environ.get(_PHYSICAL_ROOT_ENV),
        },
        "hostname": platform.node(),
        "source": source,
    }


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = runtime_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
