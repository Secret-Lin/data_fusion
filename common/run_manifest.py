from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, workspace: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        display = str(resolved.relative_to(workspace))
    except ValueError:
        display = str(resolved)
    record: dict[str, Any] = {"path": display, "exists": resolved.is_file()}
    if resolved.is_file():
        stat = resolved.stat()
        record.update(size_bytes=stat.st_size, sha256=_sha256(resolved))
    return record


def _git_metadata(workspace: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=workspace, capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1")
    return {
        "commit": commit or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
    }


def build_run_manifest(
    *,
    cfg: Any,
    workspace: str | Path,
    artifact_paths: Iterable[str | Path] = (),
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    packages = {}
    for name in ("numpy", "opencv-python", "torch", "gtsam", "trimesh", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    unique_paths = {str(Path(path)) for path in artifact_paths if str(path).strip()}
    return {
        "schema_version": 1,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "git": _git_metadata(root),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "command": sys.argv,
            "packages": packages,
            **_jsonable(runtime or {}),
        },
        "configuration": _jsonable(cfg),
        "artifacts": [_file_record(Path(path), root) for path in sorted(unique_paths)],
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
