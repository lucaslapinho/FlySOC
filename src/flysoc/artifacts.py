"""Unique run directories, strict serialization and reproducibility manifests."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from typing import Any
import uuid

import numpy as np
import pandas as pd

from .config import ROOT


def run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:6]}"


def safe_run_path(identifier: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Run identifier contains unsafe path characters")
    return ROOT / "results" / identifier


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "PyYAML", "pytest"]
    source_files = sorted([*ROOT.glob("src/flysoc/*.py"), *ROOT.glob("scripts/*.py"),
                           *ROOT.glob("tests/*.py"), ROOT / "pyproject.toml", ROOT / "requirements.txt"])
    return {"python": platform.python_version(), "platform": platform.platform(),
            "project_path": str(ROOT),
            "dependencies": {p: importlib.metadata.version(p) for p in packages},
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files}}
