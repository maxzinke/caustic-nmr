"""Provenance stamping: which package, model and calibrator produced an output.

Every writer (NEF, NMR-STAR, CSV, JSON) records the package version, the
model file name and its SHA-256, the calibrator version and an ISO-8601
UTC timestamp, so any prediction file can be traced back to the exact
artifacts that produced it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from caustic._version import __version__

PROGRAM_NAME = "caustic-nmr"
BUNDLED_MODEL_NAME = "best_v2_carbons.onnx"
BUNDLED_CALIBRATOR_NAME = "sa16_calibrator_v2.json"


def bundled_model_path() -> Path:
    """Path of the ONNX model shipped inside the wheel."""
    return Path(str(files("caustic.data") / BUNDLED_MODEL_NAME))


def bundled_calibrator_path() -> Path:
    """Path of the post-prediction calibrator shipped inside the wheel."""
    return Path(str(files("caustic.data") / BUNDLED_CALIBRATOR_NAME))


@lru_cache(maxsize=16)
def file_sha256(path: str) -> str:
    """SHA-256 hex digest of a file (cached per path)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrator_version(path: str | Path | None = None) -> str:
    """The ``version`` field of a calibrator JSON (bundled one by default)."""
    p = Path(path) if path is not None else bundled_calibrator_path()
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("version", "unknown"))
    except (OSError, ValueError):
        return "unknown"


def model_provenance(
    model_path: str | Path,
    *,
    calibrator_applied: bool,
    calibrator_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the provenance record attached to a :class:`ShiftPrediction`."""
    p = Path(model_path)
    try:
        sha = file_sha256(str(p.resolve()))
    except OSError:
        sha = "unknown"
    return {
        "program": PROGRAM_NAME,
        "version": __version__,
        "model_file": p.name,
        "model_sha256": sha,
        "calibrator": calibrator_version(calibrator_path) if calibrator_applied else "none",
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def resolve_provenance(prediction: Any) -> dict[str, Any]:
    """Return the prediction's provenance, or a best-effort default."""
    prov = getattr(prediction, "provenance", None)
    if prov:
        return dict(prov)
    return {
        "program": PROGRAM_NAME,
        "version": __version__,
        "model_file": "unknown",
        "model_sha256": "unknown",
        "calibrator": "unknown",
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def provenance_line(prov: dict[str, Any]) -> str:
    """One-line human/machine-readable stamp used in file headers."""
    return (
        f"{prov.get('program', PROGRAM_NAME)} {prov.get('version', __version__)} "
        f"model={prov.get('model_file', 'unknown')} "
        f"sha256={str(prov.get('model_sha256', 'unknown'))[:12]} "
        f"calibrator={prov.get('calibrator', 'unknown')} "
        f"date={prov.get('date', 'unknown')}"
    )
