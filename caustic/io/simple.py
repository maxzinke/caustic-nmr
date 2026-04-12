"""CSV and JSON writers for :class:`ShiftPrediction`.

Flat one-row-per-(residue × nucleus) formats for users who just want to
load the predictions into pandas, Excel, or a JSON-consuming tool. The
CSV schema matches what downstream scripts can read without external
NMR-format libraries.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from caustic.features import BACKBONE_NUCLEI
from caustic.inference import ShiftPrediction


_CSV_COLUMNS = (
    "chain_code",
    "sequence_code",
    "residue_name",
    "atom_name",
    "element",
    "isotope_number",
    "shift_ppm",
    "sigma_ppm",
)

_NUCLEUS_META: dict[str, tuple[str, int]] = {
    "H":  ("H", 1),
    "HA": ("H", 1),
    "N":  ("N", 15),
    "CA": ("C", 13),
    "CB": ("C", 13),
    "C":  ("C", 13),
}


def _iupac_atom_name(residue_name: str, nucleus: str) -> str:
    """Return the IUPAC/NEF-compatible atom label for ``(residue, nucleus)``.

    Glycine HA is emitted as ``HA%`` (degenerate) because the shift
    predictor outputs a single averaged HA value, not a stereospecific
    HA2 / HA3 pair. Matches the NEF and NMR-STAR writers so all four
    formats stay consistent.
    """
    if residue_name == "GLY" and nucleus == "HA":
        return "HA%"
    return nucleus


def _rows(
    prediction: ShiftPrediction,
    chain_code: str = "A",
) -> list[dict[str, Any]]:
    n_res = len(prediction.seq_ids)
    names = prediction.residue_names or (["UNK"] * n_res)
    if len(names) != n_res:
        names = (list(names) + ["UNK"] * n_res)[:n_res]

    rows: list[dict[str, Any]] = []
    for nuc in BACKBONE_NUCLEI:
        means = prediction.mean.get(nuc)
        stds = prediction.std.get(nuc)
        if means is None:
            continue
        element, isotope = _NUCLEUS_META[nuc]
        for i in range(n_res):
            m = float(means[i])
            if not np.isfinite(m):
                continue
            s = float(stds[i]) if stds is not None else float("nan")
            rows.append(
                {
                    "chain_code": chain_code,
                    "sequence_code": int(prediction.seq_ids[i]),
                    "residue_name": names[i],
                    "atom_name": _iupac_atom_name(names[i], nuc),
                    "element": element,
                    "isotope_number": isotope,
                    "shift_ppm": m,
                    "sigma_ppm": s if np.isfinite(s) else None,
                }
            )
    return rows


def write_csv(
    prediction: ShiftPrediction,
    path: str | Path,
    *,
    chain_code: str = "A",
) -> Path:
    """Write a flat one-row-per-shift CSV.

    Columns: ``chain_code, sequence_code, residue_name, atom_name,
    element, isotope_number, shift_ppm, sigma_ppm``. NaN / missing
    predictions are skipped. Missing σ (rare) is written as an empty
    cell.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(prediction, chain_code=chain_code)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: (row[k] if row.get(k) is not None else "")
                    for k in _CSV_COLUMNS
                }
            )
    return out


def write_json(
    prediction: ShiftPrediction,
    path: str | Path,
    *,
    chain_code: str = "A",
    indent: int = 2,
) -> Path:
    """Write a JSON document with a top-level ``shifts`` array.

    Each element mirrors the CSV schema with proper numeric types.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(prediction, chain_code=chain_code)
    doc = {
        "source": prediction.pdb_path,
        "num_conformers": int(prediction.num_conformers),
        "num_residues": len(prediction.seq_ids),
        "nuclei": list(BACKBONE_NUCLEI),
        "shifts": rows,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=indent)
    return out
