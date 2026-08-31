"""SA16 v2 slim post-prediction calibration.

After the ONNX GNN produces per-(residue, nucleus) predictions, this
module applies:

  1. Global per-nucleus offset (μ_global) — captures the residual
     systematic bias between training-time and observed shift
     distributions. Small (<0.03 ppm at v0.3.0 because the carbon-
     cleaned training set already absorbed most of it).

  2. CYS-CB modifier — adjusts CB predictions on cysteines based on
     their oxidation state (disulfide / metal-bound / reduced-free).
     The largest single SA16 correction: disulfide CB carries a
     +1.36 ppm shift relative to the model's default that the GNN
     under-predicts on average. Disulfide partners are detected with
     a simple Sγ-Sγ distance gate (<2.5 Å between any two CYS).

What's NOT in this slim version: per-(DSSP class × aromatic × rSASA)
stratum offsets. They require an aromatic-ring neighbour count that the
packaged graph pipeline does not compute. The slim version keeps the
two dominant levers (global offsets, CYS-CB oxidation state); the
measured cost of leaving the strata out is reported in
``docs/BENCHMARKS.md`` of the repository. See the ``description`` field
of ``data/sa16_calibrator_v2.json``.

Apply automatically inside ``predict_shifts_onnx``; opt out by setting
``apply_calibrator=False``.
"""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np


# Atom-index conventions (must match the model's nucleus ordering).
_NUC_INDEX = {"H": 0, "HA": 1, "N": 2, "CA": 3, "CB": 4, "C": 5}
_CYS_AA_THREE = "CYS"


def load_calibrator(path: str | Path | None = None) -> dict[str, Any]:
    """Load the slim SA16 calibrator JSON.

    Default loads the v0.3.0 shipped artefact at
    ``caustic/data/sa16_calibrator_v2.json``.
    """
    if path is None:
        # importlib.resources to support both editable installs and wheels.
        path = files("caustic.data") / "sa16_calibrator_v2.json"
    with open(path) as f:
        return json.load(f)


def detect_disulfides(
    positions: np.ndarray,
    residue_names: list[str],
    sg_atom_indices: list[int | None],
    cutoff: float = 2.5,
) -> set[int]:
    """Return the set of residue indices that are in a Sγ-Sγ bond.

    Parameters
    ----------
    positions: (N_atoms, 3) coordinates.
    residue_names: 3-letter codes per residue (length R).
    sg_atom_indices: for each residue, the atom index of its Sγ (or None
        if no Sγ atom — non-CYS or missing).
    cutoff: max Sγ-Sγ distance to count as a disulfide bond (Å). 2.5 Å is
        the standard cutoff (typical S-S bond length ~2.05 Å).
    """
    cys_indices = [i for i, name in enumerate(residue_names) if name == _CYS_AA_THREE]
    disulfide_partners: set[int] = set()
    cys_with_sg = [i for i in cys_indices if sg_atom_indices[i] is not None]
    if len(cys_with_sg) < 2:
        return disulfide_partners
    sg_coords = np.array([positions[sg_atom_indices[i]] for i in cys_with_sg], dtype=np.float64)
    for ai, i in enumerate(cys_with_sg):
        for aj, j in enumerate(cys_with_sg):
            if aj <= ai:
                continue
            d = float(np.linalg.norm(sg_coords[ai] - sg_coords[aj]))
            if d <= cutoff:
                disulfide_partners.add(i)
                disulfide_partners.add(j)
    return disulfide_partners


def find_cys_sg_indices(g) -> list[int | None]:
    """For each residue in the PyG graph, return the atom index of its
    Sγ atom (CYS only). None for non-CYS or missing Sγ."""
    R = int(g.num_residues)
    out: list[int | None] = [None] * R
    if not hasattr(g, "atom_names") and not hasattr(g, "element"):
        return out
    # The graph stores per-atom: element, residue_index, optionally atom_name.
    # Heuristic: a CYS Sγ is an atom in a CYS residue whose element index = 5
    # (sulfur). Use the per-atom residue_idx field to bucket.
    elem = g.element.detach().cpu().numpy() if hasattr(g.element, "detach") else np.asarray(g.element)
    res_idx = g.residue_idx.detach().cpu().numpy() if hasattr(g.residue_idx, "detach") else np.asarray(g.residue_idx)
    # Caustic's element encoding: H=0, C=1, N=2, O=3, S=4 (per features.py
    # ELEMENT_VOCAB). Sulfur is index 4.
    SULFUR_IDX = 4
    sulfur_mask = (elem == SULFUR_IDX)
    if not sulfur_mask.any():
        return out
    sulfur_atoms = np.where(sulfur_mask)[0]
    for a in sulfur_atoms:
        r = int(res_idx[a])
        if 0 <= r < R and out[r] is None:
            out[r] = int(a)
    return out


def apply_calibrator(
    mean_arr: np.ndarray,         # [R, 6] predicted means (NaN where invalid)
    residue_names: list[str],
    disulfide_residue_indices: set[int],
    calibrator: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply slim SA16 calibration. Returns a NEW [R, 6] array.

    Steps:
      1. Add μ_global[nucleus] to every valid prediction
      2. For CYS residues in disulfide bonds, add the disulfide CB modifier
         (residues NOT detected as disulfide are treated as reduced_free)

    Metal-bound and heme-thioether CYS are NOT detected here (would need
    HETATM analysis); they fall through to reduced_free. The misclass
    cost is small (~0.06 ppm difference for metal_bound).
    """
    if calibrator is None:
        calibrator = load_calibrator()
    out = mean_arr.copy()

    # 1. Global per-nucleus offsets
    g_off = calibrator.get("global_offsets", {})
    for nuc, ni in _NUC_INDEX.items():
        delta = float(g_off.get(nuc, 0.0))
        if delta != 0.0:
            mask = np.isfinite(out[:, ni])
            out[mask, ni] += delta

    # 2. CYS-CB modifier
    cys_mods = calibrator.get("cys_modifiers", {})
    cb_idx = _NUC_INDEX["CB"]
    delta_disulfide = float(cys_mods.get("disulfide", {}).get("offset", 0.0))
    delta_reduced = float(cys_mods.get("reduced_free", {}).get("offset", 0.0))
    for r, name in enumerate(residue_names):
        if name != _CYS_AA_THREE:
            continue
        if not np.isfinite(out[r, cb_idx]):
            continue
        if r in disulfide_residue_indices:
            out[r, cb_idx] += delta_disulfide
        else:
            out[r, cb_idx] += delta_reduced

    return out
