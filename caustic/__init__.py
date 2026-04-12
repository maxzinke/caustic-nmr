"""CAUSTIC: Conformation-Aware Uncertainty and Shift predicTion from
proteIn Conformer ensembles.

A 400K-parameter SchNet graph neural network that predicts H, HA, N, CA,
CB, and C' backbone shifts from any PDB, mmCIF, or AlphaFold model. Ships
as a lightweight ONNX Runtime package with calibrated uncertainty estimates
and NEF / NMR-STAR / CSV / JSON output.

Quick start::

    caustic input.pdb -o shifts.nef

Python API::

    from caustic import predict_shifts
    result = predict_shifts("input.pdb")
    print(result.mean["CA"])  # per-residue CA shifts in ppm
"""
from __future__ import annotations

__version__ = "0.1.0"

from .inference import ShiftPrediction, predict_shifts, predict_shifts_onnx

__all__ = [
    "__version__",
    "ShiftPrediction",
    "predict_shifts",
    "predict_shifts_onnx",
]
