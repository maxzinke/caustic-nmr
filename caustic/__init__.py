"""CAUSTIC: Conformation-Aware Uncertainty and Shift predicTion from
proteIn Conformer ensembles.

A PaiNN equivariant graph neural network (~741K parameters) that predicts
H, HA, N, CA, CB and C' backbone chemical shifts from any PDB, mmCIF or
AlphaFold model. Inference runs through ONNX Runtime; the model weights
and a post-prediction calibrator are bundled inside the package. Output
formats: NEF, NMR-STAR, CSV, JSON — every file carries a provenance
stamp (package version, model SHA-256, calibrator version, date).

The bundled calibrator (``sa16_v2_carbons_slim``) applies global
per-nucleus offsets plus a CYS-CB modifier for disulfide-bonded
cysteines (Sγ-Sγ distance gate). It is on by default; opt out with
``predict_shifts_onnx(..., apply_calibrator=False)``.

Benchmark numbers live in ``docs/BENCHMARKS.md`` and ``benchmarks/`` of
the repository, not here.

Quick start::

    caustic input.pdb -o shifts.nef

Python API::

    from caustic import predict_shifts_onnx
    result = predict_shifts_onnx("input.pdb")
    print(result.mean["CA"])  # per-residue CA shifts in ppm
"""
from __future__ import annotations

from ._version import __version__
from .inference import ShiftPrediction, predict_shifts, predict_shifts_onnx

__all__ = [
    "__version__",
    "ShiftPrediction",
    "predict_shifts",
    "predict_shifts_onnx",
]
