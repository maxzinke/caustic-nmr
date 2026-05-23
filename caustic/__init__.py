"""CAUSTIC: Conformation-Aware Uncertainty and Shift predicTion from
proteIn Conformer ensembles.

A ~740K-parameter PaiNN equivariant graph neural network that predicts
H, HA, N, CA, CB, and C' backbone shifts from any PDB, mmCIF, or
AlphaFold model. Ships as a lightweight ONNX Runtime package with
calibrated uncertainty estimates and NEF / NMR-STAR / CSV / JSON output.

v0.3.0 — trained on carbon-aggressive label-noise-cleaned BMRB labels:
−4.37% relative composite MAE on cc.test (n=614, paired bootstrap,
CI [−0.041, −0.024] excludes zero) vs the previous PaiNN baseline.
Heavy atoms benefit most: CA −5.7%, CB −5.5%, C −3.7%; H/HA/N
unchanged (proton noise is biological, not referencing drift).

v0.3.0 also embeds slim SA16 post-prediction calibration: global
per-nucleus offsets + a +1.36 ppm CYS-CB modifier for disulfide-bonded
cysteines (detected via Sγ-Sγ distance gate). On by default; opt out
with ``predict_shifts_onnx(..., apply_calibrator=False)``.

Quick start::

    caustic input.pdb -o shifts.nef

Python API::

    from caustic import predict_shifts
    result = predict_shifts("input.pdb")
    print(result.mean["CA"])  # per-residue CA shifts in ppm
"""
from __future__ import annotations

__version__ = "0.3.0"

from .inference import ShiftPrediction, predict_shifts, predict_shifts_onnx

__all__ = [
    "__version__",
    "ShiftPrediction",
    "predict_shifts",
    "predict_shifts_onnx",
]
