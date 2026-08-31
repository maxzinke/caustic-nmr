"""Bundled assets shipped with the caustic package.

Currently contains:
    best_v2_carbons.onnx — production checkpoint (unchanged since 0.3.0)
        PaiNN backbone, 741,024 parameters, trained on carbon-aggressive
        label-noise-cleaned BMRB labels. Benchmark numbers live in
        docs/BENCHMARKS.md and are regenerated from benchmarks/.
    sa16_calibrator_v2.json — post-prediction offsets (global + cysteine CB).
    Both files are CC BY 4.0 (see LICENSE-WEIGHTS).

Use ``importlib.resources`` to resolve the path::

    from importlib.resources import files
    ckpt = files("caustic.data") / "best_v2_carbons.onnx"
"""
