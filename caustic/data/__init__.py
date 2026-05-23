"""Bundled assets shipped with the caustic package.

Currently contains:
    best_v2_carbons.onnx — v0.3.0 production checkpoint
        PaiNN backbone, ~740K params, trained on carbon-aggressive
        label-noise-cleaned BMRB labels. -4.37% relative composite MAE
        on cc.test (n=614) vs the previous PaiNN baseline.

Use ``importlib.resources`` to resolve the path::

    from importlib.resources import files
    ckpt = files("caustic.data") / "best_v2_carbons.onnx"
"""
