"""Output format writers for :class:`ShiftPrediction`.

Public API::

    from caustic.io import (
        write_nef,
        write_nmrstar,
        write_csv,
        write_json,
    )

All writers take a :class:`ShiftPrediction` and an output path. They share
a common masking convention: positions where the prediction is ``NaN``
(residue has no valid target atom — e.g. PRO amide H, residues with
incomplete backbone, GLY CB) are simply skipped from the output. The
NEF/NMR-STAR layers therefore describe only the residues/atoms the model
actually predicted, which is what downstream NMR software expects.
"""
from __future__ import annotations

from caustic.io.nef import write_nef
from caustic.io.simple import write_csv, write_json

# NMR-STAR writer imports are lazy because it optionally pulls pynmrstar;
# the re-export happens via a function wrapper to keep that optional.


def write_nmrstar(*args, **kwargs):
    """Lazy wrapper — see :func:`caustic.io.nmrstar.write_nmrstar`."""
    from caustic.io.nmrstar import write_nmrstar as _impl

    return _impl(*args, **kwargs)


__all__ = ["write_nef", "write_nmrstar", "write_csv", "write_json"]
