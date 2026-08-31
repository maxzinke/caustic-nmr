"""Every module imports; the public API and version are exposed."""
from __future__ import annotations

import importlib
import pkgutil

import pytest


def test_package_imports_and_exposes_api():
    import caustic

    assert caustic.__version__.count(".") == 2
    assert callable(caustic.predict_shifts_onnx)
    assert callable(caustic.predict_shifts)
    assert caustic.ShiftPrediction


@pytest.mark.parametrize(
    "modname",
    sorted(
        m.name
        for m in pkgutil.walk_packages(importlib.import_module("caustic").__path__, "caustic.")
    ),
)
def test_every_module_imports(modname):
    importlib.import_module(modname)


def test_cli_module_has_entry_point():
    from caustic.cli import main

    assert callable(main)
