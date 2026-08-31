"""The bundled release artifacts are exactly the files of record."""
from __future__ import annotations

import hashlib
import json

import pytest

from caustic.provenance import (
    BUNDLED_CALIBRATOR_NAME,
    BUNDLED_MODEL_NAME,
    bundled_calibrator_path,
    bundled_model_path,
    calibrator_version,
    file_sha256,
)

ONNX_SHA256 = "ebc7bbc2fc59327a50384105207958948cd90d7b5c7ea1ec2906b473e02948b2"
ONNX_SIZE = 3_045_952
ONNX_PARAMS = 741_024
CALIBRATOR_SHA256 = "32d277df600a8e6b2f84e2f7ccaaaf6d9de1332664df61e51b36f31fdf267bf4"
CALIBRATOR_VERSION = "sa16_v2_carbons_slim"


def test_bundled_model_is_the_file_of_record():
    p = bundled_model_path()
    assert p.name == BUNDLED_MODEL_NAME
    assert p.exists()
    assert p.stat().st_size == ONNX_SIZE
    assert file_sha256(str(p)) == ONNX_SHA256


def test_bundled_calibrator_is_the_file_of_record():
    p = bundled_calibrator_path()
    assert p.name == BUNDLED_CALIBRATOR_NAME
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CALIBRATOR_SHA256
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["version"] == CALIBRATOR_VERSION
    assert calibrator_version() == CALIBRATOR_VERSION
    assert set(doc["global_offsets"]) == {"H", "HA", "N", "CA", "CB", "C"}
    assert set(doc["cys_modifiers"]) == {"disulfide", "metal_bound", "reduced_free"}


def test_parameter_count_matches_method_doc():
    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(bundled_model_path()))
    n = 0
    for init in model.graph.initializer:
        size = 1
        for d in init.dims:
            size *= d
        n += size
    assert n == ONNX_PARAMS
