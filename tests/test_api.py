"""Python API regression: the bundled model + calibrator on 1UBQ.

Reference values were measured with caustic-nmr 0.3.0/0.4.0 on the bundled
`best_v2_carbons.onnx` (SHA-256 ebc7bbc2…) with the default calibrator. If
they move, the model or the pipeline changed — see CONTRIBUTING.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from caustic import ShiftPrediction, predict_shifts_onnx
from caustic.provenance import bundled_model_path

CA_FIRST5 = [54.53, 55.31, 59.73, 56.07, 60.82]
EXPECTED_VALID = {"H": 73, "HA": 76, "N": 76, "CA": 76, "CB": 70, "C": 76}


@pytest.fixture(scope="module")
def ubq_prediction(ubq_pdb) -> ShiftPrediction:
    return predict_shifts_onnx(str(ubq_pdb), str(bundled_model_path()))


def test_shapes_and_coverage(ubq_prediction):
    pred = ubq_prediction
    assert len(pred.seq_ids) == 76
    assert len(pred.residue_names) == 76
    assert pred.residue_names[:3] == ["MET", "GLN", "ILE"]
    assert pred.num_conformers == 1
    for nuc, n_valid in EXPECTED_VALID.items():
        assert pred.mean[nuc].shape == (76,)
        assert pred.std[nuc].shape == (76,)
        assert int(np.sum(np.isfinite(pred.mean[nuc]))) == n_valid, nuc
    # PRO amide H and GLY CB have no target atom → NaN
    pro = [i for i, r in enumerate(pred.residue_names) if r == "PRO"]
    gly = [i for i, r in enumerate(pred.residue_names) if r == "GLY"]
    assert pro and all(np.isnan(pred.mean["H"][i]) for i in pro)
    assert gly and all(np.isnan(pred.mean["CB"][i]) for i in gly)


def test_reference_values(ubq_prediction):
    ca = ubq_prediction.mean["CA"][:5]
    np.testing.assert_allclose(ca, CA_FIRST5, atol=0.05)
    sig = ubq_prediction.std["CA"][np.isfinite(ubq_prediction.std["CA"])]
    assert 0.2 < float(np.median(sig)) < 2.0


def test_provenance_is_attached(ubq_prediction):
    prov = ubq_prediction.provenance
    assert prov["program"] == "caustic-nmr"
    assert prov["model_file"] == "best_v2_carbons.onnx"
    assert prov["model_sha256"].startswith("ebc7bbc2fc59")
    assert prov["calibrator"] == "sa16_v2_carbons_slim"
    assert prov["date"].endswith("Z")


def test_calibrator_opt_out_changes_only_offsets(ubq_pdb, ubq_prediction):
    raw = predict_shifts_onnx(str(ubq_pdb), str(bundled_model_path()), apply_calibrator=False)
    assert raw.provenance["calibrator"] == "none"
    d = ubq_prediction.mean["CA"] - raw.mean["CA"]
    d = d[np.isfinite(d)]
    # 1UBQ has no cysteine, so the calibrator is a pure per-nucleus constant
    assert np.allclose(d, d[0], atol=1e-5)
    assert abs(float(d[0]) - 0.024677) < 1e-4
