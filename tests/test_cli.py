"""End-to-end: the installed `caustic` CLI on the bundled examples."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

PYTHON = [sys.executable, "-m", "caustic.cli"]


def _run(args, cwd, env_extra=None):
    env = dict(os.environ)
    env.pop("CAUSTIC_WEIGHTS", None)  # the bundled model must be found on its own
    if env_extra:
        env.update(env_extra)
    return subprocess.run(PYTHON + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def test_version_reports_model_and_calibrator(tmp_path):
    r = _run(["--version"], tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert out[0].startswith("caustic-nmr 0.")
    assert out[1].startswith("model: best_v2_carbons.onnx sha256=ebc7bbc2fc59")
    assert out[2] == "calibrator: sa16_v2_carbons_slim"


@pytest.fixture(scope="module")
def outputs(ubq_pdb, tmp_path_factory):
    """Run all four formats once; return {fmt: path}."""
    d = tmp_path_factory.mktemp("cli")
    paths = {}
    for fmt, suffix in (("csv", ".csv"), ("nef", ".nef"), ("star", ".str"), ("json", ".json")):
        out = d / f"ubq{suffix}"
        r = _run([str(ubq_pdb), "--format", fmt, "-o", str(out)], d)
        assert r.returncode == 0, r.stderr
        assert "76 residues" in r.stdout
        assert out.exists()
        paths[fmt] = out
    return paths


def test_csv_has_provenance_header_and_rows(outputs):
    lines = outputs["csv"].read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# caustic-nmr 0.")
    assert "model=best_v2_carbons.onnx" in lines[0]
    assert "sha256=ebc7bbc2fc59" in lines[0]
    assert "calibrator=sa16_v2_carbons_slim" in lines[0]
    assert lines[1].startswith("# model_sha256=ebc7bbc2fc59")
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))
    assert {r["atom_name"] for r in rows} == {"H", "HA", "HA%", "N", "CA", "CB", "C"}
    assert sum(1 for r in rows if r["atom_name"] == "CA") == 76
    assert sum(1 for r in rows if r["atom_name"] == "H") == 73


def test_nef_meta_and_rows(outputs):
    text = outputs["nef"].read_text(encoding="utf-8")
    assert text.startswith("data_caustic\n# caustic-nmr 0.")
    assert "_nef_nmr_meta_data.program_version   0." in text
    assert "_nef_nmr_meta_data.creation_date     20" in text
    assert "predicted_from_structure" not in text
    assert "unknown" not in text
    assert text.count("\n      A ") == 76 * 6 - 3 - 6  # rows: 6 nuclei × 76 − 3 PRO H − 6 GLY CB


def test_nmrstar_meta_and_rows(outputs):
    text = outputs["star"].read_text(encoding="utf-8")
    assert text.startswith("data_caustic_predicted\n# caustic-nmr 0.")
    assert "_Assigned_chem_shift_list.Program_version  0." in text
    assert "unknown" not in text
    pynmrstar = pytest.importorskip("pynmrstar")
    entry = pynmrstar.Entry.from_file(str(outputs["star"]))
    n = sum(len(lp.data) for sf in entry.get_saveframes_by_category("assigned_chemical_shifts") for lp in sf.loops)
    assert n == 76 * 6 - 3 - 6


def test_json_has_provenance_object(outputs):
    doc = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert doc["provenance"]["program"] == "caustic-nmr"
    assert doc["provenance"]["model_sha256"].startswith("ebc7bbc2fc59")
    assert doc["provenance"]["calibrator"] == "sa16_v2_carbons_slim"
    assert doc["num_residues"] == 76
    assert len(doc["shifts"]) == 76 * 6 - 3 - 6


def test_alphafold_model_runs(af_cif, tmp_path):
    out = tmp_path / "af.csv"
    r = _run([str(af_cif), "--format", "csv", "-o", str(out)], tmp_path)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    n_ca = sum(1 for ln in out.read_text(encoding="utf-8").splitlines() if ",CA," in ln)
    assert n_ca > 100


def test_missing_weights_message_names_all_locations(ubq_pdb, tmp_path):
    r = _run([str(ubq_pdb), "--onnx-model", str(tmp_path / "nope.onnx")], tmp_path)
    assert r.returncode != 0
    assert "bundled model" in r.stderr
    assert "CAUSTIC_WEIGHTS" in r.stderr
    assert "github.com/maxzinke/caustic-nmr" in r.stderr
    assert "<HF_HUB_URL>" not in r.stderr
