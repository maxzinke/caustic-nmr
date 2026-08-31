"""Reference point: the shipped ONNX model run on the *production* graphs.

The internal record was measured on graphs built by the training pipeline
(``$CAUSTIC_DATA_HOME/shift_predictor_graphs/<bmrb_id>.pt``: medoid conformer,
DSSP 3-state pre-stored, NMR-ensemble features, BMRB sample conditions). The
public path rebuilds the graph from the structure file with the package's
``structure_to_graph`` and does not have all of those. Running the same ONNX
weights + the same slim calibrator on the production graphs isolates the
effect of the *graph features* from everything else; the difference between
this column (``caustic_recordgraph``) and the public column (``caustic``) is
the public-path parity gap.

Medoid conformer only (the record additionally averaged up to 5 alternative
conformers). Requires the private graph cache and the training repository on ``sys.path``
(the cached objects are its ``ProteinData``).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.resources as ir
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
GRAPH_DIR = Path(os.environ.get("CAUSTIC_DATA_HOME", Path.home() / ".caustic-data")) / "shift_predictor_graphs"
NOFT = Path(os.environ.get("TRAINING_REPO_DIR", Path.home() / "training-repo"))
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]


def main() -> None:
    logging.disable(logging.WARNING)
    sys.path.insert(0, str(NOFT))
    import torch
    import caustic
    from caustic.export import load_onnx_session, run_onnx_inference
    from caustic.calibrate import apply_calibrator, detect_disulfides, find_cys_sg_indices, load_calibrator
    from caustic.inference import _residue_types_to_three_letter

    onnx_path = ir.files("caustic.data") / "best_v2_carbons.onnx"
    session = load_onnx_session(str(onnx_path))
    calibrator = load_calibrator()
    inputs = [r for r in csv.DictReader(open(DATA / "test_inputs.csv", newline="")) if r["status"] == "ok"]
    log = {"meta": {"caustic_version": caustic.__version__, "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                    "graph_dir": str(GRAPH_DIR), "conformers": "medoid graph only"}, "entries": {}}
    rows = []
    t0 = time.time()
    for i, r in enumerate(inputs):
        eid = r["bmrb_id"]
        try:
            g = torch.load(str(GRAPH_DIR / f"{eid}.pt"), weights_only=False)
            mean_np, _ = run_onnx_inference(session, g)
            residue_names = _residue_types_to_three_letter(g.residue_types)
            sg = find_cys_sg_indices(g)
            pos = g.pos.detach().cpu().numpy()
            dis = detect_disulfides(pos, residue_names, sg)
            cal = apply_calibrator(np.asarray(mean_np), residue_names, dis, calibrator)
            seq_ids = [int(s) for s in g.seq_ids.tolist()]
            n = 0
            for k, sid in enumerate(seq_ids):
                for ni, nuc in enumerate(NUCLEI):
                    v = float(cal[k, ni])
                    if v == v:
                        rows.append([eid, str(getattr(g, "pdb_id", r["pdb_id"])), r["chain_id"], sid, residue_names[k], nuc, f"{v:.4f}", ""])
                        n += 1
            log["entries"][eid] = {"status": "ok", "n_rows": n, "graph_pdb_id": str(getattr(g, "pdb_id", "")),
                                   "dssp_present": bool(getattr(g, "dssp_3state", None) is not None)}
        except Exception as exc:
            log["entries"][eid] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(inputs)} {time.time() - t0:.0f}s", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "predictions_caustic_recordgraph.csv", "w", newline="") as f:
        f.write(f"# caustic-nmr {caustic.__version__} ONNX on production graphs (medoid only) + slim calibrator; "
                f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        w = csv.writer(f)
        w.writerow(["bmrb_id", "pdb_id", "chain_id", "pdb_seq_id", "resname", "nucleus", "pred", "sigma"])
        w.writerows(rows)
    n_err = sum(1 for e in log["entries"].values() if e["status"] != "ok")
    log["summary"] = {"n_entries": len(log["entries"]), "n_error": n_err, "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(RESULTS / "caustic_recordgraph_run_log.json", "w"), indent=1)
    print(json.dumps(log["summary"]))


if __name__ == "__main__":
    main()
