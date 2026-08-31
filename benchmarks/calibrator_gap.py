"""D7 — how much does the shipped *slim* calibrator give up against the full
calibrator of record?

The package ships ``sa16_calibrator_v2.json`` = global per-nucleus offsets +
CYS-CB modifiers. The calibrator the internal record was measured with
(``calibrator_ensemble`` in the training repository's report
``20260522_sa16_v2_carbons_combo.json``)
has the same two components plus per-(nucleus, DSSP class, aromatic-ring bin,
atom-rSASA bin) stratum offsets. Because the shared components are identical,
full − slim = the stratum offset alone, so the full calibrator can be applied
on top of the public-path predictions without re-running the model.

The three stratum features come from the **production medoid graphs**
(``$CAUSTIC_DATA_HOME/shift_predictor_graphs/<id>.pt``): the DSSP 3-state label is
pre-stored on those graphs at build time and the public package cannot compute it
(``structure_to_graph`` zero-fills feature [30]; a first version of this script used the
public graphs and its ``dssp_nonzero_frac`` counter caught every residue reading "coil").
Ring count = ``target_chemistry[:, :, 12] * 8`` and atom rSASA = ``target_atom_rsasa``
are computed with the package's feature functions on the same production graph; bins as
in the calibrator fit (rSASA < 0.05 buried, < 0.25 partly_buried, < 0.55
exposed, else highly_exposed; ring count 0 none, < 0.5 trace, < 1.5 one, else
multi; strata with no fitted offset get 0).

Writes ``results/predictions_caustic_fullcal.csv`` (same columns as the
CAUSTIC file) and ``results/calibrator_gap.json``; ``rescore.py`` then reports
``caustic_fullcal`` like any other method (paired against the slim CAUSTIC
column, with CIs).

Requires the full calibrator JSON (private report): pass ``--full-calibrator``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
GRAPH_DIR = Path(os.environ.get("CAUSTIC_DATA_HOME", Path.home() / ".caustic-data")) / "shift_predictor_graphs"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
DEFAULT_FULL = Path(os.environ.get("TRAINING_REPO_DIR", Path.home() / "training-repo")) / "reports" / "caustic" / "active" / "20260522_sa16_v2_carbons_combo.json"


def bin_rsasa(v: float) -> str:
    if not math.isfinite(v) or v < 0:
        return "missing"
    return "buried" if v < 0.05 else "partly_buried" if v < 0.25 else "exposed" if v < 0.55 else "highly_exposed"


def bin_ring(count: float) -> str:
    if not math.isfinite(count) or count <= 0:
        return "none"
    return "trace" if count < 0.5 else "one" if count < 1.5 else "multi"


def dssp_name(v: float) -> str:
    if not math.isfinite(v):
        return "unknown"
    return "helix" if abs(v - 1.0) < 0.5 else "strand" if abs(v + 1.0) < 0.5 else "coil" if abs(v) < 0.5 else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-calibrator", default=str(DEFAULT_FULL))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.disable(logging.WARNING)

    combo = json.load(open(a.full_calibrator))
    cal = combo["calibrator_ensemble"]
    strata = {ast.literal_eval(k) if k.startswith("(") else k: v["offset"] for k, v in cal["stratum_offsets"].items()}
    import caustic
    from caustic.calibrate import load_calibrator
    slim = load_calibrator()
    same_global = all(abs(float(slim["global_offsets"][n]) - float(cal["global_offsets"][n])) < 1e-9 for n in NUCLEI)
    same_cys = all(abs(float(slim["cys_modifiers"][k]["offset"]) - float(cal["cys_modifiers"][k]["offset"])) < 1e-9
                   for k in slim["cys_modifiers"])
    print(f"shared components identical: global_offsets={same_global} cys_modifiers={same_cys}")
    assert same_global and same_cys, "slim and full calibrators differ in shared components; gap = strata only is invalid"

    import torch
    from caustic.graph import compute_target_chemistry, compute_target_atom_rsasa

    # slim predictions (public path)
    preds: dict[str, dict[tuple[int, str], tuple[float, float]]] = {}
    import gzip as _gzip
    cpath = RESULTS / "predictions_caustic.csv"
    if not cpath.exists():
        cpath = RESULTS / "predictions_caustic.csv.gz"
    with (_gzip.open(cpath, "rt", newline="") if cpath.suffix == ".gz" else open(cpath, newline="")) as f:
        f.readline()
        for r in csv.DictReader(f):
            preds.setdefault(r["bmrb_id"], {})[(int(r["pdb_seq_id"]), r["nucleus"])] = (float(r["pred"]), r["sigma"])
    inputs = {r["bmrb_id"]: r for r in csv.DictReader(open(DATA / "test_inputs.csv", newline=""))}

    out_rows, log = [], {"entries": {}, "strata_counts": Counter(), "n_offset_applied": 0, "n_no_offset": 0}
    t0 = time.time()
    eids = list(preds)[: a.limit]
    for i, eid in enumerate(eids):
        r = inputs[eid]
        try:
            g = torch.load(str(GRAPH_DIR / f"{eid}.pt"), weights_only=False)
            dssp_arr = getattr(g, "dssp_3state", None)
            if dssp_arr is None:
                raise RuntimeError("production graph lacks dssp_3state")
            dssp_arr = dssp_arr.detach().cpu().numpy()
            chem = compute_target_chemistry(g).numpy()
            rsasa = compute_target_atom_rsasa(g).numpy()
            seq_ids = [int(s) for s in g.seq_ids.tolist()]
            n_applied = 0
            for k, sid in enumerate(seq_ids):
                dssp = dssp_name(float(dssp_arr[k]) if k < len(dssp_arr) else float("nan"))
                for ni, nuc in enumerate(NUCLEI):
                    p = preds[eid].get((sid, nuc))
                    if p is None:
                        continue
                    key = (nuc, dssp, bin_ring(float(chem[k, ni, 12]) * 8.0), bin_rsasa(float(rsasa[k, ni])))
                    off = strata.get(key)
                    log["strata_counts"][f"{key[1]}|{key[2]}|{key[3]}"] += 1
                    if off is None:
                        log["n_no_offset"] += 1
                        off = 0.0
                    else:
                        n_applied += 1
                    out_rows.append([eid, r["pdb_id"], r["chain_id"], sid, "", nuc, f"{p[0] + off:.4f}", p[1]])
            log["n_offset_applied"] += n_applied
            log["entries"][eid] = {"status": "ok", "n_applied": n_applied,
                                   "dssp_nonzero_frac": float(np.mean(np.abs(dssp_arr) > 0.5))}
        except Exception as exc:
            log["entries"][eid] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(eids)} {time.time() - t0:.0f}s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "predictions_caustic_fullcal.csv", "w", newline="") as f:
        f.write(f"# caustic-nmr {caustic.__version__} public predictions + record stratum offsets "
                f"(strata features from production medoid graphs) "
                f"({Path(a.full_calibrator).name}); generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        w = csv.writer(f)
        w.writerow(["bmrb_id", "pdb_id", "chain_id", "pdb_seq_id", "resname", "nucleus", "pred", "sigma"])
        w.writerows(out_rows)
    log["strata_counts"] = dict(log["strata_counts"].most_common())
    n_err = sum(1 for e in log["entries"].values() if e["status"] != "ok")
    log["summary"] = {"n_entries": len(log["entries"]), "n_error": n_err, "seconds": round(time.time() - t0, 1),
                      "mean_dssp_nonzero_frac": float(np.mean([e["dssp_nonzero_frac"] for e in log["entries"].values()
                                                               if e["status"] == "ok"]))}
    json.dump(log, open(RESULTS / "calibrator_gap.json", "w"), indent=1)
    print(json.dumps(log["summary"]))


if __name__ == "__main__":
    main()
