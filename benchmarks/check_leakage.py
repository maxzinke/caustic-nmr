"""Leakage checks for the benchmark split.

1. No BMRB id of the test split appears in the train or validation split
   (exit 1 otherwise).
2. No test PDB id appears in the train/val structure set (``bmrb_to_pdb`` is
   not shipped, so this check runs only when the private cache is present and
   is reported, not asserted — the split is by BMRB entry sequence similarity,
   see docs/DATA.md).
3. Overlap of the test PDB ids with the competitor reference/training
   databases that are inspectable on disk:
   * UCBShift2 ``refDB/pdbs`` (transfer-prediction reference set; a hit means
     UCBShift-Y can copy experimental shifts of the same PDB entry),
   * SPARTA+ ``tab/homology.tab`` is a per-residue table without PDB ids —
     SPARTA+'s 580-protein training set is not enumerable from the
     distribution, so this overlap cannot be computed (reported as such),
   * LEGOLAS: training set not shipped with the weights (reported as such).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CSPRED = Path(os.environ.get("CSPRED_DIR", r"C:\Users\maxim\Documents\coding\noft\deps\CSpred"))
SPARTAP = Path(os.environ.get("SPARTAP_DIR", r"C:\tmp\SPARTA+"))
GRAPH_DIR = Path(os.environ.get("CRYSTALLINE_FID_HOME", Path.home() / ".crystalline_fid")) / "shift_predictor_graphs"


def ids(name: str) -> set[str]:
    return {ln.strip() for ln in open(DATA / "splits" / f"{name}_bmrb_ids.txt") if ln.strip()}


def main() -> int:
    train, val, test = ids("train"), ids("val"), ids("test")
    report: dict = {"split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                    "test_in_train": sorted(test & train), "test_in_val": sorted(test & val),
                    "train_in_val": sorted(train & val)}
    inputs = [r for r in csv.DictReader(open(DATA / "test_inputs.csv", newline="")) if r["status"] == "ok"]
    test_pdb = {r["pdb_id"].lower() for r in inputs}
    report["n_test_pdb_ids"] = len(test_pdb)

    # 2. structure-level overlap with train/val (needs the private mapping)
    m = GRAPH_DIR / "bmrb_to_pdb.json"
    if m.exists():
        b2p = json.load(open(m))
        trainval_pdb = {p.lower() for e in (train | val) for p in b2p.get(e, [])}
        shared = sorted(test_pdb & trainval_pdb)
        report["test_pdb_in_trainval_structures"] = {"n": len(shared), "ids": shared,
                                                     "note": "same PDB entry linked to a train/val BMRB entry "
                                                             "(different deposition of the same protein); the split "
                                                             "is by sequence similarity, not PDB id"}
    else:
        report["test_pdb_in_trainval_structures"] = "bmrb_to_pdb.json not available"

    # 3. competitor databases
    ref = CSPRED / "refDB" / "pdbs"
    if ref.exists():
        ref_ids = {p.name[:4].lower() for p in ref.glob("*.pdb")}
        hit = sorted(test_pdb & ref_ids)
        report["ucbshift_refDB"] = {"n_refDB_entries": len(ref_ids), "n_test_pdb_in_refDB": len(hit), "ids": hit}
    else:
        report["ucbshift_refDB"] = "refDB not available"
    report["spartap_training_set"] = "not enumerable from the distribution (no PDB id list shipped)"
    report["legolas_training_set"] = "not shipped with the weights"

    RESULTS.mkdir(exist_ok=True)
    json.dump(report, open(RESULTS / "leakage_report.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) or "ids" not in v else {kk: vv for kk, vv in v.items() if kk != "ids"})
                      for k, v in report.items()}, indent=1))
    leak = report["test_in_train"] or report["test_in_val"] or report["train_in_val"]
    print("SPLIT LEAK" if leak else "split OK: no test id in train/val")
    return 1 if leak else 0


if __name__ == "__main__":
    sys.exit(main())
