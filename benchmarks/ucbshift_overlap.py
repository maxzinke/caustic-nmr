"""UCBShift2 fairness slice: refDB-overlap vs non-overlap test proteins.

UCBShift2's transfer (Y) module aligns the query against its shipped refDB; 67 of the
693 distinct test PDB ids are in that database (``check_leakage.py``), so for those
proteins the method has effectively seen the answer sheet's structure. This script
splits the paired CAUSTIC/UCBShift2 comparison by that flag and records how often the
Y module changed the prediction (ucbshift != ucbshift_x).

Writes ``results/ucbshift_overlap.json``; the numbers are quoted in
``docs/BENCHMARKS.md`` §8.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]


def main() -> None:
    df = pd.read_csv(gzip.open(RESULTS / "per_residue.csv.gz", "rt"), comment="#",
                     dtype={"bmrb_id": str})
    overlap_pdb = set(json.load(open(RESULTS / "leakage_report.json"))["ucbshift_refDB"]["ids"])
    df["in_ref"] = df.pdb_id.str.lower().isin(overlap_pdb)

    out: dict = {"n_refdb_overlap_pdb_ids": len(overlap_pdb), "slices": {}}
    d = df.dropna(subset=["truth", "ucbshift", "caustic"])
    for flag, g in d.groupby("in_ref"):
        lbl = "refdb_overlap" if flag else "non_overlap"
        out["slices"][lbl] = {
            "n_proteins": int(g.bmrb_id.nunique()),
            "n_residues": int(len(g)),
            "mae_ucbshift": {n: round(float((s.ucbshift - s.truth).abs().mean()), 4)
                             for n, s in g.groupby("nucleus")},
            "mae_caustic": {n: round(float((s.caustic - s.truth).abs().mean()), 4)
                            for n, s in g.groupby("nucleus")},
        }
    d2 = df.dropna(subset=["ucbshift", "ucbshift_x"])
    out["y_module_changed_prediction_frac"] = {
        ("refdb_overlap" if k else "non_overlap"): round(float(v), 4)
        for k, v in d2.assign(y_active=d2.ucbshift != d2.ucbshift_x)
                     .groupby("in_ref").y_active.mean().items()}

    json.dump(out, open(RESULTS / "ucbshift_overlap.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
