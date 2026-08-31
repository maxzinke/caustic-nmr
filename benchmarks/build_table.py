"""Build ``results/per_residue.csv`` — the single table of record.

One row per (bmrb_id, bmrb_seq_id, nucleus) that has a reference shift in
``data/truth_test.csv``. Predictions are joined through ``data/residue_map.csv``
(PDB residue number -> BMRB seq_id, the alignment the production pipeline
used); a residue without a prediction from a method gets an empty cell for
that method (never dropped, never filled).

Columns: bmrb_id, pdb_id, seq_id, resname, nucleus, truth, in_cleaned_set,
label_dropped, caustic, caustic_sigma, sparta, legolas, ucbshift, ucbshift_x, caustic_fullcal, caustic_recordgraph

``in_cleaned_set`` = the entry is not one of the 744 whole-entry blocklist
drops (the 617-entry "cleaned test" slice); ``label_dropped`` = this
particular label is on the per-label blocklist.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
METHODS = ["caustic", "sparta", "legolas", "ucbshift", "ucbshift_x", "caustic_fullcal", "caustic_recordgraph"]


def read_predictions(path: Path) -> dict[tuple[str, int, str], tuple[str, str]]:
    out = {}
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.exists():
            return out
        path = gz
    opener = (lambda p: gzip.open(p, "rt", newline="")) if path.suffix == ".gz" else (lambda p: open(p, newline=""))
    with opener(path) as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            out[(r["bmrb_id"], int(r["pdb_seq_id"]), r["nucleus"])] = (r["pred"], r.get("sigma", ""))
    return out


def main() -> None:
    residue_map: dict[tuple[str, int], int] = {}
    for r in csv.DictReader(open(DATA / "residue_map.csv", newline="")):
        residue_map[(r["bmrb_id"], int(r["bmrb_seq_id"]))] = int(r["pdb_seq_id"])
    pdb_of = {r["bmrb_id"]: r["pdb_id"] for r in csv.DictReader(open(DATA / "test_inputs.csv", newline=""))}
    cleaned = {ln.strip() for ln in open(DATA / "cleaned_test_ids.txt") if ln.strip()}
    dropped = {(r["bmrb_id"], int(r["seq_id"]), r["nucleus"])
               for r in csv.DictReader(open(DATA / "test_label_drops.csv", newline=""))}
    preds = {m: read_predictions(RESULTS / f"predictions_{m}.csv") for m in METHODS}
    header_lines = {}
    for m in METHODS:
        p = RESULTS / f"predictions_{m}.csv"
        if p.exists():
            with open(p) as f:
                header_lines[m] = f.readline().strip().lstrip("# ")

    n = defaultdict(int)
    rows = []
    for r in csv.DictReader(open(DATA / "truth_test.csv", newline="")):
        eid, sid, nuc = r["bmrb_id"], int(r["seq_id"]), r["nucleus"]
        pdb_sid = residue_map.get((eid, sid))
        row = [eid, pdb_of.get(eid, ""), sid, r["comp_id"], nuc, r["value"],
               "1" if eid in cleaned else "0", "1" if (eid, sid, nuc) in dropped else "0"]
        n["truth"] += 1
        if pdb_sid is None:
            n["unmapped_residue"] += 1
        for m in METHODS:
            v = preds[m].get((eid, pdb_sid, nuc)) if pdb_sid is not None else None
            if m == "caustic":
                row += [v[0] if v else "", v[1] if v else ""]
            else:
                row.append(v[0] if v else "")
            if v:
                n[m] += 1
        rows.append(row)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "per_residue.csv.gz"
    with gzip.open(out, "wt", newline="") as f:
        f.write("# per-residue table of record; generated "
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}; sources: {json.dumps(header_lines)}\n")
        w = csv.writer(f)
        w.writerow(["bmrb_id", "pdb_id", "seq_id", "resname", "nucleus", "truth", "in_cleaned_set", "label_dropped",
                    "caustic", "caustic_sigma", "sparta", "legolas", "ucbshift", "ucbshift_x", "caustic_fullcal", "caustic_recordgraph"])
        w.writerows(rows)
    print(json.dumps(dict(n), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
