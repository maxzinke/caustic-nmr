"""Run the public ``caustic`` package on every test structure.

This is the public path: the installed package's ``predict_shifts_onnx`` with
the bundled ONNX weights, default settings (all NMR conformers up to
``max_conformers``, median-aggregated, slim SA16 calibrator applied), on the
mmCIF file the production pipeline aligned for each BMRB entry
(``data/test_inputs.csv``; chain fixed to the aligned chain).

Output ``results/predictions_caustic.csv`` (one row per residue x nucleus, PDB
numbering; ``results/caustic_run_log.json`` records version, hashes, per-entry
status and wall time). Resumable and shardable::

    python run_caustic.py --shard 0/4 &   # ... one process per shard
    python run_caustic.py --merge         # -> results/predictions_caustic.csv

Set ``--max-conformers`` to reproduce the record (6) instead of the package
default (20); the output file name then carries the suffix ``_k6``.
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
TMP = HERE / "tmp" / "caustic"
PDB_DIR = Path(os.environ.get("CAUSTIC_BENCH_PDB_DIR", Path.home() / ".crystalline_fid" / "pdb_cache"))
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
COLS = ["bmrb_id", "pdb_id", "chain_id", "pdb_seq_id", "resname", "nucleus", "pred", "sigma"]


def read_inputs() -> list[dict]:
    with open(DATA / "test_inputs.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def suffix(k: int | None) -> str:
    return "" if k is None else f"_k{k}"


def run_shard(idx: int, n: int, max_conformers: int | None) -> None:
    import caustic
    from caustic import predict_shifts_onnx
    from caustic.export import load_onnx_session

    onnx_path = ir.files("caustic.data") / "best_v2_carbons.onnx"
    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    session = load_onnx_session(str(onnx_path))
    kwargs = {} if max_conformers is None else {"max_conformers": max_conformers}

    TMP.mkdir(parents=True, exist_ok=True)
    out_csv = TMP / f"shard{suffix(max_conformers)}_{idx}of{n}.csv"
    out_log = TMP / f"shard{suffix(max_conformers)}_{idx}of{n}.log.json"
    log = json.load(open(out_log)) if out_log.exists() else {"entries": {}}
    done = set(log["entries"])
    log.setdefault("meta", {}).update({
        "caustic_version": caustic.__version__,
        "caustic_file": str(Path(caustic.__file__).resolve()),
        "onnx_sha256": onnx_sha,
        "max_conformers": max_conformers if max_conformers is not None else "package default",
        "python": sys.version.split()[0],
        "snapshot_commit": os.environ.get("CAUSTIC_SNAPSHOT_COMMIT", ""),
    })

    rows = read_inputs()[idx::n]
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        for i, r in enumerate(rows):
            eid = r["bmrb_id"]
            if eid in done:
                continue
            cif = PDB_DIR / f"{r['pdb_id']}.cif"
            t0 = time.time()
            try:
                res = predict_shifts_onnx(str(cif), str(onnx_path), chain_id=r["chain_id"] or None,
                                          session=session, **kwargs)
                n_rows = 0
                for k, sid in enumerate(res.seq_ids):
                    rn = res.residue_names[k]
                    for nuc in NUCLEI:
                        p = float(res.mean[nuc][k])
                        s = float(res.std[nuc][k])
                        if p != p:  # NaN
                            continue
                        w.writerow([eid, r["pdb_id"], r["chain_id"], int(sid), rn, nuc, f"{p:.4f}", f"{s:.4f}"])
                        n_rows += 1
                log["entries"][eid] = {"status": "ok", "n_conformers": int(res.num_conformers),
                                       "n_rows": n_rows, "seconds": round(time.time() - t0, 2)}
            except Exception as exc:  # recorded, never skipped silently
                log["entries"][eid] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300],
                                       "seconds": round(time.time() - t0, 2)}
            f.flush()
            json.dump(log, open(out_log, "w"), indent=1)
            print(f"[shard {idx}/{n}] {i + 1}/{len(rows)} {eid} {log['entries'][eid]['status']} "
                  f"{log['entries'][eid]['seconds']}s", flush=True)


def merge(max_conformers: int | None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    shards = sorted(TMP.glob(f"shard{suffix(max_conformers)}_*of*.csv"))
    if max_conformers is None:
        shards = [s for s in shards if "_k" not in s.name]
    logs = [json.load(open(s.with_suffix("").with_suffix(".log.json"))) for s in shards]
    merged = {"meta": logs[0]["meta"] if logs else {}, "entries": {}}
    for lg in logs:
        merged["entries"].update(lg["entries"])
    seconds = sum(e.get("seconds", 0) for e in merged["entries"].values())
    n_ok = sum(1 for e in merged["entries"].values() if e["status"] == "ok")
    merged["summary"] = {"n_entries": len(merged["entries"]), "n_ok": n_ok,
                         "n_error": len(merged["entries"]) - n_ok,
                         "total_prediction_seconds": round(seconds, 1),
                         "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = RESULTS / f"predictions_caustic{suffix(max_conformers)}.csv"
    with open(out, "w", newline="") as fo:
        fo.write(f"# caustic-nmr {merged['meta'].get('caustic_version')} public path; "
                 f"onnx sha256 {merged['meta'].get('onnx_sha256')}; "
                 f"snapshot {merged['meta'].get('snapshot_commit')}; "
                 f"max_conformers={merged['meta'].get('max_conformers')}; "
                 f"generated {merged['summary']['merged_at']}\n")
        w = csv.writer(fo)
        w.writerow(COLS)
        for s in shards:
            with open(s, newline="") as fi:
                rd = csv.reader(fi)
                next(rd)
                w.writerows(rd)
    json.dump(merged, open(RESULTS / f"caustic{suffix(max_conformers)}_run_log.json", "w"), indent=1)
    print(json.dumps(merged["summary"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-conformers", type=int, default=None)
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    if a.merge:
        merge(a.max_conformers)
    else:
        i, n = (int(x) for x in a.shard.split("/"))
        run_shard(i, n, a.max_conformers)


if __name__ == "__main__":
    main()
