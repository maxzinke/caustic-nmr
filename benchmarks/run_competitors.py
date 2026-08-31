"""Run SPARTA+, LEGOLAS and UCBShift2 on the prepared single-model PDB files.

Inputs: ``tmp/pdb_prepared/<bmrb_id>.pdb`` from ``prepare_pdbs.py`` (model 1,
aligned chain, missing atoms completed, hydrogens added) and
``data/test_conditions.csv`` (BMRB sample pH, passed to UCBShift2 ``--pH``
where known).

Outputs per tool: ``results/predictions_<tool>.csv`` (same columns as the
CAUSTIC file) and ``results/<tool>_run_log.json`` (program version, exact
command line, per-entry status incl. every crash / timeout, wall time). For
UCBShift2 the transfer-module diagnostics are kept too
(``results/predictions_ucbshift_x.csv`` = ML-only "UCBShift-X" column of the
same run; per-entry best reference score/coverage in the log) so the effect of
its reference database on the comparison can be quantified.

Every failure is recorded as an entry with ``status: error`` — never dropped.

    python run_competitors.py --tool sparta
    python run_competitors.py --tool legolas
    python run_competitors.py --tool ucbshift --workers 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
PDBS = HERE / "tmp" / "pdb_prepared"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
COLS = ["bmrb_id", "pdb_id", "chain_id", "pdb_seq_id", "resname", "nucleus", "pred", "sigma"]

SPARTAP_DIR = Path(os.environ.get("SPARTAP_DIR", r"C:\tmp\SPARTA+"))
CSPRED_WIN = Path(os.environ.get("CSPRED_DIR", r"C:\Users\maxim\Documents\coding\noft\deps\CSpred"))
NOFT_DIR = Path(os.environ.get("NOFT_DIR", r"C:\Users\maxim\Documents\coding\noft"))
UCB_PY = os.environ.get("UCBSHIFT_PYTHON", "/root/ucbshift_env/bin/python3")


def wsl_path(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}" if len(s) > 1 and s[1] == ":" else s


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs() -> list[dict]:
    rows = [r for r in csv.DictReader(open(DATA / "test_inputs.csv", newline="")) if r["status"] == "ok"]
    cond = {r["bmrb_id"]: r for r in csv.DictReader(open(DATA / "test_conditions.csv", newline=""))}
    for r in rows:
        r["ph"] = cond.get(r["bmrb_id"], {}).get("ph", "")
    return rows


def pdb_resnames(pdb: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for ln in open(pdb):
        if ln.startswith("ATOM"):
            out.setdefault(int(ln[22:26]), ln[17:20].strip())
    return out


# ---------------------------------------------------------------- SPARTA+ --
_SPARTA_ATOM = {"HN": "H", "HA": "HA", "N": "N", "CA": "CA", "CB": "CB", "C": "C", "HA2": "HA2", "HA3": "HA3"}


def sparta_version() -> str:
    bin_ = SPARTAP_DIR / "bin" / "SPARTA+.static.winxp"
    proc = subprocess.run([str(bin_)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Version\s+([\d.]+)\s*\(Build\s+([\d.]+)\)", (SPARTAP_DIR / "README").read_text(errors="ignore"))
    return f"SPARTA+ {m.group(1)} build {m.group(2)}" if m else proc.stdout[:120]


def run_sparta_one(eid: str, pdb: Path, resnames: dict[int, str]) -> tuple[list, dict]:
    bin_ = SPARTAP_DIR / "bin" / "SPARTA+.static.winxp"
    with tempfile.TemporaryDirectory(prefix=f"sparta_{eid}_") as td:
        cmd = [str(bin_), "-in", str(pdb), "-spartaDir", str(SPARTAP_DIR)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=td)
        pred = Path(td) / "pred.tab"
        if proc.returncode != 0 or not pred.exists():
            raise RuntimeError(f"rc={proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")
        rows, gly = [], {}
        for ln in open(pred):
            parts = ln.split()
            if len(parts) < 9 or not parts[0].lstrip("-").isdigit():
                continue
            resid, atom, shift, sigma = int(parts[0]), parts[2], float(parts[4]), float(parts[8])
            nuc = _SPARTA_ATOM.get(atom)
            if nuc is None:
                continue
            if nuc in ("HA2", "HA3"):
                gly.setdefault(resid, []).append((shift, sigma))
                continue
            rows.append((resid, nuc, shift, sigma))
        for resid, vals in gly.items():  # GLY HA = mean of HA2/HA3 (truth carries one HA value)
            rows.append((resid, "HA", sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals)))
        return rows, {"cmd": " ".join(cmd)}


# ---------------------------------------------------------------- LEGOLAS --
def legolas_version() -> dict:
    mdir = NOFT_DIR / "deps" / "legolas" / "test" / "ens_models"
    files = sorted(mdir.glob("ens_model_*.pt"))
    return {"source": "https://github.com/roitberg-group/legolas (test/ens_models, 30 files)",
            "n_model_files": len(files),
            "model_files_sha256_of_concatenated_sha256s": hashlib.sha256(
                "".join(sha256(f) for f in files).encode()).hexdigest(),
            "model_file_mtime": datetime.fromtimestamp(files[0].stat().st_mtime).isoformat() if files else None,
            "wrapper": "crystalline_fid.crystalline.data.legolas_runner (AEV + 30-model ensemble, "
                       "constants copied from the LEGOLAS repository)"}


def run_legolas_one(eid: str, pdb: Path, resnames: dict[int, str]) -> tuple[list, dict]:
    sys.path.insert(0, str(NOFT_DIR))
    from crystalline_fid.crystalline.data.legolas_runner import run_legolas
    preds = run_legolas(str(pdb))
    rows = []
    for p in preds:
        for nuc, v in p.shifts.items():
            if nuc in NUCLEI and v == v:
                rows.append((int(p.resid), nuc, float(v), float(p.sigma.get(nuc, float("nan")))))
    return rows, {}


# --------------------------------------------------------------- UCBShift --
def ucbshift_version() -> dict:
    head = subprocess.run(["git", "-C", str(CSPRED_WIN), "log", "-1", "--format=%H %ad"],
                          capture_output=True, text=True).stdout.strip()
    n_models = len(list((CSPRED_WIN / "models").glob("*.sav")))
    n_ref = len(list((CSPRED_WIN / "refDB" / "pdbs").glob("*.pdb")))
    return {"source": "https://github.com/JerryJohnsonLee/CSpred", "git_head": head,
            "n_model_files_sav": n_models, "n_refDB_pdbs": n_ref,
            "mode": "full (UCBShift-X ML + UCBShift-Y transfer with BLAST + mTM-align; NOT --shiftx_only)",
            "external": "blast 2.9.0+, mTM-align, mkdssp from CSpred/bins; run under WSL Ubuntu"}


def run_ucbshift_one(eid: str, pdb: Path, resnames: dict[int, str], worker: int = 0, ph: str = "") -> tuple[list, dict]:
    out_dir = HERE / "tmp" / "ucbshift"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{eid}.csv"
    # CSpred stalls in uninterruptible drvfs I/O when run from /mnt/c (22 GB of model
    # pickles); UCBSHIFT_CSPRED_WSL points at a WSL-native copy of the same checkout
    # (version() still reads the Windows checkout - identical content, verified by git head).
    cspred = os.environ.get("UCBSHIFT_CSPRED_WSL") or wsl_path(CSPRED_WIN)
    ph_arg = f" --pH {float(ph):g}" if ph not in ("", None) else ""
    inner = (f"mkdir -p /tmp/ucbw{worker} && cd /tmp/ucbw{worker} && "
             f'export PATH="{cspred}/bins:{cspred}/bins/ncbi-blast-2.9.0+/bin:$PATH" && '
             f"{UCB_PY} {cspred}/CSpred.py {wsl_path(pdb)}{ph_arg} -o {wsl_path(out_csv)}")
    cmd = ["wsl.exe", "bash", "-c", inner]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    meta = {"cmd": inner, "seconds_tool": round(time.time() - t0, 1)}
    if proc.returncode != 0 or not out_csv.exists():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-6:])
        raise RuntimeError(f"rc={proc.returncode}: {tail[-400:]}")
    rows, rows_x, best = [], [], {}
    with open(out_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            resid = int(float(r["RESNUM"]))
            for nuc in NUCLEI:
                v = r.get(f"{nuc}_UCBShift", "")
                x = r.get(f"{nuc}_X", "")
                if v == "" and nuc == "HA" and r.get("RESNAME") == "GLY":
                    hs = [float(r[k]) for k in ("HA2_UCBShift", "HA3_UCBShift") if r.get(k, "") != ""]
                    v = sum(hs) / len(hs) if hs else ""
                if v != "":
                    rows.append((resid, nuc, float(v), float("nan")))
                if x != "":
                    rows_x.append((resid, nuc, float(x), float("nan")))
            for nuc in ("CA", "H"):
                s = r.get(f"{nuc}_BEST_REF_SCORE", "")
                if s not in ("", None):
                    best[nuc] = max(best.get(nuc, 0.0), float(s))
    meta["best_ref_score"] = best
    meta["n_rows_x"] = len(rows_x)
    meta["_rows_x"] = rows_x
    return rows, meta


TOOLS = {
    "sparta": (run_sparta_one, sparta_version),
    "legolas": (run_legolas_one, legolas_version),
    "ucbshift": (run_ucbshift_one, ucbshift_version),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=TOOLS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-unprepared", action="store_true",
                    help="silently skip entries whose prepared PDB does not exist yet (prepare_pdbs.py still running); "
                         "re-run without the flag afterwards so a missing file is recorded as an error")
    a = ap.parse_args()
    run_one, version = TOOLS[a.tool]

    tmp = HERE / "tmp" / a.tool
    tmp.mkdir(parents=True, exist_ok=True)
    log_path = tmp / "run_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {"entries": {}}
    log["meta"] = {"tool": a.tool, "version": version(), "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    inputs = read_inputs()[: a.limit]
    todo = [r for r in inputs if r["bmrb_id"] not in log["entries"] or not (tmp / f"{r['bmrb_id']}.rows.json").exists()]
    if a.skip_unprepared:
        todo = [r for r in todo if (PDBS / f"{r['bmrb_id']}.pdb").exists()]
    print(f"{a.tool}: {len(todo)} to run ({len(inputs) - len(todo)} done)", flush=True)
    import threading
    lock = threading.Lock()

    def task(r: dict, worker: int) -> None:
        eid = r["bmrb_id"]
        pdb = PDBS / f"{eid}.pdb"
        t0 = time.time()
        try:
            if not pdb.exists():
                raise FileNotFoundError(f"prepared PDB missing: {pdb}")
            resnames = pdb_resnames(pdb)
            if a.tool == "ucbshift":
                rows, meta = run_one(eid, pdb, resnames, worker=worker, ph=r["ph"])
            else:
                rows, meta = run_one(eid, pdb, resnames)
            rows_x = meta.pop("_rows_x", None)
            json.dump({"rows": rows, "rows_x": rows_x, "resnames": resnames}, open(tmp / f"{eid}.rows.json", "w"))
            entry = {"status": "ok", "n_rows": len(rows), "seconds": round(time.time() - t0, 1), **meta}
        except subprocess.TimeoutExpired:
            entry = {"status": "error", "error": "timeout", "seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            entry = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500], "seconds": round(time.time() - t0, 1)}
        with lock:
            log["entries"][eid] = entry
            json.dump(log, open(log_path, "w"), indent=1)
        print(f"[{a.tool}] {eid} {entry['status']} {entry['seconds']}s", flush=True)

    if a.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(task, r, i % a.workers) for i, r in enumerate(todo)]
            for f in futs:
                f.result()
    else:
        for r in todo:
            task(r, 0)

    # ---- merge ---------------------------------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    by_id = {r["bmrb_id"]: r for r in inputs}
    outs = {"": RESULTS / f"predictions_{a.tool}.csv"}
    if a.tool == "ucbshift":
        outs["x"] = RESULTS / "predictions_ucbshift_x.csv"
    handles = {}
    for key, path in outs.items():
        fo = open(path, "w", newline="")
        fo.write(f"# {a.tool}{' (UCBShift-X ML-only column of the same run)' if key else ''}; "
                 f"{json.dumps(log['meta']['version'])}; generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        w = csv.writer(fo)
        w.writerow(COLS)
        handles[key] = (fo, w)
    for eid in sorted(log["entries"], key=int):
        p = tmp / f"{eid}.rows.json"
        if log["entries"][eid]["status"] != "ok" or not p.exists():
            continue
        d = json.load(open(p))
        r = by_id[eid]
        rn = {int(k): v for k, v in d["resnames"].items()}
        for key, rows in (("", d["rows"]), ("x", d.get("rows_x") or [])):
            if key not in handles:
                continue
            for resid, nuc, pred, sigma in rows:
                handles[key][1].writerow([eid, r["pdb_id"], r["chain_id"], resid, rn.get(resid, ""), nuc,
                                          f"{pred:.4f}", "" if sigma != sigma else f"{sigma:.4f}"])
    for fo, _ in handles.values():
        fo.close()
    n_ok = sum(1 for e in log["entries"].values() if e["status"] == "ok")
    log["summary"] = {"n_entries": len(log["entries"]), "n_ok": n_ok, "n_error": len(log["entries"]) - n_ok,
                      "errors": {k: v["error"] for k, v in log["entries"].items() if v["status"] != "ok"},
                      "total_seconds": round(sum(e.get("seconds", 0) for e in log["entries"].values()), 1)}
    json.dump(log, open(RESULTS / f"{a.tool}_run_log.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in log["summary"].items() if k != "errors"}))


if __name__ == "__main__":
    main()
