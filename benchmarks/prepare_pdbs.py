"""Write the single-model PDB files the competitor programs are run on.

For every test entry in ``data/test_inputs.csv``:

1. read the mmCIF the production pipeline aligned (``pdb_id``), keep **model 1**
   and the aligned **chain only**, drop waters and other hetero groups;
2. complete missing heavy atoms and add hydrogens with PDBFixer
   (``addMissingAtoms`` + ``addMissingHydrogens(pH 7.0)``; residue numbering
   and chain id preserved, no residues built) — UCBShift2 and LEGOLAS require
   explicit hydrogens for their H/HA predictions and crash on incomplete
   side chains; SPARTA+ ignores hydrogens;
3. write ``tmp/pdb_prepared/<bmrb_id>.pdb`` with a PDB ``HEADER`` record —
   Biopython's DSSP wrapper (used inside UCBShift2) rejects files without one.

CAUSTIC is NOT run on these files; it reads the original mmCIF (all models).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "tmp" / "pdb_prepared"
PDB_DIR = Path(os.environ.get("CAUSTIC_BENCH_PDB_DIR", Path.home() / ".crystalline_fid" / "pdb_cache"))


def prepare_one(pdb_id: str, chain_id: str, out_path: Path) -> dict:
    import gemmi
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    st = gemmi.read_structure(str(PDB_DIR / f"{pdb_id}.cif"))
    st.setup_entities()
    n_models = len(st)
    while len(st) > 1:
        del st[1]
    model = st[0]
    for ch in list(model):
        if ch.name != chain_id:
            model.remove_chain(ch.name)
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    raw = out_path.with_suffix(".raw.pdb")
    st.write_pdb(str(raw))
    n_heavy_before = sum(1 for ch in st[0] for r in ch for a in r if a.element.name != "H")

    fixer = PDBFixer(filename=str(raw))
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # never build missing residues
    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    tmp = out_path.with_suffix(".fixed.pdb")
    with open(tmp, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    lines = [ln for ln in open(tmp) if ln.startswith(("ATOM", "HETATM", "TER", "END"))]
    header = f"HEADER    BENCHMARK INPUT MODEL 1 CHAIN {chain_id:<1}        01-JAN-00   {pdb_id.upper():<4}              \n"
    out_path.write_text(header + "".join(lines))
    tmp.unlink()
    raw.unlink()
    n_atoms = sum(1 for ln in lines if ln.startswith("ATOM"))
    return {"n_models_in_cif": n_models, "n_heavy_atoms_model1": n_heavy_before,
            "n_missing_heavy_atoms_added": n_missing_atoms, "n_atoms_written": n_atoms}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = HERE / "tmp" / "pdb_prepared_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {}
    rows = [r for r in csv.DictReader(open(DATA / "test_inputs.csv", newline="")) if r["status"] == "ok"]
    t0 = time.time()
    for i, r in enumerate(rows):
        eid = r["bmrb_id"]
        out = OUT / f"{eid}.pdb"
        if eid in log and out.exists():
            continue
        try:
            log[eid] = {"status": "ok", **prepare_one(r["pdb_id"], r["chain_id"], out)}
        except Exception as exc:
            log[eid] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        if (i + 1) % 25 == 0:
            json.dump(log, open(log_path, "w"), indent=1)
            print(f"{i + 1}/{len(rows)} prepared, {time.time() - t0:.0f}s", flush=True)
    json.dump(log, open(log_path, "w"), indent=1)
    n_err = sum(1 for v in log.values() if v["status"] != "ok")
    print(f"done: {len(log)} entries, {n_err} errors, {time.time() - t0:.0f}s")
    if n_err:
        print({k: v for k, v in log.items() if v["status"] != "ok"})


if __name__ == "__main__":
    sys.exit(main())
