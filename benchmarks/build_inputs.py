"""Build the benchmark input files under ``benchmarks/data/`` from the private
training caches.

This is the ONE step that depends on the private ``noft`` checkout and the
``~/.crystalline_fid`` caches; everything else under ``benchmarks/`` runs on
the public package plus the files this script writes. It is kept in the repo
so the provenance of every shipped data file is explicit.

What it writes (all under ``benchmarks/data/``):

* ``splits/{train,val,test}_bmrb_ids.txt`` — the production split
  (``splits.connected_components.json``; its SHA-256 must equal the checkpoint's
  ``splits_hash``).
* ``test_inputs.csv`` — one row per test entry: the PDB entry + chain the
  production pipeline aligned to the BMRB chemical-shift sequence (first PDB id
  in ``bmrb_to_pdb.json`` that aligns at >= 90 % identity, same rule as
  ``crystalline_fid/shift_predictor/dataset.py::_build_one_graph_worker``),
  number of models, experimental method, sequence.
* ``residue_map.csv`` — PDB residue number -> BMRB seq_id for every mapped
  residue (from ``crystalline_fid.structure.validate_sequence_mapping``).
* ``truth_test.csv`` — the reference shifts the model was evaluated against:
  the crystalline cache (``crystalline_all.npz``; H/HA/N/CA/CB/C only, one value
  per (entry, seq_id, nucleus), built from the BMRB entry by
  ``crystalline_fid/crystalline/data/dataset_builder.py``) with the production
  range filter (``DataConfig.min_shift``/``max_shift``) applied.
* ``cleaned_test_ids.txt`` and ``test_label_drops.csv`` — the v2 label
  blocklist restricted to the test split (whole-entry drops and per-label drops).
* ``label_blocklist_v2_sweep_carbons.json.gz`` — the full blocklist.
* ``provenance.json`` — hashes, counts and paths of every source file.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NOFT = Path(os.environ.get("NOFT_DIR", r"C:\Users\maxim\Documents\coding\noft"))
CACHE = Path(os.environ.get("CRYSTALLINE_FID_HOME", Path.home() / ".crystalline_fid"))
GRAPH_DIR = CACHE / "shift_predictor_graphs"
PDB_DIR = CACHE / "pdb_cache"
CRYST = CACHE / "crystalline_cache"
BLOCKLIST = NOFT / "results" / "caustic_label_blocklists" / "label_blocklist_v2_sweep_carbons.json"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
# DataConfig defaults, crystalline_fid/shift_predictor/config.py:224-229
MAX_SHIFT = {"H": 15.0, "HA": 15.0, "N": 200.0, "CA": 100.0, "CB": 100.0, "C": 200.0}
MIN_SHIFT = {"H": -2.0, "HA": -2.0, "N": 80.0, "CA": 30.0, "CB": 0.0, "C": 150.0}
MIN_IDENTITY = 0.9  # dataset.py::_build_one_graph_worker
MIN_OBSERVED = 5    # dataset.py::build_graphs (n_observed < 5 -> skipped)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sys.path.insert(0, str(NOFT))
    from crystalline_fid.structure import validate_sequence_mapping
    from crystalline_fid.shift_predictor.dataset import build_chem_shift_sequence

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(exist_ok=True)
    prov: dict = {"sources": {}}

    # --- split -----------------------------------------------------------
    split_path = GRAPH_DIR / "splits.connected_components.json"
    splits = json.load(open(split_path))
    prov["sources"]["splits"] = {"path": str(split_path), "sha256": sha256(split_path),
                                 "sizes": {k: len(v) for k, v in splits.items()}}
    for k in ("train", "val", "test"):
        ids = sorted(splits[k], key=int)
        (DATA / "splits" / f"{k}_bmrb_ids.txt").write_text("\n".join(ids) + "\n")
    test_ids = sorted(splits["test"], key=int)

    # --- truth from the crystalline cache --------------------------------
    npz = np.load(CRYST / "crystalline_all.npz")
    meta = json.load(open(CRYST / "crystalline_all_meta.json"))
    shifts, masks = npz["shifts"], npz["masks"]
    entry_ids, seq_ids, comp_ids = meta["entry_ids"], meta["seq_ids"], meta["comp_ids"]
    prov["sources"]["crystalline_all.npz"] = {"path": str(CRYST / "crystalline_all.npz"),
                                              "sha256": sha256(CRYST / "crystalline_all.npz"),
                                              "n_rows": int(len(entry_ids))}
    test_set = set(test_ids)
    entries: dict[str, dict] = defaultdict(lambda: {"shifts": {}, "seq_ids": [], "comp_ids": [], "n_observed": 0})
    n_range_dropped = defaultdict(int)
    for i, eid in enumerate(entry_ids):
        eid = str(eid)
        if eid not in test_set:
            continue
        e = entries[eid]
        sid = int(seq_ids[i])
        e["seq_ids"].append(sid)
        e["comp_ids"].append(str(comp_ids[i]))
        res = {}
        for j, nuc in enumerate(NUCLEI):
            if masks[i, j] > 0 and not np.isnan(shifts[i, j]):
                v = float(shifts[i, j])
                if MIN_SHIFT[nuc] <= v <= MAX_SHIFT[nuc]:
                    res[nuc] = v
                else:
                    n_range_dropped[nuc] += 1
        if res:
            e["shifts"][sid] = res
            e["n_observed"] += 1
    prov["truth_range_dropped"] = dict(n_range_dropped)

    # --- blocklist -------------------------------------------------------
    bl = json.load(open(BLOCKLIST))
    prov["sources"]["blocklist"] = {"path": str(BLOCKLIST), "sha256": sha256(BLOCKLIST),
                                    "metadata": bl["metadata"]}
    with open(BLOCKLIST, "rb") as fi, gzip.open(DATA / "label_blocklist_v2_sweep_carbons.json.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    drop_entries = set(bl["drop_entries"])
    cleaned = [e for e in test_ids if e not in drop_entries]
    (DATA / "cleaned_test_ids.txt").write_text("\n".join(cleaned) + "\n")
    with open(DATA / "test_label_drops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmrb_id", "seq_id", "nucleus"])
        n_drops = 0
        for eid in test_ids:
            for sid, nucs in bl["blocklist"].get(eid, {}).items():
                for nuc in nucs:
                    w.writerow([eid, int(sid), nuc])
                    n_drops += 1
    prov["test_label_drops"] = n_drops
    prov["cleaned_test_ids"] = len(cleaned)

    # --- sample conditions (pH is passed to UCBShift2 as its --pH input) ---
    # the conditions file the graph builder used (pH known for only a minority of entries;
    # the model itself does not consume pH — it is passed to UCBShift2 where known)
    sc_path = GRAPH_DIR / "sample_conditions.json"
    sc = json.load(open(sc_path))
    with open(DATA / "test_conditions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmrb_id", "ph", "temperature_k", "ionic_strength_m"])
        for eid in test_ids:
            c = sc.get(eid, {})
            w.writerow([eid, c.get("ph", ""), c.get("temperature_k", ""), c.get("ionic_strength_m", "")])
    prov["sources"]["sample_conditions"] = {"path": str(sc_path), "sha256": sha256(sc_path),
                                            "n_test_with_ph": sum(1 for e in test_ids if "ph" in sc.get(e, {}))}

    # --- structure choice + residue map ----------------------------------
    import gemmi
    bmrb_to_pdb = json.load(open(GRAPH_DIR / "bmrb_to_pdb.json"))
    status = defaultdict(int)
    inputs_rows, map_rows, truth_rows = [], [], []
    for eid in test_ids:
        e = entries.get(eid)
        if e is None or e["n_observed"] < MIN_OBSERVED:
            status["too_few_shifts"] += 1
            inputs_rows.append([eid, "", "", "", "", "", "", "skip_too_few_shifts", ""])
            continue
        bmrb_seq, bmrb_sids = build_chem_shift_sequence(e)
        chosen = None
        cands = bmrb_to_pdb.get(eid) or []
        any_cif = False
        for pdb_id in dict.fromkeys(cands):  # de-dup, keep order
            cif = PDB_DIR / f"{pdb_id}.cif"
            if not cif.exists():
                continue
            any_cif = True
            try:
                mapping, chain = validate_sequence_mapping(
                    bmrb_seq, cif, return_chain_id=True, min_identity=MIN_IDENTITY, bmrb_seq_ids=bmrb_sids)
            except Exception:
                mapping = None
            if mapping:
                chosen = (pdb_id, cif, chain, mapping)
                break
        if chosen is None:
            st = "skip_no_cif" if not any_cif else "skip_bad_align"
            status[st] += 1
            inputs_rows.append([eid, "|".join(cands), "", "", "", "", bmrb_seq, st, ""])
            continue
        pdb_id, cif, chain, mapping = chosen
        st = gemmi.read_structure(str(cif))
        try:
            method = st.info["_exptl.method"] if "_exptl.method" in st.info else ""
        except Exception:
            method = ""
        n_models = len(st)
        prod_graph = GRAPH_DIR / f"{eid}.pt"
        status["mapped"] += 1
        inputs_rows.append([eid, pdb_id, chain, n_models, method, len(bmrb_seq), bmrb_seq, "ok",
                            "yes" if prod_graph.exists() else "no"])
        for pdb_sid, bmrb_sid in sorted(mapping.items()):
            map_rows.append([eid, pdb_id, chain, int(pdb_sid), int(bmrb_sid)])
        for sid in sorted(e["shifts"]):
            comp = e["comp_ids"][e["seq_ids"].index(sid)]
            for nuc, v in e["shifts"][sid].items():
                truth_rows.append([eid, sid, comp, nuc, f"{v:.3f}"])

    with open(DATA / "test_inputs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmrb_id", "pdb_id", "chain_id", "n_models", "exptl_method", "n_residues_bmrb",
                    "bmrb_sequence", "status", "production_graph_exists"])
        w.writerows(inputs_rows)
    with open(DATA / "residue_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmrb_id", "pdb_id", "chain_id", "pdb_seq_id", "bmrb_seq_id"])
        w.writerows(map_rows)
    with open(DATA / "truth_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bmrb_id", "seq_id", "comp_id", "nucleus", "value"])
        w.writerows(truth_rows)
    prov["status"] = dict(status)
    prov["n_truth_rows"] = len(truth_rows)
    prov["n_mapped_residues"] = len(map_rows)

    # --- consistency check against the production graphs -----------------
    try:
        import torch
        mismatch, checked = [], 0
        for r in inputs_rows:
            if r[7] != "ok" or r[8] != "yes":
                continue
            d = torch.load(str(GRAPH_DIR / f"{r[0]}.pt"), weights_only=False)
            checked += 1
            if str(getattr(d, "pdb_id", "")).lower() != str(r[1]).lower():
                mismatch.append((r[0], r[1], str(getattr(d, "pdb_id", ""))))
        prov["production_graph_pdb_check"] = {"checked": checked, "mismatch": mismatch}
    except Exception as exc:  # torch/torch_geometric not importable
        prov["production_graph_pdb_check"] = {"error": repr(exc)}

    json.dump(prov, open(DATA / "provenance.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in prov.items() if k != "sources"}, indent=1))
    print("splits sha256:", prov["sources"]["splits"]["sha256"])


if __name__ == "__main__":
    main()
