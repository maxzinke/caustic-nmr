# Benchmarks

Everything the README and `docs/BENCHMARKS.md` claim is regenerated from one per-residue
table produced by the scripts in this directory. Ten-minute tour below; per-section detail
in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md); one protein end-to-end in
[WALKTHROUGH.md](WALKTHROUGH.md).

## What is measured

Backbone chemical-shift MAE (H, HA, N, CA, CB, C') on the **production test split**:
735 BMRB entries (`data/splits/test_bmrb_ids.txt`), held out from training by 3-mer cosine
sequence similarity ≤ 0.5 under single linkage ([docs/DATA.md](../docs/DATA.md) §4). The
SHA-256 of the split file is stamped inside the trained checkpoint. Two slices:

- **full** — every reference shift of the 735 entries (primary).
- **cleaned** — the 617 entries not on the whole-entry label blocklist, minus per-label
  drops (`data/cleaned_test_ids.txt`, `data/test_label_drops.csv`). This is close to the
  slice the internal 0.3.0 record used (614 = the subset where both internal pipelines had
  predictions). Reported second, because the blocklist is a training-side detector
  ([docs/LIMITATIONS.md](../docs/LIMITATIONS.md) §3).

Reference shifts (`data/truth_test.csv`, 350,875 labels) come from the BMRB entries;
sanity checks on them (ranges, duplicates, per-protein referencing offsets) are in
[results/truth_sanity.md](results/truth_sanity.md).

## The methods

| Column | What ran | Version record |
|---|---|---|
| `caustic` | Public path: installed `caustic-nmr` wheel, `predict_shifts_onnx`, bundled ONNX + slim calibrator, package defaults, on the same mmCIF/chain the production pipeline aligned | package version + ONNX SHA-256 in the CSV header |
| `sparta` | SPARTA+ **2.90 build 2017.143.12.12** on prepared single-model PDBs | `results/sparta_run_log.json` |
| `legolas` | LEGOLAS (github.com/roitberg-group/legolas), 30-model ensemble; model-file hash in the CSV header | `results/legolas_run_log.json` |
| `ucbshift` | UCBShift2 (CSpred @ `844e265f`, 2026-04-03) in **full mode** (X ML + Y transfer with BLAST + mTM-align) — *not* `--shiftx_only` — with the BMRB sample pH where known | `results/ucbshift_run_log.json` |
| `ucbshift_x` | The UCBShift-X (ML-only) column of the same run | same log |
| `caustic_fullcal` | `caustic` + the record's stratum calibrator (D7 gap measurement, `calibrator_gap.py`) | `results/calibrator_gap.json` |
| `caustic_recordgraph` | Shipped ONNX on the production training-pipeline graphs (medoid conformer) — isolates the public-path graph-parity gap | `results/caustic_recordgraph_run_log.json` |

Competitor inputs are single-model PDBs prepared by `prepare_pdbs.py` (model 1, the
aligned chain, PDBFixer-completed heavy atoms, hydrogens at pH 7). CAUSTIC reads the raw
mmCIF (all conformers; median aggregation) — that asymmetry is deliberate: each method
gets the input form it is designed for, and it is disclosed wherever numbers are quoted.

**Crash accounting.** Every failure is a logged entry (`status: error` with the message),
never a silent drop; per-tool counts are in each run log's `summary` and quoted in
`docs/BENCHMARKS.md`. Known: BMRB 30140 (PDB 5kwo) contains D-amino acids whose CCD
definitions have no ideal coordinates, so no competitor input could be prepared — it is an
error entry for all three competitors.

**Pairing rule.** A CAUSTIC-vs-X comparison uses exactly the residues where the reference
value and *both* predictions exist (per nucleus); the "all methods" table uses the
residues every method predicted; unpaired coverage tables are labelled as such and never
used for a claim. Definitions live at the top of `rescore.py` and nowhere else.

**Leakage.** `check_leakage.py` asserts the split lists are disjoint (they are) and
reports overlap between the test structures and each competitor's reference data —
e.g. 67 of 693 test PDB ids appear in UCBShift2's `refDB` (its transfer module has seen
those structures; see `results/leakage_report.json` and the discussion in
`docs/BENCHMARKS.md`). 13 test PDB ids are also linked to train/val BMRB entries (same
structure deposited under another BMRB id); the split is by sequence, not PDB id.

## Regenerate

```bash
pip install caustic-nmr pandas scipy         # the public package is the only model dependency
python rescore.py                            # tables from results/per_residue.csv.gz
python rescore.py --bootstrap 2000           # + protein-level paired bootstrap CIs and sign tests
python rescore.py --check                    # recompute and diff against results/summary.json (CI runs this)
python check_leakage.py                      # split disjointness + reference-DB overlap (CI runs this)
python truth_sanity.py                       # reference-shift range/duplicate/referencing report
```

## Re-run from scratch

```bash
python build_inputs.py        # split lists, truth, residue map, inputs (needs the local caches)
python prepare_pdbs.py        # competitor single-model PDBs (gemmi + pdbfixer + openmm)
python run_caustic.py --shard 0/6 &   # ... 6 shards; then: python run_caustic.py --merge
python run_competitors.py --tool sparta     # needs SPARTA+ (SPARTAP_DIR)
python run_competitors.py --tool legolas    # needs the LEGOLAS checkout + torch
python run_competitors.py --tool ucbshift   # needs CSpred under WSL (see run log meta)
python build_table.py         # -> results/per_residue.csv.gz
python calibrator_gap.py      # D7: + record stratum offsets  -> caustic_fullcal
```

`build_inputs.py` needs the private BMRB/PDB caches and is only for rebuilding the inputs
from upstream; everything under `data/` and `results/` is committed so that `rescore.py`
and `check_leakage.py` run from a plain checkout.

## Files

- `data/` — split id lists, `test_inputs.csv` (BMRB→PDB/chain mapping used), `truth_test.csv`,
  `residue_map.csv` (PDB numbering ↔ BMRB seq id), blocklist + cleaned-set ids, sample conditions.
- `results/per_residue.csv.gz` — the single table of record (one row per reference shift,
  all methods' predictions).
- `results/summary.json`, `results/tables.md` — output of `rescore.py`.
- `results/*_run_log.json` — per-entry status, versions, wall times.
- `results/truth_sanity.md`, `results/leakage_report.json`, `results/figures/`.
