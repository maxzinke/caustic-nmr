# Benchmarks

How every number in this repository was measured. The tables are regenerated from
[`benchmarks/results/per_residue.csv.gz`](../benchmarks/results/per_residue.csv.gz) by
[`benchmarks/rescore.py`](../benchmarks/rescore.py); CI re-derives them and fails on
drift. Companions: [METHOD.md](METHOD.md), [DATA.md](DATA.md),
[LIMITATIONS.md](LIMITATIONS.md), [../benchmarks/README.md](../benchmarks/README.md).

## 1. Test set

The production test split: **735 BMRB entries**, separated from training by 3-mer cosine
sequence similarity ≤ 0.5 under single linkage ([DATA.md](DATA.md) §4); the SHA-256 of
the split file is stamped inside the trained checkpoint. Reference shifts: **350,875
labels** from the BMRB entries themselves. Two slices are reported:

- **full** (primary): all 735 entries.
- **cleaned** (secondary): the 617 entries not on the whole-entry label blocklist, minus
  per-label drops. The blocklist is a *training-side* referencing-drift detector, so this
  slice preferentially removes entries on which every predictor looks worse — reported for
  continuity with the internal 0.3.0 record (measured on the 614 of these entries where
  both internal pipelines had predictions), never as the headline
  ([LIMITATIONS.md](LIMITATIONS.md) §3).

Reference-shift sanity (ranges, duplicates, per-protein referencing offsets):
[`benchmarks/results/truth_sanity.md`](../benchmarks/results/truth_sanity.md). Out of
350,875 labels, 35 (0.01 %) fall outside broad plausibility windows and 64 proteins show
a per-nucleus median referencing offset beyond 0.5 ppm (H/HA) or 2 ppm (heavy atoms);
they are left in — the reference is taken as deposited.

## 2. Methods compared

| Method | Exact form run | Version of record |
|---|---|---|
| **CAUSTIC** | The public path: installed `caustic-nmr` wheel, `predict_shifts_onnx`, bundled ONNX + slim calibrator, package defaults (all conformers up to 20, median), on the mmCIF + chain the production pipeline aligned | package + ONNX SHA-256 in the CSV header |
| **SPARTA+** | `SPARTA+ 2.90 build 2017.143.12.12` | `sparta_run_log.json` |
| **LEGOLAS** | github.com/roitberg-group/legolas, 30-model ensemble (model-file hash recorded) | `legolas_run_log.json` |
| **UCBShift2** | CSpred @ `844e265f` (2026-04-03), **full mode** — X (ML) + Y (transfer: BLAST + mTM-align against its 1,968-structure refDB) — with the BMRB sample pH where known. *Not* `--shiftx_only`. | `ucbshift_run_log.json` |
| UCBShift-X | The ML-only column of the same UCBShift2 run (shown to separate the transfer module's contribution) | same log |

Competitors receive single-model PDBs (model 1, aligned chain, PDBFixer-completed heavy
atoms, hydrogens at pH 7); CAUSTIC reads the raw mmCIF including all conformers. Each
method gets the input form it is designed for; the asymmetry is disclosed here and in
`benchmarks/README.md`.

Two internal reference columns (not competitors): `caustic_fullcal` = public predictions
plus the record's stratum calibrator (§6), and `caustic_recordgraph` = the shipped ONNX
run on the production training-pipeline graphs (medoid conformer), isolating the
public-path graph-parity gap (§7).

## 3. Metrics, pairing, uncertainty

Definitions live in code at the top of `rescore.py`; in prose:

- **MAE** per nucleus = mean |prediction − reference| in ppm.
- **Pairing**: a CAUSTIC-vs-X row uses exactly the residues where the reference and
  *both* predictions exist, per nucleus. The all-methods table uses residues every
  method predicted. Unpaired ("coverage") tables score each method on its own
  predictions and are never used for a claim.
- **Composite** per protein = per-nucleus MAE, weighted mean over the nuclei the protein
  has (H 1, HA 1, N 1, CA 1.5, CB 2, C 1) — the definition of the internal record,
  copied from the record script.
- **CIs**: protein-level paired bootstrap (B = 2000, seed 42) — proteins resampled with
  replacement, so correlated residues within a protein do not shrink the interval.
- **Sign test**: fraction of proteins with the lower composite, exact binomial p.

**Crash accounting.** Every failure is a logged per-entry error, never a silent drop.
Final counts: CAUSTIC **735/735**, SPARTA+ **734/735**, LEGOLAS **734/735**,
UCBShift2 **732/735**. The failure shared by all competitors is BMRB 30140 (PDB 5kwo),
whose D-amino acids (DAR, DSG) have no ideal CCD coordinates, so no competitor input
could be prepared; UCBShift2 additionally crashed on BMRB 15013 and 17089
("residue number list is not ordered" in its SPARTA+-style feature reader — those PDBs
genuinely have non-monotonic author numbering). Full messages in the run logs.

## 4. Results — full test set (primary)

Per-protein weighted composite (paired by protein; Δ < 0 favours CAUSTIC;
CIs = protein-level bootstrap, B = 2000, seed 42):

| vs | n prot. | CAUSTIC | Competitor | Δ [95 % CI] | rel. | proteins better | sign-test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPARTA+ | 733 | 0.787 | 1.062 | −0.276 [−0.318, −0.243] | −25.9 % | 720/733 (98.2 %) | 1.1e−193 |
| LEGOLAS | 734 | 0.787 | 1.216 | −0.429 [−0.450, −0.412] | −35.3 % | 731/734 (99.6 %) | 1.5e−213 |
| **UCBShift2 (full)** | 732 | 0.787 | 0.889 | −0.102 [−0.127, −0.083] | −11.5 % | 615/732 (84.0 %) | 2.1e−82 |
| UCBShift-X (ML only) | 732 | 0.787 | 0.950 | −0.163 [−0.184, −0.148] | −17.1 % | 695/732 (94.9 %) | 2.6e−158 |

Per-nucleus MAE (ppm) on the common residue set of all four methods
(344,487 residues, 731 proteins):

| Nucleus | n | CAUSTIC | SPARTA+ | LEGOLAS | UCBShift2 |
|---|---:|---:|---:|---:|---:|
| H | 62,197 | **0.309** | 0.426 | 0.525 | 0.343 |
| HA | 51,541 | **0.175** | 0.233 | 0.251 | 0.192 |
| N | 61,978 | **1.713** | 2.321 | 2.727 | 1.918 |
| CA | 64,504 | **0.764** | 0.975 | 1.091 | 0.834 |
| CB | 58,264 | **0.860** | 1.082 | 1.311 | 0.920 |
| C | 46,003 | **0.804** | 1.029 | 1.109 | 0.919 |

Every pairwise per-nucleus difference has a bootstrap CI excluding zero (closest case:
CAUSTIC−UCBShift2 on CB, −0.060 [−0.081, −0.039]). The complete tables — per-nucleus
paired rows with CIs, coverage, and the unpaired variants — are generated into
[`benchmarks/results/tables.md`](../benchmarks/results/tables.md) by `rescore.py`.

## 5. Results — cleaned slice (secondary)

Same protocol on the 617-entry cleaned slice (composite, paired; Δ in ppm):
CAUSTIC 0.753 vs SPARTA+ 1.045 (Δ −0.292 [−0.343, −0.254], −27.9 %), LEGOLAS 1.217
(Δ −0.463 [−0.495, −0.437], −38.1 %), UCBShift2 0.862 (Δ −0.109 [−0.139, −0.087],
−12.6 %, 501/615 proteins, p = 8.9e−59). The ranking and
roughly the margins of §4 are unchanged; only absolute MAEs drop, as expected when
referencing-drift-flagged entries are removed. Full tables in
[`benchmarks/results/tables.md`](../benchmarks/results/tables.md).

## 6. The shipped calibrator vs the calibrator of record (D7)

The package ships the *slim* SA16 v2 calibrator (global per-nucleus offsets + cysteine
CB modifiers). The internal record was measured with a calibrator that additionally
applied per-(nucleus × DSSP class × aromatic-ring bin × rSASA bin) stratum offsets.
`benchmarks/calibrator_gap.py` applies those stratum offsets on top of the public-path
predictions (the shared components are asserted identical, so the difference is the
strata alone).

Measured (full slice, paired; negative = full calibrator better): H −0.24 %,
HA −1.99 %, N −0.05 %, CA −1.05 %, CB −0.81 %, C −0.26 %; composite +0.55 % in the full
calibrator's favour (+0.0043 [+0.004, +0.005]). The gain is real but small — under the
±0.005 ppm seed-to-seed noise floor ([LIMITATIONS.md](LIMITATIONS.md) §2) — and the
stratum features require the DSSP label that only the private training graphs carry
(the public package cannot compute it, see §7), so **the package keeps the slim
calibrator** and this section documents the measured cost. Numbers:
`benchmarks/results/calibrator_gap.json`, column `caustic_fullcal` in the table of
record; the stratum features for this measurement were taken from the production
graphs (`benchmarks/calibrator_gap.py`).

## 7. Does the public path reproduce the internal record?

The 0.3.0 record (composite 0.69301 on the 614-entry cleaned slice) was measured on
graphs built by the training pipeline. Differences between that record and the public
path measured here: graph construction from the raw structure file instead of cached
production graphs; up to 20 conformers (median) instead of max 6; the slim instead of the
full calibrator; and 617 vs 614 entries. The `caustic_recordgraph` column bounds the
graph-construction share of the gap.

Measured on the cleaned slice: public path **0.753** vs record **0.693** (+8.7 %).
Decomposition: the calibrator explains −0.003 (public + record strata = 0.750); the
graph features do **not** explain it — the same ONNX on the production medoid graphs
(real DSSP, training-pipeline contacts) scores **worse** (0.767; the public path's
median over all conformers beats the medoid single-graph: composite Δ −0.014
[−0.016, −0.011] on this slice, CI excluding zero). The remaining gap is attributable to
the record pipeline's per-conformer graph ensemble (up to 6 separately built,
fully-featured conformer graphs averaged after prediction) and its rebuilt v50c graph
cache, neither of which the public package reproduces; the record's input cache no
longer exists, so the decomposition cannot be carried further. **All numbers published
in this repository are the public path's**; the record value is quoted only here, as
provenance.

One input-parity fact discovered by this measurement: the DSSP secondary-structure
feature is zero-filled on the public path ([METHOD.md](METHOD.md) §2,
[LIMITATIONS.md](LIMITATIONS.md) §8).

## 8. Competitor fairness

- **Versions and commands** are recorded per run log (§2) and in each predictions-file header.
- **UCBShift2 runs in full mode.** An earlier internal comparison used `--shiftx_only`
  (alignment module disabled); no number from that comparison appears in this repository.
- **Reference-database overlap**: 67 of the 693 distinct test PDB ids appear in
  UCBShift2's shipped refDB — for those proteins its transfer module has effectively seen
  the answer sheet's structure. Overlap ids: `benchmarks/results/leakage_report.json`.
  Measured (`benchmarks/ucbshift_overlap.py` → `results/ucbshift_overlap.json`): on the
  **71 overlap proteins** UCBShift2's transfer module changed 92.8 % of predictions and
  its MAE collapses to H 0.083 / CA 0.493 / CB 0.499 ppm — far better than CAUSTIC
  there (H 0.294 / CA 0.712 / CB 0.817), as expected when the answer sheet's structure
  is in the reference set. On the **661 non-overlap proteins** (7.7 % of predictions
  changed by the transfer module) CAUSTIC is better on every nucleus (e.g. CA 0.770 vs
  0.872, N 1.718 vs 2.021). The §4 headline *includes* the overlap proteins, i.e. it is
  conservative against CAUSTIC. SPARTA+ and LEGOLAS ship no enumerable training-set
  list, so their overlap with the test set could not be computed; both are trained on
  BMRB-linked structures and some overlap is likely — recorded as unquantifiable in
  `results/leakage_report.json`.
- **Split hygiene**: train/val/test BMRB id lists are disjoint (asserted by
  `check_leakage.py` in CI). 13 test PDB ids are also linked to train/val BMRB entries
  (the same structure deposited under a different BMRB id); the split is by sequence
  similarity, not PDB id, and these are listed in the leakage report.
- **What is not protected**: fold-level similarity below the sequence threshold
  ([LIMITATIONS.md](LIMITATIONS.md) §9).
- **Why the competitors' raw predictions are published here.** `benchmarks/results/`
  ships each tool's per-residue output (`predictions_legolas.csv.gz`,
  `predictions_sparta.csv.gz`, `predictions_ucbshift*.csv.gz`) so that anyone can
  re-score the comparison without installing four programs, and so that a disagreement
  with the numbers in §4 can be traced to a specific residue. These files are **output
  generated by running each tool on public PDB structures**, not copies of the tools
  themselves. Their terms differ: **LEGOLAS** is MIT; **SPARTA+** is distributed under
  terms that forbid redistributing *the software* without the authors' permission and
  are silent on generated output; the exact **CSpred/UCBShift2** checkout used here
  includes the UC Regents licence, which permits educational, research and not-for-profit
  use and redistribution of the software with its notices retained. No software from
  any competitor is redistributed here. Where a tool's terms do not address generated
  output, the durable publication record should prefer derived per-residue errors and
  aggregate statistics over mirroring full predictions; see the benchmark data notes.

## 9. Figures

`benchmarks/figure_summary.py` regenerates the README figure
(`benchmarks/results/figures/benchmark_summary.png`) from the table of record —
per-nucleus MAE of all four methods on the common residue set. Reference-shift
histograms per nucleus: `benchmarks/results/figures/truth_hist_*.png`
(from `truth_sanity.py`).
