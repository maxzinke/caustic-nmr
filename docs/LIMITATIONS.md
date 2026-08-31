# Limitations

What CAUSTIC does not do, where its numbers should be read with care, and which
engineering choices constrain its use. Companion documents: [METHOD.md](METHOD.md),
[DATA.md](DATA.md), [BENCHMARKS.md](BENCHMARKS.md), [../README.md](../README.md),
[../CHANGELOG.md](../CHANGELOG.md).

## 1. Dependencies

Inference runs through ONNX Runtime, but graph construction still builds PyTorch tensors:
`caustic/features.py`, `model.py` and `ensemble.py` import `torch` at module level and
`inference.py` imports `features` unconditionally. **PyTorch is therefore a hard
dependency** (`pyproject.toml`), and the CLI help text that says "no PyTorch needed" for
the ONNX backend describes an intention, not the current code. `torch_geometric` is
optional (a fallback `ProteinData` container is used when it is absent, `graph.py:312`).

## 2. Single seed

The production model was trained once (seed 42). Seed-to-seed variation of the composite
MAE was estimated at roughly ±0.005 ppm on the internal validation runs (from the training
notes, not re-measured for this release). Differences smaller than that between CAUSTIC
versions, or between CAUSTIC and another predictor, should not be read as real.

## 3. The evaluation set is filtered by a training-side detector

The label blocklist ([DATA.md](DATA.md) §3) was built on train, validation **and test**.
Entries flagged for whole-entry drop are absent from the "safe-test" subset, which is why
the internal comparison of record uses **614** of the 735 `cc.test` entries. Removing
test entries with a detector that was tuned to improve training is a form of selection:
it preferentially discards entries whose deposited shifts disagree with random-coil
expectations, which are also entries on which any predictor looks worse. Numbers on the
614-entry subset are therefore not comparable to numbers on the full 735-entry set, and
[BENCHMARKS.md](BENCHMARKS.md) reports both, with the full set as the primary one.

Related: the training labels carry measurable referencing drift. A random-coil detector at
tight thresholds flagged 7.3 % of training labels (CB 12.3 %, HA 18.7 %) in the audit that
preceded the blocklist; a LACS-style detector flagged 0.84 %. Cleaning improved the carbon
nuclei and left H, HA and N unchanged, which the training notes attribute to proton and
nitrogen noise being dominated by dynamics and exchange rather than referencing. Gains
under about 1 % relative MAE may be inside the label-noise floor.

## 4. AlphaFold inputs

AlphaFold models are detected and their pLDDT column is converted to a pseudo-B-factor
(`100 − pLDDT`) that enters the node features ([METHOD.md](METHOD.md) §2). **No
pLDDT-dependent widening of σ is implemented**: the reported uncertainty of a residue with
low pLDDT comes only from the network's log-variance head, which was trained on
experimental structures. Treat σ on low-confidence AlphaFold regions as a lower bound.
Earlier versions of the README described a "pLDDT-calibrated sigma widening"; that feature
does not exist in the code.

## 5. Uncertainty calibration

σ is `exp(0.5 · logvar)` from a head trained with a Student-t negative log-likelihood
(ν per nucleus between 4.6 and 6.7). There is **no isotonic or quantile calibration step**
in the package; the `--quantile` CLI option only sets the coverage label written into the
NEF/NMR-STAR uncertainty column. Empirical coverage of the reported σ is given in
[BENCHMARKS.md](BENCHMARKS.md) where measured; until then treat σ as a relative ranking of
residues rather than a calibrated 68 % interval.

## 6. Calibrator is a reduced version of the one used for the record

The shipped calibrator applies global per-nucleus offsets (all < 0.03 ppm) and a CB
offset for cysteines (+1.36 ppm disulfide, +0.32 ppm otherwise). The calibrator with which
the internal comparison of record was made additionally applied per-(DSSP × aromatic ×
rSASA) stratum offsets. Metal-bound cysteines are not detected (they receive the
reduced/free offset; the calibrator's own estimate of that misclassification cost is
≈0.25 ppm on CB for those residues). The measured effect of omitting the stratum offsets
is reported in [BENCHMARKS.md](BENCHMARKS.md).

## 7. Conformers and ensembles

For multi-model files the prediction is the median over up to 20 conformers and σ is the
mean log-variance ([METHOD.md](METHOD.md) §5). The model was trained with stochastic
conformer sampling, so it approximates a population average; predictions from a single
conformer (`--ensemble first`, or any X-ray / AlphaFold input) are noisier and were not
separately benchmarked. Conformer count is a model input (feature slot 11, `n/20` capped
at 1), so an ensemble truncated to 20 models is treated the same as a 20-model ensemble.

## 8. Chains, residues, atoms

- One chain per call: the first polymer chain unless `--chain` is given. Inter-chain
  contacts are not seen (the other chains are not in the graph), so shifts at
  oligomer interfaces and ligand-binding sites are predicted as if the partner were absent.
  Hetero atoms, metals and waters are not graph nodes in the production configuration
  (`include_hetero_nodes=False`, `config.py:35`).
- Residues whose backbone atoms are incomplete cannot be targeted; their entries in the
  output are NaN. Residues with fewer than the canonical heavy atoms may still be
  predicted from what is present.
- Thirty non-standard residue names are mapped to canonical parents
  ([METHOD.md](METHOD.md) §2); the modification itself (phosphate, selenium, hydroxyl,
  methylation) is invisible to the model, so shifts of modified residues and their
  neighbours are predicted as for the parent residue. Any other residue name becomes
  `UNK`, which the model has effectively never seen.
- Chains longer than 500 residues trigger a warning but are processed
  (`large_protein_warn_threshold`); memory grows with atom count × 32 neighbours.
- Sample temperature is fixed at 298 K at inference and pH is not an input
  ([METHOD.md](METHOD.md) §2). Shifts of titratable residues at unusual pH, and the
  temperature dependence of amide shifts, are not modelled.
- **The DSSP secondary-structure feature is zero at inference.** Feature [30] of the
  residue geometry block was pre-stored on the training-pipeline graphs by DSSP; the
  package's graph builder cannot compute it and zero-fills it (every residue reads as
  "coil" to the network). The measured cost is small — the public path scores within
  2 % of the same model on production graphs, in the public path's favour
  ([BENCHMARKS.md](BENCHMARKS.md) §7) — but inputs are not bit-identical to training.
- Synthesised H/HA positions are idealised; on structures with real hydrogens the model
  uses those, so a small systematic difference between "with H" and "without H" inputs
  should be expected (not quantified here — not verified).

## 9. What the split does and does not protect against

The train/test separation is by 3-mer cosine similarity ≤ 0.5 under single linkage
([DATA.md](DATA.md) §4). This is a sequence-level hold-out. It does not guarantee that a
test protein has no structural homologue in training (fold-level similarity with low
sequence identity), nor that the same protein does not appear under different BMRB
entries with different constructs when their 3-mer profiles fall below the threshold. The
shift-prediction literature usually reports sequence-identity hold-outs, so CAUSTIC's
protocol is comparable to, not stricter than, common practice.

## 10. Comparisons with other predictors

Any published comparison must state, for each competitor, the exact version, the command
line, its default options (in particular whether UCBShift2 was run with or without its
alignment module), how crashes and missing predictions were counted, and how much of the
test set overlaps with that predictor's own training database. The comparison of record
for this release, with those disclosures, is in [BENCHMARKS.md](BENCHMARKS.md); earlier
internal numbers that did not meet these requirements are not repeated in this repository.
