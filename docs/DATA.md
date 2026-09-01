# Data

Where the training data came from, how it was curated, and how the train/validation/test
split was made. The dataset itself is not redistributed (see §6); the split id lists and the
label blocklist are published under [`../benchmarks/`](../benchmarks/README.md). Companion
documents: [METHOD.md](METHOD.md), [BENCHMARKS.md](BENCHMARKS.md),
[LIMITATIONS.md](LIMITATIONS.md), [../LICENSE-WEIGHTS](../LICENSE-WEIGHTS).

## 1. Sources

| Source | What is used | Access |
|---|---|---|
| BMRB (Biological Magnetic Resonance Data Bank) | assigned backbone chemical shifts (H, HA, N, CA, CB, C'), sequence, sample temperature and pH | BMRB API v2 (`https://api.bmrb.io/v2/…`, training-repository `dataset_builder.py:33`) |
| wwPDB | atomic coordinates of the structure(s) linked to each BMRB entry; NMR ensembles are used as multi-conformer inputs | mmCIF download, cached locally |
| RCSB search API | homologous structures (>95 % sequence identity) for entries without a directly linked PDB id | `https://search.rcsb.org/rcsbsearch/v2/query` (`structure_expansion.py:25`) |
| AlphaFold DB | predicted models for a small number of entries without an experimental structure | 83 entries in the cache map (`bmrb_to_af2.json`) |

The BMRB→PDB map in the training cache holds **7,645** BMRB entries; the split used for
the production model covers **4,904** of them (see §4). A BMRB entry can map to several
PDB ids (e.g. BMRB 10002 → 1VEX ×2); the graph builder uses one structure per entry and,
for NMR entries, all of its models as conformers.

## 2. Labels

Labels are the deposited chemical shifts, residue-matched to the structure sequence.
Physically impossible values are discarded with per-nucleus windows (ppm): H [−2, 15],
HA [−2, 15], N [80, 200], CA [30, 100], CB [0, 100], C [150, 200] (`DataConfig.max_shift`
/ `min_shift`, `caustic/config.py:225-230`). No re-referencing is applied to the labels;
instead, entries and (entry, nucleus) pairs with detectable referencing drift are removed
by the blocklist described in §3.

Sample conditions (temperature, pH) come from the BMRB entry metadata
(`sample_conditions.json`, 5,120 entries in the cache). Entries whose recorded temperature
is outside **273–320 K** are dropped (`temperature_filter_min/max`); temperature is a model
input (`(T − 298)/20`), pH is not used by the production model (`use_solchem: false`).

## 3. Curation

**Residue-level geometry filter.** Each residue in the training cache carries a geometry
class from the training repository's `geometry_labels.py` policy, which combines an X-ray
consensus (where available) with the NMR-ensemble backbone variance: `XG` (X-ray gold —
trusted X-ray geometry confirmed well-defined by NMR), `XU` (X-ray usable but NMR shows
disorder), `NG` (NMR good — no X-ray, low ensemble variance), `D` (disordered) and `C`
(conflict). Only residues of class **NG or XG** contribute labels (`geometry_labels:
["NG", "XG"]`). An entry is kept only if at least **50 %** of its residues have a complete
backbone (`min_backbone_completeness: 0.5`).

**Label blocklist ("carbon-aggressive cleaning").** Two referencing-drift detectors were
run per entry and nucleus on the training, validation and test partitions
(`scripts/phase1_build_blocklist.py`, variant `v2`, 2026-05-21):

- a random-coil mean-offset detector: the mean secondary shift of an entry (observed minus
  random-coil value) is compared to an expected mean (0 for H/HA/N, +0.3 CA, +0.2 CB,
  −0.1 C ppm); an (entry, nucleus) is flagged when |offset| exceeds **0.35 ppm (H, HA),
  2.0 ppm (N) or 1.0 ppm (CA, CB, C)** over at least 10 residues;
- a LACS-style secondary-structure-conditional detector.

Flags are unioned. A flagged (entry, nucleus) masks every label of that nucleus in the
entry ("fine drop"). An entry is dropped entirely when ≥ 3 nuclei are flagged, any offset
exceeds 2× its threshold, or the fine drop would remove most of its labels
(`WHOLE_DROP_*` constants). Result
(`label_blocklist_v2_sweep_carbons.json`, `metadata`):

| Nucleus | fine-dropped labels | labels removed with whole-entry drops |
|---|---|---|
| H | 2,434 | 48,595 |
| HA | 3,094 | 37,321 |
| N | 13,206 | 48,348 |
| CA | 46,803 | 51,874 |
| CB | 88,190 | 46,149 |
| C | 20,081 | 29,735 |

744 entries are dropped whole; 1,705 entries carry fine masks. In the test partition 374 of
735 entries have at least one flag (CB 253, CA 177, C 77, N 55, H 24, HA 24). The
training configuration comment records the resulting drop rates as CB 34.17 %, CA 22.73 %,
C 15.97 %, HA 11.35 % of labels (fine + whole; the fine-only shares recomputed from the
blocklist metadata against the training label counts CA 305,806 / CB 276,689 / C 217,789
are CA 15.3 %, CB 31.9 %, C 9.2 %).

Why carbons: an earlier, symmetric cleaning pass improved CA/CB/C' but not H/HA/N; the v2
thresholds therefore tighten the carbons (1.5 → 1.0 ppm) and loosen HA (0.25 → 0.35 ppm).
The same blocklist is applied to the *evaluation* set — see [LIMITATIONS.md](LIMITATIONS.md)
§3 for why that matters.

**Disulfides.** A metal-aware disulfide side-car (`disulfide_sidecar_path`) overrides the
distance-based disulfide flag for cysteines coordinating metals (not verified in detail;
file not inspected).

## 4. Split

The split is **sequence-based, single-linkage, leak-safe** (training repository
`crystalline/data/splits.py:48-128`, `cluster_by_similarity` with
`linkage="connected_components"`):

```python
vectors = np.array([_sequence_to_3mer_freq(sequences[eid]) for eid in entry_ids])
sim_matrix = vectors @ vectors.T          # cosine similarity of 3-mer frequency vectors
np.fill_diagonal(sim_matrix, 0.0)
adj = (sim_matrix > threshold).astype(np.uint8)   # threshold = 0.5
_, labels = connected_components(csr_matrix(adj), directed=False)
```

Two entries are in the same cluster whenever a chain of pairwise 3-mer cosine similarities
> 0.5 connects them, so **no pair of entries in different partitions has similarity above
0.5**. Clusters are then assigned to partitions. The production split file
(`splits.connected_components.json`) contains

| Partition | entries | published as |
|---|---|---|
| train | 3,433 | `benchmarks/splits/train_ids.txt` |
| validation | 736 | `benchmarks/splits/val_ids.txt` |
| test (`cc.test`) | 735 | `benchmarks/splits/test_ids.txt` |

Ids are BMRB entry ids. The SHA-256 of the split file is stamped into the training
checkpoint (`splits_hash`), so the model can be tied to exactly this partition.

An older partition of the same sizes (`splits.json`, "complete"-linkage, documented in the
code as leaky) exists in the training cache and shares only 101 test ids with the
production split; benchmark numbers computed against it are not comparable and are not
used here.

Label counts in the test partition before the blocklist (from the blocklist metadata):
CA 65,485 · CB 59,151 · C 46,942 · H 63,173 · HA 52,283 · N 63,022. After whole-entry
drops the evaluated "safe-test" subset has **614** entries.

## 5. Sizes at a glance

| Quantity | Value |
|---|---|
| BMRB entries with a mapped structure (cache) | 7,645 |
| entries in the production split | 4,904 (3,433 / 736 / 735) |
| entries with AlphaFold-DB models instead of experimental structures | 83 |
| training labels CA / CB / C before cleaning | 305,806 / 276,689 / 217,789 |
| model parameters | 741,024 |

The number of NMR-ensemble versus single-model structures in the training set was not
counted for this document (not verified).

## 6. Licences and redistribution

- **BMRB.** Deposited data are released under **CC0 1.0** — a public-domain dedication
  with no attribution requirement and no restriction on redistribution or commercial
  use. This follows from BMRB's wwPDB membership and was confirmed directly with BMRB
  on 2026-08-06. Citation is therefore courtesy rather than obligation, and BMRB asks
  for Hoch *et al.*, *Nucleic Acids Res.* **51**, D368 (2023),
  doi:[10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050), plus the
  depositors. No BMRB record is redistributed with this package in any case; the
  training labels are not shipped.
- **wwPDB.** Coordinate data are in the public domain under CC0 1.0
  (https://www.wwpdb.org/about/usage-policies, checked 2026-09-01). Training coordinates
  are not redistributed. The package source distribution includes `examples/1ubq.pdb`
  as a CC0-licensed worked example.
- **AlphaFold DB.** Predictions are distributed under CC BY 4.0 according to EMBL-EBI
  (https://alphafold.ebi.ac.uk/faq, checked 2026-09-01). The package source distribution
  includes `examples/AF-P01112-F1-model_v6.cif`; that original ModelCIF file embeds the
  CC BY 4.0 link, DeepMind copyright and disclaimer, and the AlphaFold citation.
- **This package.** Code is MIT ([../LICENSE](../LICENSE)); the model weights and the
  calibrator are CC BY 4.0 ([../LICENSE-WEIGHTS](../LICENSE-WEIGHTS)). The published split
  id lists and the blocklist are lists of BMRB ids and residue indices, not BMRB content.
