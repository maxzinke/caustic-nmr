# Method

How CAUSTIC turns a protein structure into backbone chemical shifts with per-residue
uncertainties. Everything below is read from the code in this repository or from the
training configuration of record; file references are given so each statement can be
checked. Companion documents: [DATA.md](DATA.md) (training data and split),
[BENCHMARKS.md](BENCHMARKS.md) (accuracy), [LIMITATIONS.md](LIMITATIONS.md),
[../benchmarks/README.md](../benchmarks/README.md), [../CHANGELOG.md](../CHANGELOG.md).

## 1. Overview

CAUSTIC predicts the six backbone nuclei H, HA, N, CA, CB and C' for every residue of one
polymer chain. The pipeline is

1. structure file → atom graph (one graph per conformer);
2. PaiNN equivariant message passing over the atom graph;
3. per-residue read-out combining the target atom's embedding with hand-computed
   geometry, environment, temperature and exposure features;
4. per-nucleus heads producing a mean and a log-variance;
5. median aggregation across conformers (NMR ensembles);
6. a small post-hoc offset calibration.

The shipped model is an ONNX export (`caustic/data/best_v2_carbons.onnx`, sha256
`ebc7bbc2fc59327a50384105207958948cd90d7b5c7ea1ec2906b473e02948b2`) with
**741,024 trainable parameters** across 112 initializers, opset 17, exported from
PyTorch 2.8.0 (computed with `onnx` on the shipped file). The training checkpoint reports
741,030 entries in its state dict; the difference of 6 is the unused `_shift_mean` buffer
(6 values) that the exporter drops because the production model runs in residual mode
(`model.py:405-412`), where only `_aa_means` and `_shift_std` are needed. Both are present
in the ONNX (`model._aa_means` [21, 6]; `_shift_std` folded into a [1, 6] constant).

## 2. Input handling

**Formats.** PDB, mmCIF and their `.gz` variants, parsed with gemmi
(`graph.py:712-783`). Only one chain is predicted per call: the first polymer chain unless
`chain_id` / `--chain` is given (`graph.py:771-783`).

**Conformers.** `count_models` reports the number of models in the file. With
`use_ensemble=True` (default; CLI `--ensemble median`) up to `max_conformers=20` models are
converted to graphs and run independently (`inference.py:213-248`).

**Residue names.** The 20 canonical amino acids map to indices 0–19; 30 non-standard
names map to their canonical parent (`features.py:298-335`): HSE/HSD/HSP/HIE/HID/HIP→HIS,
GLH→GLU, ASH→ASP, CYX→CYS, MSE→MET, HYP→PRO, SEP→SER, TPO→THR, PTR→TYR,
CSO/CSD/OCS/CME→CYS, PCA→GLN, KCX/MLZ/MLY/M3L/LLP/ALY/PYL→LYS, FME→MET, SEC→CYS. Anything
else becomes the UNK class (index 20).

**Missing hydrogens.** When a structure has no backbone H/HA (X-ray, AlphaFold), they are
synthesised geometrically (`GraphConfig.missing_hydrogens="geometric"`, `config.py:22`):
the amide H is placed 0.99 Å from N along the bisector opposite N–CA and N–C(i−1); HA is
placed 1.09 Å from CA opposite the sum of the N, C and CB unit vectors; PRO gets no amide H
(`graph.py:100-135`).

**AlphaFold models.** Detected by filename (`AF-*-F*-model_v*`, `af_*`, `alphafold_*`) or
by a single-model file whose CA B-factors all lie in [0, 100] with median > 50
(`graph.py:243-275`). For such files the B-factor column is treated as pLDDT and inverted
to a pseudo-B-factor `100 − pLDDT` before it enters the node features (`graph.py:870-873`).
Nothing else changes — see [LIMITATIONS.md](LIMITATIONS.md) §4.

**Temperature.** The model has a temperature input `(T − 298) / 20` (training repository,
`dataset.py:1366-1378`). At inference the package does not read a temperature, so the
feature is zero, i.e. every prediction is made for **298 K** (`export.py:553-558`).

**Secondary structure (DSSP).** Feature [30] of the residue-geometry block is a DSSP
3-state label (helix +1 / sheet −1 / coil 0) that was **pre-stored on the training
graphs** at build time (`features.py:84-85`, `graph.py:1763-1768`). The package's graph
builder cannot run DSSP, so at inference this feature is **zero for every residue** —
the network sees "coil" everywhere. The measured effect of the whole
public-vs-production-graph input difference is < 2 % composite MAE, in the public
path's favour ([BENCHMARKS.md](BENCHMARKS.md) §7); see
[LIMITATIONS.md](LIMITATIONS.md) §8.

## 3. Graph construction

Nodes are atoms (heavy atoms plus hydrogens when present or synthesised) and, in addition,
one virtual node per aromatic ring (PHE, TYR, TRP ×2, HIS) at the ring centroid with its
normal vector (`features.py:365-376`, `GraphConfig.include_ring_nodes=True`). Each node
carries an element class (C, N, O, S, H, RING, UNK), an amino-acid class, an atom role
(backbone N/CA/C/O, CB, backbone H/HA, side-chain heavy, side-chain H, ring, ligand,
metal, water — `features.py:339-361`), and its B-factor.

Edges connect atoms within **8.0 Å**, capped at **32 neighbours** per atom
(`GraphConfig.cutoff`, `max_neighbors`). Edge features are 20 Gaussian radial basis
functions evenly spaced on [0, 8 Å] with width 0.4 Å (`features.py:rbf_expansion`), a
16-dimensional embedding of the sequence separation (capped at 32), and a same-residue flag.
Backbone hydrogen-bond edges carry an `is_hbond` flag and the donor/acceptor cosine
(`hb_cos`) as extra ONNX inputs (`graph.py:438`, `export.py:590-601`).

## 4. Network

**Node encoding** (`model.py:230-237`). Three 128-dimensional embeddings (element, amino
acid, role) and the scaled B-factor are concatenated and projected to 128 with SiLU.

**PaiNN trunk** (`model.py:74-148`, `model.py:250-263`). Four equivariant interaction
layers with hidden width 128. Each node carries a scalar channel s ∈ ℝ¹²⁸ and a vector
channel v ∈ ℝ¹²⁸ˣ³. The vector channel is initialised from the node's position relative to
its residue's CA through a learned linear map, and every layer:

- computes a filter `W = filter_net(edge_feat) · envelope(d)` with a cosine cutoff
  envelope that goes to zero at 8 Å, split into scalar, vector-scale and vector-filter
  parts;
- aggregates scalar messages `W_s · s_src` and vector messages `W_v · v_src`;
- updates `s ← LayerNorm(s + MLP([agg_s, ‖agg_v‖]))` and
  `v ← LayerNorm(v + gate(s) · agg_v)`.

After the last layer the scalar channel is augmented with the per-channel vector norm,
`s ← s + ‖v‖` (`model.py:262`), and only s is used downstream. Dropout 0.3 is applied in
every interaction and head (`ModelConfig.dropout`).

**Read-out features** (`model.py:264-355`). For each residue and nucleus the input to the
head is the concatenation of

| Block | Width | Source |
|---|---|---|
| Target atom embedding | 128 | trunk output at the target atom (H, HA, N, CA, CB or C) |
| Residue geometry | 52 → 32 | φ/ψ/ω/χ1 sin/cos, disulfide flags and S–S geometry, ensemble circular variances, Haigh–Mallion ring-current terms per nucleus, H-bond distances and angles, rSASA and half-sphere exposure, Buckingham electric field along N–H and CA–HA, DSSP 3-state, deuteration fractions, nearest-aromatic χ1/χ2, chain-aware terminal flags, H-bond partner directions (`features.py:60-114`) |
| Target environment | 15 → 16 | shell-resolved counts of backbone, hydrophobic C, polar O/N, S and aromatic atoms within 0–4 and 4–8 Å, nearest ring centroid distance and orientation, nearest disulfide S (`features.py:120-147`) |
| Temperature | 1 → 8 | `(T − 298)/20` |
| Per-target-atom rSASA | 1 → 8 | Shrake–Rupley exposure of the target atom itself (H/HA inherit N/CA) (`config.py:138-146`) |

**Branches and heads.** Protons (H, HA) and heavy nuclei (N, CA, CB, C) go through
separate two-layer MLPs of width 128, then a per-nucleus head (128 → 64 → 2) yielding a
z-scored mean and a log-variance (`model.py:152-186`). The model runs in *residual mode*:
`shift = aa_mean[aa, nucleus] + z · shift_std[nucleus]`, where `aa_mean` is a 21 × 6
table of amino-acid-type means fitted on the training set and `shift_std` =
(0.64, 0.50, 5.17, 4.84, 12.72, 2.19) ppm for (H, HA, N, CA, CB, C) (`model.py:290-296`,
`405-412`). The reported σ is `exp(0.5 · logvar)`.

Note that `caustic/model.py` in this repository does not define the temperature and
per-atom-rSASA projections (`temp_proj`, `per_atom_rsasa_proj`) that the shipped ONNX
contains; they are exported by `export.py:306-316` from the training-side model. The
PyTorch path (`predict_shifts` with a `.pt` checkpoint) therefore cannot load the
production checkpoint; use the ONNX path.

## 5. Ensemble aggregation

For multi-model files each conformer is predicted separately; the mean shift is the
**median** across conformers and the log-variance is the mean across conformers
(`inference.py:255-264`). Positions where every conformer is NaN (e.g. PRO amide H) stay
NaN. The PyTorch path uses the same median but derives σ from the median absolute deviation
across conformers (`ensemble.py:72-76`).

## 6. Post-hoc calibration

`predict_shifts_onnx(..., apply_calibrator=True)` (default) applies the "slim SA16 v2"
calibrator from `caustic/data/sa16_calibrator_v2.json` (`calibrate.py`):

1. a global per-nucleus offset: C −0.0117, CA +0.0247, CB −0.0050, H +0.0026,
   HA +0.0077, N +0.0296 ppm;
2. a CB modifier for cysteines: +1.361 ppm if the residue's Sγ is within 2.5 Å of another
   cysteine's Sγ (disulfide), otherwise +0.319 ppm (treated as reduced/free). The
   calibrator also holds a metal-bound class (+0.065 ppm) that the package does not detect.

The calibrator of record additionally contains per-(DSSP × aromatic × rSASA) stratum
offsets that are not shipped; see [LIMITATIONS.md](LIMITATIONS.md) §6 and
[BENCHMARKS.md](BENCHMARKS.md) for the measured effect.

## 7. Training

Training code lives outside this repository (see D3 in the release plan); this section
records what was run. Configuration of record
(`configs/shift_predictor/active/shift_predictor_d31_clean_safe_v2_carbons.yaml` in the
training repository, quoted verbatim below with one path corrected):

```yaml
graph:
  cutoff: 8.0
  max_neighbors: 32
  include_hydrogens: true
  num_rbf: 20
  max_seq_sep: 32

model:
  hidden_dim: 128
  num_layers: 4
  num_rbf: 20
  dropout: 0.3
  predict_uncertainty: true
  use_geometry: true
  use_branches: true
  use_residual: true
  use_conditions: false
  use_target_env: true
  use_solchem: false
  use_temperature: true
  backbone: painn
  use_per_atom_rsasa: true
  per_atom_rsasa_dim: 8

ensemble:
  aggregation: median
  conformer_dropout: 0.2
  min_conformers: 2
  max_conformers: 20

data:
  splits_filename: "splits.connected_components.json"
  geometry_labels: ["NG", "XG"]
  min_backbone_completeness: 0.5
  max_shift: {H: 15.0, HA: 15.0, N: 200.0, CA: 100.0, CB: 100.0, C: 200.0}
  min_shift: {H: -2.0, HA: -2.0, N: 80.0, CA: 30.0, CB: 0.0, C: 150.0}
  # original value was a cache path (cache/caustic/...); the tracked file is:
  label_blocklist_path: "results/caustic_label_blocklists/label_blocklist_v2_sweep_carbons.json"
  disulfide_sidecar_path: "results/shift_predictor_disulfide_sidecar.json"
  temperature_filter_min: 273.0
  temperature_filter_max: 320.0

training:
  lr: 5.0e-4
  weight_decay: 5.0e-3
  batch_size: 4
  grad_accumulation: 4
  max_epochs: 100
  warmup_epochs: 5
  scheduler: cosine
  min_lr: 1.0e-6
  center_loss_weight: 0.7
  uncertainty_loss_weight: 0.2
  consistency_loss_weight: 0.1
  center_loss_type: huber
  huber_delta: 1.0
  uncertainty_loss_type: student_t
  student_t_dfs: {H: 6.5, HA: 5.4, N: 6.7, CA: 6.6, CB: 6.1, C: 4.6}
  nucleus_weights: {H: 1.0, HA: 1.0, N: 1.0, CA: 1.5, CB: 2.0, C: 1.0}
  phase: single
  mixed_precision: true
  gradient_clip: 1.0
  patience: 15
  seed: 42
  ensemble_sampling: true
  ensemble_p_medoid: 0.2
```

**Loss** (training repository `shift_predictor/losses.py:87-215`). Per nucleus *n* with
weight *wₙ*:

```
L_n = 0.7 · Huber_δ=1(z_pred − z_true)                     (z-scored by shift_mean/shift_std)
    + 0.2 · [ ½·logvar + log_norm + ½(ν+1)·log(1 + (y−μ)²/(ν·σ²)) ]   (Student-t NLL, ν per nucleus)
L   = Σ_n w_n · L_n,   w = (1, 1, 1, 1.5, 2.0, 1) for (H, HA, N, CA, CB, C)
```

The `consistency_loss_weight: 0.1` term is defined in the configuration; the loss module
quoted above contains a "centered-pattern" term whose weight is 0 in this run, and no
separate consistency term was located in `losses.py` (not verified where, or whether, the
0.1 weight is consumed).

**Optimisation.** AdamW (`training.py:272`), lr 5 × 10⁻⁴, weight decay 5 × 10⁻³,
effective batch 16 graphs (4 × 4 accumulation), 5 warm-up epochs then cosine annealing to
10⁻⁶, mixed precision, gradient clipping at 1.0, early stopping with patience 15. Each
training graph is one conformer; with `ensemble_sampling: true` the loader substitutes a
random alternative conformer from the entry's NMR ensemble (medoid with probability 0.2)
so that the same labels are seen from different conformers (`dataset.py:1105-1130`,
`1448`). Conformer dropout 0.2 applies to the median aggregator during training.

**Run of record** (checkpoint metadata, from the survey of the training checkpoint; not
re-loaded here): best epoch 58, validation loss 0.30744, early stop at epoch 73, seed 42,
training-repository commit `8dbd5b6a`, feature pipeline version 52. A single seed was
trained (see [LIMITATIONS.md](LIMITATIONS.md) §2).

## 8. Export

`export.py` wraps the trained model in an ONNX-traceable module with 19 named inputs
(`element, amino_acid, atom_role, bfactor, edge_index, edge_dist, seq_sep, same_residue,
target_indices, target_mask, residue_types, residue_geometry[R,52],
target_environment[R,6,15], temperature_feature[R,1], target_atom_rsasa[R,6], pos[N,3],
node_normal[N,3], is_hbond[E], hb_cos[E]`) and two outputs (`pred_mean`, `pred_logvar`,
each [R, 6]). Agreement between the PyTorch and ONNX forward passes was checked at export
time on ten test proteins (max abs diff 1.5 × 10⁻⁵ ppm, from the export report in the
training repository; not re-run here).
