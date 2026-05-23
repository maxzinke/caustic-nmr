"""Configuration dataclasses for the shift predictor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GraphConfig:
    """PDB → graph construction parameters."""

    cutoff: float = 8.0  # spatial edge cutoff (Angstroms)
    max_neighbors: int = 32  # max edges per node (0 = unlimited)
    include_hydrogens: bool = True  # include H atoms from the structure file
    num_rbf: int = 20  # radial basis function dimension
    max_seq_sep: int = 32  # cap for sequence separation feature
    # Step 4 inference hardening --------------------------------------------
    # How to handle structures that have no backbone H / HA atoms (typical
    # for X-ray PDBs and AlphaFold models). "geometric" synthesises H and HA
    # positions from N/CA/C/CB backbone geometry so the model can still
    # predict those shifts; "skip" preserves legacy behaviour and returns
    # NaN for missing hydrogens; "strict" raises if any are absent.
    missing_hydrogens: Literal["geometric", "skip", "strict"] = "geometric"
    # Emit a WARNING log when the chain exceeds this many residues. Not a
    # hard cap — just a heads-up so the caller can opt into chunking.
    large_protein_warn_threshold: int = 500
    # Minimum residues in the chain. <3 is too small for meaningful
    # message passing; raise a clear error instead of crashing downstream.
    min_residues: int = 1
    # d48: include aromatic ring virtual nodes (centroids of PHE/TYR/TRP/HIS
    # rings) appended to the per-atom arrays before cKDTree edge build.
    # d47-era PaiNN trains on True (default). MACE configs set False so
    # structure_to_graph produces real-only graphs at training and inference,
    # avoiding the adapter strip-and-rebuild path.
    include_ring_nodes: bool = True
    # d49: include non-polymer heavy atoms that sit near the extracted
    # polymer chain as real graph nodes before aromatic ring virtual nodes.
    # These nodes make metals, cofactors, ligands, and ordered solvent visible
    # to message passing and to the d49 chemistry feature builders.
    include_hetero_nodes: bool = False
    hetero_radius: float = 8.0
    # d50: collect all nearby hetero atoms as compact summary context even
    # when only a capped subset, or no subset, is promoted to graph nodes.
    collect_hetero_context: bool = False
    hetero_summary_radius: float = 8.0
    include_water_nodes: bool = False
    max_hetero_nodes: int = 128  # 0 = unlimited
    max_hetero_nodes_per_residue: int = 4  # 0 = unlimited


@dataclass
class ModelConfig:
    """GNN model architecture parameters."""

    hidden_dim: int = 256
    num_layers: int = 6
    num_rbf: int = 20  # must match GraphConfig.num_rbf
    dropout: float = 0.1
    predict_uncertainty: bool = True
    use_geometry: bool = True      # concatenate backbone torsion features at prediction
    use_branches: bool = True      # separate proton vs carbon/nitrogen branch MLPs
    use_residual: bool = False     # predict residual from AA-type mean instead of absolute shift
    use_conditions: bool = False   # legacy shallow pH-only path (overlaps with use_solchem — leave False when use_solchem=True)
    use_film: bool = False         # AA-conditioned FiLM modulation in branch heads
    use_target_env: bool = True    # per-target-atom local environment features
    use_solchem: bool = False      # D15: solution-chemistry features (pH, ionic strength, titratable-neighbor context)
    use_potenci_baseline: bool = False  # D16: use data.potenci_rc as per-residue prediction baseline instead of aa_mean. Implicitly absorbs random-coil pH/temperature/ionic-strength effects at the target level. Requires data.potenci_rc on the graph; falls back to aa_mean where RC is NaN.
    use_temperature: bool = False  # D17: feed sample temperature (T-298)/20 through a small MLP and concat into head input. Uses data.sample_temperature directly.
    backbone: str = "schnet"       # "schnet" (scalar) | "painn" (equivariant) | "mace" (equivariant 3-body / 4-body, d48)
    balance_branch_inputs: bool = False  # D13: LayerNorm on trunk/geo/env before concat
    geo_dim_override: int = 0            # D13: override geo_proj output dim (0 = H//4 default)
    env_dim_override: int = 0            # D13: override env_proj output dim (0 = H//8 default)
    # D26a: higher-order v-channel invariants at the prediction head.
    # PaiNN already reads ||v_c|| (per-channel norm) into the scalar channel
    # inside every interaction layer. What's NOT already captured is the
    # cross-channel Gram matrix <v_a, v_b> -- pairwise dot products between
    # different hidden vector channels. These are rotation-invariant and carry
    # novel 2-body directional information the scalar-only head currently
    # discards. Implemented as a low-rank Gram: v -> v_proj=[N,K,3] via learned
    # K-dim mixer, gram = v_proj . v_proj -> [N,K,K], upper-triangular flattened
    # through a small MLP to v_invariants_dim features appended at the head.
    use_v_invariants: bool = False       # D26a: enable higher-order v-channel invariants
    v_invariants_rank: int = 16          # K dim of the low-rank v projection (K*(K+1)/2 unique gram entries)
    v_invariants_dim: int = 32           # output feature dim appended to head input
    # D27: per-target-atom rSASA. Each of the 6 target nuclei per residue
    # gets its own atom-sphere-normalized exposure (n_exposed /
    # n_sphere_points from Shrake-Rupley) rather than inheriting the
    # per-residue aggregate from ``residue_geometry[:, 25]``. Stored on
    # the graph as ``data.target_atom_rsasa`` of shape [R, 6] and
    # projected into the head input via a small MLP. Hydrogen targets
    # (H, HA) inherit from their parent heavy atom (N, CA respectively).
    use_per_atom_rsasa: bool = False
    per_atom_rsasa_dim: int = 8          # output feature dim of the projection MLP
    # D36: backbone-frame projection of the final PaiNN v channel. Per target
    # atom, project v in[N_target, H, 3] onto the residue's orthonormal
    # backbone frame (e1=N→Cα, e2=Cα→C′ perpendicularised, e3=e1×e2) to get
    # a [H, 3] tensor in the residue-local frame. Flatten to 3H and bottleneck
    # through an MLP. Physically principled because NMR chemical-shift tensors
    # are defined in the molecular frame; the raw equivariant v is in the lab
    # frame and the current read-out only uses its norm (direction discarded).
    use_frame_projection: bool = False
    frame_projection_dim: int = 32
    # D37: per-target-atom frames. Each of the 6 target nuclei gets its own
    # local frame anchored on its bonded neighbours (e.g., HN frame uses
    # N→H, N frame uses Cα→N→prev_C, C′ frame uses Cα→C→next_N).
    # Hypothesis: shifts depend on the target's local electronic
    # environment, not the residue's average backbone frame.
    # Boundary residues (first/last for N/C′ frames, Gly for CB frame) get
    # NaN-gated to zero contribution.
    use_per_target_frame: bool = False
    # D38: multi-layer v projection. Concat v from EVERY PaiNN interaction
    # layer (1, 2, 3, 4) and project the multi-scale stack onto the
    # residue frame. Final-only v captures long-range neighborhood; earlier
    # layers carry shorter-range directional info. Multi-scale directional
    # is the natural extension of D36's frame-projection win. Frame
    # projection input grows from 3*H to 3*(num_layers*H).
    use_multi_layer_v: bool = False
    # D42: H-bond frame projection. Add a SECOND frame defined by the
    # backbone amide H-bond partner geometry, project v onto it (alongside
    # the residue frame), expose partner-orientation info to the head.
    # Hits sheet CB (cross-strand H-bond context), amide H/N (H-bond
    # geometry directly), terminal C' (fallback directional axis).
    # Requires data.hb_nh_partner_dir and data.hb_co_partner_dir of shape
    # [R, 3] each — populated at graph build time. Atoms without an
    # H-bond partner get a zero direction; the validity bit gates the
    # extra projection to zero contribution.
    use_hb_frame: bool = False
    # D43: CYS-specific CB sub-head. Branches the CB prediction by AA:
    # non-CYS residues use the regular CB head; CYS residues route through
    # a dedicated sub-head that takes branch_out + 4 S-S partner geometry
    # features (residue_geometry slots [42-45]) as input. Targets the
    # 3.4× CYS SS CB ratio that survived D31 -> D38.
    use_cys_cb_subhead: bool = False
    # D44: per-target-atom packing density. For each of the 6 target
    # nuclei, count atoms within 4 A by chemical class (C/N/O/S/aromatic).
    # 5 features per target nucleus, populated at graph build time as
    # data.target_atom_packing of shape [R, 6, 5]. Targets buried CB
    # (Q0 still 1.6× exposed in D38). Cache rebuild required.
    use_per_atom_packing: bool = False
    per_atom_packing_dim: int = 8
    # D40: tensor (outer-product) features. Currently FrameProjection reads
    # v in the residue frame as 3 scalar projections per channel:
    # (<v,e1>, <v,e2>, <v,e3>). These are the LINEAR components of v in
    # frame coords. The MLP at depth 1 can mix them linearly but can't form
    # pairwise products. D40 augments with the 6 unique outer-product
    # components per channel:
    #     <v,e1>², <v,e2>², <v,e3>², <v,e1>·<v,e2>, <v,e1>·<v,e3>, <v,e2>·<v,e3>
    # These are the components of v⊗v in the residue frame — per-channel
    # rank-2 tensor info. Diagonals are CSA-tensor-like; off-diagonals
    # encode direction tilts the linear path misses. All 6 are rotation-
    # invariant scalars (frame rotates with molecule).
    # Flag REQUIRES use_frame_projection=True. Input grows 3*C -> 9*C.
    use_tensor_components: bool = False
    # D36: NH order parameter S²_NH = |⟨n̂_NH⟩|² across NMR ensemble. Direct
    # signal for conformational averaging at the residue level — chemical
    # shifts average over ps-ns motion that a static PDB snapshot can't see.
    # NaN sentinel for single-structure entries; projection gate uses a
    # validity bit so missing S² contributes zero.
    use_s2_nh: bool = False
    s2_nh_dim: int = 4
    # D49: optional chemistry tensors. Kept separate from existing
    # residue_geometry/target_environment so older checkpoints remain
    # loadable with their historical projection widths.
    use_edge_chemistry: bool = False
    use_residue_chemistry: bool = False
    residue_chemistry_dim: int = 16
    use_target_chemistry: bool = False
    target_chemistry_dim: int = 16
    use_nc_contact_chemistry: bool = False
    nc_contact_chemistry_dim: int = 8

    # ---------------------------------------------------------------------
    # D48: MACE trunk parameters (only used when backbone == "mace").
    # MACE's EquivariantProductBasisBlock computes 3-body and 4-body
    # equivariant features per layer — replaces the d47-era hand-engineered
    # ring-current / H-bond-geometry / electric-field features that PaiNN's
    # 2-body trunk couldn't extract. See plan
    # ~/.claude/plans/make-the-big-mace-nifty-sunrise.md for the full
    # rationale and gate criteria.
    # ---------------------------------------------------------------------
    mace_max_ell: int = 3
    mace_num_bessel: int = 8
    mace_num_polynomial_cutoff: int = 5
    mace_hidden_irreps: str = "128x0e + 128x1o"
    mace_correlation: int = 3
    mace_r_max: float = 8.0
    mace_avg_num_neighbors: float = 26.68  # measured on d48_clean train split, 3431 graphs
    mace_radial_MLP: list[int] | None = None


@dataclass
class EnsembleConfig:
    """Conformer ensemble aggregation parameters."""

    aggregation: Literal["median", "mean", "attention"] = "median"
    conformer_dropout: float = 0.2  # drop fraction during training
    min_conformers: int = 2  # keep at least this many
    max_conformers: int = 20  # cap for memory


@dataclass
class DataConfig:
    """Dataset paths and filtering."""

    cache_dir: str = "~/.crystalline_fid/crystalline_cache"
    bmrb_cache_dir: str = "~/.crystalline_fid/bmrb_cache"
    pdb_cache_dir: str = "~/.crystalline_fid/pdb_cache"
    graph_cache_dir: str = "~/.crystalline_fid/shift_predictor_graphs"
    # Caustic Rescue Plan B1: filename of the train/val/test partition under
    # graph_cache_dir. Default is the legacy 'splits.json' (linkage='complete',
    # leaky). The leak-safe alternative written by
    # scripts/audit_caustic_split_leakage.py --write-split is
    # 'splits.connected_components.json'.
    splits_filename: str = "splits.json"
    # Bulk BMRB metadata caches (from scripts/plot_sample_conditions.py etc.)
    conditions_cache: str = "scripts/bmrb_sample_conditions.json"
    components_cache: str = "scripts/bmrb_sample_components.json"
    # Quality filtering
    geometry_labels: tuple[str, ...] = ("NG", "XG")
    min_backbone_completeness: float = 0.5
    # Target formulation
    use_potenci_targets: bool = False  # POTENCI secondary shifts instead of AA-residual
    # Shift outlier filtering (ppm)
    max_shift: dict[str, float] = field(default_factory=lambda: {
        "H": 15.0, "HA": 15.0, "N": 200.0, "CA": 100.0, "CB": 100.0, "C": 200.0,
    })
    min_shift: dict[str, float] = field(default_factory=lambda: {
        "H": -2.0, "HA": -2.0, "N": 80.0, "CA": 30.0, "CB": 0.0, "C": 150.0,
    })
    # D17: label blocklist produced by scripts/audit_training_labels.py.
    # JSON with shape {"drop_entries": [...], "blocklist": {eid: {ri: [nuc, ...]}}}.
    # Empty string = blocklist disabled.
    label_blocklist_path: str = ""
    # D17: metal-aware disulfide override produced by
    # scripts/audit_disulfide_metals.py. JSON with shape
    # {"overrides": {eid: {ri: bool}}}. Empty string = disabled.
    disulfide_sidecar_path: str = ""
    # D17: drop training entries whose recorded temperature is outside the
    # physical range [273, 320] K — catches BMRB metadata typos (e.g. 25K
    # instead of 25 C, -20 C frozen, etc.) and proteins run under extreme
    # cryo conditions the model has no way of representing.
    temperature_filter_min: float = 273.0
    temperature_filter_max: float = 320.0
    # Optional runtime guard: drop extreme graph-size outliers from
    # ShiftDataset construction. 0 disables each cap.
    max_graph_atoms: int = 0
    max_graph_edges: int = 0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    grad_accumulation: int = 4
    max_epochs: int = 100
    warmup_epochs: int = 5
    scheduler: Literal["cosine", "plateau"] = "cosine"
    min_lr: float = 1e-6
    # Loss
    center_loss_weight: float = 0.7
    uncertainty_loss_weight: float = 0.2
    consistency_loss_weight: float = 0.1
    center_loss_type: Literal["huber", "l1", "mse"] = "huber"
    huber_delta: float = 1.0
    # D51: train the within-entry/per-nucleus chemical-shift pattern rather
    # than the absolute reference position. The trainer subtracts each
    # graph+nucleus mean from predictions and labels before applying this
    # loss. A small absolute center/NLL loss can be kept as an auxiliary so
    # predictions remain roughly located before anchor-offset placement.
    centered_pattern_loss_weight: float = 0.0
    centered_pattern_loss_type: Literal["huber", "l1", "mse"] = "huber"
    centered_pattern_huber_delta: float = 1.0
    uncertainty_loss_type: Literal["gaussian", "student_t"] = "gaussian"
    # Per-nucleus Student-t degrees of freedom (fixed, from M3 calibration on
    # D24-ens3). Only used when uncertainty_loss_type == "student_t".
    student_t_dfs: dict[str, float] = field(default_factory=lambda: {
        "H": 6.5, "HA": 5.4, "N": 6.7, "CA": 6.6, "CB": 6.1, "C": 4.6,
    })
    nucleus_weights: dict[str, float] = field(default_factory=lambda: {
        "H": 1.0, "HA": 1.0, "N": 0.5, "CA": 0.5, "CB": 0.5, "C": 0.5,
    })
    # Step 1.5: per-sample subgroup weights (1.0 = disabled)
    subgroup_weights: dict = field(default_factory=lambda: {
        "pro_n": 1.0,
        "cys_ss_cb_geom": 1.0,
        "bb_helix_ca": 1.0,
        "frequency_balance": False,
    })
    # Training phase
    phase: Literal["single", "ensemble", "pseudo_ensemble", "calibration"] = "single"
    # Infrastructure
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/shift_predictor"
    resume: bool = False  # resume from latest checkpoint
    warm_start: str = ""  # load model weights from this path, fresh optimizer
    log_every: int = 50
    eval_every_epoch: int = 1
    patience: int = 15
    seed: int = 42
    # Parallel backfill workers for ShiftDataset construction. 0 or 1
    # preserves the legacy serial loop; higher values spawn a
    # ProcessPoolExecutor over the cold-cache Shrake-Rupley pass (the
    # dominant first-epoch cost on a fresh graph cache). Sweet spot is
    # typically ``min(cpu_count - 1, 8)``.
    graph_build_workers: int | None = None
    backfill_workers: int = 0
    # Persist derived feature backfills into graph .pt files during dataset
    # construction. Disable for read-only/shared graph caches.
    persist_backfill: bool = True
    # Caustic Rescue Plan §10 Candidate A: stochastic NMR-conformer sampling.
    # When True, ShiftDataset.__getitem__ randomly substitutes the medoid's
    # pos / edge_index / edge_dist with one of the alternative-conformer
    # views from a sister <eid>.alt.pt file (built by
    # scripts/build_alt_conformer_pos.py). Same shift label across
    # different conformers → model implicitly learns conformer-invariant
    # prediction = population mean. Off by default for backwards compat.
    ensemble_sampling: bool = False
    ensemble_p_medoid: float = 0.2
    # Per-entry referencing offset: a learnable 6-vector b_entry per training
    # BMRB entry, added to pred_mean before the loss. Absorbs systematic
    # per-entry chemical-shift referencing drift (15N ~0.5 ppm scale,
    # carbons ~0.1-0.3 ppm, protons smaller) so the model learns
    # structure->shift physics instead of bookkeeping. Offset=0 for novel
    # proteins at inference (sentinel entry_offset_idx=-1). Off by default.
    referencing_offset_enabled: bool = False
    referencing_offset_l2: float = 1e-3
