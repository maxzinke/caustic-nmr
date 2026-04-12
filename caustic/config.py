"""Configuration dataclasses for the Caustic."""
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
    use_conditions: bool = False   # condition on sample pH
    use_film: bool = False         # AA-conditioned FiLM modulation in branch heads
    use_target_env: bool = True    # per-target-atom local environment features
    backbone: str = "schnet"       # "schnet" (scalar) or "painn" (equivariant)
    balance_branch_inputs: bool = False  # D13: LayerNorm on trunk/geo/env before concat
    geo_dim_override: int = 0            # D13: override geo_proj output dim (0 = H//4 default)
    env_dim_override: int = 0            # D13: override env_proj output dim (0 = H//8 default)


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
