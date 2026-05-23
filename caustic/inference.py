"""Inference API: predict backbone chemical shifts from structure."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from caustic.config import GraphConfig, ModelConfig
from caustic.features import BACKBONE_NUCLEI
from caustic.graph import structure_to_graph, compute_residue_geometry, compute_target_environment, count_models

logger = logging.getLogger(__name__)


@dataclass
class ShiftPrediction:
    """Per-residue backbone shift predictions for one protein."""

    seq_ids: list[int]
    residue_names: list[str]
    # Per-nucleus predictions: {nucleus: [N_residues] array or None}
    mean: dict[str, np.ndarray | None] = field(default_factory=dict)
    std: dict[str, np.ndarray | None] = field(default_factory=dict)
    # Source info
    pdb_path: str = ""
    num_conformers: int = 1


# 20-letter canonical amino acid order used by the graph builder's
# residue_types tensor (indices 0..19). Position 20 is the UNK sentinel.
_CANONICAL_AA_THREE: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)


def _residue_types_to_three_letter(residue_types: Any) -> list[str]:
    """Convert a ``residue_types`` tensor back into 3-letter names.

    Out-of-range or UNK indices map to ``"UNK"``.
    """
    out: list[str] = []
    try:
        arr = residue_types.tolist()
    except AttributeError:
        arr = list(residue_types)
    for idx in arr:
        i = int(idx)
        if 0 <= i < len(_CANONICAL_AA_THREE):
            out.append(_CANONICAL_AA_THREE[i])
        else:
            out.append("UNK")
    return out


def predict_shifts(
    structure_path: str | Path,
    checkpoint_path: str | Path,
    model_config: ModelConfig | None = None,
    graph_config: GraphConfig | None = None,
    chain_id: str | None = None,
    use_ensemble: bool = True,
    max_conformers: int = 20,
    device: str | None = None,
) -> ShiftPrediction:
    """Predict backbone chemical shifts from a PDB/mmCIF structure.

    Args:
        structure_path: Path to .pdb or .cif file.
        checkpoint_path: Path to model checkpoint (.pt).
        model_config: Model architecture config (must match checkpoint).
        graph_config: Graph construction parameters.
        chain_id: Chain to predict (None = first polymer).
        use_ensemble: If True and structure has multiple models, average.
        max_conformers: Cap on number of conformers to use.
        device: "cuda", "cpu", or None (auto).

    Returns:
        ShiftPrediction with per-residue mean and std arrays.
    """
    import torch
    from caustic.model import ShiftPredictor
    from caustic.ensemble import EnsembleAggregator
    from caustic.config import EnsembleConfig

    if model_config is None:
        model_config = ModelConfig()
    if graph_config is None:
        graph_config = GraphConfig()

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load model
    model = ShiftPredictor(model_config).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Determine conformers
    n_models = count_models(structure_path)
    if use_ensemble and n_models > 1:
        n_use = min(n_models, max_conformers)
        model_indices = list(range(n_use))
    else:
        model_indices = [0]

    # Build graphs for each conformer
    graphs = []
    for mi in model_indices:
        try:
            g = structure_to_graph(
                structure_path, chain_id=chain_id, model_idx=mi, config=graph_config,
            )
            if not hasattr(g, "residue_geometry") or g.residue_geometry is None:
                g.residue_geometry = compute_residue_geometry(g)
            if getattr(model_config, "use_target_env", False):
                if not hasattr(g, "target_environment") or g.target_environment is None:
                    g.target_environment = compute_target_environment(g)
            graphs.append(g)
        except Exception as e:
            logger.warning("Failed model %d: %s", mi, e)

    if not graphs:
        raise ValueError(f"Could not build any graphs from {structure_path}")

    # Reference graph for residue info
    ref = graphs[0]
    num_res = int(ref.num_residues)
    seq_ids = ref.seq_ids.tolist()
    residue_names = _residue_types_to_three_letter(ref.residue_types)

    # Run inference
    conformer_means: list[dict[str, torch.Tensor]] = []
    conformer_logvars: list[dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(dev)
            pm, plv = model(g)
            conformer_means.append(pm)
            conformer_logvars.append(plv)

    # Aggregate
    if len(graphs) > 1:
        agg = EnsembleAggregator(EnsembleConfig(aggregation="median"))
        agg_mean, agg_logvar = agg(conformer_means, conformer_logvars)
    else:
        agg_mean = conformer_means[0]
        agg_logvar = conformer_logvars[0]

    # Unpack into full-length arrays (NaN for missing)
    result = ShiftPrediction(
        seq_ids=seq_ids,
        residue_names=residue_names,
        pdb_path=str(structure_path),
        num_conformers=len(graphs),
    )

    for nuc_idx, nuc in enumerate(BACKBONE_NUCLEI):
        mean_arr = np.full(num_res, np.nan, dtype=np.float32)
        std_arr = np.full(num_res, np.nan, dtype=np.float32)

        if nuc in agg_mean and agg_mean[nuc].numel() > 0:
            # The mask from the reference graph tells us which residues have target atoms
            mask = ref.target_mask[:, nuc_idx].cpu().numpy()
            valid_idx = np.where(mask)[0]
            vals = agg_mean[nuc].cpu().numpy()
            lvars = agg_logvar[nuc].cpu().numpy() if nuc in agg_logvar else np.zeros_like(vals)

            n = min(len(valid_idx), len(vals))
            mean_arr[valid_idx[:n]] = vals[:n]
            std_arr[valid_idx[:n]] = np.exp(0.5 * lvars[:n])

        result.mean[nuc] = mean_arr
        result.std[nuc] = std_arr

    return result


def predict_shifts_onnx(
    structure_path: str | Path,
    onnx_path: str | Path,
    *,
    model_config: ModelConfig | None = None,
    graph_config: GraphConfig | None = None,
    chain_id: str | None = None,
    use_ensemble: bool = True,
    max_conformers: int = 20,
    session: Any | None = None,
    apply_calibrator: bool = True,
) -> ShiftPrediction:
    """Predict backbone shifts via a pre-exported ONNX model.

    Drop-in replacement for :func:`predict_shifts` that runs inference
    through ``onnxruntime`` instead of PyTorch. The returned
    :class:`ShiftPrediction` is bit-identical to the PyTorch path's output
    (within numerical tolerance) so downstream callers — PRISM, the CLI,
    the HF Space — don't need to branch on the backend.

    PyTorch is still required for graph construction right now (the
    ``structure_to_graph`` path builds torch tensors). Step 2.5 will lift
    that to a pure-numpy graph builder so the full inference stack can
    run without importing torch at all.

    Parameters
    ----------
    structure_path: ``.pdb`` / ``.cif`` / ``.pdb.gz`` / ``.cif.gz`` file.
    onnx_path: path to a ``.onnx`` produced by
        :func:`caustic.export.export_to_onnx`.
    model_config: optional ``ModelConfig`` used when computing the
        per-target-atom environment tensor. When ``None`` the default is
        used (matches D8's ``use_target_env=True``).
    graph_config: optional ``GraphConfig`` forwarded to
        ``structure_to_graph``.
    chain_id: chain to predict. ``None`` picks the first polymer chain.
    use_ensemble: if the structure has multiple models (NMR ensemble),
        run each conformer through ONNX and median-aggregate.
    max_conformers: cap on the number of conformers used.
    session: optional pre-opened ``onnxruntime.InferenceSession``. When
        the caller predicts many structures, reuse a single session rather
        than re-loading the ``.onnx`` each call.
    apply_calibrator: when True (default), apply the slim SA16 post-prediction
        calibration after ONNX inference: global per-nucleus offsets +
        CYS-CB disulfide modifier. Disulfide bonds detected via Sγ-Sγ
        distance gate. Set False to get raw ONNX outputs.
    """
    # Local imports so onnxruntime / the export module stay optional at
    # package import time — callers that only use the PyTorch path never
    # pay the import cost.
    from caustic.export import load_onnx_session, run_onnx_inference

    if model_config is None:
        model_config = ModelConfig()
    if graph_config is None:
        graph_config = GraphConfig()

    if session is None:
        session = load_onnx_session(onnx_path)

    n_models = count_models(structure_path)
    if use_ensemble and n_models > 1:
        model_indices = list(range(min(n_models, max_conformers)))
    else:
        model_indices = [0]

    per_conf_mean: list[np.ndarray] = []
    per_conf_logvar: list[np.ndarray] = []
    ref = None
    for mi in model_indices:
        try:
            g = structure_to_graph(
                structure_path, chain_id=chain_id, model_idx=mi, config=graph_config,
            )
            if not hasattr(g, "residue_geometry") or g.residue_geometry is None:
                g.residue_geometry = compute_residue_geometry(g)
            if getattr(model_config, "use_target_env", False):
                if not hasattr(g, "target_environment") or g.target_environment is None:
                    g.target_environment = compute_target_environment(g)
            mean_np, logvar_np = run_onnx_inference(session, g)
            per_conf_mean.append(mean_np)
            per_conf_logvar.append(logvar_np)
            if ref is None:
                ref = g
        except Exception as e:
            logger.warning("ONNX inference failed on model %d: %s", mi, e)

    if not per_conf_mean or ref is None:
        raise ValueError(f"Could not run ONNX inference on any conformer of {structure_path}")

    # Aggregate across conformers — median on mean, mean on logvar (same
    # semantics as EnsembleAggregator(aggregation="median") in the torch path).
    # Suppress the "All-NaN slice" / "Mean of empty slice" RuntimeWarnings
    # that fire on positions where every conformer correctly reports NaN
    # (e.g. PRO H) — that's the desired semantics, not a bug.
    import warnings as _warnings

    stacked_mean = np.stack(per_conf_mean, axis=0)   # [K, R, 6]
    stacked_lv = np.stack(per_conf_logvar, axis=0)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=RuntimeWarning)
        agg_mean = np.nanmedian(stacked_mean, axis=0)    # [R, 6]
        agg_logvar = np.nanmean(stacked_lv, axis=0)       # [R, 6]

    num_res = int(ref.num_residues)
    seq_ids = ref.seq_ids.tolist()
    residue_names = _residue_types_to_three_letter(ref.residue_types)

    if apply_calibrator:
        # SA16 v2 slim: global per-nucleus offsets + CYS-CB disulfide
        # modifier (Sγ-Sγ distance gate). See caustic/calibrate.py.
        try:
            from caustic.calibrate import (
                apply_calibrator as _apply_cal,
                detect_disulfides,
                find_cys_sg_indices,
                load_calibrator,
            )
            calibrator = load_calibrator()
            sg_indices = find_cys_sg_indices(ref)
            ref_pos = ref.pos.detach().cpu().numpy() if hasattr(ref.pos, "detach") else np.asarray(ref.pos)
            disulfide_set = detect_disulfides(ref_pos, residue_names, sg_indices)
            agg_mean = _apply_cal(agg_mean, residue_names, disulfide_set, calibrator)
            logger.info(
                "Applied SA16 v2 slim calibration (%d disulfide CYS detected)",
                len(disulfide_set),
            )
        except Exception as e:
            logger.warning("Calibration failed (%s); returning uncalibrated predictions.", e)

    result = ShiftPrediction(
        seq_ids=seq_ids,
        residue_names=residue_names,
        pdb_path=str(structure_path),
        num_conformers=len(per_conf_mean),
    )

    for nuc_idx, nuc in enumerate(BACKBONE_NUCLEI):
        mean_arr = np.asarray(agg_mean[:, nuc_idx], dtype=np.float32)
        lv = np.asarray(agg_logvar[:, nuc_idx], dtype=np.float32)
        std_arr = np.where(
            np.isnan(lv),
            np.float32("nan"),
            np.exp(0.5 * lv, where=~np.isnan(lv), out=np.zeros_like(lv, dtype=np.float32)),
        ).astype(np.float32)
        result.mean[nuc] = mean_arr
        result.std[nuc] = std_arr

    return result
