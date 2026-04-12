"""ONNX export for the Caustic.

``ShiftPredictor.forward`` takes a PyTorch Geometric ``Data`` object and uses
boolean masking + dict outputs — neither of those traces cleanly through
``torch.onnx.export``. This module wraps the trained model in a flat-tensor
facade that always computes all six backbone nuclei and returns ``[R, 6]``
mean and log-variance tensors, then runs the standard export path.

The resulting ``.onnx`` file accepts variable num_atoms, num_edges, and
num_residues via ONNX dynamic axes, so one export handles proteins of any
size. Post-processing (target-mask application, denormalisation into ppm)
happens in the wrapper itself, matching what the PyTorch inference path
already does. Callers can run the exported model through
``onnxruntime.InferenceSession`` without any of PyTorch / PyTorch Geometric
as runtime dependencies — ONNX Runtime + numpy + gemmi is enough.

Use::

    from caustic.export import export_to_onnx
    onnx_path = export_to_onnx(
        "checkpoints/shift_predictor/best.pt",
        "checkpoints/shift_predictor/best.onnx",
    )
"""
from __future__ import annotations

from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

import numpy as np


_INPUT_NAMES = [
    "element",
    "amino_acid",
    "atom_role",
    "bfactor",
    "edge_index",
    "edge_dist",
    "seq_sep",
    "same_residue",
    "target_indices",
    "target_mask",
    "residue_types",
    "residue_geometry",
    "target_environment",
]

_OUTPUT_NAMES = ["pred_mean", "pred_logvar"]


def _model_config_from_checkpoint(checkpoint_path: str | Path):
    """Extract a ``ModelConfig`` from a saved checkpoint, falling back to defaults."""
    import torch

    from caustic.config import ModelConfig

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = state.get("model_config")
    if raw is None:
        return ModelConfig()
    valid = {f.name for f in dc_fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in raw.items() if k in valid})


def _model_config_from_yaml(yaml_path: str | Path):
    """Load a ``ModelConfig`` from a training YAML (``configs/shift_predictor.yaml``)."""
    import yaml

    from caustic.config import ModelConfig

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    model_section = data.get("model", {})
    valid = {f.name for f in dc_fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in model_section.items() if k in valid})


def _resolve_model_config(
    checkpoint_path: str | Path,
    model_config: Any | None,
    config_yaml: str | Path | None,
):
    """Pick the right ``ModelConfig`` for an export call.

    Priority: explicit kwarg > YAML > checkpoint-embedded > defaults.
    Raised as its own helper so ``export_to_onnx`` and ``validate_onnx``
    stay in sync.
    """
    if model_config is not None:
        return model_config
    if config_yaml is not None:
        return _model_config_from_yaml(config_yaml)
    return _model_config_from_checkpoint(checkpoint_path)


class ExportableShiftPredictor:
    """Flat-tensor facade for ``ShiftPredictor`` used during ONNX tracing.

    Wraps a trained ``ShiftPredictor`` so the forward pass:
    - takes explicit tensors instead of a PyG ``Data`` object,
    - unrolls the per-nucleus loop across all six backbone nuclei,
    - avoids boolean masking (always computes [R, 6] outputs),
    - returns a tuple of tensors (mean, logvar) — no Python dicts,
    - denormalises z-score outputs into ppm inline.

    Not a subclass of ``nn.Module`` in the usual sense — we compose it
    around an existing model. The wrapper itself is ``nn.Module`` so
    ``torch.onnx.export`` picks up the module hierarchy.
    """

    def __new__(cls, model: Any):
        import torch.nn as nn

        if not hasattr(cls, "_impl_class"):
            cls._impl_class = _build_impl_class(nn.Module)
        return cls._impl_class(model)


def _build_impl_class(base_module):
    """Build the wrapper class lazily so torch is optional at import time."""
    import torch

    from caustic.features import BACKBONE_NUCLEI, rbf_expansion
    from caustic.model import _CARBON_NUCLEI, _PROTON_NUCLEI

    class _ExportableShiftPredictorImpl(base_module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Cache a few config values so forward() doesn't dip into self.model.config
            # inside the trace (tracer can't follow Python attribute access on a
            # dataclass attribute of a submodule when it's in a conditional).
            cfg = model.config
            self.use_residual = bool(getattr(cfg, "use_residual", False))
            self.use_branches = bool(getattr(cfg, "use_branches", True))
            self.use_geometry = bool(getattr(cfg, "use_geometry", True))
            self.use_target_env = bool(getattr(cfg, "use_target_env", False))
            self.use_conditions = bool(getattr(cfg, "use_conditions", False))
            self.backbone_name = str(getattr(cfg, "backbone", "schnet"))
            self.num_rbf = int(cfg.num_rbf)
            self.cutoff = float(model._cutoff)
            self.balance_branch_inputs = bool(getattr(model, "balance_branch_inputs", False))
            if self.backbone_name != "schnet":
                raise ValueError(
                    "ONNX export currently supports the SchNet backbone only. "
                    f"Got {self.backbone_name!r}."
                )
            if self.use_conditions:
                raise ValueError(
                    "ONNX export does not yet carry sample-pH conditioning. "
                    "Disable use_conditions in the ModelConfig or extend export.py."
                )

        def _trunk(
            self,
            element: torch.Tensor,
            amino_acid: torch.Tensor,
            atom_role: torch.Tensor,
            bfactor: torch.Tensor,
            edge_index: torch.Tensor,
            edge_dist: torch.Tensor,
            seq_sep: torch.Tensor,
            same_residue: torch.Tensor,
        ) -> torch.Tensor:
            m = self.model
            h_el = m.element_emb(element)
            h_aa = m.aa_emb(amino_acid)
            h_role = m.role_emb(atom_role)
            bf = bfactor.unsqueeze(-1) / 100.0
            x = m.input_proj(torch.cat([h_el, h_aa, h_role, bf], dim=-1))

            rbf = rbf_expansion(edge_dist, self.num_rbf, self.cutoff)
            sep_emb = m.seq_sep_emb(seq_sep)
            same = same_residue.unsqueeze(-1)
            edge_feat = torch.cat([rbf, sep_emb, same], dim=-1)

            for interaction in m.interactions:
                x = interaction(x, edge_index, edge_feat, edge_dist, self.cutoff)
            return x

        def forward(
            self,
            element: torch.Tensor,         # [N_atoms]  long
            amino_acid: torch.Tensor,      # [N_atoms]  long
            atom_role: torch.Tensor,       # [N_atoms]  long
            bfactor: torch.Tensor,         # [N_atoms]  float
            edge_index: torch.Tensor,      # [2, N_edges] long
            edge_dist: torch.Tensor,       # [N_edges]  float
            seq_sep: torch.Tensor,         # [N_edges]  long
            same_residue: torch.Tensor,    # [N_edges]  float
            target_indices: torch.Tensor,  # [R, 6]     long
            target_mask: torch.Tensor,     # [R, 6]     bool (or uint8)
            residue_types: torch.Tensor,   # [R]        long
            residue_geometry: torch.Tensor,   # [R, 34]     float
            target_environment: torch.Tensor, # [R, 6, 15]  float (or empty if unused)
        ):
            m = self.model
            x = self._trunk(
                element, amino_acid, atom_role, bfactor,
                edge_index, edge_dist, seq_sep, same_residue,
            )  # [N_atoms, H]

            R = target_indices.shape[0]
            H = x.shape[1]

            # Gather target-atom features: [R, 6, H].
            # target_indices is 0 where a target atom is missing; the mask
            # applied at the end tells the caller which positions are real.
            ti_flat = target_indices.reshape(-1)  # [R*6]
            emb = x.index_select(0, ti_flat).reshape(R, 6, H)  # [R, 6, H]

            # --- Geometry features (broadcast across the nucleus axis) ---
            if self.use_geometry:
                g = m.geo_proj(residue_geometry)  # [R, geo_dim]
                g_exp = g.unsqueeze(1).expand(-1, 6, -1)  # [R, 6, geo_dim]
            else:
                g_exp = None

            # --- Per-nucleus environment features ---
            if self.use_target_env:
                env_proj = m.env_proj(target_environment)  # [R, 6, env_dim]
            else:
                env_proj = None

            # --- D13 balance option ---
            if self.balance_branch_inputs:
                emb = m.trunk_ln(emb)
                if g_exp is not None:
                    g_exp = m.geo_ln(g_exp)
                if env_proj is not None:
                    env_proj = m.env_ln(env_proj)

            parts = [emb]
            if g_exp is not None:
                parts.append(g_exp)
            if env_proj is not None:
                parts.append(env_proj)
            feat = torch.cat(parts, dim=-1)  # [R, 6, head_input]

            if self.use_branches:
                # Split the nucleus axis into proton (0:2) and carbon (2:6).
                # Run each branch over its family as a single flat (R*k, head_input)
                # matmul — far easier to trace than a Python loop over nuclei.
                feat_proton = feat[:, :len(_PROTON_NUCLEI), :]          # [R, 2, in]
                feat_carbon = feat[:, len(_PROTON_NUCLEI):, :]          # [R, 4, in]
                in_dim = feat.shape[-1]

                p_flat = feat_proton.reshape(-1, in_dim)
                c_flat = feat_carbon.reshape(-1, in_dim)
                p_hidden = m.proton_branch(p_flat).reshape(R, len(_PROTON_NUCLEI), -1)
                c_hidden = m.carbon_branch(c_flat).reshape(R, len(_CARBON_NUCLEI), -1)

                # Apply each nucleus-specific head. The per-head net is a small
                # MLP; unrolling keeps the ONNX graph static.
                outs = []
                for i, nuc in enumerate(_PROTON_NUCLEI):
                    outs.append(m.proton_heads[nuc](p_hidden[:, i, :]))
                for i, nuc in enumerate(_CARBON_NUCLEI):
                    outs.append(m.carbon_heads[nuc](c_hidden[:, i, :]))
                out = torch.stack(outs, dim=1)  # [R, 6, 2]
            else:
                outs = []
                for i, nuc in enumerate(BACKBONE_NUCLEI):
                    outs.append(m.heads[nuc](feat[:, i, :]))
                out = torch.stack(outs, dim=1)

            # --- Denormalise z-score to ppm ---
            shift_mean = m._shift_mean.view(1, 6)   # [1, 6]
            shift_std = m._shift_std.view(1, 6)     # [1, 6]
            z_mean = out[:, :, 0]                    # [R, 6]
            logvar = out[:, :, 1]                    # [R, 6]

            if self.use_residual:
                aa_mean = m._aa_means[residue_types]  # [R, 6]
                pred_mean = aa_mean + z_mean * shift_std
            else:
                pred_mean = z_mean * shift_std + shift_mean

            # Mask positions where a target atom doesn't exist (NaN is
            # easier to detect downstream than zero).
            mask_f = target_mask.to(pred_mean.dtype)
            nan = torch.full_like(pred_mean, float("nan"))
            pred_mean = torch.where(target_mask, pred_mean, nan)
            pred_logvar = torch.where(target_mask, logvar, nan)
            return pred_mean, pred_logvar

    return _ExportableShiftPredictorImpl


def _dummy_inputs(num_atoms: int = 60, num_residues: int = 6, num_edges: int = 120):
    """Produce a self-consistent dummy batch for tracing.

    Atoms are distributed roughly evenly across residues, target_indices
    points at the first atom of each residue for all nuclei, and edges
    form a small complete graph around residue 0 to exercise scatter_add.
    """
    import torch

    element = torch.randint(0, 5, (num_atoms,), dtype=torch.long)
    amino_acid = torch.randint(0, 20, (num_atoms,), dtype=torch.long)
    atom_role = torch.randint(0, 10, (num_atoms,), dtype=torch.long)
    bfactor = torch.rand(num_atoms) * 50.0

    src = torch.randint(0, num_atoms, (num_edges,), dtype=torch.long)
    dst = torch.randint(0, num_atoms, (num_edges,), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_dist = torch.rand(num_edges) * 6.0 + 1.0
    seq_sep = torch.randint(0, 32, (num_edges,), dtype=torch.long)
    same_residue = (torch.randint(0, 2, (num_edges,))).float()

    base = torch.arange(num_residues, dtype=torch.long) * (num_atoms // num_residues)
    target_indices = base.unsqueeze(-1).expand(-1, 6).contiguous()
    target_mask = torch.ones(num_residues, 6, dtype=torch.bool)
    residue_types = torch.randint(0, 20, (num_residues,), dtype=torch.long)
    residue_geometry = torch.randn(num_residues, 34)
    target_environment = torch.randn(num_residues, 6, 15)

    return (
        element,
        amino_acid,
        atom_role,
        bfactor,
        edge_index,
        edge_dist,
        seq_sep,
        same_residue,
        target_indices,
        target_mask,
        residue_types,
        residue_geometry,
        target_environment,
    )


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    model_config: Any | None = None,
    config_yaml: str | Path | None = None,
    opset: int = 17,
    verbose: bool = False,
) -> str:
    """Export a trained Caustic checkpoint to ONNX.

    Parameters
    ----------
    checkpoint_path: path to a ``.pt`` checkpoint (loadable via torch.load).
    output_path: where to write the ``.onnx`` file.
    model_config: optional explicit ``ModelConfig`` that matches the
        checkpoint's architecture. Use this when the checkpoint was saved
        without an embedded config (D8 checkpoints in this repo don't).
    config_yaml: optional path to a training YAML
        (e.g. ``configs/shift_predictor.yaml``). Read as a fallback when
        ``model_config`` is not given.
    opset: ONNX opset version. 17 is the default (same as neural/export.py);
        needs to be ≥13 because scatter_add_ is the ScatterND op.
    verbose: pass through to ``torch.onnx.export``.

    Returns
    -------
    Absolute path to the exported ``.onnx`` file.
    """
    import torch

    from caustic.model import ShiftPredictor

    cfg = _resolve_model_config(checkpoint_path, model_config, config_yaml)
    model = ShiftPredictor(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = ExportableShiftPredictor(model)
    wrapper.eval()

    inputs = _dummy_inputs()

    dynamic_axes = {
        "element": {0: "num_atoms"},
        "amino_acid": {0: "num_atoms"},
        "atom_role": {0: "num_atoms"},
        "bfactor": {0: "num_atoms"},
        "edge_index": {1: "num_edges"},
        "edge_dist": {0: "num_edges"},
        "seq_sep": {0: "num_edges"},
        "same_residue": {0: "num_edges"},
        "target_indices": {0: "num_residues"},
        "target_mask": {0: "num_residues"},
        "residue_types": {0: "num_residues"},
        "residue_geometry": {0: "num_residues"},
        "target_environment": {0: "num_residues"},
        "pred_mean": {0: "num_residues"},
        "pred_logvar": {0: "num_residues"},
    }

    output_path = str(Path(output_path).resolve())
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            opset_version=opset,
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=verbose,
        )
    return output_path


# ----------------------------------------------------------------------
# ONNX Runtime inference path
# ----------------------------------------------------------------------


def _data_to_numpy(g) -> dict[str, np.ndarray]:
    """Convert a PyG ``Data`` object produced by ``structure_to_graph`` into
    the flat numpy dict the ONNX session expects."""
    te = getattr(g, "target_environment", None)
    if te is None:
        # The exported graph expects a real target_environment tensor; when
        # the model config doesn't use it, ship an all-zeros placeholder of
        # the right shape so the session input count matches the graph.
        R = int(g.num_residues)
        te_np = np.zeros((R, 6, 15), dtype=np.float32)
    else:
        te_np = te.detach().cpu().numpy().astype(np.float32)

    return {
        "element": g.element.detach().cpu().numpy().astype(np.int64),
        "amino_acid": g.amino_acid.detach().cpu().numpy().astype(np.int64),
        "atom_role": g.atom_role.detach().cpu().numpy().astype(np.int64),
        "bfactor": g.bfactor.detach().cpu().numpy().astype(np.float32),
        "edge_index": g.edge_index.detach().cpu().numpy().astype(np.int64),
        "edge_dist": g.edge_dist.detach().cpu().numpy().astype(np.float32),
        "seq_sep": g.seq_sep.detach().cpu().numpy().astype(np.int64),
        "same_residue": g.same_residue.detach().cpu().numpy().astype(np.float32),
        "target_indices": g.target_indices.detach().cpu().numpy().astype(np.int64),
        "target_mask": g.target_mask.detach().cpu().numpy().astype(bool),
        "residue_types": g.residue_types.detach().cpu().numpy().astype(np.int64),
        "residue_geometry": g.residue_geometry.detach().cpu().numpy().astype(np.float32),
        "target_environment": te_np,
    }


def run_onnx_inference(session, g) -> tuple[np.ndarray, np.ndarray]:
    """Run a single-conformer forward pass through an ONNX Runtime session.

    Parameters
    ----------
    session: an ``onnxruntime.InferenceSession``.
    g: a PyG ``Data`` object from ``structure_to_graph`` with
       ``residue_geometry`` (and ``target_environment`` when the model uses
       it) already computed.

    Returns ``(pred_mean, pred_logvar)`` as ``[R, 6]`` float32 numpy arrays.
    Positions without a valid target atom are NaN (matching the PyTorch
    path's convention).
    """
    inputs = _data_to_numpy(g)
    outputs = session.run(None, inputs)
    return outputs[0], outputs[1]


def load_onnx_session(onnx_path: str | Path):
    """Open an ONNX Runtime ``InferenceSession`` on the CPU execution provider."""
    import onnxruntime as ort

    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def validate_onnx(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    structure_path: str | Path,
    *,
    model_config: Any | None = None,
    config_yaml: str | Path | None = None,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Validate an exported ONNX model against the PyTorch model.

    Runs both paths on the same structure and returns the per-nucleus max
    absolute error between the mean predictions. Raises if any nucleus
    exceeds ``atol``.

    Accepts the same ``model_config`` / ``config_yaml`` resolver inputs as
    ``export_to_onnx`` so the two paths stay synchronised.
    """
    import torch

    from caustic.config import GraphConfig
    from caustic.graph import (
        compute_residue_geometry,
        compute_target_environment,
        structure_to_graph,
    )
    from caustic.model import ShiftPredictor
    from caustic.features import BACKBONE_NUCLEI

    cfg = _resolve_model_config(checkpoint_path, model_config, config_yaml)
    model = ShiftPredictor(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    g = structure_to_graph(structure_path, config=GraphConfig())
    if not hasattr(g, "residue_geometry") or g.residue_geometry is None:
        g.residue_geometry = compute_residue_geometry(g)
    if getattr(cfg, "use_target_env", False) and (
        not hasattr(g, "target_environment") or g.target_environment is None
    ):
        g.target_environment = compute_target_environment(g)

    # PyTorch reference — the original model uses boolean masking, so we
    # rebuild a full [R, 6] array to diff against.
    R = int(g.num_residues)
    ref_mean = np.full((R, 6), np.nan, dtype=np.float32)
    with torch.no_grad():
        pm, _plv = model(g)
    tm_np = g.target_mask.cpu().numpy()
    for i, nuc in enumerate(BACKBONE_NUCLEI):
        if nuc in pm and pm[nuc].numel() > 0:
            ref_mean[tm_np[:, i], i] = pm[nuc].cpu().numpy()

    session = load_onnx_session(onnx_path)
    onnx_mean, _onnx_logvar = run_onnx_inference(session, g)

    diffs: dict[str, float] = {}
    for i, nuc in enumerate(BACKBONE_NUCLEI):
        valid = tm_np[:, i]
        if not valid.any():
            diffs[nuc] = 0.0
            continue
        d = np.abs(onnx_mean[valid, i] - ref_mean[valid, i])
        diffs[nuc] = float(d.max())

    worst_nuc = max(diffs, key=diffs.get)
    if diffs[worst_nuc] > atol:
        raise ValueError(
            f"ONNX / PyTorch mismatch on {worst_nuc}: "
            f"{diffs[worst_nuc]:.6f} > atol {atol}"
        )
    return diffs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export Caustic to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--validate-on", default=None,
                    help="Optional structure (.pdb/.cif) for parity check")
    args = ap.parse_args()

    out = export_to_onnx(args.checkpoint, args.output, opset=args.opset)
    print(f"Exported to: {out}")
    if args.validate_on:
        diffs = validate_onnx(args.checkpoint, out, args.validate_on)
        print(f"Validation passed (per-nucleus max abs diff): {diffs}")
