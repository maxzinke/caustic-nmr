"""ONNX export for the shift predictor.

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
    "solchem_features",
    "temperature_feature",
    "target_atom_rsasa",
    "atom_valid",
    # 2026-05-22: PaiNN backbone support. Position vectors give the edge
    # displacement vectors used in PaiNN's directional message channel;
    # node_normal initialises the equivariant v channel for ring nodes.
    # is_hbond / hb_cos are edge-level h-bond signals that get concatenated
    # into edge_feat. SchNet ignores pos and node_normal; both PaiNN and
    # SchNet consume is_hbond + hb_cos since edge_chemistry features were
    # added to the trunk edge_feat in d47.
    "pos",
    "node_normal",
    "is_hbond",
    "hb_cos",
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
    """Load a ``ModelConfig`` from a training YAML."""
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
            cfg = model.config
            self.use_residual = bool(getattr(cfg, "use_residual", False))
            self.use_branches = bool(getattr(cfg, "use_branches", True))
            self.use_geometry = bool(getattr(cfg, "use_geometry", True))
            self.use_target_env = bool(getattr(cfg, "use_target_env", False))
            self.use_solchem = bool(getattr(cfg, "use_solchem", False))
            self.use_temperature = bool(getattr(cfg, "use_temperature", False))
            self.use_conditions = bool(getattr(cfg, "use_conditions", False))
            self.use_per_atom_rsasa = bool(getattr(cfg, "use_per_atom_rsasa", False))
            self.use_cys_cb_subhead = bool(getattr(cfg, "use_cys_cb_subhead", False))
            self.backbone_name = str(getattr(cfg, "backbone", "schnet"))
            self.num_rbf = int(cfg.num_rbf)
            self.cutoff = float(model._cutoff)
            self.balance_branch_inputs = bool(getattr(model, "balance_branch_inputs", False))
            if self.backbone_name not in ("schnet", "painn"):
                raise ValueError(
                    "ONNX export supports schnet and painn backbones. "
                    f"Got {self.backbone_name!r}."
                )
            self.use_frame_projection = bool(getattr(cfg, "use_frame_projection", False))
            self.use_multi_layer_v = bool(getattr(cfg, "use_multi_layer_v", False))
            self.use_v_invariants = bool(getattr(cfg, "use_v_invariants", False))
            self.use_s2_nh = bool(getattr(cfg, "use_s2_nh", False))
            self.use_edge_chemistry = bool(getattr(cfg, "use_edge_chemistry", False))
            self.use_residue_chemistry = bool(getattr(cfg, "use_residue_chemistry", False))
            self.use_target_chemistry = bool(getattr(cfg, "use_target_chemistry", False))
            unsupported = [
                name for name, on in (
                    ("use_frame_projection", self.use_frame_projection),
                    ("use_multi_layer_v", self.use_multi_layer_v),
                    ("use_v_invariants", self.use_v_invariants),
                    ("use_s2_nh", self.use_s2_nh),
                    ("use_edge_chemistry", self.use_edge_chemistry),
                    ("use_residue_chemistry", self.use_residue_chemistry),
                    ("use_target_chemistry", self.use_target_chemistry),
                )
                if on
            ]
            if unsupported:
                raise ValueError(
                    "ONNX export does not yet handle these PaiNN feature flags: "
                    f"{unsupported}. Disable them in the ModelConfig (default for v2 "
                    "production) or extend export.py."
                )
            if self.use_conditions:
                raise ValueError(
                    "ONNX export does not yet carry sample-pH conditioning. "
                    "Disable use_conditions in the ModelConfig or extend export.py."
                )
            if self.use_cys_cb_subhead:
                raise ValueError(
                    "ONNX export does not support use_cys_cb_subhead=True (D46+). "
                    "The CYS-CB sub-head uses Python conditionals that don't trace."
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
            pos: torch.Tensor,
            node_normal: torch.Tensor,
            is_hbond: torch.Tensor,
            hb_cos: torch.Tensor,
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
            edge_feat = torch.cat(
                [rbf, sep_emb, same, is_hbond.unsqueeze(-1), hb_cos.unsqueeze(-1)],
                dim=-1,
            )

            if self.backbone_name == "schnet":
                for interaction in m.interactions:
                    x = interaction(x, edge_index, edge_feat, edge_dist, self.cutoff)
                return x

            # PaiNN: equivariant vector channel runs alongside scalar.
            src_idx, dst_idx = edge_index[0], edge_index[1]
            raw_disp = pos.index_select(0, src_idx) - pos.index_select(0, dst_idx)
            inv_dist = 1.0 / edge_dist.clamp(min=1e-6).unsqueeze(-1)
            edge_vec = raw_disp * inv_dist

            H = x.size(1)
            v = node_normal.unsqueeze(1).expand(-1, H, -1).contiguous()

            for interaction in m.interactions:
                x, v = interaction(
                    x, v, edge_index, edge_feat, edge_dist, edge_vec, self.cutoff,
                )
            x = x + v.norm(dim=-1)
            return x

        def forward(
            self,
            element: torch.Tensor,
            amino_acid: torch.Tensor,
            atom_role: torch.Tensor,
            bfactor: torch.Tensor,
            edge_index: torch.Tensor,
            edge_dist: torch.Tensor,
            seq_sep: torch.Tensor,
            same_residue: torch.Tensor,
            target_indices: torch.Tensor,
            target_mask: torch.Tensor,
            residue_types: torch.Tensor,
            residue_geometry: torch.Tensor,
            target_environment: torch.Tensor,
            solchem_features: torch.Tensor,
            temperature_feature: torch.Tensor,
            target_atom_rsasa: torch.Tensor,
            atom_valid: torch.Tensor,
            pos: torch.Tensor,
            node_normal: torch.Tensor,
            is_hbond: torch.Tensor,
            hb_cos: torch.Tensor,
        ):
            del atom_valid
            m = self.model
            x = self._trunk(
                element, amino_acid, atom_role, bfactor,
                edge_index, edge_dist, seq_sep, same_residue,
                pos, node_normal, is_hbond, hb_cos,
            )

            R = target_indices.shape[0]
            H = x.shape[1]

            ti_flat = target_indices.reshape(-1)
            emb = x.index_select(0, ti_flat).reshape(R, 6, H)

            if self.use_geometry:
                g = m.geo_proj(residue_geometry)
                g_exp = g.unsqueeze(1).expand(-1, 6, -1)
            else:
                g_exp = None

            if self.use_target_env:
                env_proj = m.env_proj(target_environment)
            else:
                env_proj = None

            if self.use_solchem:
                s = m.solchem_proj(solchem_features)
                s_exp = s.unsqueeze(1).expand(-1, 6, -1)
            else:
                s_exp = None

            if self.use_temperature:
                t = m.temp_proj(temperature_feature)
                t_exp = t.unsqueeze(1).expand(-1, 6, -1)
            else:
                t_exp = None

            if self.use_per_atom_rsasa:
                rsasa_in = target_atom_rsasa.unsqueeze(-1)
                rsasa_feat = m.per_atom_rsasa_proj(rsasa_in)
            else:
                rsasa_feat = None

            if self.balance_branch_inputs:
                emb = m.trunk_ln(emb)
                if g_exp is not None:
                    g_exp = m.geo_ln(g_exp)
                if env_proj is not None:
                    env_proj = m.env_ln(env_proj)
                if s_exp is not None:
                    s_exp = m.solchem_ln(s_exp)
                if t_exp is not None:
                    t_exp = m.temp_ln(t_exp)

            parts = [emb]
            if g_exp is not None:
                parts.append(g_exp)
            if env_proj is not None:
                parts.append(env_proj)
            if s_exp is not None:
                parts.append(s_exp)
            if t_exp is not None:
                parts.append(t_exp)
            if rsasa_feat is not None:
                parts.append(rsasa_feat)
            feat = torch.cat(parts, dim=-1)

            if self.use_branches:
                feat_proton = feat[:, :len(_PROTON_NUCLEI), :]
                feat_carbon = feat[:, len(_PROTON_NUCLEI):, :]
                in_dim = feat.shape[-1]

                p_flat = feat_proton.reshape(-1, in_dim)
                c_flat = feat_carbon.reshape(-1, in_dim)
                p_hidden = m.proton_branch(p_flat).reshape(R, len(_PROTON_NUCLEI), -1)
                c_hidden = m.carbon_branch(c_flat).reshape(R, len(_CARBON_NUCLEI), -1)

                outs = []
                for i, nuc in enumerate(_PROTON_NUCLEI):
                    outs.append(m.proton_heads[nuc](p_hidden[:, i, :]))
                for i, nuc in enumerate(_CARBON_NUCLEI):
                    outs.append(m.carbon_heads[nuc](c_hidden[:, i, :]))
                out = torch.stack(outs, dim=1)
            else:
                outs = []
                for i, nuc in enumerate(BACKBONE_NUCLEI):
                    outs.append(m.heads[nuc](feat[:, i, :]))
                out = torch.stack(outs, dim=1)

            shift_mean = m._shift_mean.view(1, 6)
            shift_std = m._shift_std.view(1, 6)
            z_mean = out[:, :, 0]
            logvar = out[:, :, 1]

            if self.use_residual:
                aa_mean = m._aa_means[residue_types]
                pred_mean = aa_mean + z_mean * shift_std
            else:
                pred_mean = z_mean * shift_std + shift_mean

            nan = torch.full_like(pred_mean, float("nan"))
            pred_mean = torch.where(target_mask, pred_mean, nan)
            pred_logvar = torch.where(target_mask, logvar, nan)
            return pred_mean, pred_logvar

    return _ExportableShiftPredictorImpl


def _dummy_inputs(num_atoms: int = 60, num_residues: int = 6, num_edges: int = 120):
    """Produce a self-consistent dummy batch for tracing."""
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
    from caustic.features import NUM_GEO_FEATURES, NUM_SOLCHEM_FEATURES, NUM_TARGET_ENV_FEATURES
    residue_geometry = torch.randn(num_residues, NUM_GEO_FEATURES)
    target_environment = torch.randn(num_residues, 6, NUM_TARGET_ENV_FEATURES)
    solchem_features = torch.randn(num_residues, NUM_SOLCHEM_FEATURES)
    temperature_feature = torch.zeros(num_residues, 1)
    target_atom_rsasa = torch.rand(num_residues, 6)
    atom_valid = torch.ones(num_residues, 6, dtype=torch.bool)
    pos = torch.randn(num_atoms, 3) * 5.0
    node_normal = torch.zeros(num_atoms, 3)
    is_hbond = torch.zeros(num_edges)
    hb_cos = torch.zeros(num_edges)

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
        solchem_features,
        temperature_feature,
        target_atom_rsasa,
        atom_valid,
        pos,
        node_normal,
        is_hbond,
        hb_cos,
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
    """Export a trained shift-predictor checkpoint to ONNX."""
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
        "solchem_features": {0: "num_residues"},
        "temperature_feature": {0: "num_residues"},
        "target_atom_rsasa": {0: "num_residues"},
        "atom_valid": {0: "num_residues"},
        "pos": {0: "num_atoms"},
        "node_normal": {0: "num_atoms"},
        "is_hbond": {0: "num_edges"},
        "hb_cos": {0: "num_edges"},
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


def _data_to_numpy(g, expected_input_names: set[str] | None = None) -> dict[str, np.ndarray]:
    """Convert a PyG ``Data`` object produced by ``structure_to_graph`` into
    the flat numpy dict the ONNX session expects.

    ``expected_input_names`` lets the caller (``run_onnx_inference``) pass in
    the exact input names the ONNX graph wants, so we don't send extra keys
    to an older D14-era export that doesn't know about temperature/solchem,
    and don't forget them for D17+ exports that do.
    """
    R = int(g.num_residues)

    te = getattr(g, "target_environment", None)
    if te is None:
        te_np = np.zeros((R, 6, 15), dtype=np.float32)
    else:
        te_np = te.detach().cpu().numpy().astype(np.float32)

    base = {
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

    if expected_input_names is None or "solchem_features" in expected_input_names:
        from caustic.features import NUM_SOLCHEM_FEATURES
        sc = getattr(g, "solchem_features", None)
        if sc is None:
            base["solchem_features"] = np.zeros((R, NUM_SOLCHEM_FEATURES), dtype=np.float32)
        else:
            base["solchem_features"] = sc.detach().cpu().numpy().astype(np.float32)

    if expected_input_names is None or "temperature_feature" in expected_input_names:
        tf = getattr(g, "temperature_feature", None)
        if tf is None:
            base["temperature_feature"] = np.zeros((R, 1), dtype=np.float32)
        else:
            base["temperature_feature"] = tf.detach().cpu().numpy().astype(np.float32)

    if expected_input_names is None or "target_atom_rsasa" in expected_input_names:
        rsasa = getattr(g, "target_atom_rsasa", None)
        if rsasa is None:
            base["target_atom_rsasa"] = np.zeros((R, 6), dtype=np.float32)
        else:
            base["target_atom_rsasa"] = rsasa.detach().cpu().numpy().astype(np.float32)

    if expected_input_names is None or "atom_valid" in expected_input_names:
        av = getattr(g, "atom_valid", None)
        if av is None:
            base["atom_valid"] = np.ones((R, 6), dtype=bool)
        else:
            base["atom_valid"] = av.detach().cpu().numpy().astype(bool)

    # 2026-05-22: PaiNN inputs. SchNet ONNX sessions don't have these in
    # their expected_input_names so we only emit them for PaiNN exports.
    N = int(g.element.shape[0])
    if expected_input_names is None or "pos" in expected_input_names:
        ppos = getattr(g, "pos", None)
        if ppos is None:
            base["pos"] = np.zeros((N, 3), dtype=np.float32)
        else:
            base["pos"] = ppos.detach().cpu().numpy().astype(np.float32)
    if expected_input_names is None or "node_normal" in expected_input_names:
        nn = getattr(g, "node_normal", None)
        if nn is None:
            base["node_normal"] = np.zeros((N, 3), dtype=np.float32)
        else:
            base["node_normal"] = nn.detach().cpu().numpy().astype(np.float32)
    E = int(g.edge_index.shape[1])
    if expected_input_names is None or "is_hbond" in expected_input_names:
        hb = getattr(g, "is_hbond", None)
        if hb is None:
            base["is_hbond"] = np.zeros(E, dtype=np.float32)
        else:
            base["is_hbond"] = hb.detach().cpu().numpy().astype(np.float32)
    if expected_input_names is None or "hb_cos" in expected_input_names:
        hbc = getattr(g, "hb_cos", None)
        if hbc is None:
            base["hb_cos"] = np.zeros(E, dtype=np.float32)
        else:
            base["hb_cos"] = hbc.detach().cpu().numpy().astype(np.float32)

    if expected_input_names is not None:
        base = {k: v for k, v in base.items() if k in expected_input_names}
    return base


def run_onnx_inference(session, g) -> tuple[np.ndarray, np.ndarray]:
    """Run a single-conformer forward pass through an ONNX Runtime session.

    Returns ``(pred_mean, pred_logvar)`` as ``[R, 6]`` float32 numpy arrays.
    Positions without a valid target atom are NaN (matching the PyTorch
    path's convention).
    """
    expected = {i.name for i in session.get_inputs()}
    inputs = _data_to_numpy(g, expected_input_names=expected)
    outputs = session.run(None, inputs)
    return outputs[0], outputs[1]


def load_onnx_session(onnx_path: str | Path):
    """Open an ONNX Runtime ``InferenceSession`` on the CPU execution provider."""
    import onnxruntime as ort

    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
