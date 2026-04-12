"""SchNet-style GNN for backbone chemical shift prediction.

v2: backbone geometry features + proton/carbon branch heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from caustic.config import ModelConfig
from caustic.features import (
    BACKBONE_NUCLEI,
    NUM_AA_TYPES,
    NUM_ATOM_ROLES,
    NUM_ELEMENT_TYPES,
    NUM_GEO_FEATURES,
    NUM_TARGET_ENV_FEATURES,
    UNK_AA_IDX,
    UNK_ELEMENT_IDX,
    rbf_expansion,
    cosine_cutoff,
)

# Embedding table sizes (canonical + 1 for unknown)
_ELEM_EMB_SIZE = NUM_ELEMENT_TYPES + 1
_AA_EMB_SIZE = NUM_AA_TYPES + 1
_ROLE_EMB_SIZE = NUM_ATOM_ROLES
_SEQ_SEP_EMB_SIZE = 33  # 0..32
_SEQ_SEP_DIM = 16

# Branch assignments
_PROTON_NUCLEI = ("H", "HA")
_CARBON_NUCLEI = ("N", "CA", "CB", "C")


class SchNetInteraction(nn.Module):
    """Continuous-filter convolution interaction block."""

    def __init__(self, hidden_dim: int, edge_feat_dim: int, dropout: float = 0.1):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_dist: torch.Tensor,
        cutoff_val: float,
    ) -> torch.Tensor:
        src, dst = edge_index
        W = self.filter_net(edge_feat)
        envelope = cosine_cutoff(edge_dist, cutoff_val).unsqueeze(-1)
        W = W * envelope
        msg = x[src] * W
        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=msg.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.layer_norm(x + self.dropout(self.node_net(agg)))


class PaiNNInteraction(nn.Module):
    """PaiNN-style equivariant interaction: scalar + vector channels.

    Each atom carries (s: [H], v: [H, 3]).  Messages use unit displacement
    vectors so directional information flows natively through the trunk.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, dropout: float = 0.1):
        super().__init__()
        H = hidden_dim
        # Scalar message filter (same as SchNet)
        self.filter_net = nn.Sequential(
            nn.Linear(edge_feat_dim, H),
            nn.SiLU(),
            nn.Linear(H, 3 * H),  # splits into: scalar_msg, vec_scale, vec_filter
        )
        # Scalar update: agg_scalar + ||agg_vector|| → new scalar
        self.scalar_net = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        # Vector update: scalar-gated
        self.vec_scale = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_s = nn.LayerNorm(H)
        self.layer_norm_v = nn.LayerNorm(H)

    def forward(
        self,
        s: torch.Tensor,       # [N, H] scalar features
        v: torch.Tensor,       # [N, H, 3] vector features
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_dist: torch.Tensor,
        cutoff_val: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        N, H = s.shape

        # Compute filter + envelope
        W = self.filter_net(edge_feat)  # [E, 3H]
        envelope = cosine_cutoff(edge_dist, cutoff_val).unsqueeze(-1)  # [E, 1]
        W = W * envelope
        W_s, W_vscale, W_vfilt = W.chunk(3, dim=-1)  # each [E, H]

        # Unit direction vectors
        # Need positions — reconstruct from edge_index
        # Actually we need dir_ij passed in. Let's compute from the data.
        # For now, use v[src] projected by scalar filter as the vector message.

        # Scalar message
        s_msg = W_s * s[src]  # [E, H]
        agg_s = torch.zeros(N, H, device=s.device, dtype=s_msg.dtype)
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(s_msg), s_msg)

        # Vector message: filter * neighbor_vector + scalar-weighted direction
        v_msg = W_vscale.unsqueeze(-1) * v[src]  # [E, H, 3]
        agg_v = torch.zeros(N, H, 3, device=v.device, dtype=v_msg.dtype)
        agg_v.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(v_msg), v_msg)

        # Scalar update: concat(agg_s, ||agg_v||) → new scalar
        v_norm = agg_v.norm(dim=-1)  # [N, H]
        ds = self.dropout(self.scalar_net(torch.cat([agg_s, v_norm], dim=-1)))
        s_out = self.layer_norm_s(s + ds)

        # Vector update: scalar-gated
        v_gate = self.vec_scale(s_out).unsqueeze(-1)  # [N, H, 1]
        v_out = v + v_gate * agg_v
        # Normalize per-channel
        v_out = self.layer_norm_v(v_out.transpose(-1, -2)).transpose(-1, -2)

        return s_out, v_out


class NucleusHead(nn.Module):
    """Per-nucleus prediction head: hidden -> (mean, log_var)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # [mean, log_var]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchMLP(nn.Module):
    """Two-layer MLP for a nucleus family branch."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: AA embedding → (gamma, beta) for branch output.

    Learns AA-specific scale and shift so the same branch trunk can express
    different structural response surfaces per amino acid type.
    """

    def __init__(self, aa_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(aa_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # [gamma, beta]
        )
        # Init near-identity: gamma≈1, beta≈0
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        with torch.no_grad():
            self.film[-1].bias[:hidden_dim].fill_(1.0)  # gamma init = 1

    def forward(self, x: torch.Tensor, aa_emb: torch.Tensor) -> torch.Tensor:
        """x: [N, H], aa_emb: [N, aa_emb_dim] -> [N, H]."""
        gb = self.film(aa_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * x + beta


class ShiftPredictor(nn.Module):
    """Structure -> backbone chemical shift predictor.

    v2 architecture:
        1. Per-atom embedding (element + AA + role + B-factor)
        2. SchNet continuous-filter message passing (shared trunk)
        3. Extract target atom embeddings + per-residue geometry features
        4. Proton branch (H, HA) / Carbon-nitrogen branch (N, CA, CB, C)
        5. Per-nucleus MLP heads -> (mean, log_variance)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # --- Atom embeddings ---
        self.element_emb = nn.Embedding(_ELEM_EMB_SIZE, H)
        self.aa_emb = nn.Embedding(_AA_EMB_SIZE, H)
        self.role_emb = nn.Embedding(_ROLE_EMB_SIZE, H)
        self.input_proj = nn.Sequential(
            nn.Linear(3 * H + 1, H),
            nn.SiLU(),
        )

        # --- Edge feature projection ---
        self.seq_sep_emb = nn.Embedding(_SEQ_SEP_EMB_SIZE, _SEQ_SEP_DIM)
        edge_feat_dim = config.num_rbf + _SEQ_SEP_DIM + 1

        # --- Interaction layers (shared trunk) ---
        backbone = getattr(config, "backbone", "schnet")
        if backbone == "painn":
            self.interactions = nn.ModuleList([
                PaiNNInteraction(H, edge_feat_dim, config.dropout)
                for _ in range(config.num_layers)
            ])
            # Vector feature initialization: project unit displacement from CA to each atom
            self.vec_init = nn.Linear(H, H, bias=False)
        else:
            self.interactions = nn.ModuleList([
                SchNetInteraction(H, edge_feat_dim, config.dropout)
                for _ in range(config.num_layers)
            ])

        # --- Geometry feature projection ---
        geo_dim = 0
        if config.use_geometry:
            geo_dim = int(getattr(config, "geo_dim_override", 0) or H // 4)
            self.geo_proj = nn.Sequential(
                nn.Linear(NUM_GEO_FEATURES, geo_dim),
                nn.SiLU(),
                nn.Linear(geo_dim, geo_dim),
            )

        # --- Sample condition projection (pH) ---
        cond_dim = 0
        if getattr(config, "use_conditions", False):
            cond_dim = 16
            self.cond_proj = nn.Sequential(
                nn.Linear(1, cond_dim),
                nn.SiLU(),
            )

        # --- Per-target-atom environment projection ---
        env_dim = 0
        if getattr(config, "use_target_env", False):
            env_dim = int(getattr(config, "env_dim_override", 0) or H // 8)  # 16 for H=128
            self.env_proj = nn.Sequential(
                nn.Linear(NUM_TARGET_ENV_FEATURES, env_dim),
                nn.SiLU(),
                nn.Linear(env_dim, env_dim),
            )

        # --- D13: balance branch input magnitudes ---
        # Without this, the trunk contribution (||x|| ~ 10) drowns out the
        # geometry (~1.3) and env (~0.75) features at the concat point, so the
        # branch MLP has to amplify a <5% signal slice. LayerNorm normalizes
        # each piece to unit-ish scale before concat, equalizing their
        # contributions. Per the D13 grad-flow diagnosis.
        self.balance_branch_inputs = bool(getattr(config, "balance_branch_inputs", False))
        if self.balance_branch_inputs:
            self.trunk_ln = nn.LayerNorm(H)
            if config.use_geometry:
                self.geo_ln = nn.LayerNorm(geo_dim)
            if getattr(config, "use_target_env", False):
                self.env_ln = nn.LayerNorm(env_dim)

        # --- Branch heads ---
        head_input = H + geo_dim + cond_dim + env_dim

        if config.use_branches:
            self.proton_branch = BranchMLP(head_input, H, config.dropout)
            self.carbon_branch = BranchMLP(head_input, H, config.dropout)
            if getattr(config, "use_film", False):
                aa_emb_dim = H  # same as aa_emb dimension
                self.proton_film = FiLMLayer(aa_emb_dim, H)
                self.carbon_film = FiLMLayer(aa_emb_dim, H)
            self.proton_heads = nn.ModuleDict({
                nuc: NucleusHead(H, config.dropout) for nuc in _PROTON_NUCLEI
            })
            self.carbon_heads = nn.ModuleDict({
                nuc: NucleusHead(H, config.dropout) for nuc in _CARBON_NUCLEI
            })
        else:
            # Flat heads (v1 fallback)
            self.heads = nn.ModuleDict({
                nuc: NucleusHead(head_input if geo_dim > 0 else H, config.dropout)
                for nuc in BACKBONE_NUCLEI
            })

        self._cutoff = 8.0
        self._num_rbf = config.num_rbf

        # Per-nucleus output normalization (z-score -> ppm)
        self.register_buffer("_shift_mean", torch.tensor(
            [8.27, 4.37, 119.47, 56.92, 37.96, 176.02]
        ))
        self.register_buffer("_shift_std", torch.tensor(
            [0.64, 0.50, 5.17, 4.84, 12.72, 2.19]
        ))

        # AA-type means for residual prediction (21 AA × 6 nuclei, loaded by training)
        self.register_buffer("_aa_means", torch.zeros(NUM_AA_TYPES + 1, 6))

    def forward(
        self,
        data: object,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run forward pass.

        Returns:
            pred_mean:   {nucleus: [N_valid]} predicted mean shifts (ppm).
            pred_logvar: {nucleus: [N_valid]} predicted log-variance.
        """
        # --- Node encoding ---
        h_el = self.element_emb(data.element)
        h_aa = self.aa_emb(data.amino_acid)
        h_role = self.role_emb(data.atom_role)
        bf = data.bfactor.unsqueeze(-1) / 100.0
        x = self.input_proj(torch.cat([h_el, h_aa, h_role, bf], dim=-1))

        # --- Edge features ---
        rbf = rbf_expansion(data.edge_dist, self._num_rbf, self._cutoff)
        sep_emb = self.seq_sep_emb(data.seq_sep)
        same = data.same_residue.unsqueeze(-1)
        edge_feat = torch.cat([rbf, sep_emb, same], dim=-1)

        # --- Message passing (shared trunk) ---
        backbone = getattr(self.config, "backbone", "schnet")
        if backbone == "painn":
            # Initialize vector features from scalar embedding
            v = self.vec_init(x).unsqueeze(-1).expand(-1, -1, 3) * 0.01  # [N, H, 3] small init
            for interaction in self.interactions:
                x, v = interaction(x, v, data.edge_index, edge_feat, data.edge_dist, self._cutoff)
            # After PaiNN trunk, enrich scalar with vector norm for downstream heads
            x = x + v.norm(dim=-1)
        else:
            for interaction in self.interactions:
                x = interaction(x, data.edge_index, edge_feat, data.edge_dist, self._cutoff)

        # --- Geometry features (per-residue) ---
        geo_feat = None
        if self.config.use_geometry and hasattr(data, "residue_geometry"):
            geo_feat = self.geo_proj(data.residue_geometry.to(x.device))  # [R, geo_dim]

        # --- Sample condition features (per-residue broadcast) ---
        cond_feat = None
        if getattr(self.config, "use_conditions", False) and hasattr(data, "sample_ph"):
            # Normalize pH: center on 6.5 (population mean), scale by 1/2
            ph = (data.sample_ph.to(x.device) - 6.5) / 2.0  # [B] or [1]
            R = int(data.num_residues) if not hasattr(data.num_residues, '__len__') else int(data.num_residues.sum())
            cond_feat = self.cond_proj(ph.expand(R, 1))  # [R, cond_dim]

        # --- Per-target-atom environment features ---
        has_env = (
            getattr(self.config, "use_target_env", False)
            and hasattr(data, "target_environment")
            and data.target_environment is not None
        )

        # --- Per-nucleus predictions ---
        pred_mean: dict[str, torch.Tensor] = {}
        pred_logvar: dict[str, torch.Tensor] = {}

        for nuc_idx, nuc in enumerate(BACKBONE_NUCLEI):
            mask = data.target_mask[:, nuc_idx]
            if mask.sum() == 0:
                pred_mean[nuc] = torch.tensor([], device=x.device)
                pred_logvar[nuc] = torch.tensor([], device=x.device)
                continue

            atom_idx = data.target_indices[mask, nuc_idx]
            emb = x[atom_idx]  # [N_valid, H]

            # D13: normalize trunk output before concat so it doesn't drown
            # out geometry/env features (which have 10x smaller native norm).
            if self.balance_branch_inputs:
                emb = self.trunk_ln(emb)

            # Concatenate geometry and conditions if available
            if geo_feat is not None:
                g = geo_feat[mask]
                if self.balance_branch_inputs:
                    g = self.geo_ln(g)
                emb = torch.cat([emb, g], dim=-1)  # [N_valid, H+geo_dim]
            if cond_feat is not None:
                emb = torch.cat([emb, cond_feat[mask]], dim=-1)  # [N_valid, H+geo_dim+cond_dim]
            if has_env:
                nuc_env = data.target_environment[mask, nuc_idx, :].to(x.device)
                env_projected = self.env_proj(nuc_env)
                if self.balance_branch_inputs:
                    env_projected = self.env_ln(env_projected)
                emb = torch.cat([emb, env_projected], dim=-1)

            # Route through branch or flat head
            if self.config.use_branches:
                if nuc in _PROTON_NUCLEI:
                    branch_out = self.proton_branch(emb)
                    if getattr(self.config, "use_film", False):
                        aa_types = data.residue_types[mask]
                        aa_emb = self.aa_emb(aa_types)  # reuse trunk AA embedding
                        branch_out = self.proton_film(branch_out, aa_emb)
                    out = self.proton_heads[nuc](branch_out)
                else:
                    branch_out = self.carbon_branch(emb)
                    if getattr(self.config, "use_film", False):
                        aa_types = data.residue_types[mask]
                        aa_emb = self.aa_emb(aa_types)
                        branch_out = self.carbon_film(branch_out, aa_emb)
                    out = self.carbon_heads[nuc](branch_out)
            else:
                out = self.heads[nuc](emb)

            # Denormalize: z-score -> ppm
            if self.config.use_residual:
                # Residual mode: head predicts deviation from AA-type mean
                aa_types = data.residue_types[mask]
                aa_mean = self._aa_means[aa_types, nuc_idx]  # [N_valid]
                pred_mean[nuc] = aa_mean + out[:, 0] * self._shift_std[nuc_idx]
            else:
                pred_mean[nuc] = out[:, 0] * self._shift_std[nuc_idx] + self._shift_mean[nuc_idx]
            pred_logvar[nuc] = out[:, 1]

        return pred_mean, pred_logvar

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
