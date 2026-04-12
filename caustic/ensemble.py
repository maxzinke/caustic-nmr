"""Ensemble aggregation over multiple conformers."""
from __future__ import annotations

import torch
import torch.nn as nn

from caustic.config import EnsembleConfig
from caustic.features import BACKBONE_NUCLEI


class EnsembleAggregator(nn.Module):
    """Aggregate per-conformer shift predictions into a single estimate.

    Supports median, mean, and learned attention pooling.
    During training, applies conformer dropout for robustness.
    """

    def __init__(self, config: EnsembleConfig, hidden_dim: int = 256):
        super().__init__()
        self.config = config

        if config.aggregation == "attention":
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.attn = None

    def forward(
        self,
        conformer_means: list[dict[str, torch.Tensor]],
        conformer_logvars: list[dict[str, torch.Tensor]],
        conformer_embeddings: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Aggregate across conformers.

        Args:
            conformer_means: List of C dicts, each {nucleus: [R_valid]}.
            conformer_logvars: Same shape as means.
            conformer_embeddings: Optional, for attention pooling.

        Returns:
            agg_mean, agg_logvar: Aggregated predictions.
        """
        C = len(conformer_means)
        if C == 0:
            return {}, {}

        # Conformer dropout during training
        if self.training and C > self.config.min_conformers:
            n_drop = int(C * self.config.conformer_dropout)
            n_keep = max(self.config.min_conformers, C - n_drop)
            indices = torch.randperm(C)[:n_keep].tolist()
            conformer_means = [conformer_means[i] for i in indices]
            conformer_logvars = [conformer_logvars[i] for i in indices]
            if conformer_embeddings is not None:
                conformer_embeddings = [conformer_embeddings[i] for i in indices]
            C = n_keep

        agg_mean: dict[str, torch.Tensor] = {}
        agg_logvar: dict[str, torch.Tensor] = {}

        for nuc in BACKBONE_NUCLEI:
            # Stack across conformers: [C, R_valid]
            stacked = []
            for cm in conformer_means:
                if nuc in cm and cm[nuc].numel() > 0:
                    stacked.append(cm[nuc])
            if not stacked:
                continue

            # All conformers must have same number of valid residues
            n_res = stacked[0].shape[0]
            stack = torch.stack(stacked, dim=0)  # [C, R]

            if self.config.aggregation == "median":
                agg_mean[nuc] = stack.median(dim=0).values
                # Uncertainty from conformer spread (MAD)
                mad = (stack - agg_mean[nuc].unsqueeze(0)).abs().median(dim=0).values
                agg_logvar[nuc] = (mad.clamp(min=1e-4) * 1.4826).log() * 2  # MAD → σ → logvar

            elif self.config.aggregation == "mean":
                agg_mean[nuc] = stack.mean(dim=0)
                agg_logvar[nuc] = stack.var(dim=0).clamp(min=1e-6).log()

            elif self.config.aggregation == "attention" and self.attn is not None:
                # Use embeddings if available, else fall back to mean
                if conformer_embeddings is not None:
                    emb_stack = []
                    for ce in conformer_embeddings:
                        if nuc in ce and ce[nuc].numel() > 0:
                            emb_stack.append(ce[nuc])
                    if emb_stack:
                        embs = torch.stack(emb_stack, dim=0)  # [C, R, H]
                        weights = self.attn(embs).squeeze(-1)  # [C, R]
                        weights = torch.softmax(weights, dim=0)
                        agg_mean[nuc] = (stack * weights).sum(dim=0)
                        var = (weights * (stack - agg_mean[nuc].unsqueeze(0)) ** 2).sum(dim=0)
                        agg_logvar[nuc] = var.clamp(min=1e-6).log()
                    else:
                        agg_mean[nuc] = stack.mean(dim=0)
                        agg_logvar[nuc] = stack.var(dim=0).clamp(min=1e-6).log()
                else:
                    agg_mean[nuc] = stack.mean(dim=0)
                    agg_logvar[nuc] = stack.var(dim=0).clamp(min=1e-6).log()

        return agg_mean, agg_logvar
