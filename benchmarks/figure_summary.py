"""Summary figure: per-nucleus MAE of every method on the common residue set.

Reads ``results/per_residue.csv.gz`` (full slice, residues all four methods predicted)
and writes ``results/figures/benchmark_summary.png`` — the figure the README embeds.
Deliberately simple: grouped bars, one group per nucleus, value labels, n per nucleus.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
METHODS = [("caustic", "CAUSTIC", "#2f5d8a"), ("sparta", "SPARTA+", "#8a8378"),
           ("ucbshift", "UCBShift2", "#b0885a"), ("legolas", "LEGOLAS", "#a3a8b8")]


def main() -> None:
    df = pd.read_csv(gzip.open(RESULTS / "per_residue.csv.gz", "rt"), comment="#",
                     dtype={"bmrb_id": str})
    core = [m for m, _, _ in METHODS]
    d = df.dropna(subset=["truth"] + core)
    mae = {m: {} for m in core}
    ns = {}
    for nuc, g in d.groupby("nucleus"):
        ns[nuc] = len(g)
        for m in core:
            mae[m][nuc] = float((g[m] - g.truth).abs().mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200,
                             gridspec_kw={"width_ratios": [5, 1.15]})
    for ax, nucs in zip(axes, [["H", "HA", "CA", "CB", "C"], ["N"]]):
        for j, (m, label, color) in enumerate(METHODS):
            xs = [i + (j - 1.5) * 0.19 for i in range(len(nucs))]
            ys = [mae[m][n] for n in nucs]
            ax.bar(xs, ys, width=0.18, color=color, label=label if ax is axes[0] else None)
        ax.set_xticks(range(len(nucs)))
        ax.set_xticklabels([f"{n}\nn={ns[n]:,}" for n in nucs], fontsize=8)
        ax.set_ylabel("MAE (ppm)" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(True)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, ncols=4, loc="upper left")
    n_prot = d.bmrb_id.nunique()
    fig.suptitle(f"Backbone chemical-shift MAE — {n_prot} held-out proteins, "
                 f"common residues of all four methods", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / "figures" / "benchmark_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
