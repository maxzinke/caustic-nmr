"""Sanity checks on the reference shifts in ``data/truth_test.csv``.

Writes ``results/truth_sanity.md`` and per-nucleus histograms under
``results/figures/truth_hist_<nucleus>.png``. Checks: value ranges against
physically plausible windows, duplicates, GLY CB (must not exist), PRO H (must
not exist), and per-protein referencing outliers (median offset from the
residue-type mean of the whole test set > 2 ppm for carbons/N, > 0.5 ppm for
protons — the signature of a mis-referenced deposition).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIG = RESULTS / "figures"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
PLAUSIBLE = {"H": (5.0, 12.0), "HA": (1.5, 7.0), "N": (95.0, 140.0), "CA": (40.0, 70.0), "CB": (12.0, 75.0), "C": (165.0, 185.0)}
OFFSET_TOL = {"H": 0.5, "HA": 0.5, "N": 2.0, "CA": 2.0, "CB": 2.0, "C": 2.0}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(csv.DictReader(open(DATA / "truth_test.csv", newline="")))
    by_nuc = defaultdict(list)
    seen, dup = set(), 0
    gly_cb = pro_h = 0
    by_nuc_aa = defaultdict(list)
    per_protein = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["bmrb_id"], r["seq_id"], r["nucleus"])
        if key in seen:
            dup += 1
        seen.add(key)
        v = float(r["value"])
        by_nuc[r["nucleus"]].append(v)
        by_nuc_aa[(r["nucleus"], r["comp_id"])].append(v)
        per_protein[r["bmrb_id"]][r["nucleus"]].append((r["comp_id"], v))
        if r["nucleus"] == "CB" and r["comp_id"] == "GLY":
            gly_cb += 1
        if r["nucleus"] == "H" and r["comp_id"] == "PRO":
            pro_h += 1
    aa_mean = {k: float(np.mean(v)) for k, v in by_nuc_aa.items() if len(v) >= 20}

    L = ["# Reference-shift sanity report (test split)\n",
         f"{len(rows):,} labels, {len(per_protein)} proteins, {dup} duplicate (entry, seq_id, nucleus) keys, "
         f"{gly_cb} GLY CB labels, {pro_h} PRO H labels.\n",
         "| Nucleus | n | min | p0.1 | median | p99.9 | max | outside plausible window | window |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for n in NUCLEI:
        v = np.array(by_nuc[n])
        lo, hi = PLAUSIBLE[n]
        out = int(((v < lo) | (v > hi)).sum())
        L.append(f"| {n} | {len(v):,} | {v.min():.2f} | {np.percentile(v, 0.1):.2f} | {np.median(v):.2f} | "
                 f"{np.percentile(v, 99.9):.2f} | {v.max():.2f} | {out} ({100 * out / len(v):.3f}%) | {lo}–{hi} |")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(v, bins=200, color="#2f5d8a")
        ax.set_title(f"{n} reference shifts, test split (n={len(v):,})")
        ax.set_xlabel("ppm")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(FIG / f"truth_hist_{n}.png", dpi=110)
        plt.close(fig)

    # referencing outliers
    L.append("\n## Per-protein referencing check\n")
    L.append("Median (value − residue-type mean) per protein and nucleus; flagged when |offset| exceeds "
             "0.5 ppm (H/HA) or 2 ppm (N/CA/CB/C) with ≥ 10 labels.\n")
    L.append("| Nucleus | proteins checked | flagged | worst (bmrb_id, offset ppm) |")
    L.append("|---|---:|---:|---|")
    flagged_all = {}
    for n in NUCLEI:
        flagged, worst = [], (None, 0.0)
        checked = 0
        for eid, d in per_protein.items():
            vals = [v - aa_mean[(n, aa)] for aa, v in d.get(n, []) if (n, aa) in aa_mean]
            if len(vals) < 10:
                continue
            checked += 1
            off = float(np.median(vals))
            if abs(off) > OFFSET_TOL[n]:
                flagged.append((eid, off))
            if abs(off) > abs(worst[1]):
                worst = (eid, off)
        flagged_all[n] = flagged
        L.append(f"| {n} | {checked} | {len(flagged)} | {worst[0]}, {worst[1]:+.2f} |")
    L.append("\nFlagged entries (bmrb_id: nucleus offset):\n")
    agg = defaultdict(dict)
    for n, fl in flagged_all.items():
        for eid, off in fl:
            agg[eid][n] = off
    for eid in sorted(agg, key=int):
        L.append(f"- {eid}: " + ", ".join(f"{n} {o:+.2f}" for n, o in agg[eid].items()))
    L.append(f"\n{len(agg)} of {len(per_protein)} proteins flagged on at least one nucleus.")
    (RESULTS / "truth_sanity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"... flagged proteins: {len(agg)}; report -> {RESULTS / 'truth_sanity.md'}")


if __name__ == "__main__":
    main()
