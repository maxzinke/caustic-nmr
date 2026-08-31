"""Regenerate every benchmark table from ``results/per_residue.csv.gz``.

Metric definitions (the only ones used anywhere in the repository):

* **MAE** for method *m* on nucleus *n* = mean over residues of |pred − truth|.
* **Paired set** for a CAUSTIC-vs-*m* comparison = residues with a reference
  value AND a prediction from both CAUSTIC and *m* (per nucleus). Both MAEs in
  a paired row are computed on exactly the same residues. The "all methods"
  table uses the residues every listed method predicted. The "unpaired"
  table scores each method on whatever it predicted — it is reported only to
  show coverage and is never used for a claim.
* **Per-protein composite** (the definition used for the internal record,
  ``noft/scripts/phase5_final_pipeline_bootstrap.py::composite_per_protein``):
  per protein, MAE per nucleus, then the weighted mean over the nuclei the
  protein has, weights H 1, HA 1, N 1, CA 1.5, CB 2, C 1; the headline
  composite is the mean of that over proteins.
* **Bootstrap CI**: proteins are resampled with replacement (B draws, fixed
  seed); every statistic is recomputed on the resample; the 2.5/97.5
  percentiles are reported. Residue-level MAE differences are bootstrapped
  the same way (resampling proteins, not residues, so correlated residues
  within a protein do not shrink the interval).
* **Sign test**: fraction of proteins whose composite is lower with CAUSTIC
  than with *m*, with the two-sided exact binomial p-value.

Slices: ``full`` = every test label; ``cleaned`` = entries not on the
whole-entry blocklist and labels not on the per-label blocklist (the slice
the internal −4.37 % record was measured on).

    python rescore.py                    # summary.json + tables.md, no CIs
    python rescore.py --bootstrap 2000   # with CIs (default seed 42)
    python rescore.py --check            # recompute and diff against summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NUCLEI = ["H", "HA", "N", "CA", "CB", "C"]
WEIGHTS = {"H": 1.0, "HA": 1.0, "N": 1.0, "CA": 1.5, "CB": 2.0, "C": 1.0}
METHODS = ["caustic", "sparta", "legolas", "ucbshift", "ucbshift_x", "caustic_fullcal", "caustic_recordgraph"]
LABEL = {"caustic": "CAUSTIC", "sparta": "SPARTA+", "legolas": "LEGOLAS", "ucbshift": "UCBShift2",
         "ucbshift_x": "UCBShift-X (ML only)", "caustic_fullcal": "CAUSTIC + record stratum calibrator",
         "caustic_recordgraph": "CAUSTIC on production graphs (medoid)"}


def load_table() -> pd.DataFrame:
    p = RESULTS / "per_residue.csv.gz"
    with gzip.open(p, "rt") as f:
        df = pd.read_csv(f, comment="#", dtype={"bmrb_id": str})
    for m in METHODS + ["truth", "caustic_sigma"]:
        if m in df:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def apply_slice(df: pd.DataFrame, slice_: str) -> pd.DataFrame:
    if slice_ == "cleaned":
        return df[(df.in_cleaned_set == 1) & (df.label_dropped == 0)]
    return df


def per_protein_composite(sub: pd.DataFrame, method: str) -> pd.Series:
    err = (sub[method] - sub.truth).abs()
    d = sub.assign(err=err).dropna(subset=["err"])
    pn = d.groupby(["bmrb_id", "nucleus"]).err.mean().unstack()
    w = pd.Series({n: WEIGHTS[n] for n in pn.columns})
    num = (pn * w).sum(axis=1, min_count=1)
    den = pn.notna().mul(w, axis=1).sum(axis=1)
    return num / den


def boot_ci(values_by_protein: dict[str, np.ndarray], stat, B: int, seed: int) -> tuple[float, float]:
    """Resample proteins; stat(list_of_protein_arrays) -> float."""
    ids = list(values_by_protein)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        idx = rng.integers(0, len(ids), len(ids))
        out.append(stat([values_by_protein[ids[i]] for i in idx]))
    s = np.sort(out)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def residue_mae_stats(sub: pd.DataFrame, a: str, b: str, B: int, seed: int) -> dict:
    """Paired residue-level MAE of a and b on the residues both predicted."""
    d = sub.dropna(subset=["truth", a, b])
    if d.empty:
        return {"n_residues": 0}
    ea = (d[a] - d.truth).abs().to_numpy()
    eb = (d[b] - d.truth).abs().to_numpy()
    res = {"n_residues": int(len(d)), "n_proteins": int(d.bmrb_id.nunique()),
           f"mae_{a}": float(ea.mean()), f"mae_{b}": float(eb.mean()),
           "delta": float(ea.mean() - eb.mean())}
    res["delta_rel_pct"] = 100.0 * res["delta"] / res[f"mae_{b}"]
    if B:
        diff = ea - eb
        groups = {k: v.to_numpy() for k, v in pd.Series(diff).groupby(d.bmrb_id.to_numpy())}
        lo, hi = boot_ci(groups, lambda arrs: float(np.concatenate(arrs).mean()), B, seed)
        res["delta_ci95"] = [lo, hi]
        res["ci_excludes_zero"] = bool(hi < 0 or lo > 0)
    return res


def composite_stats(sub: pd.DataFrame, a: str, b: str, B: int, seed: int) -> dict:
    ca, cb = per_protein_composite(sub, a), per_protein_composite(sub, b)
    common = ca.index.intersection(cb.index)
    ca, cb = ca.loc[common], cb.loc[common]
    if len(common) == 0:
        return {"n_proteins": 0}
    delta = (ca - cb)
    n_better = int((delta < 0).sum())
    from scipy.stats import binomtest
    p = binomtest(n_better, len(common), 0.5).pvalue if len(common) else float("nan")
    res = {"n_proteins": int(len(common)), f"composite_{a}": float(ca.mean()), f"composite_{b}": float(cb.mean()),
           "delta": float(delta.mean()), "delta_rel_pct": float(100 * delta.mean() / cb.mean()),
           "n_proteins_better": n_better, "frac_better": n_better / len(common), "sign_test_p": float(p)}
    if B:
        groups = {k: np.array([v]) for k, v in delta.items()}
        lo, hi = boot_ci(groups, lambda arrs: float(np.concatenate(arrs).mean()), B, seed)
        res["delta_ci95"] = [lo, hi]
        res["ci_excludes_zero"] = bool(hi < 0 or lo > 0)
    return res


def compute(df: pd.DataFrame, slice_: str, B: int, seed: int) -> dict:
    sub = apply_slice(df, slice_)
    present = [m for m in METHODS if m in sub and sub[m].notna().any()]
    out = {"slice": slice_, "n_truth_labels": int(sub.truth.notna().sum()), "n_proteins": int(sub.bmrb_id.nunique()),
           "methods": present, "bootstrap_B": B, "seed": seed,
           "coverage": {}, "unpaired_mae": {}, "paired_vs_caustic": {}, "all_methods_common": {},
           "composite_vs_caustic": {}}
    for m in present:
        d = sub.dropna(subset=["truth", m])
        out["coverage"][m] = {"n_residues": int(len(d)), "n_proteins": int(d.bmrb_id.nunique()),
                              "frac_of_truth": float(len(d) / max(1, out["n_truth_labels"]))}
        out["unpaired_mae"][m] = {n: float((g[m] - g.truth).abs().mean()) for n, g in d.groupby("nucleus")}
    for m in present:
        if m == "caustic":
            continue
        out["paired_vs_caustic"][m] = {n: residue_mae_stats(sub[sub.nucleus == n], "caustic", m, B, seed)
                                       for n in NUCLEI}
        out["composite_vs_caustic"][m] = composite_stats(sub, "caustic", m, B, seed)
    core = [m for m in present if m not in ("ucbshift_x", "caustic_fullcal", "caustic_recordgraph")]
    common = sub.dropna(subset=["truth"] + core)
    out["all_methods_common"] = {"methods": core, "n_residues": int(len(common)),
                                 "n_proteins": int(common.bmrb_id.nunique()),
                                 "mae": {n: {m: float((g[m] - g.truth).abs().mean()) for m in core}
                                         for n, g in common.groupby("nucleus")},
                                 "n_per_nucleus": {n: int(len(g)) for n, g in common.groupby("nucleus")}}
    return out


def fmt_ci(d: dict) -> str:
    if "delta_ci95" not in d:
        return ""
    lo, hi = d["delta_ci95"]
    return f" [{lo:+.3f}, {hi:+.3f}]"


def markdown(summary: dict) -> str:
    L = []
    for slice_, s in summary["slices"].items():
        L.append(f"## Slice `{slice_}` — {s['n_proteins']} proteins, {s['n_truth_labels']:,} reference shifts\n")
        L.append("### Paired per-nucleus MAE (ppm), CAUSTIC vs each method on the residues both predicted\n")
        L.append("| Method | Nucleus | n res. | n prot. | CAUSTIC | Method | Δ (CAUSTIC − method) | rel. |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for m, per_n in s["paired_vs_caustic"].items():
            for n in NUCLEI:
                d = per_n[n]
                if not d.get("n_residues"):
                    continue
                L.append(f"| {LABEL[m]} | {n} | {d['n_residues']:,} | {d['n_proteins']} | {d['mae_caustic']:.3f} | "
                         f"{d[f'mae_{m}']:.3f} | {d['delta']:+.3f}{fmt_ci(d)} | {d['delta_rel_pct']:+.1f}% |")
        L.append("\n### Per-protein weighted composite (record definition), paired by protein\n")
        L.append("| Method | n prot. | CAUSTIC | Method | Δ | rel. | proteins better | sign-test p |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m, d in s["composite_vs_caustic"].items():
            if not d.get("n_proteins"):
                continue
            L.append(f"| {LABEL[m]} | {d['n_proteins']} | {d['composite_caustic']:.4f} | {d[f'composite_{m}']:.4f} | "
                     f"{d['delta']:+.4f}{fmt_ci(d)} | {d['delta_rel_pct']:+.2f}% | "
                     f"{d['n_proteins_better']}/{d['n_proteins']} ({100 * d['frac_better']:.1f}%) | {d['sign_test_p']:.2g} |")
        c = s["all_methods_common"]
        L.append(f"\n### All methods on the common residue set ({c['n_residues']:,} residues, {c['n_proteins']} proteins)\n")
        L.append("| Nucleus | n | " + " | ".join(LABEL[m] for m in c["methods"]) + " |")
        L.append("|---|---:|" + "---:|" * len(c["methods"]))
        for n in NUCLEI:
            if n in c["mae"]:
                L.append(f"| {n} | {c['n_per_nucleus'][n]:,} | " + " | ".join(f"{c['mae'][n][m]:.3f}" for m in c["methods"]) + " |")
        L.append("\n### Coverage and unpaired MAE (each method on its own predictions — not comparable across rows)\n")
        L.append("| Method | residues | proteins | " + " | ".join(NUCLEI) + " |")
        L.append("|---|---:|---:|" + "---:|" * len(NUCLEI))
        for m, cov in s["coverage"].items():
            um = s["unpaired_mae"][m]
            L.append(f"| {LABEL[m]} | {cov['n_residues']:,} | {cov['n_proteins']} | " +
                     " | ".join(f"{um[n]:.3f}" if n in um else "—" for n in NUCLEI) + " |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0, help="B resamples (0 = no CIs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slices", default="full,cleaned")
    ap.add_argument("--check", action="store_true", help="recompute and compare with results/summary.json")
    ap.add_argument("--markdown", action="store_true", help="also write results/tables.md")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_table()
    summary = {"slices": {s: compute(df, s, a.bootstrap, a.seed) for s in a.slices.split(",")}}
    if a.check:
        ref = json.load(open(RESULTS / "summary.json"))
        drift = []

        def walk(x, y, path=""):
            if isinstance(x, dict):
                for k in x:
                    if k in ("bootstrap_B", "delta_ci95", "ci_excludes_zero"):
                        continue
                    if k not in y:
                        drift.append(f"{path}/{k} missing")
                    else:
                        walk(x[k], y[k], f"{path}/{k}")
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if not (math.isnan(x) and math.isnan(y)) and abs(x - y) > 1e-6:
                    drift.append(f"{path}: {x} != {y}")
            elif x != y:
                drift.append(f"{path}: {x} != {y}")
        walk(ref["slices"], summary["slices"])
        print("\n".join(drift) if drift else "OK: summary.json reproduces")
        return 1 if drift else 0
    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1)
    if a.markdown or True:
        (RESULTS / "tables.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
