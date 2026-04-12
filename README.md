# CAUSTIC

**C**onformation-**A**ware **U**ncertainty and **S**hift predic**T**ion from prote**I**n **C**onformer ensembles

Predict NMR backbone chemical shifts (H, HA, N, CA, CB, C') from any PDB, mmCIF, or AlphaFold protein structure.

A 400K-parameter SchNet graph neural network trained on 4,834 BMRB-linked experimental structures. Beats SPARTA+, LEGOLAS, and UCBShift2 on all 6 backbone nuclei on a 712-protein homology-separated test set.

## Quick start

```bash
pip install caustic

# Predict shifts from a PDB file
caustic myprotein.pdb

# Output as CSV instead of NEF
caustic myprotein.pdb --format csv -o shifts.csv

# AlphaFold model — pLDDT-calibrated uncertainties applied automatically
caustic AF-P12345-F1-model_v4.cif
```

## Python API

```python
from caustic import predict_shifts_onnx

result = predict_shifts_onnx("myprotein.pdb", "path/to/best.onnx")

# Per-residue arrays
print(result.mean["CA"])   # CA shifts in ppm
print(result.std["CA"])    # calibrated sigma
print(result.residue_names)  # ['MET', 'VAL', 'LEU', ...]
```

## Output formats

| Format | Flag | Use case |
|--------|------|----------|
| **NEF** (default) | `--format nef` | CCPN Analysis v3, modern NMR software |
| NMR-STAR 3.x | `--format star` | BMRB deposition |
| CSV | `--format csv` | pandas, Excel |
| JSON | `--format json` | programmatic consumption |

## Features

- **AlphaFold-aware**: auto-detects AF models, applies pLDDT-calibrated sigma widening
- **Missing hydrogens**: synthesises backbone H/HA from geometry when absent (X-ray PDBs, AF models)
- **Non-standard residues**: MSE, HYP, SEP, TPO, PTR, CSO, PCA, and 20+ more mapped to canonical parents
- **NMR ensembles**: median aggregation across conformers
- **Calibrated uncertainties**: isotonic CDF calibration on experimental structures, per-pLDDT scaling on AF inputs
- **NEF naming conventions**: GLY HA emitted as `HA%` (degenerate wildcard) per NEF v1.1 spec

## License

MIT
