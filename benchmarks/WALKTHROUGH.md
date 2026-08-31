# One protein, end to end

BMRB **18812** — DUF3349 domain (PDB **2M0N**, solution NMR ensemble), one of the 735
held-out test entries. Everything below is reproducible from a plain checkout plus
`pip install caustic-nmr`.

## 1. Predict

```bash
# the structure the benchmark used (any mmCIF/PDB of 2M0N works the same way)
caustic 2m0n.cif --format csv -o 18812_shifts.csv
```

Or through the API, exactly as the benchmark driver does (`run_caustic.py`):

```python
from caustic import predict_shifts_onnx
from importlib.resources import files

res = predict_shifts_onnx("2m0n.cif", str(files("caustic.data") / "best_v2_carbons.onnx"))
```

All conformers of the NMR ensemble are used (median aggregation); the output carries the
package version and model SHA-256 in its header.

## 2. Compare with the deposited shifts

The reference values come from the BMRB entry itself and live, joined to every method's
prediction, in `results/per_residue.csv.gz` (534 reference shifts for this entry). First
CA rows:

| seq | res | reference | CAUSTIC | σ | SPARTA+ | LEGOLAS |
|---:|---|---:|---:|---:|---:|---:|
| 10 | GLY | 45.30 | 45.23 | 0.38 | 45.48 | 46.87 |
| 11 | THR | 62.00 | 61.92 | 0.40 | 61.79 | 61.55 |
| 12 | LEU | 55.40 | 55.32 | 0.41 | 54.42 | 54.53 |
| 13 | GLU | 56.90 | 56.61 | 0.44 | 55.93 | 56.78 |
| 14 | ALA | 52.60 | 52.66 | 0.42 | 51.98 | 52.37 |

## 3. Score it

```python
import gzip, pandas as pd
df = pd.read_csv(gzip.open("results/per_residue.csv.gz", "rt"), comment="#", dtype={"bmrb_id": str})
d = df[df.bmrb_id == "18812"]
for m in ["caustic", "sparta", "legolas"]:
    sub = d.dropna(subset=["truth", m])
    print(m, (sub[m] - sub.truth).abs().groupby(sub.nucleus).mean().round(3).to_dict())
```

MAE (ppm) on the 534 shifts of this one protein, all three methods on the same residues:

| Method | H | HA | N | CA | CB | C |
|---|---:|---:|---:|---:|---:|---:|
| CAUSTIC | **0.232** | **0.151** | **1.168** | **0.829** | **0.941** | **0.760** |
| SPARTA+ | 0.378 | 0.194 | 2.221 | 0.886 | 1.263 | 0.846 |
| LEGOLAS | 0.525 | 0.207 | 2.711 | 0.999 | 1.392 | 0.933 |

One protein proves nothing on its own — the aggregate tables with confidence intervals
are in [results/tables.md](results/tables.md) and [docs/BENCHMARKS.md](../docs/BENCHMARKS.md);
`python rescore.py --bootstrap 2000` regenerates them.
