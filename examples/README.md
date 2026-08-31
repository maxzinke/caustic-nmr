# Examples

Two input structures and the outputs the released build produces for the first one.

| File | What | Source |
|---|---|---|
| `1ubq.pdb` | Human ubiquitin, X-ray 1.8 Å, 76 residues, no hydrogens (the CLI synthesises backbone H/HA) | https://files.rcsb.org/download/1UBQ.pdb |
| `AF-P01112-F1-model_v6.cif` | AlphaFold model of human HRAS (UniProt P01112), 189 residues, pLDDT in the B-factor column | https://alphafold.ebi.ac.uk/entry/P01112 |
| `1ubq_shifts.nef` | Output of `caustic 1ubq.pdb -o 1ubq_shifts.nef` (NEF 1.1) | regenerated with this release |
| `1ubq_shifts.csv` | Output of `caustic 1ubq.pdb --format csv -o 1ubq_shifts.csv` | regenerated with this release |

```bash
caustic --version                                   # package, model SHA-256, calibrator
caustic 1ubq.pdb -o 1ubq_shifts.nef                 # NEF (default)
caustic 1ubq.pdb --format csv -o 1ubq_shifts.csv    # CSV: two '#' provenance lines, then a header
caustic AF-P01112-F1-model_v6.cif --format json -o hras.json
```

Every output starts with (or, for JSON, contains) a provenance stamp such as

```
# caustic-nmr 0.4.0 model=best_v2_carbons.onnx sha256=ebc7bbc2fc59 calibrator=sa16_v2_carbons_slim date=2026-08-30T16:40:12Z
```

Read the CSV with `pandas.read_csv("1ubq_shifts.csv", comment="#")`.
