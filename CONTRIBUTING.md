# Contributing

Thanks for your interest. Bug reports, reproducible benchmark discrepancies and
documentation fixes are the most useful contributions.

## Development install

```bash
git clone https://github.com/maxzinke/caustic-nmr.git
cd caustic-nmr
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

PyTorch is a hard dependency (graph construction); `onnxruntime` runs the
model. The model weights and calibrator ship inside the package — no download.

## Checks to run before opening a pull request

```bash
ruff check caustic tests          # lint (must be clean)
pytest -q                         # tests (~1 min; runs the CLI on examples/1ubq.pdb)
python -m build && twine check dist/*
```

CI runs the same on Python 3.10–3.13 × Ubuntu/Windows.

## Ground rules

- **Do not change predictions silently.** `tests/test_api.py` pins reference
  values for 1UBQ; if a change moves them, say why in the PR and update
  `docs/BENCHMARKS.md` by re-running `benchmarks/` (see `benchmarks/README.md`).
- The bundled `caustic/data/*.onnx` and `*.json` are release artifacts: replacing
  them requires updating `tests/test_assets.py` hashes, `LICENSE-WEIGHTS`,
  `CHANGELOG.md` and the benchmark record together.
- Every output format must keep its provenance stamp (`caustic/provenance.py`).
- Conventional-commit subjects (`fix(cli): …`, `docs: …`); no attribution trailers.

## Where things live

| What | Where |
|---|---|
| Inference API | `caustic/inference.py` |
| Graph / feature pipeline | `caustic/graph.py`, `caustic/features.py`, `caustic/physics_features.py` |
| Network definition (for reference / `--backend torch`) | `caustic/model.py` |
| ONNX export + session | `caustic/export.py` |
| Post-prediction calibration | `caustic/calibrate.py` |
| Writers | `caustic/io/` |
| Method / data / benchmark / limitations | `docs/` |
| Benchmark inputs, results, rescoring | `benchmarks/` |

## Reporting a benchmark discrepancy

Open an issue with the structure file (or its PDB/AlphaFold id), the exact
command, the `# caustic-nmr …` provenance line from the output, and the
reference shifts you compared against (BMRB id if applicable).
