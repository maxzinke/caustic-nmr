# Changelog

All notable changes to `caustic-nmr` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] — 2026-08-31

Release-readiness version: same model weights and calibrator as 0.3.0
(ONNX SHA-256 `ebc7bbc2…2948b2`, calibrator `sa16_v2_carbons_slim`), so
predictions are unchanged; everything around them is fixed or new.

### Fixed
- `caustic <structure>` failed on a fresh install because the CLI never looked
  at the model bundled inside the package (it searched `<pkg>/weights/best.onnx`,
  `$CAUSTIC_WEIGHTS` and `~/.caustic/best.onnx` only). The bundled model is now
  resolved first; the two overrides remain.
- The "weights not found" message named a literal `<HF_HUB_URL>` placeholder.
- 16 `NameError`-class defects in `graph.py` (`torch` used in annotations
  without being importable at check time) and the remaining lint findings
  (unused imports/variables, an ambiguous variable name in `potenci.py`).
- `--backend` help text claimed no PyTorch was needed; PyTorch is a hard
  dependency for graph construction in both backends.
- `caustic --help` raised `UnicodeEncodeError` when stdout was redirected under
  an OEM console codepage (cp437/cp850), because the help strings contained em
  dashes. The help text is now ASCII-only.
- The sdist shipped `tests/` but not `examples/`, so the test suite could not run
  inside an unpacked sdist; the example structures are now included.
- `benchmarks/check_leakage.py` rewrote the committed leakage report on every run,
  silently degrading it when the competitor reference databases were absent.
  Writing now requires `--write`.

### Added
- Provenance stamp in every output: package version, model file and SHA-256,
  calibrator version and ISO-8601 UTC date — as `#` comment lines in NEF /
  NMR-STAR / CSV, as a `provenance` object in JSON, and in
  `_nef_nmr_meta_data.program_version` / `creation_date` and
  `_Assigned_chem_shift_list.Program_version`.
- `caustic --version` prints the package version, the model file with its
  SHA-256, and the calibrator version.
- `ShiftPrediction.provenance` field; `caustic.provenance` module.
- Test suite (`tests/`): CLI end-to-end on 1UBQ in all four formats, Python
  API regression values, bundled-asset hashes and parameter count, import smoke.
- CI (GitHub Actions): Python 3.10–3.13 × Ubuntu/Windows; lint, tests, wheel
  build, `twine check`; benchmark regeneration + leakage check job.
- `examples/` with 1UBQ (PDB) and an AlphaFold model, plus regenerated outputs.
- `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `docs/METHOD.md`, `docs/DATA.md`, `docs/BENCHMARKS.md`, `docs/LIMITATIONS.md`
  and a `benchmarks/` directory with the benchmark inputs, per-residue results
  and rescoring script.

### Changed
- Licence split: `LICENSE` is now the pure MIT text (so licence detection
  works); the bundled ONNX weights and calibrator are CC BY 4.0 under
  `LICENSE-WEIGHTS` (earlier copies remain MIT).
- Package metadata: full author name and e-mail, `license-files`, per-version
  Python classifiers, Repository / Issues / Changelog / Web app / Model weights
  URLs, `dev` extra; project moved to `github.com/maxzinke/caustic-nmr`, branch
  `main`.
- NMR-STAR default entry id `crystalline_predicted` → `caustic_predicted`.
- `ruff` is upper-bounded in the `dev` extra so a new linter release cannot turn
  CI red without a repository change.
- README rewritten; the previous README listed two features that have no
  implementation in the package ("pLDDT-calibrated sigma widening" and
  "isotonic CDF calibration") — both removed from the feature list. The
  competitor comparison is being re-run on the production test split through
  the public package before any number is quoted again (see
  `docs/BENCHMARKS.md`).

### Removed
- Stale `space/` copy of the Hugging Face demo (v0.1.0 code); the live Space
  installs this package from PyPI instead.

### Known issues
- **The PyTorch backend cannot load the production model.** `caustic/model.py`
  (`ShiftPredictor`) has no `temp_proj` / `per_atom_rsasa_proj` layers, although
  the shipped ONNX network and its training configuration (`use_temperature`,
  `use_per_atom_rsasa`) have them. `predict_shifts()` / `caustic --backend torch`
  therefore fail with missing state-dict keys on the production `.pt`; the error
  now says so explicitly. The ONNX path (`predict_shifts_onnx()`, the CLI
  default) is the supported one. See `docs/LIMITATIONS.md`.

## [0.3.0] — 2026-05-23

### Changed
- Production model switched to `D31_clean_safe_v2_carbons`: PaiNN backbone
  (~741K parameters) trained on carbon-aggressive label-noise-cleaned BMRB
  labels; ONNX export `best_v2_carbons.onnx` bundled in the wheel.
- Feature pipeline synchronised for the v0.3.0 ONNX inputs (`pos`,
  `node_normal`, `is_hbond`, `hb_cos`).

### Added
- Slim SA16 post-prediction calibration (global per-nucleus offsets + CYS-CB
  oxidation-state modifier with Sγ–Sγ disulfide gate), on by default.
- Fallback `ProteinData` container when `torch_geometric` is not installed.
- Distribution renamed to `caustic-nmr` (2026-08-07); `LICENSE` and `.mailmap`
  added.

### Fixed
- Function-level relative imports in the synchronised modules.

## [0.1.0] — 2026-04-12

### Added
- Initial release: SchNet-style GNN (D14 checkpoint, ~404K parameters),
  ONNX Runtime inference, NEF / NMR-STAR / CSV / JSON writers, CLI, and the
  Gradio demo for Hugging Face Spaces. Renamed to CAUSTIC throughout.

[Unreleased]: https://github.com/maxzinke/caustic-nmr/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/maxzinke/caustic-nmr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/maxzinke/caustic-nmr/releases/tag/v0.3.0
[0.1.0]: https://github.com/maxzinke/caustic-nmr/commit/51af307
