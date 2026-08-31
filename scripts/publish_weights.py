"""Mirror the bundled CAUSTIC weights to the HuggingFace Hub.

The package ships its weights inside the wheel (``caustic/data/best_v2_carbons.onnx``,
3 MB, and ``sa16_calibrator_v2.json``), so nothing is downloaded at run time. The Hub
repository is the *citable* home of the same bytes: it carries the model card, a
DataCite DOI, and a tag per release, so a paper can point at exactly the weights a
version of the package used.

WHAT IT CHECKS BEFORE UPLOADING
-------------------------------
Every file is hashed and compared against ``scripts/weights_manifest.json``. A mismatch
aborts, because the manifest is what ``tests/test_assets.py`` and this script agree on:
if you rebuilt a file, update the manifest first and let the mismatch be the reminder.

The upload is tagged (``v<package version>``), not left on a moving branch.

USAGE
-----
    hf auth login                          # needs a WRITE token
    python scripts/publish_weights.py --dry-run
    python scripts/publish_weights.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "caustic" / "data"
MANIFEST = REPO_ROOT / "scripts" / "weights_manifest.json"
HF_REPO = "SiXa18/caustic-weights"


def _package_version() -> str:
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)["project"]
    if "version" in project:
        return project["version"]
    # dynamic version: read caustic/_version.py without importing torch-heavy modules
    import re

    init = (REPO_ROOT / "caustic" / "_version.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_card(version: str, manifest: dict) -> str:
    rows = "\n".join(
        f"| `{name}` | {entry['size']:,} | `{entry['sha256']}` |"
        for name, entry in manifest["files"].items()
    )
    return f"""---
license: cc-by-4.0
library_name: onnx
pipeline_tag: other
tags:
  - protein-nmr
  - chemical-shifts
  - graph-neural-network
  - painn
  - onnx
---

# CAUSTIC model weights

Model parameters and the post-prediction calibrator for
[CAUSTIC](https://github.com/maxzinke/caustic-nmr), which predicts protein backbone
chemical shifts (H, HA, N, CA, CB, C') from a 3D structure.

**You do not need to download these by hand.** The same bytes are bundled inside the
`caustic-nmr` Python package (`pip install caustic-nmr=={version}`). This repository is
the citable, tagged, DOI-carrying home of the exact files a package version uses.

| File | Bytes | SHA-256 |
|---|---|---|
{rows}

Tag `v{version}` of this repository corresponds to `caustic-nmr` {version}.

## What the files are

- `best_v2_carbons.onnx` — PaiNN equivariant graph neural network (741,024 parameters,
  ONNX opset 17) trained on BMRB-linked experimental structures with carbon-aggressive
  label-noise cleaning. Architecture, features and training recipe:
  [docs/METHOD.md](https://github.com/maxzinke/caustic-nmr/blob/main/docs/METHOD.md).
- `sa16_calibrator_v2.json` — per-nucleus global offsets and cysteine CB modifiers applied
  after prediction (the "slim" SA16 v2 calibrator).

Training data, split protocol and licences:
[docs/DATA.md](https://github.com/maxzinke/caustic-nmr/blob/main/docs/DATA.md).
Benchmark protocol and numbers:
[docs/BENCHMARKS.md](https://github.com/maxzinke/caustic-nmr/blob/main/docs/BENCHMARKS.md).

## Licence

These files are released under **CC BY 4.0** (see
[LICENSE-WEIGHTS](https://github.com/maxzinke/caustic-nmr/blob/main/LICENSE-WEIGHTS)).
The package code is MIT. Attribution: *CAUSTIC model weights, Maximilian Zinke, 2026,
https://github.com/maxzinke/caustic-nmr*.

## How to cite

See [CITATION.cff](https://github.com/maxzinke/caustic-nmr/blob/main/CITATION.cff).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="hash and print; upload nothing")
    ap.add_argument("--write-manifest", action="store_true",
                    help="(re)write scripts/weights_manifest.json from caustic/data and exit")
    args = ap.parse_args()

    version = _package_version()
    files = {
        "best_v2_carbons.onnx": DATA_DIR / "best_v2_carbons.onnx",
        "sa16_calibrator_v2.json": DATA_DIR / "sa16_calibrator_v2.json",
    }

    if args.write_manifest:
        manifest = {
            "package_version": version,
            "hf_repo": HF_REPO,
            "hf_revision": f"v{version}",
            "files": {n: {"size": p.stat().st_size, "sha256": _sha256(p)} for n, p in files.items()},
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST}")
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for name, path in files.items():
        got = _sha256(path)
        want = manifest["files"][name]["sha256"]
        status = "OK " if got == want else "MISMATCH"
        ok &= got == want
        print(f"{status} {name}  {path.stat().st_size:,} B  {got[:16]}...")
    if not ok:
        print("hash mismatch against scripts/weights_manifest.json — aborting", file=sys.stderr)
        return 1
    if manifest["package_version"] != version:
        print(f"manifest is for {manifest['package_version']}, pyproject says {version} — "
              "run --write-manifest first", file=sys.stderr)
        return 1

    tag = f"v{version}"
    card = _model_card(version, manifest)
    if args.dry_run:
        print(f"\nwould upload {len(files)} files + README.md to {HF_REPO} and tag {tag}")
        print(card)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)
    for name, path in files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=HF_REPO,
                        repo_type="model", commit_message=f"weights for caustic-nmr {version}")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=HF_REPO, repo_type="model",
                    commit_message=f"model card for caustic-nmr {version}")
    api.upload_file(path_or_fileobj=str(REPO_ROOT / "LICENSE-WEIGHTS"), path_in_repo="LICENSE",
                    repo_id=HF_REPO, repo_type="model", commit_message="CC BY 4.0")
    api.create_tag(HF_REPO, tag=tag, repo_type="model", exist_ok=True)
    print(f"uploaded and tagged {HF_REPO}@{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
