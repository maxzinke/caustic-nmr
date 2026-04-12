"""Command-line interface for Caustic.

Usage::

    # Predict shifts and save as NEF (default)
    caustic input.pdb

    # Explicit output format and path
    caustic input.pdb -o shifts.nef --format nef

    # AlphaFold model with explicit chain
    caustic AF-P12345-F1-model_v4.cif --chain A --format csv

    # NMR ensemble — median aggregation across conformers
    caustic nmr_ensemble.pdb --ensemble median

    # Batch: predict all structures in a directory
    caustic structures/*.pdb --format json -o results/

    # Use the ONNX backend (default) or PyTorch
    caustic input.pdb --backend onnx
    caustic input.pdb --backend torch --checkpoint model.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("caustic")


def _find_default_onnx() -> Path | None:
    """Locate the bundled ONNX weights.

    Search order:
    1. ``<package_dir>/weights/best.onnx`` (pip-installed editable or wheel)
    2. ``$CAUSTIC_WEIGHTS`` environment variable
    3. ``~/.caustic/best.onnx`` (user download)
    """
    import os

    pkg = Path(__file__).resolve().parent
    for candidate in [
        pkg / "weights" / "best.onnx",
        Path(os.environ.get("CAUSTIC_WEIGHTS", "")) if os.environ.get("CAUSTIC_WEIGHTS") else None,
        Path.home() / ".caustic" / "best.onnx",
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _find_default_checkpoint() -> Path | None:
    """Locate a PyTorch checkpoint for the ``--backend torch`` path."""
    import os

    pkg = Path(__file__).resolve().parent
    for candidate in [
        pkg / "weights" / "best.pt",
        Path(os.environ.get("CAUSTIC_CHECKPOINT", "")) if os.environ.get("CAUSTIC_CHECKPOINT") else None,
        Path.home() / ".caustic" / "best.pt",
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="caustic",
        description="Predict NMR backbone chemical shifts from 3D protein structure.",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="PDB / mmCIF / .gz structure file(s). Supports glob patterns.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output file path (single input) or directory (batch). "
            "Default: <input_stem>.nef next to the input file."
        ),
    )
    parser.add_argument(
        "--format", "-f",
        choices=["nef", "star", "csv", "json"],
        default="nef",
        help="Output format (default: nef).",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Chain ID to predict. Default: first polymer chain.",
    )
    parser.add_argument(
        "--ensemble",
        choices=["median", "first"],
        default="median",
        help="NMR ensemble handling: 'median' aggregates conformers, 'first' uses model 0.",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "torch"],
        default="onnx",
        help="Inference backend (default: onnx — no PyTorch needed).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint (torch backend only).",
    )
    parser.add_argument(
        "--onnx-model",
        default=None,
        help="Path to .onnx model (overrides built-in weights).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.68,
        help="Confidence interval coverage for value_uncertainty (default: 0.68 = 1-sigma).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__import__('caustic').__version__}",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # Resolve inputs (expand globs on Windows where the shell doesn't)
    import glob as _glob

    input_paths: list[Path] = []
    for pattern in args.input:
        matches = _glob.glob(pattern)
        if matches:
            input_paths.extend(Path(m) for m in matches)
        else:
            input_paths.append(Path(pattern))

    if not input_paths:
        parser.error("No input files found.")

    # Resolve model weights
    if args.backend == "onnx":
        onnx_path = Path(args.onnx_model) if args.onnx_model else _find_default_onnx()
        if onnx_path is None or not onnx_path.exists():
            parser.error(
                "ONNX weights not found. Download them with:\n"
                "  mkdir -p ~/.caustic && curl -L -o ~/.caustic/best.onnx "
                "<HF_HUB_URL>\n"
                "Or pass --onnx-model /path/to/best.onnx"
            )
        from .export import load_onnx_session
        session = load_onnx_session(onnx_path)
    else:
        checkpoint = Path(args.checkpoint) if args.checkpoint else _find_default_checkpoint()
        if checkpoint is None or not checkpoint.exists():
            parser.error(
                "PyTorch checkpoint not found. Pass --checkpoint /path/to/best.pt"
            )
        session = None  # torch path doesn't use a session

    from .config import GraphConfig, ModelConfig
    graph_config = GraphConfig()
    model_config = ModelConfig()

    # Import the right writer
    from .io import write_csv, write_json, write_nef, write_nmrstar
    writers = {"nef": write_nef, "star": write_nmrstar, "csv": write_csv, "json": write_json}
    write_fn = writers[args.format]
    suffix_map = {"nef": ".nef", "star": ".str", "csv": ".csv", "json": ".json"}
    out_suffix = suffix_map[args.format]

    # Process each input
    use_ensemble = args.ensemble == "median"
    n_ok = 0
    for inp in input_paths:
        if not inp.exists():
            logger.error("File not found: %s", inp)
            continue

        # Determine output path
        if args.output and len(input_paths) == 1:
            out = Path(args.output)
        elif args.output:
            out = Path(args.output) / (inp.stem + out_suffix)
        else:
            out = inp.with_suffix(out_suffix)

        try:
            if args.backend == "onnx":
                from .inference import predict_shifts_onnx
                pred = predict_shifts_onnx(
                    str(inp), str(onnx_path),
                    model_config=model_config,
                    graph_config=graph_config,
                    chain_id=args.chain,
                    use_ensemble=use_ensemble,
                    session=session,
                )
            else:
                from .inference import predict_shifts
                pred = predict_shifts(
                    str(inp), str(checkpoint),
                    model_config=model_config,
                    graph_config=graph_config,
                    chain_id=args.chain,
                    use_ensemble=use_ensemble,
                )

            written = write_fn(pred, out)
            n_ok += 1
            if args.verbose or len(input_paths) == 1:
                n_res = len(pred.seq_ids)
                import numpy as np
                n_valid = {
                    nuc: int(np.sum(np.isfinite(pred.mean[nuc])))
                    for nuc in pred.mean
                }
                print(f"{inp.name} -> {written.name}  ({n_res} residues, {n_valid})")

        except Exception as e:
            logger.error("Failed on %s: %s", inp.name, e)
            if args.verbose:
                import traceback
                traceback.print_exc()

    if len(input_paths) > 1:
        print(f"Done: {n_ok}/{len(input_paths)} structures processed.")


if __name__ == "__main__":
    main()
