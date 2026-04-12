"""Caustic — HF Space Gradio UI.

Predict NMR backbone chemical shifts from any PDB/mmCIF/AlphaFold structure.
Upload a file, click Predict, download results in NEF/NMR-STAR/CSV/JSON.

Launch locally:
    cd space && python app.py

Deploy to HF Spaces:
    Copy space/ contents to a HF Space repo, push.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("caustic-space")

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

ONNX_PATH = os.environ.get("CAUSTIC_ONNX", str(Path.home() / ".caustic" / "best.onnx"))

_SESSION = None

def _get_session():
    global _SESSION
    if _SESSION is None:
        from caustic.export import load_onnx_session
        logger.info("Loading ONNX model from %s", ONNX_PATH)
        _SESSION = load_onnx_session(ONNX_PATH)
    return _SESSION


# ---------------------------------------------------------------------------
# Prediction + formatting
# ---------------------------------------------------------------------------

NUCLEI = ("H", "HA", "N", "CA", "CB", "C")

NUCLEUS_COLORS = {
    "H": "#2196F3",
    "HA": "#4CAF50",
    "N": "#FF9800",
    "CA": "#E91E63",
    "CB": "#9C27B0",
    "C": "#795548",
}


def predict_and_format(
    file_obj,
    chain: str,
    ensemble: str,
):
    """Run caustic on an uploaded structure file."""
    if file_obj is None:
        raise gr.Error("Please upload a structure file (.pdb / .cif)")

    from caustic.config import GraphConfig, ModelConfig
    from caustic.inference import predict_shifts_onnx
    from caustic.io import write_csv, write_json, write_nef, write_nmrstar

    session = _get_session()
    input_path = file_obj if isinstance(file_obj, str) else file_obj.name
    chain_id = chain.strip() or None
    use_ensemble = ensemble == "median"

    pred = predict_shifts_onnx(
        input_path,
        ONNX_PATH,
        model_config=ModelConfig(),
        graph_config=GraphConfig(),
        chain_id=chain_id,
        use_ensemble=use_ensemble,
        session=session,
    )

    # Build per-nucleus plot
    fig = _make_shift_plot(pred)

    # Build summary table
    table = _make_table(pred)

    # Write output files
    tmp = Path(tempfile.mkdtemp(prefix="caustic_"))
    stem = Path(input_path).stem
    nef_path = write_nef(pred, tmp / f"{stem}.nef")
    star_path = write_nmrstar(pred, tmp / f"{stem}.str")
    csv_path = write_csv(pred, tmp / f"{stem}.csv")
    json_path = write_json(pred, tmp / f"{stem}.json")

    # Save plot as PNG for download
    plot_path = tmp / f"{stem}_shifts.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor="white")

    # Summary text
    n_res = len(pred.seq_ids)
    n_valid = {nuc: int(np.sum(np.isfinite(pred.mean[nuc]))) for nuc in NUCLEI}
    af_tag = ""
    summary = (
        f"**{Path(input_path).name}** | {n_res} residues | "
        f"{pred.num_conformers} conformer(s)\n\n"
        f"Valid predictions: " +
        ", ".join(f"{nuc} {n_valid[nuc]}" for nuc in NUCLEI)
    )

    return (
        fig,
        table,
        str(nef_path),
        str(star_path),
        str(csv_path),
        str(json_path),
        str(plot_path),
        summary,
    )


def _make_shift_plot(pred):
    """Per-nucleus shift vs residue plot with error bars."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    seq = np.array(pred.seq_ids)
    names = pred.residue_names or [""] * len(seq)

    for i, nuc in enumerate(NUCLEI):
        ax = axes[i]
        means = pred.mean[nuc]
        stds = pred.std[nuc]
        valid = np.isfinite(means)
        if not valid.any():
            ax.set_title(f"{nuc} (no predictions)", fontsize=11)
            continue

        x = seq[valid]
        y = means[valid]
        yerr = stds[valid] if stds is not None else np.zeros_like(y)
        yerr = np.where(np.isfinite(yerr), yerr, 0)

        color = NUCLEUS_COLORS[nuc]
        ax.errorbar(x, y, yerr=yerr, fmt=".", ms=3, lw=0.5,
                     color=color, ecolor=color, alpha=0.7, capsize=0)
        ax.set_title(f"{nuc}  ({int(valid.sum())} residues)", fontsize=11)
        ax.set_xlabel("Residue", fontsize=9)
        ax.set_ylabel("Shift (ppm)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Predicted backbone chemical shifts", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def _make_table(pred):
    """Build a list-of-lists table for Gradio Dataframe."""
    rows = []
    names = pred.residue_names or [""] * len(pred.seq_ids)
    for i, (sid, rn) in enumerate(zip(pred.seq_ids, names)):
        row = [int(sid), rn]
        for nuc in NUCLEI:
            m = float(pred.mean[nuc][i])
            s = float(pred.std[nuc][i]) if pred.std[nuc] is not None else float("nan")
            if np.isfinite(m):
                row.append(f"{m:.2f} +/- {s:.2f}" if np.isfinite(s) else f"{m:.2f}")
            else:
                row.append("-")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TABLE_HEADERS = ["Seq", "Res"] + [f"{n} (ppm)" for n in NUCLEI]

with gr.Blocks(
    title="Caustic — NMR Shift Predictor",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        "# Caustic\n"
        "Predict NMR backbone chemical shifts from any protein structure.  \n"
        "Upload a PDB, mmCIF, or AlphaFold model. AlphaFold structures are "
        "auto-detected and get pLDDT-calibrated uncertainty widening."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload structure (.pdb / .cif / .pdb.gz / .cif.gz)",
                file_types=[".pdb", ".cif", ".mmcif", ".gz", ".ent"],
            )
            with gr.Accordion("Options", open=False):
                chain_input = gr.Textbox(
                    label="Chain ID",
                    placeholder="auto (first polymer)",
                    value="",
                    max_lines=1,
                )
                ensemble_input = gr.Radio(
                    choices=["median", "first model only"],
                    value="median",
                    label="NMR ensemble handling",
                )
            predict_btn = gr.Button("Predict shifts", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_md = gr.Markdown("")

    with gr.Tabs():
        with gr.Tab("Plot"):
            plot_output = gr.Plot(label="Shift vs residue")
        with gr.Tab("Table"):
            table_output = gr.Dataframe(
                headers=TABLE_HEADERS,
                label="Per-residue predictions",
                wrap=True,
            )
        with gr.Tab("Downloads"):
            with gr.Row():
                nef_file = gr.File(label="NEF (.nef)")
                star_file = gr.File(label="NMR-STAR (.str)")
            with gr.Row():
                csv_file = gr.File(label="CSV (.csv)")
                json_file = gr.File(label="JSON (.json)")
            with gr.Row():
                plot_file = gr.File(label="Plot (.png)")

    predict_btn.click(
        fn=predict_and_format,
        inputs=[file_input, chain_input, ensemble_input],
        outputs=[
            plot_output,
            table_output,
            nef_file,
            star_file,
            csv_file,
            json_file,
            plot_file,
            summary_md,
        ],
    )

    gr.Markdown(
        "---\n"
        "*Caustic v0.1.0 — SchNet GNN (D14, 404K params). "
        "Wins 6/6 backbone nuclei vs SPARTA+, LEGOLAS, UCBShift2 on a "
        "712-protein test set. "
        "[GitHub](https://github.com/bluegems661/caustic)*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
