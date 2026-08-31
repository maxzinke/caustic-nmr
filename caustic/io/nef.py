"""NEF (NMR Exchange Format) chemical shift writer.

NEF is the wwPDB-endorsed modern NMR exchange format, supported natively
by CCPN Analysis v3. It's STAR-based (same textual syntax as NMR-STAR)
but uses a namespaced vocabulary (``nef_*``, ``_nef_*``). For chemical
shifts, the relevant saveframe is ``nef_chemical_shift_list`` with a
single loop ``_nef_chemical_shift`` whose rows carry one shift each.

This module hand-rolls a NEF chemical-shift-list writer — the format is
small and stable enough that adding a runtime dependency on ``pynmrstar``
isn't worth the install cost, especially for the standalone packaging in
Step 2.5. The output is validated against the NEF v1.1 specification
columns for ``_nef_chemical_shift``.

Reference: https://github.com/NMRExchangeFormat/NEF/blob/master/specification/mmcif_nef.dic
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from caustic.features import BACKBONE_NUCLEI
from caustic.inference import ShiftPrediction
from caustic.provenance import provenance_line, resolve_provenance


# Per-nucleus element / isotope number metadata. Matches the NEF
# convention where backbone nuclei are identified by their element +
# isotope; e.g. 1H, 13C, 15N. Only backbone nuclei are emitted.
_NUCLEUS_META: dict[str, tuple[str, int]] = {
    "H":  ("H", 1),
    "HA": ("H", 1),
    "N":  ("N", 15),
    "CA": ("C", 13),
    "CB": ("C", 13),
    "C":  ("C", 13),
}


def _nef_atom_name(residue_name: str, nucleus: str) -> str:
    """Return the NEF-correct atom name for a (residue, nucleus) pair.

    Reference: ``docs/NEF_ATOM_NAMING_CONVENTIONS.md`` (version 1.0.0,
    based on NEF 1.1).

    Most backbone nuclei map 1:1 with their NEF name (``H``, ``N``,
    ``CA``, ``CB``, ``C``). The one exception the Caustic hits
    is **glycine HA**: GLY has two alpha protons (HA2 pro-R, HA3 pro-S)
    and the model outputs a single averaged value, so the NEF-correct
    label is ``HA%`` (the degenerate wildcard). Anything else would
    either falsely imply stereospecificity (``HA2``/``HA3``) or
    non-stereospecific assignment (``HAx``/``HAy``), neither of which
    the Caustic claims.

    CB is implicitly skipped upstream for GLY (there is no CB so the
    prediction is NaN). This function is only invoked on residues with
    a real emitted value.
    """
    if residue_name == "GLY" and nucleus == "HA":
        return "HA%"
    return nucleus


@dataclass
class NEFWriterOptions:
    """Tuning knobs for :func:`write_nef`."""

    chain_code: str = "A"
    """Chain label written into the ``chain_code`` column."""

    list_id: int = 1
    """NEF chemical-shift-list ID. Incremented per saveframe if multiple."""

    name: str = "caustic"
    """Free-text name stored in the saveframe header. Shows up in CCPN."""

    ppm_digits: int = 4
    """Decimal digits for the ``value`` column."""

    sigma_digits: int = 4
    """Decimal digits for the ``value_uncertainty`` column."""

    program_name: str = "caustic"
    """Program identifier for the NEF metadata saveframe."""

    program_version: str | None = None
    """Program version for the NEF metadata saveframe. ``None`` = the
    version recorded in the prediction's provenance (the package version)."""


def _format_cell(value: object, digits: int | None = None) -> str:
    """Format one STAR cell — quote strings containing whitespace."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "."
    if isinstance(value, float):
        if digits is None:
            return f"{value}"
        return f"{value:.{digits}f}"
    s = str(value)
    if not s:
        return "."
    needs_quote = any(c in s for c in " \t\n'\"") or s.startswith(("_", "#"))
    if needs_quote:
        if '"' not in s:
            return f'"{s}"'
        return f"'{s}'"
    return s


def _iter_shift_rows(
    prediction: ShiftPrediction,
    opts: NEFWriterOptions,
) -> Iterable[tuple]:
    """Yield one (chain, seq_id, residue, atom, value, sigma, element, isotope) per shift."""
    n_res = len(prediction.seq_ids)
    # Fall back to UNK names if the ShiftPrediction predates the residue_names fix.
    names = prediction.residue_names or (["UNK"] * n_res)
    if len(names) != n_res:
        # Safety net: truncate or pad to match seq_ids
        if len(names) < n_res:
            names = list(names) + ["UNK"] * (n_res - len(names))
        else:
            names = list(names[:n_res])

    for nuc in BACKBONE_NUCLEI:
        means = prediction.mean.get(nuc)
        stds = prediction.std.get(nuc)
        if means is None:
            continue
        element, isotope = _NUCLEUS_META[nuc]
        for i in range(n_res):
            m = float(means[i])
            if not np.isfinite(m):
                continue
            s_val = float(stds[i]) if stds is not None else float("nan")
            atom_name = _nef_atom_name(names[i], nuc)
            yield (
                opts.chain_code,
                int(prediction.seq_ids[i]),
                names[i],
                atom_name,
                m,
                s_val,
                element,
                isotope,
            )


def write_nef(
    prediction: ShiftPrediction,
    path: str | Path,
    *,
    options: NEFWriterOptions | None = None,
) -> Path:
    """Write a NEF chemical-shift-list file.

    Writes an ``nef_nmr_meta_data`` saveframe (NEF convention for the
    header) plus a ``nef_chemical_shift_list`` saveframe containing the
    ``_nef_chemical_shift`` loop with one row per predicted shift.

    Parameters
    ----------
    prediction: the :class:`ShiftPrediction` to serialise.
    path: output file path. The parent directory is created if missing.
    options: optional :class:`NEFWriterOptions` for chain, list_id, and
        formatting knobs.

    Returns
    -------
    Resolved absolute ``Path`` to the written file.
    """
    if options is None:
        options = NEFWriterOptions()

    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prov = resolve_provenance(prediction)
    program_version = options.program_version or str(prov["version"])

    lines: list[str] = []
    # --- Global data block -------------------------------------------------
    lines.append(f"data_{_sanitize_block_name(options.name)}")
    lines.append(f"# {provenance_line(prov)}")
    lines.append(f"# model_sha256 {prov.get('model_sha256', 'unknown')}")
    lines.append("")

    # --- Metadata saveframe ----------------------------------------------
    lines.append("save_nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.sf_category       nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.sf_framecode      nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.format_name       nmr_exchange_format")
    lines.append("   _nef_nmr_meta_data.format_version    1.1")
    lines.append(
        f"   _nef_nmr_meta_data.program_name      {_format_cell(options.program_name)}"
    )
    lines.append(
        f"   _nef_nmr_meta_data.program_version   {_format_cell(program_version)}"
    )
    lines.append(
        f"   _nef_nmr_meta_data.creation_date     {_format_cell(str(prov.get('date', '.')))}"
    )
    lines.append("save_")
    lines.append("")

    # --- Chemical shift saveframe ----------------------------------------
    sf_name = f"nef_chemical_shift_list_{options.list_id}"
    lines.append(f"save_{sf_name}")
    lines.append(
        "   _nef_chemical_shift_list.sf_category   nef_chemical_shift_list"
    )
    lines.append(
        f"   _nef_chemical_shift_list.sf_framecode  {sf_name}"
    )
    lines.append("")
    lines.append("   loop_")
    lines.append("      _nef_chemical_shift.chain_code")
    lines.append("      _nef_chemical_shift.sequence_code")
    lines.append("      _nef_chemical_shift.residue_name")
    lines.append("      _nef_chemical_shift.atom_name")
    lines.append("      _nef_chemical_shift.value")
    lines.append("      _nef_chemical_shift.value_uncertainty")
    lines.append("      _nef_chemical_shift.element")
    lines.append("      _nef_chemical_shift.isotope_number")
    lines.append("")

    n_rows = 0
    for (chain, seq_id, resname, atom, value, sigma, element, isotope) in _iter_shift_rows(
        prediction, options
    ):
        lines.append(
            "      "
            + " ".join(
                _format_cell(v, digits=d)
                for v, d in zip(
                    (chain, seq_id, resname, atom, value, sigma, element, isotope),
                    (None, None, None, None, options.ppm_digits, options.sigma_digits, None, None),
                )
            )
        )
        n_rows += 1
    lines.append("")
    lines.append("   stop_")
    lines.append("save_")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _sanitize_block_name(name: str) -> str:
    """STAR data-block names must be a single non-space token."""
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return safe or "caustic"


# ---------------------------------------------------------------------------
# Minimal NEF reader (for round-trip testing only — not a public entry point)
# ---------------------------------------------------------------------------


def _read_nef_chemical_shifts(path: str | Path) -> list[dict[str, str]]:
    """Parse a ``_nef_chemical_shift`` loop into a list of row dicts.

    This is intentionally a minimal parser — good enough for our
    round-trip tests, not a general NEF parser. It locates the first
    ``loop_`` block whose tag namespace is ``_nef_chemical_shift`` and
    returns one dict per data row.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    tags: list[str] = []
    in_loop = False
    in_shift_loop = False
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "loop_":
            in_loop = True
            in_shift_loop = False
            tags = []
            i += 1
            continue
        if in_loop and stripped.startswith("_nef_chemical_shift."):
            tags.append(stripped.split(".", 1)[1])
            in_shift_loop = True
            i += 1
            continue
        if in_loop and in_shift_loop:
            if stripped == "stop_":
                in_loop = False
                in_shift_loop = False
                tags = []
                i += 1
                continue
            # Data row — split into tokens respecting simple quoting.
            tokens = _split_star_row(stripped)
            if len(tokens) == len(tags):
                rows.append(dict(zip(tags, tokens)))
        i += 1
    return rows


def _split_star_row(line: str) -> list[str]:
    """Split a STAR data row into tokens, handling simple quoted strings."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n and line[j] != quote:
                j += 1
            out.append(line[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out
