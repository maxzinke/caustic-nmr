"""NMR-STAR 3.x writer for :class:`ShiftPrediction`.

NMR-STAR 3.x is the BMRB deposition format: a tagged STAR dictionary
with saveframes and loops. This writer emits a minimal but spec-compliant
``Assigned_chem_shift_list`` saveframe with one ``Atom_chem_shift`` loop
carrying one row per predicted shift.

Hand-rolled so the standalone package doesn't need a hard dependency on
``pynmrstar``. Users who want proper round-trip validation (e.g.
depositors running BMRB's ADIT-NMR tool) can still load the output with
``pynmrstar.Entry.from_file`` — the syntax is compatible with the
reference parser.

Column set matches the production BMRB dictionary (NMR-STAR v3.2+):

- ``ID``               — row ID (1-based)
- ``Comp_index_ID``    — residue sequence number (BMRB numbering)
- ``Comp_ID``          — 3-letter residue name
- ``Atom_ID``          — atom name (H, HA, N, CA, CB, C, ...)
- ``Atom_type``        — element symbol
- ``Atom_isotope_number`` — 1, 13, 15, ...
- ``Val``              — chemical shift value in ppm
- ``Val_err``          — shift uncertainty in ppm
- ``Assign_fig_of_merit`` — set to ``.`` (unused for predictions)
- ``Ambiguity_code``   — 1 (unique resonance) — matches the default when
  BMRB depositors have no alternate assignments
- ``Assigned_chem_shift_list_ID`` — foreign key to the parent saveframe
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from caustic.features import BACKBONE_NUCLEI
from caustic.inference import ShiftPrediction


_NUCLEUS_META: dict[str, tuple[str, int]] = {
    "H":  ("H", 1),
    "HA": ("H", 1),
    "N":  ("N", 15),
    "CA": ("C", 13),
    "CB": ("C", 13),
    "C":  ("C", 13),
}


def _iupac_atom_name(residue_name: str, nucleus: str) -> str:
    """Pick an IUPAC-compatible atom label for ``(residue, nucleus)``.

    NMR-STAR (and BMRB deposition) follows the same IUPAC atom-name
    rules that NEF inherits. The Caustic's single glycine HA
    output is a degenerate average of HA2/HA3 and is emitted as ``HA%``
    for consistency with the NEF writer; BMRB depositors importing the
    file can re-label or split the wildcard at their discretion.
    """
    if residue_name == "GLY" and nucleus == "HA":
        return "HA%"
    return nucleus


@dataclass
class NMRStarWriterOptions:
    """Tuning knobs for :func:`write_nmrstar`."""

    entry_id: str = "crystalline_predicted"
    """Top-level ``data_`` block and ``Entry.ID`` value."""

    list_id: int = 1
    """Saveframe ID for the chemical shift list."""

    list_name: str = "predicted_chem_shift_list_1"
    """Saveframe framecode name (the part after ``save_``)."""

    ppm_digits: int = 4
    sigma_digits: int = 4
    ambiguity_code: int = 1
    program_name: str = "caustic"
    program_version: str = "unknown"


def _fmt(value: object, digits: int | None = None) -> str:
    """Format a STAR cell. ``None``/NaN → ``.``; floats honour ``digits``."""
    if value is None:
        return "."
    if isinstance(value, float):
        if not np.isfinite(value):
            return "."
        if digits is None:
            return f"{value}"
        return f"{value:.{digits}f}"
    s = str(value)
    if not s:
        return "."
    needs_quote = any(c in s for c in " \t\n'\"") or s.startswith(("_", "#"))
    if needs_quote:
        return f'"{s}"' if '"' not in s else f"'{s}'"
    return s


def _iter_shift_rows(prediction: ShiftPrediction) -> Iterable[tuple]:
    """Yield (comp_index_id, comp_id, atom_id, element, isotope, value, sigma)."""
    n_res = len(prediction.seq_ids)
    names = prediction.residue_names or (["UNK"] * n_res)
    if len(names) != n_res:
        names = (list(names) + ["UNK"] * n_res)[:n_res]

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
            s = float(stds[i]) if stds is not None else float("nan")
            atom_id = _iupac_atom_name(names[i], nuc)
            yield (int(prediction.seq_ids[i]), names[i], atom_id, element, isotope, m, s)


def _sanitize_block_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return safe or "crystalline"


def write_nmrstar(
    prediction: ShiftPrediction,
    path: str | Path,
    *,
    options: NMRStarWriterOptions | None = None,
) -> Path:
    """Write an NMR-STAR 3.x file containing one chemical shift list.

    Produces a minimally-compliant ``Assigned_chem_shift_list`` saveframe
    with an ``Atom_chem_shift`` loop. Sufficient for BMRB-aware NMR
    software (``pynmrstar``, NMRbox tools, PRISM) to round-trip
    (residue, atom, shift, sigma) tuples. Depositors will need to fill in
    sample conditions and other saveframes before an actual BMRB
    submission.

    Parameters
    ----------
    prediction: the :class:`ShiftPrediction` to serialise.
    path: output ``.str`` file path.
    options: optional :class:`NMRStarWriterOptions`.

    Returns
    -------
    Resolved absolute ``Path`` to the written file.
    """
    if options is None:
        options = NMRStarWriterOptions()

    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"data_{_sanitize_block_name(options.entry_id)}")
    lines.append("")
    lines.append(f"save_{options.list_name}")
    lines.append("   _Assigned_chem_shift_list.Sf_category      assigned_chemical_shifts")
    lines.append(f"   _Assigned_chem_shift_list.Sf_framecode     {options.list_name}")
    lines.append(
        f"   _Assigned_chem_shift_list.Entry_ID         {_fmt(options.entry_id)}"
    )
    lines.append(
        f"   _Assigned_chem_shift_list.ID               {options.list_id}"
    )
    lines.append(
        f"   _Assigned_chem_shift_list.Chem_shift_reference_ID  ."
    )
    lines.append(
        f"   _Assigned_chem_shift_list.Program_name     {_fmt(options.program_name)}"
    )
    lines.append(
        f"   _Assigned_chem_shift_list.Program_version  {_fmt(options.program_version)}"
    )
    lines.append("")
    lines.append("   loop_")
    lines.append("      _Atom_chem_shift.ID")
    lines.append("      _Atom_chem_shift.Comp_index_ID")
    lines.append("      _Atom_chem_shift.Comp_ID")
    lines.append("      _Atom_chem_shift.Atom_ID")
    lines.append("      _Atom_chem_shift.Atom_type")
    lines.append("      _Atom_chem_shift.Atom_isotope_number")
    lines.append("      _Atom_chem_shift.Val")
    lines.append("      _Atom_chem_shift.Val_err")
    lines.append("      _Atom_chem_shift.Assign_fig_of_merit")
    lines.append("      _Atom_chem_shift.Ambiguity_code")
    lines.append("      _Atom_chem_shift.Assigned_chem_shift_list_ID")
    lines.append("")

    row_id = 0
    for (comp_idx, resname, atom, element, isotope, value, sigma) in _iter_shift_rows(prediction):
        row_id += 1
        cells = (
            _fmt(row_id),
            _fmt(comp_idx),
            _fmt(resname),
            _fmt(atom),
            _fmt(element),
            _fmt(isotope),
            _fmt(value, digits=options.ppm_digits),
            _fmt(sigma, digits=options.sigma_digits),
            ".",  # Assign_fig_of_merit — unused for predictions
            _fmt(options.ambiguity_code),
            _fmt(options.list_id),
        )
        lines.append("      " + " ".join(cells))
    lines.append("")
    lines.append("   stop_")
    lines.append("save_")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def try_pynmrstar_validate(path: str | Path) -> tuple[bool, str]:
    """Optional round-trip sanity check via ``pynmrstar`` if installed.

    Returns ``(ok, message)``. ``ok`` is True when ``pynmrstar`` parses
    the file without errors, False otherwise. When ``pynmrstar`` is not
    installed, returns ``(True, "pynmrstar not installed — skipped")``
    so callers can treat it as a soft check.
    """
    try:
        import pynmrstar  # type: ignore
    except ImportError:
        return True, "pynmrstar not installed — skipped"
    try:
        entry = pynmrstar.Entry.from_file(str(path))
        n = 0
        for sf in entry.get_saveframes_by_category("assigned_chemical_shifts"):
            for lp in sf.loops:
                n += len(lp.data)
        return True, f"pynmrstar parsed {n} shift rows"
    except Exception as e:
        return False, f"pynmrstar parse error: {e}"
