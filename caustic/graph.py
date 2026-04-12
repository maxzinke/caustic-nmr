"""Convert PDB/mmCIF structures to PyTorch Geometric Data objects."""
from __future__ import annotations

import logging
import re
import numpy as np
from pathlib import Path
from typing import Any

from caustic.config import GraphConfig
from caustic.features import (
    AA_THREE_TO_IDX,
    BACKBONE_NUCLEI,
    ELEMENT_TO_IDX,
    NUM_ATOM_ROLES,
    NUM_ELEMENT_TYPES,
    TARGET_ATOM_NAMES,
    UNK_AA_IDX,
    UNK_ELEMENT_IDX,
    get_atom_role,
    normalize_residue_name,
)

logger = logging.getLogger(__name__)

# Suffixes we recognise as structure files. gemmi.read_structure handles both
# PDB and mmCIF transparently (including .gz), so we only use the suffix for
# validation and to decide between `read_structure` and explicit CIF block
# parsing when a file comes in without a recognised extension.
_STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}

# AlphaFold EBI database filename convention:
#   AF-<UniProt>-F<frag>-model_v<version>.{pdb,cif}
_AF_FILENAME_RE = re.compile(
    r"^AF-[A-Z0-9]+-F\d+-model_v\d+\.(pdb|cif)$", re.IGNORECASE
)


def _load_structure(path: Path) -> Any:
    """Load a PDB/mmCIF file (optionally .gz) into a gemmi.Structure.

    Centralised so every entry point into the graph builder has identical
    file-handling behaviour (case-insensitive suffixes, .gz support,
    consistent error messages).
    """
    import gemmi

    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    # Strip .gz for suffix inspection — gemmi reads gzipped files directly.
    inner = path.name
    if inner.lower().endswith(".gz"):
        inner = inner[:-3]
    suffix = Path(inner).suffix.lower()

    if suffix and suffix not in _STRUCTURE_SUFFIXES:
        logger.warning(
            "Unrecognised structure suffix %r (allowed: %s). Attempting to "
            "read via gemmi.read_structure anyway.",
            suffix, sorted(_STRUCTURE_SUFFIXES),
        )

    # gemmi.read_structure handles .pdb, .ent, .cif, .mmcif, and their .gz
    # variants. The explicit CIF block path (gemmi.cif.read →
    # make_structure_from_block) is only needed when the mmCIF uses a weird
    # block layout; keep it as a fallback below.
    try:
        return gemmi.read_structure(str(path))
    except Exception as e:
        if suffix in (".cif", ".mmcif"):
            try:
                doc = gemmi.cif.read(str(path))
                return gemmi.make_structure_from_block(doc[0])
            except Exception as e2:
                raise ValueError(
                    f"Failed to parse {path} as PDB or mmCIF: {e} / {e2}"
                ) from e2
        raise ValueError(f"Failed to parse {path}: {e}") from e


def _unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector for v; all-zero input is passed through."""
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _synthesise_missing_hydrogens(
    *,
    num_residues: int,
    residue_aa: list[int],
    target_atom_map: list[list[int]],
    target_atom_valid: list[list[bool]],
    positions: list[list[float]],
    element_ids: list[int],
    aa_ids: list[int],
    role_ids: list[int],
    residue_ids: list[int],
    bfactors: list[float],
    atom_names: list[str],
    strict: bool,
) -> int:
    """Place backbone H and HA atoms for residues that lack them.

    Used when the input structure has no hydrogens (typical for X-ray PDBs
    and AlphaFold models). Geometry is standard: amide H sits ~0.99 Å from
    N along the bisector opposite N-CA and N-C(i-1); alpha H sits ~1.09 Å
    from CA opposite the sum of N, C, and CB unit vectors. PRO residues are
    skipped for amide H (no backbone NH). The N-terminal residue falls
    back to a simpler one-neighbour placement for amide H.

    This is intentionally an approximation — real hydrogens are slightly
    better but the model's learned representation is dominated by the
    heavy-atom environment, so the resulting shift predictions are within
    a small perturbation of the "structure has hydrogens" case. When no
    canonical heavy atom is available to anchor the synthesised position,
    the residue is skipped and its H/HA target stays unmasked.

    Returns the number of atoms that were added.
    """
    from caustic.features import ELEMENT_TO_IDX, ATOM_ROLE_MAP

    # Nucleus indices in BACKBONE_NUCLEI: H=0, HA=1, N=2, CA=3, CB=4, C=5
    I_H, I_HA, I_N, I_CA, I_CB, I_C = 0, 1, 2, 3, 4, 5
    H_EL = ELEMENT_TO_IDX.get("H", 4)
    H_ROLE = ATOM_ROLE_MAP.get("H", 5)
    HA_ROLE = ATOM_ROLE_MAP.get("HA", 6)
    PRO_IDX = 14
    GLY_IDX = 7

    pos_arr = np.asarray(positions, dtype=np.float64)
    added = 0
    missing_strict: list[tuple[int, str]] = []

    for ri in range(num_residues):
        needs_h = not target_atom_valid[ri][I_H] and residue_aa[ri] != PRO_IDX
        needs_ha = not target_atom_valid[ri][I_HA]
        if not needs_h and not needs_ha:
            continue

        have_n = target_atom_valid[ri][I_N]
        have_ca = target_atom_valid[ri][I_CA]
        have_c = target_atom_valid[ri][I_C]
        if not (have_n and have_ca and have_c):
            # Cannot synthesise without the full N/CA/C backbone anchor.
            if strict:
                if needs_h:
                    missing_strict.append((ri, "H"))
                if needs_ha:
                    missing_strict.append((ri, "HA"))
            continue

        n_pos = pos_arr[target_atom_map[ri][I_N]]
        ca_pos = pos_arr[target_atom_map[ri][I_CA]]
        c_pos = pos_arr[target_atom_map[ri][I_C]]

        # -- Amide H (not PRO) --------------------------------------------
        if needs_h:
            # Previous residue's C is the peptide-bond donor neighbour.
            prev_c = None
            if ri > 0 and target_atom_valid[ri - 1][I_C]:
                prev_c = pos_arr[target_atom_map[ri - 1][I_C]]
            if prev_c is not None:
                direction = _unit(_unit(n_pos - ca_pos) + _unit(n_pos - prev_c))
            else:
                # N-terminal or broken chain — fall back to N-CA axis only.
                direction = _unit(n_pos - ca_pos)
            if float(np.linalg.norm(direction)) > 0.5:
                h_xyz = n_pos + 0.99 * direction
                new_idx = len(positions)
                positions.append(h_xyz.tolist())
                element_ids.append(H_EL)
                aa_ids.append(residue_aa[ri])
                role_ids.append(H_ROLE)
                residue_ids.append(ri)
                bfactors.append(0.0)
                atom_names.append("H")
                target_atom_map[ri][I_H] = new_idx
                target_atom_valid[ri][I_H] = True
                added += 1

        # -- Alpha H ------------------------------------------------------
        if needs_ha:
            v_n = _unit(n_pos - ca_pos)
            v_c = _unit(c_pos - ca_pos)
            if residue_aa[ri] == GLY_IDX:
                # No CB. Place a single HA out-of-plane perpendicular to the
                # N-CA-C plane, offset along the in-plane bisector.
                plane_n = _unit(np.cross(n_pos - ca_pos, c_pos - ca_pos))
                in_plane = _unit(-(v_n + v_c))
                direction = _unit(in_plane * 0.577 + plane_n * 0.816)
            elif target_atom_valid[ri][I_CB]:
                cb_pos = pos_arr[target_atom_map[ri][I_CB]]
                v_cb = _unit(cb_pos - ca_pos)
                direction = _unit(-(v_n + v_c + v_cb))
            else:
                # CB missing (non-GLY) — use N/C bisector and nudge out of plane.
                direction = _unit(-(v_n + v_c))
            if float(np.linalg.norm(direction)) > 0.5:
                ha_xyz = ca_pos + 1.09 * direction
                new_idx = len(positions)
                positions.append(ha_xyz.tolist())
                element_ids.append(H_EL)
                aa_ids.append(residue_aa[ri])
                role_ids.append(HA_ROLE)
                residue_ids.append(ri)
                bfactors.append(0.0)
                atom_names.append("HA")
                target_atom_map[ri][I_HA] = new_idx
                target_atom_valid[ri][I_HA] = True
                added += 1

    if strict and missing_strict:
        raise ValueError(
            f"missing_hydrogens='strict': could not synthesise "
            f"{len(missing_strict)} hydrogens (missing backbone anchors)."
        )

    return added


def _is_alphafold_structure(path: Path, st: Any) -> bool:
    """Detect AlphaFold-origin structures via filename or content.

    Three signals, any one of which flips the flag:
    1. Filename matches the EBI AF2 convention (AF-*-F*-model_v*).
    2. Legacy / custom filename prefix af_* or alphafold_*.
    3. Single-model structure with B-factors that look like pLDDT
       (all values in [0, 100], median > 50 — typical of pLDDT, atypical
       of X-ray B-factors which often exceed 100 and have lower medians
       for rigid core residues).

    The heuristic is intentionally permissive — false positives just mean
    the bfactor is interpreted as pLDDT, which only matters when the
    downstream packaging step wants to widen σ for low-confidence residues.
    """
    name = path.name
    stem_lower = path.stem.lower()
    if _AF_FILENAME_RE.match(name):
        return True
    if stem_lower.startswith(("af_", "alphafold_", "af-")):
        return True

    # Content heuristic: single model, all atoms with b_iso in [0, 100],
    # median of CA b_iso > 50 (pLDDT-like).
    try:
        if len(st) != 1:
            return False
        ca_b: list[float] = []
        for ch in st[0]:
            for res in ch:
                for atom in res:
                    if atom.name.strip() == "CA":
                        ca_b.append(atom.b_iso)
        if not ca_b:
            return False
        arr = np.asarray(ca_b, dtype=np.float32)
        if float(arr.min()) < 0.0 or float(arr.max()) > 100.5:
            return False
        return float(np.median(arr)) > 50.0
    except Exception:
        return False


try:
    from torch_geometric.data import Data as _Data

    class ProteinData(_Data):
        """Data subclass that offsets target_indices and residue_idx on batch."""

        def __inc__(self, key: str, value: Any, *args: Any, **kw: Any) -> Any:
            if key == "target_indices":
                return self.pos.size(0)
            if key == "residue_idx":
                return int(self.num_residues)
            return super().__inc__(key, value, *args, **kw)

        def __cat_dim__(self, key: str, value: Any, *args: Any, **kw: Any) -> Any:
            if key in ("num_atoms", "num_residues"):
                return None
            return super().__cat_dim__(key, value, *args, **kw)

except ImportError:
    ProteinData = None  # type: ignore[misc,assignment]


def structure_to_graph(
    structure_path: str | Path,
    chain_id: str | None = None,
    model_idx: int = 0,
    config: GraphConfig | None = None,
    shifts: dict[int, dict[str, float]] | None = None,
    seq_mapping: dict[int, int] | None = None,
) -> Any:
    """Convert a PDB/mmCIF structure into a PyG Data object.

    Args:
        structure_path: Path to ``.pdb`` or ``.cif`` file.
        chain_id: Chain to extract (``None`` = first polymer chain).
        model_idx: Model / conformer index (0 = first).
        config: Graph construction parameters.
        shifts: Optional shift labels ``{seq_id: {atom_name: ppm}}``.
        seq_mapping: Optional pre-computed ``{pdb_seq_id: bmrb_seq_id}`` from
            Needleman-Wunsch alignment. When provided, bypasses the heuristic
            ``_assign_shifts`` and uses the validated mapping directly.

    Returns:
        A ``torch_geometric.data.Data`` with:
        - ``pos``            [num_atoms, 3]       atom coordinates
        - ``element``        [num_atoms]           element vocab index
        - ``amino_acid``     [num_atoms]           AA vocab index
        - ``atom_role``      [num_atoms]           backbone / SC role
        - ``residue_idx``    [num_atoms]           which residue (0-based)
        - ``bfactor``        [num_atoms]           isotropic B-factor
        - ``edge_index``     [2, num_edges]
        - ``edge_dist``      [num_edges]           pairwise distance
        - ``seq_sep``        [num_edges]           |i - j| sequence sep (capped)
        - ``same_residue``   [num_edges]           1.0 if same residue
        - ``target_indices`` [num_residues, 6]     atom idx per nucleus (-1→0, see mask)
        - ``target_mask``    [num_residues, 6]     True where label exists
        - ``target_shifts``  [num_residues, 6]     shift ppm (NaN if missing)
        - ``residue_types``  [num_residues]        AA index per residue
        - ``seq_ids``        [num_residues]        PDB sequence numbers
        - ``num_atoms``      int
        - ``num_residues``   int
    """
    import torch
    import gemmi
    from scipy.spatial import cKDTree

    if config is None:
        config = GraphConfig()

    # ------------------------------------------------------------------
    # Parse structure
    # ------------------------------------------------------------------
    path = Path(structure_path)
    st = _load_structure(path)
    if len(st) == 0:
        raise ValueError(f"No models in structure {structure_path}")
    st.setup_entities()

    model_idx = max(0, min(model_idx, len(st) - 1))
    model = st[model_idx]

    # Find first polymer chain
    if chain_id is None:
        for ch in model:
            for res in ch:
                if res.entity_type == gemmi.EntityType.Polymer:
                    chain_id = ch.name
                    break
            if chain_id:
                break
    if chain_id is None:
        raise ValueError(f"No polymer chain in {structure_path}")

    chain = model[chain_id]

    # ------------------------------------------------------------------
    # Detect disulfide bonds (CYS residues with SS bridges)
    # ------------------------------------------------------------------
    disulfide_seqids: set[int] = set()
    for conn in st.connections:
        if conn.type == gemmi.ConnectionType.Disulf:
            p1, p2 = conn.partner1, conn.partner2
            if p1.res_id.seqid.num and (not chain_id or p1.chain_name == chain_id):
                disulfide_seqids.add(p1.res_id.seqid.num)
            if p2.res_id.seqid.num and (not chain_id or p2.chain_name == chain_id):
                disulfide_seqids.add(p2.res_id.seqid.num)

    # ------------------------------------------------------------------
    # Extract atoms
    # ------------------------------------------------------------------
    # AlphaFold models store pLDDT (0-100, high=confident) in the B-factor
    # column. Invert to match experimental B-factor semantics where
    # high = mobile/uncertain: pseudo_bfactor = 100 - pLDDT.
    _is_af2 = _is_alphafold_structure(path, st)
    if _is_af2:
        logger.info("Detected AlphaFold-origin structure: %s (B-factor treated as pLDDT)", path.name)

    positions: list[list[float]] = []
    element_ids: list[int] = []
    aa_ids: list[int] = []
    role_ids: list[int] = []
    residue_ids: list[int] = []
    bfactors: list[float] = []
    atom_names: list[str] = []
    plddt_per_residue: list[float] = []  # mean CA pLDDT per residue (NaN if not AF)

    residue_aa: list[int] = []
    residue_seqid: list[int] = []
    residue_disulfide: list[bool] = []  # True if CYS in SS bond
    target_atom_map: list[list[int]] = []  # [num_res][6]
    target_atom_valid: list[list[bool]] = []  # [num_res][6] — was atom found?

    # Build reverse map: atom_name → nucleus index (for target detection)
    _name_to_nuc: dict[str, int] = {}
    for nuc_idx, nuc in enumerate(BACKBONE_NUCLEI):
        for tname in TARGET_ATOM_NAMES[nuc]:
            _name_to_nuc[tname] = nuc_idx

    # Track which non-standard / unknown residue names we've already warned
    # about so users see each unique substitution once per call.
    _warned_residues: set[str] = set()

    atom_i = 0
    res_i = 0

    for residue in chain:
        if residue.entity_type != gemmi.EntityType.Polymer:
            continue

        raw_name = residue.name.strip().upper()
        aa_idx, is_nonstandard, is_unknown = normalize_residue_name(raw_name)

        if is_nonstandard and raw_name not in _warned_residues:
            logger.info(
                "Non-standard residue %s in %s normalized to canonical AA "
                "(idx=%d). Shift prediction will use the parent amino-acid "
                "embedding.",
                raw_name, path.name, aa_idx,
            )
            _warned_residues.add(raw_name)
        elif is_unknown and raw_name not in _warned_residues:
            logger.warning(
                "Unknown residue %s in %s — falling back to UNK embedding "
                "(predictions may be unreliable at this position).",
                raw_name, path.name,
            )
            _warned_residues.add(raw_name)

        residue_aa.append(aa_idx)
        residue_seqid.append(residue.seqid.num)
        residue_disulfide.append(residue.seqid.num in disulfide_seqids and aa_idx == 4)  # CYS=4
        res_targets = [0] * 6  # default 0 (safe for batching); mask tracks validity
        res_target_found = [False] * 6
        res_ca_b = float("nan")

        for atom in residue:
            if atom.is_hydrogen() and not config.include_hydrogens:
                continue

            el_name = atom.element.name
            el_idx = ELEMENT_TO_IDX.get(el_name, UNK_ELEMENT_IDX)
            role = get_atom_role(atom.name, atom.is_hydrogen())

            # Compute atom name first so we can use it both for target
            # detection and for `_atom_names` storage.
            name = atom.name.strip()

            pos = atom.pos
            positions.append([pos.x, pos.y, pos.z])
            element_ids.append(el_idx)
            aa_ids.append(aa_idx)
            role_ids.append(role)
            residue_ids.append(res_i)
            bfactors.append(100.0 - atom.b_iso if _is_af2 else atom.b_iso)
            atom_names.append(name)

            if _is_af2 and name == "CA":
                res_ca_b = float(atom.b_iso)

            # Target atom detection (first match per nucleus wins)
            nuc_idx = _name_to_nuc.get(name, -1)
            if nuc_idx >= 0 and not res_target_found[nuc_idx]:
                res_targets[nuc_idx] = atom_i
                res_target_found[nuc_idx] = True

            atom_i += 1

        target_atom_map.append(res_targets)
        target_atom_valid.append(res_target_found)
        plddt_per_residue.append(res_ca_b)
        res_i += 1

    num_atoms = atom_i
    num_residues = res_i
    if num_atoms == 0:
        raise ValueError(f"No atoms from chain {chain_id} in {structure_path}")
    if num_residues < config.min_residues:
        raise ValueError(
            f"Chain {chain_id} in {structure_path} has {num_residues} residues; "
            f"GraphConfig.min_residues={config.min_residues}."
        )

    if num_residues > config.large_protein_warn_threshold:
        logger.warning(
            "Large protein: %d residues in %s (threshold %d). Inference will "
            "run but may use significant memory; consider splitting by chain.",
            num_residues, path.name, config.large_protein_warn_threshold,
        )

    # ------------------------------------------------------------------
    # Synthesise missing backbone hydrogens (X-ray PDB / AlphaFold inputs)
    # ------------------------------------------------------------------
    if config.missing_hydrogens in ("geometric", "strict"):
        n_missing_h = sum(1 for v in target_atom_valid if not v[0])
        n_missing_ha = sum(1 for v in target_atom_valid if not v[1])
        if n_missing_h + n_missing_ha > 0:
            added = _synthesise_missing_hydrogens(
                num_residues=num_residues,
                residue_aa=residue_aa,
                target_atom_map=target_atom_map,
                target_atom_valid=target_atom_valid,
                positions=positions,
                element_ids=element_ids,
                aa_ids=aa_ids,
                role_ids=role_ids,
                residue_ids=residue_ids,
                bfactors=bfactors,
                atom_names=atom_names,
                strict=(config.missing_hydrogens == "strict"),
            )
            if added > 0:
                logger.info(
                    "Synthesised %d backbone hydrogens (%d H, %d HA missing) for %s",
                    added, n_missing_h, n_missing_ha, path.name,
                )
            num_atoms = len(positions)

    # ------------------------------------------------------------------
    # Build edges via KDTree
    # ------------------------------------------------------------------
    pos_np = np.array(positions, dtype=np.float64)
    tree = cKDTree(pos_np)
    pairs = tree.query_pairs(r=config.cutoff, output_type="ndarray")

    if len(pairs) > 0:
        # Symmetric: add both directions
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        diffs = pos_np[src] - pos_np[dst]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
    else:
        src = np.array([], dtype=np.int64)
        dst = np.array([], dtype=np.int64)
        dists = np.array([], dtype=np.float64)

    # Cap neighbors per node
    if config.max_neighbors > 0 and len(src) > 0:
        keep: list[int] = []
        for node in range(num_atoms):
            mask = src == node
            idx = np.where(mask)[0]
            if len(idx) > config.max_neighbors:
                order = np.argsort(dists[idx])[: config.max_neighbors]
                keep.extend(idx[order].tolist())
            else:
                keep.extend(idx.tolist())
        keep_arr = np.array(keep, dtype=np.int64)
        src, dst, dists = src[keep_arr], dst[keep_arr], dists[keep_arr]

    # Edge features: sequence separation + same-residue flag
    res_arr = np.array(residue_ids, dtype=np.int64)
    if len(src) > 0:
        res_src = res_arr[src]
        res_dst = res_arr[dst]
        seq_sep = np.clip(np.abs(res_src - res_dst), 0, config.max_seq_sep)
        same_res = (res_src == res_dst).astype(np.float32)
    else:
        seq_sep = np.array([], dtype=np.int64)
        same_res = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Convert to tensors
    # ------------------------------------------------------------------
    pos_t = torch.tensor(positions, dtype=torch.float32)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long) if len(src) > 0 else torch.zeros(2, 0, dtype=torch.long)
    edge_dist = torch.tensor(dists, dtype=torch.float32)
    seq_sep_t = torch.tensor(seq_sep, dtype=torch.long)
    same_res_t = torch.tensor(same_res, dtype=torch.float32)

    element_t = torch.tensor(element_ids, dtype=torch.long)
    aa_t = torch.tensor(aa_ids, dtype=torch.long)
    role_t = torch.tensor(role_ids, dtype=torch.long)
    residue_idx_t = torch.tensor(residue_ids, dtype=torch.long)
    bfactor_t = torch.tensor(bfactors, dtype=torch.float32)

    target_indices = torch.tensor(target_atom_map, dtype=torch.long)  # [R, 6]

    # ------------------------------------------------------------------
    # Shift labels
    # ------------------------------------------------------------------
    target_shifts = torch.full((num_residues, 6), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros(num_residues, 6, dtype=torch.bool)

    # Atom-valid mask: True if the target atom exists in this structure
    atom_valid = torch.tensor(target_atom_valid, dtype=torch.bool)  # [R, 6]

    if shifts is not None:
        if seq_mapping is not None:
            _assign_shifts_mapped(
                shifts, seq_mapping, residue_seqid, atom_valid,
                target_shifts, target_mask,
            )
        else:
            _assign_shifts(
                shifts, residue_seqid, residue_aa, atom_valid,
                target_shifts, target_mask,
            )
    else:
        # No labels — target_mask = atom_valid (for inference)
        target_mask = atom_valid.clone()

    residue_types = torch.tensor(residue_aa, dtype=torch.long)
    seq_ids_t = torch.tensor(residue_seqid, dtype=torch.long)
    disulfide_t = torch.tensor(residue_disulfide, dtype=torch.bool)
    plddt_t = torch.tensor(plddt_per_residue, dtype=torch.float32)  # NaN if not AF

    data = ProteinData(
        pos=pos_t,
        element=element_t,
        amino_acid=aa_t,
        atom_role=role_t,
        residue_idx=residue_idx_t,
        bfactor=bfactor_t,
        edge_index=edge_index,
        edge_dist=edge_dist,
        seq_sep=seq_sep_t,
        same_residue=same_res_t,
        target_indices=target_indices,
        atom_valid=atom_valid,
        target_mask=target_mask,
        target_shifts=target_shifts,
        residue_types=residue_types,
        seq_ids=seq_ids_t,
        num_atoms=num_atoms,
        num_residues=num_residues,
        disulfide=disulfide_t,
        plddt=plddt_t,
    )
    data._atom_names = atom_names  # stored for geometry computation, not batched
    data._is_alphafold = bool(_is_af2)  # diagnostic flag
    return data


def _assign_shifts(
    shifts: dict[int, dict[str, float]],
    pdb_seqids: list[int],
    pdb_aa_ids: list[int],
    atom_valid: "torch.Tensor",
    target_shifts: "torch.Tensor",
    target_mask: "torch.Tensor",
) -> None:
    """Assign shift labels to PDB residues, handling numbering mismatches.

    Strategy:
    1. Try direct seqid key match.
    2. If that fails, detect constant offset (PDB_seqid - BMRB_seqid).
    3. If that also fails, align by sequential position with AA-name verification.
    """
    import torch

    num_residues = len(pdb_seqids)
    bmrb_seqids = sorted(shifts.keys())
    if not bmrb_seqids:
        return

    # --- Pass 1: direct seqid match ---
    matched = 0
    for ri, sid in enumerate(pdb_seqids):
        if sid in shifts:
            for nj, nuc in enumerate(BACKBONE_NUCLEI):
                if nuc in shifts[sid] and atom_valid[ri, nj]:
                    target_shifts[ri, nj] = shifts[sid][nuc]
                    target_mask[ri, nj] = True
                    matched += 1
    if matched > 0:
        return

    # --- Pass 2: constant offset ---
    if len(pdb_seqids) > 0 and len(bmrb_seqids) > 0:
        offset = pdb_seqids[0] - bmrb_seqids[0]
        rematched = 0
        for ri, pdb_sid in enumerate(pdb_seqids):
            bmrb_sid = pdb_sid - offset
            if bmrb_sid in shifts:
                for nj, nuc in enumerate(BACKBONE_NUCLEI):
                    if nuc in shifts[bmrb_sid] and atom_valid[ri, nj]:
                        target_shifts[ri, nj] = shifts[bmrb_sid][nuc]
                        target_mask[ri, nj] = True
                        rematched += 1
        if rematched > 0:
            return

    # --- Pass 3: sequential position matching ---
    # Walk BMRB shifts in order, match to PDB residues by position
    bmrb_ordered = [shifts[s] for s in bmrb_seqids]
    n_match = min(len(bmrb_ordered), num_residues)
    for ri in range(n_match):
        nuc_dict = bmrb_ordered[ri]
        for nj, nuc in enumerate(BACKBONE_NUCLEI):
            if nuc in nuc_dict and atom_valid[ri, nj]:
                target_shifts[ri, nj] = nuc_dict[nuc]
                target_mask[ri, nj] = True


def _assign_shifts_mapped(
    shifts: dict[int, dict[str, float]],
    seq_mapping: dict[int, int],
    pdb_seqids: list[int],
    atom_valid: "torch.Tensor",
    target_shifts: "torch.Tensor",
    target_mask: "torch.Tensor",
) -> None:
    """Assign shift labels using a pre-validated PDB→BMRB sequence mapping.

    Unlike ``_assign_shifts``, this does not guess offsets or fall back to
    positional matching.  Every assignment comes from the Needleman-Wunsch
    alignment, so mismatched residues are silently skipped (they won't
    appear in ``seq_mapping``).
    """
    for ri, pdb_sid in enumerate(pdb_seqids):
        bmrb_sid = seq_mapping.get(pdb_sid)
        if bmrb_sid is None:
            continue
        nuc_dict = shifts.get(bmrb_sid)
        if nuc_dict is None:
            continue
        for nj, nuc in enumerate(BACKBONE_NUCLEI):
            if nuc in nuc_dict and atom_valid[ri, nj]:
                target_shifts[ri, nj] = nuc_dict[nuc]
                target_mask[ri, nj] = True


def compute_residue_geometry(data: Any) -> "torch.Tensor":
    """Compute per-residue geometry features from stored atomic data.

    D6 layout — 34 physics-informed features. See features.py for the
    full slot map.  Works on cached graphs — no structure file required
    (DSSP is pre-stored on graph at build time).
    """
    import torch
    import numpy as np
    from caustic.features import NUM_GEO_FEATURES

    R = int(data.num_residues)
    geo = torch.zeros(R, NUM_GEO_FEATURES, dtype=torch.float32)

    pos = data.pos             # [N_atoms, 3]
    roles = data.atom_role     # [N_atoms]  N=0 CA=1 C=2 O=3 CB=4 H=5 HA=6 sc_heavy=7 sc_H=8
    elements = data.element    # [N_atoms]  C=0 N=1 O=2 S=3 H=4
    res_idx = data.residue_idx # [N_atoms]
    atom_names = getattr(data, "_atom_names", None)

    # Build per-residue backbone lookup: {res_i: {role: atom_idx}}
    bb: dict[int, dict[int, int]] = {}
    for ai in range(pos.size(0)):
        ri = int(res_idx[ai])
        role = int(roles[ai])
        if role <= 6:  # include H=5 and HA=6 for physics features
            bb.setdefault(ri, {}).setdefault(role, ai)

    # Build per-residue atom-name lookup for chi angles
    res_atoms: dict[int, dict[str, int]] = {}
    if atom_names is not None:
        for ai, name in enumerate(atom_names):
            ri = int(res_idx[ai])
            res_atoms.setdefault(ri, {}).setdefault(name, ai)

    # Chi1 gamma atoms per AA type
    _CHI1_GAMMA: dict[int, tuple[str, ...]] = {
        0: (), 1: ("CG",), 2: ("CG",), 3: ("CG",), 4: ("SG",),
        5: ("CG",), 6: ("CG",), 7: (), 8: ("CG",), 9: ("CG1",),
        10: ("CG",), 11: ("CG",), 12: ("CG",), 13: ("CG",), 14: (),
        15: ("OG",), 16: ("OG1",), 17: ("CG",), 18: ("CG",), 19: ("CG1",),
    }

    # Helper: dihedral sin/cos from 4 atom indices
    def _dih(i1: int, i2: int, i3: int, i4: int) -> tuple[float, float]:
        p1, p2, p3, p4 = pos[i1], pos[i2], pos[i3], pos[i4]
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        n1n = n1.norm().clamp(min=1e-8)
        n2n = n2.norm().clamp(min=1e-8)
        cos_v = torch.dot(n1, n2) / (n1n * n2n)
        b2n = b2.norm().clamp(min=1e-8)
        m1 = torch.linalg.cross(n1 / n1n, b2 / b2n)
        sin_v = torch.dot(m1, n2) / (m1.norm().clamp(min=1e-8) * n2n)
        return sin_v.item(), cos_v.item()

    # ------------------------------------------------------------------
    # [0-7] Torsion angles (phi, psi, omega, chi1)
    # ------------------------------------------------------------------
    for ri in range(R):
        a = bb.get(ri, {})
        prev = bb.get(ri - 1, {})
        nxt = bb.get(ri + 1, {})

        # [0-1] phi: C(i-1) - N(i) - CA(i) - C(i)
        if 2 in prev and 0 in a and 1 in a and 2 in a:
            s, c = _dih(prev[2], a[0], a[1], a[2])
            geo[ri, 0], geo[ri, 1] = s, c

        # [2-3] psi: N(i) - CA(i) - C(i) - N(i+1)
        if 0 in a and 1 in a and 2 in a and 0 in nxt:
            s, c = _dih(a[0], a[1], a[2], nxt[0])
            geo[ri, 2], geo[ri, 3] = s, c

        # [4] cos(omega)
        if 1 in prev and 2 in prev and 0 in a and 1 in a:
            _, c = _dih(prev[1], prev[2], a[0], a[1])
            geo[ri, 4] = c

        # [5-7] chi1: N - CA - CB - Xgamma
        aa_type = int(data.residue_types[ri]) if hasattr(data, "residue_types") else -1
        ra = res_atoms.get(ri, {})
        if 0 in a and 1 in a and 4 in a and aa_type in _CHI1_GAMMA:
            for gname in _CHI1_GAMMA[aa_type]:
                if gname in ra:
                    s, c = _dih(a[0], a[1], a[4], ra[gname])
                    geo[ri, 5], geo[ri, 6] = s, c
                    geo[ri, 7] = 1.0
                    break

    # ------------------------------------------------------------------
    # [8] Disulfide flag
    # ------------------------------------------------------------------
    if hasattr(data, "disulfide"):
        geo[:, 8] = data.disulfide.float()

    # ------------------------------------------------------------------
    # [9-11] Ensemble features (phi/psi circular variance, num_conformers)
    # ------------------------------------------------------------------
    if hasattr(data, "ensemble_phi_circvar") and data.ensemble_phi_circvar is not None:
        geo[:, 9] = data.ensemble_phi_circvar
    if hasattr(data, "ensemble_psi_circvar") and data.ensemble_psi_circvar is not None:
        geo[:, 10] = data.ensemble_psi_circvar
    if hasattr(data, "ensemble_n_conformers") and data.ensemble_n_conformers is not None:
        geo[:, 11] = min(float(data.ensemble_n_conformers) / 20.0, 1.0)

    # ------------------------------------------------------------------
    # [12-17] Ring current shifts (Haigh-Mallion)
    # ------------------------------------------------------------------
    if atom_names is not None and hasattr(data, "target_indices"):
        try:
            from caustic.physics_features import compute_ring_currents
            pos_np = pos.numpy()
            rc = compute_ring_currents(
                pos_np, res_idx.numpy(),
                data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.array(data.residue_types),
                atom_names,
                data.target_indices.numpy(),
                data.target_mask.numpy().astype(bool),
                R,
            )
            geo[:, 12:18] = torch.from_numpy(rc)
        except Exception:
            pass  # leave as zeros

    # ------------------------------------------------------------------
    # [18-24] Hydrogen bond geometry
    # ------------------------------------------------------------------
    try:
        from caustic.physics_features import compute_hbond_geometry
        pos_np = pos.numpy() if not isinstance(pos, np.ndarray) else pos
        hb = compute_hbond_geometry(pos_np, roles.numpy(), res_idx.numpy(), R, bb)
        geo[:, 18:25] = torch.from_numpy(hb)
    except Exception:
        # Default: no bonds (d=1.0, angles=0, present=0)
        geo[:, 18] = 1.0
        geo[:, 22] = 1.0

    # ------------------------------------------------------------------
    # [25-27] rSASA + Half-sphere exposure
    # ------------------------------------------------------------------
    try:
        from caustic.physics_features import compute_sasa_and_hse
        sasa = compute_sasa_and_hse(
            pos, elements, res_idx,
            data.residue_types, bb, R,
            n_sphere_points=92,
        )
        geo[:, 25:28] = torch.from_numpy(sasa)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # [28-29] Buckingham electric field
    # ------------------------------------------------------------------
    if atom_names is not None:
        try:
            from caustic.physics_features import compute_electric_field
            ef = compute_electric_field(
                pos.numpy() if hasattr(pos, "numpy") else pos,
                res_idx.numpy() if hasattr(res_idx, "numpy") else res_idx,
                data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.array(data.residue_types),
                atom_names, bb, R,
            )
            geo[:, 28:30] = torch.from_numpy(ef)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # [30] DSSP secondary structure (pre-stored on graph at build time)
    # ------------------------------------------------------------------
    if hasattr(data, "dssp_3state") and data.dssp_3state is not None:
        dssp = data.dssp_3state
        if len(dssp) >= R:
            geo[:, 30] = dssp[:R].float()

    # ------------------------------------------------------------------
    # [31-33] Deuteration vector (sample-level broadcast)
    # ------------------------------------------------------------------
    if hasattr(data, "deuteration") and data.deuteration is not None:
        dv = data.deuteration  # [3]
        geo[:, 31] = dv[0]  # frac_amide
        geo[:, 32] = dv[1]  # frac_alpha
        geo[:, 33] = dv[2]  # frac_sidechain

    return geo


def compute_target_environment(data: Any) -> "torch.Tensor":
    """Compute per-target-atom local environment features from stored graph data.

    Returns [R, 6, NUM_TARGET_ENV_FEATURES] tensor.  See
    ``features.py`` for the 15-slot layout.
    """
    import torch
    import numpy as np
    from caustic.physics_features import compute_target_atom_environment

    R = int(data.num_residues)
    pos_np = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    elem_np = data.element.numpy() if hasattr(data.element, "numpy") else np.asarray(data.element)
    role_np = data.atom_role.numpy() if hasattr(data.atom_role, "numpy") else np.asarray(data.atom_role)
    ridx_np = data.residue_idx.numpy() if hasattr(data.residue_idx, "numpy") else np.asarray(data.residue_idx)
    rtypes_np = data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.asarray(data.residue_types)
    ti_np = data.target_indices.numpy() if hasattr(data.target_indices, "numpy") else np.asarray(data.target_indices)
    tm_np = (data.target_mask.numpy() if hasattr(data.target_mask, "numpy") else np.asarray(data.target_mask)).astype(bool)
    ds_np = (data.disulfide.numpy() if hasattr(data, "disulfide") and data.disulfide is not None
             else np.zeros(R, dtype=bool))

    atom_names = getattr(data, "_atom_names", None)

    env = compute_target_atom_environment(
        pos_np, elem_np, role_np, ridx_np, rtypes_np,
        atom_names, ti_np, tm_np, ds_np, R,
    )
    return torch.from_numpy(env)


def count_models(structure_path: str | Path) -> int:
    """Count conformer models in a structure file."""
    return len(_load_structure(Path(structure_path)))


def compute_ensemble_features(
    structure_path: str | Path,
    chain_id: str | None = None,
    max_conformers: int = 20,
) -> dict:
    """Extract ensemble statistics from a multi-model NMR structure.

    Computes the medoid model index, per-residue CA RMSF, and per-residue
    torsion circular variance across all conformers.

    Args:
        structure_path: Path to .pdb or .cif file.
        chain_id: Chain to extract (None = first polymer chain).
        max_conformers: Cap on number of conformers to process.

    Returns:
        Dict with keys:
        - ``medoid_idx``: int — model index of the most representative conformer
        - ``num_conformers``: int — number of models used
        - ``ca_rmsf``: ndarray [R] — per-residue CA RMSF in Angstroms
        - ``phi_circvar``: ndarray [R] — per-residue phi circular variance (0-1)
        - ``psi_circvar``: ndarray [R] — per-residue psi circular variance (0-1)
        - ``chi1_circvar``: ndarray [R] — per-residue chi1 circular variance (0-1)
        - ``chi1_has_data``: ndarray [R] — bool, whether chi1 was computable
    """
    import gemmi  # used for gemmi.EntityType.Polymer below

    path = Path(structure_path)
    st = _load_structure(path)
    st.setup_entities()

    n_models = min(len(st), max_conformers)
    if n_models <= 1:
        return {"num_conformers": n_models, "medoid_idx": 0}

    # ------------------------------------------------------------------
    # Find the target chain
    # ------------------------------------------------------------------
    if chain_id is None:
        for ch in st[0]:
            for res in ch:
                if res.entity_type == gemmi.EntityType.Polymer:
                    chain_id = ch.name
                    break
            if chain_id:
                break
    if chain_id is None:
        return {"num_conformers": 1, "medoid_idx": 0}

    # ------------------------------------------------------------------
    # Extract CA positions + backbone torsion atoms across all conformers
    # ------------------------------------------------------------------
    # Build residue list from model 0
    ref_chain = st[0][chain_id]
    residues = [r for r in ref_chain if r.entity_type == gemmi.EntityType.Polymer]
    R = len(residues)

    # Chi1 gamma atoms per AA type
    _chi1_gamma = {
        "ARG": "CG", "ASN": "CG", "ASP": "CG", "CYS": "SG",
        "GLN": "CG", "GLU": "CG", "HIS": "CG", "ILE": "CG1",
        "LEU": "CG", "LYS": "CG", "MET": "CG", "PHE": "CG",
        "SER": "OG", "THR": "OG1", "TRP": "CG", "TYR": "CG", "VAL": "CG1",
    }

    # Per-conformer arrays
    ca_coords = np.full((n_models, R, 3), np.nan, dtype=np.float64)
    # Torsion angles: store sin/cos pairs to handle circular mean properly
    phi_sin = np.full((n_models, R), np.nan)
    phi_cos = np.full((n_models, R), np.nan)
    psi_sin = np.full((n_models, R), np.nan)
    psi_cos = np.full((n_models, R), np.nan)
    chi1_sin = np.full((n_models, R), np.nan)
    chi1_cos = np.full((n_models, R), np.nan)

    def _get_pos(residue, atom_name):
        """Get atom position as numpy array, or None."""
        atom = residue.find_atom(atom_name, "\0")
        if atom is None:
            # Try common alternatives
            if atom_name == "H":
                atom = residue.find_atom("HN", "\0")
                if atom is None:
                    atom = residue.find_atom("H1", "\0")
            elif atom_name == "HA":
                atom = residue.find_atom("HA2", "\0")
        if atom is None:
            return None
        return np.array([atom.pos.x, atom.pos.y, atom.pos.z])

    def _dihedral(p1, p2, p3, p4):
        """Compute dihedral angle in radians from 4 positions."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-8 or n2_norm < 1e-8:
            return np.nan
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        cos_v = np.clip(np.dot(n1, n2), -1.0, 1.0)
        b2n = b2 / max(np.linalg.norm(b2), 1e-8)
        m1 = np.cross(n1, b2n)
        sin_v = np.dot(m1, n2)
        return np.arctan2(sin_v, cos_v)

    for mi in range(n_models):
        try:
            chain = st[mi][chain_id]
        except (KeyError, IndexError):
            continue

        model_residues = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        # Match by seqid to handle models with slightly different residue counts
        seqid_to_idx = {residues[ri].seqid.num: ri for ri in range(R)}

        for mr in model_residues:
            ri = seqid_to_idx.get(mr.seqid.num)
            if ri is None:
                continue

            # CA position
            ca = _get_pos(mr, "CA")
            if ca is not None:
                ca_coords[mi, ri] = ca

            # Backbone atoms for torsions
            n_pos = _get_pos(mr, "N")
            ca_pos = ca
            c_pos = _get_pos(mr, "C")

            # Phi: C(i-1) - N(i) - CA(i) - C(i)
            if ri > 0:
                prev_seqid = residues[ri - 1].seqid.num
                prev_res = None
                for pr in model_residues:
                    if pr.seqid.num == prev_seqid:
                        prev_res = pr
                        break
                if prev_res is not None:
                    prev_c = _get_pos(prev_res, "C")
                    if prev_c is not None and n_pos is not None and ca_pos is not None and c_pos is not None:
                        angle = _dihedral(prev_c, n_pos, ca_pos, c_pos)
                        if not np.isnan(angle):
                            phi_sin[mi, ri] = np.sin(angle)
                            phi_cos[mi, ri] = np.cos(angle)

            # Psi: N(i) - CA(i) - C(i) - N(i+1)
            if ri < R - 1:
                nxt_seqid = residues[ri + 1].seqid.num
                nxt_res = None
                for nr in model_residues:
                    if nr.seqid.num == nxt_seqid:
                        nxt_res = nr
                        break
                if nxt_res is not None:
                    nxt_n = _get_pos(nxt_res, "N")
                    if n_pos is not None and ca_pos is not None and c_pos is not None and nxt_n is not None:
                        angle = _dihedral(n_pos, ca_pos, c_pos, nxt_n)
                        if not np.isnan(angle):
                            psi_sin[mi, ri] = np.sin(angle)
                            psi_cos[mi, ri] = np.cos(angle)

            # Chi1: N - CA - CB - Xgamma
            gamma_name = _chi1_gamma.get(mr.name)
            if gamma_name is not None:
                cb_pos = _get_pos(mr, "CB")
                gamma_pos = _get_pos(mr, gamma_name)
                if n_pos is not None and ca_pos is not None and cb_pos is not None and gamma_pos is not None:
                    angle = _dihedral(n_pos, ca_pos, cb_pos, gamma_pos)
                    if not np.isnan(angle):
                        chi1_sin[mi, ri] = np.sin(angle)
                        chi1_cos[mi, ri] = np.cos(angle)

    # ------------------------------------------------------------------
    # Medoid selection: conformer with lowest mean CA RMSD to all others
    # ------------------------------------------------------------------
    # Superpose all models onto model 0 using CA atoms
    # First find residues with CA in all models
    ca_valid = np.all(~np.isnan(ca_coords[:, :, 0]), axis=0)  # [R]
    n_valid_ca = ca_valid.sum()

    if n_valid_ca < 3:
        return {"num_conformers": n_models, "medoid_idx": 0}

    ca_common = ca_coords[:, ca_valid, :]  # [M, R_valid, 3]

    # Kabsch superposition onto model 0
    ref = ca_common[0].copy()
    ref_center = ref.mean(axis=0)
    ref_centered = ref - ref_center

    aligned = np.zeros_like(ca_common)
    aligned[0] = ref_centered

    for mi in range(1, n_models):
        coords = ca_common[mi].copy()
        center = coords.mean(axis=0)
        centered = coords - center
        # Kabsch rotation
        H = centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_mat = np.diag([1.0, 1.0, d])
        R_mat = Vt.T @ sign_mat @ U.T
        aligned[mi] = centered @ R_mat.T

    # Pairwise RMSD
    rmsd_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = aligned[i] - aligned[j]
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    mean_rmsd = rmsd_matrix.mean(axis=1)
    medoid_idx = int(np.argmin(mean_rmsd))

    # ------------------------------------------------------------------
    # Per-residue CA RMSF (relative to ensemble mean after superposition)
    # ------------------------------------------------------------------
    # Expand aligned back to full residue set
    ca_rmsf = np.zeros(R, dtype=np.float64)
    # For valid residues, compute RMSF from aligned coords
    mean_pos = aligned.mean(axis=0)  # [R_valid, 3]
    deviations = aligned - mean_pos[None, :, :]  # [M, R_valid, 3]
    rmsf_valid = np.sqrt(np.mean(np.sum(deviations ** 2, axis=-1), axis=0))  # [R_valid]

    valid_indices = np.where(ca_valid)[0]
    for idx_i, ri in enumerate(valid_indices):
        ca_rmsf[ri] = rmsf_valid[idx_i]

    # ------------------------------------------------------------------
    # Circular variance for torsion angles
    # ------------------------------------------------------------------
    def _circ_var(sin_arr, cos_arr):
        """Circular variance from sin/cos arrays. Returns array [R]."""
        result = np.zeros(R)
        for ri in range(R):
            s_vals = sin_arr[:, ri]
            c_vals = cos_arr[:, ri]
            valid = ~np.isnan(s_vals)
            n = valid.sum()
            if n < 2:
                continue
            mean_sin = s_vals[valid].mean()
            mean_cos = c_vals[valid].mean()
            R_len = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
            result[ri] = 1.0 - R_len  # circular variance: 0 = no spread, 1 = uniform
        return result

    phi_circvar = _circ_var(phi_sin, phi_cos)
    psi_circvar = _circ_var(psi_sin, psi_cos)
    chi1_circvar = _circ_var(chi1_sin, chi1_cos)
    chi1_has_data = np.any(~np.isnan(chi1_sin), axis=0)  # [R]

    return {
        "medoid_idx": medoid_idx,
        "num_conformers": n_models,
        "ca_rmsf": ca_rmsf,
        "phi_circvar": phi_circvar,
        "psi_circvar": psi_circvar,
        "chi1_circvar": chi1_circvar,
        "chi1_has_data": chi1_has_data,
    }
