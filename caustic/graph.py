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
    AROMATIC_RINGS,
    BACKBONE_NUCLEI,
    ELEMENT_TO_IDX,
    NUM_EDGE_CHEM_FEATURES,
    NUM_RESIDUE_CHEM_FEATURES,
    NUM_TARGET_CHEM_FEATURES,
    NUM_TARGET_NC_CHEM_FEATURES,
    NUM_ATOM_ROLES,
    NUM_ELEMENT_TYPES,
    RING_ELEMENT_IDX,
    ROLE_LIGAND_HEAVY,
    ROLE_METAL,
    ROLE_RING,
    ROLE_WATER,
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
    from .features import ELEMENT_TO_IDX, ATOM_ROLE_MAP

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
                # CB missing (non-GLY) — build a proper tetrahedral HA
                # position pointing opposite the (phantom) CB, which sits
                # on the opposite side of the N-CA-C plane from HA.
                # Use the same in-plane bisector as GLY, but nudge along
                # the plane normal in the direction that mirrors the
                # typical CB side (positive normal). This approximates the
                # true sp3 geometry rather than placing HA in the N-CA-C
                # plane (which would put it colinear with the CB socket).
                plane_n = _unit(np.cross(n_pos - ca_pos, c_pos - ca_pos))
                in_plane = _unit(-(v_n + v_c))
                direction = _unit(in_plane * 0.577 + plane_n * 0.816)
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
            if key in (
                "num_atoms",
                "num_residues",
                "n_real_atoms",
                "n_ring_nodes",
                "n_hetero_nodes",
                "num_hetero_context_atoms",
            ):
                return None
            return super().__cat_dim__(key, value, *args, **kw)

except ImportError:
    ProteinData = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# d47_v1 features: aromatic ring virtual nodes + per-edge H-bond labels.
# See `reports/caustic/inspect_neighborhood_batch_synthesis.md` for the
# motivation: 70% of worst-predicted residues sit near an aromatic ring;
# 33% have a long-range / cross-strand H-bond partner. Both are visible in
# the geometry but unlabelled, forcing the head to extract them implicitly.
# ---------------------------------------------------------------------------
def _detect_ring_nodes(
    n_real_atoms: int,
    residue_aa: list[int],
    residue_ids: list[int],
    atom_names: list[str],
    positions: list[tuple[float, float, float]],
    bfactors: list[float],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]],
           list[int], list[int], list[float]]:
    """Find every complete aromatic ring on every aromatic residue and
    compute its centroid + plane normal.

    Returns five parallel lists (one entry per detected ring):
      ring_centroids, ring_normals, ring_residue_idxs, ring_aa_ids, ring_bfactors

    Ring detection uses ``AROMATIC_RINGS`` to look up atom-name templates
    per AA. A ring is emitted only if every named atom is present (matching
    by atom name within the residue's atoms).

    Idx alignment: ``residue_ids`` is per real atom (length = n_real_atoms).
    The returned ``ring_residue_idxs`` reuses the same residue indices so
    each ring node is bound to its parent residue (used by `same_residue`
    edge feature etc.).
    """
    # 3-letter AA code lookup. AA_THREE_TO_IDX maps "ALA"→0 + many
    # non-standard variants (HIP/HSE/HSD/... → 8, HYP → 14, etc.). For
    # ring lookup we want the CANONICAL 3-letter code per index (the one
    # AROMATIC_RINGS is keyed on), so we hard-code that 0..19 list.
    _CANONICAL_AA_THREE = (
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL",
    )
    def idx_to_canonical(i: int) -> str | None:
        return _CANONICAL_AA_THREE[i] if 0 <= i < 20 else None

    ring_centroids: list[tuple[float, float, float]] = []
    ring_normals: list[tuple[float, float, float]] = []
    ring_residue_idxs: list[int] = []
    ring_aa_ids: list[int] = []
    ring_bfactors: list[float] = []

    # Build a per-residue index of (atom_name -> atom position list index)
    by_residue: dict[int, dict[str, int]] = {}
    for i in range(n_real_atoms):
        ri = residue_ids[i]
        nm = atom_names[i]
        # Skip H-prefixed atoms — none of the aromatic ring atoms are H.
        # Also handle mismatch on duplicate atom names by keeping the first.
        if ri not in by_residue:
            by_residue[ri] = {}
        if nm not in by_residue[ri]:
            by_residue[ri][nm] = i

    for ri, name_to_idx in by_residue.items():
        aa_idx = residue_aa[ri] if ri < len(residue_aa) else None
        if aa_idx is None:
            continue
        aa_three = idx_to_canonical(int(aa_idx))
        if aa_three is None or aa_three not in AROMATIC_RINGS:
            continue
        for ring_atom_names in AROMATIC_RINGS[aa_three]:
            try:
                idxs = [name_to_idx[name] for name in ring_atom_names]
            except KeyError:
                continue  # incomplete ring (missing atom) — skip
            coords = np.asarray([positions[i] for i in idxs], dtype=np.float64)
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            # Plane normal via SVD: smallest singular value's right vector.
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            normal = vh[-1]
            n_norm = float(np.linalg.norm(normal))
            if n_norm < 1e-9:
                continue
            normal = normal / n_norm
            mean_b = float(np.mean([bfactors[i] for i in idxs]))
            ring_centroids.append((float(centroid[0]), float(centroid[1]), float(centroid[2])))
            ring_normals.append((float(normal[0]), float(normal[1]), float(normal[2])))
            ring_residue_idxs.append(int(ri))
            ring_aa_ids.append(int(aa_idx))
            ring_bfactors.append(mean_b)

    return ring_centroids, ring_normals, ring_residue_idxs, ring_aa_ids, ring_bfactors


def _detect_hbonds_per_edge(
    src: np.ndarray,
    dst: np.ndarray,
    dists: np.ndarray,
    positions: np.ndarray,
    element_ids: list[int],
    atom_names: list[str],
    n_real_atoms: int,
    max_DA: float = 3.6,
    min_cos_angle: float = -0.5,  # corresponds to angle > 120 degrees
) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge H-bond labels.

    Returns:
      is_hbond [E] float (0 or 1)
      hb_cos   [E] float (cos of donor-H..acceptor angle, 0 if not HB)

    Algorithm:
      1. Identify candidate donors (heavy atoms with at least one bonded H
         within 1.2 A) and acceptors (any non-RING N or O).
      2. For each non-self heavy donor-acceptor pair where d <= max_DA, find
         the donor's H closest to the acceptor (treating any H within 1.3 A
         of the donor as bonded), compute angle(D-H, A-H), and gate on the
         cosine threshold (cos angle <= 0 means angle >= 90; we want
         > 120 → cos <= -0.5).
      3. Flag any edge in the graph that connects (D, A) or (A, D) of an
         identified H-bond.

    Both backbone (N-H...O) and sidechain (e.g. OH-...O, NH-...N) H-bonds
    are flagged uniformly. The model can learn role-specific behaviour via
    the per-atom role embedding it already has.

    NOTE: ring virtual nodes (atom indices >= n_real_atoms) are excluded
    from H-bond detection — they have no H and are not real chemistry.
    """
    n_atoms = len(positions)
    is_hbond = np.zeros(len(src), dtype=np.float32)
    hb_cos = np.zeros(len(src), dtype=np.float32)
    if n_atoms == 0 or len(src) == 0:
        return is_hbond, hb_cos

    # Element-id 4 = "H" in features.ELEMENT_TO_IDX. Ring nodes carry
    # element RING_ELEMENT_IDX (5).
    elem_arr = np.asarray(element_ids, dtype=np.int64)
    H_ELEM = ELEMENT_TO_IDX["H"]
    N_ELEM = ELEMENT_TO_IDX["N"]
    O_ELEM = ELEMENT_TO_IDX["O"]

    real_mask = np.arange(n_atoms) < n_real_atoms
    is_h = (elem_arr == H_ELEM) & real_mask
    is_n_or_o = ((elem_arr == N_ELEM) | (elem_arr == O_ELEM)) & real_mask

    # Build "bonded H" map: for each heavy atom (N or O), find the closest
    # H within 1.3 A.
    # Use cKDTree for the ~1 A search.
    from scipy.spatial import cKDTree

    h_idxs = np.where(is_h)[0]
    heavy_idxs = np.where(is_n_or_o)[0]
    bonded_h_for_heavy: dict[int, int] = {}
    if h_idxs.size > 0 and heavy_idxs.size > 0:
        h_tree = cKDTree(positions[h_idxs])
        for hi in heavy_idxs:
            d, j = h_tree.query(positions[hi], k=1, distance_upper_bound=1.3)
            if np.isfinite(d) and j < len(h_idxs):
                bonded_h_for_heavy[int(hi)] = int(h_idxs[j])

    # For every edge that connects two N/O atoms (real, non-ring) within
    # max_DA, classify donor/acceptor and check angle.
    # An atom is a "donor" if it has a bonded H. Otherwise it's an
    # "acceptor". Both ends might be donors (e.g. SER OG ↔ THR OG, both have
    # bonded H) — in that case we try both donor-acceptor orderings and
    # take the better-aligned one.
    src_is_no = is_n_or_o[src]
    dst_is_no = is_n_or_o[dst]
    candidate = src_is_no & dst_is_no & (dists <= max_DA) & (src != dst)
    cand_idx = np.where(candidate)[0]

    for ei in cand_idx:
        a = int(src[ei])
        b = int(dst[ei])
        d = float(dists[ei])
        # Try a as donor first
        best_cos = 1.0  # angle 0 = bad (along bond), we want close to -1
        h_a = bonded_h_for_heavy.get(a)
        h_b = bonded_h_for_heavy.get(b)
        if h_a is not None:
            d_vec = positions[a] - positions[h_a]
            a_vec = positions[b] - positions[h_a]
            n1 = np.linalg.norm(d_vec) + 1e-12
            n2 = np.linalg.norm(a_vec) + 1e-12
            cos_dha = float(np.dot(d_vec, a_vec) / (n1 * n2))
            if cos_dha < best_cos:
                best_cos = cos_dha
        if h_b is not None:
            d_vec = positions[b] - positions[h_b]
            a_vec = positions[a] - positions[h_b]
            n1 = np.linalg.norm(d_vec) + 1e-12
            n2 = np.linalg.norm(a_vec) + 1e-12
            cos_dha = float(np.dot(d_vec, a_vec) / (n1 * n2))
            if cos_dha < best_cos:
                best_cos = cos_dha
        if best_cos <= min_cos_angle:
            is_hbond[ei] = 1.0
            hb_cos[ei] = best_cos

    return is_hbond, hb_cos


_METAL_ELEMENTS: frozenset[str] = frozenset({
    "Li", "Na", "K", "Rb", "Cs",
    "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb",
})
_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "H2O", "DOD"})
_HEME_RESIDUE_NAMES: frozenset[str] = frozenset({"HEM", "HEC", "HEA", "HEB"})
_OXIDIZED_CYS_NAMES: frozenset[str] = frozenset({"CYX", "CSO", "CSD", "OCS", "CME", "SEC"})

# CYS state codes stored on data.cys_state. 0 = not CYS.
_CYS_REDUCED = 1
_CYS_DISULFIDE = 2
_CYS_METAL = 3
_CYS_THIOETHER = 4
_CYS_OXIDIZED = 5


def _is_metal_element(element_name: str) -> bool:
    return element_name.strip().capitalize() in _METAL_ELEMENTS


def _hetero_atom_key(chain_name: str, residue: Any, atom: Any) -> tuple[str, str, str, str, str]:
    """Stable key for matching selected hetero atoms across conformers."""
    altloc = getattr(atom, "altloc", "")
    return (
        str(chain_name),
        str(residue.seqid),
        residue.name.strip().upper(),
        atom.name.strip(),
        str(altloc).strip(),
    )


def _hetero_priority(
    *,
    resname: str,
    element_name: str,
    atom_name: str,
    is_metal: bool,
    is_water: bool,
    covalent_residue: bool,
    nearest_dist: float,
) -> int:
    """Lower is higher priority for promotion to explicit hetero nodes."""
    if is_metal:
        return 0
    if covalent_residue or resname in _HEME_RESIDUE_NAMES:
        return 1
    el = element_name.strip().upper()
    name = atom_name.strip().upper()
    if (el in {"N", "O", "S", "P"} or name in {"NZ", "OD1", "OD2", "OE1", "OE2"}) and nearest_dist <= 4.0:
        return 2
    if (not is_water) and el in {"C", "N", "O", "S", "P"} and nearest_dist <= 5.0:
        return 3
    if is_water:
        return 5
    return 4


def _is_charged_sidechain_atom(aa_idx: int, atom_name: str) -> bool:
    name = atom_name.strip().upper()
    # ASP/GLU carboxylates, ARG guanidinium, LYS ammonium, partially
    # protonated HIS nitrogens.
    if aa_idx == 3 and name in {"OD1", "OD2"}:
        return True
    if aa_idx == 6 and name in {"OE1", "OE2"}:
        return True
    if aa_idx == 1 and name in {"NE", "CZ", "NH1", "NH2"}:
        return True
    if aa_idx == 11 and name == "NZ":
        return True
    if aa_idx == 8 and name in {"ND1", "NE2"}:
        return True
    return False


def _is_positive_sidechain_atom(aa_idx: int, atom_name: str) -> bool:
    name = atom_name.strip().upper()
    if aa_idx == 1 and name in {"NE", "CZ", "NH1", "NH2"}:
        return True
    if aa_idx == 11 and name == "NZ":
        return True
    if aa_idx == 8 and name in {"ND1", "NE2"}:
        return True
    return False


def _compute_edge_chemistry(
    src: np.ndarray,
    dst: np.ndarray,
    dists: np.ndarray,
    *,
    roles: list[int],
    elements: list[int],
    aa_ids: list[int],
    residue_ids: list[int],
    atom_names: list[str],
    node_is_hetero: list[bool],
    node_is_metal: list[bool],
    n_real_atoms: int,
    is_hbond: np.ndarray,
) -> np.ndarray:
    """Compute d49 typed edge chemistry features.

    The features are deliberately broad, low-dimensional motifs rather than a
    high-cardinality chemistry vocabulary. They tell the trunk which close
    contacts are chemically special while preserving the geometry in the
    existing distance/RBF/vector channels.
    """
    edge_chem = np.zeros((len(src), NUM_EDGE_CHEM_FEATURES), dtype=np.float32)
    if len(src) == 0:
        return edge_chem

    O_EL = ELEMENT_TO_IDX["O"]
    N_EL = ELEMENT_TO_IDX["N"]
    S_EL = ELEMENT_TO_IDX["S"]
    polar_el = {O_EL, N_EL}
    backbone_roles = {0, 1, 2, 3, 5, 6}

    for ei, (a, b, d) in enumerate(zip(src, dst, dists)):
        ai = int(a)
        bi = int(b)
        if ai >= n_real_atoms or bi >= n_real_atoms:
            continue

        ra = int(roles[ai])
        rb = int(roles[bi])
        ea = int(elements[ai])
        eb = int(elements[bi])
        aa_a = int(aa_ids[ai])
        aa_b = int(aa_ids[bi])
        sep = abs(int(residue_ids[ai]) - int(residue_ids[bi]))
        het = bool(node_is_hetero[ai]) or bool(node_is_hetero[bi])
        metal = bool(node_is_metal[ai]) or bool(node_is_metal[bi])
        charged = (
            _is_charged_sidechain_atom(aa_a, atom_names[ai])
            or _is_charged_sidechain_atom(aa_b, atom_names[bi])
        )
        polar_pair = (ea in polar_el and eb in polar_el and d <= 3.6)
        sidechain_involved = (ra not in backbone_roles) or (rb not in backbone_roles)
        sulfur_contact = (ea == S_EL or eb == S_EL) and d <= 4.0

        if polar_pair:
            edge_chem[ei, 0] = 1.0
        if bool(is_hbond[ei]) and ra in backbone_roles and rb in backbone_roles:
            edge_chem[ei, 1] = 1.0
        if (polar_pair or (charged and d <= 4.0)) and sidechain_involved:
            edge_chem[ei, 2] = 1.0
        if charged and d <= 4.0:
            edge_chem[ei, 3] = 1.0
        if metal and d <= 3.2:
            edge_chem[ei, 4] = 1.0
        if sep >= 5 and (polar_pair or charged or metal or sulfur_contact):
            edge_chem[ei, 5] = 1.0
        if sulfur_contact:
            edge_chem[ei, 6] = 1.0
        if het and d <= 4.5:
            edge_chem[ei, 7] = 1.0

    return edge_chem


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
    metal_bound_seqids: set[int] = set()
    metal_coord_counts: dict[int, int] = {}
    covalent_ligand_seqids: set[int] = set()
    covalent_hetero_residue_keys: set[tuple[str, str]] = set()
    thioether_seqids: set[int] = set()
    for conn in st.connections:
        if conn.type == gemmi.ConnectionType.Disulf:
            p1, p2 = conn.partner1, conn.partner2
            if p1.res_id.seqid.num and (not chain_id or p1.chain_name == chain_id):
                disulfide_seqids.add(p1.res_id.seqid.num)
            if p2.res_id.seqid.num and (not chain_id or p2.chain_name == chain_id):
                disulfide_seqids.add(p2.res_id.seqid.num)
        elif conn.type == gemmi.ConnectionType.MetalC:
            for p in (conn.partner1, conn.partner2):
                if p.res_id.seqid.num and (not chain_id or p.chain_name == chain_id):
                    metal_bound_seqids.add(p.res_id.seqid.num)
                    metal_coord_counts[p.res_id.seqid.num] = (
                        metal_coord_counts.get(p.res_id.seqid.num, 0) + 1
                    )
        elif conn.type != gemmi.ConnectionType.Disulf:
            # Mark polymer residues with any explicit covalent/non-metal
            # connection to a non-polymer partner. This captures thioethers,
            # PTM links, and other ligand-state chemistry without guessing
            # labels from chemical shifts.
            p1, p2 = conn.partner1, conn.partner2
            if p1.res_id.seqid.num and p1.chain_name == chain_id:
                covalent_ligand_seqids.add(p1.res_id.seqid.num)
            if p2.res_id.seqid.num and p2.chain_name == chain_id:
                covalent_ligand_seqids.add(p2.res_id.seqid.num)
            for p in (p1, p2):
                if p.res_id.seqid.num:
                    covalent_hetero_residue_keys.add(
                        (str(p.chain_name), str(p.res_id.seqid)),
                    )

    # ------------------------------------------------------------------
    # Detect CYS thioether bonds to heme prosthetic groups
    # ------------------------------------------------------------------
    # Cytochrome c proteins have CYS residues forming thioether bonds
    # (C-S-C) to a heme group (HEM, HEC, or similar HETATM). These shift
    # CYS CB from ~29 ppm to ~37-39 ppm, similar to a disulfide. Flag them
    # as is_disulfide=1 so the model uses the correct CB prior.
    _THIOETHER_CUTOFF_SQ = 5.0 * 5.0  # 5 A search radius, squared

    # Collect CYS SG positions from the target polymer chain
    _cys_sg: list[tuple[int, float, float, float]] = []  # (seqid, x, y, z)
    for residue in chain:
        if residue.entity_type != gemmi.EntityType.Polymer:
            continue
        if residue.name.strip().upper() != "CYS":
            continue
        for atom in residue:
            if atom.name.strip().upper() == "SG":
                _cys_sg.append((residue.seqid.num, atom.pos.x, atom.pos.y, atom.pos.z))
                break

    if _cys_sg:
        # Collect heme heavy-atom positions across ALL chains
        _heme_positions: list[tuple[float, float, float]] = []
        for ch in model:
            for residue in ch:
                if residue.name.strip().upper() in _HEME_RESIDUE_NAMES:
                    for atom in residue:
                        _heme_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
        if _heme_positions:
            for seqid, sx, sy, sz in _cys_sg:
                for hx, hy, hz in _heme_positions:
                    d2 = (sx - hx) ** 2 + (sy - hy) ** 2 + (sz - hz) ** 2
                    if d2 < _THIOETHER_CUTOFF_SQ:
                        disulfide_seqids.add(seqid)
                        thioether_seqids.add(seqid)
                        logger.info(
                            "CYS %d: thioether bond to heme detected (%.1f A) — "
                            "flagging as is_disulfide=1",
                            seqid, d2 ** 0.5,
                        )
                        break  # one heme contact is enough

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
    node_is_hetero: list[bool] = []
    node_is_metal: list[bool] = []
    plddt_per_residue: list[float] = []  # mean CA pLDDT per residue (NaN if not AF)

    residue_aa: list[int] = []
    residue_seqid: list[int] = []
    residue_disulfide: list[bool] = []  # True if CYS in SS bond
    residue_cys_state: list[int] = []
    residue_metal_coord_count: list[int] = []
    residue_covalent_ligand: list[bool] = []
    residue_nonstandard: list[bool] = []
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
        is_cys = aa_idx == 4
        is_disulf = residue.seqid.num in disulfide_seqids and is_cys
        residue_disulfide.append(is_disulf)  # CYS=4
        if not is_cys:
            residue_cys_state.append(0)
        elif residue.seqid.num in metal_bound_seqids:
            residue_cys_state.append(_CYS_METAL)
        elif residue.seqid.num in thioether_seqids:
            residue_cys_state.append(_CYS_THIOETHER)
        elif is_disulf:
            residue_cys_state.append(_CYS_DISULFIDE)
        elif raw_name in _OXIDIZED_CYS_NAMES or is_nonstandard:
            residue_cys_state.append(_CYS_OXIDIZED)
        else:
            residue_cys_state.append(_CYS_REDUCED)
        residue_metal_coord_count.append(int(metal_coord_counts.get(residue.seqid.num, 0)))
        residue_covalent_ligand.append(residue.seqid.num in covalent_ligand_seqids)
        residue_nonstandard.append(bool(is_nonstandard))
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
            node_is_hetero.append(False)
            node_is_metal.append(False)

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
            _pre_synth_n = len(positions)
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
            if len(positions) > _pre_synth_n:
                node_is_hetero.extend([False] * (len(positions) - _pre_synth_n))
                node_is_metal.extend([False] * (len(positions) - _pre_synth_n))
            if added > 0:
                logger.info(
                    "Synthesised %d backbone hydrogens (%d H, %d HA missing) for %s",
                    added, n_missing_h, n_missing_ha, path.name,
                )
            num_atoms = len(positions)

    # ------------------------------------------------------------------
    # d50: non-polymer heavy-atom context and capped node promotion.
    # ------------------------------------------------------------------
    hetero_context_pos: list[list[float]] = []
    hetero_context_element: list[int] = []
    hetero_context_role: list[int] = []
    hetero_context_is_metal: list[bool] = []
    hetero_context_priority: list[int] = []
    hetero_context_nearest_residue: list[int] = []
    hetero_context_nearest_dist: list[float] = []
    hetero_context_is_water: list[bool] = []
    hetero_context_keys: list[tuple[str, str, str, str, str]] = []
    hetero_node_keys: list[tuple[str, str, str, str, str]] = []
    n_hetero_nodes = 0
    if (
        getattr(config, "include_hetero_nodes", False)
        or getattr(config, "collect_hetero_context", False)
    ):
        try:
            polymer_pos = np.asarray(positions[:num_atoms], dtype=np.float64)
            polymer_resids = np.asarray(residue_ids[:num_atoms], dtype=np.int64)
            hetero_radius = float(getattr(config, "hetero_radius", config.cutoff))
            summary_radius = float(
                getattr(config, "hetero_summary_radius", hetero_radius),
            )
            collect_radius = max(hetero_radius, summary_radius)
            include_water_nodes = bool(getattr(config, "include_water_nodes", False))
            max_nodes = int(getattr(config, "max_hetero_nodes", 0) or 0)
            max_nodes_per_residue = int(
                getattr(config, "max_hetero_nodes_per_residue", 0) or 0,
            )
            candidates: list[dict[str, Any]] = []
            seen_hetero: set[tuple[str, str, str, str, str]] = set()
            for ch in model:
                for residue in ch:
                    if residue.entity_type == gemmi.EntityType.Polymer:
                        continue
                    resname = residue.name.strip().upper()
                    is_water = resname in _WATER_RESNAMES
                    covalent_residue = (
                        (str(ch.name), str(residue.seqid))
                        in covalent_hetero_residue_keys
                    )
                    for atom in residue:
                        if atom.is_hydrogen():
                            continue
                        key = _hetero_atom_key(ch.name, residue, atom)
                        if key in seen_hetero:
                            continue
                        seen_hetero.add(key)
                        pos = atom.pos
                        xyz = np.asarray([pos.x, pos.y, pos.z], dtype=np.float64)
                        if polymer_pos.size:
                            d_all = np.linalg.norm(polymer_pos - xyz, axis=1)
                            nearest_i = int(np.argmin(d_all))
                            nearest_dist = float(d_all[nearest_i])
                            nearest_residue = int(polymer_resids[nearest_i])
                        else:
                            nearest_dist = float("inf")
                            nearest_residue = -1
                        if nearest_dist > collect_radius:
                            continue
                        el_name = atom.element.name
                        is_metal = _is_metal_element(el_name)
                        if is_metal:
                            role = ROLE_METAL
                        elif is_water:
                            role = ROLE_WATER
                        else:
                            role = ROLE_LIGAND_HEAVY
                        priority = _hetero_priority(
                            resname=resname,
                            element_name=el_name,
                            atom_name=atom.name,
                            is_metal=is_metal,
                            is_water=is_water,
                            covalent_residue=covalent_residue,
                            nearest_dist=nearest_dist,
                        )
                        cand = {
                            "key": key,
                            "xyz": [float(pos.x), float(pos.y), float(pos.z)],
                            "element": ELEMENT_TO_IDX.get(el_name, UNK_ELEMENT_IDX),
                            "role": role,
                            "is_metal": is_metal,
                            "priority": priority,
                            "nearest_residue": nearest_residue,
                            "nearest_dist": nearest_dist,
                            "is_water": is_water,
                            "bfactor": 100.0 - atom.b_iso if _is_af2 else atom.b_iso,
                            "atom_name": atom.name.strip(),
                        }
                        candidates.append(cand)
                        if nearest_dist <= summary_radius:
                            hetero_context_pos.append(cand["xyz"])
                            hetero_context_element.append(int(cand["element"]))
                            hetero_context_role.append(int(cand["role"]))
                            hetero_context_is_metal.append(bool(cand["is_metal"]))
                            hetero_context_priority.append(int(cand["priority"]))
                            hetero_context_nearest_residue.append(int(cand["nearest_residue"]))
                            hetero_context_nearest_dist.append(float(cand["nearest_dist"]))
                            hetero_context_is_water.append(bool(cand["is_water"]))
                            hetero_context_keys.append(key)

            if getattr(config, "include_hetero_nodes", False) and candidates:
                selected: list[dict[str, Any]] = []
                per_residue_counts: dict[int, int] = {}
                for cand in sorted(
                    candidates,
                    key=lambda c: (
                        int(c["priority"]),
                        float(c["nearest_dist"]),
                        str(c["key"]),
                    ),
                ):
                    if float(cand["nearest_dist"]) > hetero_radius:
                        continue
                    if bool(cand["is_water"]) and not include_water_nodes:
                        continue
                    ri_near = int(cand["nearest_residue"])
                    if (
                        max_nodes_per_residue > 0
                        and per_residue_counts.get(ri_near, 0) >= max_nodes_per_residue
                    ):
                        continue
                    selected.append(cand)
                    per_residue_counts[ri_near] = per_residue_counts.get(ri_near, 0) + 1
                    if max_nodes > 0 and len(selected) >= max_nodes:
                        break

                for hetero_i, cand in enumerate(selected):
                    positions.append(cand["xyz"])
                    element_ids.append(int(cand["element"]))
                    aa_ids.append(UNK_AA_IDX)
                    role_ids.append(int(cand["role"]))
                    # Keep hetero residue ids outside the polymer range.
                    # Persistent features loop over [0, R), while edge
                    # features already store seq_sep/same_residue.
                    residue_ids.append(num_residues + hetero_i)
                    bfactors.append(float(cand["bfactor"]))
                    atom_names.append(str(cand["atom_name"]))
                    node_is_hetero.append(True)
                    node_is_metal.append(bool(cand["is_metal"]))
                    hetero_node_keys.append(cand["key"])
                n_hetero_nodes = len(selected)
                if n_hetero_nodes:
                    logger.info(
                        "Included %d/%d nearby hetero/non-polymer atoms for %s "
                        "(summary_context=%d)",
                        n_hetero_nodes, len(candidates), path.name,
                        len(hetero_context_pos),
                    )
                    num_atoms = len(positions)
        except Exception as exc:
            logger.warning("Failed to collect/include hetero context for %s: %s", path.name, exc)

    # ------------------------------------------------------------------
    # d47_v1: aromatic ring virtual nodes (feature A).
    # Insert one centroid node per detected aromatic ring (PHE/TYR/TRP/HIS).
    # Each ring node carries:
    #   - position = ring centroid
    #   - element  = RING_ELEMENT_IDX (5)
    #   - atom_role = ROLE_RING (10)
    #   - amino_acid = parent residue's AA index
    #   - residue_idx = parent residue's idx (so seq_sep / same_residue work)
    # The ring's plane normal is stored separately in ``node_normal`` and
    # used by the model's PaiNN trunk to initialise the ring node's
    # equivariant vector channel — see model.py PaiNN forward.
    #
    # d48: gated on GraphConfig.include_ring_nodes. MACE configs set this
    # False so structure_to_graph produces real-only graphs at training and
    # inference (no adapter strip-and-rebuild). PaiNN configs default True
    # (no behaviour change vs d47).
    # ------------------------------------------------------------------
    n_real_atoms = num_atoms
    if getattr(config, "include_ring_nodes", True):
        ring_centroids, ring_normals_list, ring_residue_idxs, ring_aa_ids, ring_bfactors = (
            _detect_ring_nodes(
                n_real_atoms, residue_aa, residue_ids, atom_names, positions, bfactors,
            )
        )
    else:
        ring_centroids, ring_normals_list, ring_residue_idxs, ring_aa_ids, ring_bfactors = (
            [], [], [], [], [],
        )
    n_rings = len(ring_centroids)
    if n_rings > 0:
        positions.extend(ring_centroids)
        element_ids.extend([RING_ELEMENT_IDX] * n_rings)
        aa_ids.extend(ring_aa_ids)
        role_ids.extend([ROLE_RING] * n_rings)
        residue_ids.extend(ring_residue_idxs)
        bfactors.extend(ring_bfactors)
        atom_names.extend([f"RING{i}" for i in range(n_rings)])
        node_is_hetero.extend([False] * n_rings)
        node_is_metal.extend([False] * n_rings)
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

    # Cap neighbors per node — symmetric version. Bug Q fix: the prior
    # src-only cap could drop (A,B) while keeping (B,A), producing a
    # directed-asymmetric subgraph that silently biased PaiNN message
    # passing on large dense proteins. Here we rank each node's edges
    # from BOTH sides (as source and as destination) and keep an edge
    # only if both endpoints include it in their top-K. Since edges
    # come in symmetric pairs, this yields a symmetric subgraph with
    # ≤ K neighbors per node.
    if config.max_neighbors > 0 and len(src) > 0:
        K = config.max_neighbors

        def _per_group_rank(keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
            """Within each unique key, rank vals ascending (0 = smallest).

            Equivalent to ``df.groupby(keys)[vals].rank()`` but vectorized.
            """
            order = np.lexsort((vals, keys))
            ranks = np.empty(len(order), dtype=np.int64)
            if len(order) == 0:
                return ranks
            sorted_keys = keys[order]
            new_group = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
            group_ids = np.cumsum(new_group) - 1
            group_starts = np.where(new_group)[0]
            within = np.arange(len(order)) - group_starts[group_ids]
            ranks[order] = within
            return ranks

        src_ranks = _per_group_rank(src, dists)
        dst_ranks = _per_group_rank(dst, dists)
        keep_mask = (src_ranks < K) & (dst_ranks < K)
        src = src[keep_mask]
        dst = dst[keep_mask]
        dists = dists[keep_mask]

    # Edge features: sequence separation + same-residue flag
    res_arr = np.array(residue_ids, dtype=np.int64)
    if len(src) > 0:
        res_src = res_arr[src]
        res_dst = res_arr[dst]
        seq_sep = np.clip(np.abs(res_src - res_dst), 0, config.max_seq_sep)
        hetero_arr = np.asarray(node_is_hetero, dtype=bool)
        same_res = (
            (res_src == res_dst)
            & (~hetero_arr[src])
            & (~hetero_arr[dst])
        ).astype(np.float32)
    else:
        seq_sep = np.array([], dtype=np.int64)
        same_res = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # d47_v1: per-edge H-bond labels (features B + C)
    # ------------------------------------------------------------------
    is_hbond_arr, hb_cos_arr = _detect_hbonds_per_edge(
        src, dst, dists, pos_np, element_ids, atom_names, n_real_atoms,
    )
    edge_chem_arr = _compute_edge_chemistry(
        src,
        dst,
        dists,
        roles=role_ids,
        elements=element_ids,
        aa_ids=aa_ids,
        residue_ids=residue_ids,
        atom_names=atom_names,
        node_is_hetero=node_is_hetero,
        node_is_metal=node_is_metal,
        n_real_atoms=n_real_atoms,
        is_hbond=is_hbond_arr,
    )

    # d47_v1: per-node ring-normal vector (zero for real atoms, plane normal
    # for ring nodes). Used by model.py to initialise PaiNN's equivariant
    # vector channel for ring centroids — gives the head an explicit
    # ring-axis direction, which is the missing physics behind the
    # 70%-aromatic-proximity failure mode in the worst-residue audit.
    node_normal_arr = np.zeros((num_atoms, 3), dtype=np.float32)
    for i, normal in enumerate(ring_normals_list):
        node_normal_arr[n_real_atoms + i] = normal

    # ------------------------------------------------------------------
    # Convert to tensors
    # ------------------------------------------------------------------
    pos_t = torch.tensor(positions, dtype=torch.float32)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long) if len(src) > 0 else torch.zeros(2, 0, dtype=torch.long)
    edge_dist = torch.tensor(dists, dtype=torch.float32)
    seq_sep_t = torch.tensor(seq_sep, dtype=torch.long)
    same_res_t = torch.tensor(same_res, dtype=torch.float32)
    node_normal_t = torch.from_numpy(node_normal_arr)
    is_hbond_t = torch.from_numpy(is_hbond_arr)
    hb_cos_t = torch.from_numpy(hb_cos_arr)
    edge_chem_t = torch.from_numpy(edge_chem_arr)

    element_t = torch.tensor(element_ids, dtype=torch.long)
    aa_t = torch.tensor(aa_ids, dtype=torch.long)
    role_t = torch.tensor(role_ids, dtype=torch.long)
    residue_idx_t = torch.tensor(residue_ids, dtype=torch.long)
    bfactor_t = torch.tensor(bfactors, dtype=torch.float32)
    node_is_hetero_t = torch.tensor(node_is_hetero, dtype=torch.bool)
    node_is_metal_t = torch.tensor(node_is_metal, dtype=torch.bool)
    if hetero_context_pos:
        hetero_context_pos_t = torch.tensor(hetero_context_pos, dtype=torch.float32)
        hetero_context_element_t = torch.tensor(hetero_context_element, dtype=torch.long)
        hetero_context_role_t = torch.tensor(hetero_context_role, dtype=torch.long)
        hetero_context_is_metal_t = torch.tensor(hetero_context_is_metal, dtype=torch.bool)
        hetero_context_priority_t = torch.tensor(hetero_context_priority, dtype=torch.long)
        hetero_context_nearest_residue_t = torch.tensor(
            hetero_context_nearest_residue, dtype=torch.long,
        )
        hetero_context_nearest_dist_t = torch.tensor(
            hetero_context_nearest_dist, dtype=torch.float32,
        )
        hetero_context_is_water_t = torch.tensor(hetero_context_is_water, dtype=torch.bool)
    else:
        hetero_context_pos_t = torch.zeros((0, 3), dtype=torch.float32)
        hetero_context_element_t = torch.zeros((0,), dtype=torch.long)
        hetero_context_role_t = torch.zeros((0,), dtype=torch.long)
        hetero_context_is_metal_t = torch.zeros((0,), dtype=torch.bool)
        hetero_context_priority_t = torch.zeros((0,), dtype=torch.long)
        hetero_context_nearest_residue_t = torch.zeros((0,), dtype=torch.long)
        hetero_context_nearest_dist_t = torch.zeros((0,), dtype=torch.float32)
        hetero_context_is_water_t = torch.zeros((0,), dtype=torch.bool)

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
    cys_state_t = torch.tensor(residue_cys_state, dtype=torch.long)
    metal_coord_count_t = torch.tensor(residue_metal_coord_count, dtype=torch.float32)
    covalent_ligand_t = torch.tensor(residue_covalent_ligand, dtype=torch.bool)
    residue_nonstandard_t = torch.tensor(residue_nonstandard, dtype=torch.bool)
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
        node_normal=node_normal_t,
        is_hbond=is_hbond_t,
        hb_cos=hb_cos_t,
        edge_chemistry=edge_chem_t,
        n_real_atoms=int(n_real_atoms),
        n_ring_nodes=int(n_rings),
        n_hetero_nodes=int(n_hetero_nodes),
        num_hetero_context_atoms=int(len(hetero_context_pos)),
        node_is_hetero=node_is_hetero_t,
        node_is_metal=node_is_metal_t,
        hetero_context_pos=hetero_context_pos_t,
        hetero_context_element=hetero_context_element_t,
        hetero_context_role=hetero_context_role_t,
        hetero_context_is_metal=hetero_context_is_metal_t,
        hetero_context_priority=hetero_context_priority_t,
        hetero_context_nearest_residue=hetero_context_nearest_residue_t,
        hetero_context_nearest_dist=hetero_context_nearest_dist_t,
        hetero_context_is_water=hetero_context_is_water_t,
        target_indices=target_indices,
        atom_valid=atom_valid,
        target_mask=target_mask,
        target_shifts=target_shifts,
        residue_types=residue_types,
        seq_ids=seq_ids_t,
        num_atoms=num_atoms,
        num_residues=num_residues,
        disulfide=disulfide_t,
        cys_state=cys_state_t,
        metal_coord_count=metal_coord_count_t,
        covalent_ligand=covalent_ligand_t,
        residue_nonstandard=residue_nonstandard_t,
        plddt=plddt_t,
    )
    data._atom_names = atom_names  # stored for geometry computation, not batched
    data._hetero_node_keys = hetero_node_keys
    data._hetero_context_keys = hetero_context_keys
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
    3. If neither pass matches anything, leave the label mask empty and
       rely on the caller to supply a validated seq_mapping (typically
       from Needleman-Wunsch) via :func:`_assign_shifts_mapped`.

    Removed 2026-04-22: a "Pass 3 sequential fallback" that blindly
    assigned BMRB[i] -> PDB[i] without amino-acid verification. Its
    docstring claimed "with AA-name verification" but the code never
    used ``pdb_aa_ids``; any entry whose sequences didn't align got
    silently mis-labelled. Pass 3 is not restored because ``dataset.py``
    already passes a validated ``seq_mapping`` for every training entry,
    making the fallback both unnecessary and dangerous.
    """
    import torch

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

    # --- No match ---
    # Both passes failed. Leave target_mask empty so the caller knows
    # the alignment requires a validated seq_mapping.
    logger.warning(
        "Shift-to-PDB alignment failed on both direct and offset passes; "
        "no labels assigned. Supply a seq_mapping to bypass this heuristic."
    )


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

    D6 layout — 40 physics-informed features (extended from 34 with
    continuous chi1 and aromatic-neighbor chi1/chi2). See features.py
    for the full slot map.  Works on cached graphs — no structure file
    required (DSSP is pre-stored on graph at build time).
    """
    import torch
    import numpy as np
    from .features import NUM_GEO_FEATURES

    R = int(data.num_residues)
    geo = torch.zeros(R, NUM_GEO_FEATURES, dtype=torch.float32)

    pos = data.pos             # [N_atoms, 3]
    roles = data.atom_role     # [N_atoms]  N=0 CA=1 C=2 O=3 CB=4 H=5 HA=6 sc_heavy=7 sc_H=8
    elements = data.element    # [N_atoms]  C=0 N=1 O=2 S=3 H=4
    res_idx = data.residue_idx # [N_atoms]
    atom_names = getattr(data, "_atom_names", None)

    # d47_v1: ring virtual nodes are appended at the tail. Slice to real
    # atoms so SASA, ring-current, electric-field and other per-atom
    # physics computations don't see the virtual centroids.
    n_real = int(getattr(data, "n_real_atoms", pos.size(0)))
    if n_real < pos.size(0):
        pos = pos[:n_real]
        roles = roles[:n_real]
        elements = elements[:n_real]
        res_idx = res_idx[:n_real]
        if atom_names is not None and len(atom_names) > n_real:
            atom_names = atom_names[:n_real]

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
            from .physics_features import compute_ring_currents
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
        from .physics_features import compute_hbond_geometry
        pos_np = pos.numpy() if not isinstance(pos, np.ndarray) else pos
        hb, hb_partner_dirs = compute_hbond_geometry(
            pos_np, roles.numpy(), res_idx.numpy(), R, bb,
            return_partner_dirs=True,
        )
        geo[:, 18:25] = torch.from_numpy(hb)
        # D42: H-bond partner directions in slots [46-51].
        if geo.shape[-1] >= 52:
            geo[:, 46:52] = torch.from_numpy(hb_partner_dirs)
    except Exception:
        # Default: no bonds (d=1.0, angles=0, present=0)
        geo[:, 18] = 1.0
        geo[:, 22] = 1.0

    # ------------------------------------------------------------------
    # [25-27] rSASA + Half-sphere exposure
    # ------------------------------------------------------------------
    # Track the atom-level rSASA produced by the Shrake-Rupley pass so a
    # caller that also needs target_atom_rsasa can avoid a second
    # Shrake-Rupley run. Stashed as a closure-local here and exposed via
    # ``_last_atom_rsasa_full`` on the function object so the thin public
    # entry points can read it without changing ``compute_residue_geometry``'s
    # return signature.
    _atom_rsasa_out: "np.ndarray | None" = None
    try:
        from .physics_features import compute_sasa_and_hse
        sasa, _atom_rsasa_out = compute_sasa_and_hse(
            pos, elements, res_idx,
            data.residue_types, bb, R,
            n_sphere_points=92,
        )
        geo[:, 25:28] = torch.from_numpy(sasa)
    except Exception:
        _atom_rsasa_out = None

    # ------------------------------------------------------------------
    # [28-29] Buckingham electric field
    # ------------------------------------------------------------------
    if atom_names is not None:
        try:
            from .physics_features import compute_electric_field
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

    # ------------------------------------------------------------------
    # [34-35] Continuous chi1 sin/cos (gamma-gauche effect)
    # ------------------------------------------------------------------
    # Reuses the chi1 computation from [5-6] but stores in separate slots
    # so existing indices 0-33 stay stable for backward compatibility.
    for ri in range(R):
        aa_type = int(data.residue_types[ri]) if hasattr(data, "residue_types") else -1
        ra = res_atoms.get(ri, {})
        a = bb.get(ri, {})
        if 0 in a and 1 in a and 4 in a and aa_type in _CHI1_GAMMA:
            for gname in _CHI1_GAMMA[aa_type]:
                if gname in ra:
                    s, c = _dih(a[0], a[1], a[4], ra[gname])
                    geo[ri, 34], geo[ri, 35] = s, c
                    break

    # ------------------------------------------------------------------
    # [36-39] Nearest aromatic neighbor chi1/chi2
    # ------------------------------------------------------------------
    # For each residue, find the nearest aromatic (PHE=13, TYR=18,
    # TRP=17, HIS=8) within 8 A (CA-CA distance) and store its
    # sidechain orientation as sin/cos chi1 + chi2.
    _AROMATIC_AA: frozenset[int] = frozenset({8, 13, 17, 18})

    # chi2 delta atom name per aromatic AA type:
    #   PHE: CA-CB-CG-CD1, TYR: CA-CB-CG-CD1, TRP: CA-CB-CG-CD1,
    #   HIS: CA-CB-CG-ND1
    _CHI2_DELTA: dict[int, str] = {
        8: "ND1",   # HIS
        13: "CD1",  # PHE
        17: "CD1",  # TRP
        18: "CD1",  # TYR
    }

    # Collect aromatic residue indices and their CA positions
    arom_residues: list[int] = []
    arom_ca_pos: list[np.ndarray] = []
    for ri in range(R):
        aa_type = int(data.residue_types[ri]) if hasattr(data, "residue_types") else -1
        if aa_type in _AROMATIC_AA:
            a = bb.get(ri, {})
            if 1 in a:  # has CA
                arom_residues.append(ri)
                arom_ca_pos.append(pos[a[1]].numpy() if hasattr(pos[a[1]], "numpy") else np.asarray(pos[a[1]]))

    if arom_residues:
        arom_ca_arr = np.stack(arom_ca_pos)  # [N_arom, 3]
        for ri in range(R):
            a = bb.get(ri, {})
            if 1 not in a:  # no CA for this residue
                continue
            ca_pos_ri = pos[a[1]].numpy() if hasattr(pos[a[1]], "numpy") else np.asarray(pos[a[1]])
            dists = np.sqrt(((arom_ca_arr - ca_pos_ri) ** 2).sum(axis=1))
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] > 8.0:
                continue  # no aromatic within cutoff
            arom_ri = arom_residues[nearest_idx]
            arom_aa = int(data.residue_types[arom_ri]) if hasattr(data, "residue_types") else -1
            arom_a = bb.get(arom_ri, {})
            arom_ra = res_atoms.get(arom_ri, {})

            # Aromatic neighbor chi1: N-CA-CB-Xgamma
            if 0 in arom_a and 1 in arom_a and 4 in arom_a and arom_aa in _CHI1_GAMMA:
                for gname in _CHI1_GAMMA[arom_aa]:
                    if gname in arom_ra:
                        s, c = _dih(arom_a[0], arom_a[1], arom_a[4], arom_ra[gname])
                        geo[ri, 36], geo[ri, 37] = s, c
                        break

            # Aromatic neighbor chi2: CA-CB-CG-XD
            delta_name = _CHI2_DELTA.get(arom_aa)
            gamma_name = _CHI1_GAMMA.get(arom_aa, ("",))[0] if arom_aa in _CHI1_GAMMA else ""
            if (
                delta_name
                and gamma_name
                and 1 in arom_a
                and 4 in arom_a
                and gamma_name in arom_ra
                and delta_name in arom_ra
            ):
                s, c = _dih(arom_a[1], arom_a[4], arom_ra[gamma_name], arom_ra[delta_name])
                geo[ri, 38], geo[ri, 39] = s, c

    # D32 — slots [40-45]: terminal flags + disulfide geometry.
    # Ensure pos_np is in scope even if earlier try-blocks failed.
    try:
        _ = pos_np  # defined by the hbond block above on the happy path
    except NameError:
        pos_np = pos.numpy() if not isinstance(pos, np.ndarray) else pos

    # Terminal flags are chain-aware: slot 40 = first polymer residue of
    # the extracted chain (always the case for the canonical build path,
    # since structure_to_graph already filters to one chain and only
    # polymer residues). Slot 41 = last polymer residue AND has an OXT
    # atom (C-terminal carboxylate; without OXT it's a chain break, not
    # a true terminus).
    if R > 0:
        geo[0, 40] = 1.0  # is_n_terminal
        if atom_names is not None:
            ra = res_atoms.get(R - 1, {})
            if "OXT" in ra:
                geo[R - 1, 41] = 1.0

    # Slots [42-45]: disulfide geometry for CYS-in-SS-bond residues.
    # Identify the disulfide pair from gemmi's stored connections OR by
    # falling back to SG-SG proximity within 2.6 Å (empirical cutoff).
    # When a residue flags `data.disulfide=True` but we can't locate a
    # partner, leave slots as zero.
    if hasattr(data, "disulfide") and data.disulfide is not None and atom_names is not None:
        disulf = data.disulfide.numpy() if hasattr(data.disulfide, "numpy") else np.asarray(data.disulfide)
        # Collect SG positions for every CYS residue in this chain
        cys_sg: dict[int, tuple[int, np.ndarray]] = {}
        for ai, name in enumerate(atom_names):
            if name != "SG":
                continue
            ri_atom = int(res_idx[ai])
            if 0 <= ri_atom < R:
                cys_sg[ri_atom] = (ai, pos_np[ai])
        # For each disulfide CYS, find partner by nearest SG within 2.6 Å
        for ri in range(R):
            if not disulf[ri]:
                continue
            if ri not in cys_sg:
                continue
            own_ai, own_pos = cys_sg[ri]
            best_j = None
            best_d = 2.6  # disulfide S-S is ~2.05 Å, cutoff gives some slack
            for rj, (aj, pj) in cys_sg.items():
                if rj == ri:
                    continue
                d = float(np.linalg.norm(own_pos - pj))
                if d < best_d:
                    best_d = d
                    best_j = rj
            if best_j is None:
                continue
            # SG-SG distance (normalized by 3.0 so typical 2.05 -> 0.68)
            geo[ri, 42] = float(best_d / 3.0)
            # CB-SG-SG'-CB' dihedral (chi_SS)
            own_cb = bb.get(ri, {}).get(4)  # CB role = 4
            partner_cb = bb.get(best_j, {}).get(4)
            partner_sg_ai = cys_sg[best_j][0]
            if own_cb is not None and partner_cb is not None:
                _, chi_ss_cos = _dih(own_cb, own_ai, partner_sg_ai, partner_cb)
                geo[ri, 43] = chi_ss_cos
                # CB-SG-SG' bond angle (cos)
                v_cbsg = pos_np[own_ai] - pos_np[own_cb]
                v_sgsg = pos_np[partner_sg_ai] - pos_np[own_ai]
                n1 = float(np.linalg.norm(v_cbsg))
                n2 = float(np.linalg.norm(v_sgsg))
                if n1 > 1e-6 and n2 > 1e-6:
                    geo[ri, 44] = float(np.dot(v_cbsg, v_sgsg) / (n1 * n2))
            # Sequence separation |i - j| / 50
            geo[ri, 45] = float(abs(ri - best_j) / 50.0)

    # Stash the atom-level rSASA array on the function object so
    # ``compute_persistent_features`` below can reuse it without
    # re-running Shrake-Rupley. The attribute is set per-call from the
    # calling thread and read immediately; it is NOT a thread-safe cache.
    compute_residue_geometry._last_atom_rsasa_full = _atom_rsasa_out  # type: ignore[attr-defined]

    return geo


def compute_target_environment(data: Any) -> "torch.Tensor":
    """Compute per-target-atom local environment features from stored graph data.

    Returns [R, 6, NUM_TARGET_ENV_FEATURES] tensor.  See
    ``features.py`` for the 15-slot layout.
    """
    import torch
    import numpy as np
    from .physics_features import compute_target_atom_environment

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

    # d47_v1: ring virtual nodes are appended at the tail of the per-atom
    # arrays. Per-target environment counts shells of REAL atoms only —
    # ring nodes are message-passing aids, not chemistry.
    n_real = int(getattr(data, "n_real_atoms", len(pos_np)))
    pos_np = pos_np[:n_real]
    elem_np = elem_np[:n_real]
    role_np = role_np[:n_real]
    ridx_np = ridx_np[:n_real]

    atom_names = getattr(data, "_atom_names", None)
    if atom_names is not None and len(atom_names) > n_real:
        atom_names = atom_names[:n_real]

    env = compute_target_atom_environment(
        pos_np, elem_np, role_np, ridx_np, rtypes_np,
        atom_names, ti_np, tm_np, ds_np, R,
    )
    return torch.from_numpy(env)


def _hetero_context_arrays(
    data: Any,
    *,
    pos_real: np.ndarray,
    elem_real: np.ndarray,
    role_real: np.ndarray | None = None,
    node_het: np.ndarray,
    node_metal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact hetero context arrays, falling back to hetero nodes.

    d50 caches store all nearby hetero atoms separately from the capped graph
    nodes so summary features keep seeing the full local chemistry. Older d49
    caches lack those tensors; for them, the explicit hetero nodes remain the
    only available context.
    """
    import numpy as np

    ctx_pos = getattr(data, "hetero_context_pos", None)
    if ctx_pos is not None:
        ctx_pos_np = (
            ctx_pos.numpy() if hasattr(ctx_pos, "numpy") else np.asarray(ctx_pos)
        )
        if ctx_pos_np.ndim == 2 and ctx_pos_np.shape[0] > 0:
            ctx_elem = getattr(data, "hetero_context_element", None)
            ctx_role = getattr(data, "hetero_context_role", None)
            ctx_metal = getattr(data, "hetero_context_is_metal", None)
            elem_np = (
                ctx_elem.numpy() if ctx_elem is not None and hasattr(ctx_elem, "numpy")
                else np.full(ctx_pos_np.shape[0], UNK_ELEMENT_IDX, dtype=np.int64)
            )
            role_np = (
                ctx_role.numpy() if ctx_role is not None and hasattr(ctx_role, "numpy")
                else np.full(ctx_pos_np.shape[0], ROLE_LIGAND_HEAVY, dtype=np.int64)
            )
            metal_np = (
                ctx_metal.numpy().astype(bool)
                if ctx_metal is not None and hasattr(ctx_metal, "numpy")
                else np.zeros(ctx_pos_np.shape[0], dtype=bool)
            )
            return ctx_pos_np, elem_np, role_np, metal_np

    hetero_idx = np.where(node_het)[0]
    if hetero_idx.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=bool),
        )
    roles = role_real if role_real is not None else np.full(len(pos_real), ROLE_LIGAND_HEAVY)
    return (
        pos_real[hetero_idx],
        elem_real[hetero_idx],
        roles[hetero_idx],
        node_metal[hetero_idx],
    )


def compute_residue_chemistry(data: Any) -> "torch.Tensor":
    """Compute d49 residue-level chemistry features [R, 16].

    This tensor carries discrete chemistry that is not naturally recoverable
    from canonical polymer atoms alone: CYS state, explicit metal coordination,
    nearby hetero atoms, and covalent ligand/PTM flags.
    """
    import torch
    import numpy as np
    from scipy.spatial import cKDTree

    R = int(data.num_residues)
    out = torch.zeros(R, NUM_RESIDUE_CHEM_FEATURES, dtype=torch.float32)
    if R == 0:
        return out

    residue_types = (
        data.residue_types.numpy()
        if hasattr(data.residue_types, "numpy")
        else np.asarray(data.residue_types)
    )
    seq_ids = (
        data.seq_ids.numpy()
        if hasattr(data, "seq_ids") and hasattr(data.seq_ids, "numpy")
        else np.arange(1, R + 1)
    )
    cys_state = (
        data.cys_state.numpy()
        if hasattr(data, "cys_state") and data.cys_state is not None and hasattr(data.cys_state, "numpy")
        else np.zeros(R, dtype=np.int64)
    )
    metal_coord = (
        data.metal_coord_count.numpy()
        if hasattr(data, "metal_coord_count") and data.metal_coord_count is not None and hasattr(data.metal_coord_count, "numpy")
        else np.zeros(R, dtype=np.float32)
    )
    cov_lig = (
        data.covalent_ligand.numpy().astype(bool)
        if hasattr(data, "covalent_ligand") and data.covalent_ligand is not None and hasattr(data.covalent_ligand, "numpy")
        else np.zeros(R, dtype=bool)
    )
    nonstd = (
        data.residue_nonstandard.numpy().astype(bool)
        if hasattr(data, "residue_nonstandard") and data.residue_nonstandard is not None and hasattr(data.residue_nonstandard, "numpy")
        else np.zeros(R, dtype=bool)
    )

    for ri in range(R):
        aa = int(residue_types[ri])
        state = int(cys_state[ri]) if ri < len(cys_state) else 0
        out[ri, 0] = 1.0 if aa == 4 else 0.0
        out[ri, 1] = 1.0 if state == _CYS_REDUCED else 0.0
        out[ri, 2] = 1.0 if state == _CYS_DISULFIDE else 0.0
        out[ri, 3] = 1.0 if state == _CYS_METAL else 0.0
        out[ri, 4] = 1.0 if state == _CYS_THIOETHER else 0.0
        out[ri, 5] = 1.0 if state == _CYS_OXIDIZED else 0.0
        out[ri, 6] = 1.0 if aa == 4 and int(seq_ids[ri]) <= 3 else 0.0
        out[ri, 7] = 1.0 if metal_coord[ri] > 0 else 0.0
        out[ri, 8] = min(float(metal_coord[ri]) / 4.0, 1.0)
        out[ri, 9] = 1.0
        out[ri, 11] = 1.0
        out[ri, 12] = 1.0 if aa == 8 and metal_coord[ri] > 0 else 0.0
        out[ri, 14] = 1.0 if cov_lig[ri] else 0.0
        out[ri, 15] = 1.0 if nonstd[ri] else 0.0

    pos_np = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    elem_np = data.element.numpy() if hasattr(data.element, "numpy") else np.asarray(data.element)
    ridx_np = data.residue_idx.numpy() if hasattr(data.residue_idx, "numpy") else np.asarray(data.residue_idx)
    roles_np = data.atom_role.numpy() if hasattr(data.atom_role, "numpy") else np.asarray(data.atom_role)
    atom_names = getattr(data, "_atom_names", None)
    n_real = int(getattr(data, "n_real_atoms", len(pos_np)))
    pos_np = pos_np[:n_real]
    elem_np = elem_np[:n_real]
    ridx_np = ridx_np[:n_real]
    roles_np = roles_np[:n_real]
    node_het = (
        data.node_is_hetero.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_hetero") and data.node_is_hetero is not None and hasattr(data.node_is_hetero, "numpy")
        else np.zeros(n_real, dtype=bool)
    )
    node_metal = (
        data.node_is_metal.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_metal") and data.node_is_metal is not None and hasattr(data.node_is_metal, "numpy")
        else np.zeros(n_real, dtype=bool)
    )
    hetero_pos, hetero_elem, hetero_role, hetero_is_metal = _hetero_context_arrays(
        data,
        pos_real=pos_np,
        elem_real=elem_np,
        role_real=roles_np,
        node_het=node_het,
        node_metal=node_metal,
    )

    ca_pos = np.full((R, 3), np.nan, dtype=np.float32)
    for ai in range(n_real):
        ri = int(ridx_np[ai])
        if 0 <= ri < R and int(roles_np[ai]) == 1 and np.isnan(ca_pos[ri, 0]):
            ca_pos[ri] = pos_np[ai]

    metal_pos = hetero_pos[hetero_is_metal]
    if len(metal_pos) > 0:
        tree = cKDTree(metal_pos)
        for ri in range(R):
            if np.isnan(ca_pos[ri, 0]):
                continue
            d, _ = tree.query(ca_pos[ri], k=1)
            if np.isfinite(d):
                out[ri, 9] = min(float(d) / 8.0, 1.0)
    if len(hetero_pos) > 0:
        tree = cKDTree(hetero_pos)
        for ri in range(R):
            if np.isnan(ca_pos[ri, 0]):
                continue
            close = tree.query_ball_point(ca_pos[ri], 4.0)
            out[ri, 10] = min(float(len(close)) / 8.0, 1.0)
            d, _ = tree.query(ca_pos[ri], k=1)
            if np.isfinite(d):
                out[ri, 11] = min(float(d) / 8.0, 1.0)

    # Charged sidechain atoms near each residue backbone.
    if atom_names is not None:
        charged_pos: list[np.ndarray] = []
        for ai in range(n_real):
            ri_atom = int(ridx_np[ai])
            if not (0 <= ri_atom < R):
                continue
            if _is_charged_sidechain_atom(int(residue_types[ri_atom]), str(atom_names[ai])):
                charged_pos.append(pos_np[ai])
        if charged_pos:
            ctree = cKDTree(np.asarray(charged_pos))
            for ri in range(R):
                bb_atoms = [pos_np[ai] for ai in range(n_real) if int(ridx_np[ai]) == ri and int(roles_np[ai]) in (0, 2, 3)]
                if not bb_atoms:
                    continue
                count = 0
                for p in bb_atoms:
                    count += len(ctree.query_ball_point(p, 4.0))
                out[ri, 13] = min(float(count) / 6.0, 1.0)

    return out


def compute_target_chemistry(data: Any) -> "torch.Tensor":
    """Compute d49 target-atom chemistry features [R, 6, 14]."""
    import torch
    import numpy as np
    from scipy.spatial import cKDTree

    R = int(data.num_residues)
    out = np.zeros((R, 6, NUM_TARGET_CHEM_FEATURES), dtype=np.float32)
    out[:, :, 0] = 1.0  # nearest metal sentinel
    out[:, :, 4] = 1.0  # nearest hetero sentinel
    out[:, :, 6] = 1.0  # nearest charged sentinel
    out[:, :, 13] = 1.0 # nearest ring sentinel

    pos_np = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    elem_np = data.element.numpy() if hasattr(data.element, "numpy") else np.asarray(data.element)
    role_np = data.atom_role.numpy() if hasattr(data.atom_role, "numpy") else np.asarray(data.atom_role)
    ridx_np = data.residue_idx.numpy() if hasattr(data.residue_idx, "numpy") else np.asarray(data.residue_idx)
    rtypes_np = data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.asarray(data.residue_types)
    ti_np = data.target_indices.numpy() if hasattr(data.target_indices, "numpy") else np.asarray(data.target_indices)
    tm_np = (data.target_mask.numpy() if hasattr(data.target_mask, "numpy") else np.asarray(data.target_mask)).astype(bool)
    atom_names = getattr(data, "_atom_names", None)

    n_real = int(getattr(data, "n_real_atoms", len(pos_np)))
    pos_real = pos_np[:n_real]
    elem_real = elem_np[:n_real]
    role_real = role_np[:n_real]
    ridx_real = ridx_np[:n_real]
    atom_names_real = atom_names[:n_real] if atom_names is not None and len(atom_names) > n_real else atom_names
    node_het = (
        data.node_is_hetero.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_hetero") and data.node_is_hetero is not None and hasattr(data.node_is_hetero, "numpy")
        else np.zeros(n_real, dtype=bool)
    )
    node_metal = (
        data.node_is_metal.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_metal") and data.node_is_metal is not None and hasattr(data.node_is_metal, "numpy")
        else np.zeros(n_real, dtype=bool)
    )

    metal_idx = np.where(node_metal)[0]
    hetero_pos, hetero_elem, hetero_role, hetero_is_metal = _hetero_context_arrays(
        data,
        pos_real=pos_real,
        elem_real=elem_real,
        role_real=role_real,
        node_het=node_het,
        node_metal=node_metal,
    )
    metal_pos = hetero_pos[hetero_is_metal]
    metal_tree = cKDTree(metal_pos) if len(metal_pos) else None
    hetero_tree = cKDTree(hetero_pos) if len(hetero_pos) else None

    charged_idx: list[int] = []
    polar_sc_idx: list[int] = []
    if atom_names_real is not None:
        for ai in range(n_real):
            ri = int(ridx_real[ai])
            if not (0 <= ri < R):
                continue
            aa = int(rtypes_np[ri])
            name = str(atom_names_real[ai])
            if _is_charged_sidechain_atom(aa, name):
                charged_idx.append(ai)
            if int(role_real[ai]) == 7 and int(elem_real[ai]) in (ELEMENT_TO_IDX["N"], ELEMENT_TO_IDX["O"]):
                polar_sc_idx.append(ai)
    charged_tree = cKDTree(pos_real[charged_idx]) if charged_idx else None
    polar_sc_tree = cKDTree(pos_real[polar_sc_idx]) if polar_sc_idx else None

    # Aromatic rings for direct signed per-target ring-current proxies.
    _IDX_TO_AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]
    _INTENSITY = {"PHE": 1.00, "TYR": 0.94, "TRP": 1.04, "HIS": 0.43}
    rings: list[tuple[np.ndarray, np.ndarray, float]] = []
    if atom_names_real is not None:
        res_atom_pos: dict[int, dict[str, int]] = {}
        for ai, name in enumerate(atom_names_real):
            ri = int(ridx_real[ai])
            if 0 <= ri < R:
                res_atom_pos.setdefault(ri, {})[str(name).strip()] = ai
        for ri in range(R):
            aa = int(rtypes_np[ri])
            aa3 = _IDX_TO_AA3[aa] if 0 <= aa < len(_IDX_TO_AA3) else ""
            if aa3 not in AROMATIC_RINGS:
                continue
            ra = res_atom_pos.get(ri, {})
            for ring_atom_names in AROMATIC_RINGS[aa3]:
                idxs = [ra.get(n) for n in ring_atom_names]
                if any(i is None for i in idxs):
                    continue
                coords = pos_real[idxs]
                centroid = coords.mean(axis=0)
                centered = coords - centroid
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                normal = vh[-1]
                n = np.linalg.norm(normal)
                if n < 1e-8:
                    continue
                rings.append((centroid, normal / n, _INTENSITY.get(aa3, 1.0)))

    def _ring_shift(point: np.ndarray, centroid: np.ndarray, normal: np.ndarray, intensity: float) -> tuple[float, float, float]:
        d_vec = point - centroid
        r = float(np.linalg.norm(d_vec))
        if r < 1.0:
            return 0.0, 0.0, r
        cos_theta = float(np.dot(d_vec, normal) / r)
        shift = float(intensity * 27.28 * (3.0 * cos_theta * cos_theta - 1.0) / (r ** 3))
        return shift, cos_theta, r

    for ri in range(R):
        for ni in range(6):
            if not tm_np[ri, ni]:
                continue
            ai = int(ti_np[ri, ni])
            if ai < 0 or ai >= n_real:
                continue
            target = pos_real[ai]

            if metal_tree is not None:
                close = metal_tree.query_ball_point(target, 4.0)
                out[ri, ni, 1] = min(float(len(close)) / 4.0, 1.0)
                d, _ = metal_tree.query(target, k=1)
                if np.isfinite(d):
                    out[ri, ni, 0] = min(float(d) / 8.0, 1.0)
            if hetero_tree is not None:
                close4 = hetero_tree.query_ball_point(target, 4.0)
                close8 = hetero_tree.query_ball_point(target, 8.0)
                out[ri, ni, 2] = min(float(len(close4)) / 8.0, 1.0)
                out[ri, ni, 3] = min(float(max(0, len(close8) - len(close4))) / 16.0, 1.0)
                d, _ = hetero_tree.query(target, k=1)
                if np.isfinite(d):
                    out[ri, ni, 4] = min(float(d) / 8.0, 1.0)
            if charged_tree is not None:
                close = charged_tree.query_ball_point(target, 4.0)
                out[ri, ni, 5] = min(float(len(close)) / 6.0, 1.0)
                d, _ = charged_tree.query(target, k=1)
                if np.isfinite(d):
                    out[ri, ni, 6] = min(float(d) / 8.0, 1.0)
            if polar_sc_tree is not None:
                close = polar_sc_tree.query_ball_point(target, 3.6)
                out[ri, ni, 7] = min(float(len(close)) / 6.0, 1.0)

            if rings:
                signed = 0.0
                axial = 0.0
                equatorial = 0.0
                max_abs = 0.0
                count = 0
                nearest = 999.0
                for centroid, normal, intensity in rings:
                    shift, cos_theta, d = _ring_shift(target, centroid, normal, intensity)
                    if d <= 8.5:
                        count += 1
                        nearest = min(nearest, d)
                        signed += shift
                        if abs(cos_theta) > 0.819:  # theta <35 or >145 deg
                            axial += shift
                        elif abs(cos_theta) < 0.574:  # 55-125 deg
                            equatorial += shift
                        max_abs = max(max_abs, abs(shift))
                out[ri, ni, 8] = float(np.clip(signed, -3.0, 3.0) / 3.0)
                out[ri, ni, 9] = float(np.clip(axial, -3.0, 3.0) / 3.0)
                out[ri, ni, 10] = float(np.clip(equatorial, -3.0, 3.0) / 3.0)
                out[ri, ni, 11] = float(min(max_abs, 3.0) / 3.0)
                out[ri, ni, 12] = min(float(count) / 8.0, 1.0)
                if nearest < 999.0:
                    out[ri, ni, 13] = min(float(nearest) / 8.5, 1.0)

    return torch.from_numpy(out)


def compute_target_nc_chemistry(data: Any) -> "torch.Tensor":
    """Compute d50 direct N/C contact features [R, 6, 8].

    These features give the head a short, nucleus-specific route for the two
    v49 problem classes: amide N/H donor contacts and carbonyl C/O acceptor
    contacts. They intentionally stay compact and geometry-derived; detailed
    message passing still flows through the regular graph edges.
    """
    import torch
    import numpy as np

    R = int(data.num_residues)
    out = np.zeros((R, 6, NUM_TARGET_NC_CHEM_FEATURES), dtype=np.float32)
    out[:, :, 1] = 1.0  # nearest NH acceptor/contact sentinel
    out[:, :, 5] = 1.0  # nearest CO donor/cation sentinel
    if R == 0:
        return torch.from_numpy(out)

    pos_np = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    elem_np = data.element.numpy() if hasattr(data.element, "numpy") else np.asarray(data.element)
    role_np = data.atom_role.numpy() if hasattr(data.atom_role, "numpy") else np.asarray(data.atom_role)
    ridx_np = data.residue_idx.numpy() if hasattr(data.residue_idx, "numpy") else np.asarray(data.residue_idx)
    rtypes_np = data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.asarray(data.residue_types)
    ti_np = data.target_indices.numpy() if hasattr(data.target_indices, "numpy") else np.asarray(data.target_indices)
    tm_np = (data.target_mask.numpy() if hasattr(data.target_mask, "numpy") else np.asarray(data.target_mask)).astype(bool)
    av_np = (
        data.atom_valid.numpy().astype(bool)
        if hasattr(data, "atom_valid") and data.atom_valid is not None and hasattr(data.atom_valid, "numpy")
        else tm_np
    )
    atom_names = getattr(data, "_atom_names", None)

    n_real = int(getattr(data, "n_real_atoms", len(pos_np)))
    pos_real = pos_np[:n_real]
    elem_real = elem_np[:n_real]
    role_real = role_np[:n_real]
    ridx_real = ridx_np[:n_real]
    atom_names_real = atom_names[:n_real] if atom_names is not None and len(atom_names) > n_real else atom_names
    node_het = (
        data.node_is_hetero.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_hetero") and data.node_is_hetero is not None and hasattr(data.node_is_hetero, "numpy")
        else np.zeros(n_real, dtype=bool)
    )
    node_metal = (
        data.node_is_metal.numpy().astype(bool)[:n_real]
        if hasattr(data, "node_is_metal") and data.node_is_metal is not None and hasattr(data.node_is_metal, "numpy")
        else np.zeros(n_real, dtype=bool)
    )
    hetero_pos, hetero_elem, hetero_role, _hetero_is_metal = _hetero_context_arrays(
        data,
        pos_real=pos_real,
        elem_real=elem_real,
        role_real=role_real,
        node_het=node_het,
        node_metal=node_metal,
    )

    O_EL = ELEMENT_TO_IDX["O"]
    N_EL = ELEMENT_TO_IDX["N"]
    S_EL = ELEMENT_TO_IDX["S"]
    H_EL = ELEMENT_TO_IDX["H"]
    backbone_roles = {0, 1, 2, 3, 5, 6}

    # Per-residue backbone atom lookup for carbonyl O fallback.
    bb: dict[int, dict[int, int]] = {}
    for ai in range(n_real):
        if node_het[ai]:
            continue
        ri = int(ridx_real[ai])
        role = int(role_real[ai])
        if 0 <= ri < R and role in (0, 2, 3):
            bb.setdefault(ri, {}).setdefault(role, ai)

    Contact = tuple[np.ndarray, int, int, bool, bool]

    def _atom_name(ai: int) -> str:
        if atom_names_real is None:
            return ""
        return str(atom_names_real[ai]).strip().upper()

    def _is_polymer_acceptor_like(aa_idx: int, atom_name: str, role: int, elem: int) -> bool:
        # Amide N/H should see true acceptors, not every nearby polar atom.
        # Backbone carbonyl O is allowed, but local peptide neighbors are
        # excluded later by sequence separation.
        if role == 3 and elem == O_EL:
            return True
        if role in backbone_roles:
            return False
        if elem in (O_EL, S_EL):
            return True
        if elem == N_EL and aa_idx == 8 and atom_name in {"ND1", "NE2"}:
            return True
        return False

    def _is_polymer_donor_or_cation_like(aa_idx: int, atom_name: str, role: int, elem: int) -> bool:
        if _is_positive_sidechain_atom(aa_idx, atom_name):
            return True
        # Backbone amides can donate to carbonyl oxygens, but local peptide
        # neighbors are filtered later so ordinary i/i+1 geometry cannot
        # dominate the C features.
        if role == 0 and elem == N_EL:
            return True
        if role in backbone_roles:
            return False
        donor_names_by_aa = {
            1: {"NE", "NH1", "NH2"},   # ARG
            2: {"ND2"},                 # ASN
            4: {"SG"},                  # CYS
            5: {"NE2"},                 # GLN
            8: {"ND1", "NE2"},         # HIS
            11: {"NZ"},                 # LYS
            15: {"OG"},                 # SER
            16: {"OG1"},                # THR
            17: {"NE1"},                # TRP
            18: {"OH"},                 # TYR
        }
        return atom_name in donor_names_by_aa.get(aa_idx, set())

    def _candidate_allowed(
        cand_ri: int,
        target_ri: int,
        is_sidechain_or_hetero: bool,
        is_positive: bool,
    ) -> bool:
        if cand_ri == target_ri:
            return False
        if cand_ri < 0:
            return True
        seq_sep = abs(int(cand_ri) - int(target_ri))
        if seq_sep <= 2 and not (is_sidechain_or_hetero or is_positive):
            return False
        return True

    def _nh_angle_ok(n_pos: np.ndarray | None, h_pos: np.ndarray, acc_pos: np.ndarray) -> bool:
        if n_pos is None:
            return True
        donor_vec = h_pos - n_pos
        acc_vec = acc_pos - h_pos
        denom = float(np.linalg.norm(donor_vec) * np.linalg.norm(acc_vec))
        if denom <= 1e-6:
            return True
        # Equivalent to N-H...A > ~120 degrees.
        return float(np.dot(donor_vec, acc_vec) / denom) >= 0.5

    acceptors: list[Contact] = []
    co_contacts: list[Contact] = []
    for ai in range(n_real):
        if node_het[ai]:
            continue
        elem = int(elem_real[ai])
        if elem == H_EL:
            continue
        ri = int(ridx_real[ai])
        if not (0 <= ri < R):
            continue
        role = int(role_real[ai])
        is_sidechain = role not in backbone_roles
        name = _atom_name(ai)
        aa_idx = int(rtypes_np[ri])
        positive = _is_positive_sidechain_atom(aa_idx, name)
        if _is_polymer_acceptor_like(aa_idx, name, role, elem):
            acceptors.append((pos_real[ai], ri, role, is_sidechain, positive))
        if _is_polymer_donor_or_cation_like(aa_idx, name, role, elem):
            co_contacts.append((pos_real[ai], ri, role, is_sidechain, positive))

    # Summary-only hetero context participates in N/C contact features even
    # when those atoms were not promoted to graph nodes. Treat hetero contacts
    # as sidechain-like and long-range from the polymer target.
    for hi in range(len(hetero_pos)):
        elem = int(hetero_elem[hi])
        role = int(hetero_role[hi])
        if role == ROLE_METAL:
            continue
        if elem in (O_EL, N_EL, S_EL) or elem == UNK_ELEMENT_IDX:
            acceptors.append((hetero_pos[hi], -999_999, role, True, False))
            co_contacts.append((hetero_pos[hi], -999_999, role, True, False))

    def _accumulate_nh(
        point: np.ndarray,
        n_pos: np.ndarray | None,
        target_ri: int,
        *,
        hydrogen_distance: bool,
    ) -> tuple[int, float, int, int, float]:
        count = 0
        sidechain = 0
        long_range = 0
        nearest = 999.0
        cutoff = 2.7 if hydrogen_distance else 3.5
        for cand_pos, cand_ri, _role, is_sidechain, is_positive in acceptors:
            if not _candidate_allowed(cand_ri, target_ri, is_sidechain, is_positive):
                continue
            if hydrogen_distance and not _nh_angle_ok(n_pos, point, cand_pos):
                continue
            d = float(np.linalg.norm(cand_pos - point))
            if d > cutoff:
                continue
            count += 1
            nearest = min(nearest, d)
            if is_sidechain:
                sidechain += 1
            if cand_ri < 0 or abs(int(cand_ri) - int(target_ri)) >= 5:
                long_range += 1
        return count, nearest, sidechain, long_range, cutoff

    def _accumulate_co(point: np.ndarray, target_ri: int) -> tuple[int, float, int, int]:
        count = 0
        sidechain = 0
        positive_count = 0
        nearest = 999.0
        for cand_pos, cand_ri, _role, is_sidechain, is_positive in co_contacts:
            if not _candidate_allowed(cand_ri, target_ri, is_sidechain, is_positive):
                continue
            cutoff = 4.0 if is_positive else 3.6
            d = float(np.linalg.norm(cand_pos - point))
            if d > cutoff:
                continue
            count += 1
            nearest = min(nearest, d)
            if is_sidechain:
                sidechain += 1
            if is_positive:
                positive_count += 1
        return count, nearest, sidechain, positive_count

    I_H, I_N, I_C = 0, 2, 5
    for ri in range(R):
        # Amide donor side: use H when available, falling back to N so
        # heavy-atom-only structures still get a useful direct-contact proxy.
        nh_point = None
        n_point = None
        if av_np[ri, I_N]:
            n_idx = int(ti_np[ri, I_N])
            if 0 <= n_idx < n_real:
                n_point = pos_real[n_idx]
        if av_np[ri, I_H]:
            h_idx = int(ti_np[ri, I_H])
            if 0 <= h_idx < n_real:
                nh_point = pos_real[h_idx]
        hydrogen_distance = nh_point is not None
        if nh_point is None:
            nh_point = n_point
        if nh_point is not None:
            count, nearest, sidechain, long_range, cutoff = _accumulate_nh(
                nh_point, n_point, ri, hydrogen_distance=hydrogen_distance,
            )
            vals = (
                min(float(count) / 2.0, 1.0),
                min(float(nearest) / cutoff, 1.0) if nearest < 999.0 else 1.0,
                min(float(sidechain) / 2.0, 1.0),
                min(float(long_range) / 2.0, 1.0),
            )
            for ni in (I_H, I_N):
                if tm_np[ri, ni]:
                    out[ri, ni, 0:4] = vals

        # Carbonyl acceptor side: use O when present because the contact
        # physics lands there, then expose it to the C nucleus head.
        co_point = None
        o_idx = bb.get(ri, {}).get(3)
        if o_idx is not None:
            co_point = pos_real[o_idx]
        elif av_np[ri, I_C]:
            c_idx = int(ti_np[ri, I_C])
            if 0 <= c_idx < n_real:
                co_point = pos_real[c_idx]
        if co_point is not None and tm_np[ri, I_C]:
            count, nearest, sidechain, positive_count = _accumulate_co(co_point, ri)
            out[ri, I_C, 4] = min(float(count) / 2.0, 1.0)
            out[ri, I_C, 5] = (
                min(float(nearest) / 4.0, 1.0) if nearest < 999.0 else 1.0
            )
            out[ri, I_C, 6] = min(float(sidechain) / 2.0, 1.0)
            out[ri, I_C, 7] = min(float(positive_count) / 2.0, 1.0)

    return torch.from_numpy(out)


def compute_target_atom_rsasa(
    data: Any,
    *,
    atom_rsasa_full: "np.ndarray | None" = None,
) -> "torch.Tensor":
    """Compute per-target-atom rSASA tensor [R, 6] from stored graph data.

    Maps each of the 6 target nuclei per residue to its own atom-sphere
    normalized rSASA (H→parent N, HA→parent CA, heavy atoms direct).
    See ``physics_features.compute_target_atom_rsasa`` for the mapping.

    Args:
        data: Graph Data object.
        atom_rsasa_full: Optional precomputed atom-sphere-normalized rSASA
            array (as produced by ``compute_sasa_and_hse``). When supplied,
            the expensive Shrake-Rupley pass is skipped entirely. Used by
            ``compute_persistent_features`` to share the single SASA pass
            with ``compute_residue_geometry``. When omitted, Shrake-Rupley
            runs locally and the function remains a self-contained
            lazy-backfill helper.
    """
    import torch
    import numpy as np
    from .physics_features import compute_sasa_and_hse, compute_target_atom_rsasa as _ctar

    R = int(data.num_residues)
    pos_np = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    elem_np = data.element.numpy() if hasattr(data.element, "numpy") else np.asarray(data.element)
    role_np = data.atom_role.numpy() if hasattr(data.atom_role, "numpy") else np.asarray(data.atom_role)
    ridx_np = data.residue_idx.numpy() if hasattr(data.residue_idx, "numpy") else np.asarray(data.residue_idx)
    rtypes_np = data.residue_types.numpy() if hasattr(data.residue_types, "numpy") else np.asarray(data.residue_types)
    ti_np = data.target_indices.numpy() if hasattr(data.target_indices, "numpy") else np.asarray(data.target_indices)
    tm_np = (data.target_mask.numpy() if hasattr(data.target_mask, "numpy") else np.asarray(data.target_mask)).astype(bool)

    # d47_v1: ring virtual nodes appended at tail — exclude from SASA.
    n_real = int(getattr(data, "n_real_atoms", len(pos_np)))
    pos_np = pos_np[:n_real]
    elem_np = elem_np[:n_real]
    role_np = role_np[:n_real]
    ridx_np = ridx_np[:n_real]

    # Rebuild bb_lookup from atom_role (0=N, 1=CA, 2=C, 3=O, 4=CB)
    bb: dict[int, dict[int, int]] = {}
    for ai in range(len(elem_np)):
        role = int(role_np[ai])
        if role not in (0, 1, 2, 3, 4):
            continue
        ri = int(ridx_np[ai])
        if 0 <= ri < R:
            bb.setdefault(ri, {}).setdefault(role, ai)

    if atom_rsasa_full is None:
        _sasa, atom_rsasa_full = compute_sasa_and_hse(
            pos_np, elem_np, ridx_np, rtypes_np, bb, R, n_sphere_points=92,
        )
    rsasa = _ctar(
        atom_rsasa_full, ti_np, tm_np, role_np, bb, R,
    )
    return torch.from_numpy(rsasa)


def compute_persistent_features(data: Any) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Combined single-pass computation of ``residue_geometry`` and
    ``target_atom_rsasa``.

    Equivalent to calling::

        geo = compute_residue_geometry(data)
        rsasa = compute_target_atom_rsasa(data)

    but avoids the duplicated Shrake-Rupley pass (Shrake-Rupley is the
    dominant cost of the backfill — ~2x speedup on typical graphs).

    The shared path works because ``compute_residue_geometry`` already
    runs Shrake-Rupley internally for residue_geometry slots [25-27];
    we capture the per-atom rSASA side output via the attribute
    ``compute_residue_geometry._last_atom_rsasa_full`` and pass it to
    ``compute_target_atom_rsasa`` which then skips its own SASA call.

    Returns
    -------
    (residue_geometry, target_atom_rsasa)
        residue_geometry shape [R, NUM_GEO_FEATURES], target_atom_rsasa
        shape [R, 6]. Both torch.float32 on CPU.
    """
    geo = compute_residue_geometry(data)
    atom_rsasa_full = getattr(compute_residue_geometry, "_last_atom_rsasa_full", None)
    rsasa = compute_target_atom_rsasa(data, atom_rsasa_full=atom_rsasa_full)
    return geo, rsasa


def compute_solchem_features(data: Any) -> "torch.Tensor":
    """Compute per-residue solution-chemistry features.

    Returns [R, NUM_SOLCHEM_FEATURES] tensor. See ``features.py`` for the
    12-slot layout. Works on cached graphs — needs only attributes already
    stored at build time (pos, atom_role, residue_idx, residue_types,
    sample_ph, sample_ionic_strength, has_ph, has_ionic_strength).
    """
    import torch
    import numpy as np
    from .features import (
        NUM_SOLCHEM_FEATURES,
        TITRATABLE_NEG,
        TITRATABLE_POS,
        TITRATABLE_HIS,
        TITRATABLE_THIOL,
    )

    R = int(data.num_residues)
    sc = torch.zeros(R, NUM_SOLCHEM_FEATURES, dtype=torch.float32)

    ph = float(data.sample_ph.item()) if hasattr(data, "sample_ph") else 7.0
    ion_raw = (
        float(data.sample_ionic_strength.item())
        if hasattr(data, "sample_ionic_strength")
        else 0.1
    )
    ion = max(0.0, min(5.0, ion_raw))  # clamp depositor unit errors
    has_ph = float(getattr(data, "has_ph", torch.tensor([1.0])).item())
    has_ion = float(getattr(data, "has_ionic_strength", torch.tensor([1.0])).item())

    sc[:, 0] = (ph - 6.5) / 2.0
    sc[:, 1] = (ion - 0.1) / 0.5
    sc[:, 2] = has_ph
    sc[:, 3] = has_ion

    # Per-residue CA positions (role index 1 = CA)
    pos = data.pos
    roles = data.atom_role
    res_idx = data.residue_idx
    ca_pos = np.full((R, 3), np.nan, dtype=np.float32)
    pos_np = pos.numpy() if hasattr(pos, "numpy") else np.asarray(pos)
    roles_np = roles.numpy() if hasattr(roles, "numpy") else np.asarray(roles)
    ridx_np = res_idx.numpy() if hasattr(res_idx, "numpy") else np.asarray(res_idx)
    for ai in range(pos_np.shape[0]):
        if int(roles_np[ai]) == 1:  # CA
            ri = int(ridx_np[ai])
            if 0 <= ri < R and np.isnan(ca_pos[ri, 0]):
                ca_pos[ri] = pos_np[ai]

    # Classify each residue into titratable groups from AA type
    residue_types = (
        data.residue_types.numpy()
        if hasattr(data.residue_types, "numpy")
        else np.asarray(data.residue_types)
    )
    neg_idx = np.array(
        [ri for ri in range(R) if int(residue_types[ri]) in TITRATABLE_NEG],
        dtype=np.int32,
    )
    pos_idx = np.array(
        [ri for ri in range(R) if int(residue_types[ri]) in TITRATABLE_POS],
        dtype=np.int32,
    )
    his_idx = np.array(
        [ri for ri in range(R) if int(residue_types[ri]) in TITRATABLE_HIS],
        dtype=np.int32,
    )
    thi_idx = np.array(
        [ri for ri in range(R) if int(residue_types[ri]) in TITRATABLE_THIOL],
        dtype=np.int32,
    )

    # For each residue, count and nearest-distance per group (8 A cutoff).
    # Skip residues with missing CA (nan).
    CUTOFF = 8.0
    groups = [
        (neg_idx, 5.0, 4, 8),    # slot count=[4], nearest=[8], norm=/5, /20
        (pos_idx, 5.0, 5, 9),
        (his_idx, 3.0, 6, 10),
        (thi_idx, 3.0, 7, 11),
    ]
    for ri in range(R):
        cap = ca_pos[ri]
        if np.isnan(cap[0]):
            # Leave defaults: count=0, nearest=1.0 (20 A) for all groups
            for _, _, _count_slot, nearest_slot in groups:
                sc[ri, nearest_slot] = 1.0
            continue
        for g_idx, norm_count, count_slot, nearest_slot in groups:
            if g_idx.size == 0:
                sc[ri, nearest_slot] = 1.0
                continue
            # Exclude self (a residue is not its own neighbour)
            others = g_idx[g_idx != ri]
            if others.size == 0:
                sc[ri, nearest_slot] = 1.0
                continue
            deltas = ca_pos[others] - cap
            # Mask out peers with missing CA
            valid = ~np.isnan(deltas[:, 0])
            if not np.any(valid):
                sc[ri, nearest_slot] = 1.0
                continue
            dists = np.sqrt(np.sum(deltas[valid] ** 2, axis=1))
            count_within = float(np.sum(dists < CUTOFF))
            sc[ri, count_slot] = min(count_within / norm_count, 1.0)
            nearest = float(np.min(dists))
            sc[ri, nearest_slot] = min(nearest / 20.0, 1.0)

    return sc


def recompute_potenci_rc_from_graph(data: Any) -> "torch.Tensor":
    """Compute POTENCI random-coil shifts directly from a graph's own sequence.

    Bypasses the BMRB→PDB sequence-alignment pipeline in the graph build
    path, which has a latent indexing bug: when BMRB comp_ids don't start
    at seqid 1 (His-tags, unresolved loops, truncated chains), the
    ``rc_array[bmrb_sid - 1]`` indexing in the builder's alignment loop
    assigns shifts to the wrong residue positions.

    This helper reconstructs the 1-letter sequence from the persisted
    ``data.residue_types`` (canonical AA indices 0-19), then calls POTENCI
    with the graph's own pH/temperature/ionic strength. The resulting
    [R, 6] array is trivially aligned to graph residue order because the
    sequence IS the graph's residue order.

    Returns a fresh [R, 6] float32 tensor (NaN where POTENCI has no
    random-coil value, e.g. terminals, GLY CB, PRO H). On any failure
    (import error, POTENCI exception, unknown residue), returns an
    all-NaN tensor so callers can fall back to AA means.
    """
    import torch
    import numpy as np
    try:
        from .potenci import compute_random_coil_array
    except Exception:
        R = int(data.num_residues)
        return torch.full((R, 6), float("nan"), dtype=torch.float32)

    R = int(data.num_residues)
    # Canonical 20 AA index → 1-letter
    _IDX_TO_LETTER = {
        0: "A", 1: "R", 2: "N", 3: "D", 4: "C",
        5: "Q", 6: "E", 7: "G", 8: "H", 9: "I",
        10: "L", 11: "K", 12: "M", 13: "F", 14: "P",
        15: "S", 16: "T", 17: "W", 18: "Y", 19: "V",
    }
    rtypes = (
        data.residue_types.numpy()
        if hasattr(data.residue_types, "numpy")
        else np.asarray(data.residue_types)
    )
    seq_chars = [_IDX_TO_LETTER.get(int(r), "X") for r in rtypes]
    # POTENCI can't handle unknown residues — substitute ALA into the
    # sequence so POTENCI produces a result for all positions (keeping
    # neighbor context as close as possible to the real chemistry). We
    # then mask the UNK positions themselves back to NaN below so the
    # model falls through to the AA-mean baseline for those residues.
    # Bug S: the prior code left UNK rows filled with ALA's RC values,
    # biasing the prediction at those positions.
    unk_positions = [i for i, c in enumerate(seq_chars) if c == "X"]
    seq = "".join(c if c != "X" else "A" for c in seq_chars)
    if len(seq) < 5:
        return torch.full((R, 6), float("nan"), dtype=torch.float32)

    ph = float(data.sample_ph.item()) if hasattr(data, "sample_ph") else 7.0
    temp = (
        float(data.sample_temperature.item())
        if hasattr(data, "sample_temperature")
        else 298.0
    )
    ion = (
        float(data.sample_ionic_strength.item())
        if hasattr(data, "sample_ionic_strength")
        else 0.1
    )

    try:
        rc = compute_random_coil_array(seq, pH=ph, temperature=temp, ionic_strength=ion)
    except Exception:
        return torch.full((R, 6), float("nan"), dtype=torch.float32)

    if rc.shape != (R, 6):
        return torch.full((R, 6), float("nan"), dtype=torch.float32)

    # Bug S completion: NaN-out rows for UNK residues so the model
    # baseline falls through to aa_mean rather than using ALA's RC
    # (which would bias the prediction at unknown residues).
    rc = rc.astype(np.float32)
    if unk_positions:
        rc[unk_positions, :] = np.float32("nan")

    return torch.from_numpy(rc)


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
    # N and H positions for NH order parameter S²_NH = |⟨n̂_NH⟩|² across
    # conformers. Needs overall-tumbling removal, so we apply the same
    # Kabsch rotation later that the CA medoid computation uses.
    n_coords = np.full((n_models, R, 3), np.nan, dtype=np.float64)
    h_coords = np.full((n_models, R, 3), np.nan, dtype=np.float64)

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
            h_pos = _get_pos(mr, "H")

            # Track N/H for NH order parameter (after Kabsch alignment below).
            if n_pos is not None:
                n_coords[mi, ri] = n_pos
            if h_pos is not None:
                h_coords[mi, ri] = h_pos

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

    # Per-model rigid-body transform: centered[mi] = coords[mi] - model_center[mi];
    # aligned[mi] = centered[mi] @ model_rot[mi].T + ref_center. Stored so we
    # can map N/H positions into the same frame for S²_NH.
    model_rot = np.zeros((n_models, 3, 3), dtype=np.float64)
    model_rot[0] = np.eye(3)
    model_center = np.zeros((n_models, 3), dtype=np.float64)
    model_center[0] = ref_center

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
        model_rot[mi] = R_mat
        model_center[mi] = center

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

    # ------------------------------------------------------------------
    # S²_NH order parameter — Lipari–Szabo generalised S² from the
    # ensemble unit-vector distribution. After removing overall tumbling
    # (Kabsch) the bond direction dispersion is the internal-motion
    # contribution. S² = |⟨n̂⟩|² = (<x>²+<y>²+<z>²). NaN where <2 models
    # have both N and H.
    # ------------------------------------------------------------------
    s2_nh = np.full(R, np.nan, dtype=np.float64)
    # Rotate raw N and H positions into the reference frame using the
    # stored Kabsch transform. Model 0 is already in its own frame;
    # we keep it as-is (R=I) so the code below is uniform.
    nh_vec = np.full((n_models, R, 3), np.nan, dtype=np.float64)
    for mi in range(n_models):
        # n_rot = (n - model_center) @ R.T + ref_center ; but for unit
        # vectors we only need the rotation, so:
        #   (n - h) @ R.T  is rotation-equivalent to rotating the vector.
        # (translation cancels in the difference.)
        n_minus_h = n_coords[mi] - h_coords[mi]  # [R, 3]
        valid = ~np.isnan(n_minus_h[:, 0])
        if not np.any(valid):
            continue
        rotated = n_minus_h[valid] @ model_rot[mi].T
        # NH unit vector points from N to H (conventional).
        hn_vec = -rotated
        norms = np.linalg.norm(hn_vec, axis=-1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        nh_vec[mi, valid] = hn_vec / norms

    for ri in range(R):
        v = nh_vec[:, ri, :]
        valid = ~np.isnan(v[:, 0])
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue
        mean_v = v[valid].mean(axis=0)
        s2_nh[ri] = float(np.dot(mean_v, mean_v))
    # Numerical clip: occasional round-off can push S² slightly outside
    # [0, 1] when all vectors are near-parallel.
    s2_nh = np.where(np.isnan(s2_nh), s2_nh, np.clip(s2_nh, 0.0, 1.0))

    # Emit the seq_ids the reference chain enumerated so callers can align
    # ensemble arrays to graph residues by seqid rather than position.
    # Positional alignment fails silently when graph/ensemble filter the
    # same chain but end up with different R (e.g., mid-chain HETATM).
    ensemble_seqids = np.array(
        [residues[ri].seqid.num for ri in range(R)], dtype=np.int64,
    )

    return {
        "medoid_idx": medoid_idx,
        "num_conformers": n_models,
        "seq_ids": ensemble_seqids,
        "ca_rmsf": ca_rmsf,
        "phi_circvar": phi_circvar,
        "psi_circvar": psi_circvar,
        "chi1_circvar": chi1_circvar,
        "chi1_has_data": chi1_has_data,
        "s2_nh": s2_nh,
    }
