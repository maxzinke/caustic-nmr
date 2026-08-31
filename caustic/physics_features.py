"""Physics-based per-residue geometry features for shift prediction.

Computes features that encode information the GNN cannot easily learn
from 3D coordinates alone:
  - Ring current shifts (Haigh-Mallion model)
  - Hydrogen bond geometry (NH donor, CO acceptor)
  - Solvent-accessible surface area (Shrake-Rupley)
  - Half-sphere exposure (HSE)
  - DSSP secondary structure
  - Buckingham electric field (P3 — planned)
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ring current shifts — Haigh-Mallion / Johnson-Bovey model
# ---------------------------------------------------------------------------

# Aromatic ring definitions: {3-letter AA: [(ring_name, [atom_names], intensity)]}
# Intensity relative to benzene (Haigh & Mallion, Prog NMR Spectrosc 1980)
_AROMATIC_RINGS: dict[str, list[tuple[str, list[str], float]]] = {
    "PHE": [("6ring", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], 1.00)],
    "TYR": [("6ring", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], 0.94)],
    "TRP": [
        ("6ring", ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"], 1.04),
        ("5ring", ["CG", "CD1", "CD2", "NE1", "CE2"], 0.56),
    ],
    "HIS": [("5ring", ["CG", "ND1", "CD2", "CE1", "NE2"], 0.43)],
}

# Ring current constant: B = 5.455 × 10^-6 ppm·Å³ for benzene
# This is the Johnson-Bovey constant scaled for the point-dipole approximation
_RC_CONSTANT = 5.455e-6  # ppm·Å³


def _ring_current_at_point(
    point: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    intensity: float,
) -> float:
    """Compute ring current shift contribution at a point.

    Uses the point-dipole approximation (Haigh-Mallion simplification):
        δ_rc = i × B × (3cos²θ - 1) / r³

    where r is the distance from ring centroid to the point,
    θ is the angle between the ring normal and the centroid-point vector.

    Args:
        point: [3] target nucleus position.
        centroid: [3] ring centroid.
        normal: [3] unit normal to the ring plane.
        intensity: Ring current intensity relative to benzene.

    Returns:
        Ring current shift contribution in ppm.
    """
    d = point - centroid
    r = np.linalg.norm(d)
    if r < 1.0:  # too close — unphysical
        return 0.0

    cos_theta = np.dot(d, normal) / r
    # Point-dipole: δ = i × B × (3cos²θ - 1) / r³
    # With B ≈ 27.28 ppm·Å³ for the full Johnson-Bovey model
    # We use a simpler empirical constant calibrated to reproduce
    # SPARTA+ ring current magnitudes (~0.3-1.0 ppm for H near rings)
    B = 27.28  # ppm·Å³ (empirical, from Case 1995)
    return intensity * B * (3.0 * cos_theta**2 - 1.0) / (r**3)


def compute_ring_currents(
    pos: np.ndarray,
    residue_idx: np.ndarray,
    residue_types: np.ndarray,
    atom_names: list[str],
    target_indices: np.ndarray,
    target_mask: np.ndarray,
    num_residues: int,
) -> np.ndarray:
    """Compute ring current shift contributions at each target nucleus.

    Args:
        pos: [N_atoms, 3] atomic positions.
        residue_idx: [N_atoms] residue index per atom.
        residue_types: [R] AA type index per residue (0-19).
        atom_names: List of atom name strings per atom.
        target_indices: [R, 6] atom index for each target nucleus.
        target_mask: [R, 6] bool mask for valid targets.
        num_residues: R.

    Returns:
        [R, 6] ring current shift at each target position (ppm).
        Order: H, HA, N, CA, CB, C.
    """
    # D30 fix (2026-04-22): canonical AA index order matching
    # features.AA_THREE_TO_IDX. Previous list was alphabetical-by-one-letter
    # (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y) which mismapped canonical
    # residue_types indices to wrong AA names. Consequence: ring currents
    # attributed PHE aromatic behavior to CYS (idx 4), suppressed PHE/HIS/TRP
    # ring currents entirely (idx 13/8/17 looked up as GLN/LYS/VAL), and
    # gave TYR (idx 18) TRP's ring geometry. Ring current slots [12-17] of
    # residue_geometry were garbage on every model since D14.
    _IDX_TO_AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    rc = np.zeros((num_residues, 6), dtype=np.float32)

    # Build per-residue atom name lookup
    res_atom_pos: dict[int, dict[str, int]] = {}
    for ai, name in enumerate(atom_names):
        ri = int(residue_idx[ai])
        res_atom_pos.setdefault(ri, {})[name] = ai

    # Collect all aromatic rings
    rings: list[tuple[np.ndarray, np.ndarray, float]] = []  # (centroid, normal, intensity)
    for ri in range(num_residues):
        aa_idx = int(residue_types[ri])
        if aa_idx >= len(_IDX_TO_AA3):
            continue
        aa3 = _IDX_TO_AA3[aa_idx]
        if aa3 not in _AROMATIC_RINGS:
            continue

        ra = res_atom_pos.get(ri, {})
        for ring_name, ring_atoms, intensity in _AROMATIC_RINGS[aa3]:
            atom_indices = [ra.get(name) for name in ring_atoms]
            if any(idx is None for idx in atom_indices):
                continue
            ring_pos = pos[atom_indices]  # [n_ring, 3]
            centroid = ring_pos.mean(axis=0)
            # Ring normal from cross product of two in-plane vectors
            v1 = ring_pos[1] - ring_pos[0]
            v2 = ring_pos[2] - ring_pos[0]
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len
            rings.append((centroid, normal, intensity))

    if not rings:
        return rc

    # Compute ring current at each target nucleus position
    for ri in range(num_residues):
        for ni in range(6):
            if not target_mask[ri, ni]:
                continue
            atom_idx = int(target_indices[ri, ni])
            if atom_idx < 0 or atom_idx >= len(pos):
                continue
            point = pos[atom_idx]
            total = 0.0
            for centroid, normal, intensity in rings:
                total += _ring_current_at_point(point, centroid, normal, intensity)
            rc[ri, ni] = total

    return rc


# ---------------------------------------------------------------------------
# Hydrogen bond geometry
# ---------------------------------------------------------------------------

def compute_hbond_geometry(
    pos: np.ndarray,
    atom_role: np.ndarray,
    residue_idx: np.ndarray,
    num_residues: int,
    bb_lookup: dict[int, dict[int, int]],
    return_partner_dirs: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Compute backbone hydrogen bond geometry features per residue.

    For each residue, finds the best NH→O=C (donor) and C=O←H-N (acceptor)
    hydrogen bonds and returns distance + angle features.

    Args:
        pos: [N_atoms, 3] positions.
        atom_role: [N_atoms] role indices (N=0, CA=1, C=2, O=3, H=5).
        residue_idx: [N_atoms] residue index.
        num_residues: R.
        bb_lookup: {res_i: {role: atom_idx}} for backbone atoms.
        return_partner_dirs: if True, also return a [R, 6] array of unit
            vectors pointing from each residue's amide H/N (or carbonyl O/C)
            toward its H-bond partner. Zero rows if no partner. Used by
            D42's H-bond frame projection.

    Returns:
        [R, 7] features (default), or ([R, 7], [R, 6]) if return_partner_dirs:
          [0] hb_nh_d: NH→O distance / 3.5 (1.0 if no bond)
          [1] hb_nh_cos_nho: cos(N-H···O) donor angle (0 if no bond)
          [2] hb_nh_cos_hoc: cos(H···O=C) acceptor angle (0 if no bond)
          [3] hb_nh_present: 1.0 if NH H-bond detected
          [4] hb_co_d: CO←H distance / 3.5 (1.0 if no bond)
          [5] hb_co_cos_coh: cos(C=O···H) angle (0 if no bond)
          [6] hb_co_present: 1.0 if CO H-bond detected

        If return_partner_dirs:
          partner_dirs[:, 0:3]: nh_partner_dir (NH donor → O acceptor unit vec)
          partner_dirs[:, 3:6]: co_partner_dir (CO acceptor → H donor unit vec)
    """
    hb = np.zeros((num_residues, 7), dtype=np.float32)
    partner_dirs = np.zeros((num_residues, 6), dtype=np.float32)
    # Default: no bond → distance = 3.5/3.5 = 1.0
    hb[:, 0] = 1.0
    hb[:, 4] = 1.0

    # Collect all backbone N, H, C, O positions with residue info
    n_atoms_list = []   # (ri, N_pos, H_pos_or_None)
    co_atoms_list = []  # (ri, C_pos, O_pos)

    for ri, roles in bb_lookup.items():
        n_idx = roles.get(0)  # N
        h_idx = roles.get(5)  # H (amide)
        c_idx = roles.get(2)  # C
        o_idx = roles.get(3)  # O

        if n_idx is not None:
            h_pos = pos[h_idx] if h_idx is not None else None
            n_atoms_list.append((ri, pos[n_idx], h_pos))
        if c_idx is not None and o_idx is not None:
            co_atoms_list.append((ri, pos[c_idx], pos[o_idx]))

    if not n_atoms_list or not co_atoms_list:
        return hb

    # For each residue's NH, find best O=C acceptor
    for n_ri, n_pos, h_pos in n_atoms_list:
        best_d = 3.5
        best_cos_nho = 0.0
        best_cos_hoc = 0.0
        best_partner_o = None  # D42: store the chosen O position for partner_dir
        found = False

        for co_ri, c_pos, o_pos in co_atoms_list:
            if abs(co_ri - n_ri) < 2:  # skip self and immediate neighbors
                continue

            # Use H position if available, else estimate from N position
            if h_pos is not None:
                d_ho = np.linalg.norm(h_pos - o_pos)
            else:
                # Fallback: use N···O distance - 1.0 Å as proxy
                d_no = np.linalg.norm(n_pos - o_pos)
                d_ho = d_no - 1.0
                if d_no > 3.5:
                    continue

            if d_ho >= best_d or d_ho > 2.8:
                continue

            # Compute angles
            if h_pos is not None:
                # cos(N-H···O): angle at H
                hn = n_pos - h_pos
                ho = o_pos - h_pos
                hn_n = np.linalg.norm(hn)
                ho_n = np.linalg.norm(ho)
                if hn_n > 1e-6 and ho_n > 1e-6:
                    cos_nho = np.dot(hn, ho) / (hn_n * ho_n)
                else:
                    cos_nho = 0.0

                # cos(H···O=C): angle at O
                oh = h_pos - o_pos
                oc = c_pos - o_pos
                oh_n = np.linalg.norm(oh)
                oc_n = np.linalg.norm(oc)
                if oh_n > 1e-6 and oc_n > 1e-6:
                    cos_hoc = np.dot(oh, oc) / (oh_n * oc_n)
                else:
                    cos_hoc = 0.0

                # Require reasonable geometry: N-H···O angle > 120° (cos < -0.5)
                if cos_nho > -0.3:  # angle too acute
                    continue
            else:
                cos_nho = 0.0
                cos_hoc = 0.0

            best_d = d_ho
            best_cos_nho = cos_nho
            best_cos_hoc = cos_hoc
            best_partner_o = o_pos
            found = True

        hb[n_ri, 0] = best_d / 3.5
        hb[n_ri, 1] = best_cos_nho
        hb[n_ri, 2] = best_cos_hoc
        hb[n_ri, 3] = 1.0 if found else 0.0
        # D42: partner direction = (acceptor_O − donor_anchor) / |·|.
        # Use H position when available, else fall back to N.
        if found and best_partner_o is not None:
            anchor = h_pos if h_pos is not None else n_pos
            d_vec = best_partner_o - anchor
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-6:
                partner_dirs[n_ri, 0:3] = d_vec / d_norm

    # For each residue's C=O, find best H-N donor
    for co_ri, c_pos, o_pos in co_atoms_list:
        best_d = 3.5
        best_cos_coh = 0.0
        best_partner_donor = None  # D42: store the chosen H/N position
        found = False

        for n_ri, n_pos, h_pos in n_atoms_list:
            if abs(n_ri - co_ri) < 2:
                continue

            if h_pos is not None:
                d_oh = np.linalg.norm(o_pos - h_pos)
            else:
                d_on = np.linalg.norm(o_pos - n_pos)
                d_oh = d_on - 1.0
                if d_on > 3.5:
                    continue

            if d_oh >= best_d or d_oh > 2.8:
                continue

            # cos(C=O···H) angle at O
            oc = c_pos - o_pos
            if h_pos is not None:
                oh = h_pos - o_pos
            else:
                oh = n_pos - o_pos
            oc_n = np.linalg.norm(oc)
            oh_n = np.linalg.norm(oh)
            if oc_n > 1e-6 and oh_n > 1e-6:
                cos_coh = np.dot(oc, oh) / (oc_n * oh_n)
            else:
                cos_coh = 0.0

            best_d = d_oh
            best_cos_coh = cos_coh
            best_partner_donor = h_pos if h_pos is not None else n_pos
            found = True

        hb[co_ri, 4] = best_d / 3.5
        hb[co_ri, 5] = best_cos_coh
        hb[co_ri, 6] = 1.0 if found else 0.0
        # D42: partner direction = (donor − acceptor_O) / |·|.
        if found and best_partner_donor is not None:
            d_vec = best_partner_donor - o_pos
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-6:
                partner_dirs[co_ri, 3:6] = d_vec / d_norm

    if return_partner_dirs:
        return hb, partner_dirs
    return hb


# ---------------------------------------------------------------------------
# Solvent-accessible surface area (Shrake-Rupley) + Half-sphere exposure
# ---------------------------------------------------------------------------

# van der Waals radii (Å) by element
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}
_PROBE_RADIUS = 1.40  # water probe

# Maximum SASA per residue type (extended chain, empirical, Å²)
_MAX_SASA: dict[str, float] = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}


def compute_sasa_and_hse(
    pos: np.ndarray,
    elements: np.ndarray,
    residue_idx: np.ndarray,
    residue_types: np.ndarray,
    bb_lookup: dict[int, dict[int, int]],
    num_residues: int,
    n_sphere_points: int = 92,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative SASA and half-sphere exposure per residue.

    Uses KDTree-accelerated Shrake-Rupley algorithm.
    HSE splits the CA neighborhood into upper (CA->CB direction) and
    lower hemispheres, counting CA neighbors in each.

    Args:
        pos: [N_atoms, 3] positions.
        elements: [N_atoms] element indices (C=0, N=1, O=2, S=3, H=4).
        residue_idx: [N_atoms] residue index.
        residue_types: [R] AA type indices.
        bb_lookup: {ri: {role: atom_idx}}.
        num_residues: R.
        n_sphere_points: Points on unit sphere for SASA (Fibonacci).

    Returns:
        Tuple of:
          - residue_features: [R, 3] — [rSASA (0-1), HSE_up (/50), HSE_down (/50)]
          - atom_rsasa: [N_atoms] — atom-sphere-normalized rSASA
            (= n_exposed / n_sphere_points) for heavy atoms; NaN for H
            atoms. This is the *fraction of the atom's own vdW+probe
            sphere that is solvent-exposed*, independent of AA identity
            or residue-level reference — pure per-atom granularity.
    """
    from scipy.spatial import cKDTree

    # D30 fix (2026-04-22): canonical AA index order matching
    # features.AA_THREE_TO_IDX. Previous list was alphabetical-by-one-letter
    # ("ACDEFGHIKLMNPQRSTVWY") — only ALA, SER, THR happened to align with
    # canonical indexing by coincidence. All 17 other AAs were normalized
    # by the wrong _MAX_SASA divisor. Per-residue rSASA at slot 25 of
    # residue_geometry was systematically wrong since D14.
    # Canonical order: ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
    # LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
    # One-letter:      A,   R,   N,   D,   C,   Q,   E,   G,   H,   I,
    #                  L,   K,   M,   F,   P,   S,   T,   W,   Y,   V
    _IDX_TO_AA1 = "ARNDCQEGHILKMFPSTWYV"
    _ELEM_MAP = {0: "C", 1: "N", 2: "O", 3: "S", 4: "H"}

    result = np.zeros((num_residues, 3), dtype=np.float32)

    # Generate Fibonacci sphere points
    phi_golden = (1 + math.sqrt(5)) / 2
    sphere = np.zeros((n_sphere_points, 3))
    for i in range(n_sphere_points):
        theta = math.acos(1 - 2 * (i + 0.5) / n_sphere_points)
        phi_angle = 2 * math.pi * i / phi_golden
        sphere[i] = [math.sin(theta) * math.cos(phi_angle),
                      math.sin(theta) * math.sin(phi_angle),
                      math.cos(theta)]

    # Compute radii per atom (heavy atoms only)
    elem_arr = elements.numpy() if hasattr(elements, 'numpy') else np.asarray(elements)
    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    res_idx_np = residue_idx.numpy() if hasattr(residue_idx, 'numpy') else np.asarray(residue_idx)

    # Atom-sphere-normalized rSASA per atom (NaN for H; filled for heavy atoms below)
    atom_rsasa_full = np.full(len(elem_arr), np.nan, dtype=np.float32)

    heavy_mask = elem_arr != 4
    heavy_idx = np.where(heavy_mask)[0]
    if len(heavy_idx) == 0:
        return result, atom_rsasa_full

    heavy_pos = pos_np[heavy_idx]
    heavy_radii = np.array([
        _VDW_RADII.get(_ELEM_MAP.get(int(elem_arr[ai]), "C"), 1.70) + _PROBE_RADIUS
        for ai in heavy_idx
    ])

    # KDTree for fast neighbor lookup
    max_radius = heavy_radii.max()
    tree = cKDTree(heavy_pos)

    # Per-atom SASA via Shrake-Rupley with KDTree acceleration
    atom_sasa = np.zeros(len(heavy_idx), dtype=np.float64)
    for ii in range(len(heavy_idx)):
        r_i = heavy_radii[ii]
        center = heavy_pos[ii]

        # Find all atoms that could possibly occlude test points
        # Max distance: r_i (test point radius) + max_radius (neighbor radius)
        neighbor_indices = tree.query_ball_point(center, r_i + max_radius)

        test_points = center + r_i * sphere  # [n_sphere, 3]
        n_exposed = 0
        for sp in test_points:
            exposed = True
            for jj in neighbor_indices:
                if jj == ii:
                    continue
                d = np.linalg.norm(sp - heavy_pos[jj])
                if d < heavy_radii[jj]:
                    exposed = False
                    break
            if exposed:
                n_exposed += 1

        atom_sasa[ii] = 4 * math.pi * r_i**2 * n_exposed / n_sphere_points
        # Atom-sphere normalized rSASA: fraction of this atom's own
        # vdW+probe sphere that is solvent-exposed. Equivalent to
        # atom_sasa[ii] / (4*pi*r_i**2). Pure per-atom quantity, no
        # AA-type or residue-level reference.
        atom_rsasa_full[heavy_idx[ii]] = n_exposed / n_sphere_points

    # Aggregate per-residue SASA
    res_sasa = np.zeros(num_residues, dtype=np.float64)
    for ii, ai in enumerate(heavy_idx):
        ri = int(res_idx_np[ai])
        if ri < num_residues:
            res_sasa[ri] += atom_sasa[ii]

    # Normalize by max SASA per AA type
    res_types_np = residue_types.numpy() if hasattr(residue_types, 'numpy') else np.asarray(residue_types)
    for ri in range(num_residues):
        aa_idx = int(res_types_np[ri])
        if aa_idx < len(_IDX_TO_AA1):
            aa1 = _IDX_TO_AA1[aa_idx]
            max_sasa = _MAX_SASA.get(aa1, 200.0)
            result[ri, 0] = min(res_sasa[ri] / max_sasa, 1.0)

    # Half-sphere exposure (HSE)
    ca_positions = {}
    cb_positions = {}
    for ri, roles in bb_lookup.items():
        if 1 in roles:
            ca_p = pos_np[roles[1]]
            ca_positions[ri] = ca_p
        if 4 in roles:
            cb_p = pos_np[roles[4]]
            cb_positions[ri] = cb_p

    hse_cutoff = 13.0
    if ca_positions:
        ca_ri_list = sorted(ca_positions.keys())
        ca_arr = np.array([ca_positions[ri] for ri in ca_ri_list])
        ca_tree = cKDTree(ca_arr)

        for idx_i, ri in enumerate(ca_ri_list):
            ca_i = ca_arr[idx_i]

            if ri in cb_positions:
                direction = cb_positions[ri] - ca_i
            elif ri + 1 in ca_positions:
                direction = ca_positions[ri + 1] - ca_i
            else:
                continue

            d_norm = np.linalg.norm(direction)
            if d_norm < 1e-6:
                continue
            direction = direction / d_norm

            neighbors = ca_tree.query_ball_point(ca_i, hse_cutoff)
            up_count = 0
            down_count = 0
            for idx_j in neighbors:
                if idx_j == idx_i:
                    continue
                diff = ca_arr[idx_j] - ca_i
                if np.dot(diff, direction) > 0:
                    up_count += 1
                else:
                    down_count += 1

            result[ri, 1] = up_count / 50.0
            result[ri, 2] = down_count / 50.0

    return result, atom_rsasa_full


def compute_target_atom_rsasa(
    atom_rsasa_full: np.ndarray,
    target_indices: np.ndarray,
    target_mask: np.ndarray,
    atom_role: np.ndarray,
    bb_lookup: dict[int, dict[int, int]],
    num_residues: int,
) -> np.ndarray:
    """Build per-target-atom rSASA tensor of shape [R, 6].

    Target-nucleus indexing matches BACKBONE_NUCLEI:
        0=H, 1=HA, 2=N, 3=CA, 4=CB, 5=C

    Heavy target atoms (N, CA, CB, C) look up their own rSASA directly.
    Hydrogen targets (H, HA) inherit their parent heavy atom's rSASA:
        H  → parent N  (role index 0)
        HA → parent CA (role index 1)
    Missing target atoms return 0.0 (and the corresponding mask slot
    will already be False in target_mask).

    Args:
        atom_rsasa_full: [N_atoms] atom-sphere-normalized rSASA from
            compute_sasa_and_hse (NaN for H atoms).
        target_indices: [R, 6] atom index per target nucleus.
        target_mask: [R, 6] bool — True if target atom exists.
        atom_role: [N_atoms] role indices (N=0, CA=1, C=2, O=3, CB=4,
            H=5, HA=6, sc_heavy=7, sc_H=8).
        bb_lookup: {ri: {role: atom_idx}}.
        num_residues: R.

    Returns:
        [R, 6] float32 — per-target-atom rSASA (0-1), 0.0 where absent.
    """
    R = num_residues
    out = np.zeros((R, 6), dtype=np.float32)
    N = len(atom_rsasa_full)

    for ri in range(R):
        bb = bb_lookup.get(ri, {})
        for ni in range(6):
            if not target_mask[ri, ni]:
                continue
            atom_idx = int(target_indices[ri, ni])
            if atom_idx < 0 or atom_idx >= N:
                continue

            val = atom_rsasa_full[atom_idx]
            if not np.isnan(val):
                out[ri, ni] = float(val)
                continue

            # H or HA: inherit from parent heavy atom.
            # ni 0 = H  → parent N  (bb role 0)
            # ni 1 = HA → parent CA (bb role 1)
            parent_role = 0 if ni == 0 else (1 if ni == 1 else None)
            if parent_role is None:
                continue
            parent_ai = bb.get(parent_role)
            if parent_ai is None:
                continue
            pval = atom_rsasa_full[parent_ai]
            if not np.isnan(pval):
                out[ri, ni] = float(pval)

    return out


# ---------------------------------------------------------------------------
# DSSP secondary structure
# ---------------------------------------------------------------------------

def compute_dssp_3state(
    structure_path: str,
    chain_id: str | None = None,
    model_idx: int = 0,
    num_residues: int = 0,
    seq_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 3-state secondary structure encoding per residue.

    Uses gemmi's built-in DSSP implementation if available, otherwise
    falls back to the crystalline dssp module.

    Returns:
        [R] float32: helix(H/G/I)=1.0, sheet(E/B)=-1.0, coil(T/S/C/-)=0.0
    """
    result = np.zeros(num_residues, dtype=np.float32)

    try:
        import gemmi

        path = str(structure_path)
        if path.endswith((".cif", ".mmcif")):
            doc = gemmi.cif.read(path)
            st = gemmi.make_structure_from_block(doc[0])
        else:
            st = gemmi.read_structure(path)

        if model_idx >= len(st):
            return result

        model = st[model_idx]

        # Try gemmi's built-in DSSP
        try:
            st.setup_entities()
            st.assign_subchains()
            dssp_result = gemmi.calculate_dssp(st, model_idx=model_idx)

            # Map: H/G/I -> helix, E/B -> sheet, everything else -> coil
            helix_codes = {'H', 'G', 'I'}
            sheet_codes = {'E', 'B'}

            # Build seqid -> dssp mapping from the result
            # gemmi.calculate_dssp returns a list of (chain, residue, ss_code) or similar
            # The exact API depends on gemmi version; fall through to fallback if it fails
            # Bug R: track whether the string-specific branch actually ran,
            # so we don't return an all-zero 'coil' result when a newer
            # gemmi returns a non-string iterable we don't know how to
            # consume. On non-string returns, fall through to the PDB-
            # records fallback instead of silently returning zeros.
            _dssp_handled = False
            if isinstance(dssp_result, str):
                target_chain = None
                for chain in model:
                    if chain_id is None or chain.name == chain_id:
                        target_chain = chain
                        break
                if target_chain is not None and seq_ids is not None:
                    chain_residues = list(target_chain.get_polymer())
                    ss_by_seqid = {}
                    for i, res in enumerate(chain_residues):
                        if i < len(dssp_result):
                            ss_by_seqid[res.seqid.num] = dssp_result[i]
                    for ri in range(num_residues):
                        sid = int(seq_ids[ri])
                        ss = ss_by_seqid.get(sid, ' ')
                        if ss in helix_codes:
                            result[ri] = 1.0
                        elif ss in sheet_codes:
                            result[ri] = -1.0
                    _dssp_handled = True
            elif dssp_result is not None:
                logger.warning(
                    "compute_dssp_3state: gemmi.calculate_dssp returned a "
                    "non-string result (type=%s); falling back to PDB "
                    "helix/strand records. Upgrade the DSSP handler for "
                    "the current gemmi version to restore full coverage.",
                    type(dssp_result).__name__,
                )
            if _dssp_handled:
                return result
            # else: fall through to the PDB-deposited fallback below
        except (AttributeError, TypeError):
            pass  # gemmi version doesn't have calculate_dssp

        # Fallback: use PDB-deposited helix/strand records
        helix_ranges = []
        strand_ranges = []

        for helix in st.helices:
            helix_ranges.append((
                helix.start.chain_name, helix.start.res_id.seqid.num,
                helix.end.chain_name, helix.end.res_id.seqid.num,
            ))

        for sheet in st.sheets:
            for strand in sheet.strands:
                strand_ranges.append((
                    strand.start.chain_name, strand.start.res_id.seqid.num,
                    strand.end.chain_name, strand.end.res_id.seqid.num,
                ))

        if seq_ids is not None:
            target_cid = chain_id or ""
            for ri in range(num_residues):
                sid = int(seq_ids[ri])
                for cname_s, s_start, cname_e, s_end in helix_ranges:
                    if (cname_s == target_cid or not target_cid) and s_start <= sid <= s_end:
                        result[ri] = 1.0
                        break
                else:
                    for cname_s, s_start, cname_e, s_end in strand_ranges:
                        if (cname_s == target_cid or not target_cid) and s_start <= sid <= s_end:
                            result[ri] = -1.0
                            break

    except Exception as e:
        logger.debug("DSSP computation failed: %s", e)

    # Bug R tripwire: for a real protein of ≥ 30 residues, an all-zero DSSP
    # result (100% coil) is essentially always a computation failure, not
    # a real biological signal. Log a debug-level hint so audits can spot
    # silent DSSP regressions after gemmi upgrades.
    if num_residues >= 30 and float(result.sum()) == 0.0 and float(np.abs(result).sum()) == 0.0:
        logger.debug(
            "compute_dssp_3state: all %d residues labeled coil (no H/E). "
            "Possibly a gemmi-API or PDB-records fallback miss.",
            num_residues,
        )

    return result


# ---------------------------------------------------------------------------
# Buckingham electric field (P3 — stub)
# ---------------------------------------------------------------------------

def compute_electric_field(
    pos: np.ndarray,
    residue_idx: np.ndarray,
    residue_types: np.ndarray,
    atom_names: list[str],
    bb_lookup: dict[int, dict[int, int]],
    num_residues: int,
) -> np.ndarray:
    """Compute Buckingham electric field along N-H and CA-HA bonds.

    E_z = sum_charges q_j * cos(theta_j) / (epsilon * r_j^2)
    where theta_j = angle between charge->nucleus vector and X-H bond vector.

    Charged atoms (formal charges at pH ~7):
      Asp/Glu: each carboxylate O carries -0.5e
      Lys: Nzeta carries +1e
      Arg: Czeta carries +1e (delocalized guanidinium)
      His: Ndelta/Nepsilon carry +0.5e each (protonated form, ~50% at pH 7)

    Returns:
        [R, 2]: [Efield_NH / 0.1, Efield_CaHa / 0.1]
    """
    # D29 fix (2026-04-21): canonical AA index order matching
    # features.AA_THREE_TO_IDX. The previous list had CYS at index 1
    # (should be 4), ASP at 2 (should be 3), GLU at 3 (should be 6),
    # etc. — every charged-residue lookup was reading the wrong AA from
    # the residue_types array, so the formal-charge sum was drawn from
    # essentially random positions in the protein. All checkpoints from
    # D14 onward were trained on scrambled Efield features at slots
    # 28-29 of residue_geometry.
    _IDX_TO_AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    # Charged atom definitions: {AA3: [(atom_name, charge)]}
    _CHARGED_ATOMS = {
        "ASP": [("OD1", -0.5), ("OD2", -0.5)],
        "GLU": [("OE1", -0.5), ("OE2", -0.5)],
        "LYS": [("NZ", 1.0)],
        "ARG": [("CZ", 1.0)],
        "HIS": [("ND1", 0.25), ("NE2", 0.25)],  # average protonation at pH ~7
    }

    EPSILON = 8.6  # apparent protein dielectric
    NORM = 0.1     # normalization constant

    result = np.zeros((num_residues, 2), dtype=np.float32)

    pos_np = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)

    # Build atom name -> position lookup
    res_atom_pos: dict[int, dict[str, int]] = {}
    for ai, name in enumerate(atom_names):
        ri = int(residue_idx[ai]) if hasattr(residue_idx, '__getitem__') else int(residue_idx)
        res_atom_pos.setdefault(ri, {})[name] = ai

    # Collect all charged atom positions
    charges: list[tuple[np.ndarray, float]] = []
    res_types_np = residue_types.numpy() if hasattr(residue_types, 'numpy') else np.asarray(residue_types)
    for ri in range(num_residues):
        aa_idx = int(res_types_np[ri])
        if aa_idx >= len(_IDX_TO_AA3):
            continue
        aa3 = _IDX_TO_AA3[aa_idx]
        if aa3 not in _CHARGED_ATOMS:
            continue
        ra = res_atom_pos.get(ri, {})
        for atom_name, charge in _CHARGED_ATOMS[aa3]:
            if atom_name in ra:
                charges.append((pos_np[ra[atom_name]], charge))

    if not charges:
        return result

    charge_pos = np.array([c[0] for c in charges])
    charge_vals = np.array([c[1] for c in charges])

    # For each residue, compute E_z along N-H and CA-HA bonds
    for ri in range(num_residues):
        roles = bb_lookup.get(ri, {})

        # E-field along N-H bond
        n_idx = roles.get(0)  # N
        h_idx = roles.get(5)  # H (amide)
        if n_idx is not None and h_idx is not None:
            h_pos = pos_np[h_idx]
            n_pos = pos_np[n_idx]
            bond_vec = h_pos - n_pos
            bond_len = np.linalg.norm(bond_vec)
            if bond_len > 1e-6:
                bond_unit = bond_vec / bond_len
                # E_z at H position from all charges
                d_vecs = charge_pos - h_pos  # [n_charges, 3]
                d_norms = np.linalg.norm(d_vecs, axis=1)
                valid = d_norms > 1.0  # skip very close (same residue)
                if valid.any():
                    cos_theta = np.sum(d_vecs[valid] * bond_unit, axis=1) / d_norms[valid]
                    e_z = np.sum(charge_vals[valid] * cos_theta / (EPSILON * d_norms[valid]**2))
                    result[ri, 0] = e_z / NORM

        # E-field along CA-HA bond
        ca_idx = roles.get(1)  # CA
        ha_idx = roles.get(6)  # HA
        if ca_idx is not None and ha_idx is not None:
            ha_pos = pos_np[ha_idx]
            ca_pos = pos_np[ca_idx]
            bond_vec = ha_pos - ca_pos
            bond_len = np.linalg.norm(bond_vec)
            if bond_len > 1e-6:
                bond_unit = bond_vec / bond_len
                d_vecs = charge_pos - ha_pos
                d_norms = np.linalg.norm(d_vecs, axis=1)
                valid = d_norms > 1.0
                if valid.any():
                    cos_theta = np.sum(d_vecs[valid] * bond_unit, axis=1) / d_norms[valid]
                    e_z = np.sum(charge_vals[valid] * cos_theta / (EPSILON * d_norms[valid]**2))
                    result[ri, 1] = e_z / NORM

    return result


# ---------------------------------------------------------------------------
# Per-target-atom local shielding environment
# ---------------------------------------------------------------------------

# Aromatic ring atoms by AA index (for atom classification)
_AROMATIC_ATOM_NAMES: dict[int, set[str]] = {
    13: {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},              # PHE
    18: {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},              # TYR
    17: {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},  # TRP
    8:  {"CG", "ND1", "CD2", "CE1", "NE2"},                    # HIS
}

# Chemical class indices
_CLS_BB_PEPTIDE = 0
_CLS_HYDROPHOBIC_C = 1
_CLS_POLAR_O = 2
_CLS_POLAR_N = 3
_CLS_SULFUR = 4
_CLS_AROMATIC = 5
_NUM_CLASSES = 6

# Normalization divisors per (class, shell) — keeps values in ~[0, 1]
_NORM_ENV: dict[tuple[int, int], float] = {
    (_CLS_BB_PEPTIDE, 0): 10.0, (_CLS_BB_PEPTIDE, 1): 30.0,
    (_CLS_HYDROPHOBIC_C, 0): 10.0, (_CLS_HYDROPHOBIC_C, 1): 30.0,
    (_CLS_POLAR_O, 0): 5.0, (_CLS_POLAR_O, 1): 15.0,
    (_CLS_POLAR_N, 0): 5.0, (_CLS_POLAR_N, 1): 15.0,
    (_CLS_SULFUR, 0): 3.0, (_CLS_SULFUR, 1): 5.0,
    (_CLS_AROMATIC, 0): 10.0, (_CLS_AROMATIC, 1): 30.0,
}


def compute_target_atom_environment(
    pos: np.ndarray,
    elements: np.ndarray,
    atom_role: np.ndarray,
    residue_idx: np.ndarray,
    residue_types: np.ndarray,
    atom_names: list[str] | None,
    target_indices: np.ndarray,
    target_mask: np.ndarray,
    disulfide: np.ndarray,
    num_residues: int,
) -> np.ndarray:
    """Compute per-target-atom local shielding environment features.

    For each of the 6 target nucleus atoms per residue, computes
    shell-resolved typed neighbor counts, aromatic proximity, and
    disulfide geometry.  See ``features.py`` for the 15-slot layout.

    Args:
        pos: [N_atoms, 3] atomic positions.
        elements: [N_atoms] element indices (C=0, N=1, O=2, S=3, H=4).
        atom_role: [N_atoms] role indices (N=0, CA=1, C=2, O=3, CB=4,
            H=5, HA=6, sc_heavy=7, sc_H=8).
        residue_idx: [N_atoms] residue index per atom.
        residue_types: [R] AA type index per residue.
        atom_names: [N_atoms] PDB atom name strings (for aromatic ID).
        target_indices: [R, 6] atom index per target nucleus.
        target_mask: [R, 6] bool — True if target atom exists.
        disulfide: [R] bool — True if CYS in SS bond.
        num_residues: R.

    Returns:
        [R, 6, 15] float32 environment features.
    """
    from scipy.spatial import cKDTree
    from caustic.features import NUM_TARGET_ENV_FEATURES

    R = num_residues
    env = np.zeros((R, 6, NUM_TARGET_ENV_FEATURES), dtype=np.float32)

    # Default sentinel values
    env[:, :, 12] = 1.0  # nearest_aromatic_dist (no ring nearby)
    env[:, :, 14] = 1.0  # nearest_SG_dist (no disulfide nearby)

    N = len(pos)
    if N == 0:
        return env

    # ------------------------------------------------------------------
    # Build per-atom chemical class array (heavy atoms only)
    # ------------------------------------------------------------------
    atom_class = np.full(N, -1, dtype=np.int8)  # -1 = skip (H atoms)

    # Backbone peptide: roles 0 (N), 1 (CA), 2 (C), 3 (O)
    bb_mask = (atom_role == 0) | (atom_role == 1) | (atom_role == 2) | (atom_role == 3)
    atom_class[bb_mask] = _CLS_BB_PEPTIDE

    # CB (role 4): treated as hydrophobic carbon
    atom_class[atom_role == 4] = _CLS_HYDROPHOBIC_C

    # Sidechain heavy atoms (role 7): classify by element
    sc_heavy = atom_role == 7
    atom_class[sc_heavy & (elements == 0)] = _CLS_HYDROPHOBIC_C  # C
    atom_class[sc_heavy & (elements == 2)] = _CLS_POLAR_O         # O
    atom_class[sc_heavy & (elements == 1)] = _CLS_POLAR_N         # N
    atom_class[sc_heavy & (elements == 3)] = _CLS_SULFUR           # S

    # Override aromatic atoms
    if atom_names is not None:
        for ai in range(N):
            ri = int(residue_idx[ai])
            if ri >= R:
                continue
            aa = int(residue_types[ri])
            if aa in _AROMATIC_ATOM_NAMES:
                name = atom_names[ai].strip() if isinstance(atom_names[ai], str) else atom_names[ai]
                if name in _AROMATIC_ATOM_NAMES[aa]:
                    atom_class[ai] = _CLS_AROMATIC

    # Heavy atom mask (classified atoms only)
    heavy_idx = np.where(atom_class >= 0)[0]
    if len(heavy_idx) == 0:
        return env

    heavy_pos = pos[heavy_idx]
    heavy_class = atom_class[heavy_idx]

    # ------------------------------------------------------------------
    # Build KDTree on heavy atoms
    # ------------------------------------------------------------------
    tree = cKDTree(heavy_pos)

    # ------------------------------------------------------------------
    # Collect aromatic ring centroids + normals
    # ------------------------------------------------------------------
    rings: list[tuple[np.ndarray, np.ndarray]] = []  # (centroid, unit_normal)

    # Correct integer-to-3letter mapping (matches AA_THREE_TO_IDX)
    _IDX_TO_AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    if atom_names is not None:
        res_atom_pos: dict[int, dict[str, int]] = {}
        for ai, name in enumerate(atom_names):
            ri = int(residue_idx[ai])
            n = name.strip() if isinstance(name, str) else name
            res_atom_pos.setdefault(ri, {})[n] = ai

        for ri in range(R):
            aa = int(residue_types[ri])
            aa3 = _IDX_TO_AA3[aa] if aa < 20 else ""
            if aa3 not in _AROMATIC_RINGS:
                continue
            ra = res_atom_pos.get(ri, {})
            for _, ring_atoms, _ in _AROMATIC_RINGS[aa3]:
                aidxs = [ra.get(n) for n in ring_atoms]
                if any(idx is None for idx in aidxs):
                    continue
                ring_pos = pos[aidxs]
                centroid = ring_pos.mean(axis=0)
                v1 = ring_pos[1] - ring_pos[0]
                v2 = ring_pos[2] - ring_pos[0]
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-8:
                    continue
                rings.append((centroid, normal / norm_len))

    # ------------------------------------------------------------------
    # Collect disulfide SG positions
    # ------------------------------------------------------------------
    sg_positions: list[np.ndarray] = []
    if atom_names is not None and disulfide.any():
        for ai, name in enumerate(atom_names):
            ri = int(residue_idx[ai])
            if ri < R and disulfide[ri]:
                n = name.strip() if isinstance(name, str) else name
                if n == "SG":
                    sg_positions.append(pos[ai])

    sg_arr = np.array(sg_positions) if sg_positions else None

    # ------------------------------------------------------------------
    # For each target atom, compute environment features
    # ------------------------------------------------------------------
    for ri in range(R):
        for ni in range(6):
            if not target_mask[ri, ni]:
                continue

            atom_idx = int(target_indices[ri, ni])
            if atom_idx < 0 or atom_idx >= N:
                continue

            target_pos = pos[atom_idx]

            # Query heavy-atom neighbors within 8 A
            neighbor_heavy_idxs = tree.query_ball_point(target_pos, 8.0)

            for hi in neighbor_heavy_idxs:
                real_ai = heavy_idx[hi]
                if real_ai == atom_idx:
                    continue  # skip self

                cls = int(heavy_class[hi])
                dist = np.linalg.norm(heavy_pos[hi] - target_pos)
                shell = 0 if dist < 4.0 else 1
                slot = cls * 2 + shell
                env[ri, ni, slot] += 1.0 / _NORM_ENV[(cls, shell)]

            # Aromatic proximity: nearest centroid + cos(normal)
            if rings:
                best_d = 999.0
                best_cos = 0.0
                for centroid, normal in rings:
                    d_vec = centroid - target_pos
                    d = np.linalg.norm(d_vec)
                    if 0.5 < d < best_d:
                        best_d = d
                        best_cos = np.dot(d_vec, normal) / d
                if best_d < 999.0:
                    env[ri, ni, 12] = best_d / 8.0
                    env[ri, ni, 13] = best_cos

            # Disulfide SG distance
            if sg_arr is not None:
                dists_sg = np.linalg.norm(sg_arr - target_pos, axis=1)
                valid_sg = dists_sg > 0.5  # exclude self
                if valid_sg.any():
                    env[ri, ni, 14] = dists_sg[valid_sg].min() / 8.0

    return env
