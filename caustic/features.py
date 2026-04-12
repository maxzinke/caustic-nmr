"""Feature encoding utilities: vocabularies, RBF expansion, atom roles."""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Backbone nuclei we predict (order matters — matches shift array columns)
# ---------------------------------------------------------------------------
BACKBONE_NUCLEI: tuple[str, ...] = ("H", "HA", "N", "CA", "CB", "C")
NUM_NUCLEI = 6

# Per-residue geometry features — D6 layout (34 features)
# Backbone torsions (kept from D5):
# [0-1]   sin/cos(phi), [2-3] sin/cos(psi), [4] cos(omega)
# Sidechain torsion (kept):
# [5-6]   sin/cos(chi1), [7] has_chi1
# Flags + ensemble (kept):
# [8]     is_disulfide (CYS in SS bond)
# [9]     phi circular variance (0-1, ensemble)
# [10]    psi circular variance (0-1, ensemble)
# [11]    num_conformers (n/20, capped at 1.0)
# Ring current shifts (NEW — Haigh-Mallion):
# [12-17] rc_H, rc_HA, rc_N, rc_CA, rc_CB, rc_C (/1.0 ppm)
# Hydrogen bond geometry (NEW — replaces old H-bond count):
# [18]    hb_nh_d (NH→O distance / 3.5)
# [19]    hb_nh_cos_nho (donor angle)
# [20]    hb_nh_cos_hoc (acceptor angle)
# [21]    hb_nh_present (binary)
# [22]    hb_co_d (CO←H distance / 3.5)
# [23]    hb_co_cos_coh (acceptor angle)
# [24]    hb_co_present (binary)
# Solvent accessibility (NEW — replaces burial bundle):
# [25]    rSASA (relative SASA, 0-1)
# [26]    HSE_up (half-sphere exposure, upper / 50)
# [27]    HSE_down (half-sphere exposure, lower / 50)
# Electric field (NEW — Buckingham):
# [28]    Efield_NH (along N-H bond / 0.1)
# [29]    Efield_CaHa (along CA-HA bond / 0.1)
# Secondary structure (NEW — from DSSP):
# [30]    dssp_3state (helix=+1, sheet=-1, coil=0)
# Deuteration (NEW — sample-level broadcast):
# [31]    frac_deut_amide (0-1)
# [32]    frac_deut_alpha (0-1)
# [33]    frac_deut_sidechain (0-1)
NUM_GEO_FEATURES = 34

# ---------------------------------------------------------------------------
# Per-target-atom local environment features (D8 — 15 features per nucleus)
# Stored as [R, 6, NUM_TARGET_ENV_FEATURES] where axis 1 = nucleus index
# (H=0, HA=1, N=2, CA=3, CB=4, C=5).
#
# Shell-resolved typed environment (12 features):
#   [0]  backbone_peptide_close   backbone N/CA/C/O within 0-4 A, /10
#   [1]  backbone_peptide_mid     backbone N/CA/C/O within 4-8 A, /30
#   [2]  hydrophobic_C_close      non-aromatic sidechain/CB carbon within 0-4 A, /10
#   [3]  hydrophobic_C_mid        non-aromatic sidechain/CB carbon within 4-8 A, /30
#   [4]  polar_O_close            sidechain O within 0-4 A, /5
#   [5]  polar_O_mid              sidechain O within 4-8 A, /15
#   [6]  polar_N_close            sidechain N within 0-4 A, /5
#   [7]  polar_N_mid              sidechain N within 4-8 A, /15
#   [8]  sulfur_close             S atoms within 0-4 A, /3
#   [9]  sulfur_mid               S atoms within 4-8 A, /5
#   [10] aromatic_close           aromatic ring atoms within 0-4 A, /10
#   [11] aromatic_mid             aromatic ring atoms within 4-8 A, /30
#
# Aromatic proximity (2 features):
#   [12] nearest_aromatic_dist    dist to nearest ring centroid / 8.0 (1.0 if none)
#   [13] aromatic_cos_normal      cos(target->centroid, ring normal) (0.0 if none)
#
# Disulfide geometry (1 feature):
#   [14] nearest_SG_dist          dist to nearest disulfide SG / 8.0 (1.0 if none)
NUM_TARGET_ENV_FEATURES = 15

# ---------------------------------------------------------------------------
# Amino acid vocabulary (canonical 20 → index 0-19, variants map in)
# ---------------------------------------------------------------------------
_CANONICAL_AA: dict[str, int] = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}

# Non-standard / modified residues mapped to their closest canonical parent.
# Covers the common PTMs and protonation variants seen in real PDB / mmCIF
# inputs. New entries go here rather than touching _CANONICAL_AA so the
# "canonical 20" ordering stays stable.
_NONSTANDARD_AA: dict[str, int] = {
    # Histidine protonation states
    "HSE": 8, "HSD": 8, "HSP": 8, "HIE": 8, "HID": 8, "HIP": 8,
    # Glu / Asp / Cys protonation variants
    "GLH": 6, "ASH": 3, "CYX": 4,
    # Selenomethionine → methionine (very common in SAD/MAD crystallography)
    "MSE": 12,
    # Hydroxyproline → proline
    "HYP": 14,
    # Phospho-residues → parent AA
    "SEP": 15,  # phosphoserine
    "TPO": 16,  # phosphothreonine
    "PTR": 18,  # phosphotyrosine
    # Oxidised cysteines → cysteine
    "CSO": 4,   # S-hydroxycysteine
    "CSD": 4,   # 3-sulfinoalanine
    "OCS": 4,   # cysteinesulfonic acid
    "CME": 4,   # S,S-(2-hydroxyethyl)thiocysteine
    # Pyroglutamate → glutamine
    "PCA": 5,
    # Lysine modifications → lysine
    "KCX": 11,  # lysine carbamic acid
    "MLZ": 11,  # N-methyl-lysine
    "MLY": 11,  # N-dimethyl-lysine
    "M3L": 11,  # N-trimethyl-lysine
    "LLP": 11,  # lysine-pyridoxal 5'-phosphate
    # Other frequent PTMs
    "FME": 12,  # N-formylmethionine
    "ALY": 11,  # N(6)-acetyllysine
    "SEC": 4,   # selenocysteine → cysteine (closest standard parent)
    "PYL": 11,  # pyrrolysine → lysine (closest standard parent)
}

AA_THREE_TO_IDX: dict[str, int] = {**_CANONICAL_AA, **_NONSTANDARD_AA}
NUM_AA_TYPES = 20
UNK_AA_IDX = NUM_AA_TYPES  # 20 = unknown; embeddings sized NUM_AA_TYPES + 1


def normalize_residue_name(resname: str) -> tuple[int, bool, bool]:
    """Map a 3-letter residue name to (aa_idx, is_nonstandard, is_unknown).

    is_nonstandard is True when the residue was in _NONSTANDARD_AA (e.g. MSE,
    HYP, SEP), so callers can log that a substitution happened. is_unknown is
    True when the name is not in any map and UNK_AA_IDX was returned.
    """
    name = resname.strip().upper()
    if name in _CANONICAL_AA:
        return _CANONICAL_AA[name], False, False
    if name in _NONSTANDARD_AA:
        return _NONSTANDARD_AA[name], True, False
    return UNK_AA_IDX, False, True

# ---------------------------------------------------------------------------
# Element vocabulary
# ---------------------------------------------------------------------------
ELEMENT_TO_IDX: dict[str, int] = {"C": 0, "N": 1, "O": 2, "S": 3, "H": 4}
NUM_ELEMENT_TYPES = 5
UNK_ELEMENT_IDX = NUM_ELEMENT_TYPES  # 5 = unknown; embeddings sized + 1

# ---------------------------------------------------------------------------
# Atom role categories (backbone + sidechain classification)
# ---------------------------------------------------------------------------
ATOM_ROLE_MAP: dict[str, int] = {
    "N": 0, "CA": 1, "C": 2, "O": 3,  # backbone heavy
    "CB": 4,                             # beta carbon
    "H": 5, "HA": 6, "HA2": 6, "HA3": 6,  # backbone H
    "OXT": 3,                            # C-terminal O
}
ROLE_SIDECHAIN_HEAVY = 7
ROLE_SIDECHAIN_H = 8
ROLE_UNKNOWN = 9
NUM_ATOM_ROLES = 10

# ---------------------------------------------------------------------------
# Target atom names per nucleus (used to identify prediction targets in PDB)
# ---------------------------------------------------------------------------
TARGET_ATOM_NAMES: dict[str, tuple[str, ...]] = {
    "H": ("H", "HN", "H1"),     # amide H (varies by convention)
    "HA": ("HA", "HA2", "HA3"),  # alpha H (HA2/HA3 for GLY)
    "N": ("N",),
    "CA": ("CA",),
    "CB": ("CB",),
    "C": ("C",),
}


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------
def rbf_expansion(
    distances: torch.Tensor,
    num_rbf: int = 20,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """Expand distances into Gaussian radial basis functions.

    Args:
        distances: [E] pairwise distances.
        num_rbf: Number of Gaussian centers, evenly spaced in [0, cutoff].
        cutoff: Upper bound for RBF centers.

    Returns:
        [E, num_rbf] RBF-expanded features.
    """
    centers = torch.linspace(0.0, cutoff, num_rbf, device=distances.device)
    width = (cutoff / num_rbf) if num_rbf > 1 else 1.0
    gamma = 1.0 / (width ** 2)
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cosine cutoff envelope, 1 at d=0 → 0 at d=cutoff."""
    return 0.5 * (torch.cos(distances * (3.14159265 / cutoff)) + 1.0) * (
        distances < cutoff
    ).float()


def get_atom_role(atom_name: str, is_hydrogen: bool) -> int:
    """Map PDB atom name to role index."""
    name = atom_name.strip()
    if name in ATOM_ROLE_MAP:
        return ATOM_ROLE_MAP[name]
    if is_hydrogen:
        return ROLE_SIDECHAIN_H
    return ROLE_SIDECHAIN_HEAVY
