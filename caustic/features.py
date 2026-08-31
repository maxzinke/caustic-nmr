"""Feature encoding utilities: vocabularies, RBF expansion, atom roles."""
from __future__ import annotations

import logging
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Caustic Rescue Plan A6: cache version tag
# ---------------------------------------------------------------------------
# NUM_GEO_FEATURES has grown 34 → 40 → 46 → 52 across D-runs. Three physics-
# feature bugs (Efield / ring-currents / rSASA) were also fixed mid-project
# (2026-04-21/22). All historical D-checkpoints are evaluated against the
# current cache state, not what they trained on. CACHE_VERSION is the user-
# visible tag of the current feature pipeline; cache snapshots use this name.
#
# The verify function below is a soft check — it logs a WARNING when the
# graph_cache_dir's CACHE_VERSION file disagrees with this constant. Plumb
# into training/inference in a follow-up; the constant alone is the A6
# deliverable.

CACHE_VERSION = "d50_v2_fixed_nc_contacts"  # d47 ring/HB + d50 capped hetero context + fixed N/C contacts

_logger_a6 = logging.getLogger(__name__)


def verify_cache_version(graph_cache_dir) -> str | None:
    """Read ``<graph_cache_dir>/CACHE_VERSION`` and warn on drift.

    Returns the cache tag found on disk (or ``None`` if the file is absent —
    legacy caches predate the snapshot). Never raises.
    """
    p = Path(graph_cache_dir).expanduser() / "CACHE_VERSION"
    if not p.exists():
        return None
    try:
        on_disk = p.read_text().strip()
    except OSError as e:
        _logger_a6.warning("Could not read %s: %s", p, e)
        return None
    if on_disk != CACHE_VERSION:
        _logger_a6.warning(
            "Graph cache at %s has CACHE_VERSION=%r but features.py expects %r. "
            "Re-run scripts/snapshot_cache.py or rebuild graphs.",
            graph_cache_dir, on_disk, CACHE_VERSION,
        )
    return on_disk

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
# Continuous chi1 torsion (NEW — gamma-gauche effect on N shifts):
# [34]    sin_chi1 (0.0 when chi1 undefined: GLY/ALA/PRO)
# [35]    cos_chi1 (0.0 when chi1 undefined: GLY/ALA/PRO)
# Nearest aromatic neighbor sidechain orientation (NEW — ring-current control):
# [36]    arom_nb_sin_chi1 (0.0 if no aromatic within 8 A)
# [37]    arom_nb_cos_chi1 (0.0 if no aromatic within 8 A)
# [38]    arom_nb_sin_chi2 (0.0 if no aromatic within 8 A)
# [39]    arom_nb_cos_chi2 (0.0 if no aromatic within 8 A)
# D32 (2026-04-23): six new slots targeting specific pain-points surfaced
# by the 2026-04-23 structural error audit (PRO N / CYS SS CB / terminal C').
# Chain-aware terminal flags (NOT "first/last residue by index" — must
# respect chain boundaries / OXT presence):
# [40]    is_n_terminal (1.0 if first polymer residue of the extracted chain)
# [41]    is_c_terminal (1.0 if last polymer residue AND has an OXT atom)
# Explicit disulfide geometry for CYS-in-SS-bond residues (zero for the other 19 AAs
# and for reduced CYS):
# [42]    ss_dist          SG-SG distance (A) / 3.0  (partner's SG sits ~2.05 A from own SG)
# [43]    ss_cbscb_angle   cos(CB-SG-SG'-CB') dihedral  (chi_SS)
# [44]    ss_cb_sg_sg_ang  cos(CB-SG-SG') bond angle
# [45]    ss_partner_sep   |i - j| / 50  (sequence separation to disulfide partner)
# D42 (2026-04-26): H-bond partner directions in lab frame.
# Each is a unit vector (zero if no H-bond detected). Used by the head's
# H-bond frame projection — exposes partner-orientation info the residue
# frame doesn't capture. Hits sheet CB / amide H/N / terminal C'.
# [46-48] hb_nh_partner_dir  (this residue's amide H or N → partner C=O direction)
# [49-51] hb_co_partner_dir  (this residue's carbonyl O or C → partner H-N direction)
NUM_GEO_FEATURES = 52

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
# Optional chemistry feature tensors (d49)
# ---------------------------------------------------------------------------
# These are deliberately separate from residue_geometry and target_environment
# so old checkpoints with use_geometry/use_target_env=True can still load with
# their historical projection widths. New runs opt in via ModelConfig flags.
#
# residue_chemistry [R, NUM_RESIDUE_CHEM_FEATURES]
#   [0]  is_cys
#   [1]  cys_reduced_free
#   [2]  cys_disulfide
#   [3]  cys_metal_bound
#   [4]  cys_heme_thioether
#   [5]  cys_oxidized_or_ptm
#   [6]  cys_nterm_candidate
#   [7]  residue_metal_bound
#   [8]  metal_coord_count / 4
#   [9]  nearest_metal_dist / 8 (1.0 if none)
#   [10] hetero_heavy_count_4A / 8
#   [11] nearest_hetero_dist / 8 (1.0 if none)
#   [12] his_metal_bound
#   [13] charged_sidechain_near_backbone_count / 6
#   [14] covalent_ligand_connection
#   [15] residue_nonstandard_or_modified
NUM_RESIDUE_CHEM_FEATURES = 16

# target_chemistry [R, 6, NUM_TARGET_CHEM_FEATURES]
#   [0]  nearest_metal_dist / 8 (1.0 if none)
#   [1]  metal_count_4A / 4
#   [2]  hetero_heavy_count_4A / 8
#   [3]  hetero_heavy_count_4_8A / 16
#   [4]  nearest_hetero_dist / 8 (1.0 if none)
#   [5]  charged_contact_count_4A / 6
#   [6]  nearest_charged_atom_dist / 8 (1.0 if none)
#   [7]  polar_sidechain_contact_count_3p6A / 6
#   [8]  signed_ring_current_sum clipped to +/-3 ppm, /3
#   [9]  axial_ring_current_sum clipped to +/-3 ppm, /3
#   [10] equatorial_ring_current_sum clipped to +/-3 ppm, /3
#   [11] max_abs_ring_current clipped to 3 ppm, /3
#   [12] aromatic_ring_count_8p5A / 8
#   [13] nearest_ring_dist / 8.5 (1.0 if none)
NUM_TARGET_CHEM_FEATURES = 14

# target_nc_chemistry [R, 6, NUM_TARGET_NC_CHEM_FEATURES]
#   [0] selective target-NH acceptor count / 2
#   [1] nearest target-NH acceptor distance / cutoff (1.0 if none)
#   [2] target-NH sidechain/hetero acceptor count / 2
#   [3] target-NH long-range acceptor count / 2
#   [4] selective target-CO donor/cation contact count / 2
#   [5] nearest target-CO donor/cation contact distance / 4 (1.0 if none)
#   [6] target-CO sidechain/hetero contact count / 2
#   [7] target-CO cationic sidechain contact count / 2
NUM_TARGET_NC_CHEM_FEATURES = 8
TARGET_NC_CHEMISTRY_VERSION = 2


def _version_int(value) -> int | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and int(value.numel()) > 0:
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        return int(value)
    except Exception:
        return None


def target_nc_chemistry_is_current(data) -> bool:
    """Return True when the cached direct N/C chemistry tensor is current."""
    tensor = getattr(data, "target_nc_chemistry", None)
    if tensor is None or not hasattr(tensor, "shape"):
        return False
    if len(tensor.shape) < 1 or int(tensor.shape[-1]) != NUM_TARGET_NC_CHEM_FEATURES:
        return False
    return (
        _version_int(getattr(data, "target_nc_chemistry_version", None))
        == TARGET_NC_CHEMISTRY_VERSION
    )


def stamp_target_nc_chemistry_version(data) -> None:
    """Stamp a PyG graph after recomputing target_nc_chemistry."""
    data.target_nc_chemistry_version = torch.tensor(
        [TARGET_NC_CHEMISTRY_VERSION], dtype=torch.long,
    )

# edge_chemistry [E, NUM_EDGE_CHEM_FEATURES]
#   [0] polar donor/acceptor proximity
#   [1] backbone-backbone H-bond
#   [2] sidechain-involved polar contact
#   [3] charged sidechain contact
#   [4] metal coordination/contact
#   [5] long-range typed contact (|seq sep| >= 5)
#   [6] sulfur/thiol contact
#   [7] hetero/ligand contact
NUM_EDGE_CHEM_FEATURES = 8

# ---------------------------------------------------------------------------
# Solution-chemistry features (D15 — 12 features per residue)
# Stored as [R, NUM_SOLCHEM_FEATURES]. Combines sample-level metadata with
# local titratable-neighbor context so the model can learn how protonation
# state of nearby ASP/GLU/ARG/LYS/HIS/CYS/TYR sidechains perturbs the
# backbone shifts at a given residue.
#
# Sample conditions (4 features, broadcast across residues):
#   [0]  pH normalized  (pH - 6.5) / 2.0
#   [1]  ionic strength normalized  (clamp(I, 0, 5) - 0.1) / 0.5
#   [2]  has_ph (binary, 1 if pH was recorded for this entry)
#   [3]  has_ionic_strength (binary, 1 if ionic strength was recorded)
#
# Titratable-neighbor counts within 8 A of this residue's CA (4 features):
#   [4]  count of negative sidechains (ASP, GLU) / 5
#   [5]  count of positive sidechains (ARG, LYS) / 5
#   [6]  count of HIS sidechains / 3
#   [7]  count of thiol/hydroxyl sidechains (CYS, TYR) / 3
#
# Nearest-distance to each titratable group (4 features, 1.0 if none):
#   [8]  nearest negative CA / 20.0
#   [9]  nearest positive CA / 20.0
#   [10] nearest HIS CA / 20.0
#   [11] nearest thiol/hydroxyl CA / 20.0
NUM_SOLCHEM_FEATURES = 12

# AA indices for titratable residue groups — verified against _CANONICAL_AA.
TITRATABLE_NEG: frozenset[int] = frozenset({3, 6})     # ASP, GLU
TITRATABLE_POS: frozenset[int] = frozenset({1, 11})    # ARG, LYS
TITRATABLE_HIS: frozenset[int] = frozenset({8})        # HIS
TITRATABLE_THIOL: frozenset[int] = frozenset({4, 18})  # CYS, TYR

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
# RING (5) is a virtual element used for aromatic-ring centroid nodes; it is
# not produced by any real atom in the structure. UNK is now 6.
RING_ELEMENT_IDX = 5
NUM_ELEMENT_TYPES = 6
UNK_ELEMENT_IDX = NUM_ELEMENT_TYPES  # 6 = unknown; embeddings sized + 1

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
ROLE_RING = 10  # aromatic-ring centroid virtual node
ROLE_LIGAND_HEAVY = 11
ROLE_METAL = 12
ROLE_WATER = 13
NUM_ATOM_ROLES = 14

# ---------------------------------------------------------------------------
# Aromatic ring atom-name conventions per AA. Each entry is a list of rings;
# TRP has two (6-member benzene + 5-member pyrrole). When all atoms in a ring
# are present, graph.py emits a virtual node at the centroid with element
# RING_ELEMENT_IDX, atom_role ROLE_RING, and a node-level normal vector.
# ---------------------------------------------------------------------------
AROMATIC_RINGS: dict[str, list[tuple[str, ...]]] = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TRP": [
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),  # 6-member benzene
        ("CG", "CD1", "NE1", "CE2", "CD2"),          # 5-member pyrrole
    ],
    "HIS": [("CG", "ND1", "CE1", "NE2", "CD2")],
}

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
