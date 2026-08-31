"""POTENCI random coil chemical shift prediction.

Reimplemented from Nielsen & Mulder (2018) — prediction of temperature,
neighbor and pH-corrected random coil chemical shifts for intrinsically
disordered proteins.

Reference: https://github.com/protein-nmr/POTENCI
License: MIT

Predicts random coil shifts for backbone atoms (H, HA, N, CA, CB, C)
as f(sequence, pH, temperature, ionic_strength), accounting for:
  - Sequence-neighbor effects (±2 residues)
  - Combination deviations (pentapeptide group patterns)
  - Temperature dependence (linear ppb/K)
  - pH-dependent protonation (Asp, Glu, His, Cys, Tyr, Lys, Arg + termini)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

logger = logging.getLogger(__name__)

# Nucleus ordering matching our shift predictor convention
POTENCI_NUCLEI = ("H", "HA", "N", "CA", "CB", "C")

# Map from 3-letter to 1-letter AA codes
_AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

_AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# Physical constants for pKa calculation
# ---------------------------------------------------------------------------
_R = 8.314472       # J/(mol·K)
_DIELECTRIC = 79.0  # water dielectric constant
_A_PARAM = 5.0      # distance parameter a
_B_PARAM = 7.5      # distance parameter b
_PKA_CUTOFF = 2     # submatrix half-width for pKa iteration
_PKA_NCYCLES = 5    # iterative pKa refinement cycles

# Intrinsic pKa values
_PK0 = {
    "n": 8.23, "D": 3.86, "E": 4.34, "H": 6.45,
    "C": 8.49, "K": 10.34, "R": 13.9, "Y": 9.76, "c": 3.55,
}

# ---------------------------------------------------------------------------
# Data tables — from POTENCI v1.3 (Nielsen & Mulder 2018)
# ---------------------------------------------------------------------------

# Central random coil shifts: {AA_1letter: {atom: ppm}}
_CENT_SHIFTS: dict[str, dict[str, float]] = {
    "A": {"C": 177.44069, "CA": 52.53002, "CB": 19.21113, "N": 125.40155, "H": 8.20964, "HA": 4.25629, "HB": 1.31544},
    "C": {"C": 174.33917, "CA": 58.48976, "CB": 28.06269, "N": 120.71212, "H": 8.29429, "HA": 4.44261, "HB": 2.85425},
    "D": {"C": 176.02114, "CA": 54.23920, "CB": 41.18408, "N": 121.75726, "H": 8.28460, "HA": 4.54836, "HB": 2.60054},
    "E": {"C": 176.19215, "CA": 56.50755, "CB": 30.30204, "N": 122.31578, "H": 8.35949, "HA": 4.22124, "HB": 1.92383},
    "F": {"C": 175.42280, "CA": 57.64849, "CB": 39.55984, "N": 121.30500, "H": 8.10906, "HA": 4.57507, "HB": 3.00036},
    "G": {"C": 173.83294, "CA": 45.23929, "CB": None,     "N": 110.09074, "H": 8.32746, "HA": 3.91016, "HB": None},
    "H": {"C": 175.00142, "CA": 56.20256, "CB": 30.60335, "N": 120.69141, "H": 8.27133, "HA": 4.55872, "HB": 3.03080},
    "I": {"C": 175.88231, "CA": 61.04925, "CB": 38.68742, "N": 122.37586, "H": 8.06407, "HA": 4.10574, "HB": 1.78617},
    "K": {"C": 176.22644, "CA": 56.29413, "CB": 33.02478, "N": 122.71282, "H": 8.24902, "HA": 4.25873, "HB": 1.71982},
    "L": {"C": 177.06101, "CA": 55.17464, "CB": 42.29215, "N": 123.48611, "H": 8.14330, "HA": 4.28545, "HB": 1.54067},
    "M": {"C": 175.90708, "CA": 55.50643, "CB": 32.83806, "N": 121.54592, "H": 8.24848, "HA": 4.41483, "HB": 1.97585},
    "N": {"C": 174.94152, "CA": 53.22822, "CB": 38.87465, "N": 119.92746, "H": 8.37189, "HA": 4.64308, "HB": 2.72756},
    "P": {"C": 176.67709, "CA": 63.05232, "CB": 32.03750, "N": 137.40612, "H": None,    "HA": 4.36183, "HB": 2.03318},
    "Q": {"C": 175.63494, "CA": 55.79861, "CB": 29.44174, "N": 121.49225, "H": 8.30042, "HA": 4.28006, "HB": 1.97653},
    "R": {"C": 175.92194, "CA": 56.06785, "CB": 30.81298, "N": 122.40365, "H": 8.26453, "HA": 4.28372, "HB": 1.73437},
    "S": {"C": 174.31005, "CA": 58.36048, "CB": 63.82367, "N": 117.11419, "H": 8.25730, "HA": 4.40101, "HB": 3.80956},
    "T": {"C": 174.27772, "CA": 61.86928, "CB": 69.80612, "N": 115.48126, "H": 8.11378, "HA": 4.28923, "HB": 4.15465},
    "V": {"C": 175.80621, "CA": 62.20156, "CB": 32.77934, "N": 121.71912, "H": 8.06572, "HA": 4.05841, "HB": 1.99302},
    "W": {"C": 175.92744, "CA": 57.23836, "CB": 29.56502, "N": 122.10991, "H": 7.97816, "HA": 4.61061, "HB": 3.18540},
    "Y": {"C": 175.49651, "CA": 57.82427, "CB": 38.76184, "N": 121.43652, "H": 8.05749, "HA": 4.51123, "HB": 2.91782},
}

# Neighbor corrections: {AA_1letter: {atom: [coeff_+2, coeff_+1, coeff_-1, coeff_-2]}}
_NEI_CORRS: dict[str, dict[str, list[float]]] = {}

_NEI_RAW = """C A  0.06131 -0.04544  0.14646  0.01305
C C  0.04502  0.12592 -0.03407 -0.02654
C D  0.08180 -0.08589  0.22948  0.10934
C E  0.05388  0.22264  0.06962  0.01929
C F -0.06286 -0.22396 -0.34442  0.00950
C G  0.12772  0.72041  0.16048  0.01324
C H -0.00628 -0.03355  0.13309 -0.03906
C I -0.11709  0.06591 -0.06361 -0.03628
C K  0.03368  0.15830  0.04518 -0.01576
C L -0.03877  0.11608  0.02535  0.01976
C M  0.04611  0.25233 -0.00747 -0.01624
C N  0.07068 -0.06118  0.10077  0.05547
C P -0.36018 -1.90872  0.16158 -0.05286
C Q  0.10861  0.19878  0.01596 -0.01757
C R  0.01933  0.13237  0.03606 -0.02468
C S  0.09888  0.28691  0.07601  0.01379
C T  0.05658  0.41659 -0.01103 -0.00114
C V -0.11591  0.09565 -0.03355 -0.03368
C W -0.01954 -0.19134 -0.37965  0.01582
C Y -0.08380 -0.24519 -0.32700 -0.00577
CA A  0.03588  0.03480 -0.00468 -0.00920
CA C  0.02749  0.15742  0.14376  0.03681
CA D -0.00751  0.12494  0.17354  0.14157
CA E  0.00985  0.13936  0.03289 -0.00702
CA F  0.01122  0.03732 -0.19586 -0.00377
CA G -0.00885  0.23403 -0.03184 -0.01144
CA H -0.02102  0.04621  0.03122 -0.02826
CA I -0.00656  0.05965 -0.10588 -0.04372
CA K  0.01817  0.11216 -0.00341 -0.02950
CA L  0.04507  0.07829 -0.03526  0.00858
CA M  0.07553  0.18840  0.04987 -0.01749
CA N -0.00649  0.11842  0.18729  0.06401
CA P -0.27536 -2.02189  0.01327 -0.08732
CA Q  0.06365  0.15281  0.04575 -0.01356
CA R  0.04338  0.11783  0.00345 -0.02873
CA S  0.02867  0.07846  0.09443  0.02061
CA T -0.01625  0.10626  0.03880 -0.00126
CA V -0.04935  0.04248 -0.10195 -0.03778
CA W  0.00434  0.16188 -0.08742  0.03983
CA Y  0.02782  0.02846 -0.24750  0.00759
CB A -0.00953  0.05704 -0.04838  0.00755
CB C -0.00164  0.00760 -0.03293 -0.05613
CB D  0.02064  0.09849 -0.08746 -0.06691
CB E  0.01283  0.05404 -0.01342  0.02238
CB F  0.01028  0.03363  0.18112  0.01493
CB G -0.02758  0.04383  0.06071 -0.02639
CB H -0.01760 -0.02367  0.00343  0.00415
CB I  0.02783  0.01052  0.00641  0.05090
CB K  0.00350  0.02852 -0.00408  0.01218
CB L  0.01223 -0.02940 -0.07268  0.00884
CB M -0.02925 -0.03912 -0.06587  0.03490
CB N -0.02242  0.03403 -0.09759 -0.08018
CB P  0.08431 -0.35696 -0.04680  0.05192
CB Q -0.01649 -0.01016 -0.03663  0.01723
CB R -0.01887  0.00618 -0.00385  0.02884
CB S -0.00921  0.07096 -0.06338 -0.03707
CB T  0.02601  0.04904 -0.01728  0.00781
CB V  0.03068  0.06325  0.01928  0.05011
CB W -0.07651 -0.11334  0.13806 -0.03339
CB Y  0.00082  0.01466  0.18107 -0.01181
N A  0.09963 -0.00873 -2.31666 -0.14051
N C  0.11905 -0.01296  1.15573  0.01820
N D  0.11783 -0.11817 -1.16322 -0.37601
N E  0.10825 -0.00605 -0.41856  0.01187
N F -0.12280 -0.27542  0.34635  0.09102
N G  0.10365 -0.05667 -1.50346 -0.00146
N H -0.04145 -0.26494  0.26356  0.18198
N I -0.09249  0.12136  2.75071  0.40643
N K -0.02472  0.07224 -0.07057  0.12261
N L  0.01542 -0.12800 -0.85172 -0.15460
N M -0.11266 -0.27311 -0.33192  0.09384
N N -0.00295 -0.20562 -1.00652 -0.30971
N P  0.03252  1.35296 -1.17173  0.06026
N Q  0.00900 -0.09950 -0.07389  0.08415
N R -0.07819  0.00802 -0.04821  0.08524
N S  0.12057  0.02242  0.48924 -0.25423
N T  0.04631  0.09935  1.02269  0.20228
N V -0.03610  0.21959  2.42228  0.39686
N W -0.15643 -0.19285  0.05515 -0.53172
N Y -0.10497 -0.25228  0.46023  0.01399
H A  0.01337 -0.00605 -0.04371 -0.02485
H C  0.01324  0.05107  0.12857  0.00610
H D  0.02859  0.02436 -0.06510  0.02085
H E  0.02737  0.01790  0.03740  0.01969
H F -0.02633 -0.08287 -0.11364 -0.03603
H G  0.02753  0.05640 -0.10477  0.06876
H H -0.00124 -0.02861  0.04126  0.10004
H I -0.02258 -0.00929  0.07962  0.01880
H K -0.00512 -0.00744  0.04443  0.03434
H L -0.01088 -0.01230 -0.03640 -0.03719
H M -0.01961 -0.00749 -0.00097  0.02041
H N  0.01134  0.02121 -0.01837 -0.00629
H P -0.01246  0.02956  0.13007 -0.00810
H Q  0.00783  0.00751  0.05643  0.02413
H R -0.00734  0.00546  0.07003  0.04051
H S  0.02133  0.03964  0.04978 -0.03749
H T  0.00976  0.06072  0.03531  0.01657
H V -0.01267  0.00994  0.09630  0.03420
H W -0.02348 -0.09617 -0.24207 -0.18741
H Y -0.01881 -0.07345 -0.14345 -0.06721
HA A  0.00350 -0.02371 -0.00654  0.00652
HA C  0.00660  0.01073  0.01921  0.00919
HA D  0.01717 -0.00854 -0.00802 -0.00597
HA E  0.01090 -0.01091  0.00472  0.00790
HA F -0.02271 -0.06316 -0.03057 -0.02350
HA G  0.02155 -0.00151  0.02477  0.01526
HA H -0.01132 -0.05617 -0.01514  0.01264
HA I  0.00459  0.00571  0.02984  0.00416
HA K  0.00492 -0.01788  0.00555  0.01259
HA L -0.00599 -0.01558  0.00358  0.00167
HA M  0.00100 -0.02037  0.00678  0.00930
HA N  0.00651 -0.01499 -0.00361  0.00203
HA P  0.01542  0.28350 -0.01496  0.00796
HA Q  0.00711 -0.02142  0.00734  0.00971
HA R -0.00472 -0.01414  0.00966  0.01180
HA S  0.01572  0.02791  0.03762  0.00133
HA T  0.01714  0.06590  0.03085  0.00143
HA V  0.00777  0.01505  0.02525  0.00659
HA W -0.06818 -0.08412 -0.09386 -0.06072
HA Y -0.02701 -0.05585 -0.03243 -0.02987
HB A  0.01473  0.01843  0.01428  0.00451
HB C  0.01180  0.03340  0.03081  0.00169
HB D  0.01786  0.01626  0.02221  0.01030
HB E  0.01796  0.01820  0.00835 -0.00045
HB F -0.04867 -0.09154 -0.04858 -0.00164
HB G  0.01718  0.03852  0.01043  0.00051
HB H -0.00817 -0.04557 -0.00820  0.00855
HB I  0.00446  0.00111  0.00049 -0.00283
HB K  0.01570  0.01156  0.00771  0.00646
HB L  0.00700  0.01236  0.00880  0.00150
HB M  0.01607  0.02294  0.01385 -0.00038
HB N  0.01893  0.01561  0.02760  0.01215
HB P -0.01199 -0.02752  0.00891 -0.00033
HB Q  0.01636  0.01861  0.01177 -0.00099
HB R  0.01324  0.01526  0.01082  0.00378
HB S  0.01859  0.03487  0.02890 -0.00477
HB T  0.01624  0.04073  0.01936 -0.00348
HB V  0.00380  0.00271 -0.00144 -0.00315
HB W -0.09045 -0.06895 -0.10934 -0.01948
HB Y -0.05069 -0.06698 -0.05666 -0.01193"""

# Terminal corrections: {atom: {"n": corr, "c": corr}}
_TERM_CORRS: dict[str, dict[str, float]] = {
    "C":  {"n": -0.15238, "c": -0.90166},
    "CB": {"n":  0.12064, "c":  0.06854},
    "CA": {"n": -0.04616, "c": -0.06680},
    "N":  {"n":  0.347176, "c": 0.619141},
    "H":  {"n":  0.156786, "c": 0.023189},
    "HB": {"n":  0.0052692, "c": 0.0310875},
    "HA": {"n":  0.048624, "c": 0.042019},
}

# Temperature coefficients: ppb/K per AA per atom (multiply by (T-298)/1000)
_TEMP_COEFFS: dict[str, dict[str, float]] = {
    "CA": {"A": -2.2, "C": -0.9, "D": 2.8, "E": 0.9, "F": -4.7, "G": 3.3, "H": 7.8, "I": -2.0, "K": -0.8, "L": 1.7, "M": 4.1, "N": 2.8, "P": 1.1, "Q": 2.3, "R": -1.4, "S": -1.7, "T": 0.0, "V": -2.8, "W": -2.7, "Y": -5.0},
    "CB": {"A": 4.7, "C": 1.3, "D": 6.5, "E": 4.6, "F": 2.4, "G": 0.0, "H": 15.5, "I": 4.6, "K": 2.4, "L": 4.9, "M": 9.4, "N": 5.1, "P": -0.2, "Q": 3.6, "R": 3.5, "S": 4.4, "T": 2.2, "V": 2.5, "W": 3.1, "Y": 2.9},
    "C":  {"A": -7.1, "C": -2.6, "D": -4.8, "E": -4.9, "F": -6.9, "G": -3.2, "H": 3.1, "I": -8.7, "K": -7.1, "L": -8.2, "M": -8.2, "N": -6.1, "P": -4.0, "Q": -5.7, "R": -6.9, "S": -4.7, "T": -5.2, "V": -8.1, "W": -7.9, "Y": -7.7},
    "N":  {"A": -5.3, "C": -8.2, "D": -3.9, "E": -3.7, "F": -11.2, "G": -6.2, "H": 3.3, "I": -12.7, "K": -7.6, "L": -2.9, "M": -6.2, "N": -3.3, "P": 0.0, "Q": -6.5, "R": -5.3, "S": -3.8, "T": -6.7, "V": -14.2, "W": -10.1, "Y": -12.0},
    "H":  {"A": -9.0, "C": -7.0, "D": -6.2, "E": -6.5, "F": -7.5, "G": -9.1, "H": -7.8, "I": -7.8, "K": -7.5, "L": -7.5, "M": -7.1, "N": -7.0, "P": 0.0, "Q": -7.2, "R": -7.1, "S": -7.6, "T": -7.3, "V": -7.6, "W": -7.8, "Y": -7.7},
    "HA": {"A": 0.7, "C": 0.0, "D": -0.1, "E": 0.3, "F": 0.4, "G": 0.0, "H": 0.4, "I": 0.4, "K": 0.4, "L": 0.1, "M": -0.5, "N": -2.9, "P": 0.0, "Q": 0.3, "R": 0.4, "S": 0.1, "T": 0.0, "V": 0.5, "W": 0.4, "Y": 0.5},
}

# Combination deviations: {atom: {segment_pattern: ((nei_pos, cent_group, nei_group), value)}}
_COMB_CORRS: dict[str, dict[str, tuple[tuple[int, str, str], float]]] = {}

_COMB_RAW = """C -1 G r xrGxx  0.2742
C -1 G - x-Gxx  0.0522
C -1 P P xPPxx -0.0822
C -1 P r xrPxx  0.2640
C -1 r P xPrxx -0.1027
C -1 + P xP+xx  0.0714
C -1 - - x--xx -0.0501
C -1 p r xrpxx  0.0582
C  1 G r xxGrx  0.0730
C  1 P a xxPax -0.0981
C  1 P + xxP+x -0.0577
C  1 P p xxPpx -0.0619
C  1 r r xxrrx -0.1858
C  1 r a xxrax -0.1888
C  1 r + xxr+x -0.1805
C  1 r - xxr-x -0.1756
C  1 r p xxrpx -0.1208
C  1 + P xx+Px -0.0533
C  1 - P xx-Px  0.1867
C  1 p P xxpPx  0.2321
C -2 G r rxGxx -0.1457
C -2 r p pxrxx  0.0555
C  2 P P xxPxP  0.1007
C  2 P - xxPx-  0.0634
C  2 r P xxrxP -0.1447
C  2 a r xxaxr -0.1488
C  2 a - xxax- -0.0093
C  2 + G xx+xG -0.0394
C  2 + P xx+xP  0.1016
C  2 + a xx+xa  0.0299
C  2 + + xx+x+  0.0427
C  2 - a xx-xa  0.0611
C  2 p P xxpxP -0.0753
CA -1 G P xPGxx -0.0641
CA -1 G r xrGxx  0.2107
CA -1 P P xPPxx -0.2042
CA -1 P p xpPxx  0.0444
CA -1 r G xGrxx  0.2030
CA -1 r + x+rxx -0.0811
CA -1 - P xP-xx  0.0744
CA -1 - - x--xx -0.0263
CA -1 p p xppxx -0.0094
CA  1 G P xxGPx  1.3044
CA  1 G - xxG-x -0.0632
CA  1 P G xxPGx  0.2642
CA  1 P P xxPPx  0.3025
CA  1 P r xxPrx  0.1455
CA  1 P - xxP-x  0.1188
CA  1 P p xxPpx  0.1201
CA  1 r P xxrPx -0.1958
CA  1 r - xxr-x -0.0931
CA  1 a P xxaPx -0.1428
CA  1 a - xxa-x -0.0262
CA  1 a p xxapx  0.0392
CA  1 + P xx+Px -0.1059
CA  1 + a xx+ax -0.0377
CA  1 + + xx++x -0.0595
CA  1 - P xx-Px -0.1156
CA  1 - + xx-+x  0.0316
CA  1 - p xx-px  0.0612
CA  1 p r xxprx -0.0511
CA -2 P - -xPxx -0.1028
CA -2 r r rxrxx  0.1933
CA -2 - G Gx-xx  0.0559
CA -2 - p px-xx  0.0391
CA -2 p a axpxx -0.0293
CA -2 p + +xpxx -0.0173
CA  2 G - xxGx-  0.0357
CA  2 + G xx+xG -0.0315
CA  2 - P xx-xP  0.0426
CA  2 - r xx-xr  0.0784
CA  2 - a xx-xa  0.1084
CA  2 - - xx-x-  0.0836
CA  2 p P xxpxP  0.0685
CA  2 p - xxpx- -0.0481
CB -1 P r xrPxx -0.2678
CB -1 P p xpPxx  0.0355
CB -1 r P xPrxx -0.1137
CB -1 a p xpaxx  0.0249
CB -1 + - x-+xx -0.0762
CB -1 - P xP-xx -0.0889
CB -1 - r xr-xx -0.0533
CB -1 - - x--xx  0.0496
CB -1 - p xp-xx -0.0148
CB -1 p P xPpxx  0.0119
CB -1 p r xrpxx -0.0673
CB  1 P G xxPGx -0.0522
CB  1 P P xxPPx -0.8458
CB  1 P r xxPrx -0.1573
CB  1 r r xxrrx  0.1634
CB  1 a G xxaGx -0.0393
CB  1 a r xxarx  0.0274
CB  1 a - xxa-x  0.0394
CB  1 a p xxapx  0.0149
CB  1 + G xx+Gx -0.0784
CB  1 + P xx+Px -0.1170
CB  1 - P xx-Px -0.0913
CB  1 - - xx--x  0.0284
CB  1 p P xxpPx  0.0880
CB  1 p p xxppx -0.0113
CB -2 P - -xPxx  0.0389
CB -2 P p pxPxx  0.0365
CB -2 r + +xrxx  0.0809
CB -2 a - -xaxx -0.0452
CB -2 + - -x+xx -0.0651
CB -2 - G Gx-xx -0.0883
CB -2 p G Gxpxx  0.0378
CB -2 p p pxpxx  0.0207
CB  2 r G xxrxG -0.0362
CB  2 r - xxrx- -0.0219
CB  2 a - xxax- -0.0298
CB  2 + p xx+xp  0.0189
CB  2 - - xx-x- -0.0525
N -1 G P xPGxx  0.2411
N -1 G + x+Gxx -0.1773
N -1 G - x-Gxx  0.1905
N -1 P P xPPxx -0.9177
N -1 P p xpPxx  0.2609
N -1 r G xGrxx  0.2417
N -1 r a xarxx -0.0139
N -1 r + x+rxx -0.4122
N -1 r p xprxx  0.1440
N -1 a G xGaxx -0.5177
N -1 a r xraxx  0.0890
N -1 a a xaaxx  0.1393
N -1 a p xpaxx -0.0825
N -1 + G xG+xx -0.4908
N -1 + a xa+xx  0.1709
N -1 + + x++xx  0.1868
N -1 + - x-+xx -0.0951
N -1 - P xP-xx -0.3027
N -1 - r xr-xx -0.1670
N -1 - + x+-xx -0.3501
N -1 - - x--xx  0.1266
N -1 p G xGpxx -0.1707
N -1 p - x-pxx  0.0011
N  1 G G xxGGx  0.2555
N  1 G P xxGPx -0.9725
N  1 G r xxGrx  0.0165
N  1 G p xxGpx  0.0703
N  1 r a xxrax -0.0237
N  1 a r xxarx -0.1816
N  1 a - xxa-x -0.1050
N  1 a p xxapx -0.1196
N  1 - r xx-rx -0.1762
N  1 - a xx-ax  0.0006
N  1 p P xxpPx  0.2797
N  1 p a xxpax  0.0938
N  1 p + xxp+x  0.1359
N -2 G r rxGxx -0.5140
N -2 G - -xGxx -0.0639
N -2 P P PxPxx -0.4215
N -2 r P Pxrxx -0.3696
N -2 r p pxrxx -0.1937
N -2 a - -xaxx -0.0351
N -2 a p pxaxx -0.1031
N -2 - G Gx-xx -0.2152
N -2 - P Px-xx -0.1375
N -2 - p px-xx -0.1081
N -2 p P Pxpxx -0.1489
N -2 p - -xpxx  0.0952
N  2 G - xxGx-  0.1160
N  2 r p xxrxp -0.1288
N  2 a P xxaxP  0.1632
N  2 + + xx+x+ -0.0106
N  2 + - xx+x-  0.0389
N  2 - a xx-xa -0.0815
N  2 p G xxpxG -0.0779
N  2 p p xxpxp -0.0683
H -1 G P xPGxx -0.0317
H -1 G r xrGxx  0.0549
H -1 G + x+Gxx -0.0192
H -1 G - x-Gxx  0.0138
H -1 r P xPrxx -0.0964
H -1 r - x-rxx -0.0245
H -1 a G xGaxx -0.0290
H -1 a a xaaxx  0.0063
H -1 + G xG+xx -0.0615
H -1 + r xr+xx -0.0480
H -1 + - x-+xx -0.0203
H -1 - + x+-xx -0.0232
H -1 p G xGpxx -0.0028
H -1 p P xPpxx -0.0121
H  1 G P xxGPx -0.1418
H  1 G r xxGrx  0.0236
H  1 G a xxGax  0.0173
H  1 a - xxa-x  0.0091
H  1 + P xx+Px -0.0422
H  1 + p xx+px  0.0191
H  1 - P xx-Px -0.0474
H  1 - a xx-ax  0.0102
H -2 G G GxGxx  0.0169
H -2 G r rxGxx -0.3503
H -2 a P Pxaxx  0.0216
H -2 a - -xaxx -0.0276
H -2 + - -x+xx -0.0260
H -2 - G Gx-xx  0.0273
H -2 - a ax-xx -0.0161
H -2 - - -x-xx -0.0285
H -2 p P Pxpxx -0.0101
H -2 p a axpxx -0.0157
H -2 p + +xpxx -0.0122
H -2 p p pxpxx  0.0107
H  2 G G xxGxG -0.0190
H  2 r G xxrxG  0.0472
H  2 r P xxrxP  0.0337
H  2 a + xxax+ -0.0159
H  2 + G xx+xG  0.0113
H  2 + r xx+xr -0.0307
H  2 - P xx-xP -0.0088
HA -1 P P xPPxx  0.0307
HA -1 P r xrPxx  0.0621
HA -1 r G xGrxx -0.0371
HA -1 r + x+rxx  0.0125
HA -1 r p xprxx -0.0199
HA -1 a G xGaxx  0.0073
HA -1 a a xaaxx  0.0044
HA -1 - G xG-xx  0.0116
HA -1 - r xr-xx  0.0228
HA -1 - p xp-xx  0.0074
HA  1 G G xxGGx  0.0175
HA  1 G - xxG-x  0.0107
HA  1 P a xxPax  0.0089
HA  1 - r xx-rx  0.0113
HA -2 G G GxGxx -0.0154
HA -2 P - -xPxx  0.0136
HA -2 r G Gxrxx -0.0159
HA -2 + + +x+xx -0.0137
HA -2 p - -xpxx -0.0068
HA -2 p p pxpxx  0.0046
HB -1 P r xrPxx  0.0460
HB -1 a - x-axx  0.0076
HB -1 + - x-+xx  0.0110
HB -1 - r xr-xx  0.0233
HB  1 a P xxaPx  0.0287
HB  1 + P xx+Px  0.0324
HB  1 + r xx+rx -0.0231
HB  1 p r xxprx  0.0077
HB  1 p + xxp+x -0.0074
HB -2 a P Pxaxx -0.0026
HB -2 a r rxaxx -0.0098
HB -2 - - -x-xx  0.0016
HB  2 P r xxPxr -0.0595
HB  2 P + xxPx+ -0.0145
HB  2 P - xxPx-  0.0107
HB  2 a + xxax+ -0.0015
HB  2 p r xxpxr  0.0262"""

# pH-dependent shift jumps for ionizable residues
# {residue: {atom: delta_shift}} where delta is protonated minus deprotonated
# Plus neighbor effects: {residue+"p": {...}, residue+"s": {...}} for preceding/succeeding
_PH_SHIFTS: dict[str, dict[str, float]] = {}

_PH_RAW = """D H -0.17 0.02 -0.03
D HA -0.17 0.01 -0.01
D HB -0.23
D CA 1.4 0.0 0.1
D CB 3.0
D C 1.1 -0.2 0.4
D N 1.5 0.3 0.1
E H 0.12 0.00 0.02
E HA -0.10 0.01 0.00
E HB -0.06
E CA 1.0 0.0 0.0
E CB 1.5
E C 0.6 0.1 0.1
E N 1.0 0.2 0.1
H H -0.2 -0.01 0.0
H HA -0.2 -0.01 -0.06
H HB -0.17
H CA 1.6 -0.1 0.1
H CB 2.4
H C 1.5 0.0 0.6
H N 1.8 0.3 0.5
C H 0.0
C HA -0.28 -0.01 -0.01
C HB -0.09
C CA 2.1 0.0 0.1
C CB 1.7
C C 1.9 -0.4 0.5
C N 3.6 0.4 0.6
Y H 0.0
Y HA -0.06
Y HB -0.08
Y CA 0.3
Y CB 0.1
Y C 0.4
Y N 0.6
K H 0.0
K HA -0.04
K HB -0.04
K CA 0.4
K CB 0.3
K C 0.5
K N 0.7
R H 0.0
R HA -0.07
R HB 0.05
R CA 0.2
R CB 0.9
R C 0.2
R N 0.4"""


# ---------------------------------------------------------------------------
# Table parsing (run once at module load)
# ---------------------------------------------------------------------------
def _parse_nei_corrs() -> dict[str, dict[str, list[float]]]:
    """Parse neighbor correction table into {AA: {atom: [+2, +1, -1, -2]}}."""
    dct: dict[str, dict[str, list[float]]] = {}
    for line in _NEI_RAW.strip().split("\n"):
        parts = line.split()
        atom, aa = parts[0], parts[1]
        coeffs = [float(x) for x in parts[2:6]]
        if aa not in dct:
            dct[aa] = {}
        dct[aa][atom] = coeffs
    # Terminal corrections
    for atom, term_dict in _TERM_CORRS.items():
        for term, val in term_dict.items():
            if term not in dct:
                dct[term] = {}
            if term == "n":
                dct["n"][atom] = [None, None, None, val]
            elif term == "c":
                dct["c"][atom] = [val, None, None, None]
    return dct


def _parse_comb_corrs() -> dict[str, dict[str, tuple[tuple[int, str, str], float]]]:
    """Parse combination deviation table."""
    dct: dict[str, dict[str, tuple[tuple[int, str, str], float]]] = {}
    for line in _COMB_RAW.strip().split("\n"):
        parts = line.split()
        atom = parts[0]
        nei_pos = int(parts[1])
        cent_group = parts[2]
        nei_group = parts[3]
        segment = parts[4]
        value = float(parts[5])
        if atom not in dct:
            dct[atom] = {}
        dct[atom][segment] = ((nei_pos, cent_group, nei_group), value)
    return dct


def _parse_ph_shifts() -> dict[str, dict[str, float]]:
    """Parse pH-dependent shift correction table.

    Returns {residue_key: {atom: delta}} where residue_key is:
      - single letter (e.g. "D") for the ionizable residue itself
      - letter + "p" (e.g. "Dp") for preceding neighbor effect
      - letter + "s" (e.g. "Ds") for succeeding neighbor effect
    """
    dct: dict[str, dict[str, float]] = {}
    for line in _PH_RAW.strip().split("\n"):
        parts = line.split()
        res = parts[0]
        atom = parts[1]
        delta = float(parts[2])
        if res not in dct:
            dct[res] = {}
        dct[res][atom] = delta
        # Neighbor effects (columns 4 and 5 if present)
        if len(parts) >= 5:
            prec_delta = float(parts[3])
            succ_delta = float(parts[4])
            key_p = res + "p"
            key_s = res + "s"
            if key_p not in dct:
                dct[key_p] = {}
            if key_s not in dct:
                dct[key_s] = {}
            dct[key_p][atom] = prec_delta
            dct[key_s][atom] = succ_delta
    return dct


# Initialize tables at module load
_NEI_CORRS = _parse_nei_corrs()
_COMB_CORRS = _parse_comb_corrs()
_PH_SHIFTS = _parse_ph_shifts()

# AA group classification for combination deviations
_AA_GROUPS = {
    "G": "G", "P": "P",
    "F": "r", "Y": "r", "W": "r",       # aromatic
    "L": "a", "I": "a", "V": "a",        # aliphatic
    "M": "a", "C": "a", "A": "a",
    "K": "+", "R": "+",                    # positive
    "D": "-", "E": "-",                    # negative
}
# Everything else (N, Q, S, T, H, n, c) maps to "p" (polar)


def _aa_to_group(aa: str) -> str:
    return _AA_GROUPS.get(aa, "p")


# ---------------------------------------------------------------------------
# pKa calculation (electrostatic model)
# ---------------------------------------------------------------------------
def _W(r: np.ndarray, ionic_strength: float = 0.1) -> np.ndarray:
    """Debye-Hückel charge-charge interaction energy."""
    k = np.sqrt(ionic_strength) / 3.08
    x = k * r / np.sqrt(6)
    return 332.286 * np.sqrt(6 / np.pi) * (
        1 - np.sqrt(np.pi) * x * np.exp(x ** 2) * erfc(x)
    ) / (_DIELECTRIC * r)


def _w2logp(x: np.ndarray, T: float = 293.15) -> np.ndarray:
    return x * 4181.2 / (_R * T * np.log(10))


def _henderson_hasselbalch(pH: float, pK: float, nH: float) -> float:
    return (10 ** (nH * (pK - pH))) / (1.0 + (10 ** (nH * (pK - pH))))


def _calc_pkas(seq: str, T: float = 293.15, ionic_strength: float = 0.1) -> dict[int, tuple[float, float, str]]:
    """Calculate per-site pKa values using iterative electrostatic model.

    Args:
        seq: Sequence with terminal markers, e.g. "n" + one_letter_seq + "c"
        T: Temperature in Kelvin.
        ionic_strength: In mol/L.

    Returns:
        {position: (pKa, hill_coefficient, residue_letter)}
    """
    pHs = np.arange(1.99, 10.01, 0.15)
    titratable = set(_PK0.keys())

    pos = np.array([i for i in range(len(seq)) if seq[i] in titratable])
    N = len(pos)
    if N == 0:
        return {}

    eye = np.diag(np.ones(N))
    sites = "".join(seq[i] for i in pos)
    neg = np.array([i for i in range(len(sites)) if sites[i] in "DEYc"])
    l_arr = np.array([np.abs(pos - pos[i]) for i in range(N)])
    d = _A_PARAM + np.sqrt(l_arr) * _B_PARAM

    tmp = _W(d, ionic_strength)
    tmp[eye == 1] = 0
    ww = _w2logp(tmp, T) / 2

    charges_empty = np.zeros(N)
    if len(neg):
        charges_empty[neg] = -1

    pK0s = [_PK0[c] for c in sites]
    nH0s = [0.9] * N

    titration = np.zeros((N, len(pHs)))
    smallN = min(2 * _PKA_CUTOFF + 1, N)
    alltuples = [
        [int(c) for c in np.binary_repr(i, smallN)]
        for i in range(2 ** smallN)
    ]
    gmatrix = [np.zeros((smallN, smallN)) for _ in pHs]

    for icycle in range(_PKA_NCYCLES):
        if icycle == 0:
            fractionhold = np.array([
                [_henderson_hasselbalch(pHs[p], pK0s[i], nH0s[i]) for i in range(N)]
                for p in range(len(pHs))
            ])
        else:
            fractionhold = titration.T

        for ires in range(1, N + 1):
            ileft = max(1, ires - _PKA_CUTOFF)
            iright = min(ileft + 2 * _PKA_CUTOFF, N)
            if iright == N:
                ileft = max(1, iright - 2 * _PKA_CUTOFF)

            resi = _PKA_CUTOFF + 1
            if ires < _PKA_CUTOFF + 1:
                resi = ires
            if ires > N - _PKA_CUTOFF:
                resi = min(N, 2 * _PKA_CUTOFF + 1) - (N - ires)

            for p in range(len(pHs)):
                fraction = fractionhold[p].copy()
                fraction[ileft - 1:iright] = 0
                charges = charges_empty + fraction
                ww0 = np.diag(np.dot(ww, charges) * 2)
                gmatrixfull = ww + ww0 + pHs[p] * eye - np.diag(pK0s)
                gmatrix[p] = gmatrixfull[ileft - 1:iright, ileft - 1:iright]

            E_all = np.array([
                sum(10 ** -(gmatrix[p] * np.outer(c, c)).sum() for c in alltuples)
                for p in range(len(pHs))
            ])
            E_sel = np.array([
                sum(10 ** -(gmatrix[p] * np.outer(c, c)).sum()
                    for c in alltuples if c[resi - 1] == 1)
                for p in range(len(pHs))
            ])
            titration[ires - 1] = E_sel / E_all

        try:
            sol = np.array([
                curve_fit(_henderson_hasselbalch, pHs, titration[p], [pK0s[p], nH0s[p]])[0]
                for p in range(N)
            ])
            pKs, nHs = sol.T
        except RuntimeError:
            # Fallback: use intrinsic pKa values
            pKs = np.array(pK0s)
            nHs = np.array(nH0s)

    result = {}
    for p_idx, i in enumerate(pos):
        result[i - 1] = (float(pKs[p_idx]), float(nHs[p_idx]), seq[i])
    return result


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def _pred_pent_shift(pent: str, atom: str) -> Optional[float]:
    """Predict random coil shift for one atom given a 5-residue window.

    Args:
        pent: 5-character string of AA single-letter codes (positions i-2..i+2).
              Terminals use 'n' and 'c'.
        atom: Atom name (C, CA, CB, N, H, HA, HB).

    Returns:
        Predicted shift in ppm, or None if not applicable.
    """
    center = pent[2]
    base = _CENT_SHIFTS.get(center, {}).get(atom)
    if base is None:
        return None

    # Neighbor corrections at positions +2, +1, -1, -2
    nei_positions = [2, 1, -1, -2]
    for idx, offset in enumerate(nei_positions):
        nei_aa = pent[2 + offset]
        if nei_aa in _NEI_CORRS and atom in _NEI_CORRS[nei_aa]:
            corr = _NEI_CORRS[nei_aa][atom][idx]
            if corr is not None:
                base += corr

    # Combination deviations
    if atom not in _COMB_CORRS:
        return base

    # Build group string for the pentapeptide
    grstr = "".join(_aa_to_group(pent[i]) for i in range(5))
    cent_gr = grstr[2]

    for segment, (key, combval) in _COMB_CORRS[atom].items():
        nei_pos, cent_group, nei_group = key
        if cent_group == cent_gr and grstr[2 + nei_pos] == nei_group:
            # pp combination only used when center is Ser or Thr
            if (cent_gr, nei_group) != ("p", "p") or pent[2] in "ST":
                base += combval

    return base


def _get_ph_corrections(
    seq: str,
    temperature: float,
    pH: float,
    ionic_strength: float,
) -> dict[str, dict[int, tuple[Optional[str], float]]]:
    """Compute pH-dependent shift corrections for all residues.

    Returns:
        {atom: {position: (residue, correction_ppm)}}
    """
    bb_atoms = ["C", "CA", "CB", "HA", "H", "N", "HB"]
    ion = max(0.0001, ionic_strength)

    pka_dct = _calc_pkas("n" + seq + "c", temperature, ion)

    outdct: dict[str, dict[int, list]] = {}
    for i, (pKa, nH, resi) in pka_dct.items():
        frac = _henderson_hasselbalch(pH, pKa, nH)
        frac7 = _henderson_hasselbalch(7.0, _PK0.get(resi, pKa), nH)

        if resi in "nc":
            continue

        for atn in bb_atoms:
            if atn not in outdct:
                outdct[atn] = {}

            if resi in _PH_SHIFTS and atn in _PH_SHIFTS[resi]:
                delta = _PH_SHIFTS[resi][atn]
                jumpdelta = frac * delta - frac7 * delta
                if i not in outdct[atn]:
                    outdct[atn][i] = [resi, jumpdelta]
                else:
                    outdct[atn][i][1] += jumpdelta

                # Neighbor effects
                key_p = resi + "p"
                key_s = resi + "s"
                if key_p in _PH_SHIFTS and atn in _PH_SHIFTS[key_p]:
                    for n in range(2):
                        ni = i + 2 * n - 1
                        nkey = [key_p, key_s][n]
                        if nkey in _PH_SHIFTS and atn in _PH_SHIFTS[nkey]:
                            ndelta = _PH_SHIFTS[nkey][atn]
                            njumpdelta = frac * ndelta - frac7 * ndelta
                            if ni not in outdct[atn]:
                                outdct[atn][ni] = [None, njumpdelta]
                            else:
                                outdct[atn][ni][1] += njumpdelta

    return outdct


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_random_coil(
    sequence: str,
    pH: float = 7.0,
    temperature: float = 298.0,
    ionic_strength: float = 0.1,
) -> dict[int, dict[str, float]]:
    """Compute POTENCI random coil chemical shifts for a protein sequence.

    Args:
        sequence: One-letter amino acid sequence (e.g. "MKWVTF...").
        pH: Sample pH (default 7.0).
        temperature: Sample temperature in Kelvin (default 298 K).
        ionic_strength: Ionic strength in mol/L (default 0.1 M).

    Returns:
        {residue_index: {"H": ppm, "HA": ppm, "N": ppm, "CA": ppm, "CB": ppm, "C": ppm}}
        where residue_index is 0-based. Terminal residues (first and last)
        are excluded (POTENCI requires ±2 neighbor context).
    """
    if len(sequence) < 5:
        return {}

    # pH correction (skip if pH ≈ 7.0)
    use_ph_corr = abs(pH - 7.0) > 0.01
    if use_ph_corr:
        ph_corrs = _get_ph_corrections(sequence, temperature, pH, ionic_strength)
    else:
        ph_corrs = {}

    bb_atoms = ["C", "CA", "CB", "N", "H", "HA"]
    result: dict[int, dict[str, float]] = {}

    for i in range(1, len(sequence) - 1):
        aa = sequence[i]
        if aa not in _AA_STANDARD:
            continue

        res_shifts: dict[str, float] = {}
        for atom in bb_atoms:
            # Skip impossible combinations
            if (aa, atom) in (("G", "CB"), ("P", "H")):
                continue

            # Build pentapeptide window
            if i == 1:
                pent = "n" + sequence[i - 1:i + 2] + sequence[i + 2]
            elif i == len(sequence) - 2:
                pent = sequence[i - 2] + sequence[i - 1:i + 2] + "c"
            else:
                pent = sequence[i - 2:i + 3]

            if len(pent) != 5:
                continue

            shift = _pred_pent_shift(pent, atom)
            if shift is None:
                continue

            # Temperature correction (not applied to HB)
            if atom in _TEMP_COEFFS and aa in _TEMP_COEFFS[atom]:
                shift += _TEMP_COEFFS[atom][aa] / 1000.0 * (temperature - 298.0)

            # pH correction
            if atom in ph_corrs and i in ph_corrs[atom]:
                ph_data = ph_corrs[atom][i]
                ph_corr = ph_data[1]
                if abs(ph_corr) < 9.9:
                    shift -= ph_corr

            res_shifts[atom] = shift

        if res_shifts:
            result[i] = res_shifts

    return result


def compute_random_coil_array(
    sequence: str,
    pH: float = 7.0,
    temperature: float = 298.0,
    ionic_strength: float = 0.1,
) -> np.ndarray:
    """Compute POTENCI random coil shifts as a [len(sequence), 6] array.

    Array columns follow POTENCI_NUCLEI order: (H, HA, N, CA, CB, C).
    Missing values are NaN.

    Args:
        sequence: One-letter amino acid sequence.
        pH: Sample pH.
        temperature: Temperature in Kelvin.
        ionic_strength: Ionic strength in mol/L.

    Returns:
        [N, 6] float32 array.
    """
    N = len(sequence)
    rc = np.full((N, 6), np.nan, dtype=np.float32)

    shifts = compute_random_coil(sequence, pH, temperature, ionic_strength)
    for idx, nuc_dict in shifts.items():
        for j, nuc in enumerate(POTENCI_NUCLEI):
            if nuc in nuc_dict:
                rc[idx, j] = nuc_dict[nuc]

    return rc


def sequence_3to1(seq_3letter: list[str]) -> str:
    """Convert a list of 3-letter residue codes to a 1-letter string."""
    return "".join(_AA3_TO_1.get(r, "X") for r in seq_3letter)
