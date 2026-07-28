"""
Théorie Nodale v3 — Polynômes de Jones

Le polynôme de Jones V_K(t) est l'invariant de nœud fondamental de la théorie.
Il distingue les nœuds chiraux (matière/antimatière).

Convention : nœuds main droite (right-hand), coefficients depuis Stoimenov/KnotInfo.
Le miroir (main gauche) s'obtient par t → t⁻¹.
"""
import numpy as np
from theory.postulates import GAMMA

# Type alias — Laurent polynomial {power: coefficient}
JonesPolynomial = dict[int, float]

# Jones polynomial table for 20 knots (right-hand convention)
# Source: Stoimenov knot data tables (stoimenov.net/stoimeno/homepage/ptab/)
# Notation: {n} = min power, [c] = coefficient at power 0
_JONES_TABLE: dict[str, JonesPolynomial] = {
    "0_1": {0: 1},
    "3_1": {1: 1, 3: 1, 4: -1},
    "4_1": {-2: 1, -1: -1, 0: 1, 1: -1, 2: 1},
    "5_1": {2: 1, 4: 1, 5: -1, 6: 1, 7: -1},
    "5_2": {1: 1, 2: -1, 3: 2, 4: -1, 5: 1, 6: -1},
    "6_1": {-2: 1, -1: -1, 0: 2, 1: -2, 2: 1, 3: -1, 4: 1},
    "6_2": {-1: 1, 0: -1, 1: 2, 2: -2, 3: 2, 4: -2, 5: 1},
    "6_3": {-3: -1, -2: 2, -1: -2, 0: 3, 1: -2, 2: 2, 3: -1},
    "7_1": {3: 1, 5: 1, 6: -1, 7: 1, 8: -1, 9: 1, 10: -1},
    "7_2": {1: 1, 2: -1, 3: 2, 4: -2, 5: 2, 6: -1, 7: 1, 8: -1},
    "7_3": {2: 1, 3: -1, 4: 2, 5: -2, 6: 3, 7: -2, 8: 1, 9: -1},
    "7_4": {1: 1, 2: -2, 3: 3, 4: -2, 5: 3, 6: -2, 7: 1, 8: -1},
    "7_5": {2: 1, 3: -1, 4: 3, 5: -3, 6: 3, 7: -3, 8: 2, 9: -1},
    "7_6": {-1: 1, 0: -2, 1: 3, 2: -3, 3: 4, 4: -3, 5: 2, 6: -1},
    "7_7": {-4: 1, -3: -2, -2: 3, -1: -4, 0: 4, 1: -3, 2: 3, 3: -1},
    "8_1": {-2: 1, -1: -1, 0: 2, 1: -2, 2: 2, 3: -2, 4: 1, 5: -1, 6: 1},
    "8_2": {0: 1, 1: -1, 2: 2, 3: -2, 4: 3, 5: -3, 6: 2, 7: -2, 8: 1},
    "8_3": {-4: 1, -3: -1, -2: 2, -1: -3, 0: 3, 1: -3, 2: 2, 3: -1, 4: 1},
    "9_1": {4: 1, 6: 1, 7: -1, 8: 1, 9: -1, 10: 1, 11: -1, 12: 1, 13: -1},
    "10_1": {-2: 1, -1: -1, 0: 2, 1: -2, 2: 2, 3: -2, 4: 2, 5: -2, 6: 1, 7: -1, 8: 1},
}

JONES_KNOT_NAMES = list(_JONES_TABLE.keys())

# Braid index table for 20 knots (minimal braid index b(K))
# Source: KnotInfo database (indiana.edu/~knotinfo)
_BRAID_INDEX_TABLE: dict[str, int] = {
    "0_1": 1,   # unknot
    "3_1": 2,   # trefoil — torus T(2,3)
    "4_1": 3,   # figure-eight
    "5_1": 2,   # torus T(2,5)
    "5_2": 3,
    "6_1": 3,
    "6_2": 3,
    "6_3": 4,
    "7_1": 2,   # torus T(2,7)
    "7_2": 3,
    "7_3": 4,
    "7_4": 3,
    "7_5": 4,
    "7_6": 4,
    "7_7": 4,
    "8_1": 3,
    "8_2": 3,
    "8_3": 4,
    "9_1": 2,   # torus T(2,9)
    "10_1": 3,
}

# Bridge number table for 20 knots (nombre de ponts)
# Source: Schubert normal forms, 2-bridge (rational) knot classification
# Tous les nœuds premiers ≤ 10 crossings avec ≤ 2 ponts sont rationnels (S(p,q))
_BRIDGE_NUMBER_TABLE: dict[str, int] = {
    "0_1": 1,   # unknot
    "3_1": 2,   # S(3,1) — trefoil
    "4_1": 2,   # S(5,2) — figure-eight
    "5_1": 2,   # S(5,1) — torus T(2,5)
    "5_2": 2,   # S(7,3)
    "6_1": 2,   # S(9,4)
    "6_2": 2,   # S(11,4)
    "6_3": 2,   # S(13,5)
    "7_1": 2,   # S(7,1) — torus T(2,7)
    "7_2": 2,   # S(11,5)
    "7_3": 2,   # S(13,4)
    "7_4": 2,   # S(15,4)
    "7_5": 2,   # S(17,7)
    "7_6": 2,   # S(19,7)
    "7_7": 2,   # S(21,8)
    "8_1": 2,   # S(13,5)
    "8_2": 2,   # S(17,5)
    "8_3": 2,   # S(17,4)
    "9_1": 2,   # S(9,1) — torus T(2,9)
    "10_1": 2,  # S(17,7)
}


def bridge_number(knot_name: str) -> int:
    """Nombre de ponts br(K) (bridge number).

    Le nombre de ponts est le nombre minimal de maxima locaux
    d'une projection régulière. Tous les nœuds du catalogue sont
    des nœuds à 2 ponts (rationnels / Schubert).

    Raises ValueError for unknown knot names.
    """
    if knot_name not in _BRIDGE_NUMBER_TABLE:
        raise ValueError(f"Unknown knot: {knot_name}. Available: {list(_BRIDGE_NUMBER_TABLE.keys())}")
    return _BRIDGE_NUMBER_TABLE[knot_name]


def jones_polynomial(knot_name: str) -> JonesPolynomial:
    """Return the Jones polynomial for a knot (right-hand convention).

    Raises ValueError for unknown knot names.
    """
    if knot_name not in _JONES_TABLE:
        raise ValueError(f"Unknown knot: {knot_name}. Available: {JONES_KNOT_NAMES}")
    return dict(_JONES_TABLE[knot_name])


def evaluate_jones(poly: JonesPolynomial, t: complex) -> complex:
    """Evaluate V_K(t) = sum(coeff * t^power)."""
    return sum(coeff * t**power for power, coeff in poly.items())


def jones_span(poly: JonesPolynomial) -> int:
    """Span of the Jones polynomial (max_power - min_power).

    For alternating knots, span = crossing number.
    """
    nonzero = [p for p, c in poly.items() if c != 0]
    if not nonzero:
        return 0
    return max(nonzero) - min(nonzero)


def jones_determinant(poly: JonesPolynomial) -> int:
    """Knot determinant |V_K(-1)|.

    Theorem: |V_K(-1)| = |Delta_K(-1)| for all knots.
    """
    return abs(round(evaluate_jones(poly, -1.0).real))


def jones_retraction_factor(poly: JonesPolynomial) -> float:
    """Retraction factor rho_J(K) = span(V_K) + gamma * |V_K(-1)|.

    Primary retraction factor for the theory.
    """
    return float(jones_span(poly)) + GAMMA * float(jones_determinant(poly))


def mirror_jones(poly: JonesPolynomial) -> JonesPolynomial:
    """Mirror image: V_{K*}(t) = V_K(t^{-1}), i.e., negate all powers."""
    return {-p: c for p, c in poly.items()}


def is_amphicheiral(poly: JonesPolynomial) -> bool:
    """Check if a knot is amphicheiral (equal to its mirror image).

    V_K(t) = V_K(t^{-1}) iff all nonzero coefficients are symmetric.
    """
    mirror = mirror_jones(poly)
    all_powers = set(poly.keys()) | set(mirror.keys())
    for p in all_powers:
        if abs(poly.get(p, 0) - mirror.get(p, 0)) > 1e-10:
            return False
    return True


def poly_to_str(poly: JonesPolynomial) -> str:
    """Human-readable string for a Jones polynomial."""
    if not poly:
        return "0"
    powers = sorted(poly.keys())
    terms = []
    for p in powers:
        c = poly[p]
        if c == 0:
            continue
        if p == 0:
            terms.append(f"{c:+g}")
        elif p == 1:
            terms.append(f"{c:+g}t")
        else:
            terms.append(f"{c:+g}t^{p}")
    result = " ".join(terms) if terms else "0"
    if result.startswith("+"):
        result = result[1:]
    return result.strip()


def braid_index(knot_name: str) -> int:
    """Indice de tresse minimal b(K).

    Raises ValueError for unknown knot names.
    """
    if knot_name not in _BRAID_INDEX_TABLE:
        raise ValueError(f"Unknown knot: {knot_name}. Available: {list(_BRAID_INDEX_TABLE.keys())}")
    return _BRAID_INDEX_TABLE[knot_name]


def topological_spin(knot_name: str) -> float:
    """Spin dérivé de l'indice de tresse : s(K) = (b(K) - 1) / 2.

    b=1 → spin 0 (scalaire), b=2 → spin 1/2 (fermion),
    b=3 → spin 1 (boson de jauge), b≥4 → spin 3/2+.
    """
    b = braid_index(knot_name)
    return (b - 1) / 2.0


def is_half_integer_spin(knot_name: str) -> bool:
    """Retourne True si le nœud a un spin demi-entier (b pair → spin 1/2, 3/2, ...).

    Un nœud est « fermionique » si son indice de tresse b(K) est pair,
    ce qui donne un spin s = (b-1)/2 demi-entier.
    """
    b = braid_index(knot_name)
    return b % 2 == 0


def jones_power_center(poly: JonesPolynomial) -> float:
    """Centre de puissance normalisé : (p_min + p_max) / span(V_K).

    Mesure l'asymétrie du polynôme de Jones.
    Amphichiraux → 0, chiraux → valeur fractionnaire.
    """
    nonzero = [p for p, c in poly.items() if c != 0]
    if not nonzero:
        return 0.0
    p_min = min(nonzero)
    p_max = max(nonzero)
    span = p_max - p_min
    if span == 0:
        return 0.0
    return (p_min + p_max) / span


def topological_charge(knot_name: str) -> float:
    """Charge topologique Q(K) = (p_min + p_max) / span(V_K).

    Propriétés :
    - Amphichiraux (4_1, 6_3, 8_3) → Q = 0
    - Miroir : Q(K*) = -Q(K) → antiparticule a charge opposée
    - Valeurs fractionnaires SM : 1/3, 2/3, 1
    """
    poly = jones_polynomial(knot_name)
    return jones_power_center(poly)
