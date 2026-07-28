"""Polynôme d'Alexander Δ_K(t) et genre de Seifert g(K).

Le polynôme d'Alexander est un invariant fondamental du nœud. Pour les nœuds
**alternés** (les 19 du catalogue le sont), le genre de Seifert se lit directement
sur son envergure (théorème de Crowell--Murasugi) :

    g(K) = breadth(Δ_K) / 2 = (deg Δ_K) / 2     (forme à une variable, normalisée).

On stocke Δ_K sous forme de liste de coefficients (degré croissant, forme à une
variable telle que Δ_K(t) = Σ c_i t^i, normalisée à c_0 ≠ 0). Chaque polynôme est
validé par |Δ_K(-1)| = det(K), recoupé avec le déterminant de Jones |V_K(-1)|.

Réf. : Crowell (1959), Murasugi (1958) ; valeurs KnotInfo.
"""
from __future__ import annotations

# Polynôme d'Alexander (forme à une variable, coefficients de degré croissant).
# Symétrique : c_i = c_{deg-i}. deg = 2·genre. Validé par |Δ(-1)| = det(K).
_ALEXANDER = {
    "3_1":  [1, -1, 1],
    "4_1":  [1, -3, 1],
    "5_1":  [1, -1, 1, -1, 1],
    "5_2":  [2, -3, 2],
    "6_1":  [2, -5, 2],
    "6_2":  [1, -3, 3, -3, 1],
    "6_3":  [1, -3, 5, -3, 1],
    "7_1":  [1, -1, 1, -1, 1, -1, 1],
    "7_2":  [3, -5, 3],
    "7_3":  [2, -3, 3, -3, 2],
    "7_4":  [4, -7, 4],
    "7_5":  [2, -4, 5, -4, 2],
    "7_6":  [1, -5, 7, -5, 1],
    "7_7":  [1, -5, 9, -5, 1],
    "8_1":  [3, -7, 3],
    "8_2":  [1, -3, 3, -3, 3, -3, 1],
    "8_3":  [1, -4, 7, -4, 1],
    "9_1":  [1, -1, 1, -1, 1, -1, 1, -1, 1],
    "10_1": [4, -9, 4],
}


def alexander_polynomial(name: str) -> dict[int, int]:
    """Δ_K(t) sous forme {puissance: coefficient} (forme symétrique centrée).

    Exemple : 3_1 -> {-1: 1, 0: -1, 1: 1} (soit t - 1 + t^{-1}).
    """
    coeffs = _ALEXANDER[name]
    deg = len(coeffs) - 1
    shift = deg // 2
    return {i - shift: c for i, c in enumerate(coeffs)}


def alexander_determinant(name: str) -> int:
    """|Δ_K(-1)| = det(K) (cross-check du déterminant de Jones)."""
    return abs(sum(c * ((-1) ** i) for i, c in enumerate(_ALEXANDER[name])))


def genus(name: str) -> int:
    """Genre de Seifert g(K) = breadth(Δ_K)/2 (Crowell--Murasugi, nœuds alternés)."""
    return (len(_ALEXANDER[name]) - 1) // 2


if __name__ == "__main__":
    # Validation : |Δ(-1)| doit égaler le déterminant de Jones.
    from theory.jones import jones_polynomial
    print(f"{'K':>5} | {'genre g':>7} | {'det(Δ)':>6} | {'det(Jones)':>10} | ok")
    for k in _ALEXANDER:
        poly = jones_polynomial(k)
        det_j = abs(round(sum(c * ((-1) ** p) for p, c in poly.items())))
        det_a = alexander_determinant(k)
        print(f"{k:>5} | {genus(k):>7} | {det_a:>6} | {det_j:>10} | {'✓' if det_a == det_j else '✗'}")
