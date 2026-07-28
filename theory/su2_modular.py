"""
Structure modulaire SU(2)_k pour TQNT-V5 (chap. 12.5 et V5.5 dérivation α).

À k = 135 fixé (V5 niveau unique), expose :
  - dimensions quantiques d_j pour j ∈ {0, 1/2, 1, ..., k/2}
  - matrice modulaire S_{jl}
  - poids conformes h_j
  - dimension totale D²
  - Wilson loop expectations ⟨W_K⟩ pour les nœuds catalogue

Référence : Witten "Quantum Field Theory and the Jones Polynomial",
Commun. Math. Phys. 121 (1989) 351 ; Verlinde "Fusion rules and modular
transformations in 2D conformal field theory", Nucl. Phys. B 300 (1988) 360.

API
---
    quantum_dim(j, k=135)          -> dim quantique d_j
    total_dim_squared(k=135)       -> D² = Σ_j d_j²
    modular_S(j, l, k=135)         -> élément S-matrice
    conformal_weight(j, k=135)     -> h_j
    wilson_loop(knot_name, k=135)  -> ⟨W_K⟩ (complexe)
    central_charge(k=135)          -> c = 3k/(k+2)

Notes
-----
À k=135, k+2 = 137 — coïncidence numérique fondamentale TQNT :
α_EM = 1/(k+2) = 1/137. Cette identification (V5 §10.4, §21.2) est le
verrou théorique de la cohérence V5 entre Chern-Simons et électromagnétisme.

Les unités sont :
- j est un demi-entier (j ∈ {0, 0.5, 1, 1.5, ..., k/2})
- d_j est sans dimension
- h_j est sans dimension (∈ [0, 1])
- ⟨W_K⟩ est complexe en général
"""
from __future__ import annotations

import cmath
import math
from functools import lru_cache

V5_LEVEL = 135
"""Niveau Chern-Simons unique de la TQNT-V5 (V5 §10.4)."""


@lru_cache(maxsize=4096)
def quantum_dim(j: float, k: int = V5_LEVEL) -> float:
    """Dimension quantique de la représentation de spin j à niveau k.

    d_j = sin[π(2j+1)/(k+2)] / sin[π/(k+2)]

    À j = 0 : d_0 = 1 (vide).
    """
    if j < 0 or j > k / 2:
        return 0.0
    numerator = math.sin(math.pi * (2 * j + 1) / (k + 2))
    denominator = math.sin(math.pi / (k + 2))
    return numerator / denominator


@lru_cache(maxsize=4)
def total_dim_squared(k: int = V5_LEVEL) -> float:
    """Dimension modulaire totale D² = Σ_j d_j² = (k+2) / (2 sin²(π/(k+2)))."""
    return (k + 2) / (2 * math.sin(math.pi / (k + 2)) ** 2)


def central_charge(k: int = V5_LEVEL) -> float:
    """Charge centrale c = 3k/(k+2). À k=135 : c = 405/137 ≈ 2.956."""
    return 3 * k / (k + 2)


@lru_cache(maxsize=4096)
def conformal_weight(j: float, k: int = V5_LEVEL) -> float:
    """Poids conforme h_j = j(j+1)/(k+2) du primaire de spin j.

    À j = 0 : h_0 = 0 (identité).
    À j = 1/2 : h = (1/4·3/2)/137 = 3/8 · 1/137 ≈ 0.00274.
    """
    return j * (j + 1) / (k + 2)


@lru_cache(maxsize=2048)
def modular_S(j: float, l: float, k: int = V5_LEVEL) -> float:
    """Élément de la matrice modulaire S_{jl}.

    S_{jl} = √(2/(k+2)) · sin[π(2j+1)(2l+1)/(k+2)]

    Propriétés :
      S² = (ST)³ = 1 (charge conjugaison + transformation modulaire)
      S_{0j}/S_{00} = d_j (dimensions quantiques)
    """
    prefactor = math.sqrt(2 / (k + 2))
    return prefactor * math.sin(math.pi * (2 * j + 1) * (2 * l + 1) / (k + 2))


def topological_T_phase(j: float, k: int = V5_LEVEL) -> complex:
    """Phase de la matrice diagonale T : T_{jj} = exp(2πi(h_j - c/24)).

    À utiliser dans la composition modulaire (ST)³ = 1.
    """
    return cmath.exp(2j * cmath.pi * (conformal_weight(j, k)
                                      - central_charge(k) / 24))


# ------------------------------------------------------------
# Wilson loop expectation values pour les nœuds catalogue V5
# ------------------------------------------------------------
# Pour un nœud K identifié par sa représentation R_K dans SU(2)_k,
# l'expectation Wilson loop ⟨W_K⟩ est donnée par le polynôme de Jones :
#   ⟨W_K⟩ = V_K(q) avec q = exp(2πi/(k+2))
# où V_K est le polynôme de Jones normalisé.

@lru_cache(maxsize=64)
def q_parameter(k: int = V5_LEVEL) -> complex:
    """Paramètre quantique q = exp(2πi/(k+2)).

    À k=135 : q = exp(2πi/137).
    """
    return cmath.exp(2j * cmath.pi / (k + 2))


def wilson_loop_unknot(k: int = V5_LEVEL) -> float:
    """⟨W_unknot⟩ = d_{1/2} = sin(2π/(k+2))/sin(π/(k+2)) = 2 cos(π/(k+2))."""
    return quantum_dim(0.5, k)


def wilson_loop_from_jones(jones_poly, k: int = V5_LEVEL) -> complex:
    """⟨W_K⟩ = V_K(q) évalué en q = exp(2πi/(k+2)).

    `jones_poly` : dict {power: coefficient} comme défini dans theory.jones.
    """
    q = q_parameter(k)
    total = 0j
    for power, coeff in jones_poly.items():
        total += coeff * (q ** power)
    return total


# ------------------------------------------------------------
# Diagnostic et tests rapides
# ------------------------------------------------------------
def summary(k: int = V5_LEVEL) -> dict:
    """Retourne un résumé des quantités modulaires à niveau k."""
    return {
        "k": k,
        "k_plus_2": k + 2,
        "alpha_EM": 1.0 / (k + 2),
        "central_charge_c": central_charge(k),
        "D_squared": total_dim_squared(k),
        "d_half": quantum_dim(0.5, k),
        "h_half": conformal_weight(0.5, k),
        "n_primaries": k + 1,  # j ∈ {0, 1/2, ..., k/2}
        "S_00": modular_S(0, 0, k),
        "S_half_half": modular_S(0.5, 0.5, k),
    }


if __name__ == "__main__":
    print("=" * 70)
    print(f"Structure modulaire SU(2)_{V5_LEVEL} pour TQNT-V5")
    print("=" * 70)
    s = summary()
    for key, value in s.items():
        if isinstance(value, float):
            print(f"  {key:25s} = {value:.6e}")
        else:
            print(f"  {key:25s} = {value}")

    print()
    print("Dimensions quantiques des premiers spins :")
    for j2 in [0, 1, 2, 3, 4, 5, 10, 50, 134, 135]:
        j = j2 / 2
        d = quantum_dim(j)
        h = conformal_weight(j)
        print(f"  j = {j:6.1f}   d_j = {d:>10.4f}   h_j = {h:>10.6f}")

    print()
    print(f"Total D² = {total_dim_squared():.4e}")
    print(f"α_EM = 1/(k+2) = {1.0/(V5_LEVEL+2):.10f} = {1.0/(V5_LEVEL+2)*137.036:.5f}×α_PDG")
    print()
    print("Ces quantités sont la donnée structurelle pour la dérivation")
    print("first-principles de α (V5 §13.5) — étape suivante : implémenter")
    print("le bilan détaillé χ⁰ ⇌ K(c_n) en utilisant ⟨W_K⟩.")
