"""
Théorie Nodale v3 — Champ χ (tchi)

Le champ fondamental χ satisfait l'équation de Chern-Simons SU(2).
Sa quantification sur le complément du nœud produit l'équation de Schrödinger
émergente. Le facteur de rétractation ρ(K, E) dépend à la fois du type de
nœud (via Jones) et de l'énergie.

Formule : ρ(K, E) = ρ_J(K)² / (ρ_J(K) + E/E₀)
"""
import numpy as np
from theory.jones import jones_polynomial, jones_retraction_factor
from theory.postulates import LAMBDA, E_0


def jones_rho(knot_name: str) -> float:
    """Topological retraction factor ρ_J(K) from Jones polynomial.

    ρ_J(K) = span(V_K) + γ·|V_K(-1)|
    """
    poly = jones_polynomial(knot_name)
    return jones_retraction_factor(poly)


def chi_retraction(knot_name: str, energy: float) -> float:
    """Energy-dependent retraction factor ρ(K, E).

    ρ(K, E) = ρ_J(K)² / (ρ_J(K) + E/E₀)

    Properties:
    - E = 0 → ρ = ρ_J(K)
    - E = E₀·ρ_J → ρ = ρ_J/2
    - E → ∞ → ρ → 0
    """
    rho_J = jones_rho(knot_name)
    if rho_J == 0:
        return 0.0
    return rho_J**2 / (rho_J + energy / E_0)


def characteristic_energy(knot_name: str) -> float:
    """Characteristic energy scale E_K = E₀ · ρ_J(K).

    At this energy, the retraction is halved: ρ(K, E_K) = ρ_J/2.
    """
    return E_0 * jones_rho(knot_name)


def chi_potential(
    points: np.ndarray,
    knot_name: str,
    energy: float,
    knot_tree,
    epsilon: float = 0.2,
) -> np.ndarray:
    """Retraction potential V(r, K, E) = -λ · ρ(K, E) / d(r, K)².

    Parameters
    ----------
    points : (N, 3) array — evaluation points
    knot_name : str — knot identifier
    energy : float — current energy (for ρ dependence)
    knot_tree : cKDTree — knot curve tree for distance queries
    epsilon : float — minimum distance cutoff
    """
    rho = chi_retraction(knot_name, energy)
    dists, _ = knot_tree.query(points)
    dists = np.maximum(dists, epsilon)
    return -LAMBDA * rho / dists**2
