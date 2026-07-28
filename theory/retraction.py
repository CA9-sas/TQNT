"""
Théorie Nodale v3 — Facteur de Rétractation (Jones)

    rho_J(K) = span(V_K) + gamma * |V_K(-1)|
    M(K) = alpha * rho_J(K)
    rho(K, E) = rho_J² / (rho_J + E/E₀)
"""
import numpy as np
from theory.postulates import ALPHA, R_0, E_0
from theory.jones import JonesPolynomial, jones_retraction_factor as _jones_rho


def retraction_factor(poly: JonesPolynomial) -> float:
    """Retraction factor from Jones polynomial.

    ρ_J(K) = span(V_K) + γ·|V_K(-1)|
    """
    return _jones_rho(poly)


def knot_mass(poly: JonesPolynomial) -> float:
    """[OBSOLÈTE V3] Masse depuis Jones via M(K) = α · ρ_J(K).

    Pour la formule opérationnelle V4/V5 (m = α₀ · U_EM · c_J^3,55, éq. 6.8),
    utiliser ``theory.mass_formula.mass(name, sector="auto")``. Conservée pour
    compatibilité ; émet un DeprecationWarning à chaque appel.
    """
    import warnings
    warnings.warn(
        "theory.retraction.knot_mass est OBSOLÈTE (V3, M = α · ρ_J). "
        "Utiliser theory.mass_formula.mass(name) pour la formule V5 (éq. 6.8).",
        DeprecationWarning, stacklevel=2,
    )
    return ALPHA * retraction_factor(poly)


def tightening_radius(E: float) -> float:
    return R_0 / (1.0 + E / E_0)


def energy_dependent_retraction(poly: JonesPolynomial, energy: float) -> float:
    """Energy-dependent retraction: ρ(K, E) = ρ_J² / (ρ_J + E/E₀)."""
    rho_J = retraction_factor(poly)
    if rho_J == 0:
        return 0.0
    return rho_J**2 / (rho_J + energy / E_0)
