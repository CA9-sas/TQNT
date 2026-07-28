"""
Théorie Nodale — Rayon des nœuds du nucléon par densité particulaire

Calcul du rayon effectif de chaque nœud (9₁, 7₄, 6₃) du nucléon,
libre et comprimé, avec correction métrique (P5) et prédiction du
rayon de charge du proton.

Formules clés :
  - Rayon libre : r_free(K) = ℏc / ρ_J(K)  (rayon d'exclusion P16)
  - Rétractation comprimée : ρ(K,E) = ρ_J²/(ρ_J + E/E₀)  (P4)
  - Rayon de confinement : r_conf(K_i) = d*/2 × ρ_J(K_i)/ρ_total
  - Densité particulaire : n_i = ρ(K_i,E*) / V_i
  - Facteur métrique : Ω_i = √(1 + κ·n_i)  (P5)
  - Rayon propre : r_propre(K_i) = r_conf(K_i) × Ω_i
  - Rayon du proton : r_p = 2·N_Ψ·ℏ/(m_p·c)  avec N_Ψ = 2 (P16)
"""
from __future__ import annotations

import numpy as np

from theory.jones import jones_polynomial, jones_retraction_factor
from theory.chi_field import chi_retraction, jones_rho
from theory.multi_knot import (
    cornell_equilibrium, fibonacci, ETA, SM_ASSIGNMENTS,
)
from theory.postulates import KAPPA, HBAR, C, R_0, E_0

# --- Constantes physiques SI ---
HBAR_SI = 1.054_571_817e-34   # J·s
C_SI = 2.997_924_58e8          # m/s
M_P_SI = 1.672_621_924e-27     # kg
HBAR_C_MEV_FM = 197.327         # MeV·fm
M_P_MEV = 938.272               # MeV/c²
R_P_EXP_FM = 0.8414             # fm (CODATA 2018)

# --- Configuration du nucléon (P17) ---
NUCLEON_KNOTS = ["9_1", "7_4", "6_3"]
NUCLEON_K = 18
# P16 : 7_4 est exclu (son puits effectif est dominé par 9_1 et 6_3)
ACTIVE_KNOTS = ["9_1", "6_3"]
N_PSI = len(ACTIVE_KNOTS)  # Nombre de nœuds porteurs de Ψ


def knot_free_radius(knot_name: str) -> float:
    """Rayon libre d'un nœud isolé (rayon d'exclusion P16).

    r_free(K) = ℏc / ρ_J(K)  [unités code]

    C'est la résolution minimale où le nœud peut porter une fonction d'onde.
    Inversement proportionnel à la complexité topologique.
    """
    rho_J = jones_rho(knot_name)
    if rho_J <= 0:
        return float("inf")
    return HBAR * C / rho_J


def knot_compressed_retraction(knot_name: str, E_bind: float) -> float:
    """Rétractation comprimée à l'énergie de liaison E_bind (P4).

    ρ(K, E) = ρ_J(K)² / (ρ_J(K) + E/E₀)
    """
    return chi_retraction(knot_name, E_bind)


def nucleon_cornell() -> dict:
    """Équilibre de Cornell du nucléon (k=18).

    Retourne le dict cornell_equilibrium enrichi avec E_bind.
    """
    eq = cornell_equilibrium(NUCLEON_KNOTS, NUCLEON_K)
    eq["E_bind"] = eq["mass_raw"] - eq["M_base"]
    eq["m_nucleon_mev"] = ETA * eq["mass_raw"]
    return eq


def nucleon_knot_radii() -> list[dict]:
    """Calcule les rayons de chaque nœud du nucléon.

    Pour chaque nœud K_i (9₁, 7₄, 6₃), calcule :
    - ρ_J : facteur de rétractation libre
    - r_free : rayon libre (P16, unités code)
    - ρ_comp : rétractation comprimée à E*
    - r_conf : rayon de confinement (part de d* proportionnelle à ρ_J/ρ_total)
    - compression_ratio : r_free / r_conf
    - V_conf : volume de confinement
    - n_particle : densité particulaire (ρ_comp / V_conf)
    - Omega : facteur métrique √(1 + κ·n_particle) (P5)
    - r_proper : rayon propre (r_conf × Ω)
    - active : True si le nœud porte Ψ (P16)
    """
    eq = nucleon_cornell()
    d_star = eq["d_star"]
    E_bind = eq["E_bind"]
    rho_total = eq["rho_total"]

    results = []
    for name in NUCLEON_KNOTS:
        rho_J = jones_rho(name)
        r_free = knot_free_radius(name)

        # Rétractation comprimée
        rho_comp = knot_compressed_retraction(name, E_bind)

        # Rayon de confinement : chaque nœud occupe une fraction de d*/2
        # proportionnelle à son poids topologique ρ_J/ρ_total
        r_conf = (d_star / 2.0) * (rho_J / rho_total)

        # Ratio de compression
        compression = r_free / r_conf if r_conf > 0 else float("inf")

        # Volume de confinement (sphère)
        V_conf = (4.0 / 3.0) * np.pi * r_conf**3

        # Densité particulaire
        n_particle = rho_comp / V_conf if V_conf > 0 else 0.0

        # Facteur métrique (P5)
        Omega = np.sqrt(1.0 + KAPPA * n_particle)

        # Rayon propre (distance physique corrigée par la métrique)
        r_proper = r_conf * Omega

        # Actif (P16) ?
        active = name in ACTIVE_KNOTS

        results.append({
            "name": name,
            "rho_J": rho_J,
            "r_free": r_free,
            "rho_comp": rho_comp,
            "r_conf": r_conf,
            "compression_ratio": compression,
            "V_conf": V_conf,
            "n_particle": n_particle,
            "Omega": Omega,
            "r_proper": r_proper,
            "active": active,
        })

    return results


def code_to_fm() -> float:
    """Facteur de conversion unités code → femtomètres.

    Calibration par le rayon de charge du proton :
    L₀ = R_p_exp / R_knot_code

    où R_knot_code est le rayon caractéristique de la courbe 9₁
    (composante dominante du nucléon).
    """
    from theory.knot_catalog import knot_curve
    curve = knot_curve("9_1", 500)
    R_knot_code = np.std(np.linalg.norm(curve, axis=1))
    return R_P_EXP_FM / R_knot_code


def proton_radius_prediction() -> dict:
    """Prédiction du rayon de charge du proton.

    Formule topologique :
      r_p = 2 · N_Ψ · ℏ/(m_p·c)

    où N_Ψ = 2 est le nombre de nœuds porteurs de Ψ (P16).

    Le facteur 2 par nœud actif provient de l'extension radiale
    de la fonction d'onde dans le puits de potentiel V ∝ -ρ_J/r².

    Returns
    -------
    dict avec r_p_fm, r_p_exp, error_pct, lambda_C, N_psi, factor
    """
    # Longueur d'onde de Compton du proton
    lambda_C = HBAR_C_MEV_FM / M_P_MEV  # fm

    # Prédiction : r_p = 2 × N_Ψ × λ_C
    factor = 2 * N_PSI
    r_p_pred = factor * lambda_C

    error_pct = abs(r_p_pred - R_P_EXP_FM) / R_P_EXP_FM * 100

    return {
        "r_p_fm": r_p_pred,
        "r_p_exp": R_P_EXP_FM,
        "error_pct": error_pct,
        "lambda_C_fm": lambda_C,
        "N_psi": N_PSI,
        "factor": factor,
    }


# --- Constantes pour les limites physiques ---
G_SI = 6.674_30e-11            # m³/(kg·s²)
R_SCHWARZSCHILD_FM = 2 * G_SI * M_P_SI / C_SI**2 * 1e15  # fm
LAMBDA_C_FM = HBAR_C_MEV_FM / M_P_MEV                      # fm


def cornell_energy_at_d(d: float | np.ndarray) -> float | np.ndarray:
    """Énergie totale de Cornell U(d) pour le nucléon (unités code).

    U(d) = M_base + A/d + σ_k·d

    où A et σ_k sont les paramètres du nucléon (k=18).
    """
    from theory.multi_knot import cornell_potential
    return cornell_potential(NUCLEON_KNOTS, d, NUCLEON_K)


def cornell_pressure(d: float | np.ndarray) -> float | np.ndarray:
    """Pression particulaire P(d) dans le nucléon (unités code).

    P = -dU/dV  avec V = (4π/3)(d/2)³

    dU/dd = -A/d² + σ_k
    dV/dd = π d²/2

    P(d) = 2(A/d² - σ_k) / (π d²)

    P > 0 pour d < d* (pression dégénérée, résiste à la compression)
    P = 0 pour d = d* (équilibre)
    P < 0 pour d > d* (tension)
    """
    eq = nucleon_cornell()
    A = eq["A"]
    sigma_k = eq["sigma_k"]
    d = np.asarray(d, dtype=float)
    return 2.0 * (A / d**2 - sigma_k) / (np.pi * d**2)


def proton_radius_vs_density(
    n_points: int = 200,
    d_min_factor: float = 0.05,
    d_max_factor: float = 50.0,
) -> dict:
    """Rayon du proton en fonction de la densité particulaire.

    Balaye la distance inter-nœuds d de d_min à d_max et calcule
    à chaque point :
    - d : distance inter-nœuds
    - m(d) : masse du nucléon via Cornell
    - r_p(d) = 4·ℏc/(m(d)·c²)  [fm]
    - n(d) = ρ_total / V(d)  [densité particulaire, code⁻³]
    - P(d) = pression de Cornell  [code⁻⁴]

    Les limites physiques sont calculées :
    - r_Schwarzschild : rayon de Schwarzschild du proton
    - r_Compton : longueur d'onde de Compton (λ_C)
    - r_equilibre : rayon à d = d* (valeur physique 0.84 fm)

    Returns dict with arrays and scalar limits.
    """
    eq = nucleon_cornell()
    d_star = eq["d_star"]
    rho_total = eq["rho_total"]

    d_arr = np.geomspace(d_star * d_min_factor, d_star * d_max_factor, n_points)

    # Masse via Cornell : m(d) = ETA × U(d)
    U_arr = cornell_energy_at_d(d_arr)
    m_arr = ETA * U_arr  # MeV

    # Rayon du proton : r_p(d) = 4·ℏc/(m(d)·c²) [fm]
    r_p_arr = 4.0 * HBAR_C_MEV_FM / m_arr

    # Volume et densité
    V_arr = (4.0 / 3.0) * np.pi * (d_arr / 2.0)**3
    n_arr = rho_total / V_arr

    # Pression
    P_arr = cornell_pressure(d_arr)

    # Densité et pression à l'équilibre
    V_star = (4.0 / 3.0) * np.pi * (d_star / 2.0)**3
    n_star = rho_total / V_star
    r_p_star = 4.0 * HBAR_C_MEV_FM / (ETA * eq["mass_raw"])

    # Limites physiques
    # 1. Schwarzschild : r_S = 2Gm_p/c²
    r_schwarzschild = R_SCHWARZSCHILD_FM

    # 2. Compton : λ_C = ℏc/(m_p c²)
    r_compton = LAMBDA_C_FM

    # 3. Densité au Schwarzschild (quand r_p = r_S)
    #    4·ℏc/m = r_S → m = 4·ℏc/r_S
    m_schwarzschild = 4.0 * HBAR_C_MEV_FM / r_schwarzschild
    # Trouver d correspondant : U(d) = m_S/ETA → compression extrême

    # 4. Densité au Compton (quand r_p = λ_C → facteur = 1)
    m_compton = 4.0 * HBAR_C_MEV_FM / r_compton
    # m_compton = 4 × m_p (facteur 4)

    return {
        "d": d_arr,
        "m_mev": m_arr,
        "r_p_fm": r_p_arr,
        "n": n_arr,
        "P": P_arr,
        "d_star": d_star,
        "n_star": n_star,
        "r_p_star": r_p_star,
        "r_schwarzschild": r_schwarzschild,
        "r_compton": r_compton,
        "r_p_exp": R_P_EXP_FM,
    }


def full_nucleon_analysis() -> dict:
    """Analyse complète : rayons libres, comprimés, densités, proton radius.

    Returns dict with keys:
        cornell, knot_radii, proton_radius, L0_fm
    """
    eq = nucleon_cornell()
    radii = nucleon_knot_radii()
    r_p = proton_radius_prediction()
    L0 = code_to_fm()

    return {
        "cornell": eq,
        "knot_radii": radii,
        "proton_radius": r_p,
        "L0_fm": L0,
    }
