"""Théorie Nodale — Entrelacs neutrino-photon (Hopf link minimal)

Un neutrino (reliquat ondulatoire, P14) tourne à v = c sur un cercle de
rayon R_ν = ℏc/(2 m_ν). Pour la désintégration β du neutron (m_ν ≈ 0.57 eV),
R_ν ≈ 173 nm — une échelle macroscopique.

Un photon de longueur d'onde λ_γ ~ R peut traverser ce disque et former
un **lien de Hopf** (entrelacs de deux inconnus 0₁, linking number = 1).

Ce module calcule :
  - La section efficace géométrique σ = π R²
  - La condition spectrale E_γ = ℏc/R
  - La densité spectrale de Planck
  - Le taux d'interaction et le libre parcours moyen
  - Le nombre total d'événements sur le volume de Hubble
  - Les propriétés du Hopf link résultant (masse, stabilité)

Résultat principal : le taux est **nul aujourd'hui** (CMB trop froid,
suppression exp(−4250)), mais significatif dans l'univers jeune (z > 1720,
T > 4700 K) et les environnements astrophysiques chauds.
"""
from __future__ import annotations

import math

from theory.neutrino import HBAR_C_MEV_FM, boomerang_radius


# ── Constantes physiques ──────────────────────────────────────────────

T_CMB = 2.7255               # K — température du CMB aujourd'hui
N_GAMMA_TOTAL = 411.0         # cm⁻³ — densité totale de photons du CMB
N_NU_PER_FLAVOR = 56.0        # cm⁻³ — densité de neutrinos par saveur
R_HUBBLE_M = 4.4e26           # m — rayon de Hubble
T_HUBBLE_S = 4.35e17          # s — temps de Hubble
K_B_EV_PER_K = 8.617e-5       # eV/K — constante de Boltzmann
C_CM_PER_S = 2.99792458e10    # cm/s — vitesse de la lumière
HBAR_C_EV_CM = 1.9732698e-5   # ℏc en eV·cm
FM_PER_CM = 1.0e13            # fm par cm
V_HUBBLE_CM3 = (4.0 / 3.0) * math.pi * (R_HUBBLE_M * 100) ** 3  # cm³


# ── Section efficace géométrique ──────────────────────────────────────

def linking_cross_section(R_nu_fm: float) -> float:
    """Section efficace géométrique σ = π R² pour l'entrelacs neutrino-photon.

    Le photon doit traverser le disque de rayon R_ν pour former un
    lien de Hopf avec la boucle du neutrino.

    Parameters
    ----------
    R_nu_fm : rayon du neutrino en fm

    Returns
    -------
    σ en fm²
    """
    return math.pi * R_nu_fm ** 2


# ── Condition spectrale ───────────────────────────────────────────────

def spectral_matching_energy(R_nu_fm: float) -> float:
    """Énergie du photon résonant E_γ = ℏc/R (MeV).

    La longueur d'onde du photon doit correspondre au rayon du neutrino
    pour que l'entrelacs soit topologiquement possible.

    Parameters
    ----------
    R_nu_fm : rayon du neutrino en fm

    Returns
    -------
    E_γ en MeV
    """
    return HBAR_C_MEV_FM / R_nu_fm


# ── Distribution de Planck ────────────────────────────────────────────

def planck_spectral_density(E_eV: float, T_K: float) -> float:
    """Densité spectrale de photons de Planck dn/dE (cm⁻³ eV⁻¹).

    Formule de Planck pour le nombre de photons par unité de volume
    et par unité d'énergie :

        dn/dE = (8π / (hc)³) · E² / (exp(E / k_B T) − 1)

    Parameters
    ----------
    E_eV : énergie du photon en eV
    T_K : température en Kelvin

    Returns
    -------
    dn/dE en cm⁻³ eV⁻¹
    """
    if T_K <= 0 or E_eV <= 0:
        return 0.0
    x = E_eV / (K_B_EV_PER_K * T_K)
    if x > 700:
        return 0.0  # suppression exponentielle
    prefactor = 1.0 / (math.pi ** 2 * HBAR_C_EV_CM ** 3)
    return prefactor * E_eV ** 2 / (math.exp(x) - 1.0)


def photon_density_at_wavelength(lambda_fm: float, T_K: float,
                                 bandwidth: float = 1.0) -> float:
    """Densité de photons dans une bande spectrale autour de λ (cm⁻³).

    Intègre la distribution de Planck sur une bande de largeur
    `bandwidth` eV centrée sur E = ℏc/λ.

    Parameters
    ----------
    lambda_fm : longueur d'onde en fm
    T_K : température en Kelvin
    bandwidth : largeur de bande en eV (défaut = 1 eV)

    Returns
    -------
    n_γ en cm⁻³
    """
    E_eV = HBAR_C_MEV_FM / lambda_fm * 1e6  # MeV → eV
    return planck_spectral_density(E_eV, T_K) * bandwidth


# ── Taux d'interaction ────────────────────────────────────────────────

def interaction_rate_density(n_gamma: float, n_nu: float,
                             sigma_fm2: float) -> float:
    """Taux d'interaction par unité de volume Γ/V (cm⁻³ s⁻¹).

    Γ/V = n_γ · n_ν · σ · c

    Parameters
    ----------
    n_gamma : densité de photons (cm⁻³)
    n_nu : densité de neutrinos (cm⁻³)
    sigma_fm2 : section efficace en fm²

    Returns
    -------
    Γ/V en cm⁻³ s⁻¹
    """
    sigma_cm2 = sigma_fm2 * 1e-26  # 1 fm² = 10⁻²⁶ cm²
    return n_gamma * n_nu * sigma_cm2 * C_CM_PER_S


# ── Libre parcours moyen ──────────────────────────────────────────────

def mean_free_path(n_target: float, sigma_fm2: float) -> float:
    """Libre parcours moyen l = 1/(n·σ) en cm.

    Parameters
    ----------
    n_target : densité de cibles (cm⁻³)
    sigma_fm2 : section efficace en fm²

    Returns
    -------
    l en cm
    """
    sigma_cm2 = sigma_fm2 * 1e-26
    if n_target * sigma_cm2 <= 0:
        return float('inf')
    return 1.0 / (n_target * sigma_cm2)


# ── Événements sur le volume de Hubble ────────────────────────────────

def total_events_hubble(rate_density: float) -> float:
    """Nombre total d'événements dans le volume de Hubble sur un temps de Hubble.

    N = (Γ/V) · V_H · t_H

    Parameters
    ----------
    rate_density : Γ/V en cm⁻³ s⁻¹

    Returns
    -------
    N_total (sans dimension)
    """
    return rate_density * V_HUBBLE_CM3 * T_HUBBLE_S


# ── Température critique ──────────────────────────────────────────────

def critical_temperature(R_nu_fm: float) -> float:
    """Température critique T_crit = ℏc/(2.82 k_B R) en Kelvin.

    C'est la température à laquelle le pic de Planck (loi de Wien)
    coïncide avec E_match = ℏc/R.

    Parameters
    ----------
    R_nu_fm : rayon du neutrino en fm

    Returns
    -------
    T_crit en K
    """
    E_match_eV = HBAR_C_MEV_FM / R_nu_fm * 1e6  # MeV → eV
    return E_match_eV / (2.82 * K_B_EV_PER_K)


def critical_redshift(T_crit_K: float) -> float:
    """Redshift correspondant à T_crit : z = T_crit/T_CMB − 1.

    Parameters
    ----------
    T_crit_K : température critique en K

    Returns
    -------
    z (redshift)
    """
    return T_crit_K / T_CMB - 1.0


# ── Seuil de masse pour entrelac ──────────────────────────────────────

def linking_rate_per_neutrino(m_nu_eV: float, T_K: float) -> float:
    """Taux d'interaction par neutrino individuel Γ_ν (s⁻¹).

    Γ_ν = n_γ(E_match, T) · σ · c

    Parameters
    ----------
    m_nu_eV : masse du neutrino en eV
    T_K : température de l'environnement en K

    Returns
    -------
    Γ_ν en s⁻¹
    """
    R_fm = boomerang_radius(m_nu_eV * 1e-6)
    E_match_eV = 2.0 * m_nu_eV  # ℏc/R = 2 m_ν c²
    dn = planck_spectral_density(E_match_eV, T_K)
    sigma_cm2 = linking_cross_section(R_fm) * 1e-26
    return dn * sigma_cm2 * C_CM_PER_S


def linking_onset_mass(T_K: float = T_CMB) -> float:
    """Masse seuil m_onset en eV : un neutrino forme un Hopf link
    exactement une fois pendant l'âge de l'univers (Γ_ν = 1/t_H).

    Pour T = T_CMB, m_onset ≈ 8.7 meV — remarquablement proche de
    √(Δm²_sol) ≈ 8.7 meV (masse minimale de ν₂).

    Au-dessous de m_onset, l'entrelac est cosmologiquement probable.
    Au-dessus, il est exponentiellement supprimé.

    Parameters
    ----------
    T_K : température de l'environnement en K (défaut = T_CMB)

    Returns
    -------
    m_onset en eV
    """
    target = 1.0 / T_HUBBLE_S
    lo, hi = 1e-7, 1.0  # eV
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        rate = linking_rate_per_neutrino(mid, T_K)
        if rate > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def linking_mean_free_path_neutrino(m_nu_eV: float, T_K: float) -> float:
    """Libre parcours moyen d'un neutrino avant formation d'un Hopf link (cm).

    l = 1 / (n_γ(E_match, T) · σ)

    Parameters
    ----------
    m_nu_eV : masse du neutrino en eV
    T_K : température en K

    Returns
    -------
    l en cm (inf si pas de photons disponibles)
    """
    R_fm = boomerang_radius(m_nu_eV * 1e-6)
    E_match_eV = 2.0 * m_nu_eV
    dn = planck_spectral_density(E_match_eV, T_K)
    sigma_fm2 = linking_cross_section(R_fm)
    return mean_free_path(dn, sigma_fm2)


# ── Propriétés du Hopf link ──────────────────────────────────────────

def hopf_link_properties(R_nu_fm: float) -> dict:
    """Propriétés du lien de Hopf formé entre neutrino et photon.

    Le Hopf link est l'entrelacs de deux inconnus 0₁ avec linking
    number = 1. C'est l'entrelacs le plus simple possible.

    Propriétés :
      - linking_number = 1 (invariant topologique)
      - composantes = 2 × unknot (0₁)
      - ρ_J(0₁) = 0.1 (facteur de rétractation minimal)
      - masse estimée via Cornell k=1 :
        m ≈ η · [2 · M(0₁) + λ · 2 · ρ_J(0₁) · F(1) / R_0]
      - stabilité : topologiquement stable (linking number ≠ 0)

    Parameters
    ----------
    R_nu_fm : rayon du neutrino en fm

    Returns
    -------
    dict avec clés : linking_number, components, rho_J, mass_MeV,
                     sigma_fm2, E_match_MeV, T_crit_K, stable, reason
    """
    # Invariants topologiques du Hopf link
    rho_J_unknot = 0.1  # ρ_J(0₁) minimal

    # Masse via P12 (Cornell topologique, k=1)
    # M(0₁) = α · ρ_J = 1.0 · 0.1 = 0.1
    # F(1) = 1
    # m = η · [2 · 0.1 + λ · 2 · 0.1 · 1 / R_0]
    #   = 0.1407 · [0.2 + 1.0 · 0.2 / 10.0]
    #   = 0.1407 · 0.22 = 0.030954 MeV
    eta = 0.1407  # MeV, calibration factor
    lambda_link = 1.0
    R_0 = 10.0
    M_unknot = 0.1  # α · ρ_J(0₁)
    F_1 = 1  # Fibonacci(1)

    mass_MeV = eta * (2 * M_unknot +
                      lambda_link * 2 * rho_J_unknot * F_1 / R_0)

    sigma = linking_cross_section(R_nu_fm)
    E_match = spectral_matching_energy(R_nu_fm)
    T_crit = critical_temperature(R_nu_fm)

    return {
        "linking_number": 1,
        "components": ["0_1", "0_1"],
        "rho_J": rho_J_unknot,
        "mass_MeV": mass_MeV,
        "sigma_fm2": sigma,
        "E_match_MeV": E_match,
        "T_crit_K": T_crit,
        "stable": True,
        "reason": "Linking number ≠ 0 → topologiquement stable (P15)",
    }
