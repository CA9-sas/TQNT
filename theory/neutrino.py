"""Théorie Nodale — Postulat P18 : Neutrinos comme reliquats ondulatoires

Hypothèse : lors de la cassure d'un nœud, la partie ondulatoire du champ χ
peut se détacher sous forme d'une onde longitudinale qui tourne sur elle-même.
Ce reliquat ondulatoire n'a pas de topologie (pas de croisements), d'où :
  - charge nulle (pas de charge topologique)
  - spin 1/2 (rotation longitudinale)

N'importe quel nœud peut être parent d'un neutrino. La saveur n'est pas
liée à un nœud spécifique ni à un axe spatial — c'est un effet
d'observation.

--- Masse par inertie rotationnelle ---

Le neutrino n'a aucune masse propre (pas de topologie). Sa masse provient
uniquement de l'inertie de sa rotation : l'onde tourne sur elle-même comme
un bout de corde retenu à une extrémité. La tension centripète de cette
auto-rotation crée une inertie effective qui se manifeste comme masse.

Formellement, l'onde (intrinsèquement sans masse) tourne à v = c sur un
rayon R. La quantification du spin impose :

    L = ℏ/2 = p R      →  p = ℏ/(2R)
    E = pc = ℏc/(2R)    →  m_ν = E/c² = ℏ/(2Rc)

La masse EST l'énergie rotationnelle : m_ν c² = ℏc/(2R).
Le rayon R est la seule grandeur qui détermine la masse.

--- Vitesse boomerang ---

Puisque l'onde tourne à v = c sur son rayon :

    ω = c / R_ν
    R_ν = ℏc / (2 m_ν c²)     (demi-longueur d'onde Compton réduite)
    v = ω × R_ν = c            (le neutrino est naturellement luminal)

Résultat : v = c exactement pour toute masse.

--- Oscillation de saveur ---

L'oscillation entre saveurs provient de la courbure gravitationnelle
(métrique émergente g_ij = (1 + κ|Ψ|²)δ_ij) et non d'une précession
entre axes spatiaux. Les formules PMNS standard sont conservées :

    Δm² = m₁² - m₂²
    L_osc = 4π ħc E / |Δm²|

--- Cassure de nœud et désintégration bêta ---

Tout nœud peut être parent d'un neutrino. Lors de la cassure d'un entrelacs
(ex. désintégration bêta du neutron), chaque extrémité de la cassure produit
un fragment :
  - une extrémité se replie sur elle-même → nœud simple (électron = 3₁)
  - l'autre extrémité tourne longitudinalement → reliquat ondulatoire (ν̄e)

Exemple : désintégration bêta du neutron n → p + e⁻ + ν̄e.
Le neutron (entrelacs 9₁+7₄+6₃) casse à une jonction. L'énergie de la
différence de masse (Δm = 1.3 MeV) se matérialise aux deux extrémités :
l'une se replie en 3₁ (électron), l'autre tourne en ν̄e.
"""
from __future__ import annotations

import math


# --- Constantes physiques ---

HBAR_C_MEV_FM = 197.3269804  # ℏc en MeV·fm
C_FM_PER_S = 2.99792458e23   # c en fm/s


# --- Noms de saveurs (utiles pour l'affichage, pas liés à des nœuds) ---

FLAVOR_NAMES = {
    1: "nu_e",
    2: "nu_mu",
    3: "nu_tau",
}

FLAVOR_SYMBOLS = {
    1: "νe",
    2: "νμ",
    3: "ντ",
}


# --- Masse par inertie rotationnelle ---


def mass_from_radius(radius_fm: float) -> float:
    """Masse du neutrino à partir de son rayon de rotation (MeV).

    m_ν = ℏc / (2 R c²) = ℏc / (2 R)   [en unités MeV avec R en fm]

    La masse provient uniquement de l'inertie rotationnelle : une onde
    sans masse tournant à v = c sur un rayon R acquiert une masse
    effective m = ℏ/(2Rc) par quantification du spin.

    Parameters
    ----------
    radius_fm : rayon de rotation en fm (> 0)

    Returns
    -------
    Masse en MeV
    """
    if radius_fm <= 0:
        raise ValueError(f"Rayon invalide : {radius_fm} (doit être > 0)")
    return HBAR_C_MEV_FM / (2.0 * radius_fm)


# --- Vitesse boomerang ---


def boomerang_radius(mass_mev: float) -> float:
    """Rayon d'auto-rotation du neutrino (fm).

    R_ν = ℏc / (2 m_ν)

    C'est la demi-longueur d'onde Compton réduite. Le neutrino tourne
    sur un cercle de ce rayon. C'est aussi la relation inverse de
    mass_from_radius : R et m sont deux faces de la même grandeur.

    Parameters
    ----------
    mass_mev : masse du neutrino en MeV (> 0)

    Returns
    -------
    R_ν en fm
    """
    if mass_mev <= 0:
        raise ValueError(f"Masse invalide : {mass_mev} (doit être > 0)")
    return HBAR_C_MEV_FM / (2.0 * mass_mev)


def boomerang_omega(mass_mev: float) -> float:
    """Vitesse angulaire d'auto-rotation du neutrino (s⁻¹).

    ω = c / R_ν = 2 m_ν c² / ℏ

    Parameters
    ----------
    mass_mev : masse du neutrino en MeV

    Returns
    -------
    ω en s⁻¹
    """
    R = boomerang_radius(mass_mev)
    return C_FM_PER_S / R


def boomerang_velocity(mass_mev: float) -> float:
    """Vitesse de translation du neutrino par auto-rotation (en unités de c).

    v = ω × R_ν = c

    Le neutrino est intrinsèquement luminal car son rayon d'auto-rotation
    est sa demi-longueur Compton réduite. Le résultat est exactement 1.0
    (v = c) pour toute masse — c'est une identité analytique.

    Parameters
    ----------
    mass_mev : masse du neutrino en MeV (> 0)

    Returns
    -------
    v/c = 1.0 (exactement)
    """
    # Dérivation : v = ω R = (c/R) × R = c
    # Pas besoin de calcul numérique, c'est une identité.
    if mass_mev <= 0:
        raise ValueError(f"Masse invalide : {mass_mev} (doit être > 0)")
    return 1.0


# --- Oscillations ---


def oscillation_delta_m2(m1_mev: float, m2_mev: float) -> float:
    """Δm² entre deux neutrinos de masses différentes (MeV²).

    Δm² = m₁² - m₂²

    L'oscillation de saveur résulte de la courbure gravitationnelle
    (métrique émergente) et non d'une précession entre axes spatiaux.
    Les formules PMNS standard s'appliquent.

    Parameters
    ----------
    m1_mev, m2_mev : masses des deux neutrinos en MeV

    Returns
    -------
    Δm² = m₁² - m₂² en MeV²
    """
    return m1_mev ** 2 - m2_mev ** 2


def oscillation_length(m1_mev: float, m2_mev: float, E_mev: float) -> float:
    """Longueur d'oscillation entre deux neutrinos de masses m₁ et m₂ (en fm).

    L_osc = 4π ħc E / |Δm²|

    La longueur d'oscillation dépend de l'énergie du neutrino et de la
    différence de masse quadratique. L'effet physique provient de la
    courbure gravitationnelle de la métrique émergente.

    Parameters
    ----------
    m1_mev, m2_mev : masses en MeV
    E_mev : énergie du neutrino (MeV)

    Returns
    -------
    L_osc en fm (1 fm = 10⁻¹⁵ m). Retourne inf si m1 == m2.
    """
    dm2 = abs(oscillation_delta_m2(m1_mev, m2_mev))
    if dm2 < 1e-30:
        return float('inf')
    return 4 * math.pi * HBAR_C_MEV_FM * E_mev / dm2


def oscillation_frequency(m1_mev: float, m2_mev: float, E_mev: float) -> float:
    """Fréquence d'oscillation entre deux neutrinos (MeV).

    ω = |Δm²| / (2E)

    L'effet provient de la courbure gravitationnelle de la métrique
    émergente, pas d'une précession entre axes spatiaux.

    Parameters
    ----------
    m1_mev, m2_mev : masses en MeV
    E_mev : énergie du neutrino (MeV)

    Returns
    -------
    ω en MeV (unités naturelles ħ = 1)
    """
    dm2 = abs(oscillation_delta_m2(m1_mev, m2_mev))
    return dm2 / (2 * E_mev)
