"""
V5.10b — Précession de périhélie post-newtonienne PN-1 (résultat GR standard).

PÉRIMÈTRE HONNÊTE — À LIRE EN PREMIER
-------------------------------------
Ce module implémente la formule fermée GR standard de précession du
périhélie au premier ordre post-newtonien :

    Δφ_rev = 6 π G M / (c² a (1 - e²))

C'est un résultat classique de relativité générale (Einstein 1915,
MTW §40, Will "Theory and Experiment in Gravitational Physics" §7.3).

V5 HÉRITE ce résultat à travers la dérivation V5.8 de l'équation
d'Einstein depuis le χ-field (theory/einstein_emergence.py). Reproduire
les 42,98″/siècle de Mercure NE valide PAS une spécificité TQNT : cela
valide la COHÉRENCE V5/GR dans le régime faible champ + 1PN. Aucune
prédiction unique à V5 n'est revendiquée ici.

Référence catalogue planétaire (a, e, période) :
  - Will, C. M. (2014) "The Confrontation between General Relativity
    and Experiment", Living Rev. Relativ. 17, 4.
  - JPL Horizons / NASA fact sheets pour a, e, T.

Lancement direct (auto-test rapide) :
    PYTHONPATH=. python3 theory/post_newtonian_PN1.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

# Constantes SI (CODATA 2018, cohérent avec theory/einstein_emergence.py)
G_NEWTON = 6.67430e-11      # m³ kg⁻¹ s⁻²
C_LIGHT = 2.99792458e8      # m/s
M_SUN = 1.98892e30          # kg
AU = 1.495978707e11         # m
JULIAN_YEAR = 365.25 * 86400.0  # s
ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi
CENTURY = 100.0 * JULIAN_YEAR  # s


@dataclass(frozen=True)
class PlanetData:
    """Paramètres orbitaux d'une planète.

    a : demi-grand axe (m)
    e : excentricité (sans dimension)
    T : période orbitale sidérale (s)
    observed_arcsec_per_century : précession GR observée (réf. Will 2014)
    """
    name: str
    a: float
    e: float
    T: float
    observed_arcsec_per_century: float


# Catalogue. Valeurs a, e, T : NASA planetary fact sheets.
# Valeurs observed_arcsec_per_century : Will (2014) Table 2 et
# Nyambuya (2010) Mon. Not. R. Astron. Soc. 403, 1381 pour la précession
# GR théorique (les mesures observées sont compatibles aux barres d'erreur).
PLANETS: Dict[str, PlanetData] = {
    "Mercury": PlanetData(
        name="Mercury",
        a=0.387098 * AU,
        e=0.205630,
        T=87.9691 * 86400.0,
        observed_arcsec_per_century=42.98,
    ),
    "Venus": PlanetData(
        name="Venus",
        a=0.723332 * AU,
        e=0.006772,
        T=224.701 * 86400.0,
        observed_arcsec_per_century=8.62,
    ),
    "Earth": PlanetData(
        name="Earth",
        a=1.000000 * AU,
        e=0.016710,
        T=365.256 * 86400.0,
        observed_arcsec_per_century=3.84,
    ),
    "Mars": PlanetData(
        name="Mars",
        a=1.523679 * AU,
        e=0.093400,
        T=686.971 * 86400.0,
        observed_arcsec_per_century=1.35,
    ),
    "Jupiter": PlanetData(
        name="Jupiter",
        a=5.20260 * AU,
        e=0.048498,
        T=4332.589 * 86400.0,
        observed_arcsec_per_century=0.0623,
    ),
    "Saturn": PlanetData(
        name="Saturn",
        a=9.55491 * AU,
        e=0.05555,
        T=10759.22 * 86400.0,
        observed_arcsec_per_century=0.0136,
    ),
}


def perihelion_precession_per_revolution(M: float, a: float, e: float) -> float:
    """Précession PN-1 du périhélie par révolution, en radians.

    Δφ = 6 π G M / (c² a (1 - e²))

    M : masse centrale (kg)
    a : demi-grand axe (m)
    e : excentricité

    Référence : Einstein (1915) ; MTW §40.5 eq. (40.17).
    """
    if a <= 0:
        raise ValueError("a must be > 0")
    if not (0.0 <= e < 1.0):
        raise ValueError("e must be in [0, 1)")
    return 6.0 * math.pi * G_NEWTON * M / (C_LIGHT ** 2 * a * (1.0 - e ** 2))


def perihelion_precession_arcsec_per_century(
    M: float, a: float, e: float, T: float
) -> float:
    """Précession PN-1 du périhélie en secondes d'arc par siècle.

    M : masse centrale (kg)
    a : demi-grand axe (m)
    e : excentricité
    T : période orbitale (s)

    Δφ_siècle = Δφ_rev × (siècle / T) × (180 × 3600 / π)
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    delta_phi_rev = perihelion_precession_per_revolution(M, a, e)
    revolutions_per_century = CENTURY / T
    return delta_phi_rev * revolutions_per_century * ARCSEC_PER_RAD


def validate_mercury() -> Dict[str, float]:
    """Reproduit la précession GR du périhélie de Mercure (42,98″/siècle).

    Retourne un dict avec la valeur prédite et l'écart relatif à la valeur
    observée Will (2014).
    """
    p = PLANETS["Mercury"]
    predicted = perihelion_precession_arcsec_per_century(M_SUN, p.a, p.e, p.T)
    return {
        "predicted_arcsec_per_century": predicted,
        "observed_arcsec_per_century": p.observed_arcsec_per_century,
        "relative_error": abs(predicted - p.observed_arcsec_per_century)
        / p.observed_arcsec_per_century,
    }


def validate_all_planets() -> List[Dict[str, float]]:
    """Reproduit la précession GR du périhélie pour les 6 planètes du catalogue.

    Retourne une liste de dicts par planète. Inclut Mercure, Vénus, Terre,
    Mars, Jupiter, Saturne.
    """
    results: List[Dict[str, float]] = []
    for name, p in PLANETS.items():
        predicted = perihelion_precession_arcsec_per_century(M_SUN, p.a, p.e, p.T)
        results.append(
            {
                "name": name,
                "a_AU": p.a / AU,
                "e": p.e,
                "T_days": p.T / 86400.0,
                "predicted_arcsec_per_century": predicted,
                "observed_arcsec_per_century": p.observed_arcsec_per_century,
                "relative_error": abs(predicted - p.observed_arcsec_per_century)
                / p.observed_arcsec_per_century,
            }
        )
    return results


if __name__ == "__main__":
    print("V5.10b — PN-1 précession du périhélie (formule GR standard)")
    print("=" * 70)
    print(f"{'Planet':<10} {'a (AU)':>8} {'e':>8} {'Predicted':>14} {'Observed':>14} {'Err':>8}")
    print("-" * 70)
    for r in validate_all_planets():
        print(
            f"{r['name']:<10} {r['a_AU']:>8.4f} {r['e']:>8.5f}"
            f" {r['predicted_arcsec_per_century']:>14.4f}"
            f" {r['observed_arcsec_per_century']:>14.4f}"
            f" {100.0 * r['relative_error']:>7.2f}%"
        )
    print("-" * 70)
    print("Note : valeurs en secondes d'arc par siècle.")
    print("Cf. Will, Living Rev. Relativ. 17, 4 (2014).")
