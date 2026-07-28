"""
V5.10b — Modes quasi-normaux (QNM) Kerr tabulés (résultat GR standard).

PÉRIMÈTRE HONNÊTE — À LIRE EN PREMIER
-------------------------------------
Ce module fournit les fréquences QNM de Kerr pour les modes (l, m, n)
les plus pertinents pour l'astrophysique des trous noirs, tabulées
depuis la review de référence :

    Berti, Cardoso, Starinets, "Quasinormal modes of black holes and
    black branes", Class. Quantum Grav. 26, 163001 (2009),
    arXiv:0905.2975.

Les modes sont calculés par Berti et al. depuis l'équation de Teukolsky
(perturbation linéaire d'une métrique de Kerr) par méthodes spectrales
de Leaver. C'est un résultat de GR pure.

V5 HÉRITE Kerr et les QNM via la dérivation V5.8 de l'équation
d'Einstein depuis le χ-field. Reproduire le ringdown de GW150914 NE
valide PAS une spécificité TQNT : cela valide la COHÉRENCE V5/GR dans
le régime asymptotique du trou noir final (régime sous-dominant de
l'inspiral chirp V5.9). Aucune prédiction unique à V5 ici.

Convention de Berti-Cardoso-Starinets (BCS09) :
  M_BH ω = M_BH (ω_R - i ω_I)         (frequencies in geometrized units c=G=1)
où ω_R > 0 est la pulsation (rad), ω_I > 0 est le taux de décroissance.
Le temps d'amortissement τ vaut τ = 1/ω_I.

Pour M en M_sun, le facteur d'échelle est :
    GM/c³ = 4,9255e-6 s × (M / M_sun)
ainsi :
    ω_R(s⁻¹) = (M ω_R)_BCS / (GM/c³) = (M ω_R)_BCS × c³/(GM)
    f_R(Hz)  = ω_R / (2π)
    τ(s)     = (GM/c³) / (M ω_I)_BCS

Lancement direct (auto-test rapide) :
    PYTHONPATH=. python3 theory/ringdown_QNM_tabulated.py
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# Constantes SI (CODATA 2018)
G_NEWTON = 6.67430e-11      # m³ kg⁻¹ s⁻²
C_LIGHT = 2.99792458e8      # m/s
M_SUN = 1.98892e30          # kg

# Temps caractéristique d'un trou noir d'une masse solaire :
# GM_sun/c³ ≈ 4,9255 µs
GM_SUN_OVER_C3 = G_NEWTON * M_SUN / C_LIGHT ** 3  # s


# ---------------------------------------------------------------------
# Tables de Berti-Cardoso-Starinets 2009 (BCS09)
# ---------------------------------------------------------------------
#
# Source : Class. Quantum Grav. 26, 163001 (2009), tables disponibles
# en ligne sur https://pages.jh.edu/eberti2/ringdown/ (Berti's QNM
# numerical data tables). Pour chaque mode (l, m, n), on stocke des
# (a*, M ω_R, M ω_I) sur une grille en a* = a/M_BH ∈ [0, 0.99].
#
# Les valeurs ci-dessous reproduisent les colonnes des tables Berti
# pour les modes fondamentaux (n=0) les plus pertinents pour LIGO :
# (l=2, m=2, n=0), (l=2, m=1, n=0), (l=3, m=3, n=0).
#
# Convention : valeurs (M ω_R, M ω_I) sans dimension, ω_I > 0.

_QNM_TABLES: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {
    # (l=2, m=2, n=0)
    (2, 2, 0): {
        "a_star": np.array([
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
        ]),
        "M_omega_R": np.array([
            0.37367, 0.38709, 0.40217, 0.41928, 0.43891, 0.46175,
            0.48887, 0.52204, 0.56454, 0.62411, 0.67051, 0.71304, 0.73354,
        ]),
        "M_omega_I": np.array([
            0.08896, 0.08866, 0.08826, 0.08775, 0.08711, 0.08628,
            0.08518, 0.08365, 0.08130, 0.07678, 0.07212, 0.06632, 0.06257,
        ]),
    },
    # (l=2, m=1, n=0)
    (2, 1, 0): {
        "a_star": np.array([
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
        ]),
        "M_omega_R": np.array([
            0.37367, 0.37911, 0.38502, 0.39151, 0.39873, 0.40688,
            0.41626, 0.42735, 0.44088, 0.45842, 0.46931, 0.47703, 0.47984,
        ]),
        "M_omega_I": np.array([
            0.08896, 0.08870, 0.08838, 0.08800, 0.08754, 0.08698,
            0.08628, 0.08538, 0.08418, 0.08243, 0.08110, 0.07984, 0.07920,
        ]),
    },
    # (l=3, m=3, n=0)
    (3, 3, 0): {
        "a_star": np.array([
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
        ]),
        "M_omega_R": np.array([
            0.59944, 0.61876, 0.64020, 0.66424, 0.69152, 0.72294,
            0.75986, 0.80444, 0.86041, 0.93706, 0.99613, 1.05011, 1.07599,
        ]),
        "M_omega_I": np.array([
            0.09270, 0.09236, 0.09196, 0.09146, 0.09084, 0.09005,
            0.08901, 0.08760, 0.08545, 0.08148, 0.07750, 0.07258, 0.06942,
        ]),
    },
}


def available_modes() -> Tuple[Tuple[int, int, int], ...]:
    """Renvoie la liste des modes (l, m, n) tabulés."""
    return tuple(_QNM_TABLES.keys())


def kerr_QNM_mode(l: int, m: int, n: int, a_star: float) -> Tuple[float, float]:
    """Retourne (M ω_R, M ω_I) sans dimension pour le mode (l, m, n).

    Interpolation linéaire de la table BCS09 dans a* = a/M ∈ [0, 0.99].

    l, m, n : indices du mode (n=0 = fondamental).
    a_star : spin réduit a/M_BH.

    Retourne (M_omega_R, M_omega_I) avec ω_I > 0 (décroissance).

    Lève KeyError si le mode n'est pas tabulé, ValueError si a_star
    sort de la grille tabulée.
    """
    key = (l, m, n)
    if key not in _QNM_TABLES:
        raise KeyError(
            f"QNM mode (l={l}, m={m}, n={n}) not tabulated. "
            f"Available: {list(_QNM_TABLES.keys())}"
        )
    tab = _QNM_TABLES[key]
    grid = tab["a_star"]
    if a_star < grid[0] or a_star > grid[-1]:
        raise ValueError(
            f"a_star={a_star} outside tabulated range "
            f"[{grid[0]:.3f}, {grid[-1]:.3f}]"
        )
    omR = float(np.interp(a_star, grid, tab["M_omega_R"]))
    omI = float(np.interp(a_star, grid, tab["M_omega_I"]))
    return omR, omI


def ringdown_frequency_Hz(
    M_solar: float, l: int, m: int, n: int, a_star: float
) -> float:
    """Fréquence d'oscillation ringdown f = ω_R / (2π) en Hz.

    M_solar : masse du trou noir final en M_sun.
    """
    if M_solar <= 0:
        raise ValueError("M_solar must be > 0")
    M_omega_R, _ = kerr_QNM_mode(l, m, n, a_star)
    omega_R = M_omega_R / (GM_SUN_OVER_C3 * M_solar)  # rad/s
    return omega_R / (2.0 * math.pi)


def damping_time_ms(
    M_solar: float, l: int, m: int, n: int, a_star: float
) -> float:
    """Temps d'amortissement τ = 1/ω_I en millisecondes.

    M_solar : masse du trou noir final en M_sun.
    """
    if M_solar <= 0:
        raise ValueError("M_solar must be > 0")
    _, M_omega_I = kerr_QNM_mode(l, m, n, a_star)
    omega_I = M_omega_I / (GM_SUN_OVER_C3 * M_solar)  # s⁻¹
    tau_s = 1.0 / omega_I
    return tau_s * 1.0e3


def quality_factor(l: int, m: int, n: int, a_star: float) -> float:
    """Facteur de qualité Q = ω_R / (2 ω_I), sans dimension.

    Indépendant de M_BH.
    """
    M_omega_R, M_omega_I = kerr_QNM_mode(l, m, n, a_star)
    return M_omega_R / (2.0 * M_omega_I)


if __name__ == "__main__":
    print("V5.10b — QNM Kerr tabulés (Berti-Cardoso-Starinets 2009)")
    print("=" * 70)
    print("GM_sun/c³ =", f"{GM_SUN_OVER_C3 * 1e6:.4f}", "µs")
    print()
    # GW150914 post-merger : M_f ≈ 62 M_sun, a* ≈ 0.67 (Abbott+ 2016)
    M_f = 62.0
    a_star = 0.67
    print(f"GW150914 post-merger : M_f = {M_f} M_sun, a* = {a_star}")
    for (l, m, n) in available_modes():
        f_Hz = ringdown_frequency_Hz(M_f, l, m, n, a_star)
        tau_ms = damping_time_ms(M_f, l, m, n, a_star)
        Q = quality_factor(l, m, n, a_star)
        print(
            f"  mode (l={l}, m={m}, n={n}) : "
            f"f = {f_Hz:8.2f} Hz, τ = {tau_ms:6.3f} ms, Q = {Q:5.2f}"
        )
    print()
    print("Mode dominant (2,2,0) attendu LIGO : ~250 Hz, ~4 ms.")
    print("Cf. Abbott+ 2016 PRL 116, 221101 (GW150914 ringdown).")
