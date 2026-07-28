"""
V5p27 (nouveau) — Reconnexion topologique P15 et oscillations de mésons

Cadre théorique (CLAUDE.md règle 0) :
La « saveur » (S, C, B', T) du Modèle Standard est un artefact de
comptabilité, PAS un nombre quantique fondamental TQNT. La « chiralité
Dirac » (handedness L/R sur les spineurs de Weyl) est également un
artefact SM : en TQNT, la chiralité signifie l'opération miroir
V_K*(t) = V_K(t⁻¹) sur les invariants topologiques, sans rapport avec
la handedness des spineurs.

Le vrai phénomène physique que le SM indexe par « oscillation de
saveur » est :

    Oscillations K⁰ ↔ K̄⁰, B⁰ ↔ B̄⁰, D⁰ ↔ D̄⁰

La TQNT V5 le dérive comme **reconnexion topologique P15** entre un
entrelacs L et son entrelacs miroir L*.

Note : la matrice CKM mélange des états propres de SAVEUR (artefact SM)
et la phase δ_CP confondait la chiralité Dirac avec l'opération miroir
topologique. Aucun des deux n'est dérivé ici (cf. CLAUDE.md règle 0).

Formule maîtresse (P15, V5 chap. 6/13) :

    Γ_reconnex(L) = Γ_P × exp(-λ × d) × M_raw(L)^(-ν)

avec Γ_P calibré sur Δm_K0, ν_osc = 1, λ_osc = 1 (tunnel cohérent).
La différence de masse d'oscillation est alors :

    Δm = ℏ × Γ_reconnex(L)        (ℏ = 6.582e-22 MeV·s)

Pour un entrelacs amphichiral (e.g. K⁰ = 6_1 + 6_1 + 6_1), L = L*
topologiquement : la reconnexion passe par un état intermédiaire de
suspension. L'oscillation est encore plus lente.

Référence : docs/TQNT_detaillee_V5_FR.tex chap. 6 (lifetimes), chap. 13
(P15 fragility), CLAUDE.md règle 0.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from theory.jones import (
    jones_polynomial,
    mirror_jones,
    is_amphicheiral,
    JONES_KNOT_NAMES,
)
from theory.retraction import retraction_factor
from theory.multi_knot import cornell_equilibrium, ETA, SM_ASSIGNMENTS
from theory.topological_lifetime import (
    NU_WEAK_ALG,
    LAMBDA_WEAK_ALG,
    _calibrate_algebraic,
)
import theory.topological_lifetime as _tl

# --- Constantes physiques ---
HBAR_MEV_S = 6.582119569e-22   # MeV·s
H_BAR_GEV_S = 6.582119569e-25  # GeV·s

# --- Assignations V5 des mésons neutres (catalogue diffusion) ---
# Source : docs/TQNT_catalog_V5_diffusion.xlsx onglet « 3. Particules PDG »
MESON_OSCILLATION_ASSIGNMENTS = {
    # nom : (knots, k Fibonacci, masse PDG MeV)
    "K0":   {"knots": ["6_1", "6_1", "6_1"], "k": 11, "mass_pdg": 497.611},
    "D0":   {"knots": ["3_1", "5_1", "7_2"], "k": 18, "mass_pdg": 1864.84},
    "B0":   {"knots": ["3_1", "6_2", "7_7"], "k": 22, "mass_pdg": 5279.66},
    "Bs0":  {"knots": ["6_3", "7_6"],        "k": 22, "mass_pdg": 5366.92},
}

# --- Mesures PDG (sources : PDG 2024) ---
DELTA_M_PDG = {
    # Δm en MeV (PDG)
    "K0":  3.484e-12,    # Δm_K = m(K_L) - m(K_S)
    "D0":  6.25e-12,     # Δm_D (= x_D × Γ_D), x_D = 0.0041
    "B0":  3.334e-10,    # Δm_d (oscillation B0_d)
    "Bs0": 1.169e-8,     # Δm_s (B_s oscillation, rapide)
}

# --- Opération miroir topologique TQNT (≠ chiralité Dirac) ---
# En TQNT, la « chiralité » d'un nœud K désigne strictement l'opération
# miroir V_K*(t) = V_K(t⁻¹). Un nœud est dit amphichiral si V_K = V_K*
# (e.g. 4_1, 6_3, 8_3) ; sinon il est chiral et distinct de son miroir.
# Ceci N'EST PAS la handedness Dirac L/R des spineurs de Weyl, qui est
# un artefact du SM (CLAUDE.md règle 0).
def chirality_sigma(knot_name: str) -> int:
    """Indice de miroir topologique σ(K).

    σ = 0  si K = mirror(K) (amphichiral, e.g. 4_1, 6_1, 6_3, 8_3)
    σ = ±1 sinon (chiral au sens topologique : K ≠ K*).
    Le signe ±1 vient du sgn(min(power) + max(power)) du Jones
    (lopsidedness), il n'a aucun rapport avec la handedness Dirac.
    """
    poly = jones_polynomial(knot_name)
    if is_amphicheiral(poly):
        return 0
    powers = list(poly.keys())
    asym = min(powers) + max(powers)
    return int(np.sign(asym)) if asym != 0 else 1


def link_chirality_index(knot_names: list[str]) -> int:
    """Indice de miroir topologique du link L : Σσ(K_i).

    L est amphichiral globalement ssi Σσ = 0 ET chaque composante est
    amphichirale ou s'apparie avec son miroir.
    """
    return sum(chirality_sigma(n) for n in knot_names)


def is_link_amphichiral(knot_names: list[str]) -> bool:
    """L = L* (link amphichiral au sens topologique) ssi toutes
    composantes amphichirales, ou chiralités s'annulent exactement entre
    composantes équivalentes."""
    sigmas = [chirality_sigma(n) for n in knot_names]
    if all(s == 0 for s in sigmas):
        return True
    # Si non, on requiert appariement strict ±1
    return False  # critère strict : seul l'amphichiralité pure compte


# --- Distance topologique entre L et L* (nombre de croisements à inverser) ---
def topological_distance_to_mirror(knot_names: list[str]) -> int:
    """d(L, L*) = nombre de croisements à inverser pour transformer L en L*.

    Pour une composante K_i chirale : d(K_i, K_i*) = c(K_i) (nombre de
    croisements). Pour une composante amphichirale : d = 0 (mais une
    reconnexion topologique reste possible via suspension Reidemeister).

    Distance link totale : Σ d(K_i, K_i*).
    Pour amphichiral : on assigne d = 2 (suspension Reidemeister, P15
    régime « tunnel topologique »).
    """
    from theory.knot_catalog import crossing_number
    d = 0
    for n in knot_names:
        if is_amphicheiral(jones_polynomial(n)):
            d += 2  # tunnel Reidemeister minimal
        else:
            d += crossing_number(n)
    return d


# --- Calibration P15 oscillation (calibrée sur K0) ---
# Le rate brut de la formule τ standard (avec n) est trop suppression :
# pour les oscillations, le facteur de barrière topologique est d (distance
# au miroir) et non n. De plus, le mécanisme d'oscillation est un TUNNEL
# COHÉRENT (pas une désintégration), donc Γ_osc ≪ Γ_decay.
#
# On calibre Γ_P_osc sur K0 (Δm_K = 3.484e-12 MeV → Γ_osc^K0 = 5.29e9 s^-1)
# puis on prédit Δm_D, Δm_B0, Δm_Bs0.
_GAMMA_P_OSC = None


def _calibrate_oscillation_rate():
    """Calibre Γ_P_osc sur Δm_K0 (PDG 3.484e-12 MeV)."""
    global _GAMMA_P_OSC
    if _GAMMA_P_OSC is not None:
        return
    # K0 = 6_1 + 6_1 + 6_1, k=11
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    eq = cornell_equilibrium(cfg["knots"], cfg["k"])
    M_raw = eq["mass_raw"]
    d = topological_distance_to_mirror(cfg["knots"])
    # Γ_osc = Γ_P_osc × exp(-λ × d) × M_raw^(-ν_osc)
    # Choix : ν_osc = 1 (linéaire), λ_osc = 1 (distance topologique pure)
    # → calibrage unique sur K0
    target_gamma = DELTA_M_PDG["K0"] / HBAR_MEV_S
    nu_osc = 1.0
    lam_osc = 1.0
    _GAMMA_P_OSC = target_gamma * M_raw ** nu_osc / np.exp(-lam_osc * d)


# Exposants oscillation P15 (distincts de la durée de vie)
NU_OSC = 1.0     # Γ_osc ∝ 1/M_raw (au lieu de 1/M^7.85 pour la désintégration)
LAMBDA_OSC = 1.0 # Barrière unitaire par croisement à inverser


# --- Taux de reconnexion P15 ---
def reconnection_rate(knot_names: list[str], k: int) -> dict:
    """Taux de reconnexion topologique L → L* (P15).

    Γ_reconnex = Γ_P_osc × exp(-λ_osc × d(L, L*)) × M_raw^(-ν_osc)

    où d(L, L*) est la distance topologique au miroir et Γ_P_osc est
    calibré sur Δm_K0 (PDG). ν_osc = 1, λ_osc = 1 pour l'oscillation
    (cohérent tunnel vs. désintégration ν=7.85, λ=5).

    Returns dict : rate (s^-1), M_raw, d_mirror, channel, n_components.
    """
    _calibrate_oscillation_rate()
    eq = cornell_equilibrium(knot_names, k)
    M_raw = eq["mass_raw"]
    d = topological_distance_to_mirror(knot_names)
    rate = _GAMMA_P_OSC * np.exp(-LAMBDA_OSC * d) * M_raw ** (-NU_OSC)
    return {
        "rate": rate,
        "M_raw": M_raw,
        "d_mirror": d,
        "n_components": len(knot_names),
        "channel": "weak",
        "nu": NU_OSC,
        "lambda": LAMBDA_OSC,
    }


def delta_m_oscillation(knot_names: list[str], k: int) -> dict:
    """Différence de masse d'oscillation Δm = ℏ × Γ_reconnex.

    Returns dict : delta_m_MeV, rate, M_raw, d_mirror, n_components.
    """
    r = reconnection_rate(knot_names, k)
    delta_m = HBAR_MEV_S * r["rate"]   # MeV
    return {
        "delta_m_MeV": delta_m,
        **r,
    }


# --- Table de comparaison oscillations V5 vs PDG ---
def oscillation_comparison_table() -> list[dict]:
    """Table de comparaison Δm V5 vs PDG pour K0, D0, B0, Bs0.

    Returns list of dicts : name, delta_m_v5, delta_m_pdg, log_ratio,
    factor (V5/PDG).
    """
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_oscillation(cfg["knots"], cfg["k"])
        dm_v5 = res["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        log_ratio = np.log10(dm_v5 / dm_pdg) if dm_v5 > 0 else None
        rows.append({
            "name": name,
            "knots": cfg["knots"],
            "k": cfg["k"],
            "mass_pdg_MeV": cfg["mass_pdg"],
            "delta_m_v5_MeV": dm_v5,
            "delta_m_pdg_MeV": dm_pdg,
            "log_ratio": log_ratio,
            "factor_v5_over_pdg": dm_v5 / dm_pdg if dm_pdg > 0 else None,
            "d_mirror": res["d_mirror"],
            "M_raw": res["M_raw"],
            "amphichiral": is_link_amphichiral(cfg["knots"]),
        })
    return rows
