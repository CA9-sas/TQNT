"""Théorie Nodale — Durée de vie algébrique (Cas 4 : instabilité structurelle)

Formulation algébrique de P15 avec exposants dérivés de la topologie :

    Canal faible  : ν = 5π/2 ≈ 7.854,  λ = 5
    Canal fort/EW : ν = 7/4  = 1.75,    λ = 7/4

L'indice de stabilité structurelle S(K) = ρ_J(K)² / U(K) compare la
rigidité topologique (ρ_J) à la tension géométrique (U = ∫(κ²+τ²)ds).

Résultat : RMS ≈ 0.239 décades (quasi-identique à P15 = 0.240), base
conceptuelle plus solide (π apparaît naturellement dans la courbure,
7/4 = composantes W).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from theory.jones import jones_polynomial, JONES_KNOT_NAMES
from theory.retraction import retraction_factor
from theory.abraham import geometric_integrals, REFERENCE_N50K
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium, ETA


# --- Exposants algébriques ---
NU_WEAK_ALG = 5 * np.pi / 2        # ≈ 7.854 (P15 : 7.75)
LAMBDA_WEAK_ALG = 5.0               # exposant fragilité, canal faible
NU_STRONG_ALG = Fraction(7, 4)      # = 1.75 exact
LAMBDA_STRONG_ALG = Fraction(7, 4)  # = 1.75 exact

# --- Constantes dupliquées de multi_knot (éviter couplage) ---
TAU_MU = 2.197e-6       # secondes (durée de vie du muon)
TAU_W = 3.157e-25        # secondes (durée de vie du W)
STRONG_DECAY = {"W+", "Z0", "H", "t"}

# --- Données expérimentales ---
TAU_EXP_DATA = {
    "e-": None, "u": None, "nu_tau": None,
    "d": 878.4, "s": 1.238e-8, "mu-": 2.197e-6,
    "c": 1.040e-12, "tau-": 2.906e-13, "b": 1.519e-12,
    "W+": 3.157e-25, "Z0": 2.642e-25, "H": 1.624e-22, "t": 5.0e-25,
}

# --- Calibration (lazy) ---
_TAU_P_WEAK_ALG = None
_TAU_P_STRONG_ALG = None


def structural_stability(knot_name: str, n_points: int = 50000) -> dict:
    """Indice de stabilité structurelle S(K) = ρ_J(K)² / U(K).

    U(K) = ∫(κ² + τ²)ds est la tension géométrique totale.
    Utilise REFERENCE_N50K si disponible (n_points=50000), sinon calcule.

    Returns dict: rho_J, U, S_K
    """
    poly = jones_polynomial(knot_name)
    rho_J = retraction_factor(poly)

    # U from reference table or computation
    if n_points == 50000 and knot_name in REFERENCE_N50K:
        sqrt_U = REFERENCE_N50K[knot_name]["sqrt_U"]
        U = sqrt_U ** 2
    else:
        gi = geometric_integrals(knot_name, n_points)
        U = gi["U"]

    S_K = rho_J ** 2 / U if U > 0 else float("inf")

    return {
        "rho_J": rho_J,
        "U": U,
        "S_K": S_K,
    }


def link_structural_stability(
    knot_names: list[str], n_points: int = 50000
) -> dict:
    """S_link = (Σ ρ_J)² / (Σ U) pour un entrelacs multi-nœuds.

    Returns dict: rho_total, U_total, S_link, n_components, per_knot
    """
    per_knot = {}
    rho_total = 0.0
    U_total = 0.0

    for name in knot_names:
        if name not in per_knot:
            per_knot[name] = structural_stability(name, n_points)
        rho_total += per_knot[name]["rho_J"]
        U_total += per_knot[name]["U"]

    S_link = rho_total ** 2 / U_total if U_total > 0 else float("inf")

    return {
        "rho_total": rho_total,
        "U_total": U_total,
        "S_link": S_link,
        "n_components": len(knot_names),
        "per_knot": per_knot,
    }


def _calibrate_algebraic():
    """Calibre les τ_P pour les deux canaux algébriques (lazy)."""
    global _TAU_P_WEAK_ALG, _TAU_P_STRONG_ALG
    if _TAU_P_WEAK_ALG is not None:
        return

    nu_w = NU_WEAK_ALG
    lam_w = LAMBDA_WEAK_ALG
    nu_s = float(NU_STRONG_ALG)
    lam_s = float(LAMBDA_STRONG_ALG)

    # Canal faible : calibré sur le muon
    mu_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "mu-")
    eq_mu = cornell_equilibrium(mu_cfg["knots"], mu_cfg["k"])
    n_mu = len(mu_cfg["knots"])
    _TAU_P_WEAK_ALG = (
        TAU_MU * eq_mu["mass_raw"] ** nu_w / np.exp(-lam_w * n_mu)
    )

    # Canal fort : calibré sur le W
    w_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "W+")
    eq_w = cornell_equilibrium(w_cfg["knots"], w_cfg["k"])
    n_w = len(w_cfg["knots"])
    _TAU_P_STRONG_ALG = (
        TAU_W * eq_w["mass_raw"] ** nu_s / np.exp(-lam_s * n_w)
    )


def algebraic_lifetime(
    knot_names: list[str],
    k: int,
    channel: str = "auto",
) -> float:
    """Durée de vie topologique avec exposants algébriques.

    τ = τ_P × exp(-λ × n) / M_raw^ν

    Canal faible  : ν = 5π/2, λ = 5    (calibré sur le muon)
    Canal fort/EW : ν = 7/4,  λ = 7/4  (calibré sur le W)

    Parameters
    ----------
    knot_names : configuration nœuds
    k : indice Fibonacci
    channel : "weak", "strong", ou "auto" (auto-détection via SM_ASSIGNMENTS)

    Returns
    -------
    τ en secondes
    """
    _calibrate_algebraic()
    eq = cornell_equilibrium(knot_names, k)
    M_raw = eq["mass_raw"]
    n = len(knot_names)

    if channel == "auto":
        for p in SM_ASSIGNMENTS:
            if p["knots"] == knot_names and p["k"] == k:
                channel = "strong" if p["name"] in STRONG_DECAY else "weak"
                break
        else:
            channel = "weak"

    if channel == "strong":
        nu = float(NU_STRONG_ALG)
        lam = float(LAMBDA_STRONG_ALG)
        return _TAU_P_STRONG_ALG * np.exp(-lam * n) / M_raw ** nu
    else:
        nu = NU_WEAK_ALG
        lam = LAMBDA_WEAK_ALG
        return _TAU_P_WEAK_ALG * np.exp(-lam * n) / M_raw ** nu


def sm_lifetime_table_algebraic() -> list[dict]:
    """Table comparant durées de vie algébrique vs P15 vs expérimentale.

    Returns list of dicts with keys:
        name, knots, k, n, m_model, tau_alg, tau_p15, tau_exp,
        log_ratio_alg, log_ratio_p15, channel
    """
    from theory.multi_knot import topological_lifetime_p15

    rows = []
    for p in SM_ASSIGNMENTS:
        name = p["name"]
        eq = cornell_equilibrium(p["knots"], p["k"])
        m_mod = ETA * eq["mass_raw"]
        channel = "strong" if name in STRONG_DECAY else "weak"

        tau_alg = algebraic_lifetime(p["knots"], p["k"], channel)
        tau_p15 = topological_lifetime_p15(p["knots"], p["k"], channel)
        tau_exp = TAU_EXP_DATA.get(name)

        log_alg = (
            np.log10(tau_alg / tau_exp)
            if tau_exp is not None and tau_exp > 0
            else None
        )
        log_p15 = (
            np.log10(tau_p15 / tau_exp)
            if tau_exp is not None and tau_exp > 0
            else None
        )

        rows.append({
            "name": name,
            "knots": p["knots"],
            "k": p["k"],
            "n": len(p["knots"]),
            "m_model": m_mod,
            "tau_alg": tau_alg,
            "tau_p15": tau_p15,
            "tau_exp": tau_exp,
            "log_ratio_alg": log_alg,
            "log_ratio_p15": log_p15,
            "channel": channel,
        })
    return rows


def stability_summary() -> dict:
    """Résumé : S(K) pour les 19 nœuds, S_link pour les 13 particules SM.

    Returns dict with keys:
        knot_stability: dict[str, dict] — S(K) par nœud
        particle_stability: list[dict] — S_link par particule
        rms_alg, max_alg: erreur algébrique (décades)
        rms_p15, max_p15: erreur P15 (décades)
    """
    # S(K) pour les 19 nœuds non-triviaux
    knot_stab = {}
    for name in JONES_KNOT_NAMES:
        if name == "0_1":
            continue
        knot_stab[name] = structural_stability(name)

    # S_link pour les 13 particules SM
    particle_stab = []
    for p in SM_ASSIGNMENTS:
        sl = link_structural_stability(p["knots"])
        particle_stab.append({
            "name": p["name"],
            **sl,
        })

    # Erreurs (décades)
    table = sm_lifetime_table_algebraic()
    log_alg = [
        abs(r["log_ratio_alg"]) for r in table if r["log_ratio_alg"] is not None
    ]
    log_p15 = [
        abs(r["log_ratio_p15"]) for r in table if r["log_ratio_p15"] is not None
    ]

    rms_alg = np.sqrt(np.mean(np.array(log_alg) ** 2)) if log_alg else 0.0
    max_alg = max(log_alg) if log_alg else 0.0
    rms_p15 = np.sqrt(np.mean(np.array(log_p15) ** 2)) if log_p15 else 0.0
    max_p15 = max(log_p15) if log_p15 else 0.0

    return {
        "knot_stability": knot_stab,
        "particle_stability": particle_stab,
        "rms_alg": float(rms_alg),
        "max_alg": float(max_alg),
        "rms_p15": float(rms_p15),
        "max_p15": float(max_p15),
    }
