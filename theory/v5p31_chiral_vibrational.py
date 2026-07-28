"""V5.31 — Spectre vibratoire chiral asymétrique pour fermer Bs/B.

Diagnostic V5.30c
=================

V5.30c (Σ|log| = 1.05 dex) reproduit déjà les absolus :

    K⁰  : log_ratio = 0.000  (calibré)
    D⁰  : log_ratio = -0.276
    B⁰  : log_ratio = -0.755
    B_s⁰: log_ratio = +0.020  (quasi-parfait)

mais sur-amplifie le ratio B_s/B :

    V5.30c : B_s/B = 209
    PDG    : B_s/B = 35

soit un facteur ×6.  Cette sur-amplification vient du facteur de
frustration amphichirale F(6_3) = 6!·4 = 2880 (V5.29a) qui ferme
correctement B_s⁰ en absolu mais pousse trop le ratio parce que B⁰ n'a
**aucune** composante amphichirale.

Insight V5.31 — la chiralité distingue les modes vibratoires
============================================================

Les modes propres ω_0(K) = b(K)/√c(K) de V5.30c sont **scalaires** : ils
ne voient pas la chiralité de la composante.  V5.31 introduit la
distinction structurelle suivante :

* **Amphichirals** (4_1, 6_3, 8_3) : V_K(t) = V_K(t⁻¹).  Les modes
  vibratoires sont **symétriques** sous l'opération miroir K ↔ K*.
  Ces modes symétriques *suppriment* le tunneling K ↔ K* car ils
  n'ont pas de direction préférée — ils oscillent identiquement dans
  les deux phases.

* **Chirals** (3_1, 5_1, 6_1, 6_2, 7_2, 7_6, 7_7, ...) : V_K ≠ V_K(t⁻¹).
  Les modes vibratoires sont **asymétriques** sous miroir — ils ont
  une direction préférée mesurée par σ(K) = |jones_power_center(V_K)|.
  Ces modes asymétriques *facilitent* le tunneling K ↔ K*.

Le préfacteur vibratoire chiral est donc :

    P_chiral(L) = Π_i chiral_factor(K_i)

    chiral_factor(K_amphichiral) = exp(-β · 1)
    chiral_factor(K_chiral)      = exp(+β · |σ(K)|)

où σ(K) = jones_power_center(V_K) (cf. theory/jones.py).

Effet attendu pour les 4 mésons
================================

Par composante :
* K0 (3×6_1) : 3 chirals, Σ|σ| = 3·0.333 = 1.000  →  log P = +β
* D0 (3_1, 5_1, 7_2) : 3 chirals,
  Σ|σ| = 1.667 + 1.800 + 1.286 = 4.752  →  log P = +4.752 β
* B0 (3_1, 6_2, 7_7) : 3 chirals,
  Σ|σ| = 1.667 + 0.667 + 0.143 = 2.476  →  log P = +2.476 β
* Bs0 (6_3, 7_6) : 1 amphi + 1 chiral,
  Σ = -1 + |0.714| = -0.286  →  log P = -0.286 β

V5.31 = V5.30c × P_chiral / P_chiral(K0).  K⁰ reste calibré par
construction (R_chiral(K⁰) = 1).

Calibration et limite physique attendue
========================================

Un seul paramètre β calibré sur le ratio B_s/B = 35 (PDG).  Une
analyse de premier ordre montre :

    log10(R_Bs/R_B)(β) = (σ_Bs - σ_B) · β / ln(10)
                       = (-0.286 - 2.476) · β / ln(10)
                       = -1.200 · β

On veut shifter le ratio actuel de -log10(209/35) = -0.776 dex,
donc β ≈ 0.65 fermerait le ratio.  Mais ce même β shifte D :

    log10(R_D)(β=0.65) = (4.752 - 1.000) · 0.65 / ln(10) = +1.06 dex
    log10(R_B)(β=0.65) = (2.476 - 1.000) · 0.65 / ln(10) = +0.42 dex

→ D devient log_ratio = -0.28 + 1.06 = +0.78 dex (encore mauvais)
→ B devient log_ratio = -0.76 + 0.42 = -0.34 dex (amélioré)
→ Bs devient log_ratio = +0.02 + (-0.286·0.65/ln(10)) = -0.06 dex

Donc le simple Π σ ne ferme pas D⁰.  V5.31 propose une calibration
honnête sur le RATIO B_s/B et reporte les conséquences sur les
absolus K⁰, D⁰, B⁰, B_s⁰.

Verdict honnête (cf. consigne) :
* SUCCESS : Σ|log| < 1.0 dex ET ratio Bs/B ∈ [30, 40]
* PARTIAL : amélioration mais pas tous critères
* FAILURE : V5.31 casse un système relatif à V5.30c

Référence : theory/v5p30c_vibrational.py
            theory/jones.py (is_amphicheiral, jones_power_center)
"""
from __future__ import annotations

import numpy as np

from theory.jones import (
    is_amphicheiral,
    jones_polynomial,
    jones_power_center,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import (
    ALPHA_VIB_DEFAULT,
    delta_m_v5p30c,
)


# ---------------------------------------------------------------------------
# Constantes V5.31
# ---------------------------------------------------------------------------
# β calibré sur ratio Bs/B = 35 (PDG).  Voir calibrate_beta_ratio().
BETA_CHIRAL_DEFAULT = 0.647

# Pénalité amphichirale.  Forme "exp(-β · 1)" par composante amphichirale.
AMPHI_PENALTY = 1.0

# Cache du préfacteur de référence K0
_P_CHIRAL_K0_CACHE: dict[float, float] = {}


# ---------------------------------------------------------------------------
# 1. Signature de chiralité σ(K)
# ---------------------------------------------------------------------------
def chirality_signature(knot_name: str) -> dict:
    """Signature de chiralité σ(K) d'une composante.

    Pour un nœud K :

    * **Amphichiral** (V_K = V_K(t⁻¹)) : σ(K) = -1 par convention.
      Les modes vibratoires sont symétriques sous miroir et SUPPRIMENT
      le tunneling K ↔ K*.
    * **Chiral** (V_K ≠ V_K(t⁻¹)) : σ(K) = +|power_center(V_K)|.
      Les modes vibratoires sont asymétriques et FACILITENT le tunneling.

    σ(K) = 0 correspondrait à un cas neutre — non utilisé ici (les
    amphichiraux ont la pénalité -1 explicite, et power_center=0 implique
    déjà amphichiral pour notre catalogue).

    Returns
    -------
    dict
        { knot, is_amphi, power_center, sigma }
    """
    poly = jones_polynomial(knot_name)
    amphi = is_amphicheiral(poly)
    pc = jones_power_center(poly)
    if amphi:
        sigma = -AMPHI_PENALTY
    else:
        sigma = abs(pc)
    return {
        "knot": knot_name,
        "is_amphi": bool(amphi),
        "power_center": float(pc),
        "sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# 2. Facteur chiral par composante
# ---------------------------------------------------------------------------
def chiral_factor(knot_name: str, beta: float) -> dict:
    """Facteur vibratoire chiral d'une composante.

        chiral_factor(K) = exp(β · σ(K))

    avec σ(K) = -1 si K amphichiral, |power_center| si K chiral.

    * Amphichiral : facteur = exp(-β) < 1  → SUPPRIME
    * Chiral : facteur = exp(+β·|σ|) > 1   → AMPLIFIE

    Returns
    -------
    dict
        { knot, sigma, is_amphi, factor }
    """
    sig = chirality_signature(knot_name)
    factor = float(np.exp(beta * sig["sigma"]))
    return {
        "knot": knot_name,
        "sigma": sig["sigma"],
        "is_amphi": sig["is_amphi"],
        "factor": factor,
        "beta": float(beta),
    }


# ---------------------------------------------------------------------------
# 3. Préfacteur chiral global P_chiral(L)
# ---------------------------------------------------------------------------
def p_chiral(knot_names: list[str], beta: float) -> dict:
    """Préfacteur vibratoire chiral d'un entrelacs.

        P_chiral(L) = Π_i chiral_factor(K_i)
                    = exp(β · Σ_i σ(K_i))

    Returns
    -------
    dict
        { p_chiral, sigma_sum, n_amphi, per_component }
    """
    per = [chiral_factor(n, beta) for n in knot_names]
    sigma_sum = sum(p["sigma"] for p in per)
    n_amphi = sum(1 for p in per if p["is_amphi"])
    return {
        "p_chiral": float(np.exp(beta * sigma_sum)),
        "sigma_sum": float(sigma_sum),
        "n_amphi": int(n_amphi),
        "n_chiral": int(len(per) - n_amphi),
        "beta": float(beta),
        "per_component": per,
    }


# ---------------------------------------------------------------------------
# 4. Calibration sur K⁰
# ---------------------------------------------------------------------------
def _p_chiral_K0(beta: float) -> float:
    """Préfacteur chiral de K⁰ (référence, R_chiral(K⁰) = 1)."""
    key = float(beta)
    if key in _P_CHIRAL_K0_CACHE:
        return _P_CHIRAL_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    val = p_chiral(cfg["knots"], beta)["p_chiral"]
    _P_CHIRAL_K0_CACHE[key] = float(val)
    return float(val)


# ---------------------------------------------------------------------------
# 5. Δm V5.31
# ---------------------------------------------------------------------------
def delta_m_v5p31(
    knot_names: list[str],
    mass_link_MeV: float,
    beta: float = BETA_CHIRAL_DEFAULT,
    alpha: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.31 = Δm V5.30c × R_chiral(L), avec R_chiral(K⁰) = 1.

    Parameters
    ----------
    knot_names : list[str]
    mass_link_MeV : float
    beta : float
        Paramètre chiral libre V5.31 (calibré sur ratio Bs/B = 35).
    alpha : float
        Paramètre vibratoire V5.30c (conservé à sa valeur calibrée).
    kappa_K : float
        Paramètre V5.30 (Kashaev), conservé.

    Returns
    -------
    dict
        { delta_m_MeV, delta_m_v5p30c_MeV, R_chiral, P_chiral, ... }
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha, kappa_K)
    pc_K0 = _p_chiral_K0(beta)
    pc = p_chiral(knot_names, beta)
    R_chiral = pc["p_chiral"] / pc_K0
    dm = res_v530c["delta_m_MeV"] * R_chiral
    return {
        "delta_m_MeV": float(dm),
        "delta_m_v5p30c_MeV": float(res_v530c["delta_m_MeV"]),
        "R_chiral": float(R_chiral),
        "P_chiral": pc["p_chiral"],
        "P_chiral_K0": float(pc_K0),
        "sigma_sum": pc["sigma_sum"],
        "n_amphi": pc["n_amphi"],
        "n_chiral": pc["n_chiral"],
        "beta": float(beta),
        "alpha": float(alpha),
        "kappa_K": float(kappa_K),
        "R_vib": res_v530c["R_vib"],
        "per_component": pc["per_component"],
        "components_v530c": res_v530c["components"],
    }


# ---------------------------------------------------------------------------
# 6. Calibration de β sur ratio Bs/B
# ---------------------------------------------------------------------------
def calibrate_beta_ratio(
    target_ratio: float = 35.0,
    beta_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 6001,
    alpha: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Calibre β sur le ratio B_s/B observé PDG = 35.

        β* = argmin_β  (log10(R_Bs/R_B / target))²

    Returns
    -------
    dict
        { beta, ratio_v531, target, log_diff_dex, per_meson, sum_abs_log }
    """
    cfg_B = MESON_OSCILLATION_ASSIGNMENTS["B0"]
    cfg_Bs = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    log_target = np.log10(target_ratio)
    grid = np.linspace(beta_range[0], beta_range[1], n_grid)
    diffs = np.full_like(grid, np.inf)
    for j, b in enumerate(grid):
        # Clear cache between calls
        _P_CHIRAL_K0_CACHE.clear()
        rB = delta_m_v5p31(
            cfg_B["knots"], cfg_B["mass_pdg"], float(b), alpha, kappa_K
        )
        rBs = delta_m_v5p31(
            cfg_Bs["knots"], cfg_Bs["mass_pdg"], float(b), alpha, kappa_K
        )
        dmB = rB["delta_m_MeV"]
        dmBs = rBs["delta_m_MeV"]
        if dmB <= 0 or dmBs <= 0:
            continue
        ratio = dmBs / dmB
        diffs[j] = (np.log10(ratio) - log_target) ** 2
    idx = int(np.argmin(diffs))
    best = float(grid[idx])

    # Évalue les 4 mésons à β*
    _P_CHIRAL_K0_CACHE.clear()
    per = {}
    s_abs = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p31(
            cfg["knots"], cfg["mass_pdg"], best, alpha, kappa_K
        )
        log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": log_r,
            "R_chiral": res["R_chiral"],
            "n_amphi": res["n_amphi"],
            "sigma_sum": res["sigma_sum"],
        }
        s_abs += abs(log_r)

    ratio_v531 = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "beta": best,
        "ratio_v531": float(ratio_v531),
        "target_ratio": float(target_ratio),
        "log_diff_dex": float(np.log10(ratio_v531 / target_ratio)),
        "per_meson": per,
        "sum_abs_log": float(s_abs),
        "alpha": float(alpha),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 7. Table V5.31 vs V5.30c vs PDG
# ---------------------------------------------------------------------------
def v5p31_table(
    beta: float = BETA_CHIRAL_DEFAULT,
    alpha: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> list[dict]:
    """Table de comparaison Δm V5.31 vs V5.30c vs PDG.

    Returns
    -------
    list of dicts : name, knots, delta_m_v5p31, delta_m_v5p30c,
        delta_m_pdg, log_ratio_v5p31, log_ratio_v5p30c, R_chiral,
        sigma_sum, n_amphi, beta.
    """
    _P_CHIRAL_K0_CACHE.clear()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res_v531 = delta_m_v5p31(
            cfg["knots"], cfg["mass_pdg"], beta, alpha, kappa_K
        )
        dm_v531 = res_v531["delta_m_MeV"]
        dm_v530c = res_v531["delta_m_v5p30c_MeV"]
        dm_pdg = DELTA_M_PDG[name]

        def _log(x: float) -> float | None:
            return float(np.log10(x / dm_pdg)) if x > 0 else None

        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "mass_pdg_MeV": cfg["mass_pdg"],
                "delta_m_v5p31_MeV": dm_v531,
                "delta_m_v5p30c_MeV": dm_v530c,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p31": _log(dm_v531),
                "log_ratio_v5p30c": _log(dm_v530c),
                "R_chiral": res_v531["R_chiral"],
                "P_chiral": res_v531["P_chiral"],
                "sigma_sum": res_v531["sigma_sum"],
                "n_amphi": res_v531["n_amphi"],
                "n_chiral": res_v531["n_chiral"],
                "beta": float(beta),
                "alpha": float(alpha),
                "kappa_K": float(kappa_K),
            }
        )
    return rows
