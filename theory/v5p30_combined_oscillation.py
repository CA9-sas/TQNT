"""V5.30 — Combinaison V5.29a (frustrés amphichiraux) × V5.29c (Kashaev).

Cadre V5.30
===========

V5.28 (commit 962c96f) :
    Δm = 2 ℏ ω_0 × N_paths × exp(-S_min/ℏ)

V5.29a (commit 0433d47) — chemins frustrés amphichiraux, parameter-free :
    N_paths^V5.29a = N_paths^V5.28 × Π_i F_i
    F_i = c_i! × b_i si K_i amphichiral, F_i = 1 sinon

V5.29c (commit e9b11b5) — préfacteur Kashaev / volume conjecture :
    Δm = 2 ℏ ω_0 × N_paths × exp(-S_min/ℏ) × exp(κ_K × vol_tot)

V5.30 = combinaison unifiée :

    Δm V5.30 = 2 ℏ ω_0 × N_paths^V5.28(L) × Π_i F_i(L)
                       × exp(-S_min(L)/ℏ) × exp(κ_K × vol_tot(L))

Calibration : κ_K refit globalement par χ² log sur K, D, B⁰, B_s⁰
(K⁰ contribue zéro par construction puisque le préfacteur global absorbe
ω_0·ℏ·F_total(K⁰)·exp(κ_K·vol_tot(K⁰))).

Honnêteté V5.30
================
* V5.30 a **un seul paramètre libre** au-dessus de V5.28 : κ_K (le même
  que V5.29c). Les facteurs F_i de V5.29a sont entièrement déterminés
  par la topologie (c_i, b_i, amphichiralité), donc parameter-free.
* Si V5.30 ferme la hiérarchie complète K < D < B⁰ < B_s⁰ avec un seul
  κ_K calibré globalement, c'est un SUCCÈS.
* Si V5.30 améliore mais ne ferme pas la hiérarchie, c'est PARTIAL.
* Si V5.30 casse l'un des systèmes, c'est FAILURE et il faut explorer
  V5.30c (modes vibratoires).

Référence : theory/topological_chiral_oscillation.py (V5.28, V5.29c)
            theory/v5p29_frustrated_amphichiral.py (V5.29a)
            theory/kashaev_invariants.py
"""
from __future__ import annotations

import numpy as np

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    action_min,
    delta_m_chiral_oscillation,
    delta_m_chiral_oscillation_v29c,
    KAPPA_K_DEFAULT,
)
from theory.v5p29_frustrated_amphichiral import (
    n_paths_with_frustration,
    delta_m_v5p29a,
)
from theory.kashaev_invariants import (
    total_hyperbolic_volume,
    prefactor_volume,
)


# Valeur par défaut V5.30, déterminée par calibrate_kappa_K_global() ci-dessous.
# Mise à jour empiriquement après le premier appel.
KAPPA_K_V5p30_DEFAULT = 0.611

# Cache de la calibration globale
_PREFACTOR_V5p30_MEV: dict[float, float] = {}


# ---------------------------------------------------------------------------
# 1. Élément matriciel combiné V5.30
# ---------------------------------------------------------------------------
def matrix_element_v5p30(
    knot_names: list[str], mass_link_MeV: float, kappa_K: float
) -> dict:
    """Élément matriciel topologique V5.30.

        ⟨K | H_topo | K*⟩_V5.30
            = N_paths^V5.29a × exp(-S_min/ℏ) × exp(κ_K × vol_tot)

    Le préfacteur global ω_0·ℏ est absorbé dans la calibration sur K⁰
    (voir `_calibrate_prefactor_v5p30`).

    Returns
    -------
    dict
        { matrix_element, N_paths, F_total, S_min, vol_tot, P_Kashaev, ...}
    """
    paths = n_paths_with_frustration(knot_names)
    action = action_min(knot_names, mass_link_MeV)
    pK = prefactor_volume(knot_names, kappa_K)
    vol_tot = total_hyperbolic_volume(knot_names)
    me = paths["N_paths"] * np.exp(-action["S_min"]) * pK
    return {
        "matrix_element": float(me),
        "N_paths": paths["N_paths"],
        "N_paths_v528": paths["N_paths_v528"],
        "F_total": paths["F_total"],
        "F_per_component": paths["F_per_component"],
        "n_amphichiral": paths["n_amphichiral"],
        "S_min": action["S_min"],
        "vol_tot": float(vol_tot),
        "P_Kashaev": float(pK),
        "kappa_K": float(kappa_K),
        "mass_ratio": action["mass_ratio"],
        "components": paths["components"],
    }


# ---------------------------------------------------------------------------
# 2. Calibration du préfacteur global sur K⁰
# ---------------------------------------------------------------------------
def _calibrate_prefactor_v5p30(kappa_K: float) -> float:
    """Calibre le préfacteur global V5.30 sur Δm_K⁰ (PDG).

    Pour K⁰ (6_1)³ : F_total = 1 (aucun amphichiral), donc V5.30(K⁰) ≡
    V5.29c(K⁰). Le préfacteur absorbe ω_0·ℏ·exp(κ_K·vol_tot(K⁰)).
    """
    if kappa_K in _PREFACTOR_V5p30_MEV:
        return _PREFACTOR_V5p30_MEV[kappa_K]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    me = matrix_element_v5p30(cfg["knots"], cfg["mass_pdg"], kappa_K)
    target_dm = DELTA_M_PDG["K0"]
    pref = target_dm / (2.0 * me["matrix_element"])
    _PREFACTOR_V5p30_MEV[kappa_K] = pref
    return pref


# ---------------------------------------------------------------------------
# 3. Δm V5.30
# ---------------------------------------------------------------------------
def delta_m_v5p30(
    knot_names: list[str],
    mass_link_MeV: float,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.30 = combinaison frustrés × Kashaev.

        Δm = 2 × prefactor(κ_K) × N_paths^V5.29a × exp(-S_min)
                × exp(κ_K × vol_tot)

    Parameters
    ----------
    knot_names : list[str]
    mass_link_MeV : float
    kappa_K : float
        Paramètre libre V5.30, calibré globalement sur K, D, B, B_s.

    Returns
    -------
    dict
        { delta_m_MeV, N_paths, F_total, S_min, vol_tot, P_Kashaev,
          prefactor_MeV, ...}
    """
    pref = _calibrate_prefactor_v5p30(kappa_K)
    me = matrix_element_v5p30(knot_names, mass_link_MeV, kappa_K)
    dm = 2.0 * pref * me["matrix_element"]
    return {
        "delta_m_MeV": float(dm),
        "N_paths": me["N_paths"],
        "N_paths_v528": me["N_paths_v528"],
        "F_total": me["F_total"],
        "F_per_component": me["F_per_component"],
        "n_amphichiral": me["n_amphichiral"],
        "S_min": me["S_min"],
        "vol_tot": me["vol_tot"],
        "P_Kashaev": me["P_Kashaev"],
        "kappa_K": me["kappa_K"],
        "matrix_element": me["matrix_element"],
        "prefactor_MeV": pref,
        "components": me["components"],
    }


# ---------------------------------------------------------------------------
# 4. Calibration globale de κ_K (V5.30)
# ---------------------------------------------------------------------------
def calibrate_kappa_K_global(
    systems: list[str] | None = None,
    kappa_range: tuple[float, float] = (-1.0, 2.0),
    n_grid: int = 601,
) -> dict:
    """Calibre κ_K globalement par χ²_log sur les systèmes donnés.

        χ²_log(κ_K) = Σ_i log²(Δm V5.30 / Δm PDG)_i

    K⁰ contribue zéro par construction (préfacteur global absorbe la
    calibration). Les contributions effectives viennent de D⁰, B⁰, B_s⁰.

    Parameters
    ----------
    systems : list[str] | None
        Sous-ensemble de {'K0', 'D0', 'B0', 'Bs0'}. Si None, tous les 4.
    kappa_range : (float, float)
        Bornes du scan 1D.
    n_grid : int
        Résolution du scan.

    Returns
    -------
    dict
        { kappa_K, chi2_log, per_meson }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(kappa_range[0], kappa_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, kappa in enumerate(grid):
        s = 0.0
        ok = True
        for name in systems:
            cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
            res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], float(kappa))
            dm = res["delta_m_MeV"]
            if dm <= 0 or not np.isfinite(dm):
                ok = False
                break
            log_r = np.log10(dm / DELTA_M_PDG[name])
            s += log_r * log_r
        if ok:
            chi2[j] = s
    idx = int(np.argmin(chi2))
    best = float(grid[idx])
    per = {}
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], best)
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": float(
                np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name])
            ),
        }
    return {
        "kappa_K": best,
        "chi2_log": float(chi2[idx]),
        "per_meson": per,
        "systems": systems,
    }


# ---------------------------------------------------------------------------
# 5. Table de comparaison V5.30 vs V5.28/V5.29a/V5.29c/PDG
# ---------------------------------------------------------------------------
def v5p30_table(kappa_K: float = KAPPA_K_V5p30_DEFAULT) -> list[dict]:
    """Table de comparaison Δm V5.30 vs V5.28, V5.29a, V5.29c, PDG.

    Returns
    -------
    list of dicts : name, knots, delta_m_v5p30, delta_m_v529a, delta_m_v529c,
        delta_m_v528, delta_m_pdg, log_ratio_v5p30, log_ratio_v529a,
        log_ratio_v529c, log_ratio_v528, F_total, vol_tot, P_Kashaev,
        N_paths.
    """
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res_v530 = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], kappa_K)
        res_v528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
        res_v29a = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
        res_v29c = delta_m_chiral_oscillation_v29c(
            cfg["knots"], cfg["mass_pdg"], KAPPA_K_DEFAULT
        )
        dm_v530 = res_v530["delta_m_MeV"]
        dm_v528 = res_v528["delta_m_MeV"]
        dm_v29a = res_v29a["delta_m_MeV"]
        dm_v29c = res_v29c["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]

        def _log(x: float) -> float | None:
            return float(np.log10(x / dm_pdg)) if x > 0 else None

        rows.append({
            "name": name,
            "knots": cfg["knots"],
            "mass_pdg_MeV": cfg["mass_pdg"],
            "delta_m_v5p30_MeV": dm_v530,
            "delta_m_v529a_MeV": dm_v29a,
            "delta_m_v529c_MeV": dm_v29c,
            "delta_m_v528_MeV": dm_v528,
            "delta_m_pdg_MeV": dm_pdg,
            "log_ratio_v5p30": _log(dm_v530),
            "log_ratio_v529a": _log(dm_v29a),
            "log_ratio_v529c": _log(dm_v29c),
            "log_ratio_v528": _log(dm_v528),
            "F_total": res_v530["F_total"],
            "n_amphichiral": res_v530["n_amphichiral"],
            "vol_tot": res_v530["vol_tot"],
            "P_Kashaev": res_v530["P_Kashaev"],
            "N_paths": res_v530["N_paths"],
            "N_paths_v528": res_v530["N_paths_v528"],
            "S_min": res_v530["S_min"],
            "kappa_K": res_v530["kappa_K"],
        })
    return rows
