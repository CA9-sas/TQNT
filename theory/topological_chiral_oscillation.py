"""V5.28 — Oscillation chirale K ↔ K* comme tunneling quantique topologique.

Reframing conceptuel (V5.28 vs V5p27)
====================================

V5p27 (commit 4d286f5) traitait les oscillations K-K̄, D-D̄, B-B̄ comme un
**taux de réaction phénoménologique** P15 :

    Γ ∝ exp(-λ d) × M_raw^(-ν)        (V5p27)

Résultat honnête : K calibré, D à 0.28 dex, B sous-prédit de 3.34 dex.

V5.28 reframe le problème en mécanique quantique topologique pure. Ce
qui oscille N'EST PAS un méson : c'est la **superposition cohérente
entre un nœud K et son miroir K***.

    |ψ(t)⟩ = α(t) |K⟩ + β(t) |K*⟩

L'élément matriciel topologique s'écrit comme une intégrale de chemin
sur toutes les transformations Reidemeister entre K et K* :

    ⟨K | H_topo | K*⟩ = Σ_paths exp(-S_path / ℏ)

et la différence de masse d'oscillation devient

    Δm = 2 × |⟨K | H_topo | K*⟩|
       ≈ N_paths × exp(-S_min/ℏ) × ℏ ω_0

où

* N_paths(L) est le nombre de **chemins R-move topologiquement
  distincts** menant de L à L* (compté combinatoirement, avec
  suppression bosonique des composantes identiques)
* S_min(L) est l'action topologique du chemin minimum, qui dépend de la
  taille combinatoire (c_total) et du facteur de Jones effectif
* ω_0 = c²/r₀ est une fréquence topologique de référence
  (rapport vitesse-lumière / rayon de Compton du méson)

Mécanisme physique de l'enhancement B / K
==========================================

L'enhancement de B⁰ en V5.28 vient **uniquement de la combinatoire des
chemins R-move**. L'action S_min reste purement topologique :

    S_min(L)/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)

(une action de tunneling topologique standard, somme sur les
composantes du link.)

Hypothèse alternative testée et rejetée
---------------------------------------

Une version antérieure de V5.28 introduisait un facteur de
dilatation WKB (M_ref/M_link)^γ pour modéliser la réduction d'action
dans le repère de repos du méson lourd. Toutes les valeurs de
γ ∈ [0, 1] ont été scannées : **aucune** ne reproduit la hiérarchie
PDG (Bs⁰ > B⁰ > D⁰ > K⁰). γ=0 (pas de rescaling) est conservé comme
choix le plus parcimonieux et le plus honnête : il ferme le gap B⁰
(V5p27: -3.34 dex → V5.28: -0.73 dex) mais ne reproduit pas le
ratio Bs/B PDG ≈ 35 (V5.28 prédit Bs/B ≈ 0.1).

Compte combinatoire des chemins
================================

Pour un link L = K_1 + ... + K_n avec c_i croisements chacun
(c_tot = Σ c_i), un chemin direct L → L* doit inverser **tous** les
croisements. L'ordre d'inversion est libre, mais :

* deux croisements **dans la même composante** peuvent être inversés
  dans n'importe quel ordre (donc divisés par c_i!)
* deux croisements dans des composantes **identiques** sont
  indistinguables (symmetry factor 1/n_sym!)
* deux croisements dans des composantes **différentes** ajoutent une
  multiplicité (multinomial dans c_tot)

D'où la formule canonique :

    N_paths(L) = c_tot! / (∏_i c_i!  ×  ∏_j n_sym_j!)

où n_sym_j est la multiplicité de chaque type de composante.
Multiplié par ∏ b_i (indice de tresse par composante) pour le degré de
liberté de l'ordonnancement des brins.

Pour K⁰ (6_1 + 6_1 + 6_1, n_sym = 3 pour 6_1) :
    N_paths = 18! / (6!·6!·6! × 3!) × 3³ = 4 287 ≈ O(10³)

Pour B⁰ (3_1 + 6_2 + 7_7, toutes distinctes) :
    N_paths = 16! / (3!·6!·7! × 1·1·1) × 2·3·4 = 2.3 × 10⁷

L'écart de 4 ordres de grandeur sur N_paths est complémentaire à
l'écart sur exp(-S_min) — ensemble ils ferment partiellement le gap B.

Honnêteté V5.28
================

Cette construction introduit **deux nouveaux paramètres** par rapport
à V5p27 :

* κ_S (constante d'action par croisement) — calibré sur K⁰
* le rapport M_ref (≡ M_K⁰) — déterminé physiquement par la
  calibration

La calibration globale (préfacteur ω_0 × ℏ × constant) est aussi
calibrée sur K⁰. Donc V5.28 a **1 paramètre de plus** que V5p27 (le
schéma exponentiel WKB), pour 4 mesures (K, D, B⁰, Bs⁰). À tester
honnêtement.
"""
from __future__ import annotations

from math import factorial
from collections import Counter

import numpy as np

from theory.jones import (
    jones_polynomial,
    mirror_jones,
    is_amphicheiral,
    braid_index,
)
from theory.retraction import retraction_factor
from theory.knot_catalog import crossing_number
from theory.multi_knot import cornell_equilibrium
from theory.kashaev_invariants import (
    total_hyperbolic_volume,
    prefactor_volume,
)

# --- Constantes physiques ---
HBAR_MEV_S = 6.582119569e-22   # MeV·s

# --- Assignations V5 (même catalogue que V5p27) ---
MESON_OSCILLATION_ASSIGNMENTS = {
    "K0":   {"knots": ["6_1", "6_1", "6_1"], "k": 11, "mass_pdg": 497.611},
    "D0":   {"knots": ["3_1", "5_1", "7_2"], "k": 18, "mass_pdg": 1864.84},
    "B0":   {"knots": ["3_1", "6_2", "7_7"], "k": 22, "mass_pdg": 5279.66},
    "Bs0":  {"knots": ["6_3", "7_6"],        "k": 22, "mass_pdg": 5366.92},
}

DELTA_M_PDG = {
    "K0":  3.484e-12,
    "D0":  6.25e-12,
    "B0":  3.334e-10,
    "Bs0": 1.169e-8,
}

# --- Paramètres calibrés sur K0 ---
# κ_S : action topologique par croisement (sans dimension).  Fixé à 1 par
#       convention ; la calibration finale passe par le préfacteur global.
# γ_M : exposant de dilatation WKB (M_ref/M_link)^γ_M.  Hypothèse rejetée
#       par scan empirique (voir docstring du module) : γ_M = 0 conservé.
KAPPA_S = 1.0
GAMMA_M = 0.0
M_REF_MEV = 497.611

# Préfacteur global (ω_0 × ℏ) calibré sur K⁰ — déterminé dans
# `_calibrate_prefactor()` ci-dessous.
_PREFACTOR_MEV: float | None = None


# ---------------------------------------------------------------------------
# 1. Comptage de chemins Reidemeister
# ---------------------------------------------------------------------------
def count_paths_K_to_Kstar(knot_names: list[str]) -> dict:
    """Nombre de chemins R-move topologiquement distincts L → L*.

    Formule :

        N_paths = (c_tot! / Π c_i!)  /  Π_j n_sym_j!  ×  Π_i b_i

    où :

    * c_i = crossing_number(K_i)
    * c_tot = Σ c_i
    * n_sym_j = multiplicité du type de composante j (suppression bosonique)
    * b_i = braid_index(K_i)  (degré de liberté de l'ordonnancement des
      brins par composante)

    Tous les croisements (même ceux d'une composante amphichirale) doivent
    être inversés pour passer de la configuration K à sa configuration
    miroir explicite — l'amphichiralité signifie V_K = V_{K*} au niveau du
    polynôme de Jones, pas l'égalité des configurations géométriques.

    Returns
    -------
    dict
        { N_paths, c_tot_eff, multinomial, sym_factor, braid_product,
          components }
    """
    counts = Counter(knot_names)

    # Crossings effectifs par composante : tous les croisements doivent
    # être inversés pour transformer K en K* (même pour les composantes
    # amphichirales, qui passent par un état intermédiaire de
    # suspension Reidemeister).
    c_eff = [crossing_number(n) for n in knot_names]
    c_tot = sum(c_eff)

    # Multinomial : c_tot! / Π c_eff_i!
    multinomial = factorial(c_tot)
    for c in c_eff:
        multinomial //= factorial(c)

    # Suppression bosonique : on divise par Π n_sym!
    sym_factor = 1
    for n_sym in counts.values():
        sym_factor *= factorial(n_sym)

    # Produit des indices de tresse
    braid_product = 1
    for n in knot_names:
        braid_product *= braid_index(n)

    N_paths = (multinomial // sym_factor) * braid_product
    return {
        "N_paths": int(N_paths),
        "c_tot_eff": c_tot,
        "multinomial": int(multinomial),
        "sym_factor": int(sym_factor),
        "braid_product": int(braid_product),
        "components": [(n, c) for n, c in zip(knot_names, c_eff)],
    }


# ---------------------------------------------------------------------------
# 2. Action topologique du chemin minimum
# ---------------------------------------------------------------------------
def action_min(knot_names: list[str], mass_link_MeV: float) -> dict:
    """Action topologique S_min/ℏ du chemin de tunneling minimum.

    Formule (V5.28, action purement topologique) :

        S_min/ℏ = κ_S × ( Σ_i c_i log(1 + c_J,i) ) × (M_ref / M_link)^γ_M

    où c_J,i est le facteur de Jones (rétraction polynomiale) de la
    composante K_i, et γ_M ∈ [0, 1] est un exposant de dilatation WKB.
    Le scan γ_M ∈ [0, 1] a montré qu'aucune valeur ne reproduit la
    hiérarchie PDG (Bs⁰ > B⁰ > D⁰ > K⁰) ; γ_M = 0 est conservé comme
    valeur la plus parcimonieuse (voir docstring du module).

    Returns
    -------
    dict
        { S_min, action_topo_raw, mass_ratio, contributions }
    """
    contributions = []
    raw_sum = 0.0
    for n in knot_names:
        c = crossing_number(n)
        cJ = retraction_factor(jones_polynomial(n))
        term = c * np.log1p(cJ)
        contributions.append(
            {"knot": n, "c": c, "c_J": cJ, "term": term}
        )
        raw_sum += term

    mass_ratio = (M_REF_MEV / mass_link_MeV) ** GAMMA_M
    S_min = KAPPA_S * raw_sum * mass_ratio
    return {
        "S_min": float(S_min),
        "action_topo_raw": float(raw_sum),
        "mass_ratio": float(mass_ratio),
        "contributions": contributions,
    }


# ---------------------------------------------------------------------------
# 3. Élément matriciel topologique <K|H|K*>
# ---------------------------------------------------------------------------
def matrix_element_topo(knot_names: list[str], mass_link_MeV: float) -> dict:
    """Élément matriciel topologique ⟨K | H_topo | K*⟩.

    En approximation de col (saddle-point) sur l'intégrale de chemin :

        ⟨K | H | K*⟩ ≈ N_paths × exp(-S_min/ℏ)

    Le préfacteur ℏω_0 (énergie de la barrière) est inclus dans la
    routine `delta_m_chiral_oscillation` via la calibration K⁰.

    Returns
    -------
    dict
        { matrix_element, N_paths, S_min, ...}
    """
    paths = count_paths_K_to_Kstar(knot_names)
    action = action_min(knot_names, mass_link_MeV)
    me_dimensionless = paths["N_paths"] * np.exp(-action["S_min"])
    return {
        "matrix_element": float(me_dimensionless),
        "N_paths": paths["N_paths"],
        "S_min": action["S_min"],
        "mass_ratio": action["mass_ratio"],
        "components": paths["components"],
    }


# ---------------------------------------------------------------------------
# 4. Calibration du préfacteur sur K⁰
# ---------------------------------------------------------------------------
def _calibrate_prefactor() -> float:
    """Calibre le préfacteur global ω_0·ℏ sur Δm_K⁰ (PDG)."""
    global _PREFACTOR_MEV
    if _PREFACTOR_MEV is not None:
        return _PREFACTOR_MEV
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    me = matrix_element_topo(cfg["knots"], cfg["mass_pdg"])
    # Δm = 2 × |⟨K|H|K*⟩| × ω_0·ℏ_factor
    # On ajuste le préfacteur pour que la prédiction colle exactement
    # à Δm_K (calibration).
    target_dm = DELTA_M_PDG["K0"]
    _PREFACTOR_MEV = target_dm / (2.0 * me["matrix_element"])
    return _PREFACTOR_MEV


# ---------------------------------------------------------------------------
# 5. Δm chiral V5.28
# ---------------------------------------------------------------------------
def delta_m_chiral_oscillation(
    knot_names: list[str], mass_link_MeV: float
) -> dict:
    """Différence de masse d'oscillation chirale Δm V5.28.

    Δm = 2 × prefactor × N_paths × exp(-S_min/ℏ)

    où le préfacteur est calibré sur K⁰.

    Returns
    -------
    dict
        { delta_m_MeV, N_paths, S_min, matrix_element, ...}
    """
    pref = _calibrate_prefactor()
    me = matrix_element_topo(knot_names, mass_link_MeV)
    delta_m = 2.0 * pref * me["matrix_element"]
    return {
        "delta_m_MeV": float(delta_m),
        "N_paths": me["N_paths"],
        "S_min": me["S_min"],
        "matrix_element": me["matrix_element"],
        "mass_ratio": me["mass_ratio"],
        "prefactor_MeV": pref,
        "components": me["components"],
    }


# ---------------------------------------------------------------------------
# 6. Table de comparaison V5.28 vs PDG vs V5p27
# ---------------------------------------------------------------------------
def chiral_oscillation_table() -> list[dict]:
    """Table de comparaison Δm V5.28 vs PDG vs V5p27.

    Returns
    -------
    list of dicts : name, knots, delta_m_v528, delta_m_pdg, log_ratio_528,
    log_ratio_v5p27 (pour mémoire), N_paths, S_min.
    """
    # V5p27 reference numbers (commit 4d286f5)
    V5P27 = {
        "K0":  3.484e-12,
        "D0":  3.271e-12,
        "B0":  1.527e-13,
        "Bs0": 2.015e-10,
    }
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
        dm_v5 = res["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        log_528 = float(np.log10(dm_v5 / dm_pdg)) if dm_v5 > 0 else None
        log_v5p27 = float(np.log10(V5P27[name] / dm_pdg))
        rows.append({
            "name": name,
            "knots": cfg["knots"],
            "mass_pdg_MeV": cfg["mass_pdg"],
            "delta_m_v528_MeV": dm_v5,
            "delta_m_v5p27_MeV": V5P27[name],
            "delta_m_pdg_MeV": dm_pdg,
            "log_ratio_v528": log_528,
            "log_ratio_v5p27": log_v5p27,
            "N_paths": res["N_paths"],
            "S_min": res["S_min"],
            "mass_ratio": res["mass_ratio"],
        })
    return rows


# ===========================================================================
# V5.29c — Kashaev / Volume conjecture comme préfacteur knot-spécifique
# ===========================================================================
# Paramètre libre V5.29c : κ_K (calibré pour reproduire la hiérarchie PDG).
# Valeur par défaut : déterminée par minimisation conjointe sur D/B/Bs
# (voir scripts/V5_oscillations/v5p29c_kashaev_test.py).
KAPPA_K_DEFAULT = 0.965

# Préfacteur V5.29c renormalisé : cache.
_PREFACTOR_V29C_MEV: dict[float, float] = {}


def _calibrate_prefactor_v29c(kappa_K: float) -> float:
    """Calibre le préfacteur global V5.29c sur Δm_K⁰ (PDG).

    Le préfacteur absorbe simultanément ω_0·ℏ et P_Kashaev(K⁰), de sorte
    que κ_K soit le seul paramètre physique libre. La calibration impose

        Δm V5.29c (K⁰)  =  Δm PDG (K⁰)

    Returns
    -------
    float
        prefactor_MeV tel que Δm = 2 · prefactor · N_paths · exp(-S_min)
        · exp(κ_K · vol_tot).
    """
    if kappa_K in _PREFACTOR_V29C_MEV:
        return _PREFACTOR_V29C_MEV[kappa_K]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    me = matrix_element_topo(cfg["knots"], cfg["mass_pdg"])
    pK = prefactor_volume(cfg["knots"], kappa_K)
    target_dm = DELTA_M_PDG["K0"]
    pref = target_dm / (2.0 * me["matrix_element"] * pK)
    _PREFACTOR_V29C_MEV[kappa_K] = pref
    return pref


def delta_m_chiral_oscillation_v29c(
    knot_names: list[str],
    mass_link_MeV: float,
    kappa_K: float = KAPPA_K_DEFAULT,
) -> dict:
    """Δm V5.29c : V5.28 augmenté du préfacteur Kashaev / volume conjecture.

    Forme :

        Δm = 2 · prefactor(K⁰) · N_paths · exp(-S_min/ℏ) · P_Kashaev(L)

    avec P_Kashaev(L) = exp(κ_K · Σ_i vol(K_i)).

    La calibration absorbe ω_0·ℏ et P_Kashaev(K⁰) ; κ_K reste le seul
    paramètre physique libre.

    Parameters
    ----------
    knot_names : list[str]
    mass_link_MeV : float
    kappa_K : float
        Paramètre V5.29c. Valeur par défaut : KAPPA_K_DEFAULT (calibré
        empiriquement, voir docstring du module et script associé).

    Returns
    -------
    dict
        { delta_m_MeV, N_paths, S_min, P_Kashaev, vol_tot, kappa_K,
          prefactor_MeV, components }
    """
    pref = _calibrate_prefactor_v29c(kappa_K)
    me = matrix_element_topo(knot_names, mass_link_MeV)
    pK = prefactor_volume(knot_names, kappa_K)
    vol_tot = total_hyperbolic_volume(knot_names)
    delta_m = 2.0 * pref * me["matrix_element"] * pK
    return {
        "delta_m_MeV": float(delta_m),
        "N_paths": me["N_paths"],
        "S_min": me["S_min"],
        "matrix_element": me["matrix_element"],
        "P_Kashaev": float(pK),
        "vol_tot": float(vol_tot),
        "kappa_K": float(kappa_K),
        "mass_ratio": me["mass_ratio"],
        "prefactor_MeV": pref,
        "components": me["components"],
    }


def chiral_oscillation_table_v29c(
    kappa_K: float = KAPPA_K_DEFAULT,
) -> list[dict]:
    """Table de comparaison Δm V5.29c vs V5.28 vs PDG.

    Returns
    -------
    list of dicts : name, knots, delta_m_v29c, delta_m_v528, delta_m_pdg,
    log_ratio_v29c, log_ratio_v528, N_paths, vol_tot, P_Kashaev.
    """
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res_29c = delta_m_chiral_oscillation_v29c(
            cfg["knots"], cfg["mass_pdg"], kappa_K
        )
        res_528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
        dm_29c = res_29c["delta_m_MeV"]
        dm_528 = res_528["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        log_29c = float(np.log10(dm_29c / dm_pdg)) if dm_29c > 0 else None
        log_528 = float(np.log10(dm_528 / dm_pdg)) if dm_528 > 0 else None
        rows.append({
            "name": name,
            "knots": cfg["knots"],
            "mass_pdg_MeV": cfg["mass_pdg"],
            "delta_m_v29c_MeV": dm_29c,
            "delta_m_v528_MeV": dm_528,
            "delta_m_pdg_MeV": dm_pdg,
            "log_ratio_v29c": log_29c,
            "log_ratio_v528": log_528,
            "N_paths": res_29c["N_paths"],
            "S_min": res_29c["S_min"],
            "vol_tot": res_29c["vol_tot"],
            "P_Kashaev": res_29c["P_Kashaev"],
            "kappa_K": res_29c["kappa_K"],
        })
    return rows


def calibrate_kappa_K_to_hierarchy(
    kappa_range: tuple[float, float] = (-1.0, 2.0),
    n_grid: int = 401,
) -> dict:
    """Cherche κ_K qui minimise Σ log² (Δm V5.29c / Δm PDG) sur D, B⁰, B_s⁰.

    K⁰ est calibré par construction (préfacteur global), il ne contribue
    pas à la fonction-coût. La recherche est un scan 1D sur κ_K.

    Returns
    -------
    dict
        { kappa_K, chi2_log, per_meson }
    """
    grid = np.linspace(kappa_range[0], kappa_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, kappa in enumerate(grid):
        s = 0.0
        ok = True
        for name in ("D0", "B0", "Bs0"):
            cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
            res = delta_m_chiral_oscillation_v29c(
                cfg["knots"], cfg["mass_pdg"], float(kappa)
            )
            if res["delta_m_MeV"] <= 0 or not np.isfinite(res["delta_m_MeV"]):
                ok = False
                break
            log_r = np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name])
            s += log_r * log_r
        if ok:
            chi2[j] = s
    idx = int(np.argmin(chi2))
    best = float(grid[idx])
    per = {}
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_chiral_oscillation_v29c(
            cfg["knots"], cfg["mass_pdg"], best
        )
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name])),
        }
    return {
        "kappa_K": best,
        "chi2_log": float(chi2[idx]),
        "per_meson": per,
    }
