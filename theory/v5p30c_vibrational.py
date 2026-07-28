"""V5.30c — Modes vibratoires du nœud comme préfacteur dynamique.

Cadre V5.30c
============

V5.30 (commit 42ddf0c) combine V5.29a (chemins frustrés amphichiraux) et
V5.29c (préfacteur Kashaev / volume) avec un seul paramètre libre κ_K :

    Δm V5.30 = 2 ℏ ω_0 × N_paths^V5.29a × exp(-S_min/ℏ) × exp(κ_K · vol_tot)

Résultats V5.30 (κ_K = 0.611) : Σ|log_ratio| = 1.96 dex (vs V5.28 = 6.00),
hiérarchie K<D<B<Bs reproduite, **mais** ratio B_s/B = 1471 (PDG = 35) :
sur-amplification d'un facteur ×42.

Diagnostic statique
-------------------
La rigidité statique (topologie + frustration + Kashaev) ne distingue pas
« Bs absolu correct » et « ratio Bs/B correct ».  Tous les leviers V5.30
agissent dans le même sens : ils amplifient Bs⁰ plus que B⁰ parce que 6_3
est amphichiral (F=2880) et que vol(6_3+7_6) > vol(3_1+6_2+7_7).

Il manque un mécanisme **dynamique** qui dépende de la structure de
couplage entre composantes, pas seulement de la somme topologique.

Hypothèse V5.30c — modes vibratoires
====================================

Un nœud (et un entrelacs de nœuds) n'est pas un objet rigide : c'est un
tube élastique fermé qui supporte des modes propres de vibration.

Pour une composante isolée K_i :

    ω_0(K_i) ∝ b(K_i) / √(c(K_i))
        (rapport tension topologique / masse linéique du tube)

* le numérateur b(K_i) = indice de tresse mesure la « tension topologique »
  (rigidité torsionnelle du strand-bundle)
* le dénominateur √c(K_i) est proxy de √m_L ∝ √(longueur de tube)

Modes harmoniques : ω_n = n · ω_0(K_i)
Modes intra-composante : Π_n ω_n^α (sur n_modes)

Pour un entrelacs L = K_1 + ... + K_{n_comp} :

* modes INTRA : un bouquet de modes ω_0(K_i) par composante
* modes INTER : couplage rigide entre composantes, dont la dureté
  monte avec le nombre n_comp de composantes (rigidité ∝ n_comp).
  Plus il y a de composantes, plus le système est gelé.

Forme adoptée (parameter-frugal, 1 seul nouveau paramètre α)
-----------------------------------------------------------

On combine intra et inter en un seul scalaire par link :

    Ω_link(L) = n_comp(L) · ⟨ω_0⟩_geom(L)
    ⟨ω_0⟩_geom(L) = ( Π_i b_i / √c_i )^{1/n_comp}     (moyenne géométrique)

et le préfacteur vibratoire dynamique :

    P_vib(L) = Ω_link(L)^α

calibré par K⁰ (absorption dans le préfacteur global), donc on travaille
effectivement avec :

    R_vib(L) = ( Ω_link(L) / Ω_link(K⁰) )^α

Formule V5.30c finale
=====================

    Δm V5.30c(L) = Δm V5.30(L) × R_vib(L)
                 = 2 ℏ ω_0_calibré × N_paths · exp(-S_min) · exp(κ_K vol_tot)
                       × (n_comp · ⟨b/√c⟩_geom / Ω_K0)^α

Honnêteté V5.30c
================

* α est calibré globalement (χ²_log) sur K⁰, D⁰, B⁰, B_s⁰.
* κ_K reste fixé à sa valeur V5.30 calibrée (= 0.611) — V5.30c ne refit
  pas κ_K, sinon on aurait 2 paramètres de plus que V5.28.
* V5.30c teste : un seul paramètre dynamique additionnel suffit-il à
  améliorer simultanément l'absolu Bs et le ratio Bs/B ?
* SUCCESS : tous |log_ratio| < 0.5 dex ET |log10(ratio_Bs_B / 35)| < 0.3.
* PARTIAL : amélioration de Σ|log| sans fermer tous les critères.
* FAILURE : V5.30c casse un système relatif à V5.30.

Référence : theory/v5p30_combined_oscillation.py (V5.30)
            theory/v5p29_frustrated_amphichiral.py (V5.29a)
            theory/topological_chiral_oscillation.py (V5.28)
            theory/jones.py (braid_index, is_amphicheiral)
            theory/knot_catalog.py (crossing_number)
"""
from __future__ import annotations

import numpy as np

from theory.jones import braid_index
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import (
    KAPPA_K_V5p30_DEFAULT,
    delta_m_v5p30,
)


# ---------------------------------------------------------------------------
# Constantes V5.30c
# ---------------------------------------------------------------------------
# α calibré globalement par χ²_log sur K, D, B⁰, B_s⁰ (voir
# calibrate_alpha_global()).  La valeur par défaut est mise à jour après
# la première calibration.  Forme physique : P_vib = (n_comp · ⟨ω₀⟩)^α.
ALPHA_VIB_DEFAULT = 9.43

# Préfacteur global V5.30c (cache par (κ_K, α))
_PREFACTOR_V5p30c_MEV: dict[tuple[float, float], float] = {}


# ---------------------------------------------------------------------------
# 1. Fréquence vibratoire fondamentale d'une composante
# ---------------------------------------------------------------------------
def vibrational_frequency(knot_name: str) -> dict:
    """Fréquence vibratoire fondamentale ω_0(K) d'une composante.

    Modèle de corde-tube fermée :

        ω_0(K) = b(K) / √c(K)

    Interprétation :

    * b(K) = indice de tresse — proxy de la tension topologique
      (rigidité torsionnelle du tube nodal).  Plus de tresses ⇒ plus
      de strands en parallèle ⇒ tension transverse plus haute.
    * c(K) = nombre de croisements — proxy de la longueur totale L_tot
      (chaque croisement ajoute ~ une longueur unité de tube).
    * m_L ∝ L_tot ∝ c(K)  ⇒  ω_0 ∝ √(T/m_L) ∝ b / √c.

    Sans dimension physique : ω_0(K) est une fréquence topologique
    normalisée (en unités de la fréquence de référence du tube).

    Returns
    -------
    dict
        { knot, b, c, omega0 }
    """
    b = braid_index(knot_name)
    c = crossing_number(knot_name)
    omega0 = b / np.sqrt(c)
    return {
        "knot": knot_name,
        "b": int(b),
        "c": int(c),
        "omega0": float(omega0),
    }


# ---------------------------------------------------------------------------
# 2. Facteur de couplage inter-composantes
# ---------------------------------------------------------------------------
def coupling_factor(knot_names: list[str]) -> dict:
    """Facteur de couplage inter-composantes du link L.

    Modèle : un link à n_comp composantes possède n_comp − 1 modes de
    couplage inter-composantes (équivalent à n_comp − 1 ressorts entre
    composantes), plus le mode global de translation rigide.  La
    rigidité collective scale globalement comme n_comp.

        C_inter(L) = n_comp(L)

    Pour les composantes intra, on prend la moyenne géométrique des
    fréquences fondamentales ω_0(K_i), pour pondérer également chaque
    composante :

        ⟨ω_0⟩_geom(L) = ( Π_i ω_0(K_i) )^{1/n_comp}

    Et la fréquence vibratoire effective du link :

        Ω_link(L) = C_inter(L) · ⟨ω_0⟩_geom(L)
                  = n_comp · ( Π_i b_i / √c_i )^{1/n_comp}

    Returns
    -------
    dict
        { n_comp, omega0_geom, omega_link, per_component }
    """
    per = [vibrational_frequency(n) for n in knot_names]
    n_comp = len(knot_names)
    # Moyenne géométrique des ω_0(K_i)
    log_sum = sum(np.log(p["omega0"]) for p in per)
    omega0_geom = float(np.exp(log_sum / n_comp))
    omega_link = float(n_comp * omega0_geom)
    return {
        "n_comp": int(n_comp),
        "omega0_geom": omega0_geom,
        "omega_link": omega_link,
        "per_component": per,
    }


# ---------------------------------------------------------------------------
# 3. Préfacteur vibratoire P_vib(L; α)
# ---------------------------------------------------------------------------
def p_vibrational(knot_names: list[str], alpha: float) -> dict:
    """Préfacteur vibratoire dynamique P_vib(L) = Ω_link(L)^α.

    Le préfacteur est dimensionné par calibration sur K⁰ (cf.
    `_calibrate_prefactor_v5p30c`) : ce qui compte physiquement est le
    rapport R_vib(L) = P_vib(L) / P_vib(K⁰).

    Returns
    -------
    dict
        { p_vib, omega_link, n_comp, alpha }
    """
    cpl = coupling_factor(knot_names)
    p_vib = float(cpl["omega_link"] ** alpha)
    return {
        "p_vib": p_vib,
        "omega_link": cpl["omega_link"],
        "omega0_geom": cpl["omega0_geom"],
        "n_comp": cpl["n_comp"],
        "alpha": float(alpha),
        "per_component": cpl["per_component"],
    }


# ---------------------------------------------------------------------------
# 4. Calibration du préfacteur global V5.30c sur K⁰
# ---------------------------------------------------------------------------
def _calibrate_prefactor_v5p30c(kappa_K: float, alpha: float) -> float:
    """Calibre le préfacteur global V5.30c (ω_0·ℏ·R_vib(K⁰)) sur K⁰ PDG.

    En appliquant P_vib(K⁰) à Δm_K⁰_V5.30 (qui était déjà calibré à
    Δm_PDG), il faut renormaliser pour préserver l'exactitude sur K⁰.
    """
    key = (float(kappa_K), float(alpha))
    if key in _PREFACTOR_V5p30c_MEV:
        return _PREFACTOR_V5p30c_MEV[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    dm_v530_K = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], kappa_K)[
        "delta_m_MeV"
    ]
    pv_K = p_vibrational(cfg["knots"], alpha)["p_vib"]
    # Δm V5.30c(K⁰) doit valoir Δm_PDG(K⁰)
    # Δm V5.30c(L) = Δm V5.30(L) × (P_vib(L) / P_vib(K⁰)) × pref_corr
    # avec pref_corr = 1 puisque V5.30 calibre K⁰ exactement.  Le ratio
    # R_vib(K⁰) = P_vib(K⁰)/P_vib(K⁰) = 1, donc V5.30c(K⁰) = V5.30(K⁰) =
    # PDG(K⁰).  On stocke P_vib(K⁰) comme dénominateur.
    _PREFACTOR_V5p30c_MEV[key] = float(pv_K)
    return float(pv_K)


# ---------------------------------------------------------------------------
# 5. Δm V5.30c
# ---------------------------------------------------------------------------
def delta_m_v5p30c(
    knot_names: list[str],
    mass_link_MeV: float,
    alpha: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.30c = Δm V5.30 × R_vib(L) = Δm V5.30 × (Ω_link / Ω_K0)^α.

    Parameters
    ----------
    knot_names : list[str]
    mass_link_MeV : float
    alpha : float
        Paramètre vibratoire libre V5.30c (calibré globalement).
    kappa_K : float
        Paramètre V5.30 (Kashaev), conservé à sa valeur calibrée V5.30.

    Returns
    -------
    dict
        { delta_m_MeV, delta_m_v530_MeV, R_vib, P_vib, omega_link, ...}
    """
    res_v530 = delta_m_v5p30(knot_names, mass_link_MeV, kappa_K)
    pv_K = _calibrate_prefactor_v5p30c(kappa_K, alpha)
    pv = p_vibrational(knot_names, alpha)
    R_vib = pv["p_vib"] / pv_K
    dm = res_v530["delta_m_MeV"] * R_vib
    return {
        "delta_m_MeV": float(dm),
        "delta_m_v530_MeV": float(res_v530["delta_m_MeV"]),
        "R_vib": float(R_vib),
        "P_vib": pv["p_vib"],
        "P_vib_K0": float(pv_K),
        "omega_link": pv["omega_link"],
        "omega0_geom": pv["omega0_geom"],
        "n_comp": pv["n_comp"],
        "alpha": float(alpha),
        "kappa_K": float(kappa_K),
        "N_paths": res_v530["N_paths"],
        "F_total": res_v530["F_total"],
        "vol_tot": res_v530["vol_tot"],
        "P_Kashaev": res_v530["P_Kashaev"],
        "S_min": res_v530["S_min"],
        "per_component": pv["per_component"],
        "components": res_v530["components"],
    }


# ---------------------------------------------------------------------------
# 6. Calibration globale de α (V5.30c)
# ---------------------------------------------------------------------------
def calibrate_alpha_global(
    systems: list[str] | None = None,
    alpha_range: tuple[float, float] = (-5.0, 15.0),
    n_grid: int = 4001,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Calibre α globalement par χ²_log sur les systèmes donnés.

        χ²_log(α) = Σ_i log²( Δm V5.30c(L_i) / Δm PDG_i )

    K⁰ contribue zéro par construction (R_vib(K⁰) = 1).  Les contributions
    effectives viennent de D⁰, B⁰, B_s⁰.

    Returns
    -------
    dict
        { alpha, chi2_log, per_meson, sum_abs_log }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(alpha_range[0], alpha_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, a in enumerate(grid):
        s = 0.0
        ok = True
        for name in systems:
            cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
            res = delta_m_v5p30c(
                cfg["knots"], cfg["mass_pdg"], float(a), kappa_K
            )
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
    s_abs = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p30c(cfg["knots"], cfg["mass_pdg"], best, kappa_K)
        log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": log_r,
            "R_vib": res["R_vib"],
        }
        s_abs += abs(log_r)
    return {
        "alpha": best,
        "chi2_log": float(chi2[idx]),
        "sum_abs_log": float(s_abs),
        "per_meson": per,
        "systems": systems,
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 7. Table de comparaison V5.30c vs V5.30 vs PDG
# ---------------------------------------------------------------------------
def v5p30c_table(
    alpha: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> list[dict]:
    """Table de comparaison Δm V5.30c vs V5.30 vs PDG.

    Returns
    -------
    list of dicts : name, knots, delta_m_v5p30c, delta_m_v5p30,
        delta_m_pdg, log_ratio_v5p30c, log_ratio_v5p30, R_vib,
        omega_link, n_comp, alpha.
    """
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res_v530c = delta_m_v5p30c(
            cfg["knots"], cfg["mass_pdg"], alpha, kappa_K
        )
        res_v530 = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], kappa_K)
        dm_v530c = res_v530c["delta_m_MeV"]
        dm_v530 = res_v530["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]

        def _log(x: float) -> float | None:
            return float(np.log10(x / dm_pdg)) if x > 0 else None

        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "mass_pdg_MeV": cfg["mass_pdg"],
                "delta_m_v5p30c_MeV": dm_v530c,
                "delta_m_v5p30_MeV": dm_v530,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p30c": _log(dm_v530c),
                "log_ratio_v5p30": _log(dm_v530),
                "R_vib": res_v530c["R_vib"],
                "P_vib": res_v530c["P_vib"],
                "omega_link": res_v530c["omega_link"],
                "omega0_geom": res_v530c["omega0_geom"],
                "n_comp": res_v530c["n_comp"],
                "F_total": res_v530c["F_total"],
                "vol_tot": res_v530c["vol_tot"],
                "alpha": float(alpha),
                "kappa_K": float(kappa_K),
            }
        )
    return rows
