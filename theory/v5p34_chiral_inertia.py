"""V5.34 — Inertie chirale collective + force centripète.

Insight V5.34 (utilisateur)
===========================

V5.33 a établi que l'interaction EM mutuelle entre composantes d'un
entrelacs est physiquement réelle : la matrice de couplage K_{ij} avec
des termes hors-diagonale τ_ij ∝ μ_i μ_j / r_ij² produit un mode
collectif dont l'eigenvalue maximale module Δm.  MAIS l'amplitude
réelle ne couvre qu'environ 0,5 décade entre K⁰ et B_s⁰, alors que la
hiérarchie PDG s'étale sur 3,4 décades.

Le mécanisme manquant identifié par l'utilisateur : ce qu'on calcule
en V5.33 est une **force** (couple τ_ij) sans **inertie** associée.
Or, le mode collectif K ↔ K* est un mouvement de rotation — la
chiralité globale tourne autour de zéro — qui doit obéir à

    ω_χ² = F_torque / I_χ                                (1)

où F_torque est l'amplitude scalaire du couple mutuel (V5.33) et
I_χ est l'**inertie chirale collective** du système.  Le ratio
F/I est la fréquence carrée du mode chiral.  La masse d'oscillation
suit alors :

    Δm V5.34(L) = 2 ℏ × ω_χ(L) × préfacteur(K⁰)         (2)

Comme F_torque varie peu d'un méson à l'autre (au facteur Lk_eff
près) mais que I_χ peut varier de plusieurs décades selon la forme
analytique choisie, c'est I_χ qui peut potentiellement injecter
les décades manquantes.

Cinq formules candidates I_χ
============================

Formule 1 — **Sommative simple** (« moment d'inertie additif ») :

    I_1(L) = Σ_i c_i × R_i²,  R_i = c_i / b_i

   Analogie : I = Σ m_i r_i² du solide.  Masse = c, bras de
   levier = R = c/b (proxy de la longueur de bras divisée par
   le nombre de strands).

Formule 2 — **Multiplicative log-linéaire** :

    I_2(L) = Π_i (c_i × b_i)

   Inspirée des produits d'invariants ; l'inertie totale grandit
   exponentiellement avec le nombre de composantes (extensivité
   multiplicative).

Formule 3 — **Avec coupling de linking** :

    I_3(L) = Σ_i I_i + Σ_{i<j} J_ij × |Lk_eff(K_i, K_j)|
    I_i = c_i × R_i²,  J_ij = √(c_i c_j) × b_i b_j

   Forme additive corrigée d'un terme d'interaction inter-composantes.

Formule 4 — **Chirale spécifique** :

    I_4(L) = Σ_i c_i × b_i^β × (1 + ε × amphi_i)
    amphi_i = 1 si is_amphicheiral(K_i), 0 sinon, ε = 1.

   Met l'accent sur l'amphichiralité : les composantes amphichirales
   (6_3, 4_1, 8_3, …) doublent leur contribution à I (frustration
   chirale → inertie accrue contre le flip).  β contrôle la pente.

Formule 5 — **Volume Kashaev exponentiel** :

    I_5(L) = exp(κ × vol_tot(L)),  vol_tot = Σ_i vol_hyp(K_i)

   La forme exponentielle est la SEULE qui peut naturellement
   produire plusieurs décades de variation entre K⁰ (3×6_1,
   vol=12,76) et B⁰ (3_1+6_2+7_7, vol=10,5) ; mais le scaling
   est dominé par les volumes hyperboliques tabulés, donc
   le résultat est en partie « importé de Kashaev ».

Force F_torque
==============

Pour conserver V5.33 comme référence rigoureuse, on définit la
force scalaire à partir de la matrice de couplage :

    F_torque(L) = Σ_{i≠j} |τ_ij|     (somme des amplitudes
                                       off-diagonales)
    F_torque(L) = τ_00(L)             si n_comp = 1
                  (= ω_self² du seul nœud → force = rigidité propre)

C'est la « force centripète totale » sentie par le système, qui
résiste à F = I × ω².

Calibration
===========

UN paramètre global (notation κ_34 selon la formule) calibré sur
{K⁰, D⁰, B⁰, B_s⁰} par χ²_log.  K⁰ contribue zéro par construction
(normalisation R_χ = ω_χ(L) / ω_χ(K⁰)).

Honnêteté V5.34
================

* Σ|log| < 1,0 dex sur {K, D, B_d, B_s} → **succès partiel** : on
  passe sous le plafond V5.30c/V5.33 (≈ 1,05 dex).
* Σ|log| < 0,5 dex → **breakthrough**.
* Σ|log| ≥ 1,0 dex → **plafond confirmé**, l'inertie chirale comme
  facteur multiplicatif unique ne suffit pas non plus.
* Si la formule 5 (Kashaev) gagne, c'est légitime mais à signaler :
  le succès est partiellement importé du volume hyperbolique.

Référence : theory/v5p33_mutual_EM_oscillation.py
            theory/kashaev_invariants.py (hyperbolic_volume)
            theory/jones.py (braid_index, is_amphicheiral)
            theory/knot_catalog.py (crossing_number)
"""
from __future__ import annotations

import numpy as np

from theory.jones import braid_index, is_amphicheiral, jones_polynomial
from theory.kashaev_invariants import hyperbolic_volume
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, delta_m_v5p30c
from theory.v5p33_mutual_EM_oscillation import (
    coupling_matrix,
    effective_linking,
    mutual_torque,
    self_frequency,
)


# ---------------------------------------------------------------------------
# Constantes physiques V5.34
# ---------------------------------------------------------------------------
HBAR_MEV_S = 6.582119569e-22   # MeV·s

# α de V5.33 utilisé pour construire la force F_torque (modèle dipolaire
# pleinement attractif/repoussant).  Hérité, pas refit en V5.34.
ALPHA_V5p33_TORQUE = -1.0       # |τ_ij| ne dépend pas du signe de α

# Cinq formules candidates pour I_χ
FORMULAS = ("F1_sum", "F2_mult", "F3_coupling", "F4_chiral", "F5_kashaev")

# Cache du préfacteur (ω_χ(K⁰)) par (formula, kappa)
_OMEGA_K0_CACHE: dict[tuple[str, float], float] = {}


def _clear_cache() -> None:
    _OMEGA_K0_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Inertie chirale I_χ — cinq variantes
# ---------------------------------------------------------------------------
def chiral_inertia(
    knot_names: list[str],
    formula: str = "F1_sum",
    kappa: float = 1.0,
) -> dict:
    """Inertie chirale collective I_χ(L) selon la formule choisie.

    Parameters
    ----------
    knot_names : list[str]
    formula : str
        Une des cinq formules : F1_sum, F2_mult, F3_coupling,
        F4_chiral, F5_kashaev.
    kappa : float
        Paramètre de la formule :
        * F1, F2, F3 : exposant global sur I (puissance).  kappa=1
          rend la définition canonique ; kappa libre n'ajoute pas
          d'expressivité supplémentaire car le préfacteur global
          calibré sur K⁰ absorbe toute puissance.  Conservé pour
          uniformité d'interface ; le test de fit le laisse fixe
          ou comme paramètre unique (cf. formule 5).
        * F4 : exposant β sur b dans c × b^β × (1 + amphi).
        * F5 : coefficient κ dans exp(κ × vol_tot).

    Returns
    -------
    dict
        { I_chi, formula, kappa, n_comp, components,
          vol_tot, amphi_count }
    """
    if formula not in FORMULAS:
        raise ValueError(
            f"Formule inconnue : {formula!r}. "
            f"Choix : {FORMULAS}"
        )
    n = len(knot_names)
    components = []
    for k in knot_names:
        b = braid_index(k)
        c = crossing_number(k)
        amphi = bool(is_amphicheiral(jones_polynomial(k)))
        components.append({"knot": k, "b": int(b), "c": int(c), "amphi": amphi})
    vol_tot = float(sum(hyperbolic_volume(comp["knot"]) for comp in components))
    amphi_count = sum(1 for comp in components if comp["amphi"])

    if formula == "F1_sum":
        # I = Σ c × R², R = c/b ⇒ I = Σ c³ / b²
        I = float(sum(
            comp["c"] * (comp["c"] / comp["b"]) ** 2 for comp in components
        ))
    elif formula == "F2_mult":
        # I = Π (c × b)
        prod = 1.0
        for comp in components:
            prod *= comp["c"] * comp["b"]
        I = float(prod)
    elif formula == "F3_coupling":
        # I = Σ I_i + Σ_{i<j} J_ij × |Lk_eff|
        I_self = sum(
            comp["c"] * (comp["c"] / comp["b"]) ** 2 for comp in components
        )
        I_int = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ki, kj = knot_names[i], knot_names[j]
                lk = effective_linking(ki, kj)["lk_eff"]
                J_ij = (
                    np.sqrt(components[i]["c"] * components[j]["c"])
                    * components[i]["b"] * components[j]["b"]
                )
                I_int += J_ij * abs(lk)
        I = float(I_self + I_int)
    elif formula == "F4_chiral":
        # I = Σ c × b^β × (1 + amphi).  β = kappa ici.
        beta = kappa
        I = float(sum(
            comp["c"] * (comp["b"] ** beta) * (1.0 + (1.0 if comp["amphi"] else 0.0))
            for comp in components
        ))
    else:  # F5_kashaev
        # I = exp(κ × vol_tot)
        I = float(np.exp(kappa * vol_tot))

    return {
        "I_chi": I,
        "formula": formula,
        "kappa": float(kappa),
        "n_comp": int(n),
        "components": components,
        "vol_tot": vol_tot,
        "amphi_count": int(amphi_count),
    }


# ---------------------------------------------------------------------------
# 2. Force centripète F_torque (V5.33)
# ---------------------------------------------------------------------------
def torque_force_v5p33(
    knot_names: list[str],
    alpha: float = ALPHA_V5p33_TORQUE,
) -> dict:
    """Force centripète scalaire F_torque dérivée du couplage V5.33.

    Pour n_comp ≥ 2 :
        F_torque = Σ_{i≠j} |τ_ij|  (somme des amplitudes off-diag)
    Pour n_comp = 1 :
        F_torque = ω_self²(K)      (rigidité propre = force restauratrice)

    Le signe de α n'affecte pas la somme des |τ_ij|, on prend α = -1
    par convention (cohérent avec α < 0 V5.33).

    Returns
    -------
    dict
        { F_torque, n_comp, off_diag_sum, diag_min }
    """
    n = len(knot_names)
    if n == 1:
        sf = self_frequency(knot_names[0])
        F = float(sf["omega_self_sq"])
        return {
            "F_torque": F,
            "n_comp": 1,
            "off_diag_sum": 0.0,
            "diag_min": F,
        }
    M = coupling_matrix(knot_names, alpha)["matrix"]
    off = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            off += abs(M[i, j])
    diag_min = float(np.min(np.diag(M)))
    F = float(off)
    if F == 0.0:
        # Pas de couplage net (ex. : amphichiralité écrante).  On
        # retombe sur la rigidité propre minimale pour éviter
        # ω² = 0 / I = 0.
        F = diag_min
    return {
        "F_torque": F,
        "n_comp": int(n),
        "off_diag_sum": float(off),
        "diag_min": diag_min,
    }


# ---------------------------------------------------------------------------
# 3. Fréquence de chiralité ω_χ = √(F / I)
# ---------------------------------------------------------------------------
def omega_chirality(
    knot_names: list[str],
    formula: str = "F1_sum",
    kappa: float = 1.0,
    alpha_torque: float = ALPHA_V5p33_TORQUE,
) -> dict:
    """ω_χ(L) = √(F_torque(L) / I_χ(L)) — fréquence du mode chiral.

    Returns
    -------
    dict
        { omega_chi, F_torque, I_chi, formula, kappa, n_comp }
    """
    F = torque_force_v5p33(knot_names, alpha_torque)
    I = chiral_inertia(knot_names, formula, kappa)
    if I["I_chi"] <= 0:
        raise ValueError(
            f"I_chi <= 0 pour {knot_names!r} (formule {formula}, "
            f"kappa={kappa}). Impossible de calculer ω_χ."
        )
    omega = float(np.sqrt(F["F_torque"] / I["I_chi"]))
    return {
        "omega_chi": omega,
        "F_torque": F["F_torque"],
        "I_chi": I["I_chi"],
        "formula": formula,
        "kappa": float(kappa),
        "n_comp": F["n_comp"],
        "vol_tot": I["vol_tot"],
        "amphi_count": I["amphi_count"],
    }


# ---------------------------------------------------------------------------
# 4. Calibration du préfacteur sur K⁰
# ---------------------------------------------------------------------------
def _omega_chi_K0(formula: str, kappa: float) -> float:
    """ω_χ(K⁰; formula, kappa) — référence pour la normalisation."""
    key = (formula, float(kappa))
    if key in _OMEGA_K0_CACHE:
        return _OMEGA_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = omega_chirality(cfg["knots"], formula, kappa)["omega_chi"]
    if oc <= 0:
        raise ValueError(
            f"ω_χ(K⁰) = {oc} pour formula={formula}, kappa={kappa}"
        )
    _OMEGA_K0_CACHE[key] = float(oc)
    return float(oc)


# ---------------------------------------------------------------------------
# 5. Δm V5.34
# ---------------------------------------------------------------------------
def delta_m_v5p34(
    knot_names: list[str],
    mass_link_MeV: float,
    formula: str = "F1_sum",
    kappa: float = 1.0,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.34(L) = Δm V5.30c(L) × R_χ(L), R_χ = ω_χ(L) / ω_χ(K⁰).

    V5.34 module V5.30c (préfacteur vibratoire scalaire — qui capture
    déjà ~3 décades de la hiérarchie) par le rapport R_χ entre la
    fréquence chirale du link et celle de K⁰.  C'est la même
    architecture que V5.33 (Δm = Δm V5.30c × R_EM) mais avec un
    nouveau facteur dérivé du rapport force/inertie.

    Par construction Δm V5.34(K⁰) = Δm V5.30c(K⁰) (R_χ = 1).

    Parameters
    ----------
    knot_names : list[str]
    mass_link_MeV : float
        Masse PDG du méson (transmis à V5.30c).
    formula : str
    kappa : float
    alpha_vib, kappa_K : float
        Paramètres V5.30c hérités (non refit en V5.34).

    Returns
    -------
    dict
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    oc = omega_chirality(knot_names, formula, kappa)
    oc_K0 = _omega_chi_K0(formula, kappa)
    R = oc["omega_chi"] / oc_K0
    dm = res_v530c["delta_m_MeV"] * R
    return {
        "delta_m_MeV": float(dm),
        "delta_m_v5p30c_MeV": float(res_v530c["delta_m_MeV"]),
        "R_chi": float(R),
        "omega_chi": oc["omega_chi"],
        "omega_K0": float(oc_K0),
        "F_torque": oc["F_torque"],
        "I_chi": oc["I_chi"],
        "formula": formula,
        "kappa": float(kappa),
        "n_comp": oc["n_comp"],
        "vol_tot": oc["vol_tot"],
        "amphi_count": oc["amphi_count"],
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 6. Calibration du paramètre unique par χ²_log
# ---------------------------------------------------------------------------
def calibrate_kappa_global(
    formula: str = "F5_kashaev",
    systems: list[str] | None = None,
    kappa_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 4001,
) -> dict:
    """Calibre l'unique paramètre kappa par χ²_log sur K, D, B_d, B_s.

        χ²_log(κ) = Σ_i log²( Δm V5.34(L_i; κ) / Δm_PDG_i )

    K⁰ contribue zéro par construction (normalisation).

    Returns
    -------
    dict
        { kappa, chi2_log, sum_abs_log, ratio_Bs_B, per_meson, formula }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(kappa_range[0], kappa_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, k in enumerate(grid):
        _clear_cache()
        s = 0.0
        ok = True
        try:
            for name in systems:
                cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
                res = delta_m_v5p34(
                    cfg["knots"], cfg["mass_pdg"], formula, float(k)
                )
                dm = res["delta_m_MeV"]
                if dm <= 0 or not np.isfinite(dm):
                    ok = False
                    break
                log_r = np.log10(dm / DELTA_M_PDG[name])
                s += log_r * log_r
        except (ValueError, ZeroDivisionError, OverflowError):
            ok = False
        if ok and np.isfinite(s):
            chi2[j] = s
    idx = int(np.argmin(chi2))
    best = float(grid[idx])
    _clear_cache()
    per = {}
    s_abs = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], formula, best)
        log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": log_r,
            "R_chi": res["R_chi"],
            "omega_chi": res["omega_chi"],
            "F_torque": res["F_torque"],
            "I_chi": res["I_chi"],
        }
        s_abs += abs(log_r)
    ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "kappa": best,
        "chi2_log": float(chi2[idx]),
        "sum_abs_log": float(s_abs),
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "formula": formula,
        "systems": systems,
    }


# ---------------------------------------------------------------------------
# 7. Comparaison globale des 5 formules
# ---------------------------------------------------------------------------
def compare_5_formulas(
    systems: list[str] | None = None,
) -> dict:
    """Calibre chaque formule indépendamment et compare Σ|log|.

    Plages de recherche par formule (heuristiques) :
    * F1 : κ ∈ [0.5, 2]  (exposant cosmétique sans effet : on prend
      la valeur fixe κ=1 pour neutralité ; le test est sans paramètre
      libre, ce qui est honnête : F1 n'a aucun degré de liberté
      au-delà de la normalisation K⁰).
    * F2, F3 : idem, sans paramètre libre (κ=1).
    * F4 : β ∈ [0.5, 4] (exposant sur b).
    * F5 : κ ∈ [-3, 3] (coefficient exponentiel sur vol_tot).

    Returns
    -------
    dict
        { F1_sum, F2_mult, F3_coupling, F4_chiral, F5_kashaev,
          best_formula, summary }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    results = {}
    # F1, F2, F3 : aucun degré de liberté physique (kappa=1 fixe).
    for formula in ("F1_sum", "F2_mult", "F3_coupling"):
        _clear_cache()
        per = {}
        s_abs = 0.0
        for name in ("K0", "D0", "B0", "Bs0"):
            cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
            res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], formula, 1.0)
            log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
            per[name] = {
                "delta_m_MeV": res["delta_m_MeV"],
                "delta_m_pdg_MeV": DELTA_M_PDG[name],
                "log_ratio": log_r,
                "R_chi": res["R_chi"],
                "omega_chi": res["omega_chi"],
                "F_torque": res["F_torque"],
                "I_chi": res["I_chi"],
            }
            s_abs += abs(log_r)
        ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
        results[formula] = {
            "kappa": 1.0,
            "sum_abs_log": float(s_abs),
            "ratio_Bs_B": float(ratio_BsB),
            "per_meson": per,
            "n_params": 0,
        }
    # F4 : β libre dans [-3, 8] (laisse l'optimum global apparaître)
    f4 = calibrate_kappa_global(
        formula="F4_chiral",
        systems=systems,
        kappa_range=(-3.0, 8.0),
        n_grid=4401,
    )
    f4["n_params"] = 1
    results["F4_chiral"] = f4
    # F5 : κ libre dans [-3, 3]
    f5 = calibrate_kappa_global(
        formula="F5_kashaev",
        systems=systems,
        kappa_range=(-3.0, 3.0),
        n_grid=4001,
    )
    f5["n_params"] = 1
    results["F5_kashaev"] = f5

    best_name = min(results, key=lambda f: results[f]["sum_abs_log"])
    summary = [
        {
            "formula": name,
            "sum_abs_log": results[name]["sum_abs_log"],
            "ratio_Bs_B": results[name]["ratio_Bs_B"],
            "kappa": results[name].get("kappa", 1.0),
            "n_params": results[name]["n_params"],
        }
        for name in FORMULAS
    ]
    results["best_formula"] = best_name
    results["summary"] = summary
    return results


# ---------------------------------------------------------------------------
# 8. Table V5.34 (formule donnée)
# ---------------------------------------------------------------------------
def v5p34_table(formula: str = "F5_kashaev", kappa: float = 1.0) -> list[dict]:
    """Table de comparaison Δm V5.34 vs PDG pour une formule donnée."""
    _clear_cache()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], formula, kappa)
        dm = res["delta_m_MeV"]
        dm_v530c = res["delta_m_v5p30c_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        log_r = float(np.log10(dm / dm_pdg)) if dm > 0 else None
        log_r_530c = (
            float(np.log10(dm_v530c / dm_pdg)) if dm_v530c > 0 else None
        )
        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "delta_m_v5p34_MeV": dm,
                "delta_m_v5p30c_MeV": dm_v530c,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p34": log_r,
                "log_ratio_v5p30c": log_r_530c,
                "R_chi": res["R_chi"],
                "omega_chi": res["omega_chi"],
                "F_torque": res["F_torque"],
                "I_chi": res["I_chi"],
                "n_comp": res["n_comp"],
                "vol_tot": res["vol_tot"],
                "formula": formula,
                "kappa": float(kappa),
            }
        )
    return rows
