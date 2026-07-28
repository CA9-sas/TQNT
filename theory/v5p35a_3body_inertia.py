"""V5.35a — Terme 3-corps dans l'inertie chirale.

Insight V5.35a (utilisateur)
============================

V5.34 (F3_coupling, sans paramètre libre) a établi

    I_χ^V5.34(L) = Σ_i c_i × R_i²
                 + Σ_{i<j} J_ij × |Lk_eff(K_i, K_j)|,
        R_i  = c_i / b_i,  J_ij = √(c_i·c_j) × b_i·b_j.

Résultat (rappel) — Σ|log| = 0.82 dex sur {K, D, B_d, B_s} :

    K⁰  log = 0.000   (calibré)
    D⁰  log = -0.12   (excellent)
    B_s⁰ log = -0.04  (excellent)
    B⁰  log = -0.66   (GAP RÉSIDUEL)

DIAGNOSTIC.  B⁰ = 3_1 + 6_2 + 7_7 est le seul système 3-knot
constitué de trois composantes chirales TOUTES DISTINCTES de
complexité très différente (c = 3, 6, 7).  Le coupling pairwise
ne « voit » qu'une moyenne de deux complexités à la fois et rate
l'asymétrie réelle à 3 composantes.  Les autres systèmes :

* K⁰  = 3 × 6_1  → permutation parfaite (3 composantes identiques)
* D⁰  = 3_1 + 5_1 + 7_2 → asymétrie modérée (c=3, 5, 7)
* B_s⁰ = 6_3 + 7_6 → 2 composantes seulement (pas de 3-corps)

V5.35a ajoute donc un TERME 3-CORPS à l'inertie chirale.

Formule retenue
===============

I_χ^V5.35a(L) = I_χ^V5.34(L) + γ_3 × I_3body(L)

avec

    I_3body(L) = Σ_{i<j<k} K_ijk × (c_i·c_j·c_k)^{1/2}
                 × |Lk_eff_ij · Lk_eff_jk · Lk_eff_ik|^{1/3}

    K_ijk = b_i × b_j × b_k     (structurel, fonction de b)

Ce choix est physiquement motivé :

1. **Triple produit** des linkings effectifs élevé à 1/3 : la
   géométrie d'un croisement à trois brins a un volume de l'ordre
   du produit des longueurs, dont l'échelle linéaire ∝ produit^{1/3}.
2. **Préfacteur (c_i c_j c_k)^{1/2}** : analogue exact à √(c_i c_j)
   du terme à 2-corps, étendu à 3-corps par moyenne géométrique.
3. **K_ijk = b_i b_j b_k** : pour deux composantes le coefficient
   à 2-corps est J_ij = √(c_i c_j) b_i b_j ; le 3-corps suit le même
   pattern en cohérence dimensionnelle (Σ b → produit b).

Pour B_s⁰ (n=2) → I_3body = 0 (somme vide), pas de modification.
Pour K⁰ (3 × 6_1 identiques) → I_3body amplifié de façon « plate »
puisque tout est égal ; comme on normalise R_χ = ω_χ(L)/ω_χ(K⁰),
seule la modification relative compte.  K⁰ reste à log = 0 par
construction.

Asymétrie attendue :

* K⁰ : c_tot^{1/2} × |Lk^3|^{1/3} = (6·6·6)^{1/2} × |Lk|
       = √216 × Lk_eff (très grand)
* B⁰ : (3·6·7)^{1/2} × |Lk_31_62 · Lk_62_77 · Lk_31_77|^{1/3}
       (modéré — composantes hétérogènes)
* D⁰ : intermédiaire

Donc augmenter γ_3 réduit I_3body relatif pour B⁰, augmente ω_χ(B⁰)
relatif à ω_χ(K⁰), donc augmente Δm V5.35a(B⁰).  L'écart log = -0.66
(sous-estimation) devrait se réduire.

Calibration
===========

UN paramètre libre γ_3, calibré par χ²_log sur {K⁰, D⁰, B⁰, B_s⁰}.
K⁰ contribue 0 (normalisation).  γ_3 ≥ 0 par convention physique
(l'inertie est positive).  γ_3 = 0 → V5.34 F3 exactement.

Honnêteté V5.35a
================

* Σ|log|^V5.34_F3 = 0.82 dex (baseline F3, kappa=1, sans paramètre)
* Σ|log|^V5.35a < 0.50 → SUCCESS (terme 3-corps utile)
* Σ|log|^V5.35a ≥ 0.82 → ÉCHEC (le 3-corps ne corrige rien)
* On rapporte aussi : si B⁰ s'améliore au prix d'une dégradation
  de D⁰ ou B_s⁰ → c'est un échec partiel, on le dit.

Référence : theory/v5p34_chiral_inertia.py (F3_coupling baseline)
            theory/v5p33_mutual_EM_oscillation.py (effective_linking)
            theory/topological_chiral_oscillation.py (MESON_OSCILLATION_ASSIGNMENTS)
"""
from __future__ import annotations

import numpy as np

from theory.jones import braid_index
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, delta_m_v5p30c
from theory.v5p33_mutual_EM_oscillation import effective_linking
from theory.v5p34_chiral_inertia import (
    ALPHA_V5p33_TORQUE,
    chiral_inertia as chiral_inertia_v5p34,
    torque_force_v5p33,
)


# ---------------------------------------------------------------------------
# Cache du préfacteur (ω_χ(K⁰)) par gamma_3
# ---------------------------------------------------------------------------
_OMEGA_K0_CACHE: dict[float, float] = {}


def _clear_cache() -> None:
    _OMEGA_K0_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Terme 3-corps
# ---------------------------------------------------------------------------
def three_body_term(knot_names: list[str]) -> dict:
    """Terme 3-corps I_3body(L) de l'inertie chirale V5.35a.

    Définition :
        I_3body(L) = Σ_{i<j<k} K_ijk × (c_i c_j c_k)^{1/2}
                     × |Lk_eff_ij · Lk_eff_jk · Lk_eff_ik|^{1/3}

    avec K_ijk = b_i b_j b_k.

    Pour n < 3, retourne 0 (somme vide).

    Returns
    -------
    dict
        { I_3body, n_comp, n_triplets, triplets }
        triplets : liste des contributions individuelles (debug).
    """
    n = len(knot_names)
    if n < 3:
        return {
            "I_3body": 0.0,
            "n_comp": int(n),
            "n_triplets": 0,
            "triplets": [],
        }

    # Précalculer c, b pour chaque composante
    cs = [crossing_number(k) for k in knot_names]
    bs = [braid_index(k) for k in knot_names]

    triplets: list[dict] = []
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ki, kj, kk = knot_names[i], knot_names[j], knot_names[k]
                lk_ij = effective_linking(ki, kj)["lk_eff"]
                lk_jk = effective_linking(kj, kk)["lk_eff"]
                lk_ik = effective_linking(ki, kk)["lk_eff"]
                lk_prod_abs = abs(lk_ij * lk_jk * lk_ik)
                # |Lk_eff_ij * Lk_eff_jk * Lk_eff_ik|^{1/3}
                lk_term = lk_prod_abs ** (1.0 / 3.0) if lk_prod_abs > 0 else 0.0
                # (c_i c_j c_k)^{1/2}
                c_term = float(np.sqrt(cs[i] * cs[j] * cs[k]))
                # K_ijk = b_i b_j b_k
                K_ijk = float(bs[i] * bs[j] * bs[k])
                contrib = K_ijk * c_term * lk_term
                total += contrib
                triplets.append(
                    {
                        "i": int(i),
                        "j": int(j),
                        "k": int(k),
                        "knots": (ki, kj, kk),
                        "K_ijk": K_ijk,
                        "c_term": c_term,
                        "lk_term": float(lk_term),
                        "contrib": float(contrib),
                    }
                )
    return {
        "I_3body": float(total),
        "n_comp": int(n),
        "n_triplets": len(triplets),
        "triplets": triplets,
    }


# ---------------------------------------------------------------------------
# 2. Inertie chirale V5.35a = V5.34 F3 + γ_3 × I_3body
# ---------------------------------------------------------------------------
def chiral_inertia_v5p35a(
    knot_names: list[str],
    gamma_3: float = 0.0,
) -> dict:
    """Inertie chirale V5.35a : V5.34 F3 + γ_3 × terme 3-corps.

    Parameters
    ----------
    knot_names : list[str]
    gamma_3 : float
        Coefficient du terme 3-corps.  γ_3 = 0 redonne V5.34 F3.

    Returns
    -------
    dict
        { I_chi, I_chi_v5p34, I_3body, gamma_3, n_comp, components }
    """
    base = chiral_inertia_v5p34(knot_names, formula="F3_coupling", kappa=1.0)
    tb = three_body_term(knot_names)
    I_total = float(base["I_chi"] + gamma_3 * tb["I_3body"])
    return {
        "I_chi": I_total,
        "I_chi_v5p34": float(base["I_chi"]),
        "I_3body": tb["I_3body"],
        "gamma_3": float(gamma_3),
        "n_comp": int(base["n_comp"]),
        "components": base["components"],
        "n_triplets": tb["n_triplets"],
    }


# ---------------------------------------------------------------------------
# 3. Fréquence chirale V5.35a
# ---------------------------------------------------------------------------
def omega_chirality_v5p35a(
    knot_names: list[str],
    gamma_3: float = 0.0,
    alpha_torque: float = ALPHA_V5p33_TORQUE,
) -> dict:
    """ω_χ_V5.35a = √(F_torque / I_χ_V5.35a)."""
    F = torque_force_v5p33(knot_names, alpha_torque)
    I = chiral_inertia_v5p35a(knot_names, gamma_3)
    if I["I_chi"] <= 0:
        raise ValueError(
            f"I_chi <= 0 pour {knot_names!r} (gamma_3={gamma_3}). "
            "Impossible de calculer ω_χ."
        )
    omega = float(np.sqrt(F["F_torque"] / I["I_chi"]))
    return {
        "omega_chi": omega,
        "F_torque": F["F_torque"],
        "I_chi": I["I_chi"],
        "I_chi_v5p34": I["I_chi_v5p34"],
        "I_3body": I["I_3body"],
        "gamma_3": float(gamma_3),
        "n_comp": F["n_comp"],
    }


def _omega_chi_K0(gamma_3: float) -> float:
    """ω_χ(K⁰; γ_3) — référence pour la normalisation."""
    key = float(gamma_3)
    if key in _OMEGA_K0_CACHE:
        return _OMEGA_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = omega_chirality_v5p35a(cfg["knots"], gamma_3)["omega_chi"]
    if oc <= 0:
        raise ValueError(f"ω_χ(K⁰) = {oc} pour gamma_3={gamma_3}")
    _OMEGA_K0_CACHE[key] = float(oc)
    return float(oc)


# ---------------------------------------------------------------------------
# 4. Δm V5.35a
# ---------------------------------------------------------------------------
def delta_m_v5p35a(
    knot_names: list[str],
    mass_link_MeV: float,
    gamma_3: float = 0.0,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.35a(L) = Δm V5.30c(L) × R_χ_v5p35a(L).

    R_χ_v5p35a = ω_χ_v5p35a(L) / ω_χ_v5p35a(K⁰).

    Par construction Δm V5.35a(K⁰) = Δm V5.30c(K⁰) (R_χ = 1).
    γ_3 = 0 → V5.34 F3 exactement.
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    oc = omega_chirality_v5p35a(knot_names, gamma_3)
    oc_K0 = _omega_chi_K0(gamma_3)
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
        "I_chi_v5p34": oc["I_chi_v5p34"],
        "I_3body": oc["I_3body"],
        "gamma_3": float(gamma_3),
        "n_comp": oc["n_comp"],
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 5. Calibration de γ_3 par χ²_log
# ---------------------------------------------------------------------------
def calibrate_gamma_3(
    systems: list[str] | None = None,
    gamma_range: tuple[float, float] = (0.0, 5.0),
    n_grid: int = 5001,
) -> dict:
    """Calibre γ_3 par χ²_log sur K, D, B_d, B_s.

        χ²_log(γ_3) = Σ_i log²( Δm V5.35a(L_i; γ_3) / Δm_PDG_i )

    K⁰ contribue zéro par construction (normalisation).

    Returns
    -------
    dict
        { gamma_3, chi2_log, sum_abs_log, per_meson, ratio_Bs_B,
          systems, grid_size }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(gamma_range[0], gamma_range[1], n_grid)
    chi2 = np.full_like(grid, np.inf)
    for j, g in enumerate(grid):
        _clear_cache()
        s = 0.0
        ok = True
        try:
            for name in systems:
                cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
                res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], float(g))
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
        res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], best)
        log_r = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": log_r,
            "R_chi": res["R_chi"],
            "omega_chi": res["omega_chi"],
            "F_torque": res["F_torque"],
            "I_chi": res["I_chi"],
            "I_chi_v5p34": res["I_chi_v5p34"],
            "I_3body": res["I_3body"],
        }
        s_abs += abs(log_r)
    ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "gamma_3": best,
        "chi2_log": float(chi2[idx]),
        "sum_abs_log": float(s_abs),
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "systems": systems,
        "grid_size": int(n_grid),
        "gamma_range": tuple(float(x) for x in gamma_range),
    }


# ---------------------------------------------------------------------------
# 6. Table V5.35a
# ---------------------------------------------------------------------------
def v5p35a_table(gamma_3: float) -> list[dict]:
    """Table de comparaison Δm V5.35a vs PDG pour γ_3 donné."""
    _clear_cache()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], gamma_3)
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
                "delta_m_v5p35a_MeV": dm,
                "delta_m_v5p30c_MeV": dm_v530c,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p35a": log_r,
                "log_ratio_v5p30c": log_r_530c,
                "R_chi": res["R_chi"],
                "omega_chi": res["omega_chi"],
                "F_torque": res["F_torque"],
                "I_chi": res["I_chi"],
                "I_chi_v5p34": res["I_chi_v5p34"],
                "I_3body": res["I_3body"],
                "gamma_3": float(gamma_3),
                "n_comp": res["n_comp"],
            }
        )
    return rows
