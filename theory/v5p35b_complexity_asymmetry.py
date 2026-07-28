"""V5.35b — Asymétrie de complexité dans le coupling inertiel.

Insight V5.35b (utilisateur)
============================

V5.34 (commit 875cf3d) a établi la formule F3 (couplage de linking) :

    I_χ(L) = Σ_i c_i × R_i² + Σ_{i<j} J_ij × |Lk_eff(K_i, K_j)|
    J_ij   = √(c_i c_j) × b_i b_j

avec Σ|log| = 0.82 dex sur {K⁰, D⁰, B⁰, B_s⁰}.

Reste B⁰ sous-prédit de −0.66 dex.  Observation : B⁰ = 3_1 + 6_2 + 7_7
porte une forte ASYMÉTRIE de complexité entre composantes (c = 3, 6, 7)
alors que K⁰ = 3 × 6_1 n'a aucune asymétrie (c uniformes).

L'idée physique : quand deux composantes ont des nombres de croisement
très différents, le couplage inertiel mutuel J_ij doit être modifié par
un facteur d'asymétrie A_ij — le nœud léger « porte » l'inertie du
nœud lourd, ou inversement le couplage est plus rigide (boost) /
plus mou (damping) selon le régime.

V5.35b modifie J_ij sans toucher au reste de la chaîne V5.34 :

    J_ij V5.35b = J_ij V5.34 × A_ij(c_i, c_j; β)

Deux formes candidates pour A_ij
================================

* Forme « rel » (asymétrie relative, paramètre β additif) :

      A_ij = 1 + β × |c_i − c_j| / max(c_i, c_j)

  β = 0 redonne V5.34.  β > 0 amplifie le couplage asymétrique
  (boost d'inertie) ; β < 0 le damp.

* Forme « ratio » (ratio puissance, paramètre γ multiplicatif) :

      A_ij = (max(c_i, c_j) / min(c_i, c_j))^γ

  γ = 0 redonne V5.34.  γ > 0 boost, γ < 0 damp.

UN paramètre libre au total (β ou γ).  Par construction A_ij = 1
quand c_i = c_j → K⁰ inchangé (calibration préservée).

Calibration
===========

β (ou γ) calibré par χ²_log sur {K⁰, D⁰, B⁰, B_s⁰}.  K⁰ contribue
zéro par construction (R_χ = ω_χ(L) / ω_χ(K⁰), et K⁰ a A_ij ≡ 1
donc inchangé).

Honnêteté V5.35b
================

* Σ|log| V5.34 (F3) = 0.820 dex (référence).
* Si Σ|log| V5.35b < 0.5 dex et B⁰ ≈ 0 → succès clair.
* Si Σ|log| V5.35b ∈ [0.5, 0.8) → amélioration partielle :
  l'asymétrie capture une partie du déficit B⁰ mais pas tout.
* Si Σ|log| V5.35b ≥ 0.8 dex → l'asymétrie ne suffit pas.
* Si B⁰ reste loin de zéro même après calibration → la rigidité de
  J_ij n'est pas le bon levier pour B⁰.  Honnête à reporter.

Référence : theory/v5p34_chiral_inertia.py
            theory/v5p33_mutual_EM_oscillation.py
            theory/topological_chiral_oscillation.py
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
    torque_force_v5p33,
)


HBAR_MEV_S = 6.582119569e-22

FORMS = ("rel", "ratio")

# Cache du préfacteur ω_χ(K⁰) par (form, beta)
_OMEGA_K0_CACHE: dict[tuple[str, float], float] = {}


def _clear_cache() -> None:
    _OMEGA_K0_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Facteur d'asymétrie A_ij(c_i, c_j; β)
# ---------------------------------------------------------------------------
def asymmetry_factor(c_i: int, c_j: int, beta: float, form: str = "rel") -> float:
    """Facteur d'asymétrie A_ij(c_i, c_j; β).

    Parameters
    ----------
    c_i, c_j : int
        Nombres de croisement des deux composantes.
    beta : float
        Paramètre libre (β additif si form="rel", γ multiplicatif si
        form="ratio").
    form : str
        "rel" : A = 1 + β × |c_i − c_j| / max(c_i, c_j)
        "ratio" : A = (max / min)^β

    Returns
    -------
    float
        A_ij ≥ 0.  Par construction A_ij = 1 si c_i = c_j.
    """
    if form not in FORMS:
        raise ValueError(f"form inconnue : {form!r}.  Choix : {FORMS}")
    cmax = max(int(c_i), int(c_j))
    cmin = min(int(c_i), int(c_j))
    if cmin <= 0:
        raise ValueError(
            f"Nombre de croisement non positif : c_i={c_i}, c_j={c_j}"
        )
    if form == "rel":
        A = 1.0 + beta * (cmax - cmin) / cmax
    else:  # ratio
        A = (cmax / cmin) ** beta
    # Floor à 0 pour éviter I négatif (en pratique β > -1 garde A > 0)
    return float(max(A, 0.0))


# ---------------------------------------------------------------------------
# 2. Inertie chirale V5.35b : F3_coupling × A_ij
# ---------------------------------------------------------------------------
def chiral_inertia_v5p35b(
    knot_names: list[str],
    beta: float = 0.0,
    form: str = "rel",
) -> dict:
    """I_χ V5.35b(L) = Σ I_i + Σ_{i<j} J_ij × |Lk_eff| × A_ij(c_i, c_j; β).

    β = 0 (ou γ = 0) → redonne exactement F3_coupling V5.34.

    Returns
    -------
    dict
        { I_chi, I_self, I_int, components, pair_terms, n_comp,
          form, beta }
    """
    n = len(knot_names)
    components = []
    for k in knot_names:
        b = int(braid_index(k))
        c = int(crossing_number(k))
        components.append({"knot": k, "b": b, "c": c})

    I_self = float(sum(
        comp["c"] * (comp["c"] / comp["b"]) ** 2 for comp in components
    ))

    I_int = 0.0
    pair_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = knot_names[i], knot_names[j]
            lk = effective_linking(ki, kj)["lk_eff"]
            ci, cj = components[i]["c"], components[j]["c"]
            bi, bj = components[i]["b"], components[j]["b"]
            J_v34 = float(np.sqrt(ci * cj) * bi * bj)
            A = asymmetry_factor(ci, cj, beta, form)
            J_v35b = J_v34 * A
            term = J_v35b * abs(lk)
            I_int += term
            pair_terms.append(
                {
                    "knot_i": ki,
                    "knot_j": kj,
                    "c_i": ci,
                    "c_j": cj,
                    "lk_eff": float(lk),
                    "J_v34": J_v34,
                    "A_ij": A,
                    "J_v35b": J_v35b,
                    "term": float(term),
                }
            )
    return {
        "I_chi": float(I_self + I_int),
        "I_self": I_self,
        "I_int": float(I_int),
        "components": components,
        "pair_terms": pair_terms,
        "n_comp": int(n),
        "form": form,
        "beta": float(beta),
    }


# ---------------------------------------------------------------------------
# 3. Fréquence chirale V5.35b : ω_χ = √(F_torque / I_χ V5.35b)
# ---------------------------------------------------------------------------
def omega_chirality_v5p35b(
    knot_names: list[str],
    beta: float = 0.0,
    form: str = "rel",
    alpha_torque: float = ALPHA_V5p33_TORQUE,
) -> dict:
    F = torque_force_v5p33(knot_names, alpha_torque)
    I = chiral_inertia_v5p35b(knot_names, beta, form)
    if I["I_chi"] <= 0:
        raise ValueError(
            f"I_chi ≤ 0 pour {knot_names!r} (β={beta}, form={form})"
        )
    omega = float(np.sqrt(F["F_torque"] / I["I_chi"]))
    return {
        "omega_chi": omega,
        "F_torque": F["F_torque"],
        "I_chi": I["I_chi"],
        "I_self": I["I_self"],
        "I_int": I["I_int"],
        "form": form,
        "beta": float(beta),
        "n_comp": F["n_comp"],
    }


def _omega_chi_K0(form: str, beta: float) -> float:
    key = (form, float(beta))
    if key in _OMEGA_K0_CACHE:
        return _OMEGA_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = omega_chirality_v5p35b(cfg["knots"], beta, form)["omega_chi"]
    if oc <= 0:
        raise ValueError(f"ω_χ(K⁰) = {oc} (form={form}, β={beta})")
    _OMEGA_K0_CACHE[key] = float(oc)
    return float(oc)


# ---------------------------------------------------------------------------
# 4. Δm V5.35b — préfacteur V5.30c × R_χ V5.35b
# ---------------------------------------------------------------------------
def delta_m_v5p35b(
    knot_names: list[str],
    mass_link_MeV: float,
    beta: float = 0.0,
    form: str = "rel",
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.35b(L) = Δm V5.30c(L) × R_χ V5.35b(L).

    R_χ V5.35b(L) = ω_χ V5.35b(L) / ω_χ V5.35b(K⁰)

    Par construction K⁰ a A_ij ≡ 1 (toutes ses composantes ont c = 6),
    donc ω_χ V5.35b(K⁰) = ω_χ V5.34(K⁰) — la calibration K⁰ est
    indépendante de β.

    Returns
    -------
    dict
        { delta_m_MeV, delta_m_v5p30c_MeV, R_chi, omega_chi,
          omega_K0, F_torque, I_chi, I_int, form, beta, n_comp }
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    oc = omega_chirality_v5p35b(knot_names, beta, form)
    oc_K0 = _omega_chi_K0(form, beta)
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
        "I_self": oc["I_self"],
        "I_int": oc["I_int"],
        "form": form,
        "beta": float(beta),
        "n_comp": oc["n_comp"],
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# 5. Calibration globale du paramètre β (ou γ)
# ---------------------------------------------------------------------------
def calibrate_beta_global(
    form: str = "rel",
    systems: list[str] | None = None,
    beta_range: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 6001,
    metric: str = "sum_abs_log",
) -> dict:
    """Calibre β sur {K⁰, D⁰, B⁰, B_s⁰}.

    metric : "sum_abs_log" (Σ|log|, ℓ1) ou "chi2_log" (χ², ℓ²).
    On utilise ℓ1 par défaut : c'est la métrique cible explicite
    de V5.35b (« Σ|log| < 0.5 dex »).  ℓ² (V5.34 historique) reste
    disponible mais pénalise excessivement le résidu B⁰ et déplace
    l'optimum vers le bord de la grille.

    Returns
    -------
    dict
        { beta, chi2_log, sum_abs_log, ratio_Bs_B, per_meson, form,
          systems, metric }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    if metric not in ("sum_abs_log", "chi2_log"):
        raise ValueError(
            f"metric inconnue : {metric!r}. "
            f"Choix : 'sum_abs_log' ou 'chi2_log'"
        )
    grid = np.linspace(beta_range[0], beta_range[1], n_grid)
    score = np.full_like(grid, np.inf)
    for j, b in enumerate(grid):
        _clear_cache()
        s_chi2 = 0.0
        s_abs = 0.0
        ok = True
        try:
            for name in systems:
                cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
                res = delta_m_v5p35b(
                    cfg["knots"], cfg["mass_pdg"], float(b), form
                )
                dm = res["delta_m_MeV"]
                if dm <= 0 or not np.isfinite(dm):
                    ok = False
                    break
                lr = np.log10(dm / DELTA_M_PDG[name])
                s_chi2 += lr * lr
                s_abs += abs(lr)
        except (ValueError, ZeroDivisionError, OverflowError):
            ok = False
        if ok:
            s = s_abs if metric == "sum_abs_log" else s_chi2
            if np.isfinite(s):
                score[j] = s
    idx = int(np.argmin(score))
    best = float(grid[idx])
    _clear_cache()
    per = {}
    s_abs = 0.0
    s_chi2 = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p35b(cfg["knots"], cfg["mass_pdg"], best, form)
        lr = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": lr,
            "R_chi": res["R_chi"],
            "omega_chi": res["omega_chi"],
            "F_torque": res["F_torque"],
            "I_chi": res["I_chi"],
            "I_int": res["I_int"],
        }
        s_abs += abs(lr)
        s_chi2 += lr * lr
    ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "beta": best,
        "chi2_log": float(s_chi2),
        "sum_abs_log": float(s_abs),
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "form": form,
        "metric": metric,
        "systems": list(systems),
    }


# ---------------------------------------------------------------------------
# 6. Comparaison des deux formes "rel" et "ratio"
# ---------------------------------------------------------------------------
def compare_forms() -> dict:
    """Calibre les deux formes et retourne un résumé comparatif."""
    out = {}
    for form in FORMS:
        out[form] = calibrate_beta_global(form=form)
    best = min(out, key=lambda f: out[f]["sum_abs_log"])
    out["best_form"] = best
    return out


# ---------------------------------------------------------------------------
# 7. Table de résultats V5.35b
# ---------------------------------------------------------------------------
def v5p35b_table(beta: float, form: str = "rel") -> list[dict]:
    """Tableau Δm V5.35b vs PDG pour les 4 systèmes."""
    _clear_cache()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p35b(cfg["knots"], cfg["mass_pdg"], beta, form)
        dm = res["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "delta_m_v5p35b_MeV": dm,
                "delta_m_v5p30c_MeV": res["delta_m_v5p30c_MeV"],
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p35b": float(np.log10(dm / dm_pdg)),
                "R_chi": res["R_chi"],
                "omega_chi": res["omega_chi"],
                "F_torque": res["F_torque"],
                "I_chi": res["I_chi"],
                "I_int": res["I_int"],
                "form": form,
                "beta": float(beta),
                "n_comp": res["n_comp"],
            }
        )
    return rows
