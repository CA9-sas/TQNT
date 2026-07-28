"""V5.35d — 3-corps MULTIPLICATIF RENORMALISANT sur l'inertie chirale V5.35b.

Contexte
========

V5.35b (commit 8c7ea0e) — asymétrie de complexité dans le coupling
J_ij — descend Σ|log| de V5.34 (0.820 dex) à **0.563 dex** sur
{K⁰, D⁰, B⁰, B_s⁰} avec β = -1.3 (forme « rel »).  Tous les mésons
ferment à l'exception du B⁰ :

    K⁰   log_r = +0.000  (calibré)
    D⁰   log_r = +0.000  (accidentel — exact)
    B_s⁰ log_r = -0.001  (accidentel — exact)
    B⁰   log_r = -0.563  (RÉSIDU)

V5.35a 3-corps **additif** (commit 2044036) a échoué : I_3body amplifié
indistinctement pour K⁰ (3 composantes identiques 6_1) casse la
calibration K⁰, donc le boost net sur B⁰ se trouve annulé par la
re-normalisation.

V5.35d retente le 3-corps mais de manière **multiplicative
renormalisante**, en pondérant le coupling V5.35b par un facteur
structurel f(c, b) qui **doit être nul** pour K⁰ (composantes c, b
identiques) et **non-nul** pour B⁰ (composantes hétérogènes).

Idée géométrique
================

Le 3-corps multiplicatif s'écrit :

    I_χ^V5.35d(L) = I_χ^V5.35b(L) × ∏_{i<j<k} [ 1 + δ × f(c_i, b_i; c_j, b_j; c_k, b_k) ]

Pour n < 3 (cas B_s⁰) le produit est vide → I_χ^V5.35d = I_χ^V5.35b
(aucun effet, exactement comme V5.35a 3-corps).

Pour n = 3 et f(K⁰) = 0 : I_χ^V5.35d(K⁰) = I_χ^V5.35b(K⁰).  La
normalisation R_χ = ω_χ(L) / ω_χ(K⁰) est inchangée sur K⁰, donc la
calibration est strictement préservée — c'est la propriété qui manquait
à V5.35a.

Sept formes candidates f
========================

Toutes sont des fonctions structurelles symétriques en (i, j, k).  Pour
chacune, on note c_min, c_mid, c_max les complexités triées et
similarly pour b.

* **var_c**     : f = Var(c) / mean(c)²
                  Mesure la dispersion relative des complexités.
                  K⁰ (c=6,6,6) → 0 ; D⁰ (3,5,7) → 8/3 / 25 ≈ 0.107 ;
                  B⁰ (3,6,7) → 2.89/28.4 ≈ 0.102.  Très proches → ne
                  distingue probablement pas B⁰ de D⁰.

* **seq_asym**  : f = |c_min + c_max − 2·c_mid| / c_mid
                  Asymétrie séquentielle (position du milieu).
                  K⁰ → 0 ; D⁰ (3,5,7) → 0 (milieu équilibré) ;
                  B⁰ (3,6,7) → |3+7−12|/6 = 2/6 = 0.333.  Distingue
                  D⁰ de B⁰ — candidat fort.

* **prod_norm** : f = (c_i c_j c_k) / (c_i + c_j + c_k)³
                  Produit normalisé.  K⁰ → 216/5832 = 0.037 ;
                  D⁰ (3,5,7) → 105/3375 = 0.0311 ;
                  B⁰ (3,6,7) → 126/4096 = 0.0308.  Quasi-identique
                  pour tous — n'utilise pas l'asymétrie.  **Non nul sur
                  K⁰** : pour préserver la calibration, on **soustrait
                  la valeur K⁰** (forme « prod_norm » devient
                  f − f_K⁰).

* **minmax_mid** : f = (c_max − c_mid)(c_mid − c_min) / c_mid²
                   Asymétrie min/max autour du milieu.
                   K⁰ → 0 ; D⁰ (3,5,7) → 2×2/25 = 0.16 ;
                   B⁰ (3,6,7) → 1×3/36 = 0.083.  D⁰ > B⁰ → boost
                   relatif favorable au B⁰ (effet contraire).
                   Testé par complétude — δ peut être négatif.

* **irreg_idx** : f = Σ|c_i − c_mean| / Σ c_i
                  Index d'irrégularité (MAD normalisé).
                  K⁰ → 0 ; D⁰ (3,5,7) → 4/15 ≈ 0.267 ;
                  B⁰ (3,6,7) → 2.667/16 ≈ 0.167.  D⁰ > B⁰ → idem
                  minmax_mid.

* **braid_asym** : f = Var(b) / mean(b)²
                   Asymétrie sur l'indice de tresse.
                   K⁰ (b=3,3,3) → 0 ;
                   D⁰ (b=2,2,3) → (1/9+1/9+4/9)/3 / (7/3)² → après calcul :
                     mean(b)=7/3, Var=2/9, Var/mean² = (2/9)/(49/9) = 2/49 ≈ 0.041
                   B⁰ (b=2,3,4) → mean=3, Var=2/3, Var/mean² = 2/3 / 9 = 0.074
                   B⁰ > D⁰ d'un facteur ≈ 1.8 — favorable, candidat fort.

* **braid_prod** : f = (b_i b_j b_k − b_min³) / b_min³
                   Excès du produit b sur le cas uniforme.
                   K⁰ → 0 ; D⁰ (2,2,3) → (12−8)/8 = 0.5 ;
                   B⁰ (2,3,4) → (24−8)/8 = 2.0.  B⁰ > D⁰ d'un facteur
                   4 — candidat le plus discriminant.

Le candidat 7 (« braid_prod ») exploite la structure d'enlacement
inter-brins : un triplet de braids très inégaux nécessite plus de
volumes de phase pour la coordination que des braids uniformes.

Calibration
===========

UN paramètre δ par formule.  Calibré par minimisation de Σ|log| sur
{K⁰, D⁰, B⁰, B_s⁰}.  K⁰ contribue 0 par construction (f(K⁰) = 0 ou
recentrée pour les 7 formes).

Honnêteté V5.35d
================

* Baseline V5.35b : Σ|log| = 0.563 dex (B⁰ = -0.563).
* Cible : Σ|log| < 0.4 dex ET |B⁰| < 0.2 dex → SUCCESS.
* 0.4 ≤ Σ|log| < 0.55 → PARTIAL.
* Σ|log| ≥ 0.55 → ÉCHEC (recommandation V5.36).
* Si aucune des 7 formes ne ferme B⁰ → la structure 3-corps multiplicative
  ne contient pas le bon levier — rapport HONNÊTE.

Référence : theory/v5p35b_complexity_asymmetry.py
            theory/v5p34_chiral_inertia.py
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
from theory.v5p34_chiral_inertia import (
    ALPHA_V5p33_TORQUE,
    torque_force_v5p33,
)
from theory.v5p35b_complexity_asymmetry import (
    chiral_inertia_v5p35b,
)


# ---------------------------------------------------------------------------
# Sept formes candidates f(c, b)
# ---------------------------------------------------------------------------
FORMS = (
    "var_c",
    "seq_asym",
    "prod_norm",
    "minmax_mid",
    "irreg_idx",
    "braid_asym",
    "braid_prod",
)

# Valeur de prod_norm pour K⁰ = (6, 6, 6) à soustraire pour annuler le
# 3-corps sur K⁰ (sinon la calibration K⁰ se casse).
_PROD_NORM_K0 = 216.0 / (18.0 ** 3)  # = 0.037037...


def f_triplet(
    c_i: int, c_j: int, c_k: int,
    b_i: int, b_j: int, b_k: int,
    form: str,
) -> float:
    """Évalue f(c, b) pour un triplet, selon la forme demandée.

    Toutes les formes sont symétriques en (i, j, k) ; ce qui compte est
    l'ensemble {(c_i, b_i), (c_j, b_j), (c_k, b_k)} pas l'ordre.

    Garantie : f = 0 pour le triplet K⁰ (c=6,6,6, b=3,3,3) sauf
    « prod_norm » où l'on recentre f := f − f(K⁰).

    Returns
    -------
    float
        Valeur de la fonction structurelle f.  Peut être positive ou
        négative pour prod_norm (recentrée) ; positive sinon.
    """
    if form not in FORMS:
        raise ValueError(f"Forme inconnue : {form!r}.  Choix : {FORMS}")

    cs = sorted([int(c_i), int(c_j), int(c_k)])
    bs = sorted([int(b_i), int(b_j), int(b_k)])
    c_min, c_mid, c_max = cs
    b_min, b_mid, b_max = bs
    c_mean = (c_min + c_mid + c_max) / 3.0
    b_mean = (b_min + b_mid + b_max) / 3.0

    if c_min <= 0 or b_min <= 0:
        raise ValueError(
            f"c ou b non positif : c=({c_i},{c_j},{c_k}), "
            f"b=({b_i},{b_j},{b_k})"
        )

    if form == "var_c":
        # Variance(c) / mean(c)²
        var = ((c_min - c_mean) ** 2 + (c_mid - c_mean) ** 2
               + (c_max - c_mean) ** 2) / 3.0
        return float(var / (c_mean ** 2))

    if form == "seq_asym":
        # |c_min + c_max − 2 c_mid| / c_mid
        return float(abs(c_min + c_max - 2.0 * c_mid) / c_mid)

    if form == "prod_norm":
        # (c_i c_j c_k) / (c_i + c_j + c_k)³ − f(K⁰)
        prod = c_min * c_mid * c_max
        denom = (c_min + c_mid + c_max) ** 3
        return float(prod / denom - _PROD_NORM_K0)

    if form == "minmax_mid":
        # (c_max − c_mid)(c_mid − c_min) / c_mid²
        return float((c_max - c_mid) * (c_mid - c_min) / (c_mid ** 2))

    if form == "irreg_idx":
        # Σ|c_i − c_mean| / Σ c_i
        s = abs(c_min - c_mean) + abs(c_mid - c_mean) + abs(c_max - c_mean)
        return float(s / (c_min + c_mid + c_max))

    if form == "braid_asym":
        # Variance(b) / mean(b)²
        var = ((b_min - b_mean) ** 2 + (b_mid - b_mean) ** 2
               + (b_max - b_mean) ** 2) / 3.0
        return float(var / (b_mean ** 2))

    if form == "braid_prod":
        # (b_i b_j b_k − b_min³) / b_min³ — excès relatif sur le cas uniforme
        prod = b_min * b_mid * b_max
        return float((prod - b_min ** 3) / (b_min ** 3))

    raise AssertionError(f"forme non couverte : {form}")


# ---------------------------------------------------------------------------
# Facteur 3-corps multiplicatif global Π [1 + δ f]
# ---------------------------------------------------------------------------
def three_body_mult_factor(
    knot_names: list[str],
    delta: float,
    form: str,
) -> dict:
    """Π_{i<j<k} [1 + δ × f(c_i, b_i; c_j, b_j; c_k, b_k)].

    Pour n < 3, retourne 1.0 (produit vide).

    Returns
    -------
    dict
        { factor, n_triplets, triplets }
        triplets : liste des (1 + δ f) individuels (debug).
    """
    n = len(knot_names)
    if n < 3:
        return {"factor": 1.0, "n_triplets": 0, "triplets": []}

    cs = [crossing_number(k) for k in knot_names]
    bs = [braid_index(k) for k in knot_names]
    triplets: list[dict] = []
    factor = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                f_val = f_triplet(
                    cs[i], cs[j], cs[k],
                    bs[i], bs[j], bs[k],
                    form,
                )
                local = 1.0 + delta * f_val
                # On garde la possibilité d'un facteur négatif théoriquement,
                # mais on floore à un minimum positif pour éviter I_chi ≤ 0.
                if local <= 0.0:
                    local = 1e-12
                factor *= local
                triplets.append(
                    {
                        "i": i, "j": j, "k": k,
                        "knots": (knot_names[i], knot_names[j], knot_names[k]),
                        "c_triplet": (cs[i], cs[j], cs[k]),
                        "b_triplet": (bs[i], bs[j], bs[k]),
                        "f_val": f_val,
                        "local_factor": local,
                    }
                )
    return {
        "factor": float(factor),
        "n_triplets": len(triplets),
        "triplets": triplets,
    }


# ---------------------------------------------------------------------------
# Inertie chirale V5.35d : V5.35b × facteur 3-corps multiplicatif
# ---------------------------------------------------------------------------
def chiral_inertia_v5p35d(
    knot_names: list[str],
    delta: float,
    form: str,
    beta_v5p35b: float = -1.3,
    form_v5p35b: str = "rel",
) -> dict:
    """I_χ V5.35d(L) = I_χ V5.35b(L) × Π_{i<j<k}[1 + δ f(...)].

    Paramètres β_v5p35b et form_v5p35b sont fixés au point calibré de
    V5.35b (β = -1.3, forme « rel »), gelés ici comme baseline.

    Returns
    -------
    dict
        { I_chi, I_chi_v5p35b, factor_3body, delta, form, n_comp,
          n_triplets }
    """
    base = chiral_inertia_v5p35b(knot_names, beta_v5p35b, form_v5p35b)
    tb = three_body_mult_factor(knot_names, delta, form)
    I_total = float(base["I_chi"] * tb["factor"])
    return {
        "I_chi": I_total,
        "I_chi_v5p35b": float(base["I_chi"]),
        "factor_3body": tb["factor"],
        "delta": float(delta),
        "form": form,
        "n_comp": int(base["n_comp"]),
        "n_triplets": tb["n_triplets"],
    }


# ---------------------------------------------------------------------------
# Fréquence chirale V5.35d
# ---------------------------------------------------------------------------
def omega_chirality_v5p35d(
    knot_names: list[str],
    delta: float,
    form: str,
    beta_v5p35b: float = -1.3,
    form_v5p35b: str = "rel",
    alpha_torque: float = ALPHA_V5p33_TORQUE,
) -> dict:
    F = torque_force_v5p33(knot_names, alpha_torque)
    I = chiral_inertia_v5p35d(
        knot_names, delta, form, beta_v5p35b, form_v5p35b
    )
    if I["I_chi"] <= 0:
        raise ValueError(
            f"I_chi ≤ 0 pour {knot_names!r} (δ={delta}, form={form})"
        )
    omega = float(np.sqrt(F["F_torque"] / I["I_chi"]))
    return {
        "omega_chi": omega,
        "F_torque": F["F_torque"],
        "I_chi": I["I_chi"],
        "I_chi_v5p35b": I["I_chi_v5p35b"],
        "factor_3body": I["factor_3body"],
        "delta": float(delta),
        "form": form,
        "n_comp": F["n_comp"],
    }


# Cache du préfacteur ω_χ(K⁰) par (form, delta)
_OMEGA_K0_CACHE: dict[tuple[str, float], float] = {}


def _clear_cache() -> None:
    _OMEGA_K0_CACHE.clear()


def _omega_chi_K0(form: str, delta: float) -> float:
    key = (form, float(delta))
    if key in _OMEGA_K0_CACHE:
        return _OMEGA_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = omega_chirality_v5p35d(
        cfg["knots"], delta, form
    )["omega_chi"]
    if oc <= 0:
        raise ValueError(f"ω_χ(K⁰) = {oc} (form={form}, δ={delta})")
    _OMEGA_K0_CACHE[key] = float(oc)
    return float(oc)


# ---------------------------------------------------------------------------
# Δm V5.35d — préfacteur V5.30c × R_χ V5.35d
# ---------------------------------------------------------------------------
def delta_m_v5p35d(
    knot_names: list[str],
    mass_link_MeV: float,
    delta: float,
    form: str,
    beta_v5p35b: float = -1.3,
    form_v5p35b: str = "rel",
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.35d(L) = Δm V5.30c(L) × R_χ V5.35d(L).

    R_χ V5.35d(L) = ω_χ V5.35d(L) / ω_χ V5.35d(K⁰).

    Pour les formes où f(K⁰) = 0 (var_c, seq_asym, minmax_mid, irreg_idx,
    braid_asym, braid_prod), facteur_3body(K⁰) ≡ 1, donc ω_χ V5.35d(K⁰)
    = ω_χ V5.35b(K⁰) — calibration K⁰ strictement préservée.

    Pour « prod_norm » la définition est recentrée (f − f_K⁰) pour
    donner f(K⁰) = 0 également.

    Returns
    -------
    dict
        Résultats Δm pour ce système.
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    oc = omega_chirality_v5p35d(
        knot_names, delta, form, beta_v5p35b, form_v5p35b
    )
    oc_K0 = _omega_chi_K0(form, delta)
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
        "I_chi_v5p35b": oc["I_chi_v5p35b"],
        "factor_3body": oc["factor_3body"],
        "delta": float(delta),
        "form": form,
        "n_comp": oc["n_comp"],
        "alpha_vib": float(alpha_vib),
        "kappa_K": float(kappa_K),
    }


# ---------------------------------------------------------------------------
# Calibration de δ par forme (Σ|log| sur K⁰, D⁰, B⁰, B_s⁰)
# ---------------------------------------------------------------------------
def calibrate_delta(
    form: str,
    systems: list[str] | None = None,
    delta_range: tuple[float, float] = (-10.0, 50.0),
    n_grid: int = 6001,
) -> dict:
    """Calibre δ pour une forme f donnée par Σ|log|.

    Returns
    -------
    dict
        { delta, sum_abs_log, chi2_log, ratio_Bs_B, per_meson,
          form, systems }
    """
    if systems is None:
        systems = ["K0", "D0", "B0", "Bs0"]
    grid = np.linspace(delta_range[0], delta_range[1], n_grid)
    score = np.full_like(grid, np.inf)
    for j, d in enumerate(grid):
        _clear_cache()
        s_abs = 0.0
        ok = True
        try:
            for name in systems:
                cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
                res = delta_m_v5p35d(
                    cfg["knots"], cfg["mass_pdg"], float(d), form
                )
                dm = res["delta_m_MeV"]
                if dm <= 0 or not np.isfinite(dm):
                    ok = False
                    break
                lr = np.log10(dm / DELTA_M_PDG[name])
                s_abs += abs(lr)
        except (ValueError, ZeroDivisionError, OverflowError):
            ok = False
        if ok and np.isfinite(s_abs):
            score[j] = s_abs
    idx = int(np.argmin(score))
    best = float(grid[idx])
    _clear_cache()
    per = {}
    s_abs = 0.0
    s_chi2 = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        res = delta_m_v5p35d(cfg["knots"], cfg["mass_pdg"], best, form)
        lr = float(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
        per[name] = {
            "delta_m_MeV": res["delta_m_MeV"],
            "delta_m_pdg_MeV": DELTA_M_PDG[name],
            "log_ratio": lr,
            "R_chi": res["R_chi"],
            "factor_3body": res["factor_3body"],
            "omega_chi": res["omega_chi"],
            "F_torque": res["F_torque"],
            "I_chi": res["I_chi"],
            "I_chi_v5p35b": res["I_chi_v5p35b"],
        }
        s_abs += abs(lr)
        s_chi2 += lr * lr
    ratio_BsB = per["Bs0"]["delta_m_MeV"] / per["B0"]["delta_m_MeV"]
    return {
        "delta": best,
        "sum_abs_log": float(s_abs),
        "chi2_log": float(s_chi2),
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "form": form,
        "systems": list(systems),
    }


# ---------------------------------------------------------------------------
# Calibrate best form across the 7 candidates
# ---------------------------------------------------------------------------
def calibrate_best_form(
    delta_range: tuple[float, float] = (-10.0, 50.0),
    n_grid: int = 6001,
) -> dict:
    """Calibre les 7 formes f, retourne celle minimisant Σ|log|.

    Returns
    -------
    dict
        { 'best_form': str, 'baseline_v5p35b': 0.563, '<form>': {...}, ... }
    """
    out: dict = {"baseline_v5p35b": 0.5632863317993378}
    for form in FORMS:
        out[form] = calibrate_delta(
            form, delta_range=delta_range, n_grid=n_grid
        )
    best = min(FORMS, key=lambda f: out[f]["sum_abs_log"])
    out["best_form"] = best
    return out


# ---------------------------------------------------------------------------
# Table V5.35d
# ---------------------------------------------------------------------------
def v5p35d_table(delta: float, form: str) -> list[dict]:
    """Table Δm V5.35d vs PDG pour (δ, form) donné."""
    _clear_cache()
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p35d(cfg["knots"], cfg["mass_pdg"], delta, form)
        dm = res["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "delta_m_v5p35d_MeV": dm,
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio_v5p35d": float(np.log10(dm / dm_pdg)),
                "R_chi": res["R_chi"],
                "factor_3body": res["factor_3body"],
                "omega_chi": res["omega_chi"],
                "F_torque": res["F_torque"],
                "I_chi": res["I_chi"],
                "I_chi_v5p35b": res["I_chi_v5p35b"],
                "delta": float(delta),
                "form": form,
                "n_comp": res["n_comp"],
            }
        )
    return rows
