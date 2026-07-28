"""V5.29b — Coupling c_J asymétrique inter-composantes.

Motivation physique
===================

V5.28 (`theory/topological_chiral_oscillation.py`) écrit l'action
topologique du chemin minimum comme une somme **additive** sur les
composantes :

    S_min/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)        (V5.28)

Cette forme suppose que chaque composante du link contribue
indépendamment à l'action de tunneling chiral L ↔ L*.  Physiquement,
les composantes d'un link entrelacé n'oscillent PAS indépendamment :
leur couplage topologique dépend de l'**asymétrie** de leurs facteurs de
Jones c_J,i.  Deux composantes à c_J très voisins ont des fréquences
topologiques (ω_i ∝ c_J,i) proches et couplent bien ; deux composantes à
c_J très différents (e.g. 3_1 avec c_J=3.3 vs 7_2 avec c_J=8.1) ont des
fréquences détunées et couplent **moins bien** — leur tunneling effectif
coûte plus d'action.

V5.29b ajoute donc un terme de couplage inter-composantes :

    S_min/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)
            + κ_C × Σ_{i<j} |c_J,i − c_J,j|             (V5.29b, linéaire)

où κ_C est un **nouveau paramètre** calibré sur l'observation
expérimentale D⁰ (V5.28 sur-prédit D⁰ de +1,97 dex, à corriger).

Honnêteté V5.29b
================

* κ_C est **un paramètre supplémentaire** par rapport à V5.28 (donc V5.29b a
  3 paramètres contre 2 en V5.28 et 1 en V5p27, pour 4 mesures K, D, B, Bs).
* L'asymétrie |c_J,i − c_J,j| est **symétrique** sur i ↔ j — elle ne sait
  pas distinguer "petit + grand + grand" (B⁰ : 3.3, 7.1, 9.1) de
  "petit + moyen + grand" (D⁰ : 3.3, 5.5, 8.1).  En fait B⁰ a une
  somme |Δc_J| **supérieure** à D⁰ (11.6 vs 9.6).  Conséquence
  prédictible : si κ_C est calibré sur D⁰, B⁰ encaisse une pénalité
  encore plus forte → le gap B⁰ que V5.28 avait fermé risque de
  rouvrir.  Ce module l'expose explicitement ; ne pas masquer ce résultat.
* Pour K⁰ (3 composantes identiques 6_1 + 6_1 + 6_1), |Δc_J,i_j| = 0 pour
  toute paire → S_min_V5p29b = S_min_V5.28.  La calibration de K⁰
  (préfacteur ω_0·ℏ) reste identique : V5.29b colle automatiquement à
  PDG sur K⁰ par construction.

Référence : theory/topological_chiral_oscillation.py (V5.28).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from theory.jones import jones_polynomial
from theory.knot_catalog import crossing_number
from theory.retraction import retraction_factor
from theory.topological_chiral_oscillation import (
    KAPPA_S,
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    count_paths_K_to_Kstar,
)


# ---------------------------------------------------------------------------
# Paramètres V5.29b
# ---------------------------------------------------------------------------
# κ_C : intensité du couplage c_J asymétrique.  Calibré sur D⁰ par
# ``calibrate_kappa_C()``.  La valeur par défaut ci-dessous est la valeur
# calibrée (gardée comme constante pour reproductibilité des tests ; un appel
# à ``calibrate_kappa_C()`` recalcule la calibration à partir de zéro).
KAPPA_C: float = 0.4734689527187159

# Forme de couplage par défaut : 'linear' = |c_J,i − c_J,j|.  Autres formes
# disponibles :
#   'quadratic'   : (c_J,i − c_J,j)²
#   'logratio'    : |log(c_J,i / c_J,j)|
COUPLING_FORM_DEFAULT = "linear"

# Cache du préfacteur ω_0·ℏ (calibré sur K⁰).
_PREFACTOR_MEV: float | None = None


# ---------------------------------------------------------------------------
# 1. Facteur de couplage c_J asymétrique
# ---------------------------------------------------------------------------
def _pair_coupling(cJ_i: float, cJ_j: float, form: str) -> float:
    """Élément pairwise du couplage c_J asymétrique."""
    if form == "linear":
        return abs(cJ_i - cJ_j)
    if form == "quadratic":
        return (cJ_i - cJ_j) ** 2
    if form == "logratio":
        if cJ_i <= 0 or cJ_j <= 0:
            return 0.0
        return abs(np.log(cJ_i / cJ_j))
    raise ValueError(f"Unknown coupling form: {form}")


def asymmetry_coupling(
    knot_names: Sequence[str],
    kappa_C: float = None,
    form: str = COUPLING_FORM_DEFAULT,
) -> dict:
    """Couplage c_J asymétrique entre composantes du link.

    Formule (V5.29b, linéaire) :

        ΔS_couplage / ℏ = κ_C × Σ_{i<j} |c_J,i − c_J,j|

    Le couplage s'annule pour des composantes à c_J identiques (e.g. K⁰
    = 6_1 + 6_1 + 6_1) : la sym factor combinatoire de N_paths n'est PAS
    redondante, c'est une suppression indépendante.

    Parameters
    ----------
    knot_names : sequence of str
        Composantes du link (e.g. ["3_1", "5_1", "7_2"] pour D⁰).
    kappa_C : float, optional
        Si None, utilise ``KAPPA_C`` (calibré sur D⁰).
    form : {"linear", "quadratic", "logratio"}
        Forme du couplage pairwise.

    Returns
    -------
    dict
        { delta_S, asymmetry_raw, pairs, c_J_values }
    """
    if kappa_C is None:
        kappa_C = KAPPA_C

    cJ_values = [retraction_factor(jones_polynomial(n)) for n in knot_names]
    n = len(cJ_values)

    pairs = []
    asymmetry_raw = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            contrib = _pair_coupling(cJ_values[i], cJ_values[j], form)
            pairs.append({
                "i": i,
                "j": j,
                "knot_i": knot_names[i],
                "knot_j": knot_names[j],
                "c_J_i": cJ_values[i],
                "c_J_j": cJ_values[j],
                "contribution": contrib,
            })
            asymmetry_raw += contrib

    delta_S = kappa_C * asymmetry_raw
    return {
        "delta_S": float(delta_S),
        "asymmetry_raw": float(asymmetry_raw),
        "pairs": pairs,
        "c_J_values": cJ_values,
        "kappa_C": kappa_C,
        "form": form,
    }


# ---------------------------------------------------------------------------
# 2. Action totale V5.29b
# ---------------------------------------------------------------------------
def action_min_v5p29b(
    knot_names: Sequence[str],
    kappa_C: float = None,
    form: str = COUPLING_FORM_DEFAULT,
) -> dict:
    """Action topologique totale V5.29b (additive + couplage c_J).

    Formule :

        S_min/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)         (terme V5.28)
                + κ_C × Σ_{i<j} |c_J,i − c_J,j|        (couplage V5.29b)

    Note : on conserve γ_M = 0 (rejet de l'hypothèse WKB de V5.28).

    Returns
    -------
    dict
        { S_min, S_topo_additive, S_coupling, ... }
    """
    # Terme additif V5.28
    s_additive = 0.0
    for n in knot_names:
        c = crossing_number(n)
        cJ = retraction_factor(jones_polynomial(n))
        s_additive += c * np.log1p(cJ)
    s_additive *= KAPPA_S

    # Terme de couplage V5.29b
    coup = asymmetry_coupling(knot_names, kappa_C=kappa_C, form=form)

    S_min = s_additive + coup["delta_S"]
    return {
        "S_min": float(S_min),
        "S_topo_additive": float(s_additive),
        "S_coupling": float(coup["delta_S"]),
        "asymmetry_raw": coup["asymmetry_raw"],
        "kappa_C": coup["kappa_C"],
        "form": coup["form"],
        "c_J_values": coup["c_J_values"],
    }


# ---------------------------------------------------------------------------
# 3. Calibration du préfacteur sur K⁰ (V5.29b)
# ---------------------------------------------------------------------------
def _calibrate_prefactor_v5p29b(
    kappa_C: float = None, form: str = COUPLING_FORM_DEFAULT
) -> float:
    """Calibre le préfacteur global ω_0·ℏ sur Δm_K⁰ (PDG).

    Pour K⁰ (6_1 × 3), le couplage c_J asymétrique vaut 0 (composantes
    identiques) → S_min_v29b(K⁰) = S_min_v28(K⁰), et le préfacteur
    coïncide avec celui de V5.28.  Cette routine recalcule néanmoins le
    préfacteur explicitement (sans réutiliser le cache de V5.28) pour
    isoler V5.29b.
    """
    global _PREFACTOR_MEV
    if _PREFACTOR_MEV is not None:
        return _PREFACTOR_MEV
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    paths = count_paths_K_to_Kstar(cfg["knots"])
    action = action_min_v5p29b(cfg["knots"], kappa_C=kappa_C, form=form)
    me = paths["N_paths"] * np.exp(-action["S_min"])
    target = DELTA_M_PDG["K0"]
    _PREFACTOR_MEV = target / (2.0 * me)
    return _PREFACTOR_MEV


def _reset_prefactor_cache() -> None:
    """Réinitialise le cache du préfacteur (utile pour tests / recalibration)."""
    global _PREFACTOR_MEV
    _PREFACTOR_MEV = None


# ---------------------------------------------------------------------------
# 4. Δm V5.29b
# ---------------------------------------------------------------------------
def delta_m_v5p29b(
    knot_names: Sequence[str],
    mass_link_MeV: float = None,
    kappa_C: float = None,
    form: str = COUPLING_FORM_DEFAULT,
) -> dict:
    """Différence de masse d'oscillation chirale V5.29b.

    Δm = 2 × prefactor × N_paths × exp(-S_min/ℏ)

    où S_min inclut le couplage c_J asymétrique.

    Parameters
    ----------
    knot_names : sequence of str
        Composantes du link.
    mass_link_MeV : float, optional
        Non utilisé (γ_M = 0 → pas de dépendance en masse).  Gardé pour
        signature compatible avec V5.28.
    kappa_C : float, optional
        Si None, utilise ``KAPPA_C`` (calibré sur D⁰).
    form : str
        Forme du couplage pairwise.

    Returns
    -------
    dict
        { delta_m_MeV, N_paths, S_min, ...}
    """
    pref = _calibrate_prefactor_v5p29b(kappa_C=kappa_C, form=form)
    paths = count_paths_K_to_Kstar(list(knot_names))
    action = action_min_v5p29b(list(knot_names), kappa_C=kappa_C, form=form)
    me = paths["N_paths"] * np.exp(-action["S_min"])
    delta_m = 2.0 * pref * me
    return {
        "delta_m_MeV": float(delta_m),
        "N_paths": paths["N_paths"],
        "S_min": action["S_min"],
        "S_topo_additive": action["S_topo_additive"],
        "S_coupling": action["S_coupling"],
        "matrix_element": float(me),
        "prefactor_MeV": pref,
        "kappa_C": action["kappa_C"],
        "form": action["form"],
        "components": [(n, crossing_number(n)) for n in knot_names],
        "c_J_values": action["c_J_values"],
    }


# ---------------------------------------------------------------------------
# 5. Calibration de κ_C sur D⁰
# ---------------------------------------------------------------------------
def calibrate_kappa_C(form: str = COUPLING_FORM_DEFAULT) -> float:
    """Calibre κ_C pour que Δm_D⁰ V5.29b colle à PDG.

    Procédure :

    1.  V5.28 sur-prédit Δm_D⁰ de +1.97 dex → on cherche κ_C tel que
        ΔS_couplage / ln(10) ≈ 1.97  (couvre exp(-ΔS) ≈ 10^(-1.97)).
    2.  Le préfacteur reste calibré sur K⁰ (où le couplage est nul) :
        K⁰ n'est PAS affecté par κ_C.
    3.  La solution est κ_C = (1.97 × ln(10)) / asymmetry_raw(D⁰).

    Returns
    -------
    float
        Valeur calibrée de κ_C.
    """
    cfg_D = MESON_OSCILLATION_ASSIGNMENTS["D0"]
    coup_D = asymmetry_coupling(cfg_D["knots"], kappa_C=1.0, form=form)
    # Pour V5.28, D⁰ est à log_ratio = +1.974 (commit 962c96f).
    log_ratio_v528_D0 = 1.974
    # On veut S_couplage = ln(10) × log_ratio (correction de 1.97 dex
    # en facteur exp(-ΔS)).
    target_delta_S = log_ratio_v528_D0 * np.log(10.0)
    return target_delta_S / coup_D["asymmetry_raw"]


# ---------------------------------------------------------------------------
# 6. Table de comparaison V5.29b vs V5.28 vs PDG
# ---------------------------------------------------------------------------
def v5p29b_oscillation_table(
    kappa_C: float = None, form: str = COUPLING_FORM_DEFAULT
) -> list[dict]:
    """Table V5.29b vs V5.28 vs PDG pour K⁰, D⁰, B⁰, Bs⁰.

    Renvoie une liste de dicts contenant les Δm V5.29b et leur log_ratio
    PDG ; le log_ratio_v528 est repris du commit 962c96f (V5.28).
    """
    # V5.28 reference numbers (commit 962c96f), pour mémoire :
    V5P28 = {
        "K0":  3.484e-12,
        "D0":  5.881e-10,
        "B0":  6.211e-11,
        "Bs0": 5.842e-12,
    }
    rows = []
    # Recalibre le préfacteur à chaque appel pour permettre des scans κ_C.
    _reset_prefactor_cache()
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p29b(
            cfg["knots"], cfg["mass_pdg"], kappa_C=kappa_C, form=form
        )
        dm = res["delta_m_MeV"]
        pdg = DELTA_M_PDG[name]
        log_ratio = float(np.log10(dm / pdg)) if dm > 0 else None
        log_ratio_v528 = float(np.log10(V5P28[name] / pdg))
        rows.append({
            "name": name,
            "knots": cfg["knots"],
            "delta_m_v5p29b_MeV": dm,
            "delta_m_v528_MeV": V5P28[name],
            "delta_m_pdg_MeV": pdg,
            "log_ratio_v5p29b": log_ratio,
            "log_ratio_v528": log_ratio_v528,
            "N_paths": res["N_paths"],
            "S_min": res["S_min"],
            "S_coupling": res["S_coupling"],
            "kappa_C": res["kappa_C"],
        })
    return rows
