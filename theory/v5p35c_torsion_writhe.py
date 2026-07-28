"""V5.35c — Torsion (writhe) au lieu du linking dans l'inertie chirale.

Hypothèse V5.35c
================

V5.34 F3_coupling utilise dans le terme inter-composantes de l'inertie
chirale :

    I_3(L) = Σ_i I_i + Σ_{i<j} J_ij × |Lk_eff(i,j)|
    J_ij   = √(c_i c_j) × b_i b_j
    Lk_eff = √(c_i c_j) × χ_ij    (χ_ij = -1 si amphi, +1 sinon)

Ce |Lk_eff| capture **l'existence** de l'enlacement mais perd
l'information de **chiralité signée** : deux composantes chirales
peuvent s'enlacer dans le **même sens** (writhes additifs) ou en
**sens opposés** (writhes destructifs).  L'inertie de torsion devrait
sentir cette différence.

V5.35c remplace donc |Lk_eff| par un **writhe signé inter-composantes**
w_ij ∈ {-1, 0, +1} construit à partir des chiralités locales :

    w_ij_signed =  +1  si les deux composantes ont la même chiralité
                   -1  si chiralités opposées (énantiomères)
                    0  si au moins une est amphichirale (writhe nul)

Et la contribution inter-composantes à l'inertie devient :

    J_ij_v5p35c = √(c_i c_j) × b_i × b_j × w_ij_signed
    I_3'(L)     = Σ_i I_i + Σ_{i<j} J_ij_v5p35c

Comparé à V5.34 :
* la magnitude √(c_i c_j) × b_i b_j est conservée ;
* le facteur scalaire |Lk_eff| = √(c_i c_j) × 1 (ou 1) est remplacé
  par un signe pur ± 1 (ou 0).  L'unité change : V5.34 voit
  c × b² × √(c c), V5.35c voit √(c c) × b² × sign.  Cela retire un
  facteur √(c c) global de l'inter, mais le préfacteur sur K⁰ absorbe
  l'unité.  Ce qui compte est la **dispersion relative** entre liens.

Théorème de Călugăreanu (référence) :
    Lk = Tw + Wr
La V5.35c isole le terme Wr en mettant le sign of writhe au cœur de
l'inter-couplage ; le |Lk_eff| de V5.34 mélangeait |Tw + Wr| sans
distinction de signe.

Choix de chiralité par défaut
=============================

Sans information physique additionnelle (orientation absolue du
nœud dans l'entrelacs), on suppose que :

* Toutes les composantes chirales d'un même entrelacs sont **alignées**
  (chiralité positive par défaut) → w_ij = +1 entre chirales.
* Les composantes amphichirales ont writhe = 0 → w_ij = 0 dès qu'une
  composante est amphi.

C'est l'**hypothèse minimale** : zéro paramètre libre nouveau, on
prédit simplement que B_s⁰ (avec 6_3 amphi) doit voir son inter-couplage
chiral disparaître complètement.

Cette hypothèse est testable : si Σ|log| augmente par rapport à V5.34
F3_coupling (0,82 dex), c'est que le |Lk_eff| capturait quelque chose
de réel (i.e. il existe un couplage même avec amphi), et V5.35c échoue
honnêtement.

Tables writhe
=============

Tables standard (Rolfsen / KnotInfo, convention crossings signés).
Sign arbitraire (donné par convention) pour les chiraux ; magnitude
seule retenue pour les amphi (toujours 0).

    3_1 :  ±3  (chiral)
    4_1 :   0  (amphi)
    5_1 :  ±5
    5_2 :  ±5
    6_1 :  ±2
    6_2 :  ±4
    6_3 :   0  (amphi)
    7_1 :  ±7
    7_2 :  ±7
    7_3 :  ±7
    7_4 :  ±7
    7_5 :  ±7
    7_6 :  ±5
    7_7 :  ±3
    8_3 :   0  (amphi)

(Les knots non listés sont étendus paresseusement par fall-back via
is_amphicheiral : amphi → w=0, sinon w prend la convention +c.)

Référence : theory/v5p34_chiral_inertia.py
            theory/v5p33_mutual_EM_oscillation.py
            theory/jones.py
"""
from __future__ import annotations

import numpy as np

from theory.jones import braid_index, is_amphicheiral, jones_polynomial
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, delta_m_v5p30c
from theory.v5p34_chiral_inertia import (
    ALPHA_V5p33_TORQUE,
    HBAR_MEV_S,  # noqa: F401
    torque_force_v5p33,
)


# ---------------------------------------------------------------------------
# 1. Table writhe (standard knot tables)
# ---------------------------------------------------------------------------
# Convention : sign arbitraire (positif par défaut) pour les knots chiraux,
# 0 pour les amphichiraux.  La magnitude est la valeur tabulée Rolfsen.
_WRITHE_TABLE: dict[str, int] = {
    "0_1": 0,    # unknot
    "3_1": 3,    # trefoil — chiral, w = +3 convention right-handed
    "4_1": 0,    # figure-eight — amphichiral
    "5_1": 5,    # cinquefoil T(2,5)
    "5_2": 5,
    "6_1": 2,    # Stevedore knot
    "6_2": 4,
    "6_3": 0,    # amphichiral
    "7_1": 7,    # T(2,7)
    "7_2": 7,
    "7_3": 7,
    "7_4": 7,
    "7_5": 7,
    "7_6": 5,
    "7_7": 3,
    "8_1": 6,
    "8_2": 6,
    "8_3": 0,    # amphichiral
    "8_12": 0,   # amphichiral
    "8_17": 0,   # negative amphichiral (writhe = 0)
    "8_18": 0,   # amphichiral
    "9_1": 9,
    "10_1": 8,
}


def knot_writhe(knot_name: str) -> int:
    """Writhe |w(K)| standard pour un nœud K (magnitude, ≥ 0).

    Cherche d'abord dans la table tabulée.  Sinon, fall-back :
    * amphichiral → 0
    * sinon → c(K) (proxy : tout chiral non-tabulé prend writhe = c
      comme borne supérieure, suffisant pour le test V5.35c qui
      n'utilise que le SIGNE).

    Returns
    -------
    int
        |w(K)| ≥ 0.
    """
    if knot_name in _WRITHE_TABLE:
        return int(_WRITHE_TABLE[knot_name])
    amphi = bool(is_amphicheiral(jones_polynomial(knot_name)))
    if amphi:
        return 0
    return int(crossing_number(knot_name))


def is_chiral(knot_name: str) -> bool:
    """True si writhe(K) ≠ 0 (chiral), False si amphichiral."""
    return knot_writhe(knot_name) != 0


# ---------------------------------------------------------------------------
# 2. Writhe signé inter-composantes
# ---------------------------------------------------------------------------
def writhe_inter_components(
    knot_i: str,
    knot_j: str,
    sign_i: int = +1,
    sign_j: int = +1,
) -> dict:
    """Writhe signé inter-composantes w_ij ∈ {-1, 0, +1}.

        w_ij =  0                       si l'une des deux est amphichirale,
        w_ij =  sign(s_i × s_j)         si les deux sont chirales,

    où s_i, s_j sont les chiralités relatives ±1.  Par défaut s_i =
    s_j = +1 → w_ij = +1 (alignement chiral, hypothèse minimale).

    Returns
    -------
    dict
        { knot_i, knot_j, w_ij_signed, w_i, w_j, is_chiral_i, is_chiral_j }
    """
    w_i = knot_writhe(knot_i)
    w_j = knot_writhe(knot_j)
    if w_i == 0 or w_j == 0:
        w_ij = 0
    else:
        s_i = int(np.sign(sign_i)) if sign_i != 0 else +1
        s_j = int(np.sign(sign_j)) if sign_j != 0 else +1
        w_ij = int(s_i * s_j)
    return {
        "knot_i": knot_i,
        "knot_j": knot_j,
        "w_ij_signed": int(w_ij),
        "w_i": int(w_i),
        "w_j": int(w_j),
        "is_chiral_i": w_i != 0,
        "is_chiral_j": w_j != 0,
    }


# ---------------------------------------------------------------------------
# 3. Inertie chirale V5.35c (variante de F3_coupling)
# ---------------------------------------------------------------------------
def chiral_inertia_v5p35c(
    knot_names: list[str],
    chiralities: list[int] | None = None,
) -> dict:
    """Inertie chirale I_χ V5.35c — writhe signé inter-composantes.

        I_χ(L) = Σ_i I_i + Σ_{i<j} J_ij_v5p35c
        I_i           = c_i × (c_i / b_i)²              (idem V5.34 F1/F3)
        J_ij_v5p35c   = √(c_i c_j) × b_i × b_j × w_ij_signed
        w_ij_signed   = +1 same chirality, -1 opposite, 0 si une amphi

    Si I_χ ≤ 0 (cas pathologique : couplages opposés annulent I_self),
    on plafonne à I_self (l'inertie propre est toujours présente).

    Parameters
    ----------
    knot_names : list[str]
    chiralities : list[int] | None
        Chiralité ±1 par composante (par défaut +1 partout, hypothèse
        d'alignement minimal).  Les composantes amphichirales sont
        forcées à 0.

    Returns
    -------
    dict
        { I_chi, I_self, I_inter, n_comp, components, writhe_signs,
          inter_couplings }
    """
    n = len(knot_names)
    if chiralities is None:
        chiralities = [+1] * n
    if len(chiralities) != n:
        raise ValueError(
            f"chiralities a {len(chiralities)} éléments pour {n} composantes"
        )
    components = []
    writhe_signs = []
    for k, s in zip(knot_names, chiralities):
        b = braid_index(k)
        c = crossing_number(k)
        w_abs = knot_writhe(k)
        amphi = (w_abs == 0)
        eff_sign = 0 if amphi else int(np.sign(s)) if s != 0 else +1
        components.append(
            {
                "knot": k,
                "b": int(b),
                "c": int(c),
                "writhe_abs": int(w_abs),
                "amphi": bool(amphi),
                "sign_eff": int(eff_sign),
            }
        )
        writhe_signs.append(int(eff_sign))

    I_self = sum(
        comp["c"] * (comp["c"] / comp["b"]) ** 2 for comp in components
    )
    I_inter = 0.0
    inter_couplings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, bi = components[i]["c"], components[i]["b"]
            cj, bj = components[j]["c"], components[j]["b"]
            si, sj = components[i]["sign_eff"], components[j]["sign_eff"]
            if si == 0 or sj == 0:
                w_ij = 0
            else:
                w_ij = si * sj
            J_ij = np.sqrt(ci * cj) * bi * bj * w_ij
            I_inter += J_ij
            inter_couplings.append(
                {
                    "i": i,
                    "j": j,
                    "knot_i": knot_names[i],
                    "knot_j": knot_names[j],
                    "w_ij_signed": int(w_ij),
                    "J_ij": float(J_ij),
                }
            )

    I = float(I_self + I_inter)
    if I <= 0:
        # Plafond plancher : inertie propre toujours > 0 par construction.
        I = float(I_self)

    return {
        "I_chi": I,
        "I_self": float(I_self),
        "I_inter": float(I_inter),
        "n_comp": int(n),
        "components": components,
        "writhe_signs": writhe_signs,
        "inter_couplings": inter_couplings,
    }


# ---------------------------------------------------------------------------
# 4. ω_χ V5.35c = √(F_torque / I_χ V5.35c)
# ---------------------------------------------------------------------------
def omega_chirality_v5p35c(
    knot_names: list[str],
    chiralities: list[int] | None = None,
    alpha_torque: float = ALPHA_V5p33_TORQUE,
) -> dict:
    """ω_χ(L) = √(F_torque(L) / I_χ V5.35c(L)).

    F_torque hérité strict de V5.33/V5.34 (aucune modification).
    """
    F = torque_force_v5p33(knot_names, alpha_torque)
    I = chiral_inertia_v5p35c(knot_names, chiralities)
    if I["I_chi"] <= 0:
        raise ValueError(
            f"I_chi V5.35c <= 0 pour {knot_names!r}"
        )
    omega = float(np.sqrt(F["F_torque"] / I["I_chi"]))
    return {
        "omega_chi": omega,
        "F_torque": F["F_torque"],
        "I_chi": I["I_chi"],
        "I_self": I["I_self"],
        "I_inter": I["I_inter"],
        "n_comp": F["n_comp"],
        "writhe_signs": I["writhe_signs"],
    }


# ---------------------------------------------------------------------------
# 5. Cache pour ω_χ(K⁰)
# ---------------------------------------------------------------------------
_OMEGA_K0_CACHE: dict[tuple, float] = {}


def _clear_cache() -> None:
    _OMEGA_K0_CACHE.clear()


def _omega_chi_K0_v5p35c() -> float:
    key = ("K0",)
    if key in _OMEGA_K0_CACHE:
        return _OMEGA_K0_CACHE[key]
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    oc = omega_chirality_v5p35c(cfg["knots"])["omega_chi"]
    if oc <= 0:
        raise ValueError(f"ω_χ(K⁰) V5.35c = {oc}")
    _OMEGA_K0_CACHE[key] = float(oc)
    return float(oc)


# ---------------------------------------------------------------------------
# 6. Δm V5.35c
# ---------------------------------------------------------------------------
def delta_m_v5p35c(
    knot_names: list[str],
    mass_link_MeV: float,
    chiralities: list[int] | None = None,
    alpha_vib: float = ALPHA_VIB_DEFAULT,
    kappa_K: float = KAPPA_K_V5p30_DEFAULT,
) -> dict:
    """Δm V5.35c(L) = Δm V5.30c(L) × R_χ(L),
                     R_χ = ω_χ V5.35c(L) / ω_χ V5.35c(K⁰).

    Par construction Δm V5.35c(K⁰) = Δm V5.30c(K⁰) (R_χ = 1).

    Aucun paramètre libre nouveau : zero-parameter test du remplacement
    Lk_eff → writhe signé.
    """
    res_v530c = delta_m_v5p30c(knot_names, mass_link_MeV, alpha_vib, kappa_K)
    oc = omega_chirality_v5p35c(knot_names, chiralities)
    oc_K0 = _omega_chi_K0_v5p35c()
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
        "I_inter": oc["I_inter"],
        "n_comp": oc["n_comp"],
        "writhe_signs": oc["writhe_signs"],
    }


# ---------------------------------------------------------------------------
# 7. Table de comparaison V5.35c vs V5.34 vs PDG
# ---------------------------------------------------------------------------
def v5p35c_table(
    chirality_hypothesis: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Table Δm V5.35c sur les 4 systèmes K⁰/D⁰/B⁰/B_s⁰.

    Parameters
    ----------
    chirality_hypothesis : dict[str, list[int]] | None
        Optionnel : pour chaque méson, liste des chiralités par
        composante (sinon : +1 partout).

    Returns
    -------
    list[dict]
        Une ligne par méson.
    """
    _clear_cache()
    if chirality_hypothesis is None:
        chirality_hypothesis = {}
    rows = []
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        chir = chirality_hypothesis.get(name)
        res = delta_m_v5p35c(cfg["knots"], cfg["mass_pdg"], chir)
        dm = res["delta_m_MeV"]
        dm_pdg = DELTA_M_PDG[name]
        log_r = float(np.log10(dm / dm_pdg)) if dm > 0 else None
        rows.append(
            {
                "name": name,
                "knots": cfg["knots"],
                "delta_m_v5p35c_MeV": dm,
                "delta_m_v5p30c_MeV": res["delta_m_v5p30c_MeV"],
                "delta_m_pdg_MeV": dm_pdg,
                "log_ratio": log_r,
                "R_chi": res["R_chi"],
                "I_chi": res["I_chi"],
                "I_self": res["I_self"],
                "I_inter": res["I_inter"],
                "writhe_signs": res["writhe_signs"],
                "n_comp": res["n_comp"],
            }
        )
    return rows


def summary_v5p35c(
    chirality_hypothesis: dict[str, list[int]] | None = None,
) -> dict:
    """Σ|log|, ratio B_s/B, et table V5.35c."""
    rows = v5p35c_table(chirality_hypothesis)
    per = {r["name"]: r for r in rows}
    sum_abs_log = float(
        sum(abs(r["log_ratio"]) for r in rows if r["log_ratio"] is not None)
    )
    ratio_BsB = per["Bs0"]["delta_m_v5p35c_MeV"] / per["B0"]["delta_m_v5p35c_MeV"]
    return {
        "sum_abs_log": sum_abs_log,
        "ratio_Bs_B": float(ratio_BsB),
        "per_meson": per,
        "rows": rows,
    }
