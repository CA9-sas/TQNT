r"""
Vérification du VEV électrofaible du multiplet EW corrigé — ÉCHEC révélateur.

Le VEV se dérive (ew_vev_derivation.py) de m_W et m_Z SEULS, via l'angle de
Weinberg à l'arbre :
    sin²θ_W = 1 − (m_W/m_Z)²   ;   g = √(4πα)/sinθ_W   ;   vev = 2 m_W/g.
CRUCIAL : sin²θ_W ne dépend que du RAPPORT m_W/m_Z, INDÉPENDANT de l'échelle de
calibration. C'est donc un test bien plus SÉVÈRE que le RMS de masse (qui, lui,
absorbe une échelle globale). Référence exp : m_W/m_Z = 0,8814 → sin²θ_W = 0,223.

RÉSULTAT (rapport prédit par la géométrie correcte, W = 7_1+7_2+8_3) :
    Z ancien   7_2+8_3+9_1 : m_W/m_Z = 0,685 → sin²θ_W = 0,531 → vev +57 %.
    Z corrigé  7_2+8_3+6_3 : m_W/m_Z = 0,805 → sin²θ_W = 0,352 → vev +28 %.
Le corrigé est MEILLEUR que l'ancien (0,352 vs 0,531) mais reste LOIN de 0,223.

PIRE — c'est déjà le MEILLEUR possible : la recherche sur tous les Z = 7_2+8_3+X
BOSONiques montre que 6_3 (0,805) est le X qui APPROCHE le plus 0,881. Les autres
bosons donnent soit m_W/m_Z ≥ 1 (Z ≤ W, non physique : 7_1 redonne {7_1,7_2,8_3}=W),
soit pire. AUCUNE assignation bosonique de Z ne reproduit le rapport électrofaible.

DIAGNOSTIC : la cause est le pattern MIXED-SIGN (W -5,8 %, Z +3,2 %). Le VEV veut
que W et Z aient la MÊME erreur fractionnaire (le rapport serait alors exact) ;
la géométrie leur donne des erreurs de SIGNE OPPOSÉ → le rapport casse. C'est
exactement le « mixed-sign, no global recalibration » de [[electroweak-sector-true-knots]],
mais rendu QUANTITATIF et FATAL par le VEV.

VERDICT (le plus sévère de la série) : le multiplet corrigé PASSE l'ordre de
masse (RMS 5,6 %), la charge (Z propre, top entier) et le spin (parités
correctes), mais ÉCHOUE le VEV (sin²θ_W = 0,35 vs 0,22, vev +28 %). La géométrie
des nœuds ne suit PAS le rapport m_W/m_Z. Deux conclusions possibles :
  (a) les masses EW ne sont pas fidèlement géométriques (lecture (a) confirmée) ;
  (b) W lui-même est mal assigné (pas seulement Z/top).
La réassignation géométrique est INSUFFISANTE pour la précision qu'exige le VEV.
Ne PAS présenter le multiplet corrigé comme reproduisant le secteur EW complet.
"""
from __future__ import annotations
import math
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.jones import braid_index
from theory.abraham import REFERENCE_N50K as GEO
from simulation.experiments.ew_vev_derivation import VEV_EXP

AMPHI = {"4_1", "6_3", "8_3"}
ALPHA = 1 / 137.035999
MW_EXP = 80.379


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def PhiF(K):
    d = ots._axiom_data(K)
    return d["Uk"] ** (5 / 3) * d["Ut"]


def rawmass(cfg, s=0.45):
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)
    comp = [x.strip().replace("-", "_") for x in cfg.split("+")]
    n = len(comp); cnt = Counter(comp)
    screen = 1.0
    for k, cc in cnt.items():
        if cc >= 2:
            screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (cc - 1)
    b = [_m_naive(k) * _T(k) ** P_CAS for k in comp]
    holder = (sum(np.sqrt(bi) for bi in b)) ** 2
    G = sum(PhiF(k) for k in comp) / PhiF("3_1")
    pair = 1.0
    for i, j in itertools.combinations(range(n), 2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI:
            pair *= s
    return holder * G ** Q_CAS * pair * screen


def vev(cfgW, cfgZ):
    r = rawmass(cfgW) / rawmass(cfgZ)          # indépendant de l'échelle
    s2W = 1 - r ** 2
    if s2W <= 0:
        return r, s2W, None
    g = math.sqrt(4 * math.pi * ALPHA) / math.sqrt(s2W)
    return r, s2W, 2 * MW_EXP / g


def main():
    _install_ideal()
    print("VEV via m_W/m_Z (indépendant de l'échelle). Réf. exp : "
          "0,8814 → sin²θ_W=0,223 → vev≈246.\n")
    for lbl, cZ in [("Z ancien  7_2+8_3+9_1", "7_2+8_3+9_1"),
                    ("Z corrigé 7_2+8_3+6_3", "7_2+8_3+6_3")]:
        r, s2, v = vev("7_1+7_2+8_3", cZ)
        print(f"  {lbl:24s} m_W/m_Z={r:.4f} sin²θ_W={s2:.4f} "
              f"vev={v:.0f} ({(v / VEV_EXP - 1) * 100:+.0f}%)")
    print("\nRecherche du meilleur Z bosonique pour le rapport (W=7_1+7_2+8_3) :")
    mW = rawmass("7_1+7_2+8_3")
    cands = []
    for X in GEO:
        comp = ["7_2", "8_3", X]
        boson = sum(braid_index(k) % 2 == 0 for k in comp) % 2 == 0
        if not boson:
            continue
        r = mW / rawmass("+".join(comp))
        cands.append((abs(r - 0.881), X, r, 1 - r ** 2))
    for _, X, r, s2 in sorted(cands)[:5]:
        note = "← corrigé" if X == "6_3" else ("← ancien" if X == "9_1" else "")
        print(f"  X={X:5s} m_W/m_Z={r:.4f} sin²θ_W={s2:+.4f} {note}")
    print("\n→ 6_3 est déjà le meilleur Z bosonique ; aucun ne reproduit 0,881.")
    print("  VEV ÉCHOUE (mixed-sign W/Z). Ordre+charge+spin OK, mais pas le VEV.")


if __name__ == "__main__":
    main()
