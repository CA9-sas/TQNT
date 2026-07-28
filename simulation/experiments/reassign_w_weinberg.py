r"""
Réassigner W (avec Z) pour corriger l'angle de Weinberg — succès sur l'angle,
coût sur l'échelle absolue.

Le VEV échouait (verify_ew_vev.py) car m_W/m_Z=0,805 → sin²θ_W=0,352 (vs 0,223).
On libère W AUSSI et on cherche un couple (W,Z) bosonique, partageant 8_3, avec
m_W/m_Z ≈ 0,881, Z neutre (Σχ=0), W chargé (|Σχ|≤3), ordre W<Z<H<top préservé.

RÉSULTAT — l'angle se corrige :
    W = 8_3+7_4+8_3 ,  Z = 8_3+4_1+6_3
    m_W/m_Z = 0,887 → sin²θ_W = 0,214  (mesuré 0,223 ; à 4 %).
  Contrôles : W boson (b=4,3,4 → 2 fermions) ✓, Q=+1 (Σχ=+1, Lk=0) ✓ ;
              Z boson ✓, Q=0 (Σχ=0, Lk=0) ✓.
  BONUS : Z = 8_3+4_1+6_3 est ENTIÈREMENT amphichiral → Z⁰ intrinsèquement neutre
  ET auto-conjugué, physiquement idéal pour le boson neutre.

COÛT — l'échelle absolue casse :
    W -16 %, Z -17 %, H +28 %, top +13 %  (RMS 19 %).
  W,Z (nœuds légers, nécessaires pour le rapport) sortent trop légers face à H,top
  (qui portent le 10_1 lourd). Fixer le RAPPORT (angle) sacrifie l'ÉCHELLE. C'est
  un DÉPLACEMENT de la tension mixed-sign : du couple (W,Z) vers le bloc
  (W,Z)-contre-(H,top).

OPTION écartée : garder Z=7_2+8_3+6_3 et ne bouger que W ne permet PAS d'atteindre
0,223 (meilleurs : 0,178 avec 8_3+7_4+7_6, ou 0,271 avec 8_3+7_6+8_1). Il faut
réassigner W ET Z ensemble.

VERDICT : l'angle de Weinberg — l'observable EW profonde qui avait fait échouer le
VEV — EST récupérable par réassignation (sin²θ_W 0,35→0,214, à 4 % de la mesure),
avec un Z⁰ pleinement amphichiral. MAIS pas simultanément avec le spectre absolu :
le rapport et l'échelle se disputent. Le secteur EW a une tension à deux niveaux
(angle + échelle) que la géométrie des nœuds ne résout pas d'un coup. Progrès réel
sur l'angle ; à NE PAS présenter comme un secteur EW complet reproduit.
"""
from __future__ import annotations
import math
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.jones import braid_index
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
TORUS = {"3_1", "5_1", "7_1", "9_1"}
ALPHA = 1 / 137.035999
EXP = {"W": 80.379, "Z": 91.188, "H": 125.25, "top": 172.5}
WIN = {"W": "8_3+7_4+8_3", "Z": "8_3+4_1+6_3",
       "H": "7_1+8_3+10_1", "top": "8_3+4_1+10_1"}


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


def chi(k):
    if k in AMPHI:
        return 0
    if k in TORUS:
        return 2
    if k == "7_2":
        return 0
    return 1


def parity(cfg):
    return "boson" if sum(braid_index(k) % 2 == 0 for k in cfg.split("+")) % 2 == 0 else "fermion"


def main():
    _install_ideal()
    r = rawmass(WIN["W"]) / rawmass(WIN["Z"])
    s2W = 1 - r ** 2
    print("Paire Weinberg réassignée (W ET Z) :\n")
    print(f"  m_W/m_Z = {r:.3f} → sin²θ_W = {s2W:.3f}  (mesuré 0,223)\n")
    for k in ["W", "Z"]:
        comp = WIN[k].split("+")
        Schi = sum(chi(x) for x in comp)
        q = "Q=+1 (Lk=0)" if k == "W" else "Q=0 (Lk=0)"
        print(f"  {k}: {WIN[k]:14s} spin={parity(WIN[k])} Σχ={Schi:+d} → {q}")
    print("\n  Masses absolues (échelle geomean, H/top fixes) :")
    off = np.mean([math.log(EXP[k]) - math.log(rawmass(WIN[k])) for k in EXP])
    for k in ["W", "Z", "H", "top"]:
        p = rawmass(WIN[k]) * math.exp(off)
        print(f"    {k:4s} préd={p:7.1f} exp={EXP[k]:7.1f} ({(p - EXP[k]) / EXP[k] * 100:+.0f}%)")
    rms = math.sqrt(np.mean([((rawmass(WIN[k]) * math.exp(off) - EXP[k]) / EXP[k]) ** 2
                             for k in EXP])) * 100
    print(f"\n  RMS absolu = {rms:.0f}%  — l'angle est corrigé, l'échelle W,Z-vs-H,top casse.")


if __name__ == "__main__":
    main()
