r"""
Réassignation de Z et top avec des nœuds cohérents avec la géométrie correcte.

ew_sector_coefficient.py a montré que les assignations EW autoritatives
(calibrées sur l'ancien Ω_ts phénoménologique) BROUILLENT l'ordre sur la
géométrie correcte : les coupables sont le 9_1 (torus, Φ_F=10920) qui gonfle Z,
et le 8_2 (Φ_F=2337 trop léger) qui met top SOUS H.

STRATÉGIE (substitution minimale, chaîne du multiplet). On garde le marqueur EW
8_3 dans chaque, on cale l'échelle sur W et H (qui ne sont pas cassés), et on ne
remplace QUE le nœud coupable — en respectant la structure de substitution
1-nœud du multiplet (Z partage 7_2+8_3 avec W ; top partage 8_3+10_1 avec H).

⚠️ MISE À JOUR SPIN : le top 8_3+10_1+9_1 ci-dessous est le meilleur EN MASSE, mais
il est RÉFUTÉ par le spin (le 9_1, b=2, en fait un BOSON alors que top=fermion —
voir verify_ew_spin.py). Le top spin-VALIDE est 8_3+4_1+10_1 (-6,6 %). Ce script
reste la recherche MASSE-SEULE ; le multiplet final spin-corrigé est dans
verify_ew_spin.py.

RÉSULTAT (masse seule) :
    Z   : 7_2+8_3+9_1  →  7_2+8_3+6_3   (+3,2 %)  [spin: boson ✓]
    top : 8_2+8_3+10_1 →  8_3+10_1+9_1  (+2,3 %)  [spin: BOSON ✗ → corrigé 4_1, -6,6 %]
    RMS EW 18 % → 4,7 % (masse seule) / 5,6 % (spin-corrigé) ; ORDRE W<Z<H<top ✓.

BONUS PHYSIQUE (double). (1) Le 9_1 (torus T(2,9), lourd, Φ_F=10920) qui gonflait
le boson NEUTRE Z n'y avait pas sa place : il MIGRE vers top, la particule la plus
LOURDE — le nœud lourd va au bon endroit. C'est une simple RELOCALISATION d'un
nœud entre deux membres du multiplet. (2) Le nouveau Z = 7_2+8_3+6_3 contient DEUX
amphichiraux (6_3 et 8_3) là où W n'en a qu'un (8_3). Le boson NEUTRE Z⁰ (son
propre antiparticule) devient plus AUTO-CONJUGUÉ — l'amphichiralité = invariance
miroir = neutralité matière/antimatière (cf. Règle 0).

RÉSERVES (importantes) :
  1. Réassignation par la GÉOMÉTRIE DE MASSE SEULE. Non re-vérifié : la charge
     Q=Σχ+Lk, le spin, et surtout la dérivation du VEV EW (ew_vev_derivation, sur
     laquelle les anciennes assignations étaient calibrées). Une réassignation
     COMPLÈTE doit re-passer ces contraintes.
  2. W et H gardés, encadrent ±6 % (échelle calée sur leur moyenne géométrique).
  3. NON UNIQUE (liberté d'assignation, cf. [[assignment-freedom-quantified-V6]]) :
     pour Z, 6_3 (+3,2 %) devance 10_1 (+6,7 %) ; pour top, 9_1 (+2,3 %) devance
     4_1 (-6,6 %). C'est UNE solution cohérente, pas une contrainte forcée.
  4. Ne PAS écrire dans V6 ([[research-in-brouillon-not-v6]]).
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
S_CHIRAL = 0.45


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def PhiF(K):
    d = ots._axiom_data(K)
    return d["Uk"] ** (5 / 3) * d["Ut"]


def mass(cfg, s=S_CHIRAL):
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


def main():
    _install_ideal()
    W = ("7_1+7_2+8_3", 80377.); H = ("7_1+8_3+10_1", 125250.)
    off = np.mean([np.log(mx) - np.log(mass(cfg)) for cfg, mx in (W, H)])
    pred = lambda cfg: mass(cfg) * np.exp(off)

    def best_single(fixed, target, culprit):
        cands = sorted(((abs((pred(f"{fixed[0]}+{fixed[1]}+{X}") - target) / target),
                         X, pred(f"{fixed[0]}+{fixed[1]}+{X}")) for X in GEO))
        return cands[0][1]

    zX = best_single(("7_2", "8_3"), 91188., "9_1")
    tX = best_single(("8_3", "10_1"), 172500., "8_2")
    FINAL = [("W", "7_1+7_2+8_3", 80377.), ("Z", f"7_2+8_3+{zX}", 91188.),
             ("H", "7_1+8_3+10_1", 125250.), ("top", f"8_3+10_1+{tX}", 172500.)]
    print("Spectre EW réassigné (Z, top ; W, H inchangés) :")
    preds = []
    for nm, cfg, mx in FINAL:
        p = pred(cfg); preds.append((nm, p, mx))
        amp = sum(k in AMPHI for k in cfg.split("+"))
        print(f"  {nm:4s} {cfg:16s} préd={p:8.0f} exp={mx:8.0f} "
              f"({(p - mx) / mx * 100:+.1f}%)  amphi={amp}")
    r = np.sqrt(np.mean([((p - mx) / mx) ** 2 for _, p, mx in preds])) * 100
    opred = [nm for nm, _, _ in sorted(preds, key=lambda t: t[1])]
    oexp = [nm for nm, _, _ in sorted(preds, key=lambda t: t[2])]
    print(f"\n  RMS EW = {r:.1f}%  (avant réassignation : 18 %)")
    print(f"  ordre préd : {' < '.join(opred)}  {'OK' if opred == oexp else 'BROUILLÉ'}")
    print("  Z gagne un amphichiral (6_3) → boson neutre plus auto-conjugué.")


if __name__ == "__main__":
    main()
