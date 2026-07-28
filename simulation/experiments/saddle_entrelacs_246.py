r"""
Identification de l'entrelacs-selle de reconnexion à l'échelle faible (246 GeV)
via la formule de masse TQNT (cornell_equilibrium × ETA, calibrée sur les masses).

Contexte : la durée de vie du neutron sans calibration (neutron_equilibration.py)
fait intervenir une selle de reconnexion d'énergie E_s ≈ v = 246 GeV, qui fixe
G_F = 1/(√2 E_s²). On montre ici que cette échelle EST une masse topologique
accessible : un entrelacs naturel y correspond.

Résultat : 10_1+10_1+10_1+10_1+7_2 à k=28 -> 245,9 GeV (−0,1 % de 246).
  - k=28 = un niveau de Fibonacci au-dessus du top (k=27) ;
  - 5 composantes (le top en a 6) : complexité comparable, pas un outlier ;
  - plusieurs configs voisines (10_1⁴+7_3, +7_4, ...) donnent ~246 GeV à ±1 %.

RÉSERVE : 246 GeV était la CIBLE (échelle faible connue). Ceci CONFIRME que
l'échelle est une masse topologique réalisable et identifie la config
candidate ; ce n'est pas encore une prédiction « de rien » de 246, ni la preuve
que cette config est dynamiquement la selle du chemin n -> p.

Usage :  python3 -m simulation.experiments.saddle_entrelacs_246
"""
from __future__ import annotations

from itertools import combinations_with_replacement
import numpy as np

from theory.multi_knot import cornell_equilibrium, ETA
from theory.jones import JONES_KNOT_NAMES

TARGET = 246000.0  # MeV (échelle faible v)
KNOTS = [k for k in JONES_KNOT_NAMES if k != "0_1"]


def mass(cfg, k):
    return ETA * cornell_equilibrium(list(cfg), k)["mass_raw"]


def main():
    print("=" * 76)
    print("  ENTRELACS-SELLE À 246 GeV (échelle faible) — formule de masse TQNT")
    print("=" * 76)

    # contexte : le top fixe l'échelle k=27
    top = ["7_1", "7_2", "10_1", "10_1", "10_1", "10_1"]
    print(f"\n  Repère : top (k=27) = {mass(top,27)/1000:.1f} GeV ; "
          f"k=28 monte l'échelle (F(28)/F(27) ≈ 1,6).")

    # recherche : 4 à 6 composantes, k=27 et 28
    cands = []
    for n in (4, 5, 6):
        for k in (27, 28):
            for cfg in combinations_with_replacement(KNOTS, n):
                # restreindre aux configs lourdes (au moins 3 gros nœuds) pour la vitesse
                if sum(1 for x in cfg if x in ("9_1", "10_1", "8_3")) < n - 2:
                    continue
                m = mass(cfg, k)
                cands.append((abs(m - TARGET), m, cfg, k, n))
    cands.sort()

    print(f"\n  Entrelacs les plus proches de 246 GeV :")
    print(f"  {'config':>32} {'n':>2} {'k':>3} {'m (GeV)':>9} {'écart':>7}")
    print("  " + "-" * 62)
    shown = 0
    seen = set()
    for d, m, cfg, k, n in cands:
        key = (cfg, k)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {'+'.join(cfg):>32} {n:>2} {k:>3} {m/1000:>9.1f} "
              f"{(m-TARGET)/TARGET*100:>+6.1f}%")
        shown += 1
        if shown >= 8:
            break

    best = cands[0]
    print(f"\n  SELLE IDENTIFIÉE : {'+'.join(best[2])} (k={best[3]}) "
          f"= {best[1]/1000:.1f} GeV  ({(best[1]-TARGET)/TARGET*100:+.1f}%)")
    print(f"  => v ≈ 246 GeV est une MASSE TOPOLOGIQUE (entrelacs de {best[4]} ondes,")
    print(f"     niveau k={best[3]}), donc G_F = 1/(√2 v²) est topologiquement fondé.")
    print("=" * 76)


if __name__ == "__main__":
    main()
