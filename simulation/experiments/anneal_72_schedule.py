"""Plongement idéal du 7_2 : amorce native + réchauffes graduées.

LEÇON DU PAYSAGE (2026-07-13) :
  - depuis la fermeture de tresse, SONO plafonne à rho = 118 (+72% du publié
    65.92) et les réchauffes graduées seules ne franchissent PAS la barrière
    (113.5 après 2 balayages : bassin trop profond) ;
  - la sortie n'est pas de forcer le paysage mais de CHANGER D'AMORCE : le
    twist_knot_seed natif (bande torsadée + agrafe enfilée dans l'œil,
    sélectionné par la garde d'Alexander : n_half=9, x_eye=0.10 -> (3,-5,3))
    démarre dans le bassin compact : rho = 73.7 (+11.8%) dès le serrage.

PIPELINE :
  1. amorce twist_knot_seed(9, x_eye=0.10) + vérification d'Alexander ;
  2. serrage SONO (40000 itérations) ;
  3. réchauffes graduées anneal() : compressions affines anisotropes
     (homéomorphismes -- jamais de passage de brin ; s >= 0.5 => gap >= 1.0R),
     échelle s = 0.60 -> 0.90, validation d'Alexander à chaque essai,
     arrêt anticipé à rho <= 66.5 (publié + 1%).

Sortie : output/p72_ideal.npy (meilleure courbe valide, unités de rayon).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import sono, anneal, twist_knot_seed  # noqa: E402
from repair_curves_braid import alexander  # noqa: E402

PUBLISHED = 65.92
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "p72_ideal.npy")


def guard_72(P):
    try:
        return tuple(int(x) for x in alexander(P)[0]) == (3, -5, 3)
    except Exception:
        return False


def run():
    print("=== 7_2 idéal : amorce native + réchauffes (cible %.2f) ===" % PUBLISHED,
          flush=True)
    P0 = twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10)
    assert guard_72(P0), "l'amorce ne réalise pas le 7_2"

    print("serrage initial (40000 itérations)...", flush=True)
    P, info = sono(P0, n=280, iters=40000)
    print("  rho = %.2f gap = %.3f alexander %s" %
          (info["ropelength"], info["min_gap"], "OK" if guard_72(P) else "KO"),
          flush=True)

    best, rho, journal = anneal(
        P, guard=guard_72, target=PUBLISHED * 1.01,
        rungs=(0.6, 0.7, 0.8, 0.9), trials_per_rung=2,
        iters=10000, sweeps=2, rng=np.random.default_rng(3),
        log=lambda m: print(m, flush=True),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, best)
    ok = rho <= PUBLISHED * 1.05
    print("\n=== BILAN : rho = %.2f (publié %.2f, %+.1f%%) — %s ===" %
          (rho, PUBLISHED, (rho / PUBLISHED - 1) * 100,
           "SUCCÈS" if ok else "écart résiduel"), flush=True)
    print("courbe sauvée : %s" % os.path.abspath(OUT), flush=True)


if __name__ == "__main__":
    run()
