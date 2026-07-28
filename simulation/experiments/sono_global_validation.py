"""Validation du minimiseur global (sono_global) sur ropelengths publiees.

CHANTIER beta_otimes (liste B) : « pincer exige un minimiseur global de
ropelength ». Nouveaux outils (ideal_embedding) :
  - polish_gradient : descente de longueur PAR SOMMET sous contraintes --
    exploite le mou non uniforme, invisible au retrecissement uniforme de sono
    (smoke test : 3_1 passe de +0.81% a +0.12% du publie) ;
  - sono_global : multi-departs perturbes (squeezes homeomorphes) + rechauffes
    graduees + polissage + continuation en resolution ; info['basins'] liste
    les minima par depart (diagnostic de globalite).

CIBLES (ropelength publiee, unites de rayon ; Pieranski/CKS, x2 des valeurs
demi-ropelength archivees) : 3_1 = 32.744 ; 5_2 = 52.26 ; 7_2 = 65.92.
Gardes : Alexander (type de noeud preserve).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono, sono_global, torus_knot, twist_knot_seed, resample,
)
from repair_curves_braid import alexander  # noqa: E402

PUBLISHED = {"3_1": 32.744, "5_2": 52.26, "7_2": 65.92}
ALEX = {"3_1": (1, -1, 1), "5_2": (2, -3, 2), "7_2": (3, -5, 3)}
SEEDS = {
    "3_1": lambda: torus_knot(2, 3, n=200),
    "5_2": lambda: resample(twist_knot_seed(5, x_eye=0.10), 300),
    "7_2": lambda: resample(twist_knot_seed(9, x_eye=0.10), 320),
}


def guard_for(name):
    ref = ALEX[name]
    def g(P):
        try:
            return tuple(int(x) for x in alexander(P)[0]) == ref
        except Exception:
            return False
    return g


def run():
    print("=== Validation sono_global vs ropelengths publiees ===", flush=True)
    for name in ("3_1", "5_2", "7_2"):
        pub = PUBLISHED[name]
        # baseline : sono seul
        P, info = sono(SEEDS[name](), iters=20000)
        r_sono = info["ropelength"]
        # minimiseur global
        Q, r_glob, ginfo = sono_global(
            SEEDS[name](), guard=guard_for(name), restarts=3, iters=12000,
            trials_per_rung=2, continuation=(1.5,),
            log=lambda m: print("   " + m, flush=True))
        b = sorted(r for r, ok in ginfo["basins"] if ok)
        spread = (b[-1] / b[0] - 1) * 100 if len(b) > 1 else 0.0
        print("%s : publie %.2f | sono %.2f (%+.1f%%) | GLOBAL %.2f (%+.2f%%) ; "
              "bassins valides %d, dispersion %.1f%%" % (
                  name, pub, r_sono, (r_sono / pub - 1) * 100,
                  r_glob, (r_glob / pub - 1) * 100, len(b), spread), flush=True)
    print("\nLecture : si GLOBAL <~ +1-2% partout, l'outillage est apte au", flush=True)
    print("pincement beta_otimes (l'etape suivante : nucleon repoli).", flush=True)


if __name__ == "__main__":
    run()
