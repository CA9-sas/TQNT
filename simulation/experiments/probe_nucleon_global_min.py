"""Sonde de globalite du minimum nucleon (rho=208) — rechauffes graduees sur l'entrelacs.

QUESTION : le bassin rho=208 (qui donne E=+0.7%) est-il le minimum global ?
La tendance 'plus serre -> E plus grand' signifie qu'un etat plus serre DEPASSERAIT
la cible : c'est le seul risque restant de la chaine beta_otimes.

METHODE (topologiquement sure) : compression affine anisotrope appliquee aux TROIS
composantes ensemble (homeomorphisme ambiant : types de noeuds ET nombres
d'enlacement preserves ; s >= 0.5 => gap >= 1.0R), puis re-relaxation sono_link
sans desserrage initial ; gardes Lk de Gauss + Alexander ; verdict : un etat
valide avec rho < 208 est-il trouve, et quel E donne-t-il ?

Usage : python3 probe_nucleon_global_min.py <trial 0..5> [src_prefix]
(6 essais paralleles, s in {0.55, 0.65, 0.75} x 2 directions aleatoires par s ;
RHO_REF surchargable par variable d'environnement).

RESULTATS (2 rondes, 2026-07-14) : le bassin 208 N'EST PAS global (le +0.7% etait
un artefact de plateau). 8 minima valides distincts (rho 204.5-208, quasi
degeneres : a serrage egal, E varie de ~6%) -> Delta m(n-p) reproduit a +3 +/- 3%
sans parametre ajuste ; precision bornee par la quasi-degenerescence + le modele
tube-mince. Les KO sont de VRAIS passages inter-composantes (Lk de Gauss perd un
enlacement, types de noeuds intacts ; leurs E +14-20% recoupent la mesure de la
paire +16.7%). Regle de surete : s >= 0.7 pour les entrelacs ; garde Lk requise.
beta_otimes RESTE CALIBRE (decision utilisateur, validee).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono_link, squeeze, linking_number, link_ropelength,
)
from derive_inductance_factor_test import L_gmd, energy_MeV, TARGET, FM, MU0  # noqa: E402
from repair_curves_braid import alexander  # noqa: E402

RHO_REF = float(__import__("os").environ.get("RHO_REF", "208.07"))


def run(trial, src="/tmp/nucleon_relaxed"):
    comps = [np.load(f"{src}_{i}.npy") for i in range(3)]
    s = (0.55, 0.65, 0.75)[trial // 2]
    rng = np.random.default_rng(100 + trial)
    u = rng.normal(size=3)
    print(f"trial {trial}: squeeze s={s} direction {np.round(u/np.linalg.norm(u), 2)}",
          flush=True)
    sq = [squeeze(C, u, s) for C in comps]
    # meme transformation affine pour tous -> Lk et types preserves
    comps2, info = sono_link(sq, iters=18000, loosen_start=False)
    lks = (linking_number(comps2[0], comps2[1]), linking_number(comps2[0], comps2[2]),
           linking_number(comps2[1], comps2[2]))
    alex = [tuple(int(x) for x in alexander(C)[0]) for C in comps2]
    ok = (alex[0] == (1, -1, 1) and alex[1] == (3, -5, 3) and alex[2] == (3, -5, 3)
          and lks == (-1, -1, 0))
    rho = info["ropelength"]
    X = comps2[0] * 0.209 * FM
    L, var = L_gmd(X, 0.209 * FM)
    E = energy_MeV(L)
    beat = "BAT-208" if (ok and rho < RHO_REF - 0.5) else "ne bat pas 208"
    print(f"RESULT probe{trial} s={s} rho={rho:.2f} gap={info['min_gap']:.3f} "
          f"gardes={'OK' if ok else 'KO'} L={L/MU0/FM:.3f} E={E:.3f} "
          f"({(E/TARGET-1)*100:+.1f}%) -> {beat}", flush=True)
    if ok and rho < RHO_REF - 0.5:
        for i, C in enumerate(comps2):
            np.save(f"/tmp/nucleon_tighter_{trial}_{i}.npy", C)


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "/tmp/nucleon_relaxed")
