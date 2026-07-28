"""Application du plongement ideal : pincer la geometrie de boucle de beta_otimes.

CHAINE (aucun bouton) :
  forme  = plongement IDEAL du coeur 3_1 (minimisation de ropelength, SONO valide
           sur la valeur publiee 32.74 unites de rayon) ;
  taille = rayon de tube a = d_tube = 0.209 fm (= lambdabar_C(p), chaine moment neutron) ;
  courant = surface (c=2 PINCE, cf. derive_inductance_factor_test : le twist a courant
           net nul -> circulation de charge -> repulsion -> surface) ;
  L      = GMD-Bishop (Neumann axe<->filaments de surface, valide sur le tore exact) ;
  E      = Phi_chi^2 / (2L), Phi_chi = alpha h/e (quantum de flux derive).
CIBLE : E(dLk=1) = 1.293 MeV = Delta m(n-p) ; L* = 1.750 mu0.fm.

Aussi : 7_2 ideal (depart fermeture de tresse, garde d'Alexander), pour la variante
ou la boucle qui repond est un compagnon 7_2.

RESULTATS (2026-07-13) :
  3_1 : ropelength 33.01 (+0.8% du publie 32.74, outillage VALIDE) ; L = 1.315 mu0.fm
        -> E = 1.72 MeV (+33%). Le coeur 3_1 ideal n'atterrit PAS sur L* = 1.750 :
        le circuit circulaire effectif a r_link (+6.6%) reste la meilleure geometrie.
        CAVEAT : var-theta 64% au contact -> le modele tube-mince/courant de surface
        est en limite de validite dans les zones de contact du noeud serre (effet de
        proximite non modelise) ; le +33% porte cette incertitude de modele.
  7_2 : RESOLU par changement d'amorce (anneal_72_schedule.py) : le twist_knot_seed
        natif + rechauffes graduees donne rho = 73.3 (+11.2% du publie ; plateau NON dU
        a la discretisation, N=420 idem). Inductance du 7_2 ideal : L = 3.736 mu0.fm
        -> E = 0.606 MeV (-53%) : la variante 'boucle compagne = 7_2' est REJETEE ;
        les deux calculs (3_1 et 7_2) convergent : le circuit d'enlacement a r_link
        reste la geometrie physique. NB : la fermeture de tresse plafonne a 118 (+72%),
        bassin infranchissable par rechauffes seules -- changer d'amorce, pas le paysage.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import sono, torus_knot, resample  # noqa: E402
from derive_inductance_factor_test import (  # noqa: E402
    L_gmd, energy_MeV, L_star, fourier_smooth, TARGET, D_TUBE, FM, MU0, ROPE_DIAM,
)
from repair_curves_braid import braid_curve, alexander, BRAID_WORDS  # noqa: E402

CJ_MEAN = 6.15   # parametrisation catalogue (bookkeeping beta_otimes = E / 6.15)


def report(name, P, rho, gap, rope_pub):
    a = D_TUBE
    X = P * a * FM                      # unites de rayon -> physiques (tube a)
    L, var = L_gmd(X, a * FM)
    E = energy_MeV(L)
    print(f"\n--- {name} ideal ---")
    print(f"   ropelength = {rho:.2f} (publie {rope_pub:.2f}, "
          f"{(rho/rope_pub-1)*100:+.1f}%) ; min_gap = {gap:.3f}")
    print(f"   longueur physique = {rho*a:.2f} fm (tube a = {a} fm)")
    print(f"   L = {L/MU0/FM:.3f} mu0.fm (var-theta {var:.1f}%)  [L* = {L_star()/MU0/FM:.3f}]")
    print(f"   E(dLk=1) = {E:.3f} MeV vs {TARGET} ({(E/TARGET-1)*100:+.1f}%)"
          f"  ->  beta_otimes = {E/CJ_MEAN:.3f} MeV (vs 0.210)")
    return E


def run():
    print("=== Boucle ideale : E = Phi_chi^2/(2 L[plongement ideal, a=d_tube]) ===")
    print("cible 1.293 MeV ; rappel cercles : r_link +6.6%, rope-3_1 -8.2%")

    # --- 3_1 ideal ---
    P, info = sono(torus_knot(2, 3, n=240), iters=15000)
    report("3_1 (coeur nucleon)", P, info["ropelength"], info["min_gap"],
           2 * ROPE_DIAM["3_1"])

    # --- 7_2 ideal (depart tresse reparee, garde d'Alexander) ---
    word, ns, target_alex = BRAID_WORDS["7_2"]
    P0, _, _ = braid_curve(word, ns, pts_per_seg=40)
    P0 = fourier_smooth(resample(P0, 4096), 48)
    P, info = sono(P0, n=320, iters=15000)
    try:
        alex = alexander(P)
        print(f"\n   [garde Alexander 7_2 ideal : {alex[0]} (cible {target_alex})]")
    except Exception as e:  # pragma: no cover
        print(f"\n   [garde Alexander indisponible : {e}]")
    report("7_2 (compagnon)", P, info["ropelength"], info["min_gap"],
           2 * ROPE_DIAM["7_2"])


if __name__ == "__main__":
    run()
