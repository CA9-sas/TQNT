"""Fermer le residu : les mutuelles aux compagnons (matrice 3x3 du nucleon).

RESIDU (brouillon §14) : E = E_unif(coeur seul) = -1.6 +- 1.0 % de Dm(n-p) ;
candidat nomme = les compagnons. IMAGE CIRCUIT COMPLETE : les trois tubes
maintiennent leur fluxoide ENTIER (rigidite topologique §14) ; l'etat neutron
force le coeur hors de son optimum d'UNE unite ; les compagnons ECRANTENT pour
preserver leur quantification :

    Phi_1 = Phi_chi = L11 I1 + M12 I2 + M13 I3   (coeur force)
    Phi_2 = 0      = M12 I1 + L22 I2 + M23 I3   (compagnon quantifie)
    Phi_3 = 0      = M13 I1 + M23 I2 + L33 I3
    =>  E = Phi_chi^2/2 * [L^{-1}]_{11}  =  Phi_chi^2 / (2 L_eff),
        L_eff = 1/[L^{-1}]_{11} <= L11  =>  correction POSITIVE sur E.

CALCUL : matrice complete sur la geometrie relaxee canonique
(output/nucleon_ideal_{0,1,2}.npy) --
  - selfs L11, L22, L33 : nappe exacte (L_sheet_ring, v3) ;
  - mutuelles M12, M13, M23 : Neumann axe-axe (les tubes sont separes de
    >= 2a ; corrections de section aux contacts = second ordre, notees).
Verdict : le facteur L11/L_eff ferme-t-il le +1.6 % ?
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from derive_inductance_factor_test import (  # noqa: E402
    mutual_neumann, energy_MeV, D_TUBE, MU0, FM, TARGET,
)
from derive_noyau_elliptique import L_sheet_ring  # noqa: E402
from simulation.ideal_embedding import resample, linking_number  # noqa: E402


def run():
    a = D_TUBE * FM
    root = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    comps = [np.load(os.path.join(root, "nucleon_ideal_%d.npy" % i))
             for i in range(3)]
    print("=== Fermer le residu : matrice 3x3 du nucleon ===\n", flush=True)
    print("Lk : coeur-S1 = %d, coeur-S2 = %d, S1-S2 = %d" % (
        linking_number(comps[0], comps[1]),
        linking_number(comps[0], comps[2]),
        linking_number(comps[1], comps[2])), flush=True)

    X = [resample(C, 1600) * D_TUBE * FM for C in comps]
    L = np.zeros((3, 3))
    for i in range(3):
        L[i, i] = L_sheet_ring(X[i], a, nq=48)
        print("L%d%d (nappe exacte) = %.4f mu0.fm" % (
            i + 1, i + 1, L[i, i] / MU0 / FM), flush=True)
    for i in range(3):
        for j in range(i + 1, 3):
            L[i, j] = L[j, i] = mutual_neumann(X[i], X[j])
            print("M%d%d (Neumann axes)  = %+.4f mu0.fm" % (
                i + 1, j + 1, L[i, j] / MU0 / FM), flush=True)

    Linv = np.linalg.inv(L)
    L_eff = 1.0 / Linv[0, 0]
    E0 = energy_MeV(L[0, 0])
    E1 = energy_MeV(L_eff)
    print("\ncoeur seul   : L11   = %.4f -> E = %.3f MeV (%+.2f%%)" % (
        L[0, 0] / MU0 / FM, E0, (E0 / TARGET - 1) * 100), flush=True)
    print("avec ecrans  : L_eff = %.4f -> E = %.3f MeV (%+.2f%%)" % (
        L_eff / MU0 / FM, E1, (E1 / TARGET - 1) * 100), flush=True)
    print("correction des compagnons : %+.2f points" % (
        (E1 - E0) / TARGET * 100), flush=True)
    # courants d'ecrantage relatifs (diagnostic)
    I = Linv @ np.array([1.0, 0, 0])
    print("courants relatifs (I1 normalise) : I2/I1 = %+.3f ; I3/I1 = %+.3f" % (
        I[1] / I[0], I[2] / I[0]), flush=True)
    print("\nbeta_otimes implique : %.4f (vs 0.210 calibre)" % (E1 / 6.15), flush=True)
    print("NB : mutuelles = Neumann axe-axe ; corrections de section aux", flush=True)
    print("contacts (2a) = 2nd ordre sur M (les selfs sont deja en nappe exacte).", flush=True)


if __name__ == "__main__":
    run()
