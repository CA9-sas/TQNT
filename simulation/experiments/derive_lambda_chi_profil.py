"""Profil lambda_chi/a : la microphysique du vortex (marche <1% de beta_otimes).

ETAT (brouillon §13) : le London dur est exclu par les donnees -> vortex
marginal (lambda >~ a). Ici, trois volets :

1. INVENTAIRE ZERO-BOUTON des echelles du condensat chi :
   hbar*c/sqrt(sigma_chi) = 0.472 fm = 2.26 a ; 1/Lambda_chi = 0.945 fm = 4.52 a ;
   r_link = 0.99 fm = 4.7 a ; a = d_tube = 0.209 fm = lambdabar_C(p).
   TOUTES les longueurs propres du condensat sont >= 2a : le regime MOU est
   PREDIT par les echelles de la theorie (pas seulement impose par les donnees)
   -- la discrimination du §13 devient une CONSISTANCE.

2. LONDON AMORTI (tore 1D) : la redistribution minimise E_mag + E_cin ou la
   raideur cinetique du supercourant penalise la concentration :
     E_cin = (mu0 lambda^2 / 2) int j^2 dV,  courant en coquille t = lambda
     => penalite P_i = mu0 lambda pi r_i / w par I_i^2 (modele d'interpolation,
        pas une derivation du profil exact -- honnete).
   Distribution : I ~ (M + 2P)^{-1} 1 ; on rapporte L_mag[I(lambda)] et la
   fraction realisee du gap f(lambda/a) = (L_unif - L_mag)/(L_unif - L_London).

3. ENCADREMENT RESSERRE : E in [E_unif, E_unif + f(lambda_chi/a) x gap_local],
   avec lambda_chi/a >= 2.26 (l'echelle la plus courte du condensat).

NB (puzzle consigne, pas resolu ici) : si le courant chi etait un supercourant
massif litteral en coquille t~lambda, l'inductance CINETIQUE dominerait
(L_kin/L_mag ~ 1.7 (lambda/a)^2 ~ 8 !) et casserait l'accord au % de toute la
chaine ; l'accord implique donc que la circulation chi est TOPOLOGIQUEMENT
rigide (enroulement n fixe : son energie cinetique ne varie pas avec Delta Lk)
et que seul le differentiel EM compte -- contrainte pour le programme Ising.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from derive_inductance_factor_test import (  # noqa: E402
    mutual_rings, energy_MeV, D_TUBE, MU0, FM, TARGET,
)
from derive_london_solve import ring_self, torus_london_1d  # noqa: E402
from derive_noyau_elliptique import L_sheet_ring  # noqa: E402
from simulation.ideal_embedding import resample  # noqa: E402

SQSIG_LEN = 197.327 / 417.65      # hbar c / sqrt(sigma_chi), fm
LAMCHI_LEN = 197.327 / 208.8      # 1 / Lambda_chi, fm
R_LINK = 0.99                     # fm
A_TUBE = D_TUBE                   # 0.209 fm


def torus_soft_london(R, a, lam, N=128):
    """Tore : distribution minimisant E_mag + E_cin (coquille t = lambda).

    Retourne (L_mag_de_la_distribution, fraction_du_gap_realisee)."""
    th = 2 * math.pi * (np.arange(N) + 0.5) / N
    r = R + a * np.cos(th)
    z = a * np.sin(th)
    w = 2 * math.pi * a / N
    M = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = ring_self(r[i], w) if i == j else \
                mutual_rings(r[i], r[j], z[i] - z[j])
    ones = np.ones(N)
    L_unif = float(ones @ M @ ones) / N ** 2
    sol_h = np.linalg.solve(M, ones)
    L_london = 1.0 / float(ones @ sol_h)
    P = MU0 * lam * math.pi * r / w                  # penalite cinetique
    A = M + 2 * np.diag(P)
    I = np.linalg.solve(A, ones)
    I = I / I.sum()                                  # courant total 1
    L_mag = float(I @ M @ I)                         # energie EM de la distribution
    f = (L_unif - L_mag) / max(L_unif - L_london, 1e-30)
    return L_unif, L_london, L_mag, f


def run():
    a = A_TUBE * FM
    print("=== Profil lambda_chi/a : microphysique du vortex ===\n", flush=True)

    print("[1] Inventaire zero-bouton des echelles du condensat :", flush=True)
    for nom, l_fm in (("hbar c / sqrt(sigma_chi)", SQSIG_LEN),
                      ("1/Lambda_chi", LAMCHI_LEN),
                      ("r_link", R_LINK)):
        print("    %-24s = %.3f fm = %.2f a" % (nom, l_fm, l_fm / A_TUBE), flush=True)
    print("    -> toutes >= 2a : le regime MOU est PREDIT (zero bouton) ;", flush=True)
    print("    l'echelle la plus courte donne lambda_chi/a >= %.2f.\n" % (
        SQSIG_LEN / A_TUBE), flush=True)

    print("[2] London amorti (tore, N=128) : fraction realisee du gap f(lambda/a) :",
          flush=True)
    for ra in (2.0, 3.0):
        R = ra * a
        print("  r/a = %.0f :" % ra, flush=True)
        for loa in (0.0, 0.25, 0.5, 1.0, 2.0, 2.26, 4.5, 8.0):
            lam = loa * a
            if loa == 0.0:
                Lu, Ll, Lm, f = torus_soft_london(R, a, 1e-9 * a, N=128)
            else:
                Lu, Ll, Lm, f = torus_soft_london(R, a, lam, N=128)
            print("    lambda/a = %5.2f : f = %6.1f%%  (L_mag = %.4f ; "
                  "unif %.4f ; London %.4f)" % (
                      loa, f * 100, Lm / MU0 / FM, Lu / MU0 / FM, Ll / MU0 / FM),
                  flush=True)

    print("\n[3] Encadrement resserre du nucleon :", flush=True)
    C0 = np.load(os.path.join(os.path.dirname(__file__), "..", "..",
                              "output", "nucleon_ideal_0.npy"))
    X = resample(C0, 1600) * D_TUBE * FM
    L_unif = L_sheet_ring(X, a, nq=48)
    E_unif = energy_MeV(L_unif)
    # fraction realisee a lambda_chi/a = 2.26 (r/a=2, regime nucleon typique)
    _, _, _, f226 = torus_soft_london(2.0 * a, a, SQSIG_LEN * FM, N=128)
    # gap local nucleon : borne dure = carte locale (+59% sur E, surestimee) ;
    # correction realisee = f x (E_hard - E_unif)
    E_hard_low, E_hard_high = 1.293 * 1.10, 1.293 * 1.60   # bande honnete du dur
    dE_low = f226 * (E_hard_low - E_unif)
    dE_high = f226 * (E_hard_high - E_unif)
    print("    E_unif = %.3f MeV (%+.2f%%) ; f(lambda_chi/a=2.26, r/a=2) = %.1f%%" % (
        E_unif, (E_unif / TARGET - 1) * 100, f226 * 100), flush=True)
    print("    correction realisee : +[%.3f ; %.3f] MeV -> E in [%+.1f%% ; %+.1f%%]" % (
        dE_low, dE_high,
        ((E_unif + dE_low) / TARGET - 1) * 100,
        ((E_unif + dE_high) / TARGET - 1) * 100), flush=True)
    print("    (bande du 'dur' : E_hard in [+10%, +60%] -- carte locale surestimee,", flush=True)
    print("    variationnel sous-estime ; la fraction f la comprime d'un facteur %d)" % (
        round(1 / max(f226, 1e-9))), flush=True)


if __name__ == "__main__":
    run()
