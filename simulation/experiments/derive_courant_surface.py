"""Modele de courant AU-DELA du tube-mince : nappe de surface + condition de London.

VERROU (brouillon beta_otimes §9) : la bande E in [+0.9, +2.5]% du nucleon repoli
est portee par le modele d'inductance tube-mince (L_gmd = moyenne axe<->filament,
var-theta 46-48%). Ici : le modele EXACT de la nappe, sans paramètre libre.

PHYSIQUE : le courant chi est de type London (fluxoide) -- il vit sur la surface
du tube et s'y redistribue. La quantification du fluxoide impose un flux EGAL a
travers CHAQUE contour de surface :   M . I = Phi . 1
d'ou I = Phi M^{-1} 1 et  E = Phi^2/(2 L_eff),  L_eff = 1/(1^T M^{-1} 1).
  - M = matrice de Neumann entre N filaments de surface (repere de Bishop,
    anti-glitch de Frenet) ;
  - auto-termes M_kk regularises par la GMD de bande : deux filaments separes
    de g = e^{-3/2} w (w = 2 pi a / N largeur de bande) -- methode de Grover ;
  - N -> infini : somme de Riemann de la double integrale de nappe exacte.
Deux distributions : UNIFORME (L_unif = moyenne de M, l'approximation que
L_gmd vise) et LONDON (L_eff, la distribution physique d'energie minimale /
flux egal -- se redistribue aux contacts et dans les coudes).

VALIDATION : cercle r/a = 3 et 5 vs analytique (L_torus_surface) et tube-mince
mu0 r (ln(8r/a) - 2). Puis : coeur 3_1 du nucleon relaxe (geometrie archivee
output/nucleon_ideal_0.npy, celle du +0.7%).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from derive_inductance_factor_test import (  # noqa: E402
    bishop_frame, mutual_neumann, L_torus_surface, energy_MeV,
    D_TUBE, MU0, FM, TARGET, L_gmd,
)
from simulation.ideal_embedding import resample  # noqa: E402

GMD_STRIP = math.exp(-1.5)   # GMD d'une bande plate de largeur w : g = e^{-3/2} w


def surface_filaments(P, a, ntheta):
    """N filaments sur la surface du tube (repere de Bishop)."""
    _, N, B = bishop_frame(P)
    out = []
    for th in np.linspace(0, 2 * math.pi, ntheta, endpoint=False):
        out.append(P + a * (math.cos(th) * N + math.sin(th) * B))
    return out


def L_sheet(P, a, ntheta=12):
    """Inductances de la nappe de surface : (L_uniforme, L_London).

    M_kl : Neumann filament k <-> l ; M_kk : auto-terme de bande (GMD de
    Grover, filament vs filament decale de g = e^{-3/2} w sur la surface).
    """
    _, N, B = bishop_frame(P)
    thetas = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
    fils = [P + a * (math.cos(th) * N + math.sin(th) * B) for th in thetas]
    w = 2 * math.pi * a / ntheta
    dth = GMD_STRIP * w / a
    M = np.zeros((ntheta, ntheta))
    for k in range(ntheta):
        Fk_off = P + a * (np.cos(thetas[k] + dth) * N + np.sin(thetas[k] + dth) * B)
        M[k, k] = mutual_neumann(fils[k], Fk_off)
        for l in range(k + 1, ntheta):
            M[k, l] = M[l, k] = mutual_neumann(fils[k], fils[l])
    L_unif = float(M.mean())
    ones = np.ones(ntheta)
    L_london = float(1.0 / (ones @ np.linalg.solve(M, ones)))
    return L_unif, L_london


def circle(r, n):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), 0 * t], axis=1)


def run():
    print("=== Nappe de surface + London : validation puis nucleon ===\n", flush=True)
    a = D_TUBE * FM

    # -- 1. validation cercle --
    print("[1] Cercle (analytique) :", flush=True)
    for ra in (3.0, 5.0):
        R = ra * a
        C = circle(R, 1200)
        L_thin = MU0 * R * (math.log(8 * ra) - 2)
        L_ana = L_torus_surface(R, a)
        Lg, var = L_gmd(C, a)
        for nth in (8, 12, 16):
            Lu, Ll = L_sheet(C, a, ntheta=nth)
            print("   r/a=%.0f nth=%2d : unif %.4f | London %.4f | gmd %.4f "
                  "| analytique %.4f | mince %.4f  (mu0.fm)" % (
                      ra, nth, Lu / MU0 / FM, Ll / MU0 / FM, Lg / MU0 / FM,
                      L_ana / MU0 / FM, L_thin / MU0 / FM), flush=True)

    # -- 2. nucleon (geometrie archivee du +0.7%) --
    print("\n[2] Coeur 3_1 du nucleon relaxe (nucleon_ideal_0.npy) :", flush=True)
    root = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    C = np.load(os.path.join(root, "nucleon_ideal_0.npy"))
    for npts in (1200, 1800):
        X = resample(C, npts) * D_TUBE * FM
        Lg, var = L_gmd(X, a)
        Lu, Ll = L_sheet(X, a, ntheta=12)
        for tag, L in (("gmd (modele actuel)", Lg), ("nappe uniforme", Lu),
                       ("nappe LONDON", Ll)):
            E = energy_MeV(L)
            print("   n=%4d %-20s : L = %.4f mu0.fm ; E = %.3f MeV (%+.1f%%)" % (
                npts, tag, L / MU0 / FM, E, (E / TARGET - 1) * 100), flush=True)
        print("   (var-theta gmd : %.0f%% -- obsolete pour la nappe, "
              "double somme exacte)" % var, flush=True)

    print("\nLECTURE : si London deplace E de -1 a -2.5 points vs gmd, la bande", flush=True)
    print("[+0.9, +2.5]%% se referme sur la cible ; sinon le modele de courant", flush=True)
    print("n'etait pas le residu et il faudra chercher ailleurs (honnete).", flush=True)


if __name__ == "__main__":
    run()
