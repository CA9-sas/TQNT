"""Mesure SONO du T(2,23) ideal : ropelength et rayons (17/07).

Pince le R_K de derive_alpha_pol_si_rk.py (terme dominant de la bande) et
teste au passage le fit lineaire de ropelength de la loi de masse
geometrique sur un T(2,q) profond.

RESULTATS (n=560, 40k iterations, min_gap = 1.994 ~ 2R) :
  T(2,3)  : L/D = 16.50 (+0.8% de 16.372 : validation)
  T(2,23) : L/D = 90.21 (+0.4% du fit lineaire 90.53 : la loi de masse
            geometrique est VALIDEE sur un T(2,q) profond)
            R_g/D = 2.648 ; R_max/D = 3.811 ; R_out/D = 4.311
  => le plongement ideal est un PELOTON COMPACT (pas un anneau a 2 brins) :
     R_K(c=23) = R_g..R_out = 0.057..0.093 fm (D = 0.0215 fm).
"""
import numpy as np
from simulation.ideal_embedding import sono, torus_knot, curve_length


def measure(p, q, n, iters):
    P0 = torus_knot(p, q, n=n)
    P, info = sono(P0, R=1.0, n=n, iters=iters)
    L = curve_length(P)                    # unites de rayon de tube (R=1)
    d = np.linalg.norm(P - P.mean(0), axis=1)
    return dict(L_D=L / 2, Rg_D=np.sqrt((d ** 2).mean()) / 2,
                Rmax_D=d.max() / 2, Rout_D=(d.max() + 1) / 2,
                min_gap=info.get("min_gap", float("nan")))


def run():
    for (p, q, n, iters) in [(2, 3, 200, 15000), (2, 23, 560, 40000)]:
        r = measure(p, q, n, iters)
        print("T(%d,%d): L/D=%.2f  Rg/D=%.3f  Rmax/D=%.3f  Rout/D=%.3f  "
              "min_gap=%.3f" % (p, q, r["L_D"], r["Rg_D"], r["Rmax_D"],
                                r["Rout_D"], r["min_gap"]))


if __name__ == "__main__":
    run()
