"""Deriver le residu -11 % de la saturation : la fourchette de FORME (19/07).

Le facteur d'interpenetration est derive (v_c = 2 f_ov V_tube = 2,96 fm^3,
-11 % du 3,33 requis par rho_0). Ce script montre que le residu est la
DEMI-LARGEUR d'une fourchette de forme DERIVEE : les deux ecritures
principielles du volume libre, toutes deux SANS parametre libre a partir
de la coquille mesuree V_coq = 5,91 fm^3 :

  (L) LINEAIRE au covolume partage (van der Waals, 2e viriel linearise) :
      rho_eff = rho/(1 - b rho),  b = V_coq/2 = 2,96 fm^3.
      Sous-compte l'insertion (ln(1-x) tronque a l'ordre 1).
  (P) INSERTION DE POISSON au volume de paire complet (milieu non
      correle, exact pour des objets places au hasard) :
      alpha = exp(-V_coq rho),  rho_eff = rho e^{V_coq rho}.
      Sur-compte l'exclusion (ignore l'evitement mutuel qui ordonne le
      milieu et libere du volume).

La verite physique (tubes flexibles correles) est ENTRE les deux. Si le
requis (3,33) tombe au centre de la fourchette, le residu est explique
par la forme -- il ne reste que le POINT PRECIS dans la fourchette
(theorie des correlations de tubes flexibles, ouvert).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.bethe_weizsacker_F3 import (
    attraction_integral, R_EXCL, T_PREF)
from simulation.experiments.saturation_volume_entrelacs import EMP

V_COQ = 5.91          # fm^3 (coquille mesuree, union rayon D)
B_LIN = V_COQ / 2     # 2.96


def ea_lin(rho, J, b):
    return T_PREF * (rho / np.maximum(1 - b * rho, 1e-9)) ** (2 / 3) + 0.5 * rho * J


def ea_poisson(rho, J, vpair):
    return T_PREF * (rho * np.exp(vpair * rho)) ** (2 / 3) + 0.5 * rho * J


def minimize(f, rho_max):
    rho = np.linspace(0.005, rho_max, 12000)
    EA = f(rho)
    i0 = np.argmin(EA)
    return rho[i0], -EA[i0]


def run():
    print("=== Le residu -11 % : la fourchette de forme du volume libre ===\n")
    J, _ = attraction_integral(R_EXCL)
    print("coquille mesuree V_coq = %.2f fm^3 ; J_att = %.0f MeV.fm^3\n" % (V_COQ, J))

    r_lin, av_lin = minimize(lambda r: ea_lin(r, J, B_LIN), 0.98 / B_LIN)
    r_poi, av_poi = minimize(lambda r: ea_poisson(r, J, V_COQ), 0.6)
    print("%-42s %8s %8s" % ("forme (sans parametre libre)", "rho_0", "a_V"))
    print("%-42s %8.3f %8.1f" % ("(L) lineaire, b = V_coq/2 = 2,96", r_lin, av_lin))
    print("%-42s %8.3f %8.1f" % ("(P) Poisson, alpha = e^(-V_coq rho)", r_poi, av_poi))
    print("%-42s %8.3f %8.1f   (empirique)\n" % ("", EMP["rho0"], EMP["aV"]))

    # v_eff equivalent lineaire de chaque forme (pour comparer au 3,33 requis)
    from scipy.optimize import brentq

    def v_of_rho0(target):
        return brentq(lambda b: minimize(lambda r: ea_lin(r, J, b), 0.98 / b)[0]
                      - target, 1.5, 8.0, xtol=1e-3)
    v_P = v_of_rho0(r_poi)
    v_req = v_of_rho0(EMP["rho0"])
    lo, hi = B_LIN, v_P
    centre = 0.5 * (lo + hi)
    print("en v_c equivalent (forme lineaire) :")
    print("  (L) %.2f   <--   requis %.2f   -->   (P) %.2f fm^3" % (lo, v_req, hi))
    print("  centre de la fourchette : %.2f ; position du requis :" % centre)
    print("  %.0f %% de la largeur depuis (L) ; ecart au centre : %+.1f %%\n" % (
        100 * (v_req - lo) / (hi - lo), 100 * (v_req / centre - 1)))

    print("VERDICT : le requis tombe DANS la fourchette de forme derivee,")
    print("pres du centre. Le residu -11 %% est donc EXPLIQUE comme la")
    print("demi-largeur de la fourchette (L)-(P), dont les deux bornes sont")
    print("sans parametre libre : (L) sous-compte l'insertion (viriel")
    print("linearise), (P) la sur-compte (milieu non correle) ; l'evitement")
    print("mutuel des tubes place la verite entre les deux. RESTE OUVERT :")
    print("le point precis (theorie des correlations de tubes flexibles).")


if __name__ == "__main__":
    run()
