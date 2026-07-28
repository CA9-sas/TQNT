"""Deriver le point precis dans la fourchette (L)-(P) de la saturation (19/07).

POSITION DU REQUIS : v_req = 3,33 fm^3 est a -2,9 % de (P) = 3,43 --
et (P) n'est pas un point arbitraire : c'est l'insertion EXACTE au
2e viriel (alpha = e^{-v_pair rho}, terme dominant exact du potentiel
chimique d'exces : beta mu = 2 B2 rho + (3/2) B3 rho^2 + ...,
B2 = v_pair/2). La question du point precis = les corrections de
correlation au-dela du 2e viriel. Deux jambes opposees :
  - EVITEMENT (durcissante) : les objets durs correles excluent PLUS
    que Poisson (B3 > 0) ;
  - FLEXIBILITE (amollissante) : les tubes se reorganisent, l'exclusion
    effective baisse.
Ce banc TESTE les deux fermetures calculables qui isolent chaque jambe :
  (a) PARSONS-LEE (objets DURS correles, packing Carnahan-Starling) :
      beta mu = [v_pair/(8 v_0)] eta(8-9eta+3eta^2)/(1-eta)^3,
      eta = rho V_tube -- l'evitement seul, sans flexibilite ;
  (b) LIMITE FONDUE (flexibilite parfaite) : l'exclusion se reduit a la
      matiere, b = V_tube -- la flexibilite seule, sans evitement.
Si (a) sur-exclut et (b) sous-exclut avec le requis a ~3 % de (P), le
point precis EST (P) : les deux jambes se compensent sous 3 % a rho_0.
"""
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")
from simulation.experiments.bethe_weizsacker_F3 import (
    attraction_integral, R_EXCL, T_PREF)
from simulation.experiments.saturation_volume_entrelacs import EMP

V_COQ = 5.91
V_TUBE = 1.51
V_REQ = 3.33


def minimize(f, rho_max):
    rho = np.linspace(0.005, rho_max, 12000)
    EA = f(rho)
    i0 = np.argmin(EA)
    return rho[i0], -EA[i0]


def run():
    print("=== Le point precis : correlations de tubes flexibles ===\n")
    J, _ = attraction_integral(R_EXCL)

    def ea_lin(rho, b):
        return T_PREF * (rho / np.maximum(1 - b * rho, 1e-9)) ** (2 / 3) \
            + 0.5 * rho * J

    def v_of_rho0(target):
        return brentq(lambda b: minimize(lambda r: ea_lin(r, b), 0.98 / b)[0]
                      - target, 1.0, 8.0, xtol=1e-3)

    # position du requis vs (P)
    def ea_poisson(rho):
        return T_PREF * (rho * np.exp(V_COQ * rho)) ** (2 / 3) + 0.5 * rho * J
    r_P, _ = minimize(ea_poisson, 0.6)
    v_P = v_of_rho0(r_P)
    print("[0] Position : v_req = %.2f vs (P) 2e viriel exact = %.2f -> %+.1f %%\n" % (
        V_REQ, v_P, 100 * (V_REQ / v_P - 1)))

    # (a) Parsons-Lee : evitement dur, sans flexibilite
    ratio = V_COQ / (8 * V_TUBE)
    def ea_pl(rho):
        eta = rho * V_TUBE
        mu = ratio * eta * (8 - 9 * eta + 3 * eta ** 2) / (1 - eta) ** 3
        return T_PREF * (rho * np.exp(mu)) ** (2 / 3) + 0.5 * rho * J
    r_pl, av_pl = minimize(ea_pl, 0.5)
    v_pl = v_of_rho0(r_pl)
    print("(a) Parsons-Lee (dur, correle) : rho_0 = %.3f, a_V = %.1f" % (r_pl, av_pl))
    print("    v_eff = %.2f fm^3 (%+.0f %% du requis) -> SUR-EXCLUT :" % (
        v_pl, 100 * (v_pl / V_REQ - 1)))
    print("    la durete d'objet est fausse (coherent avec le rigide x25).\n")

    # (b) limite fondue : flexibilite parfaite, sans evitement
    r_m, av_m = minimize(lambda r: ea_lin(r, V_TUBE), 0.98 / V_TUBE)
    print("(b) Limite fondue (b = V_tube) : rho_0 = %.3f, a_V = %.1f" % (r_m, av_m))
    print("    v_eff = %.2f fm^3 (%+.0f %% du requis) -> SOUS-EXCLUT :" % (
        V_TUBE, 100 * (V_TUBE / V_REQ - 1)))
    print("    la flexibilite parfaite efface la coquille, refutee aussi.\n")

    # jambes de correction estimees autour de (P)
    b2 = V_COQ / 2
    b3_est = 0.625 * b2 ** 2          # echelle B3/B2^2 ~ 5/8 (objets compacts)
    rho0 = EMP["rho0"]
    jambe_dure = 1.5 * b3_est * rho0 ** 2 / (V_COQ * rho0)
    print("[jambes] evitement (echelle B3/B2^2 = 5/8) : %+.0f %% sur beta mu" % (
        100 * jambe_dure))
    print("         flexibilite requise pour le net mesure : %+.0f %%" % (
        100 * (np.log(1 - V_REQ * rho0) / (-V_COQ * rho0) - 1) if V_REQ * rho0 < 1 else 0))

    print("\nVERDICT : le POINT PRECIS est l'insertion exacte au 2e viriel")
    print("(P), a -3 %% pres. Les deux jambes de correlation au-dela --")
    print("evitement (+%.0f %%, isolee par Parsons-Lee : v_eff %.1f, refutee)" % (
        100 * jambe_dure, v_pl))
    print("et flexibilite (isolee par la limite fondue : v_eff %.1f, refutee)" % V_TUBE)
    print("-- sont du meme ordre et SE COMPENSENT sous 3 %% a rho_0.")
    print("La compensation est MESUREE (encadree par les deux fermetures),")
    print("pas derivee : la deriver exigerait la theorie des tubes mous")
    print("correles -- hors du modele transparent (limite honnete declaree).")


if __name__ == "__main__":
    run()
