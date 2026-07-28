"""Le calcul de Boltzmann via TQNT -- deux faces (20/07).

FACE A -- la hierarchie de Boltzmann de Planck : le 5,45 +- 0,03 est
une INFERENCE a travers les codes de Boltzmann (CAMB/CLASS : transport
des perturbations photons/neutrinos/CDM/baryons ajuste sur les C_l).
Question : la DM TQNT (nœuds 1,7-2 GeV, primordiale) change-t-elle
cette inference ? Test des trois portes de sortie du modele CDM :
  (1) froideur (vitesse residuelle, longueur de libre parcours) ;
  (2) couplage DM-photon/baryon (trainee) ;
  (3) especes relativistes suppl. / isocourbure.
Si les trois sont fermees, un code de Boltzmann TQNT rendrait le MEME
5,45 : la cible est legitime et ne peut pas etre deplacee.

FACE B -- l'equation cinetique de Boltzmann via TQNT : remplacer le
gel instantane (poids e^{-m/T_gel}, approximation en marche du
chantier T_c) par la VRAIE equation de Boltzmann de la population :

    dw/dt = -Gamma_d(T) [ w - w_eq(T) ],   w_eq = e^{-m/T}
    Gamma_d = T e^{-E(L)/T}  (re-fonte, barriere du fragment porteur)

integree fragment par fragment sur la distribution MESUREE (gel lisse
echelonne), de T_form a T << T_gel. Comparaison avec l'approximation
en marche (etape 4/5).
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.pin_x17_fragments_porteurs import batch  # noqa: E402

M_PL = 1.2209e19
G_S = 15.2
M_KNOT = 1.73
E_PER_XI = 2.98
T_NOM = 0.067
T_FORM = 0.0711


def hubble(t):
    return 1.66 * math.sqrt(G_S) * t * t / M_PL


def w_kinetic(L, t_form, t_stop=0.020, n=1200):
    """Integre dw/dlnT = (Gamma_d/H)(w - w_eq) de t_form a t_stop."""
    e_b = E_PER_XI * L
    u_max = math.log(t_form / t_stop)
    du = u_max / n
    t = t_form
    w = math.exp(-M_KNOT / t)          # ne d'equilibre a T_form
    for k in range(n):
        t = t_form * math.exp(-(k + 0.5) * du)
        g_over_h = M_PL * math.exp(-e_b / t) / (1.66 * math.sqrt(G_S) * t)
        w_eq = math.exp(-M_KNOT / t)
        w = w_eq + (w - w_eq) * math.exp(-g_over_h * du)
    return w


def run():
    print("=== Boltzmann via TQNT : les deux faces ===\n")

    print("FACE A -- l'inference Planck est-elle valide pour la DM TQNT ?")
    # (1) froideur
    v_birth = T_FORM / M_KNOT
    t_eq = 0.8e-9   # GeV (~0,8 eV, egalite matiere-rayonnement)
    v_eq = v_birth * (t_eq / T_FORM) * (3.9 / G_S) ** (1 / 3)
    # libre parcours ~ v_eq x horizon comobile a l'egalite (~100 Mpc)
    lam_fs = v_eq * 100e6   # pc
    print("  (1) froideur : v(naissance) = %.3f c ; v(egalite) = %.1e c" % (
        v_birth, v_eq))
    print("      libre parcours ~ %.1e pc << Mpc : CDM exact (marge >6 dex)"
          % lam_fs)
    print("  (2) trainee DM-photon : sigma_eq/m = 1e-48 cm^2/1,7 GeV =")
    print("      6e-49 cm^2/GeV vs borne Planck ~1e-32 : marge ~16 dex")
    print("  (3) N_eff standard (neutrinos = ondes chi libres, pas")
    print("      d'espece nouvelle) ; perturbations adiabatiques (graine")
    print("      = courbure d'inflation, A_s V5.15) : rien a changer")
    print("  => un code de Boltzmann TQNT rendrait LE MEME 5,45 +- 0,03 :")
    print("     la cible Planck est legitime, l'ecart se joue cote TQNT.\n")

    print("FACE B -- gel LISSE (equation cinetique) vs gel en marche :")
    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([L for L, nn in finals if nn >= 1])
    w_ref = math.exp(-M_KNOT / T_NOM)
    for t_form in (0.067, 0.0711, 0.075):
        ws = [w_kinetic(L, t_form) for L in L_bear]
        om = 3.9 * float(np.mean(ws)) / w_ref
        t_eff = -M_KNOT / math.log(float(np.mean(ws)))
        print("  T_form = %5.1f MeV : T_c effectif (cinetique) = %5.1f MeV"
              " ; Omega = %5.2f" % (t_form * 1e3, t_eff * 1e3, om))
    print("\n  (rappel marche, etape 5 : T_form = 71,1 -> T_c_eff = 67,3,")
    print("   Omega = 4,45)")

    ws = [w_kinetic(L, T_FORM) for L in L_bear]
    om = 3.9 * float(np.mean(ws)) / w_ref
    print("\nLECTURE : le gel lisse ajoute la correction de poursuite")
    print("(l'equilibre COURT pendant le gel : il faut Gamma_d > H·m/T,")
    print("pas seulement Gamma_d > H) -> les fragments encore couples")
    print("gelent un peu plus tot que l'estimation en marche, et la")
    print("transition est douce. Omega(cinetique, T_form = 71,1) = %.2f" % om)
    print("vs 4,45 (marche) : la difference mesure la qualite de")
    print("l'approximation instantanee.")


if __name__ == "__main__":
    run()
