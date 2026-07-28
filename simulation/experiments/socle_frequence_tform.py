"""La frequence vibratoire de l'espace-temps -> T_form (20/07).

PROPOSITION (utilisateur) : on connait le pas de l'engrenage (la
longueur de Planck) et la vitesse de la vis sans fin (c) -> la
frequence vibratoire de l'espace-temps se CALCULE :

    f_socle = c / l_P     (frequence de Planck)

et son quantum d'energie est hbar omega_socle = E_P = M_Pl c^2
(IDENTITE EXACTE). Consequences structurelles :
 (i) la RIGIDITE du raccord (M_Pl, ancre de transmutation unique)
     EST la frequence du socle -- l'ancre M_Pl de toutes les
     equations de gel recoit son sens : le log ln(M_Pl/T) COMPTE LES
     DENTS D'ENGRENAGE entre l'horloge du socle et le processus ;
 (ii) le gel ancre M_Pl (etape 1, cliquet) devient le gel HORLOGE
     PAR LE SOCLE : T_form = E_b / ln(nombre de dents), avec la
     barriere MESUREE (fragment porteur / quantum de domaine).

BANC : calcul de f_socle ; T_form horloge-socle pour les deux
barrieres mesurees ; gel echelonne CINETIQUE -> Omega ; confrontation
avec la route friction (epsilon=pi, Q mesure).
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.boltzmann_cinetique_omega import (  # noqa: E402
    w_kinetic, M_KNOT, T_NOM)
from simulation.experiments.pin_x17_fragments_porteurs import batch  # noqa: E402

HBAR_SI = 1.054571817e-34
C_SI = 2.99792458e8
L_P_SI = 1.616255e-35
M_PL = 1.2209e19          # GeV
G_S = 15.2
E_PORTEUR = 2.76          # GeV (mesure, etape 4)
E_DOMAINE = 2.99          # GeV (quantum de domaine)


def t_gel_socle(e_b):
    t = 0.05
    for _ in range(100):
        t = 0.5 * t + 0.5 * e_b / math.log(M_PL / (1.66 * math.sqrt(G_S) * t))
    return t


def run():
    print("=== Frequence vibratoire de l'espace-temps -> T_form ===\n")

    f_socle = C_SI / L_P_SI
    e_quantum = HBAR_SI * 2 * math.pi * f_socle / 1.602176634e-10  # GeV (h f)
    e_red = HBAR_SI * f_socle / 1.602176634e-10                    # hbar c/l_P
    print("[1] LE CALCUL PROPOSE :")
    print("   pas de l'engrenage l_P = %.4e m ; vitesse c" % L_P_SI)
    print("   -> f_socle = c/l_P = %.4e Hz" % f_socle)
    print("   quantum : hbar c/l_P = %.4e GeV = E_P (M_Pl = %.4e)" % (
        e_red, M_PL))
    print("   IDENTITE EXACTE : la frequence vibratoire de l'espace-temps")
    print("   est la frequence de Planck ; son quantum est LA MASSE DE")
    print("   PLANCK -- la rigidite du raccord EST l'horloge du socle.\n")

    print("[2] CONSEQUENCE : les ln(M_Pl/...) des equations de gel =")
    print("   COMPTAGE DES DENTS entre l'horloge du socle et le processus")
    print("   (a T = 67 MeV : %.1f e-plis, ~e^45 dents par periode du" % (
        math.log(M_PL / 0.067)))
    print("   processus). L'ancre M_Pl de l'etape 1 est FONDEE.\n")

    print("[3] T_form HORLOGE-SOCLE (barrieres mesurees) + Omega cinetique :")
    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([L for L, nn in finals if nn >= 1])
    w_ref = math.exp(-M_KNOT / T_NOM)
    for name, e_b in (("fragment porteur (2,76 GeV mesure)", E_PORTEUR),
                      ("quantum de domaine (2,99 GeV)", E_DOMAINE)):
        tf = t_gel_socle(e_b)
        ws = [w_kinetic(L, tf) for L in L_bear]
        om = 3.9 * float(np.mean(ws)) / w_ref
        print("   %-36s T_form = %.1f MeV ; Omega = %.2f" % (
            name, tf * 1e3, om))

    print("\n[4] CONFRONTATION DES DEUX ROUTES (meme ancre socle-M_Pl) :")
    print("   route horloge-socle (cliquet)   : T_form = 61,5-66,8 ->")
    print("     Omega ~ 0,3-1,5 (bord bas de Planck)")
    print("   route friction ondelette-hbar    : T_form = 64-74 (e=pi,")
    print("     Q mesure) -> Omega ~ 1,5-16 (encadre Planck)")
    print("   -> les deux routes ENCADRENT Planck (5,45) ; elles partagent")
    print("   desormais la meme ancre (l'horloge du socle) ; le depart")
    print("   entre elles = QUI gele en premier (la plus haute des deux")
    print("   temperatures gagne : le processus se fige des que l'un des")
    print("   deux canaux decroche).\n")

    t_fric_lo, t_fric_hi = 0.064, 0.074
    t_cliq_lo, t_cliq_hi = t_gel_socle(E_PORTEUR), t_gel_socle(E_DOMAINE)
    t_eff_lo = max(t_fric_lo, t_cliq_lo)
    t_eff_hi = max(t_fric_hi, t_cliq_hi)
    ws_lo = [w_kinetic(L, t_eff_lo) for L in L_bear]
    ws_hi = [w_kinetic(L, t_eff_hi) for L in L_bear]
    om_lo = 3.9 * float(np.mean(ws_lo)) / w_ref
    om_hi = 3.9 * float(np.mean(ws_hi)) / w_ref
    print("[5] REGLE << LE PREMIER QUI DECROCHE >> : T_form = max(routes)")
    print("   -> T_form in [%.1f, %.1f] MeV -> Omega in [%.1f, %.1f]" % (
        t_eff_lo * 1e3, t_eff_hi * 1e3, om_lo, om_hi))
    print("   Planck 5,45 : %s" % (
        "DANS la fourchette" if om_lo <= 5.45 <= om_hi else "hors"))


if __name__ == "__main__":
    run()
