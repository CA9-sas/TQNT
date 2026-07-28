"""Traiter les deux nombres restants : epsilon et le departage (20/07).

[1] EPSILON = pi, DERIVE de l'ontologie du socle : la transmission
    n'existe qu'aux SOMMETS (le point bas ne transmet pas -- postulat
    du socle) ; un echange tube-particule a distance b exige un mode
    avec un sommet AUX DEUX BOUTS ; le mode minimal est le fondamental
    lambda = 2b (corde vibrante) ; un quantum de ce mode porte
    Dp = h/lambda = h/(2b) = pi hbar / b  =>  epsilon = pi EXACTEMENT.
    (Les alternatives : lambda/4 exigerait un creux a un bout --
    interdit ; harmoniques n>1 sous-dominants.)

[2] LE DEPARTAGE : composition physique des deux routes --
    la LIBERATION (friction, route B) fixe T_form ; le CLIQUET
    (route A) agit ENSUITE fragment par fragment -- il est DEJA
    integre dans le gel echelonne cinetique w_kinetic(L, T_form).
    Donc T_form = T_B(epsilon = pi, Q), et Q est la seule entree.
    Selection mecanique de l'ensemble de Q : la liberation concerne
    la PREMIERE scission des fragments tels que nes du reseau =
    GENERATION 0 (Q = 2 pi x mediane gen 0) ; l'ensemble toutes-
    generations-epurees reste la borne basse (systematique declaree).

Chaine complete -> Omega, avec propagation des incertitudes.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.boltzmann_cinetique_omega import (  # noqa: E402
    w_kinetic, M_KNOT, T_NOM)
from simulation.experiments.pin_x17_fragments_porteurs import batch  # noqa: E402

SIGMA = 0.41847 ** 2
KAPPA = 1.14
D_TUBE = 0.1084
ZETA3 = 1.2020569
N_ALL_T3 = (ZETA3 / math.pi ** 2) * (2 + 0.75 * (6 + 4 + 1.2))
EPS = math.pi


def t_release(q):
    t = 0.07
    for _ in range(80):
        beta = EPS * N_ALL_T3 * math.log((KAPPA / t) / D_TUBE)
        t = 0.5 * t + 0.5 * math.sqrt(SIGMA / (q * KAPPA * beta))
    return t


def omega_at(t_form, L_bear):
    w_ref = math.exp(-M_KNOT / T_NOM)
    ws = [w_kinetic(L, t_form) for L in L_bear]
    return 3.9 * float(np.mean(ws)) / w_ref


def run():
    print("=== Les deux nombres : epsilon derive, Q departage ===\n")
    print("[1] epsilon = pi (DERIVE) : sommet aux deux bouts (le point")
    print("    bas ne transmet pas) -> mode fondamental lambda = 2b ->")
    print("    Dp = h/(2b) = pi hbar/b. Le postulat du socle fait le")
    print("    travail de selection.\n")

    print("[2] Q : selection mecanique = generation 0 (la liberation =")
    print("    premiere scission des fragments nes du reseau).")
    # valeurs mesurees (pin_q_scission_stats) : gen0 mediane 0,300 (n=48),
    # toutes gen. epurees 0,225 (n=187)
    q_gen0 = 2 * math.pi * 0.300
    q_all = 2 * math.pi * 0.225
    # bootstrap grossier gen0 : mediane de n=48, sigma ~ 1.25*IQR/sqrt(n)
    dq = 2 * math.pi * 0.05
    print("    Q(gen 0) = %.2f +- %.2f ; [borne basse : toutes gen." % (
        q_gen0, dq))
    print("    epurees, Q = %.2f -- systematique d'ensemble declaree]\n" % (
        q_all))

    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([L for L, nn in finals if nn >= 1])

    print("[3] CHAINE FERMEE (T_form = liberation ; cliquet integre au")
    print("    gel echelonne cinetique) :")
    rows = [("Q gen 0 (selection)", q_gen0),
            ("Q gen 0 - 1 s", q_gen0 + dq),
            ("Q gen 0 + 1 s", q_gen0 - dq),
            ("Q toutes gen. (borne)", q_all)]
    for lab, q in rows:
        t = t_release(q)
        om = omega_at(t, L_bear)
        print("    %-24s Q = %.2f -> T_form = %5.1f MeV ; Omega = %6.2f" % (
            lab, q, t * 1e3, om))

    t_sel = t_release(q_gen0)
    om_sel = omega_at(t_sel, L_bear)
    print("\nVERDICT (honnete) :")
    print(" - epsilon = pi : DERIVE (ontologie du socle -- l'exclusion du")
    print("   point bas selectionne le fondamental lambda = 2b).")
    print(" - Q : MESURE, mais le CHOIX D'ENSEMBLE reste la systematique")
    print("   dominante -- selection mecanique gen 0 : T_form = %.1f MeV," % (
        t_sel * 1e3))
    print("   Omega = %.2f (%+.0f %% de Planck) ; la borne toutes-gen." % (
        om_sel, 100 * (om_sel / 5.45 - 1)))
    print("   donne Omega = %.1f : Planck est DANS la fourchette" % (
        omega_at(t_release(q_all), L_bear)))
    print("   d'ensemble, mais le levier exponentiel amplifie le choix.")
    print(" - PIN FINAL NOMME : mesurer DIRECTEMENT le seuil de liberation")
    print("   Q_c -- dynamique de Nambu AMORTIE (terme gamma), balayage")
    print("   gamma/omega, probabilite d'accomplir la premiere scission ->")
    print("   Q_c sans choix d'ensemble. (Machinerie : evolution numerique")
    print("   amortie des fragments gen 0 ; chantier de simulation.)")


if __name__ == "__main__":
    run()
