"""Etudier T_form : la temperature de liberation du reseau (20/07).

CIBLE (etape 4) : le gel echelonne sur la distribution mesuree exige
T_form ~ 70 MeV (Planck pince ; Omega explose au-dela de ~100).

MECANISME CANDIDAT : VERROUILLAGE PAR FRICTION puis LIBERATION.
Apres la condensation (T_trans ~ 155-297 MeV), le reseau est
SUR-AMORTI par la trainee du plasma (F/L = beta T^3 v, forme
d'Everett) : pas d'oscillation a l'echelle de domaine, pas de
cascade -- le reseau est fige (c'est le verrouillage). L'amortissement
(gamma = beta T^3/sigma) decroit plus vite que la frequence de domaine
(omega = T/kappa_dom) : la liberation a lieu quand gamma = omega :

    T_form = sqrt( sigma_chi / (beta kappa_dom) )

Fermee : ne contient que sigma_chi, kappa_dom et le coefficient de
friction beta (= somme des especes x transparence O(1)).

BANC : [1] beta requis par la cible ; [2] candidats de composition
(le O(1) cinetique est LE reste) ; [3] fermeture de la chaine
(T_form -> gel echelonne sur la distribution mesuree -> T_c_eff,
Omega) ; [4] candidats alternatifs pour T_form, tries.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.pin_x17_fragments_porteurs import (  # noqa: E402
    batch, t_gel)

SIGMA = 0.41847 ** 2
KAPPA = 1.14
G_S = 15.2
M_EDGE = 1.73
M_PI = 0.1395
T_NOM = 0.067
E_PER_XI = 2.98


def t_form(beta):
    return math.sqrt(SIGMA / (beta * KAPPA))


def run():
    print("=== Etudier T_form : liberation du reseau par chute de friction ===\n")

    print("[1] LE BETA REQUIS PAR LA CIBLE (T_form = sqrt(sigma/(beta kappa))) :")
    for tt in (0.067, 0.070, 0.075):
        b = SIGMA / (KAPPA * tt * tt)
        print("   T_form = %3.0f MeV  ->  beta_requis = %.1f  (= %.2f g_s)" % (
            tt * 1e3, b, b / G_S))
    print("   -> la friction requise est un O(1) x g_s : AUCUNE echelle")
    print("      exotique n'est necessaire -- le pincement Planck tombe")
    print("      dans la gamme naturelle du plasma.\n")

    print("[2] CANDIDATS DE COMPOSITION (transparence b_a par espece = le")
    print("    O(1) cinetique restant) :")
    cands = [
        ("beta = 2 g_s (b = 2 toutes especes)", 2 * G_S),
        ("beta = g_s (b = 1)", G_S),
        ("beta = 3 g_s (b = 3)", 3 * G_S),
    ]
    for lab, b in cands:
        t = t_form(b)
        print("   %-38s beta = %5.1f -> T_form = %5.1f MeV" % (
            lab, b, t * 1e3))
    print("   -> b = 2 (deux polarisations transverses du mouvement de")
    print("      corde, candidat a deriver) donne 71,1 MeV, a -1,3 %% de")
    print("      la cible 70 ; b=1 et b=3 sont hors cible (100,5 / 58,0).\n")

    print("[3] FERMETURE DE LA CHAINE (T_form = 71,1 -> gel echelonne sur")
    print("    la distribution MESUREE des porteurs) :")
    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([L for L, n in finals if n >= 1])
    tf = t_form(2 * G_S)
    ws = [math.exp(-M_EDGE / min(tf, t_gel(E_PER_XI * L))) for L in L_bear]
    w_mean = float(np.mean(ws))
    t_eff = -M_EDGE / math.log(w_mean)
    om = 3.9 * w_mean / math.exp(-M_EDGE / T_NOM)
    print("   T_form = %.1f MeV -> T_c effectif = %.1f MeV ;"
          " Omega = %.2f (Planck 5,45 : %+.0f %%)" % (
              tf * 1e3, t_eff * 1e3, om, 100 * (om / 5.45 - 1)))
    print("   CHAINE COMPLETE : sigma_chi + kappa_dom + g_s + M_Pl +")
    print("   distribution mesuree -> Omega, avec UN O(1) restant (b=2).\n")

    print("[4] CANDIDATS ALTERNATIFS pour T_form, tries :")
    # (a) chute pionique : n_pi(T) (NR) vs friction requise sigma T/kappa
    def n_pi(t):
        return 3 * (M_PI * t / (2 * math.pi)) ** 1.5 * math.exp(-M_PI / t)
    t70 = 0.070
    ratio = n_pi(t70) / (SIGMA * t70 / KAPPA)
    print("   (a) chute de la friction PIONIQUE (T ~ m_pi/2 = %.1f MeV," % (
        M_PI / 2 * 1e3))
    print("       echelle derivee E_Z3/2) : estimation cinetique AB")
    print("       F/L ~ n_pi v -> friction/requise = %.1e a 70 MeV :" % ratio)
    print("       TROP FAIBLE x%.0f sous l'estimation naive -- defavorise" % (
        1 / ratio))
    print("       (sauf section efficace hadronique >> AB : a chiffrer).")
    print("   (b) saturation de sigma(T) (~0,5 T_deconf) : graduelle, pas")
    print("       d'echelle nette -- ecarte.")
    print("   (c) pas geometrique Delta_m = 70,2 MeV : seuil local, ne")
    print("       verrouille ni ne libere rien (etape 3) -- ecarte.")
    print("   (d) beta = 2 g_s (liberation de friction generique) : SEUL")
    print("       candidat avec mecanisme + forme close + cible atteinte.\n")

    print("BILAN : T_form a un MECANISME (verrouillage par friction,")
    print("liberation a gamma = omega) et une FORME CLOSE")
    print("T_form = sqrt(sigma_chi/(beta kappa_dom)) ; la cible Planck")
    print("exige beta = %.1f = 2,06 g_s -- nature O(1) x g_s confirmee ;" % (
        SIGMA / (KAPPA * 0.070 ** 2)))
    print("le reste du chantier est UN nombre : la transparence cinetique")
    print("b (candidat b = 2, deux polarisations transverses, a deriver")
    print("par le calcul de trainee sur le tube chi).")


if __name__ == "__main__":
    run()
