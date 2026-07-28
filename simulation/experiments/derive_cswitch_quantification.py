"""Deriver c_switch depuis la quantification du flux (20/07).

LE POINT DE STRUCTURE (qui evacue le log 45) : le gel de la fenetre
de formation n'est PAS un gel cosmologique (Gamma vs H -- qui
donnerait E/T = ln(M_Pl/T) ~ 45 et T* ~ 4 MeV, refute par tout le
reste). C'est un gel MICROPHYSIQUE : les tentatives de croisement et
la dynamique du fragment partagent la MEME horloge de cascade -- la
topologie decroche quand le facteur de Boltzmann tombe sous le
rapport des horloges microphysiques :

    c_switch = ln( tau_topologie / tau_cadence )   [O(ln O(1))]

HORLOGE MESUREE : les statistiques t* de la cascade branchee donnent
UN contact par ~0,30 periode de fragment (mediane gen 0, pin de Q).
Donc le rapport des horloges est MESURE :

    c_switch = ln(1/t*) = ln(1/0,300) = 1,20.

CANDIDATS COMPARES (discipline du null) :
  - par tentative (horloge partagee stricte)      : c_switch = 1,0 ;
  - par periode (t* mesure)                       : c_switch = 1,20 ;
  - sites de croisement (ln c = ln 23)            : 3,14 -- EXCLU par
    l'argument d'horloge (les sites partagent la meme cadence, pas
    des horloges independantes) ;
  - ln(xi/D) (rapport cascade/cœur)               : 4,7 -- EXCLU : les
    tentatives sont a l'horloge du FRAGMENT (mesure t*), pas du cœur.

PROPAGATION : T_c(c_switch) -> Omega (chaine canonique, comptage
mesure, masses pleines).
"""
import math
import sys

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
D0_SIG = 0.1084 * SIGMA
T_TRANS = 0.155
EXPO = 2 * 0.6301
C_EFF = 5.03
P_OCC_EFF = (8.0 / 9.0) * 0.304
T_STAR_MEASURED = 0.300     # mediane t* gen 0 (pin de Q, mesure)


def t_c_of(c_switch):
    t = 0.05
    for _ in range(200):
        x = max(1e-9, 1 - t / T_TRANS) ** EXPO
        t = 0.5 * t + 0.5 * C_EFF * D0_SIG * math.sqrt(x) / c_switch
    return t


def run():
    print("=== c_switch depuis la quantification : horloge microphysique ===\n")

    print("[1] STRUCTURE : pas de log cosmologique.")
    print("   Le gel Gamma=H donnerait E/T ~ 45 -> T* ~ 4 MeV, Omega ~ 0 :")
    print("   contredit par la chaine entiere. La fenetre de formation se")
    print("   ferme sur l'horloge de CASCADE (les tentatives de croisement")
    print("   et l'effondrement partagent la meme dynamique microphysique)")
    print("   -> c_switch = ln(rapport d'horloges microphysiques) = O(1).\n")

    print("[2] L'HORLOGE EST MESUREE : la cascade donne un contact par")
    print("   t* = %.3f periode (mediane gen 0, pin de Q) ->" % T_STAR_MEASURED)
    c_meas = math.log(1 / T_STAR_MEASURED)
    print("   c_switch = ln(1/t*) = %.2f\n" % c_meas)

    print("[3] CANDIDATS ET PROPAGATION (chaine canonique) :")
    rows = [
        ("par tentative (partage strict)", 1.0, "retenu (borne)"),
        ("par periode, t* MESURE", c_meas, "retenu (mesure)"),
        ("sites ln(c=23)", math.log(23.0), "EXCLU (meme cadence)"),
        ("cœur ln(xi/D)", 4.7, "EXCLU (horloge fragment mesuree)"),
    ]
    for lab, cs, statut in rows:
        t = t_c_of(cs)
        b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-32s c_sw = %.2f -> T_c = %5.1f MeV,"
              " Omega = %6.2f  [%s]" % (lab, cs, t * 1e3, b["omega"], statut))

    print("\n[4] LA COMPENSATION DE t* (raffinement decisif) :")
    print("   la comparaison correcte est PAR VIE DE FRAGMENT (la fenetre")
    print("   dont chaque fragment dispose). Or, mesure/structurel : le")
    print("   premier contact EST le declencheur de la scission -> nombre")
    print("   de tentatives par vie = (1/t*) x t* = 1 EXACTEMENT (le t*")
    print("   se compense). Nombre attendu de changements topologiques")
    print("   par fragment : N_topo = e^{-E_char/T}. La fenetre se ferme")
    print("   au seuil e-fold N_topo = 1/e :")
    print("      c_switch = 1   (DERIVE ; le seuil exact est la seule")
    print("      convention residuelle)\n")

    print("[5] BANDE DE CONVENTION DE SEUIL (N_topo = 1/2, 1/e, 1/3) :")
    for lab, cs in (("seuil 1/2 (ln 2)", math.log(2.0)),
                    ("seuil 1/e (= 1)", 1.0),
                    ("seuil 1/3 (ln 3)", math.log(3.0))):
        t = t_c_of(cs)
        b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-18s c_sw = %.2f -> T_c = %5.1f MeV, Omega = %6.2f" % (
            lab, cs, t * 1e3, b["omega"]))

    t_1 = t_c_of(1.0)
    b_1 = budget(tc=t_1, g_s=15.2, p_occ=P_OCC_EFF)["omega"]
    print("\nBILAN : c_switch = 1 DERIVE (fenetre par fragment, t*")
    print("compense) -> T_c = %.1f MeV, Omega = %.2f ; les lectures a" % (
        t_1 * 1e3, b_1))
    print("grands logs (3,1-4,7 : sites, cœur) et le log cosmologique 45")
    print("sont EXCLUS par la structure ; le residu = la convention de")
    print("seuil (ln 2 a ln 3 : T_c 61-74, Omega 0,7-9) -- le meme O(1)")
    print("fin que partout ailleurs dans la chaine.")


if __name__ == "__main__":
    run()
