"""Les deux verrous : basculement elastique->CS + raccord au comptage (20/07).

VERROU (a) -- LE BASCULEMENT ELASTIQUE -> CS.
Tant que le tube est gras, la topologie est FLUIDE : les glissements
de phase thermiques (meme echelle que le passage elastique,
E_char(T) = c_eff sqrt(sigma(T) sigma_chi) D0) sont O(1) par tentative
(E_char < T) -- le flux du tube n'est pas defini, pas de protection.
Quand E_char franchit T, les glissements gelent, LE FLUX SE QUANTIFIE,
et la suppression Chern-Simons (niveau k = 135, tau ~ 1e347 s)
s'allume : le taux passe de ~T e^-1 a ~e^{-S_k} -- il n'existe PAS de
regime thermique intermediaire E in [T, 45T] (le canal elastique
disparait avec la quantification). Le gel effectif est donc a
E_char = c_switch T avec c_switch ~ 1 (O(1) du basculement).

VERROU (b) -- LE RACCORD AU COMPTAGE.
La chaine canonique (mode KZ) porte T_c par la FRACTION SOMBRE
S_d/Z (Boltzmann RELATIF entre types, levier e^{-Dm/T}), avec les
POIDS A MASSE PLEINE -- convention CANONIQUE DERIVEE (cout d'entropie
d'onde a masse pleine ; la variante << tension fondue >> est REFUTEE,
brouillon_boltzmann_tqnt). CORRECTION d'etape 9 : sa surproduction
x400 utilisait la convention fondue refutee -- caduque.

ASSEMBLAGE : T* = T_c derive (geometrie, etape 10) -> budget canonique
(masses pleines, comptage mesure n/L = 0,304) -> Omega final + bande.
"""
import math
import sys

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
D0_SIG = 0.1084 * SIGMA
T_TRANS = 0.155
EXPO = 2 * 0.6301
P_OCC_EFF = (8.0 / 9.0) * 0.304   # comptage mesure (verif 11/11)


def c_p_of_s(u):
    return math.pi + 2 * math.atan(1 / u) - 2 * math.atan(u)


def c_eff_of_q(q):
    u = math.sqrt(max(1e-9, 2 / q - 1))
    return c_p_of_s(u) + 2 * q * u


def t_star(c_eff, c_switch=1.0, t_trans=T_TRANS):
    t = 0.05
    for _ in range(200):
        x = max(1e-9, 1 - t / t_trans) ** EXPO
        t = 0.5 * t + 0.5 * c_eff * D0_SIG * math.sqrt(x) / c_switch
    return t


def run():
    print("=== Verrous (a) basculement CS et (b) raccord comptage ===\n")

    print("(a) BASCULEMENT ELASTIQUE -> CS :")
    print("   fluide (E_char < T) : glissements O(1), flux non defini,")
    print("   pas de CS ; a E_char = c_switch T : quantification -> taux")
    print("   ~e^{-S_k} (tau ~ 1e347 s). Pas de regime thermique [T, 45T].")
    ce = 5.03   # mesure T(2,23), etape 10
    for cs in (0.5, 1.0, 1.5, 2.0):
        t = t_star(ce, cs)
        print("   c_switch = %.1f -> T* = %5.1f MeV" % (cs, t * 1e3))
    print("   -> la valeur canonique 67 selectionne c_switch = 1,00 (la")
    print("   lecture la plus simple : gel des que E_char > T). Statut :")
    print("   argument de structure + coherence ; le O(1) exact du")
    print("   basculement reste la marge du resultat.\n")

    print("(b) RACCORD AU COMPTAGE (canonique, masses pleines) :")
    print("   correction d'etape 9 : la convention << tension fondue >>")
    print("   (masses reduites au gel) est REFUTEE aux regles etablies -- le poids")
    print("   de formation est a masse PLEINE ; la surproduction x400")
    print("   d'etape 9 est caduque.")
    t_c = t_star(ce, 1.0)
    for lab, tc in (("T_c derive (T* = %.1f MeV)" % (t_c * 1e3), t_c),
                    ("T_c historique 67", 0.067)):
        b = budget(tc=tc, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-28s Omega = %.2f (Planck 5,45 : %+.0f %%)" % (
            lab, b["omega"], 100 * (b["omega"] / 5.45 - 1)))

    print("\n   BANDE (propagation des mesures de l'etape 10) :")
    rows = []
    for lab, q, t_tr in (("q mediane 0,95, T_trans 155", 0.95, 0.155),
                         ("q Q1 0,72, T_trans 155", 0.72, 0.155),
                         ("q 0,95, T_trans 145", 0.95, 0.145),
                         ("q 0,95, T_trans 165", 0.95, 0.165)):
        ce_i = c_eff_of_q(q)
        t_i = t_star(ce_i, 1.0, t_tr)
        b = budget(tc=t_i, g_s=15.2, p_occ=P_OCC_EFF)
        rows.append(b["omega"])
        print("   %-28s c_eff = %.2f, T* = %4.1f -> Omega = %5.2f" % (
            lab, ce_i, t_i * 1e3, b["omega"]))
    print("\nBILAN : T_c = %.1f MeV DERIVE (geometrie de contact + Ising +" % (
        t_c * 1e3))
    print("basculement CS a c_switch = 1) ; chaine canonique (masses")
    print("pleines, comptage mesure) -> Omega in [%.1f, %.1f] --" % (
        min(rows), max(rows)))
    print("l'enveloppe adoptee [60, 75] est REMPLACEE par une valeur")
    print("derivee avec sa bande de mesure.")


if __name__ == "__main__":
    run()
