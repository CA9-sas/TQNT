"""Sourcer sigma_eff(T) sur reseau -- et confronter la chaine T_c (20/07).

SOURCE (quenched SU(3), Cardoso-Bicudo, PRD 85 077501 / 1111.1317) :
sigma(T) extraite de l'energie libre coloree-moyennee d'une paire
statique ; RESULTAT STRUCTUREL : sigma(T) est PLATE (~sigma_0)
jusqu'a T ~ 0,5 T_c, puis chute pres de T_c avec le fit
    sigma(T)/sigma_0 = 1,21 (1 - 0,990 (T/T_c)^2)^(1/2)
(forme magnetisation, exposant 1/2), operationalise avec le plafond
sigma <= sigma_0 (le fit vaut pres de T_c ; plat en-deca).

CONSEQUENCE : l'extrapolation d'Ising (1 - T/T_trans)^1,26 utilisee
depuis l'etape 9 est REFUTEE a bas T (elle fait chuter sigma des le
depart ; le reseau la garde pleine jusqu'a 0,55 T_c).

CONFRONTATION : serrage E_char = T sous chaque forme -> T_c, Omega.
"""
import math
import sys

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
D0_SIG = 0.1084 * SIGMA
T_TRANS = 0.1565
C_EFF = 4.6   # c_eff(q zone-marche ~0,81) -- statistique derivee etape 14
P_OCC_EFF = (8.0 / 9.0) * 0.304


def x_ising(t):
    return max(1e-9, 1 - t / T_TRANS) ** (2 * 0.6301)


def x_mf(t):
    return max(1e-9, 1 - t / T_TRANS)


def x_lattice(t):
    u2 = (t / T_TRANS) ** 2
    if u2 >= 1 / 0.990:
        return 1e-9
    return min(1.0, 1.21 * math.sqrt(1 - 0.990 * u2))


def t_serrage(x_of):
    t = 0.06
    for _ in range(300):
        t = 0.5 * t + 0.5 * C_EFF * D0_SIG * math.sqrt(x_of(t))
        t = min(t, 0.95 * T_TRANS)
    return t


def run():
    print("=== sigma_eff(T) source : confrontation des formes ===\n")
    print("[1] LA FORME SOURCEE (Cardoso-Bicudo, quenched SU(3)) :")
    print("   sigma/sigma_0 = min[1 ; 1,21 (1-0,990(T/Tc)^2)^(1/2)]")
    print("   -> PLATE jusqu'a ~0,55 T_c, chute pres de T_c seulement.")
    print("   T/Tc :   0,3    0,4    0,5    0,6    0,7    0,8")
    for name, fx in (("reseau", x_lattice), ("Ising (ex-chaine)", x_ising),
                     ("champ moyen", x_mf)):
        vals = "  ".join("%.2f" % fx(u * T_TRANS)
                         for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
        print("   %-18s %s" % (name, vals))
    print("   -> l'extrapolation d'Ising a bas T est REFUTEE par le")
    print("      reseau (elle sous-estime sigma d'un facteur ~2 a 0,4 Tc).\n")

    print("[2] SERRAGE SOUS CHAQUE FORME (c_eff = %.1f, stat. derivee) :" % (
        C_EFF))
    for name, fx in (("reseau (source, quenched)", x_lattice),
                     ("Ising (ex-chaine)", x_ising),
                     ("champ moyen", x_mf)):
        t = t_serrage(fx)
        b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-26s T_c = %5.1f MeV -> Omega = %8.2f (%+.0f %%)" % (
            name, t * 1e3, b["omega"], 100 * (b["omega"] / 5.45 - 1)))

    print("\nVERDICT (honnete) :")
    print(" - la forme SOURCEE (quenched) donne T_c ~ 90-95 MeV et")
    print("   Omega ~ 1e3 : SURPRODUCTION x200 -- la chaine ne ferme pas")
    print("   sous la courbe quenched ;")
    print(" - MAIS la courbe quenched est SANS ECRANTAGE (jauge pure,")
    print("   pas de brisure de corde) ; la transition canonique est le")
    print("   croisement PHYSIQUE (2+1 saveurs, canaux legers) ou")
    print("   sigma_eff est adoucie par l'ecrantage -- les donnees")
    print("   pleines-QCD a T/Tc ~ 0,4-0,6 ne sont PAS disponibles dans")
    print("   ce sourcing : LE PIN RESTE OUVERT ;")
    print(" - de plus l'echelle est ambigue : E_char vit a l'echelle du")
    print("   CŒUR (D ~ 0,1-0,15 fm), ou la << tension >> pertinente est")
    print("   l'amplitude du condensat (parametre d'ordre du croisement),")
    print("   pas la tension asymptotique -- question de matching declaree.")
    print(" - ETAT : T_c in [63,5 (Ising), ~94 (quenched)] selon la forme ;")
    print("   Omega in [1, 1e3]. Le pin exige : sigma_eff(T) pleine-QCD a")
    print("   0,4-0,6 Tc (sourcing a poursuivre), et le matching d'echelle")
    print("   cœur/asymptotique (chantier theorique).")


if __name__ == "__main__":
    run()
