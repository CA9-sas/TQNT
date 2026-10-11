"""Les phases du Big Bang du cadre, la densité d'espace des phases à la NAISSANCE
des éremnions, et ce que chaque phase peut produire comme objet sombre compact."""
from __future__ import annotations
import math

G, c, kB = 6.6710e-11, 2.998e8, 1.381e-23
Msun, AU, pc, yr = 1.989e30, 1.496e11, 3.086e16, 3.156e7
GeV, fm = 1.7827e-27, 1e-15
m_er = 1.8 * GeV
SQRT_SIGMA = 0.41847                      # GeV
SIGMA_CHI = SQRT_SIGMA ** 2               # GeV^2
XI = 2.98 / SIGMA_CHI * 0.197327 * fm     # quantum sigma_chi xi = 2,98 GeV -> xi
T_C = 62.0                                # MeV
Z_EQ = 3400.0
Q_PEARL = 2.27e-6                         # 2e-3 Msun à 21 km
Q_F = 1.1e-22                             # 3e5 Msun à 20 UA


def z_of_T(T_MeV):
    return T_MeV * 1e6 / 2.35e-4


def t_of_T(T_MeV, gstar):
    return 2.4 / math.sqrt(gstar) * (1 / T_MeV) ** 2


def n_er_moyen(T_MeV):
    return 0.265 * 9.47e-27 * (1 + z_of_T(T_MeV)) ** 3 / m_er


def Q(n, v):
    return n * m_er / v ** 3


if __name__ == "__main__":
    print("=== 1. CE QUE LIOUVILLE CONSERVE, ET DEPUIS QUAND ===")
    print("  Liouville : la densité d'espace des phases À GRAIN FIN est conservée par toute")
    print("  évolution hamiltonienne sans collision. Elle n'est PAS conservée à travers une")
    print("  transition qui CRÉE les particules : la valeur maximale est fixée À LA NAISSANCE,")
    print("  par le mécanisme de formation. Ma borne Q_gel supposait densité MOYENNE et vitesse")
    print("  THERMIQUE. Les éremnions naissent le long des cordes, en filaments, avec la")
    print("  vitesse de la cascade. Il faut refaire le compte.\n")

    n_moy = n_er_moyen(T_C)
    v_th = math.sqrt(3 * T_C * 1e6 * 1.602e-19 / m_er)
    print(f"  xi au gel (sigma_chi xi = 2,98 GeV)      : {XI/fm:.2f} fm")
    print(f"  densité moyenne d'éremnions au gel        : {n_moy:.2e} /m3")
    print(f"  vitesse thermique sqrt(3T_c/m)            : {v_th/c:.2f} c")
    print(f"  vitesse des filles de la cascade (cadre)  : 0,03 c")
    print(f"  Q_gel (moyenne, thermique) = {Q(n_moy, v_th):.2e}   <- la borne que j'avais utilisée\n")

    print("  Q à la naissance selon (surdensité de filament, vitesse de naissance) :")
    print(f"  {'n_b/n_moy':>10s} {'v_b':>7s} {'Q_naissance':>12s} {'Q_perle/Q':>10s} {'perle ?':>8s} {'Q_F/Q':>9s}")
    for f_n in (1, 1e3, 1e5, 3.5e7, 1e9):
        for v_b in (0.32, 0.03, 0.003):
            q = Q(f_n * n_moy, v_b * c)
            print(f"  {f_n:10.0e} {v_b:6.3f}c {q:12.2e} {Q_PEARL/q:10.1e} {'PERMISE' if q > Q_PEARL else 'interdite':>9s} {Q_F/q:9.1e}")
    print(f"\n  -> la perle (Q = {Q_PEARL:.1e}) est PERMISE dès que n_b/n_moy x (v_b/c)^-3 >= "
          f"{Q_PEARL/Q(n_moy, c):.1e} ;")
    print("     à v_b = 0,03 c il faut des filaments 3,5e7 fois plus denses que la moyenne.")
    print("     Le défaut F (Q = 1e-22) est permis dans TOUS les cas — ce n'est pas son problème.")
    print("     La borne « perle interdite par 4,4e10 » du 16/09 était une SURESTIMATION :")
    print("     elle vaut pour des nœuds nés thermiques et homogènes, ce qu'ils ne sont pas.\n")

    print("=== 2. LES PHASES : masse causale de matière noire et dissipation sombre ===")
    print(f"  {'phase':34s} {'T':>8s} {'t':>10s} {'M_hor':>10s} {'M_DM causale':>13s}  dissipation sombre ?")
    phases = [
        ("croisement QCD (sigma_Cornell = sigma_chi)", 156.0, 61.75, "plasma, pas encore d'éremnions"),
        ("GEL chi, naissance des nœuds, marche de G", 62.0, 60.0, "cordes rayonnent, chaleur latente — mais c>=23 NÉ DÉCOUPLÉ (Gamma/H = 0,067)"),
        ("découplage neutrinos", 1.0, 10.75, "aucune"),
        ("annihilation e+e-", 0.5, 10.75, "aucune"),
        ("BBN", 0.1, 3.36, "aucune"),
        ("M_DM causale = 3e5 Msun", 7e-3, 3.36, "AUCUNE TRANSITION CONNUE DU CADRE"),
        ("T = 1 keV", 1e-3, 3.36, "aucune"),
        ("égalité", 8e-7, 3.36, "aucune"),
    ]
    for lab, T, g, dis in phases:
        t = t_of_T(T, g)
        Mh = c ** 3 * t / G
        Mdm = Mh * (1 + Z_EQ) / (1 + z_of_T(T))
        ts = f"{t:.1e} s" if t < 3.156e7 else f"{t/yr:.1e} an"
        print(f"  {lab:34s} {T:8.1e} {ts:>10s} {Mh/Msun:10.1e} {Mdm/Msun:13.1e}  {dis}")
    print()
    print("  Lecture : la masse de matière noire dans un horizon monte comme T^-3 ; elle vaut")
    print("  4e-7 Msun au gel et n'atteint 3e5 Msun qu'à T ~ 7 keV (t ~ 3 h). Et la seule phase")
    print("  où le secteur sombre dissipe est le gel — mais le cadre DÉRIVE c_min = 23 en exigeant")
    print("  que les éremnions soient découplés À LEUR NAISSANCE (les nœuds couplés ne sont pas")
    print("  de la matière noire). La matière noire n'a JAMAIS eu de fenêtre dissipative.\n")

    print("=== 3. CE QU'UNE NAISSANCE FROIDE PERMETTRAIT : la perle primordiale du gel ===")
    Mdm_gel = (c ** 3 * t_of_T(T_C, 60) / G) * (1 + Z_EQ) / (1 + z_of_T(T_C))
    R_hor = c * t_of_T(T_C, 60)
    v_esc = math.sqrt(2 * G * Mdm_gel / R_hor)
    print(f"  M_DM causale au gel = {Mdm_gel/Msun:.1e} Msun ; v_échappement de ce nuage = {v_esc/c:.1e} c")
    print(f"  -> un amas se LIE à la marche de G si v_naissance < {v_esc/c:.1e} c.")
    print(f"     Thermique (0,32 c) : non. Cascade (0,03 c) : non, facteur {0.03/(v_esc/c):.0e}.")
    print(f"     Il faudrait des nœuds nés à moins de {v_esc:.0f} m/s dans le repère du plasma.")
    print("  -> des perles LÉGÈRES (<~ 1e-7 Msun) peuvent être primordiales SI la cascade laisse")
    print("     les nœuds quasi immobiles. C'est une question au spectre de vitesse de la cascade")
    print("     (verif_chaine_omega.py), pas à Liouville. Et c'est la fenêtre de microlentille")
    print("     OUVERTE du cadre (M <~ 5e-11 Msun, f ~ 1 permis).\n")

    print("=== 4. ET LE DÉFAUT F ===")
    print("  Il lui faut 3e5 Msun de matière noire causale : T <~ 7 keV. Aucune transition du cadre")
    print("  n'y siège. Sans transition, pas de formation primordiale ; sans dissipation, pas")
    print("  d'assemblage tardif. => la quasi-étoile à cœur sombre EXIGE une transition du secteur")
    print("  sombre à l'échelle du keV. C'est la demande théorique que la question fait remonter.")
