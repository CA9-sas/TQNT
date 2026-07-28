"""Completion de la voie vrai-N(c) : deriver c_max (la coupure de condensation).

CONTEXTE (brouillon §6) : sous le vrai N(c), Omega=5.45 n'est atteignable
qu'en regime divergent + coupure ; le lieu (alpha, c_max) est cartographie
(1.25->23 ... 1.70->63). En regime divergent, PAS de mur de rarete (les
gros noeuds sont plus abondants) : la coupure doit etre GEOMETRIQUE/CAUSALE.

TROIS MECANISMES TESTES :
 1. TEMPS D'ASSEMBLAGE : c operations sequentielles de verrouillage de
    croisement pendant la fenetre de condensation Delta_t ~ 1/(2H(T_c)).
    Taux par pas : tau_step ~ 1/T_c (thermique) ou a/c_lum (dynamique tube).
 2. KIBBLE-ZUREK cosmologique : xi_KZ = xi_0 (tau_Q/tau_0)^{1/4}
    (champ moyen nu=1/2, z=2), tau_Q = 1/H.
 3. DOMAINE GEOMETRIQUE : le noeud (rayon de tube a) doit tenir dans un
    domaine chi coherent de diametre D_dom = kappa/T_c a la formation.
    Empaquetage : boule compacte D(c) = 2a (3 l_full(c)/4)^{1/3}
    (noeud serre = volume de corde) ; variante anneau plat D = l a/pi.

Puis : INTERSECTION avec le lieu Omega=5.45 -> a quel kappa_dom correspond
chaque point du lieu ? Le O(1) geometrique devient LE cadran restant.
"""
import math

T_C = 0.067                    # GeV
G_STAR = 10.75
M_PL = 1.221e19
HBARC_FM = 0.19733             # GeV.fm
A_TUBE_FM = 0.209              # fm
L31 = 16.372
SLOPE = (38.62 - 16.372) / 6
ALPHA_LOCUS = [(1.25, 23), (1.30, 24), (1.40, 26), (1.50, 31),
               (1.60, 40), (1.70, 63)]


def l_full(c):
    return 2 * (L31 + SLOPE * (c - 3))


def D_compact_fm(c):
    return 2 * A_TUBE_FM * (3 * l_full(c) / 4) ** (1 / 3)


def D_ring_fm(c):
    return l_full(c) * A_TUBE_FM / math.pi


def run():
    H = 1.66 * math.sqrt(G_STAR) * T_C ** 2 / M_PL
    dt = 1 / (2 * H)                       # GeV^-1
    print("=== Deriver c_max : trois mecanismes ===\n")
    print("H(T_c) = %.2e GeV ; fenetre Delta_t = %.2e GeV^-1\n" % (H, dt))

    print("[1] Temps d'assemblage :")
    n_th = T_C * dt
    tau_tube = (A_TUBE_FM / HBARC_FM)      # GeV^-1 (a / c)
    n_dyn = dt / tau_tube
    print("   pas thermiques disponibles : T_c*Delta_t = %.1e ;" % n_th)
    print("   pas dynamiques (a/c)       : %.1e" % n_dyn)
    print("   -> AUCUN MUR (c_max ~ 1e19-1e20 >> 60) : l'assemblage n'est")
    print("   jamais limite par le temps. Mecanisme ECARTE.\n")

    print("[2] Kibble-Zurek cosmologique :")
    xi0 = 1 / T_C                          # GeV^-1
    xi_kz = xi0 * (dt * T_C) ** 0.25       # (tau_Q/tau_0)^{1/4}
    xi_kz_fm = xi_kz * HBARC_FM
    print("   xi_KZ = (1/T_c)(T_c/2H)^{1/4} = %.0f fm" % xi_kz_fm)
    c_kz = 3 + ((( (xi_kz_fm / (2 * A_TUBE_FM)) ** 3) * 4 / 3) / 2 - L31) / SLOPE
    print("   -> boule compacte : c_max ~ %.0e >> 60 : PAS DE MUR utile." % c_kz)
    print("   Mecanisme ECARTE (le gel KZ est bien trop tardif/grand).\n")

    print("[3] Domaine geometrique D_dom = kappa / T_c :")
    print("   (1/T_c = %.2f fm ; 1/Lambda_chi = %.2f fm ; a = %.3f fm)" % (
        HBARC_FM / T_C, HBARC_FM / 0.2088, A_TUBE_FM))
    # table c_max(kappa) pour les deux empaquetages
    print("   kappa   D_dom(fm)   c_max(boule)   c_max(anneau)")
    for kappa in (0.5, 0.74, 1.0, 1.5, 2.0, 3.0):
        D = kappa * HBARC_FM / T_C
        # boule : resoudre D_compact(c) = D
        lf = (D / (2 * A_TUBE_FM)) ** 3 * 4 / 3
        c_b = 3 + (lf / 2 - L31) / SLOPE
        # anneau : l a / pi = D
        lf_r = D * math.pi / A_TUBE_FM
        c_r = 3 + (lf_r / 2 - L31) / SLOPE
        print("   %4.2f     %5.2f        %6.1f          %6.1f" % (kappa, D, c_b, max(c_r, 0)))

    print("\n[4] Intersection avec le lieu Omega = 5.45 (empaquetage boule) :")
    print("   alpha   c_max(lieu)   kappa_dom requis   statut")
    for a, cm in ALPHA_LOCUS:
        D = D_compact_fm(cm)
        kappa = D * T_C / HBARC_FM
        statut = "corner FALSIFIABLE (Migdal)" if cm <= 26 else (
            "intermediaire" if cm <= 35 else "fuite invisible")
        print("   %.2f      %3d          %5.2f            %s" % (a, cm, kappa, statut))

    print("\nVERDICT :")
    print(" - Assemblage et Kibble-Zurek : PAS de mur (ecartes proprement).")
    print(" - Le mur GEOMETRIQUE marche EN ORDRE : un domaine coherent de")
    print("   l'ordre de la longueur thermique 1/T_c donne c_max ~ 20-60 --")
    print("   exactement la decade du lieu Omega=5.45. La derivation d'ORDRE")
    print("   est acquise ; le O(1) kappa_dom = D_dom*T_c reste le cadran :")
    print("   kappa ~ 0.7-0.8  -> c_max 23-26 -> corner falsifiable (Migdal)")
    print("   kappa ~ 1.0      -> c_max ~ 60  -> reservoir invisible (4.5 GeV)")
    print(" - La completion se reduit a UNE question : la taille du domaine")
    print("   de nucleation coherent du condensat chi a T_c (kappa_dom) --")
    print("   candidat naturel : le rayon thermique de correlation du mode")
    print("   critique (kappa=1 au gel exact ; <1 si le gel precede T_c).")
    print(" - NB : tout ceci reste conditionne aux comptages Burton (memoire)")
    print("   et au r_inf ; la structure (lieu + mur geometrique) est robuste.")


if __name__ == "__main__":
    run()
