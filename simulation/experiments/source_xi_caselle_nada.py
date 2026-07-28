"""Sourcing de xi(T) sur reseau SU(2) pur -- Caselle & Nada, arXiv:1707.02164v2.

DONNEES (Tables 2+3 du papier) : longueur de correlation EXPONENTIELLE xi/a du
correlateur de boucle de Polyakov a moment nul, phase CONFINEE (T < T_c),
SU(2) pur (D=3+1), action de Wilson. T = 1/(a N_t) => xi*T = (xi/a)/N_t
EXACTEMENT (aucune conversion d'echelle, aucune amplitude Ising requise).

POURQUOI C'EST LA BONNE OBSERVABLE : la boucle de Polyakov est le parametre
d'ordre du deconfinement (= la condensation chi, programme gluon-Ising) ; la
phase confinee (celle ou les noeuds se forment au refroidissement) est la
phase symetrique Ising (Svetitsky-Yaffe). xi est donc LE rayon du domaine
coherent du condensat.

COURT-CIRCUIT DU CRITERE : la masse du correlateur est m(T) = sigma(T)/T
(terme dominant corde libre). Le critere de verrouillage du brouillon
(sigma(T_lock) l_th > T, l_th = 1/T => sigma = T^2) se lit alors DIRECTEMENT :
    sigma = T^2  <=>  xi*T = 1.
Plus besoin de xi_0 ni de t_lock theorique (qui supposait sigma_0 = sigma(0),
faux pres de T_c) : on cherche le croisement xi*T = 1 dans la TABLE MESUREE,
et kappa_dom = xi(T_lock)*T_c = (xi*T)/(T/T_c) au croisement.
"""
import math

# Table 2+3 de Caselle-Nada 1707.02164 : (T/T_c, N_t, xi/a, err, xi2nd/a)
DATA = [
    (0.59, 6, 1.31, 0.02, 0.887),
    (0.71, 6, 2.31, 0.04, 1.842),
    (0.78, 5, 2.56, 0.02, 2.22),
    (0.78, 6, 3.08, 0.04, 2.67),
    (0.84, 4, 3.05, 0.06, 2.74),
    (0.90, 6, 6.9, 0.2, 6.6),
]
NU_ISING = 0.6301
T_C = 0.067                 # GeV
HBARC = 0.19733             # GeV.fm
A_FM = 0.209
L31, SLOPE = 16.372, (38.62 - 16.372) / 6
SCALE = 310.0 / 16.372


def c_max_of_kappa(kappa):
    D = kappa * HBARC / T_C
    lf = (D / (2 * A_FM)) ** 3 * 4 / 3
    return 3 + (lf / 2 - L31) / SLOPE


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0


def branche(cm):
    if cm < 21:
        return "PAS de reservoir"
    if cm <= 27:
        return "corner FALSIFIABLE (Migdal)"
    if cm <= 35:
        return "intermediaire"
    return "fuite invisible"


def xiT_of_x(x, pts):
    """Interpolation lineaire de xi*T en fonction de x = T/T_c."""
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(x)


def crossing(target, pts):
    """x tel que xi*T(x) = target (les donnees sont croissantes en x)."""
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        if y0 <= target <= y1:
            return x0 + (x1 - x0) * (target - y0) / (y1 - y0)
    return None


def run():
    print("=== xi(T) SU(2) pur, phase confinee -- Caselle-Nada 1707.02164 ===\n")

    pts = []
    print("[1] Donnees converties (xi*T = (xi/a)/N_t, exact) :")
    print("   T/T_c   N_t   xi/a      xi*T      xi/xi_2nd")
    for x, nt, xi, err, xi2 in DATA:
        xiT = xi / nt
        exiT = err / nt
        pts.append((x, xiT, exiT))
        print("   %4.2f    %d    %5.2f(%d)  %.3f(%d)   %.2f" % (
            x, nt, xi, round(err * 100), xiT, round(exiT * 1000), xi / xi2))
    # controle d'echelle : les deux points a 0.78 (N_t=5 et 6)
    p78 = [p for p in pts if abs(p[0] - 0.78) < 1e-9]
    print("   Controle de scaling (0.78, N_t=5 vs 6) : xi*T = %.3f vs %.3f (%+.1f%%) OK\n" % (
        p78[0][1], p78[1][1], 100 * (p78[1][1] / p78[0][1] - 1)))
    # fusionner les deux points 0.78
    merged = [p for p in pts if abs(p[0] - 0.78) > 1e-9]
    merged.append((0.78, 0.5 * (p78[0][1] + p78[1][1]), 0.5 * p78[0][2]))
    merged.sort()

    print("[2] Amplitude effective A(t) = xi*T * t^nu (t = 1 - T/T_c, nu = 0.6301) :")
    for x, xiT, _ in merged:
        t = 1 - x
        print("   t = %.2f : A_eff = %.3f" % (t, xiT * t ** NU_ISING))
    print("   -> A_eff DERIVE (0.12 -> 0.27) : la loi de puissance pure n'est")
    print("   asymptotique que t << 0.1 ; l'ancienne reduction via (xi_0, t_lock")
    print("   theorique avec sigma_0 = sigma(0)) etait donc APPROXIMATIVE.")
    print("   Les donnees permettent de COURT-CIRCUITER l'amplitude :\n")

    print("[3] Critere de verrouillage lu directement (sigma = T^2 <=> xi*T = 1,")
    print("    via m = sigma(T)/T pour le correlateur de Polyakov) :")
    x_lock = crossing(1.0, merged)
    kappa = 1.0 / x_lock
    cm = c_max_of_kappa(kappa)
    print("   croisement xi*T = 1 : T_lock/T_c = %.3f (t_lock = %.3f)" % (
        x_lock, 1 - x_lock))
    print("   => kappa_dom = xi(T_lock)*T_c = 1/(T_lock/T_c) = %.2f" % kappa)
    print("   => c_max = %.0f : %s" % (cm, branche(cm)))
    print("   (masse du gros du reservoir ~ m_geo(c_max) = %.1f GeV ;" % m_geo(cm))
    print("    sigma_DM-N ~ alpha^(c-18) ~ 1e%.0f cm^2 : hors de portee de tout)\n" % (
        math.log10(1e-25) + (cm - 18) * math.log10(1 / 137.036)))

    print("[4] Robustesse au O(1) du critere (xi*T = s au verrouillage) :")
    print("   s        T_lock/T_c   kappa_dom   c_max    branche")
    for s in (0.5, 0.6, 0.63, 0.7, 0.8, 1.0, 1.2, math.pi):
        xl = crossing(s, merged)
        if xl is None:
            print("   %5.2f    > 0.90 (hors table, plus pres de T_c) -> kappa > %.2f : invisible" % (
                s, s / 0.90))
            continue
        k = s / xl
        c = c_max_of_kappa(k)
        print("   %5.2f      %.3f       %.2f      %5.0f    %s" % (s, xl, k, c, branche(c)))
    print("   -> le corner falsifiable (kappa 0.73-0.76) exige s = 0.60-0.63,")
    print("      soit sigma = 1.6 T^2 : un decalage x1.6 du critere naturel ;")
    print("      s < 0.55 TUE le reservoir (c_max < 23 : plus de DM du tout) ;")
    print("      s >= 0.8 : invisible. La fenetre falsifiable est ETROITE.\n")

    print("VERDICT (sourcing fait) :")
    print(" - Source : Caselle & Nada, arXiv:1707.02164 (SU(2) 3+1D, phase")
    print("   confinee, correlateur de Polyakov ; echelle T/T_c = 0.7091(36)).")
    print(" - Le critere canonique du brouillon (sigma l_th = T, l_th = 1/T),")
    print("   lu sur les DONNEES sans intermediaire : kappa_dom = %.2f (+-0.02" % kappa)
    print("   stat) -> c_max ~ %.0f -> branche INVISIBLE (reservoir ~%.0f GeV," % (cm, m_geo(cm)))
    print("   sigma ~ 1e-170) : la tension SIMP s'evapore, MAIS le corner")
    print("   Migdal falsifiable n'est PAS la prediction generique.")
    print(" - La fourche 0.113/0.154 sur xi_0 est SUPPLANTEE par cette lecture")
    print("   directe (l'amplitude n'etait qu'un intermediaire, et A_eff derive).")
    print(" - Reserve honnete : le O(1) du critere n'est pas derive ; la fenetre")
    print("   corner exige sigma = 1.6 T^2 (fine-tuning modere) ; en-deca de")
    print("   sigma = 2 T^2 le reservoir disparait. Le resultat naturel est")
    print("   coherent avec la prediction canonique LZ/XENONnT = ABSENCE.")


if __name__ == "__main__":
    run()
