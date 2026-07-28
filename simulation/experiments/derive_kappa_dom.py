"""kappa_dom : le mode critique du condensat chi -- scan des candidats derivables.

CIBLE (brouillon §7) : kappa_dom = D_dom * T_c in [0.73 ; 1.01] est LE O(1)
qui decide de la branche (0.73-0.76 : corner falsifiable Migdal ;
~1 : reservoir invisible 4.5 GeV).

CANDIDATS :
 A. Longueurs thermiques generiques : 1/T (kappa=1) ; longueur d'onde
    thermique de de Broglie d'un boson sans masse lambda = pi^{2/3}/T ?
    conventions multiples -- generiques, non discriminantes.
 B. Espacement inter-quanta a T_c : d = (pi^2/(g zeta(3)))^{1/3} / T
    (g dof du champ chi : g=3 pour SU(2) adjoint ? g=1 scalaire ?).
 C. ISING-3D + CRITERE DE VERROUILLAGE (TQNT-natif : la condensation chi
    est le deconfinement SU(2) = classe Ising-3D, programme gluon-Ising) :
    - tension emergente sous T_c : sigma(T) = sigma_0 (1-T/T_c)^{2 nu}
      (exposant d'interface 3D-Ising, 2nu = 1.26) ;
    - VERROUILLAGE des noeuds quand la reconnexion thermique cesse :
      sigma(T_lock) * l_th > T_lock, l_th ~ 1/T => sigma(T_lock) = T_lock^2 ;
    - domaine = longueur de correlation Ising au verrouillage :
      xi(T_lock) = xi_0 * t_lock^{-nu}, nu = 0.63.
    => kappa_dom = (xi_0 T_c) * t_lock^{-0.63} : le O(1) devient xi_0 T_c,
       l'AMPLITUDE critique -- mesurable sur reseau SU(2) (litterature).
 D. Boucles browniennes du reseau de vortex (Vachaspati-Vilenkin) :
    remarque de coherence -- une boucle brownienne de pas a remplissant le
    domaine porte un ACN >> c_max geometrique : le noeud MINIMAL apres
    relaxation est bien borne par la GEOMETRIE (le fit du §7), pas par le
    knotting disponible. (Confirme l'empaquetage compact comme critere.)
"""
import math

T_C = 0.067
SQSIG0 = 0.41765            # GeV (sqrt sigma_chi a T=0)
HBARC = 0.19733             # GeV.fm
A_FM = 0.209
NU_ISING = 0.6301
L31, SLOPE = 16.372, (38.62 - 16.372) / 6
ZETA3 = 1.2020569


def c_max_of_kappa(kappa):
    D = kappa * HBARC / T_C
    lf = (D / (2 * A_FM)) ** 3 * 4 / 3
    return 3 + (lf / 2 - L31) / SLOPE


def branche(cm):
    if cm < 21:
        return "sous c_min=23 : PAS de reservoir (!)"
    if cm <= 27:
        return "corner FALSIFIABLE (Migdal)"
    if cm <= 35:
        return "intermediaire"
    return "fuite invisible"


def run():
    print("=== kappa_dom : scan des candidats ===\n")

    print("[A] Longueurs thermiques generiques :")
    for nom, kap in (("1/T_c", 1.0), ("1/(2 T_c)", 0.5), ("pi^{2/3}/T (deBroglie boson)", math.pi ** (2 / 3))):
        cm = c_max_of_kappa(kap)
        print("   %-30s kappa=%5.2f -> c_max=%6.1f : %s" % (nom, kap, cm, branche(cm)))
    print()

    print("[B] Espacement inter-quanta d = (pi^2/(g zeta3))^{1/3}/T :")
    for g in (1, 3):
        kap = (math.pi ** 2 / (g * ZETA3)) ** (1 / 3)
        cm = c_max_of_kappa(kap)
        print("   g=%d : kappa=%5.2f -> c_max=%6.1f : %s" % (g, kap, cm, branche(cm)))
    print()

    print("[C] Ising-3D + verrouillage sigma(T_lock) = T_lock^2 :")
    sigma0 = SQSIG0 ** 2
    # resoudre sigma0 (1-x)^{1.26} = (x T_c)^2, x = T/T_c
    lo, hi = 0.5, 0.999999
    for _ in range(80):
        x = 0.5 * (lo + hi)
        lhs = sigma0 * (1 - x) ** (2 * NU_ISING)
        rhs = (x * T_C) ** 2
        if lhs > rhs:
            lo = x
        else:
            hi = x
    x = 0.5 * (lo + hi)
    t_lock = 1 - x
    amp = t_lock ** (-NU_ISING)
    print("   t_lock = 1 - T_lock/T_c = %.4f ; xi/xi_0 = t^{-nu} = %.2f" % (t_lock, amp))
    print("   => kappa_dom = %.2f x (xi_0 T_c)" % amp)
    print("   Le O(1) restant est l'AMPLITUDE critique xi_0 T_c (reseau SU(2)) :")
    for x0 in (0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        kap = amp * x0
        cm = c_max_of_kappa(kap)
        print("     xi_0 T_c = %.2f : kappa = %.2f -> c_max = %5.1f : %s" % (
            x0, kap, cm, branche(cm)))
    # inversion : quel xi_0 T_c pour les deux branches ?
    for nom, kap_cible in (("corner falsifiable (kappa=0.74)", 0.74),
                           ("gel invisible (kappa=1.01)", 1.01)):
        print("     -> %s exige xi_0 T_c = %.3f" % (nom, kap_cible / amp))
    print()

    print("[D] Remarque boucles browniennes : une boucle de pas a remplissant")
    print("    D_dom porte n ~ (D/a)^2 ~ %.0f segments (kappa=0.74) -> ACN ~" % (
        (0.74 * HBARC / T_C / A_FM) ** 2))
    print("    n ln n >> c_max : le knotting disponible n'est PAS le goulot ;")
    print("    la borne est bien geometrique (empaquetage compact, §7). OK.\n")

    print("VERDICT :")
    print(" - Les candidats generiques (A, B) tombent DES DEUX COTES")
    print("   (kappa 0.5-2.1) : non discriminants seuls.")
    print(" - La voie TQNT-native (C, Ising-3D + verrouillage) REDUIT le")
    print("   probleme : kappa_dom = %.2f x (xi_0 T_c) -- l'amplitude critique" % amp)
    print("   du deconfinement SU(2), un nombre de RESEAU :")
    print("     xi_0 T_c ~ 0.12 -> corner falsifiable ; ~ 0.16 -> invisible.")
    print(" - PROCHAINE MARCHE (externe et checkable) : la valeur lattice de")
    print("   xi_0 T_c pour SU(2) pur au deconfinement (litterature lattice ;")
    print("   de memoire l'ordre est 0.1-0.3 -- A SOURCER). Le programme")
    print("   gluon-Ising rejoint la cosmologie par un 2e point de contact.")


if __name__ == "__main__":
    run()
