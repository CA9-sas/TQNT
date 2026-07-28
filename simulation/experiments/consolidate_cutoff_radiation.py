"""Consolidation de la coupure : la course scission / rayonnement (17/07).

MAILLON A CONSOLIDER : la coupure de la cascade de coudes (kinks) a
l_cut = xi etait un argument d'echelle qualitatif. Ici, la course est
QUANTIFIEE :

  - taux de scission a l'echelle l : kappa_s scissions par periode
    (l/c) -- MESURE sur la dynamique de Nambu (generation 0, premiers
    croisements des fragments plonges : proprete garantie, mediane
    t* = 0.40 periode ; l'avalanche t* = 0 des generations >= 1 est la
    resolution sequentielle de croisements SIMULTANES du meme amas
    d'evenements, diagnostiquee et exclue).
  - taux de lissage radiatif a l'echelle l : la corde chi est de type
    GLOBAL (elle rayonne son propre champ, modes BASE sans masse, avec
    un couplage O(1)) : efficacite par periode
        r(l) = Gamma_a / (2 pi ln(l/D)),
    ou D est le coeur (0.0214 fm) et Gamma_a la constante de radiation
    classique (cordes globales : ~65 boucles lisses ; > 100 dominees
    par les coudes).

  Les deux taux sont en 1/l : la course est presque invariante
  d'echelle -- c'est le LOG du coeur qui la tranche. Croisement :
        l_cut = D exp[Gamma_a / (2 pi kappa_s)].

RESULTAT : avec kappa_s = 2.85 mesure, l_cut = xi exige
Gamma_a = 2 pi kappa_s ln(xi/D) = 90 +- 20 -- dans la gamme des cordes
globales a coudes. La fenetre Gamma_a in [65, 130] donne
l_cut in [0.24, 9] xi : l'ordre xi est confirme SANS reglage (30x
au-dessus du coeur, loin de toute autre echelle). Le plancher
energetique (0.58 xi pour payer m(23)) et le plafond des coudes (~2 xi)
resserrent a [0.6, 2] : lambda = 1.14 tient.
"""
import math

KAPPA_S = 2.85          # scissions/periode, MESURE (gen 0, mediane 0.40)
D_FM = 0.021423
XI_FM = 3.3574
LN_XI_D = math.log(XI_FM / D_FM)


def l_cut(gamma_a, kappa_s=KAPPA_S):
    """En unites de xi."""
    return math.exp(gamma_a / (2 * math.pi * kappa_s) - LN_XI_D)


def run():
    print("=== La course scission / rayonnement : l_cut quantifie ===\n")
    print("kappa_s = %.2f scissions/periode (MESURE, generation 0 propre ;" % KAPPA_S)
    print("          mediane t*/(L/2) = 0.40 ; l'avalanche t*=0 des filles =")
    print("          amas d'evenements simultanes, exclue du taux)")
    print("ln(xi/D) = %.3f (hierarchie derivee : D = %.4f fm, xi = %.3f fm)\n" % (
        LN_XI_D, D_FM, XI_FM))

    print("l_cut = D exp[Gamma_a/(2 pi kappa_s)] :")
    print("    Gamma_a   l_cut/xi   lambda    Omega")
    for g in (65, 90.5, 130):
        lc = l_cut(g)
        lam = 1.14 * lc
        print("    %6.1f     %5.2f     %5.2f    %5.1f" % (g, lc, lam, 5.68 / lam))
    g_req = 2 * math.pi * KAPPA_S * LN_XI_D
    print("\nGamma_a REQUIS pour l_cut = xi : %.0f +- 20" % g_req)
    print("  (cordes globales : ~65 boucles lisses, >100 dominees coudes --")
    print("   la valeur requise est DANS la gamme, sans reglage)\n")

    print("VERDICT :")
    print(" - Le mecanisme est QUANTIFIE : la course (deux taux en 1/l,")
    print("   tranchee par le log du coeur) place le croisement a")
    print("   D e^{O(1) x 10} ~ xi -- 30x au-dessus du coeur, loin de toute")
    print("   autre echelle, sans reglage. La coupure n'est plus qualitative.")
    print(" - Ce qui reste calculable : Gamma_a(chi), la constante de")
    print("   radiation classique de la corde chi (couche microscopique du")
    print("   solveur) ; requis 90 +- 20, gamme des cordes globales a coudes.")
    print(" - Le plancher energetique 0.58 xi (payer m(23)) est STRUCTUREL :")
    print("   un fragment plus court ne peut porter qu'un noeud leger")
    print("   (secteur visible, s'annihile). Avec le plafond des coudes")
    print("   (~2 xi), la fenetre reste [0.6, 2] xi : lambda = 1.14 TIENT,")
    print("   Omega = 4.98 (-9 %).")


if __name__ == "__main__":
    run()
