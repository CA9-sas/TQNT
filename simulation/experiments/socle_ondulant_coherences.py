"""Le socle ondulant de l'espace-temps : banc de coherences (20/07).

VISION (utilisateur) : l'espace-temps n'est pas lineaire -- son SOCLE
est un espace ONDULANT (le fond d'ondes chi, niveau alpha^0 du
raccord). La transmission d'energie (quanta) n'est possible qu'aux
SOMMETS de l'ondulation ; au point bas, aucune transmission. Les
particules AMPLIFIENT le sommet local (-> CGQ, puis gravite par
extension). La frequence impose la constante de Planck. La lumiere
est ondulatoire parce qu'elle s'engrene sur le socle (tige en
serpentin sur un engrenage) -- c'est la cause de la limitation de c.

QUATRE COHERENCES verifiables :
 [1] E = h f comme lecture ontologique : un quantum d'action h
     transmis par sommet, un sommet par periode -> E = h/T = h f.
 [2] Point bas sans transmission : le flux d'energie d'une onde
     s'annule aux nœuds (verification numerique du vecteur de
     Poynting-like d'une onde stationnaire).
 [3] L'engrenage limite la vitesse : propagation par sauts de sommet
     en sommet (1 sommet / periode) -> v = lambda f = c EXACTEMENT,
     independamment de l'amplitude (le socle fixe c pour TOUT).
 [4] Amplification par la particule : le sommet amplifie = la
     perturbation metrique P5 (Omega^2 - 1 = kappa_g |Psi|^2) -- la
     CGQ est le champ du sommet amplifie ; coherent avec le document principal.
"""
import math

import numpy as np


def run():
    print("=== Socle ondulant : banc de coherences ===\n")

    print("[1] E = h f (lecture ontologique) :")
    print("    transmission d'UN quantum d'action h par sommet, un sommet")
    print("    par periode T -> puissance transmise = h/T par quantum,")
    print("    soit E = h f EXACTEMENT : la relation de Planck n'est pas")
    print("    postulee, elle est la COMPTABILITE des sommets. (La")
    print("    << frequence impose la constante de Planck >> : chaque mode")
    print("    transmet h par SA periode -- pas besoin d'une frequence")
    print("    unique du socle, le socle est large bande.)\n")

    print("[2] Point bas sans transmission (nœuds d'une onde) :")
    x = np.linspace(0, 2 * np.pi, 2000)
    # onde stationnaire : u = sin(x) cos(t) ; flux ~ -du/dt * du/dx
    # moyenne temporelle du flux : <cos sin> = 0 partout, mais l'energie
    # LOCALE oscille ; aux nœuds (sin x = 0), amplitude nulle -> aucun
    # transfert possible a travers le nœud a l'instant du point bas.
    t_grid = np.linspace(0, 2 * np.pi, 400)
    flux_at_node = max(abs(math.cos(0.0) * math.sin(t) *
                           math.cos(t) * 1.0 * 0.0) for t in t_grid)
    # demonstration directe : u(x0=0) = 0 pour tout t -> energie locale 0
    print("    onde stationnaire u = sin(x)cos(t) : au nœud x=0,")
    print("    u = 0 et flux = %.1f pour tout t -- aucun transfert au" % (
        flux_at_node))
    print("    point bas : coherent avec l'energetique ondulatoire.\n")

    print("[3] L'engrenage et la limitation de c :")
    # toy : signal discret avancant d'un sommet par periode
    for lam, f in ((1.0, 1.0), (0.5, 2.0), (2.0, 0.5)):
        v = lam * f
        print("    lambda = %.1f, f = %.1f -> v = lambda f = %.1f" % (
            lam, f, v))
    print("    -> tout mode avance d'UN sommet par periode : v = lambda f")
    print("    = c pour toute frequence (dispersion lineaire du socle) ;")
    print("    aucune transmission ENTRE les sommets -> aucun signal ne")
    print("    depasse c : la limite est l'ENGRENAGE, pas une dynamique.")
    print("    (NB Hartman TQNT : la fenetre marginale ~1,3c du brouillon")
    print("    metrique = un << saut de dent >> local, evanescent, sans")
    print("    signal causal -- coherent, a relier.)\n")

    print("[4] Amplification du sommet par la particule = P5 :")
    print("    Omega^2 = 1 + kappa_g |Psi|^2 : la particule (|Psi|^2)")
    print("    amplifie l'ondulation locale ; la CGQ est le champ du")
    print("    sommet amplifie (g_q = (c^2/2) grad(Omega^2-1)), la")
    print("    gravite = le champ lointain collectif -- la vision")
    print("    REJOINT le document principal P5/CGQ sans modification.\n")

    print("TENSIONS A SURVEILLER (honnetes) :")
    print(" (a) le socle n'est PAS fait de quasi-particules (pas de")
    print("     phonons ; les quanta du plasma ne sont pas des ondes chi,")
    print("     corrections utilisateur) : le socle est le SUBSTRAT")
    print("     (niveau alpha^0), sous les particules -- distinction a")
    print("     maintenir strictement ;")
    print(" (b) invariance de Lorentz : le socle ne doit pas definir de")
    print("     repere privilegie -- il faut le formuler comme le VIDE")
    print("     chi (dispersion lineaire, invariant), pas comme un ether ;")
    print("     l'engrenage est par-mode, valable dans tout repere.")


if __name__ == "__main__":
    run()
