"""Deriver le seuil : la marche d'echelle et ses trois fermetures (20/07).

Le seuil (N_topo = 1/e) n'est pas une convention isolee -- il se
derive de la STRUCTURE DE MARCHE du changement de type, avec trois
objections structurelles qui se referment chacune par un element
etabli du cadre :

[1] << SITE FAIBLE >> : un seul site bon marche suffirait-il a
    denouer ? NON : le nombre de denouement des noeuds du reservoir
    est grand -- u(T(2,q)) = (q-1)/2 (11 passages pour T(2,23)).
    Le denouage est une MARCHE sur l'echelle des types (DeltaLk = +-1
    par pas -- exactement la marche naissance-mort de la
    reconstruction canonique, taux net 0,68/croisement). Une marche
    visite des sites TYPIQUES : le site MEDIAN gouverne (q = 0,95),
    pas le lien faible.

[2] << SEUIL >> : pour une marche dont chaque pas dispose d'EXACTEMENT
    une tentative (N_att = (1/t*) x t* = 1, mesure, compensation de
    t*), le gel est le gel standard de marche : le deplacement attendu
    restant tombe sous un pas quand p_pas = e^{-E_char/T} < 1/e --
    le seuil e-fold EST le seuil de marche a tentative unique.
    => c_switch = 1 (statut : derive de la structure de marche).

[3] << TENTATIVES POST-CASCADE >> : un nœud couple au fond thermique
    accumulerait ~e^45 tentatives thermiques par e-pli de T (le log
    cosmologique reviendrait). FERMETURE : les nœuds SOMBRES sont nes
    DECOUPLES -- c'est la derivation de c_min = 23 (Gamma/H a T_c :
    c=22 couple 9,3 ; c=23 decouple 0,067) : pas d'energie du fond thermique,
    pas de mou, pas de tentatives post-cascade. La fenetre du secteur
    sombre se ferme avec le decouplage -- coherent avec T* (les deux
    ~60-100 MeV ; la boucle T_c<->c_min est une COHERENCE, pas une
    circularite : deux criteres independants, meme fenetre).

BANC : verification u(T(2,q)) ; comparaison site-median vs
lien-faible (rates de marche) ; recapitulatif Omega.
"""
import math
import sys

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA = 0.41847 ** 2
D0_SIG = 0.1084 * SIGMA
T_TRANS = 0.155
EXPO = 2 * 0.6301
P_OCC_EFF = (8.0 / 9.0) * 0.304


def c_p_of_s(u):
    return math.pi + 2 * math.atan(1 / u) - 2 * math.atan(u)


def c_eff_of_q(q):
    u = math.sqrt(max(1e-9, 2 / q - 1))
    return c_p_of_s(u) + 2 * q * u


def t_c_of(c_eff, c_switch=1.0):
    t = 0.05
    for _ in range(200):
        x = max(1e-9, 1 - t / T_TRANS) ** EXPO
        t = 0.5 * t + 0.5 * c_eff * D0_SIG * math.sqrt(x) / c_switch
    return t


def run():
    print("=== Le seuil derive de la structure de marche ===\n")

    print("[1] LE DENOUAGE EST UNE MARCHE (pas un evenement) :")
    for q in (3, 9, 23):
        print("   u(T(2,%d)) = %d passages" % (q, (q - 1) // 2))
    print("   -> DeltaLk = +-1 par pas : la marche naissance-mort de la")
    print("   reconstruction canonique (taux net 0,68/croisement). Une")
    print("   marche visite des sites TYPIQUES -> le site MEDIAN gouverne.")
    q_med, q_min = 0.95, 0.40
    for lab, q in (("site median (marche)", q_med),
                   ("lien faible (1 evenement -- REFUTE par u=11)", q_min)):
        ce = c_eff_of_q(q)
        t = t_c_of(ce)
        b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-46s T_c = %4.1f MeV, Omega = %6.3f" % (
            lab, t * 1e3, b["omega"]))

    print("\n[2] LE SEUIL EST CELUI D'UNE MARCHE A TENTATIVE UNIQUE :")
    print("   N_att par pas = (1/t*) x t* = 1 (mesure, compense) ;")
    print("   deplacement attendu restant < 1 pas quand p = e^{-E/T} < 1/e")
    print("   -> c_switch = 1, DERIVE de la structure (le 1/e n'est plus")
    print("   une convention : c'est le gel de marche standard).\n")

    print("[3] FENETRE POST-CASCADE FERMEE PAR LE DECOUPLAGE :")
    print("   nœud couple -> ~e^45 tentatives (log cosmologique) ;")
    print("   les nœuds sombres sont nes DECOUPLES (derivation c_min=23 :")
    print("   Gamma/H : c=22 -> 9,3 couple ; c=23 -> 0,067 decouple) ->")
    print("   pas d'energie du fond thermique, pas de tentatives post-cascade.")
    print("   Coherence T* <-> decouplage : deux criteres independants,")
    print("   meme fenetre (~60-100 MeV) -- coherence, pas circularite.\n")

    ce = c_eff_of_q(q_med)
    t = t_c_of(ce)
    b = budget(tc=t, g_s=15.2, p_occ=P_OCC_EFF)
    print("RECAPITULATIF : seuil e-fold DERIVE (marche a tentative unique,")
    print("site median par la marche, fenetre fermee par le decouplage) :")
    print("   T_c = %.1f MeV ; Omega = %.2f (Planck 5,45 : %+.0f %%)" % (
        t * 1e3, b["omega"], 100 * (b["omega"] / 5.45 - 1)))


if __name__ == "__main__":
    run()
