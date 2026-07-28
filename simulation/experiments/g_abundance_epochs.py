"""G selon l'abondance des nucleons : plusieurs constantes selon les epoques (19/07).

PROPOSITION (utilisateur, 2026-07-19) : la masse de reference du theoreme
G n'est pas celle d'un etat fixe mais la MOYENNE PONDEREE des nucleons
par densite de masse dans l'univers :

    G(t) = (12/7) alpha_EM^18 hbar c / m_bar(t)^2
    m_bar(t) = m_base + f_n(t) x Delta_tension     (Delta = 1,293 MeV)

ou f_n(t) est la fraction NEUTRONIQUE des baryons de l'univers a
l'epoque t. La ponderation est COSMIQUE (etat global du fond chi), pas
locale : une etoile a neutrons ne porte pas son propre G (coherent avec
le pulsar triple PSR J0337+1715, universalite a 2,6e-6) ; seule l'EPOQUE
change la constante.

Ce script calcule G aux epoques cles et confronte aux contraintes :
  - BBN (vitesse d'expansion, sensibilite de Y_p a G) ;
  - CMB (G a la recombinaison = G d'aujourd'hui, composition gelee) ;
  - LLR (derive seculaire Gdot/G, alimentee par l'helium stellaire) ;
  - metrologie actuelle (CODATA, Rosi-14).

BORNE DE VALIDITE : au-dessus de T_c = 67 MeV il n'y a pas de nucleons
(condensation chi) -- l'ancre de la chaine disparait, le regime est
celui du fond d'ondes chi (chapitres A_s/inflation), pas de G nucleonique.
"""
import math

M_BASE = 938.240        # MeV -- base octet sqrt(sigma_chi)*S (plancher, proton)
D_TENSION = 1.293       # MeV -- tension d'enlacement du neutron (derivee, beta)
G_CODATA = 6.67430e-11
U_CODATA = 2.2e-5       # incertitude relative
G_ROSI14 = 6.67191e-11  # interferometrie atomique, Nature 2014
U_ROSI14 = 1.5e-4
ALPHA = 1.0 / 137.0
HBARC_SI = 3.1615268e-26   # J.m
MEV_KG = 1.78266192e-30    # kg/MeV


def G_simple(m_mev):
    """G par mise a l'echelle depuis la valeur etablie a m_base."""
    return 6.6710e-11 * (M_BASE / m_mev) ** 2


def run():
    print("=== G selon l'abondance : G(t) = (12/7) a^18 hbar c / m_bar(t)^2 ===")
    print("    m_bar(t) = %.3f + f_n(t) x %.3f MeV\n" % (M_BASE, D_TENSION))

    epochs = [
        ("T >> 1 MeV (equilibre faible, n/p = 1)",      0.5),
        ("T_c = 67 MeV (condensation, n/p = e^-Dm/T)",  math.exp(-D_TENSION / 67.0) / (1 + math.exp(-D_TENSION / 67.0))),
        ("gel faible T ~ 0,8 MeV (n/p = 1/6)",          1.0 / 7.0),
        ("post-BBN -> aujourd'hui (Y_p = 0,245)",       0.245 / 2.0),
        ("[reference V6 : plancher, f_n = 0]",          0.0),
    ]
    print("%-46s %-7s %-9s %-12s %s" % ("Epoque", "f_n", "m_bar", "G", "vs CODATA"))
    res = {}
    for label, fn in epochs:
        m = M_BASE + fn * D_TENSION
        G = G_simple(m)
        res[label] = G
        print("%-46s %.4f  %8.3f  %.4e  %+.3f %%" % (
            label, fn, m, G, 100 * (G / G_CODATA - 1)))

    G_now = res["post-BBN -> aujourd'hui (Y_p = 0,245)"]
    G_prim = res["T >> 1 MeV (equilibre faible, n/p = 1)"]
    G_bbn = res["gel faible T ~ 0,8 MeV (n/p = 1/6)"]

    print("\n[1] Amplitude totale de l'evolution :")
    print("    G(primordial)/G(aujourd'hui) - 1 = %+.2e" % (G_prim / G_now - 1))
    print("    (G croit de ~0,10 %% entre l'equilibre faible et aujourd'hui)")

    print("\n[2] BBN : G a l'ere du gel differe de G actuel de %+.2e" % (G_bbn / G_now - 1))
    dY = 0.16 * 0.245 * abs(G_bbn / G_now - 1)   # dY_p ~ 0,16 Y dG/G (vitesse d'expansion)
    print("    effet sur Y_p ~ %.1e (obs. : Y_p = 0,245 +- 0,003) -> INVISIBLE" % dY)

    print("\n[3] CMB/aujourd'hui : composition GELEE depuis BBN -> G constant")
    print("    depuis z ~ 1e8 ; les contraintes CMB (~2 %%) ne mordent pas.")

    print("\n[4] Derive seculaire (helium stellaire) :")
    dY_stellar = 0.01           # accroissement du Y cosmique sur ~10 Gyr
    dfn = dY_stellar / 2.0
    dG = -2 * dfn * D_TENSION / (M_BASE + 0.1225 * D_TENSION)
    print("    Delta Y ~ +0,01/10 Gyr -> Gdot/G ~ %.1e /an" % (dG / 1.0e10))
    print("    borne LLR : |Gdot/G| < 1,5e-13 /an -> marge x%.0f ;" % (1.5e-13 / abs(dG / 1.0e10)))
    print("    PREDICTION : derive negative ~1e-15/an, sous la borne actuelle.")

    print("\n[5] Metrologie actuelle :")
    for lab, Gv, u in [("CODATA 2018", G_CODATA, U_CODATA), ("Rosi-14", G_ROSI14, U_ROSI14)]:
        d = G_now / Gv - 1
        print("    vs %-12s : %+.2e  (%.1f sigma de sa propre incertitude)" % (
            lab, d, abs(d) / u))
    print("    -> G(aujourd'hui) pondere = SOUS le bord bas du G puzzle,")
    print("       3,2 sigma sous Rosi-14 (la variante plancher f_n=0 etait a")
    print("       <1 sigma de Rosi-14) : c'est le cout observationnel du modele.")

    print("\n[6] Etoile a neutrons : PAS de G propre (ponderation cosmique, pas")
    print("    locale) -- coherent avec PSR J0337+1715 (universalite 2,6e-6)")
    print("    et MICROSCOPE (1e-15). La composition LOCALE ne change pas G.")

    print("\n[7] Chaine Omega : G(T_c) = %.4e (f_n ~ 0,495) au lieu de 6,6710e-11" % res[
        "T_c = 67 MeV (condensation, n/p = e^-Dm/T)"])
    print("    -> dH/H = %+.1e a T_c : invisible (sauts Vassiliev x137," % (
        0.5 * (res["T_c = 67 MeV (condensation, n/p = e^-Dm/T)"] / 6.6710e-11 - 1)))
    print("    enveloppe T_c dominante). Omega = 3,9 +- 0,4 inchange.")

    print("\nBORNE : T > T_c = 67 MeV -> pas de nucleons, l'ancre disparait ;")
    print("le regime pre-condensation est celui du fond d'ondes chi (A_s, V5.15).")


if __name__ == "__main__":
    run()
