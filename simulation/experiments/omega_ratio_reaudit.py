"""Re-audit d'Omega_DM/Omega_b = 5.45 sous les masses geometriques (3e retroaction).

FORMULE CANONIQUE (V6 eq:dm-ratio) :
    Omega_DM/Omega_b = sum_{c>c*} m(c) N(c) e^{-alpha c} / sum_{c<=c*} ...
avec, dans le calcul historique : m(c) = FEM <=10 puis ~c^6.65 (ERE V5),
N(c) = Hoste-Thistlethwaite <=16 puis 2.68^c/c, alpha = 1.20, c* = 18.
Resultat historique : 5.5 (Planck 5.45 +- 0.28).

CE QUE LA REFONTE CHANGE :
  1. MASSES : la loi canonique V6 par noeud est M_topo = scale*l(K)
     (geometrique, quasi lineaire) -- le c^6.65 est de l'ere V5. Or le 5.5
     reposait sur le boost massique du cote sombre : m(23)/m(3) ~ 7e5 sous
     c^6.65, contre ~5.5 sous la geometrie.
  2. c_min : 23 derive (decouplage/survie) -- le cote sombre commence a 23 ;
     les c=19-22 (formes mais recouples/detruits) retournent au fond thermique :
     variante ou leur masse est recyclee cote visible.
  3. RACCORD N(c) : la table HT croit en ~4.7-5.5/croisement jusqu'a c=16,
     mais l'asymptotique 2.68^c/c DECROIT au raccord (N(17) < N(16) !) --
     defaut du modele canonique ; variante N-continue (croissance geometrique
     amortie vers 2.68).

VARIANTES CALCULEES : {masse V5, masse geo} x {N canonique, N continu}
x {c* = 18, c_min = 23 (avec/sans recyclage 19-22)}.
"""
import math

ALPHA = 1.20
HT = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165, 11: 552,
      12: 2176, 13: 9988, 14: 46972, 15: 253293, 16: 1388705}
SCALE = 310.0 / 16.372
SLOPE = (38.62 - 16.372) / 6
L31 = 16.372
CMAX = 60


def m_v5(c):
    return 0.511e-3 * (c / 3.0) ** 6.65     # GeV (m_e a c=3)


def m_geo(c):
    return SCALE * (L31 + SLOPE * (c - 3)) / 1000.0


def N_canon(c):
    if c in HT:
        return float(HT[c])
    return 2.68 ** c / c


def N_continu(c):
    """Raccord continu : HT <=16, puis croissance geometrique du dernier ratio
    HT (5.48) relaxee exponentiellement vers 2.68 (echelle 10 croisements)."""
    if c in HT:
        return float(HT[c])
    N = float(HT[16])
    for cc in range(17, c + 1):
        r = 2.68 + (5.48 - 2.68) * math.exp(-(cc - 16) / 10.0)
        N *= r
    return N


def ratio(mfun, Nfun, cstar_dark, recycle_19_22=False):
    """Somme visible c=3..18 ; sombre c=cstar_dark..CMAX ;
    si recycle : le poids massique 19..cstar-1 est ajoute au visible."""
    w = lambda c: mfun(c) * Nfun(c) * math.exp(-ALPHA * c)
    vis = sum(w(c) for c in range(3, 19))
    if recycle_19_22:
        vis += sum(w(c) for c in range(19, cstar_dark))
    dark = sum(w(c) for c in range(cstar_dark, CMAX + 1))
    return dark / vis


def run():
    print("=== Re-audit Omega_DM/Omega_b (cible Planck : 5.45 +- 0.28) ===\n")
    cases = [
        ("HISTORIQUE  : masse V5 (c^6.65), N canonique, c*=19", m_v5, N_canon, 19, False),
        ("masse V5, N canonique, c_min=23", m_v5, N_canon, 23, False),
        ("masse GEO, N canonique, c*=19", m_geo, N_canon, 19, False),
        ("masse GEO, N canonique, c_min=23", m_geo, N_canon, 23, False),
        ("masse GEO, N canonique, c_min=23 + recyclage 19-22", m_geo, N_canon, 23, True),
        ("masse GEO, N continu,   c_min=23", m_geo, N_continu, 23, False),
        ("masse V5,  N continu,   c*=19", m_v5, N_continu, 19, False),
    ]
    for nom, mf, Nf, cd, rec in cases:
        r = ratio(mf, Nf, cd, rec)
        print(" %-55s : %8.3f  (%+5.1f%% vs 5.45)" % (nom, r, (r / 5.45 - 1) * 100))

    print("\nDiagnostic du raccord N(c) canonique : N(16)=%.2e (HT) vs N(17)=%.2e" % (
        N_canon(16), N_canon(17)))
    print("(asymptotique 2.68^c/c) : DECROISSANT au raccord -- defaut de modele.\n")

    print("VERDICT :")
    print(" - Le 5.5 historique etait PORTE par la loi de masse V5 (boost")
    print("   m(23)/m(3) ~ %.0e du cote sombre)." % (m_v5(23) / m_v5(3)))
    print(" - Sous la loi geometrique canonique V6 (+ c_min derive), la")
    print("   formule canonique donne ~%.2f : ECHEC par un facteur ~%.0f." % (
        ratio(m_geo, N_canon, 23), 5.45 / ratio(m_geo, N_canon, 23)))
    print(" - Aucune variante honnete (N continu, recyclage) ne remonte a 5.45")
    print("   sans reintroduire un boost massique du cote sombre.")
    print(" - Le succes 'Omega_DM/Omega_b = 5.5' est donc CONDITIONNE A LA LOI")
    print("   DE MASSE V5 (comme le 0.076%% d'alpha_cond) : sous le document principal")
    print("   actuel il est PERDU en l'etat -- a declasser en 'en revision'")
    print("   avec les sorties possibles : (i) loi de masse sombre non")
    print("   geometrique (a justifier sans reintroduire c^6.65 ad hoc) ;")
    print("   (ii) formule de ratio differente (budget d'energie de")
    print("   condensation, pas somme ponderee) ; (iii) N(c) reel (croissance")
    print("   >2.68 : inconnu mathematiquement au-dela de c~19).")


if __name__ == "__main__":
    run()
