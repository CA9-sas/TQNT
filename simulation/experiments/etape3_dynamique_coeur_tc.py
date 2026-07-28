"""Etape 3 : la dynamique de coeur -- origine non lineaire de la barriere (20/07).

ACQUIS des etapes 1-2 : aucun canal LINEAIRE de corde ne fournit la
barriere ~sigma_chi*xi (contact ~1/L refute, echange 0,08 neutre).
La barriere doit venir du COEUR (tube D = 0,0214 fm, physique non
lineaire). Deux classes de candidats :

 (A) COUTS LOCAUX de coeur (un evenement au voisinage du tube) :
     - passage de brin (boucle serree pi*D*sigma, 2pi*D*sigma) ;
     - pas geometrique (un croisement serre, Dm = 70,2 MeV) ;
     - noyau de nouage minimal (trefle serre, m_l = 310 MeV) ;
     - poche de fusion des coeurs (Lambda_chi^4 * D^3).
     TEST : le gel Gamma = H exige E_b = T * ln[M_Pl/(1,66 sqrt(g) T)]
     ~ 45 T ; un candidat local gele a T = E_b/45 -- chiffre.

 (B) CLIQUET RADIATIF (la vraie non-linearite de coeur) : un nœud
     forme est SERRE au diametre de coeur ; la protection de serrage
     (la meme physique que la suppression de Vassiliev alpha^c et la
     stabilite CS tau ~ 1e347 s, DEJA etablies) interdit tout passage
     de brin sans re-ouverture complete. Re-equilibrer la population
     de nœuds exige donc de RE-FONDRE le fragment porteur entier --
     dont l'energie est MESUREE par la cascade : la corde paie
     x1,7 la masse du nœud (E_frag = 1,7 m(c_min) = 2,94 GeV),
     degeneree a 2 % avec le quantum de domaine sigma_chi*xi = 2,99.
     La radiation des coudes (O(1) periodes, fortement couplee --
     physique de coeur documentee) rend la fonte IRREVERSIBLE dans
     le sens direct : la barriere est le cout du RETOUR.
"""
import math

M_PL = 1.2209e19
SQRT_SIGMA = 0.41847
SIGMA = SQRT_SIGMA ** 2               # GeV^2
HBARC = 0.19733
SIGMA_FM = SIGMA / HBARC              # GeV/fm
D_TUBE = 0.0214                       # fm
KAPPA_DOM = 1.14
G_S = 15.2
M_EDGE = 1.73                         # GeV m(c=23)
F_PAY = 1.7                           # la corde paie x1,7 la masse (mesure)
DM_STEP = 0.0702
M_L = 0.310                           # trefle serre (m_l)
LAMBDA_CHI = 0.2092
T_NOM = 0.067


def log_term(t):
    return math.log(M_PL / (1.66 * math.sqrt(G_S) * t))


def t_gel_fixed(e_b):
    """Gel Gamma=H pour une barriere FIXE (independante de T)."""
    t = 0.05
    for _ in range(100):
        t = 0.5 * t + 0.5 * e_b / log_term(t)
    return t


def run():
    print("=== Etape 3 : dynamique de coeur -- la barriere du gel ===\n")
    print("Le gel exige E_b ~ 45 T (log de M_Pl). Candidats :\n")

    print("(A) COUTS LOCAUX de coeur -- chiffres et elimines :")
    locals_ = [
        ("boucle serree pi D sigma", math.pi * D_TUBE * SIGMA_FM),
        ("boucle serree 2pi D sigma", 2 * math.pi * D_TUBE * SIGMA_FM),
        ("pas geometrique (1 croisement)", DM_STEP),
        ("noyau minimal (trefle serre, m_l)", M_L),
        ("poche de fusion Lambda^4 D^3",
         (LAMBDA_CHI ** 4 / HBARC ** 3) * D_TUBE ** 3),
    ]
    for name, e in locals_:
        t = t_gel_fixed(e)
        print("   %-36s E = %8.4f GeV -> gel a T = %6.2f MeV" % (
            name, e, t * 1e3))
    print("   -> TOUS gelent a T << 60-75 MeV : la classe locale est")
    print("      ELIMINEE par le log de gel (multi-GeV requis).\n")

    print("(B) CLIQUET RADIATIF -- la re-fonte du fragment porteur :")
    print("   Protection de serrage (Vassiliev alpha^c + CS tau~1e347 s,")
    print("   etablies) : pas de passage de brin sans re-ouverture ->")
    print("   la barriere = energie du fragment porteur de nœud.")
    e_frag = F_PAY * M_EDGE
    e_dom = KAPPA_DOM * SIGMA / 0.0667   # quantum de domaine a T~T_c
    cands = [
        ("fragment porteur : 1,7 x m(23) [MESURE cascade]", e_frag),
        ("quantum de domaine : kappa sigma/T [reference]", e_dom),
    ]
    for name, e in cands:
        t = t_gel_fixed(e)
        om = 3.9 * math.exp(-M_EDGE * (1 / t - 1 / T_NOM))
        print("   %-46s E = %.2f GeV -> T_c = %.1f MeV ; Omega = %.1f" % (
            name, e, t * 1e3, om))
    print("   -> les deux lectures coherentes sont DEGENEREES a %.1f %%" % (
        abs(e_frag / e_dom - 1) * 100))
    print("      (E_frag = 1,7 m ~ sigma xi : c'est le MEME objet vu")
    print("      par la cascade et par le reseau).\n")

    print("   Pourquoi PAS le fragment moyen (0,82 sigma xi = %.2f GeV) :" % (
        0.82 * KAPPA_DOM * SIGMA / 0.0667))
    print("   le re-equilibrage de la population passe par les fragments")
    print("   PORTEURS (c'est leur contenu topologique qui change) --")
    print("   selection mecanique, pas numerique.\n")

    print("SENSIBILITES (barriere fixe -> T lineaire en E_b) :")
    for lab, f in (("x1,7 -> x1,5", 1.5), ("x1,7 -> x1,9", 1.9)):
        t = t_gel_fixed(f * M_EDGE)
        om = 3.9 * math.exp(-M_EDGE * (1 / t - 1 / T_NOM))
        print("   %-18s T_c = %.1f MeV ; Omega = %.2f" % (lab, t * 1e3, om))
    t_lo, t_hi = t_gel_fixed(1.5 * M_EDGE), t_gel_fixed(1.9 * M_EDGE)
    print("   (le pin final = la precision du x1,7 de la cascade)\n")

    print("BILAN :")
    t_c = t_gel_fixed(e_frag)
    print(" - la dynamique de coeur SELECTIONNE la classe fragment (multi-")
    print("   GeV) : les couts locaux gelent a 0,05-7 MeV, elimines ;")
    print(" - mecanisme porteur = CLIQUET RADIATIF : serrage -> protection")
    print("   Vassiliev/CS -> re-equilibrage seulement par re-fonte du")
    print("   fragment porteur (E = 1,7 m, MESURE) ;")
    print(" - T_c = %.1f MeV (porteur) a %.1f MeV (domaine), soit" % (
        t_c * 1e3, t_gel_fixed(e_dom) * 1e3))
    print("   [-2,1 %%, -0,4 %%] du 67 adopte ; enveloppe [%.0f, %.0f] MeV" % (
        t_lo * 1e3, t_hi * 1e3))
    print("   sous +-12 %% sur le x1,7 ;")
    print(" - aval : Omega = 2,3-3,6 (-58 a -35 %% de Planck) -- ordre de")
    print("   grandeur maintenu, la comparaison fine attend le pin du 1,7.")


if __name__ == "__main__":
    run()
