"""Amplification annulaire (sommets voisins, pas le sommet du dessous) :
est-ce que ca change les calculs ? (21/07)

Vision utilisateur affinee : la particule amplifie les sommets AUTOUR
d'elle, pas le sommet directement en dessous. Trois verifications :

[1] LE PROFIL ANNULAIRE EST DEJA CELUI DU CADRE. La densite |Psi|^2 d'un
    noeud est localisee AUTOUR du tube (le confinement du solveur est
    V = -lambda_V rho_J/d^2 avec d = distance a la COURBE : les etats
    lies vivent sur un voisinage tubulaire, pas au centre geometrique de
    la boucle). P5 (Omega^2 = 1 + kappa_g|Psi|^2) amplifie donc la ou
    |Psi|^2 est -- l'anneau, PAS le centre. La vision annulaire est la
    lecture CORRECTE de P5 sur le socle.

[2] LE CHAMP LOINTAIN NE DEPEND PAS DU PROFIL. Toutes les predictions
    gravitationnelles (G a -0,05 %, Newton, Schwarzschild, Mercure,
    GW150914...) sont du champ lointain : elles ne dependent que de
    l'integrale totale de |Psi|^2 (le monopole). La correction annulaire
    est quadrupolaire, ~ (R_noeud/r)^2 -- chiffree ci-dessous : elle est
    a des dizaines d'ordres sous toute mesure. AUCUN nombre ne bouge.

[3] LE CENTRE NON AMPLIFIE A DEJA UN NOM : P20. << La matiere repousse
    l'espace-temps ; la zone d'exclusion genere la courbure >> (document principal).
    La vision annulaire = P5 (amplification autour, portee par |Psi|^2)
    + P20 (exclusion au coeur) lus ENSEMBLE sur le socle. Les chaines de
    pres (V_NN par recouvrements, moments de Biot-Savart, hyperfins
    tau_perp, d_tube = lambdabar_C) vivent deja sur le tube : rien a
    refaire.
"""
import numpy as np


def ring_potential_axis(r, ring_r):
    """Potentiel (unites GM=1) d'un anneau de rayon R, sur l'axe, a
    distance r du centre : -1/sqrt(r^2+R^2). Point : -1/r."""
    return -1.0 / np.sqrt(r * r + ring_r * ring_r)


def run():
    print(__doc__)
    print("=" * 72)
    print("[2] CHIFFRAGE : anneau vs point -- correction du champ lointain")
    print("=" * 72)
    ring = 1.0e-15                    # taille du noeud ~ fm
    cases = [("echelle atomique (1 A)", 1e-10),
             ("echelle du laboratoire (1 m)", 1.0),
             ("echelle terrestre (6,4e6 m)", 6.4e6)]
    for name, r in cases:
        phi_ring = ring_potential_axis(r, ring)
        phi_point = -1.0 / r
        rel = abs(phi_ring / phi_point - 1)
        print("  %-32s ecart relatif anneau/point = %.1e" % (name, rel))
    print("  -> correction ~ (R/r)^2/2 : quadrupolaire, 10-20 a 60 ordres")
    print("     sous les precisions experimentales. G, Newton, Mercure,")
    print("     GW : INCHANGES (ils ne voient que le monopole).")

    print("\n" + "=" * 72)
    print("[4] << G ETAIT L'AMPLITUDE DE L'ONDE >> : c'est l'identite du document principal")
    print("=" * 72)
    # G = kappa_g c^2 / (8 pi m_base) -- kappa_g est le COEFFICIENT
    # D'AMPLITUDE de la reponse du socle (Omega^2 = 1 + kappa_g |Psi|^2)
    kappa_g = 3.1201e-53              # m (EXACT, document principal raccord)
    c = 299792458.0
    m_base = 938.240e6 * 1.602176634e-19 / c**2   # kg
    g_pred = kappa_g * c * c / (8 * np.pi * m_base)
    print("  kappa_g = %.4e m (le coefficient d'amplitude du socle)" % kappa_g)
    print("  G = kappa_g c^2/(8 pi m_base) = %.4e m^3 kg^-1 s^-2" % g_pred)
    print("  G_predit (document principal)             = 6.6710e-11        (ecart %.1e)"
          % abs(g_pred / 6.6710e-11 - 1))
    print("  -> G EST l'amplitude de la reponse ondulatoire du socle, par")
    print("     unite de masse : l'amplification des sommets voisins a pour")
    print("     coefficient kappa_g|Psi|^2, et G = kappa_g c^2/(8 pi m_base).")
    print("     La vision << G = amplitude de l'onde >> est l'identite")
    print("     canonique P19/raccord, pas un changement.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  La vision annulaire NE CHANGE AUCUN CALCUL :")
    print("  [1] elle est deja la lecture correcte de P5 (|Psi|^2 est")
    print("      tubulaire/annulaire -- le solveur confine sur la courbe) ;")
    print("  [2] le champ lointain ne depend que du monopole (verifie) ;")
    print("  [3] les chaines de pres vivent deja sur le tube ; et le centre")
    print("      non amplifie est P20 (zone d'exclusion) -- la vision")
    print("      annulaire = P5 + P20 lus ensemble sur le socle.")
    print("  Seule l'ILLUSTRATION change (schema : sommets voisins amplifies,")
    print("  sommet central non amplifie/exclu) -- et une precision de")
    print("  lecture S4 pourrait etre consignee (brouillon/V6 sur demande).")


if __name__ == "__main__":
    run()
