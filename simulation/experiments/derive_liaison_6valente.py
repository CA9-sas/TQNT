"""La liaison 6-valente dissoute : Steiner -> double-Y + pont (22/07).

LA CIBLE : la systematique dominante du H_cc (+-40 MeV, « liaison
6-valente inconnue », S36).

[1] LE THEOREME DE STEINER : dans un reseau minimal de tubes a tension
    uniforme, TOUT sommet interne est 3-valent a 120 degres (un sommet
    de degre >= 4 se scinde toujours en reduisant la longueur --
    theoreme classique des arbres de Steiner). => LE SOMMET 6-VALENT
    N'EXISTE PAS : le dibaryon compact = DEUX jonctions Y (120 degres,
    baryon-standard) reliees par un PONT.
    Consequences : (i) la base 2 SURVIT (2 enroulements de trialite,
    un par jonction -- la base compte les enroulements, pas les
    sommets) ; (ii) la geometrie angulaire est celle du baryon : la
    liaison 1/2 se transfere TELLE QUELLE -- l'« inconnue 6-valente »
    n'existe pas ; (iii) le pont porte un caractere Z3 (conduit :
    caractere a J1, conjugue a J2 -- neutralite des deux jonctions
    preservee ; le pont est un tube MESONIQUE entre deux jonctions
    baryoniques).
[2] LE PONT : la seule piece nouvelle -- un segment de tube de tension
    sigma_chi, longueur bornee par la geometrie du tube : l'equilibre
    est au contact des jonctions, l in [0, D] (D = 0,0214 fm, le
    diametre canonique) => E_pont = sigma_chi l in [0, +19 MeV].
    LA SYSTEMATIQUE +-40 DEVIENT [+0, +19], BORNEE PAR MECANISME.
"""
import numpy as np

SIGMA = 0.175117                        # GeV^2
PS = np.pi * SIGMA
ML, MS_, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1
HBARC = 0.19733                         # GeV.fm
D_TUBE = 0.0214 / HBARC                 # GeV^-1
E_BR_MAX = SIGMA * D_TUBE               # GeV


def mass(core, n, ebr=0.0):
    return np.sqrt((core + ebr)**2 + PS * n)


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE PONT CHIFFRE")
    print("=" * 78)
    print("  E_pont(max) = sigma_chi x D = %.4f GeV^2 x %.4f GeV^-1 = %.1f MeV"
          % (SIGMA, D_TUBE, 1e3 * E_BR_MAX))
    print("""  -> la fenetre du pont [0, D] borne l'energie a [0, +19 MeV] :
  la systematique dominante passe de +-40 (inconnue) a [+0, +19]
  (bornee par la geometrie canonique du tube).""")

    print("=" * 78)
    print("[B] LES TROIS DIBARYONS, FENETRE DU PONT INCLUSE")
    print("=" * 78)
    cases = [
        ("H  (2s+4l)", 2 * MS_ + 4 * ML, 2 + D1,
         [("Lambda Lambda", 2.23136)]),
        ("H_cc (2c+4l)", 2 * MC + 4 * ML, 2 + 3 * D1,
         [("Xi_cc + N", 4.5599), ("Lambda_c Lambda_c", 4.57292)]),
        ("H_bb (2b+4l)", 2 * MB + 4 * ML, 2 + 9 * D1,
         [("Xi_bb + N", 11.2789), ("Lambda_b Lambda_b", 11.23920)]),
    ]
    for nom, core, n, seuils in cases:
        m0 = mass(core, n)
        m1 = mass(core, n, E_BR_MAX)
        mc_ = mass(core, n, E_BR_MAX / 2)
        print("  %-14s M = [%.0f, %.0f] MeV (central %.0f, pont D/2)" % (
            nom, 1e3 * m0, 1e3 * m1, 1e3 * mc_))
        for snom, sm in seuils:
            print("      vs %-22s [%+.0f, %+.0f] MeV" % (
                snom + " (%.0f)" % (1e3 * sm),
                1e3 * (m0 - sm), 1e3 * (m1 - sm)))
    print("""
  -> H LEGER : [+279, +295] au-dessus de Lambda Lambda -- l'ABSENCE
  est ROBUSTE au pont (la postdiction des nulls tient) ;
  -> H_bb : [+45, +64] au-dessus de Lambda_b Lambda_b -- non lie,
  robuste ;
  -> H_cc : LA FENETRE [4554, 4573] est PINCEE ENTRE LES SEUILS --
  elle s'etend du juste-sous Xi_cc N (-6) au contact de
  Lambda_c Lambda_c (-0,5) : la phenomenologie s'AFFINE.""")

    print("=" * 78)
    print("[C] LA PHENOMENOLOGIE AFFINEE DU H_cc")
    print("=" * 78)
    print("""  Central (pont D/2) : M = 4563 +- 10 MeV --
  ENTRE les deux seuils (Xi_cc N a 4560, Lambda_c Lambda_c a 4573) :
  - branche HAUTE (2/3 de la fenetre, M > 4560) : RESONANCE ETROITE
    Xi_cc N -- desintegration forte a petit espace de phase
    (Gamma ~ MeV), canaux Xi_cc++ n / Xi_cc+ p : detection via
    Xi_cc reconstruit + nucleon ; TOUJOURS sous Lambda_c Lambda_c
    (pas de canal ouvert vers 2 Lambda_c : etroitesse garantie) ;
  - branche BASSE (M < 4560) : lie sous les deux seuils ->
    desintegration faible, vertex deplace ~30 um (S36).
  Dans LES DEUX branches : une structure dans [4554, 4573], JAMAIS
  au-dessus de Lambda_c Lambda_c -- falsifiable : un pic Lambda_c
  Lambda_c au-dessus de 4573 refuterait la lecture compacte.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  LA LIAISON 6-VALENTE EST DISSOUTE PAR THEOREME (Steiner : pas de
  sommet de degre > 3 dans un reseau minimal) : le dibaryon compact =
  double-Y (120 degres, baryon-standard, liaison 1/2 transferee) +
  PONT mesonique (conduit de caractere Z3). La seule piece nouvelle --
  le pont -- est BORNEE par la geometrie canonique du tube :
  [0, +19 MeV]. Resultat : H_cc = 4563 +- 10 MeV, pince entre les
  seuils, avec les deux branches phenomenologiques nettes ; H leger
  absent et H_bb non lie ROBUSTES. La systematique dominante est
  passee d'une inconnue O(1) a une fenetre mecanisee de 19 MeV.""")


if __name__ == "__main__":
    main()
