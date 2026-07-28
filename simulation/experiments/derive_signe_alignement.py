"""Le signe de l'alignement j_K=1/2 : dissous par la monotonie de l'echelle (22/07).

CIBLE : le dernier « lu » du secteur des bandes -- l'alignement du doublet
j_K=1/2 admettait trois lectures (S17) : X = 2+0, 2-1/2, 2-1. Les trois
sont DEGENEREES a j_K=3/2 (toutes donnent le +1/2 demontre) : la
discrimination est structurelle.

[1] THEOREME DE MONOTONIE (structurel) : toutes les contributions de
    l'echelle canonique sont >= 0 -- regle close n = 1 + 3^g[bascule] +
    min(3^g(d-1),1)[...] + 1/2[J=3/2] : base 1, bascule, x3, exces,
    rung, TOUS positifs ; V-P : le P (anti-aligne) est le SOL, le V paie
    +1 (jamais -1/2 symetrique) ; rung baryonique : J=1/2 paie 0. La
    torsion stockee est un EXCES SUR LE SOL : pas de barreau negatif.
    => l'anti-aligne j_K=1/2 EST le sol du couple L x s_K : cout 0,
    X(1/2) = 2 EXACTEMENT. La question du signe se dissout (il n'y a
    rien a soustraire).

[2] LA COMPARAISON REPAREE : la bande predit des CENTROIDES ponderes
    (2J+1) -- la pratique du S16 pour j_K=3/2 (8 doublets <= 0,8 %).
    Le « D0* prefere -1/2 » du S17 comparait le MEMBRE BAS seul : biais
    vers le bas -> artefact.

[3] MEMBRES via le splitting derive (S27) : Delta = (3/16) sigma_chi/m_Q
    par unite ; distorsion DK (seuil 2363, double pole) declaree.
"""
import numpy as np

SIGMA = 0.41847 ** 2
PI_SIGMA = np.pi * SIGMA
ML, MC, MB = 0.310, 1.557, 4.952
CORE_C, CORE_B = MC + ML, MB + ML


def band_mass(core, n):
    return np.sqrt(core**2 + PI_SIGMA * n)


def main():
    print(__doc__)
    print("=" * 78)
    print("[2] CHARME j_K=1/2 : les trois lectures vs le CENTROIDE (2J+1)")
    print("=" * 78)
    d0, d1p = 2.343, 2.412               # D0*(2300) PDG 2343(10) ; D1(2430) 2412(9)
    cen = (1 * d0 + 3 * d1p) / 4
    print("  mesure : D0* = 2343(10), D1' = 2412(9) -> centroide %.1f MeV" % (1e3 * cen))
    print("  %-28s %6s %10s %9s" % ("lecture", "X", "M_pred", "ecart"))
    for nom, X in [("X = 2   (monotone, align 0)", 2.0),
                   ("X = 3/2 (symetrique -1/2)", 1.5),
                   ("X = 1   (Clebsch L.s -1)", 1.0)]:
        m = band_mass(CORE_C, 2 + X)     # charme : 3^g = 1 (base de generation)
        print("  %-28s %6.1f %10.1f %+8.1f %%" % (
            nom, X, 1e3 * m, 100 * (m / cen - 1)))
    print("""  -> au CENTROIDE (l'objet que la bande predit), la lecture MONOTONE
  X=2 est la meilleure (-0,4 %%, dans la precision ~1 %% des bandes) ;
  -1/2 (-2,9 %%) et -1 (-5,4 %%) sont EXCLUES. Le « D0* prefere -1/2 »
  du S17 etait l'artefact membre-vs-centroide.
  Controle 3/2 (meme machinerie) : n=9/2 -> %.1f vs 2446,5 (-0,2 %%).""" % (
        1e3 * band_mass(CORE_C, 4.5)))

    print("=" * 78)
    print("[3] LES MEMBRES (splitting S27) ET LA DISTORSION DECLAREE")
    print("=" * 78)
    cen_c = band_mass(CORE_C, 4.0)
    split_c = (3 / 16.0) * SIGMA / MC
    p0 = cen_c - 0.75 * split_c
    p1 = cen_c + 0.25 * split_c
    print("""  X=2 + splitting (3/16)sigma/m_c = %.1f MeV :
    D0* pred %.1f vs 2343 (%+.1f %%) ; D1' pred %.1f vs 2412 (%+.1f %%)
  -> residus ~1 %% du signe attendu de la DISTORSION DK (seuil D K a
  2363 MeV : attraction du D0* vers le bas, double pole -- declaree
  aux S16-17/27) ; l'ordre observe D0* < D1' est reproduit.""" % (
        1e3 * split_c, 1e3 * p0, 100 * (p0 / d0 - 1),
        1e3 * p1, 100 * (p1 / d1p - 1)))

    print("=" * 78)
    print("[4] PREDICTION AFFUTEE : la fourchette bottom s'effondre en un point")
    print("=" * 78)
    cen_b = band_mass(CORE_B, 2 + 3 * 2.0)          # bottom : 3^g = 3
    split_b = (3 / 16.0) * SIGMA / MB
    print("""  bottom j_K=1/2 (n = 2 + 3x2 = 8) :
    centroide B0*/B1' = %.0f MeV (le HAUT de l'ancienne fourchette
    5591-5665 -- le 5591 etait la lecture -1/2, le 5517 la lecture -1) ;
    membres avec le splitting %.1f MeV : B0* ~ %.0f, B1' ~ %.0f.
  FALSIFIABLE : une observation du centroide a ~5591 (ou ~5517)
  refuterait la monotonie de l'echelle ; a ~5665, elle la confirme --
  et fermerait le DERNIER element du secteur des bandes.""" % (
        1e3 * cen_b, 1e3 * split_b,
        1e3 * (cen_b - 0.75 * split_b), 1e3 * (cen_b + 0.25 * split_b)))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  Le « signe de l'alignement » est DISSOUS, pas choisi : l'echelle
  canonique est MONOTONE (aucun barreau negatif -- base, bascule, x3,
  1/2, exces, V-P, rung baryonique : tous des exces sur le sol) ;
  l'anti-aligne j_K=1/2 est le SOL du couple L x s_K -> X(1/2) = 2
  exactement. Les donnees (centroide charme -0,4 %) le confirment une
  fois l'artefact membre-vs-centroide repare ; les lectures negatives
  sont exclues (-2,9/-5,4 %). Prediction : B0*/B1' centroide 5665 MeV
  (plus une fourchette). LE SECTEUR DES BANDES N'A PLUS AUCUN OUVERT.""")


if __name__ == "__main__":
    main()
