"""Le test de ropelength du trefle pour k = 135 (25/07).

CONVENTIONS D'ABORD (piege classique) : la ropelength mathematique
standard est L/r (longueur/rayon) ; la chaine k utilise L/D
(longueur/diametre) = moitie. ACPR (table canonique du depot, audit
22/07) : l(3_1) = 32,7448 en L/r = 16,3724 en L/D -- c'est une borne
SUPERIEURE (la meilleure configuration connue, ridgerunner) ; la
borne INFERIEURE rigoureuse est 31,32 en L/r = 15,66 en L/D
(Denne-Diao-Sullivan).

DEUX LECTURES DE k+2 = (8 pi/3) l(3_1) :
 [A] LE VERROU ENTIER (robuste) : la fermeture de phase QUANTIFIE le
     compte -> k+2 = round((8 pi/3) l). Le tube physique realise la
     configuration fermable la plus proche de l'ideal ; l'ecart est
     une DEFORMATION du second ordre (cout ~ epsilon^2, negligeable).
 [B] L'IDENTITE EXACTE (sharp) : l_min(3_1) = 411/(8 pi) exactement
     -- une prediction de mathematiques pures, falsifiable.
NOTRE SONO (valide a +0,8 % du 16,37) ne peut pas departager B :
le test de B est l'etat de l'art (ridgerunner) et les bornes
rigoureuses. On execute les deux verdicts honnetement.
"""
import math

L_OVER_D_ACPR = 32.7448 / 2      # 16.3724 (borne sup, meilleure config)
L_OVER_D_MIN = 31.32 / 2         # 15.66 (borne inf rigoureuse)
PRED_B = 411 / (8 * math.pi)     # 16.3532
ALPHA = 1 / 137.036


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LE VERROU ENTIER : k = 135 TIENT SUR LA GEOMETRIE ETABLIE")
    print("=" * 78)
    n_bare = 8 * math.pi / 3 * L_OVER_D_ACPR
    k2 = round(n_bare)
    lo = (k2 - 0.5) * 3 / (8 * math.pi)
    hi = (k2 + 0.5) * 3 / (8 * math.pi)
    strain = n_bare / k2 - 1
    print("""  compte nu : (8 pi/3) x %.4f = %.2f  ->  round = %d  ->  k = %d
  FENETRE du verrou : l(3_1) dans [%.3f, %.3f] (L/D) -- la valeur
  ACPR (%.4f) est confortablement DEDANS (et meme la borne inf
  rigoureuse %.2f donne round((8pi/3)x15,66) = %d : hors fenetre --
  la fenetre est physiquement discriminante, pas triviale).
  DEFORMATION de quantification : epsilon = %+.2f %% -- cout
  elastique du second ordre ~ epsilon^2 ~ %.1e : negligeable
  (le tube fermable a 137 cellules est quasi ideal).
  => AU RANG DU VERROU ENTIER, k = 135 SORT DE LA GEOMETRIE DU
  TREFLE TELLE QUE MESUREE -- aujourd'hui, sans attendre.""" % (
        L_OVER_D_ACPR, n_bare, k2, k2 - 2, lo, hi, L_OVER_D_ACPR,
        L_OVER_D_MIN, round(8 * math.pi / 3 * L_OVER_D_MIN),
        100 * strain, strain**2))

    print("=" * 78)
    print("[B] L'IDENTITE EXACTE : DEFAVORISEE, NON REFUTEE")
    print("=" * 78)
    gap = (L_OVER_D_ACPR - PRED_B) / PRED_B
    print("""  prediction sharp : l_min = 411/(8 pi) = %.4f (L/D)
  meilleure config connue (ACPR/ridgerunner) : %.4f (borne SUP)
  -> la prediction exige que le VRAI minimum soit %.2f %% SOUS la
  meilleure configuration connue. Etat de l'art : ridgerunner est
  estime converge a ~1e-4 relatif (analyse de criticite) -- la
  variante exacte est DEFAVORISEE ; mais la borne inferieure
  rigoureuse (%.2f) est loin : elle n'est PAS refutee.
  Ce qui trancherait : une borne INFERIEURE rigoureuse > %.4f
  (refutation nette) ou une configuration sous %.4f (support fort).
  Notre SONO (+0,8 %%) ne peut pas y contribuer : cible externe
  (litterature des bornes de ropelength).""" % (
        PRED_B, L_OVER_D_ACPR, 100 * gap, L_OVER_D_MIN, PRED_B,
        L_OVER_D_ACPR))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  LA CHAINE k = 135 FERME AU RANG DU VERROU ENTIER :
     k + 2 = round((8 pi/3) l(3_1)) = 137   [l ACPR etabli]
  avec une fenetre discriminante (la borne inf rigoureuse tombe
  HORS fenetre : le resultat n'est pas un artefact de robustesse) et
  une deformation de quantification negligeable (+0,12 %%, second
  ordre). CONDITIONNEL aux trois identifications declarees (alpha =
  constante du vide -> quantification sur le tube nu ; m_l(nu) =
  (3/4) sqrt(sigma_chi) ; capture = demi-tour -> m_l(phys) a +0,10 %%).
  La variante exacte (411/(8 pi)) reste la version sharp :
  defavorisee par ridgerunner, non refutee par les bornes -- cible
  mathematique externe nommee.
  SI LES IDENTIFICATIONS TIENNENT : k = 135 (donc alpha = 1/137) et
  m_l sont fixes par LA GEOMETRIE DU TREFLE + le gonflement de la
  charge -- les deux plus gros ouverts du cadre, fermes par la meme
  vision (rails -> capture -> demi-tour -> gonflement).""")


if __name__ == "__main__":
    main()
