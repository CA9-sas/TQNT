"""k = 135 par le gonflement magnetique du tube (25/07 -- vision
utilisateur : « le passage dans les rails charge magnetiquement le
tube de flux, le gonfle, et produit la matiere topologique »).

LE POINT D'APPUI EXISTANT : le candidat « cellule de phase »
(mecanisme_flux_cs.py, 2026-07-17) comptait les cellules de phase
autour du tube : N0 = 2 pi lambda_sigma / D = 138,6 -- contre
k+2 = 137 requis. Il fut ecarte (« coincidence, legerement
defavorisee ») pour DEUX manques : (a) aucun mecanisme d'ajustement
(pourquoi N bougerait ?), (b) aucune raison d'etre ENTIER.
LA VISION FOURNIT LES DEUX :

 [K1] LE MECANISME D'AJUSTEMENT : l'entree dans le rail cree le
      courant persistant ENROULE (la capture magnetique etablie,
      « enroulement perpendiculaire ») ; ce courant charge le tube
      comme un solenoide -> la pression magnetique GONFLE le tube
      jusqu'a l'equilibre avec la tension sigma_chi. Le diametre
      physique D est donc DYNAMIQUE : D(charge) > D(nu). La matiere
      topologique est un tube magnetiquement gonfle -- et du meme
      coup l'echelle de transmutation D_tube (la cible m_l !)
      recoit un candidat de mecanisme.
 [K2] LA RAISON D'ETRE ENTIER : la phase du courant enroule doit se
      FERMER univoquement autour du tube -> le nombre de cellules de
      phase sur la circonference est QUANTIFIE : N entier. Le
      gonflement fournit le degre de liberte, la fermeture fournit le
      cran : le tube gonfle jusqu'au verrouillage entier.
 [K3] L'AUTO-COHERENCE (le bootstrap) : le gonflement est d'ordre
      alpha (energie EM d'une charge unite / tension) -- mais alpha
      EST 1/(k+2), fixe par le meme comptage. Le point fixe :
         k+2 = N0 (1 - c alpha),  alpha = 1/(k+2)
      -> (k+2)^2 - N0 (k+2) + c N0 = 0.
      L'AMPLEUR REQUISE : (138,6-137)/138,6 = 1,154 % = 1,58 alpha --
      exactement l'ORDRE d'un gonflement de charge unite. Le
      coefficient O(1) : c* = 1,582 ~ pi/2 (a +0,7 %) -- NOTE SANS
      MECANISME (regle du test nul), cible de derivation nommee.
"""
import math

N0 = 138.6            # cellules de phase mesurees (tube NU, 17/07)
K_TARGET = 135
ALPHA = 1 / 137.036


def main():
    print(__doc__)
    print("=" * 78)
    print("[K3] LE BOOTSTRAP, CHIFFRE")
    print("=" * 78)
    shift = (N0 - 137.0) / N0
    c_star = shift / ALPHA
    print("""  decalage requis : (N0 - 137)/N0 = %.3f %% = %.3f alpha
  -> l'ordre du gonflement magnetique d'une charge unite (O(alpha)) :
     LE MECANISME A LA BONNE TAILLE. Coefficient requis c* = %.3f
     (pi/2 = %.4f : +0,7 %% -- COINCIDENCE NOTEE, pas retenue sans
     mecanisme).""" % (100 * shift, shift / ALPHA, c_star, math.pi / 2))
    # point fixe avec c = pi/2
    c = math.pi / 2
    disc = N0**2 - 4 * c * N0
    x = (N0 + math.sqrt(disc)) / 2
    print("""  POINT FIXE avec c = pi/2 pose : (k+2)^2 - N0(k+2) + c N0 = 0
     -> k+2 = %.2f  (la racine physique ; l'autre, %.2f, est le
        regime sur-gonfle, rejete)
     -> verrouillage entier : k+2 = 137, k = %d.
  Si c est derive (et vaut ~pi/2), k = 135 SORT du gonflement.""" % (
        x, (N0 - math.sqrt(disc)) / 2, K_TARGET))

    print("=" * 78)
    print("CE QUE LA VISION UNIFIE (si elle tient)")
    print("=" * 78)
    print("""  1. k = 135 : compte des cellules de phase du tube GONFLE,
     verrouille entier par la fermeture du courant -- l'entier le
     plus mysterieux du cadre recevrait un mecanisme ;
  2. D_tube (et donc m_l, la transmutation) : le diametre serait
     l'equilibre pression magnetique / tension sigma_chi -- la cible
     « deriver le rayon lambda_sigma » recevrait le meme mecanisme ;
  3. la boucle se ferme ontologiquement : rails -> magnetisation
     (etabli) -> gonflement -> tube physique -> cellules de phase ->
     alpha -> qui fixe la force du gonflement : alpha est le POINT
     FIXE de l'auto-coherence de la matiere topologique.
  4. coherences etablies : l'anneau conducteur de la DM (courant de
     rail verrouille) = le meme courant ; la capture par enroulement
     perpendiculaire = la source de la charge ; la refutation
     historique du flux CS energetique (R ~ sqrt(k), mauvais sens)
     n'est PAS contredite -- ici R ne suit pas k, il suit l'equilibre
     magnetique, et k COMPTE des cellules.""")

    print("=" * 78)
    print("LES DEUX MAILLONS A DERIVER (cibles nommees) -- ET LES RISQUES")
    print("=" * 78)
    print("""  (a) le coefficient c : l'equilibre solenoide (pression B^2/8pi
      du courant de capture enroule vs tension du tube) doit donner
      le gonflement relatif d'une charge unite -- c'est un calcul
      d'electrodynamique de tube, pas une inconnue de principe ;
      s'il donne c ~ pi/2, le bootstrap ferme ; sinon, refutation
      propre ;
  (b) la barre d'erreur de N0 = 138,6 : la mesure de 17/07 portait
      ~1 %% d'incertitude (pas de 0,72 %% sur m_l) -- il faut la
      resserrer pour que le decalage de 1,15 %% soit significatif ;
      si N0 = 137,0 +- 0,5 nu, le gonflement n'a rien a faire et la
      cellule de phase revient seule.
  RISQUES DECLARES : c = pi/2 est de la numerologie tant que (a)
  n'est pas fait ; le verrouillage entier suppose la fermeture de
  phase demontree sur le tube gonfle ; et 138,6 -> 137 saute DEUX
  entiers (138 en route) -- le cran de verrouillage doit selectionner
  137, pas 138 : la condition de fermeture doit etre sur k+2 (135+2 :
  2 cellules pour les extremites/framing ?) ou le gonflement doit
  depasser le premier cran -- a trancher par (a).
  STATUT : hypothese mecanisante FORTE (bonne taille, bon ordre,
  unifie k et D_tube), brouillon, deux cibles calculables.""")


if __name__ == "__main__":
    main()
