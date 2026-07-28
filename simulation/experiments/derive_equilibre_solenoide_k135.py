"""L'equilibre solenoide du tube gonfle : derivation du coefficient c
(25/07 -- maillon (a) du bootstrap k = 135).

TROIS PIECES :
 [A] LA REPONSE ELASTIQUE DU TUBE (rigoureuse dans le modele declare).
     Forme standard d'un tube de flux : E(R) = a/R^2 + b R^2 (coeur
     cinetique + volume de champ), minimum a R0 avec chaque terme
     = sigma_chi/2. Une energie ajoutee VERROUILLEE EN FLUX (donc en
     1/R^2 -- c'est le cas magnetique : « Lk verrouille la
     circulation », etabli) deplace l'equilibre de :
        delta R / R = u_B / (2 sigma_chi).
 [B] L'ENERGIE DE L'ENROULEMENT : LE DEMI-TOUR (une identification
     nouvelle, declaree). La capture est l'enroulement perpendiculaire
     magnetique (etabli, S50) ; l'identification : l'enroulement de
     capture stocke UN DEMI-TOUR du ruban -- LE quantum universel du
     cadre, pi sigma_chi (le meme qui fait l'hyperfin V-P, verifie
     universel a 0,7 %). Sa fraction ELECTROMAGNETIQUE est alpha (le
     portail universel par croisement, etabli) :
        u_B = alpha x pi sigma_chi.
 [C] ASSEMBLAGE : delta R/R = alpha pi/2  ->  c = pi/2 EXACTEMENT.
     Le bootstrap (k+2)^2 - N0 (k+2) + (pi/2) N0 = 0 se ferme, et il
     s'INVERSE en prediction : N0 = (k+2)/(1 - (pi/2) alpha) -- a
     confronter a la mesure du 17/07.
"""
import math

ALPHA = 1 / 137.036
N0_MES = 138.6         # cellules de phase mesurees (17/07, ~1 %)
C = math.pi / 2


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LA REPONSE DU TUBE (modele declare, calcul exact)")
    print("=" * 78)
    print("""  E(R) = a/R^2 + bR^2, min a R0 : a = b R0^4, E(R0) = sigma_chi,
  chaque terme = sigma_chi/2. Ajout u_add en 1/R^2 (flux verrouille) :
     R^4 = (a + a_B)/b -> delta R/R = a_B/(4a) = u_B/(2 sigma_chi).
  (Une energie en R^2 ou en R DEGONFLERAIT : le signe du gonflement
  est fixe par le verrouillage de flux -- la circulation persistante
  se comprime en 1/R^2, elle pousse. C'est le meme verrou que
  l'anneau conducteur de la matiere noire.)""")

    print("=" * 78)
    print("[B] u_B = alpha x pi sigma_chi (l'identification du demi-tour)")
    print("=" * 78)
    print("""  L'enroulement de capture stocke un DEMI-TOUR du ruban : le
  quantum pi sigma_chi -- le MEME objet que l'hyperfin V-P universel
  (verifie a 0,7 % de pi a B_s), que le d = sqrt2 de Koide (ruban
  k=2), que l'excès d-1 des echelles : c'est LE quantum du cadre.
  Sa fraction EM est alpha (portail universel par croisement).
  -> u_B = alpha pi sigma_chi   [1 identification NOUVELLE :
     capture = demi-tour ; les 2 autres pieces sont etablies]""")

    print("=" * 78)
    print("[C] c = pi/2 EXACT, et le bootstrap s'inverse en prediction")
    print("=" * 78)
    dr = ALPHA * math.pi / 2
    print("  delta R/R = alpha pi/2 = %.5f  ->  c = pi/2 (exact)" % dr)
    # point fixe
    disc = N0_MES**2 - 4 * C * N0_MES
    x = (N0_MES + math.sqrt(disc)) / 2
    print("  point fixe avec N0 mesure (138,6) : k+2 = %.2f -> k = %d" % (
        x, round(x) - 2))
    # inversion : prediction de N0
    n0_pred = 137.0 / (1 - C * ALPHA)
    print("""  INVERSION (la lecture la plus propre) : k = 135 etant fixe par
  alpha, la chaine PREDIT la mesure de cellules :
     N0 = (k+2)/(1 - (pi/2) alpha) = %.2f
     mesure (17/07)               = %.1f   (ecart %+.2f %%)
  -> la mesure du 17/07, ecartee alors comme coincidence, devient une
  VERIFICATION a +0,01 %% de la chaine gonflement (dans sa barre ~1 %% :
  l'accord fin est partiellement chanceux, DECLARE -- mais le central
  tombe pile).""" % (n0_pred, N0_MES, 100 * (N0_MES - n0_pred) / n0_pred))

    print("=" * 78)
    print("LE SAUT DE 138, DISSOUS")
    print("=" * 78)
    print("""  L'objection « 138,6 -> 137 enjambe 138 » supposait une DESCENTE.
  Le bootstrap n'est pas une descente : c'est un POINT FIXE -- le
  systeme auto-coherent (le gonflement depend d'alpha qui depend du
  comptage) n'a qu'UNE solution physique, 137,01, et le cran entier
  verrouille 137. L'entier 138 n'est jamais un etat du systeme :
  il n'est pas auto-coherent.""")

    print("=" * 78)
    print("BILAN ET STATUTS")
    print("=" * 78)
    print("""  c = pi/2 est DERIVE au rang des identifications canoniques :
   - reponse du tube : EXACTE dans le modele declare (E = a/R^2+bR^2,
     ajout verrouille en flux) ;
   - u_B = alpha pi sigma_chi : UNE identification nouvelle (capture =
     demi-tour du ruban) + deux pieces etablies (quantum pi sigma_chi ;
     portail alpha) -- le meme rang que les theoremes conditionnels du
     cadre (boucle <-> demi-tour) ;
   - bootstrap : k = 135 SORT, et la chaine PREDIT N0 = 138,59
     (mesure 138,6).
  RESTE : (i) resserrer N0 (~1 % -> l'accord fin deviendrait un test
  reel) ; (ii) consolider l'identification capture = demi-tour (lien
  au couple helicoide +-4,59 de S50 : cible nommee) ; (iii) le modele
  de tube a/R^2 + bR^2 est declare (standard, mais un autre profil
  changerait le 1/2 de la reponse -- pas le pi).
  SI CELA TIENT : k = 135, alpha = 1/137 ET D_tube/m_l recoivent UN
  SEUL mecanisme -- le gonflement magnetique de la capture -- et
  alpha est le point fixe de l'auto-coherence de la matiere.""")


if __name__ == "__main__":
    main()
