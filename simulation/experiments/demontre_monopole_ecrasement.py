"""Le maillon dur de l'hypothese de l'ecrasement : la demonstration
monopole-seule -- et son verdict honnete (24/07).

QUESTION : l'amplification x3 de la reponse de H a la densite locale
(T2-T3) peut-elle etre un effet de monopole sans toucher l'infall par
paires (RSD) ? L'ANALYSE TRANCHE, EN TROIS COUPS -- et le verdict
n'est pas celui qu'on esperait :

[1] LE SIGNE. Le sourcage instantane local (H^2 ~ rho du volume,
    v = sqrt(2GM(R)/R) applique region par region) donne pour un
    VIDE (delta < 0) : M plus petit -> v PLUS PETIT -> le vide
    s'etend PLUS LENTEMENT. Or la relativite generale etablit (et
    l'observation exige) l'inverse : un vide DECELERE MOINS et
    s'etend plus vite a temps tardif. La version instantanee du
    sourcage predit le MAUVAIS SIGNE pour la tension H0.

[2] LA STRUCTURE. Imposer la trajectoire parabolique (energie nulle)
    region par region GELE la formation des structures : une region
    sur-dense recevrait PLUS de vitesse (M plus grand -> v plus
    grand) et s'echapperait -- les surdensites ne s'effondreraient
    jamais, les vides ne se creuseraient jamais. Un univers
    partout-parabolique reste homogene : contradiction avec
    l'existence meme du vide KBC que T3 invoquait.

[3] LA RESTAURATION. Pour retrouver le bon signe, il faut que la
    poussee porte la MEMOIRE de l'histoire (le vide a decelere moins
    depuis toujours) -- c'est exactement Birkhoff : chaque region
    suit v^2 = 2GM/R + 2E avec E fixe par l'histoire. Mais alors la
    reponse de H a delta redevient EXACTEMENT celle de la relativite
    generale (Delta H/H = -f delta/3) : L'AMPLIFICATION x3 MEURT.

CE QUI SURVIT : T1, au niveau GLOBAL -- l'univers ENTIER est la
region sans exterieur, la seule ou l'echange d'energie de bord est
nul : la condition parabolique globale v = sqrt(2GM/R) est le bilan
exact de la poussee -> Friedmann plat + Omega_tot = 1 en theoreme.
Les perturbations chevauchent ce fond exactement comme en RG.
"""
import math

OM = 0.31
F = OM ** 0.55
H_CMB, H_LOC = 67.4, 73.0
DELTA_KBC = -0.30


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] LE TEST DU SIGNE (chiffre)")
    print("=" * 78)
    d = DELTA_KBC
    h_inst = math.sqrt(1 + d) - 1          # sourcage instantane
    h_gr = -F * d / 3                      # RG lineaire
    print("""  vide delta = %.2f :
     sourcage instantane : Delta H/H = sqrt(1+delta)-1 = %+.1f %%
     relativite generale : Delta H/H = -f delta/3      = %+.1f %%
     tension observee    :                              +8,3 %%
  -> le sourcage instantane donne un vide qui s'etend PLUS LENTEMENT
  (H local plus BAS) : signe OPPOSE a la tension. Le calcul T2-T3
  du premier passage portait une erreur de signe (H ~ sqrt(rho)
  repond POSITIVEMENT a la densite) -- consigne.""" % (
        d, 100 * h_inst, 100 * h_gr))

    print("=" * 78)
    print("[2] LE TEST DE STRUCTURE")
    print("=" * 78)
    print("""  Parabolique partout = marginalement lie partout :
     region sur-dense (delta > 0) : M plus grand -> v = sqrt(2GM/R)
     plus GRAND -> la surdensite recoit la vitesse d'echappement
     exacte -> elle ne s'effondre JAMAIS ;
     region sous-dense : v plus petit -> le vide ne se creuse jamais.
  Un univers partout-parabolique n'a NI galaxies NI vide KBC : la
  version locale-instantanee se contredit elle-meme (elle invoque un
  vide qu'elle interdit de former).""")

    print("=" * 78)
    print("[3] LA RESTAURATION = BIRKHOFF = RG EXACTEMENT")
    print("=" * 78)
    print("""  Le bon signe exige la memoire : v^2 = 2GM/R + 2E(histoire), E > 0
  pour un vide (il a toujours moins decelere). Mais des que E est la,
  la reponse lineaire de H a delta est LA REPONSE RG : -f delta/3.
  L'amplification x2,9 n'existe dans AUCUNE version coherente :
   - instantanee : mauvais signe + gele les structures ;
   - avec memoire : identique a la RG (et c'est heureux : RSD,
     lentillage et croissance sont RG a ~10 %).
  Le point 4 du brouillon (le risque declare) etait le bon instinct :
  il n'y a PAS de demonstration monopole-seule, parce qu'il n'y a
  pas d'amplification a demontrer.""")

    print("=" * 78)
    print("CE QUI SURVIT -- ET C'EST SUBSTANTIEL")
    print("=" * 78)
    print("""  [T1 GLOBAL] L'univers entier est la SEULE region sans exterieur :
  aucun echange d'energie de bord -> la condition parabolique
  s'applique EXACTEMENT la, et la seulement :
     v(R) = sqrt(2GM(R)/R) = HR,  H^2 = 8 pi G rho/3,  Omega_tot = 1.
  L'hypothese de l'ecrasement MECANISE Friedmann et la platitude :
  l'expansion est la poussee de la trame qui paie les puits -- plus
  une condition initiale ; Omega_tot = 1 est un theoreme (toute
  courbure detectee refuterait). Les perturbations chevauchent ce
  fond comme en RG (Birkhoff) : ZERO conflit avec RSD/lentillage.
  [H0/S8] La position d'elimination du brouillon est RESTAUREE :
  H0 = 67,4 global ; le 73 local redevient systematique OU vide reel
  traite en RG standard (-f delta/3 : le vide KBC a -0,3 donnerait
  +5,3 %, une part substantielle de la tension -- mais c'est de la
  RG, pas de la TQNT ; et l'existence du vide a cette profondeur est
  elle-meme une tension LCDM). Sirenes locales : retour a la
  prediction 67,4 (a la part RG du vide pres).
  VERDICT : hypothese TRANCHEE honnetement -- l'amplification est
  REFUTEE (3 coups), la mecanisation de Friedmann+platitude est le
  produit positif durable. A ne JAMAIS reintroduire : le sourcage
  local instantane de H (mauvais signe + gel des structures).""")


if __name__ == "__main__":
    main()
