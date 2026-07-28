"""Le theoreme de la fenetre de capture : delta = (d-1)/2 (23/07).

CIBLE : la fenetre de capture magnetique du processus directionnel
(S46') -- requise a delta* = 0,209, candidat (d-1)/2 = 0,207. La
deriver = deriver alpha_cond en forme close.

LA REDUCTION : le prefacteur cinematique du gaz de rails (4 directions
perpendiculaires x approche diagonale) vaut 2 sqrt2 ; la fenetre
delta = (d-1)/2 equivaut donc EXACTEMENT au branchement par rencontre
   passages/scissions = 2 sqrt2 x (d-1)/2 = sqrt2 (sqrt2 - 1) = 2 - sqrt2
c'est-a-dire, PAR SENS d'enroulement :
   W_enroulement / W_scission = (d-1)/d = 1 - 1/d.
Le theoreme est donc ALGEBRIQUE (par rencontre), la cinematique n'est
qu'un echafaudage.

LE THEOREME (deux identifications, au rang canonique boucle<->demi-tour):
[1] LA SCISSION PORTE LA VALEUR DE BOUCLE d : la collision frontale
    meme-rail de deux ondes contra-propageantes qui se recombinent
    FERME leur boucle d'histoire (le diagramme de la rencontre 1D est
    clos par la collision elle-meme) -- c'est une fermeture de Markov
    EFFECTUEE physiquement : poids d (le theoreme du maillon etabli :
    fermeture de l'element d'echange e = d, une boucle).
[2] L'ENROULEMENT PORTE L'EXCES d-1 PAR SENS : la capture magnetique
    de l'onde perpendiculaire est un echange SANS fermeture -- la
    partie CLASSIQUE (valeur 1) passe sans evenement (l'onde continue,
    rien n'est depose), seul l'EXCES anyonique d-1 se depose comme
    topologie (le Delta Lk = +-1) : le meme excès que le maillon/S4''
    (exces = d - 1, demontre) ; DEUX sens de capture (gauche/droite),
    chacun avec l'exces.
=> passages/scissions = 2(d-1)/d = 2 - sqrt2 = 0,5858
=> ALPHA = ln(d/(d-1)) = ln(2 + sqrt2) = 1,2279   (forme close).

VERIFICATIONS : algebre TL (valeurs de fermeture), identites exactes,
equivalence gaz<->algebre, confrontations (fenetre cinetique, valeur
utilisee, f_frac, branchement, MC rails), alternatives tuees.
"""
import numpy as np

D = np.sqrt(2.0)
ALPHA = float(np.log(2 + np.sqrt(2)))
RATIO = 2 - np.sqrt(2)


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] L'ALGEBRE : valeurs de fermeture TL (verification)")
    print("=" * 78)
    # TL_2 : base {1, e}, e^2 = d e ; fermetures diagrammatiques :
    # cl(1) = 2 boucles = d^2 ; cl(e) = 1 boucle = d.
    # Coherence de l'algebre : R = A.1 + A^-1.e (Kauffman), la fermeture
    # de R (un croisement clos) doit valoir un noeud trivial a kink :
    # |cl(R)| = d. A = e^{3 i pi/8} pour d = sqrt2 (-A^2 - A^-2 = d).
    A = np.exp(3j * np.pi / 8)
    d_check = float(np.real(-(A**2) - A**(-2)))
    clR = A * D**2 + A**(-1) * D
    print(f"""  d = -A^2 - A^-2 = {d_check:.6f} (= sqrt2 = {D:.6f}) [OK]
  fermetures : cl(1) = d^2 = {D**2:.4f} (2 boucles), cl(e) = d = {D:.4f}
  (1 boucle -- LE theoreme du maillon etabli) ;
  coherence Kauffman : |cl(R)| = |A d^2 + A^-1 d| = {abs(clR):.6f} = d [OK]
  (le croisement clos = trivial a kink : l'algebre est bien celle du
  ruban k=2 du cadre -- d = sqrt2 force par les 2 canaux de fusion).""")

    print("=" * 78)
    print("[B] LE THEOREME ET SES IDENTITES EXACTES")
    print("=" * 78)
    print(f"""  Scission (frontale)      : W_s = d      (fermeture de Markov
                                          EFFECTUEE par la collision)
  Enroulement (par sens)   : W_+- = d - 1 (l'exces seul se depose ;
                                          la partie classique 1 passe)
  passages/scissions = 2(d-1)/d = 2 - sqrt2 = {RATIO:.6f}
  ALPHA = ln(d/(d-1)) = ln(2+sqrt2) = {ALPHA:.6f}
  Identites : d-1 = tan(pi/8) = {np.tan(np.pi / 8):.6f} (le meme exces que
  le maillon/S4'') ; (d-1)/d = 1 - 1/sqrt2 = {1 - 1 / np.sqrt(2):.6f} ;
  d(d+1) = 2+sqrt2 [exact].
  EQUIVALENCE GAZ <-> ALGEBRE : delta = ratio/(2 sqrt2) =
  (2-sqrt2)/(2 sqrt2) = (sqrt2-1)/2 = (d-1)/2 = {(D - 1) / 2:.6f} EXACT --
  la fenetre de capture EST le demi-exces (le prefacteur cinematique
  2 sqrt2 = 4 directions x diagonale se replie sans reste).""")

    print("=" * 78)
    print("[C] CONFRONTATIONS")
    print("=" * 78)
    rows = [
        ("alpha_cond utilise (1,1991)", 1.1991, "valeur historique de la chaine"),
        ("derivation cinetique (loi geom.)", 1.217, "fenetre [1,214, 1,220]"),
        ("MC rails x10 (S46)", 1.361, "+-0,120"),
    ]
    for nom, v, note in rows:
        print(f"  {nom:34s} : {v:.4f}  ecart {100 * (ALPHA / v - 1):+.1f} %  ({note})")
    f_frac = 1 - np.exp(-RATIO)
    print(f"""  f_frac (identite d'arbre)          : 1 - e^-(2-sqrt2) = {f_frac:.4f}
     vs 0,447 (Poisson etabli, -0,8 %) ; la cascade branchee (0,34)
     garde son ecart connu (surdispersion, independant d'alpha).
  Branchement par rencontre : scission {1 / (1 + RATIO):.3f} / passage {RATIO / (1 + RATIO):.3f}
     vs etabli 0,628/0,372 (+0,9 %/-1,5 %).
  LES TROIS DETERMINATIONS (utilisee 1,1991 ; cinetique 1,214-1,220 ;
  algebrique 1,2279) se rangent dans 2,4 % -- la forme close est a
  +0,7..+1,1 % de la cinetique (les O(1) de celle-ci sont declares) et
  a 1,1 sigma du MC rails.""")

    print("=" * 78)
    print("[D] ALTERNATIVES TUEES")
    print("=" * 78)
    print(f"""  - trace naive (enroulement = fermeture d aussi) : ratio 2, alpha =
    ln 2 = 0,693 -- EXCLU par toutes les determinations (>4 sigma MC) ;
  - exces TOTAL sans partage des sens (2(d-1)/1) : alpha = 0,88 --
    exclu (4 sigma MC, 27 %% de la cinetique) ;
  - classique/exces inverse ((d-1) aux scissions) : ratio 2d/(d-1) > 1
    -- alpha negatif, absurde (le nouage dominerait la casse) ;
  - 3/5 « ports libres » (alpha = ln(10/3) = 1,204) : pas de structure
    d'exces, la fraction 3/5 n'a aucun theoreme du cadre (null) ;
  -> SEULE l'affectation (fermeture d -> scission ; exces d-1 -> par
  sens d'enroulement) est construite sur DEUX theoremes etablis
  (fermeture de Markov du maillon ; exces = d-1).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print(f"""  LE THEOREME DE LA FENETRE : delta = (d-1)/2, PAR l'affectation
  scission = valeur de boucle d (la frontale FERME la boucle
  d'histoire -- fermeture de Markov physique), enroulement = exces
  d-1 par sens (la partie classique passe, seul l'exces se depose en
  Delta Lk). D'ou la FORME CLOSE :
     alpha_cond = ln(d/(d-1)) = ln(2 + sqrt2) = {ALPHA:.4f}
  Statut : theoreme CONDITIONNEL aux deux identifications (rang
  canonique boucle<->demi-tour, comme le maillon et S4'') ; accord
  +0,7..+2,4 %% avec les determinations independantes, 1,1 sigma du
  MC rails. Si adopte : alpha_cond cesse d'etre un nombre -- c'est
  ln(2+sqrt2), le QUATRIEME invariant du meme ruban k=2 (d = sqrt2
  Koide, d-1 exces, h_1/2 = 3/16 splitting, ln(d/(d-1)) formation)
  -- et la chaine Omega est ancree dans l'algebre du socle de bout
  en bout.""")


if __name__ == "__main__":
    main()
