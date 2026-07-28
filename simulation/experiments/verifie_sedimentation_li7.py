"""Contre-verification de la sedimentation stellaire du Li-7
(25/07 -- scepticisme utilisateur : « tire par les cheveux »).

VERDICT DE LA RE-VERIFICATION : les nombres de premiers principes
tiennent, mais la presentation d'hier SUR-VENDAIT (« se ferme aux
barres pres »). Les objections REELLES, chiffrees :

 [O1] LE PROBLEME DE LA PLATITUDE (l'objection centrale) : le plateau
      observe est plat a ~0,05 dex sur 500 K de Teff et ~2 dex de
      [Fe/H]. Or la diffusion PURE predit une pente ~0,3-0,4 dex sur
      cette gamme (zones convectives d'epaisseurs tres differentes).
      La « turbulence calibree » doit donc SUPPRIMER la pente d'un
      facteur ~6-8 TOUT EN GARDANT une moyenne de 0,35 dex : ce
      n'est pas une prediction, c'est un REGLAGE -- la platitude est
      IMPOSEE au modele, pas expliquee par lui. L'intuition « tire
      par les cheveux » vise juste ICI.
 [O2] LE MELTDOWN : sous [Fe/H] ~ -2,5, le plateau SE CASSE (dispersion
      0,3-0,5 dex vers le bas, Sbordone+ 2010) -- les modeles de
      diffusion+turbulence ne le predisent pas (ils donnent une
      depletion douce et REGULIERE en metallicite). Un mecanisme qui
      explique la platitude a -2 doit expliquer pourquoi elle meurt
      a -3 : il ne le fait pas.
 [O3] LE RESIDU : requis 0,47 dex ; modeles calibres 0,30-0,40 ->
      il MANQUE 0,07-0,17 dex meme apres le reglage (+0,9 a +2,5
      sigma hier -- j'avais choisi la borne favorable).
 [O4] LES SIGNATURES CONTESTEES : les differences Fe/Mg de NGC 6397
      (Korn+ 2006) dependent de l'echelle de Teff choisie ; les
      re-analyses et les autres amas donnent des amplitudes plus
      faibles ou mitigees. Soutien REEL mais pas decisif.

CE QUI RESTE VRAI (l'autre plateau de la balance) :
 [P1] la destruction stellaire du Li est un FAIT massif : le Soleil
      est appauvri de ~2,2 dex (facteur ~160) -- le mecanisme est
      surpuissant quand le transport atteint les couches chaudes ;
 [P2] le Li interstellaire du Petit Nuage de Magellan (Howk+ 2012)
      sort a ~la valeur BBN : le reservoir NON appauvri est au niveau
      primordial -- fort soutien a « primordial = BBN, deficit = dans
      les etoiles » ;
 [P3] il n'y a PAS d'alternative BBN survivante : notre propre mur
      (~GeV/baryon pour toute injection EM) tue les solutions legeres,
      les injections hadroniques sont etranglees par le D, et les deux
      voies TQNT sont refutees proprement.
"""

# chiffres cles (litterature declaree)
PLAT_SCATTER = 0.05      # dex, dispersion du plateau (5900-6400 K)
PENTE_DIFF = 0.35        # dex, pente Teff de la diffusion pure sur la gamme
DEP_REQ = 0.47           # dex
DEP_MODELES = (0.30, 0.40)
SOLEIL_DEP = 2.2         # dex
SMC = 2.68               # 12+log(Li/H) ISM SMC
BBN = 2.69


def main():
    print(__doc__)
    print("=" * 78)
    print("BILAN CHIFFRE DE LA BALANCE")
    print("=" * 78)
    print("""  CONTRE :
   platitude observee %.2f dex  vs  pente de diffusion pure ~%.2f dex
     -> la turbulence doit supprimer la pente x%.0f en gardant la
        moyenne : REGLAGE, pas prediction ;
   meltdown sous [Fe/H] ~ -2,5 : non predit ;
   residu apres reglage : %.2f-%.2f dex manquants ;
   signatures d'amas : amplitude debattue.
  POUR :
   Soleil : -%.1f dex (le mecanisme est surpuissant, fait etabli) ;
   ISM du SMC : %.2f vs BBN %.2f (reservoir non appauvri = primordial) ;
   aucune alternative BBN survivante (mur ~GeV/baryon, notre theoreme).""" % (
        PLAT_SCATTER, PENTE_DIFF, PENTE_DIFF / PLAT_SCATTER,
        DEP_REQ - DEP_MODELES[1], DEP_REQ - DEP_MODELES[0],
        SOLEIL_DEP, SMC, BBN))

    print("=" * 78)
    print("VERDICT REVISE")
    print("=" * 78)
    print("""  La formulation d'hier (« le probleme se ferme aux barres pres »)
  est RETIREE. Etat honnete :
   - la sedimentation/destruction stellaire est le CANDIDAT PRINCIPAL
     (le Soleil et le SMC pesent lourd, et rien d'autre ne survit) ;
   - mais elle N'EST PAS une fermeture : la platitude du plateau est
     IMPOSEE par un reglage de turbulence (facteur ~7 de suppression
     de pente), le meltdown n'est pas predit, et il manque
     0,07-0,17 dex apres reglage.
  LE VRAI ENONCE DU PROBLEME DU LITHIUM CHANGE : ce n'est plus
  « pourquoi manque-t-il du Li ? » (reponse probable : il a coule et
  brule), c'est « POURQUOI LE PLATEAU EST-IL SI PLAT ET SI FIN si
  chaque etoile a depleta a sa facon ? » -- une question de transport
  stellaire, ouverte, qui n'appelle ni nouvelle physique ni TQNT.
  STATUT AU REGISTRE : Li-7 HERITE ; candidat astrophysique principal
  NON FERME (reglage fin de turbulence non explique) ; aucune
  revendication TQNT ; l'instinct utilisateur (« tire par les
  cheveux ») etait fonde sur le point exact : la platitude.""")


if __name__ == "__main__":
    main()
