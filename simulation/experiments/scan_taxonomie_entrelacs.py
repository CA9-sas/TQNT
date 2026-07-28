"""Taxonomie complete des entrelacs sous les regles du document principal (22/07).

QUESTION : avec toutes les regles (echelle, monotonie, etat/transport,
cable, verrou/bascule, saturation, Vassiliev, c_min, Q=Sigma_chi+Lk,
stabilite P15), peut-on determiner les entrelacs POSSIBLES et
DETECTABLES ? Y a-t-il des configurations SANS categorie ? De NOUVELLES
categories ? Des configurations INTERDITES ?

Quatre volets : [A] la table des categories (peuplee) ; [B] les
familles sans categorie (3 identifiees) ; [C] les categories nouvelles
envisageables ; [D] les interdits (10 theoremes d'impossibilite).
"""


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] LA TABLE DES CATEGORIES (l'espace est presque partitionne)")
    print("=" * 78)
    print("""  1. BASE (c=0)          : ondes chi libres -- photon, neutrino (pas
     de topologie ; steriles INTERDITS). Detection : directe.
  2. NOEUDS VISIBLES (3 <= c <= ~10, charges/couples) :
     - leptons = modes Z3 du trefle (exactement 3 : e, mu, tau) ;
     - noeuds constituants : 3_1 (l), 5_2 (s), c, b (identite
       geometrique = mur de transmutation) ;
     - toriques 5_1/7_1/9_1 : composants EW/marqueurs.
  3. COMPOSITES VISIBLES (les 5 sous-secteurs de masse) :
     mesons (K+Kbar, coeur 4_1) ; baryons (jonction Y, 3 jambes,
     cable) ; EW (non confines, quartet 8_3) ; molecules (hadrons
     socle + demi-tours) ; compression (quarkonia, B_c, tau<0).
     Detection : spectroscopie -- 89+226 etats scores.
  4. DESERT DES ENTRELACS (12-80 GeV, 617 configs) : TRANSITOIRES
     (cascade P15 ~1e-18 s). Detection : aucune (largeurs enormes).
  5. SOMBRE (c >= 23) :
     - reservoir : noeuds premiers ISOLES T(2,q>=23)/T(3,q>=12),
       bord 1,7-2,2 GeV (detection : gravitationnelle seule,
       sigma_eq ~1e-48 -- absence LZ/XENONnT/DS-20k PREDITE) ;
     - 6 STRICTS amphichiraux (1,07-1,64 GeV) : raies Fermi-LAT
       1,10/1,64 (la fenetre) + 2 extensions EW-neutres 1,75/1,89 ;
     - 111 REALISABLES (Q=0 par Lk entier, ~1-3 GeV) : transitoires
       ou contextuels (P22) ; absence collisionneur PREDITE (alpha^dc).
  -> sur les 1539 entrelacs du catalogue : ~30 realises visibles,
  117 sombres (6+111), 617 desert, le reste = charges instables
  (cascade vers les planchers e/p). CHAQUE population a sa categorie
  ET son statut de detection.""")

    print("=" * 78)
    print("[B] LES FAMILLES SANS CATEGORIE (3 identifiees)")
    print("=" * 78)
    print("""  B1. HYBRIDES visible-sombre (noeud c<18 enlace a un noeud c>=23) :
      aucune categorie canonique. Creation supprimee (alpha^dc), et le
      theoreme d'instabilite (entrelacs noirs n>=2) ne couvre que le
      sombre-sombre. S'ils se formaient (primordial), le membre
      visible interagit -> le lien casse (P15) : proposition =
      TRANSITOIRES DE FRONTIERE (portails), population primordiale
      residuelle a estimer.
  B2. COMPOSES A DOUBLE JONCTION (2 jonctions Y : tetra/pentaquarks
      COMPACTS, vs les molecules du document principal) : le document principal lit TOUS les
      exotiques LHC comme molecules (mediane 0,3 %) -- la famille
      compacte est INOCCUPEE. Le document principal nomme deja son premier membre
      predit : les DIBARYONS LOURDS BASE 2 (deux trialites). Categorie
      declaree mais non peuplee : la distinguer des molecules =
      largeurs + base 2 (torsion doublee).
  B3. SOMMES CONNEXES (noeuds composes K1#K2, non premiers) : le
      catalogue est en noeuds PREMIERS ; les sommes connexes n'ont pas
      de categorie. P15 les fissionne (reconnexion au col) :
      proposition = RESONANCES DE FISSION (transitoires, largeur =
      taux P15 au col) -- a categoriser.""")

    print("=" * 78)
    print("[C] CATEGORIES NOUVELLES ENVISAGEABLES")
    print("=" * 78)
    print("""  C1. Dibaryons base-2 (predits, base derivee) : H-dibaryon lourd
      TQNT -- signature : torsion de base doublee, canaux Lambda_c
      Lambda_c / Xi_cc N ; LHCb/Belle II peuvent chercher.
  C2. Portails de frontiere (B1) : uniquement pertinents en
      cosmologie (P22 aux frontieres matiere/vide) -- pas de
      signature collisionneur.
  C3. Resonances de fission (B3) : largeur P15 calculable -- si un
      jour un etat tres large sans case (type D0*(2300) historique)
      resiste aux lectures moleculaires, tester la somme connexe.
  C4. PAS de categorie nouvelle pour les XYZ observes : la lecture
      moleculaire couvre tout (0,3 %) -- razor du document principal.""")

    print("=" * 78)
    print("[D] LES INTERDITS : 10 THEOREMES D'IMPOSSIBILITE")
    print("=" * 78)
    print("""  D1. Lambda_ss (paire ss spin 0) : INTERDIT (verrou Pauli) --
      confirme PDG (aucun Lambda etrange-etrange).
  D2. 4e GENERATION : INTERDITE (saturation Z3, derivee) -- jamais de
      hadrons g=3, aucune 4e marche.
  D3. NEUTRINOS STERILES : INTERDITS (le neutrino n'a aucune
      topologie) -- toute confirmation refuterait.
  D4. LA BANDE INTERDITE c = 19-22 : un noeud charge a c>18 serait
      invisible-mais-charge (contradiction Vassiliev) ; un noeud a
      c<23 ne survit pas cosmologiquement (couplage au plasma,
      N_coll>1). AUCUNE population stable entre 19 et 22 croisements :
      le desert topologique EST une prediction.
  D5. Omega_bbb au cadre torsion : INTERDIT (compression seule,
      liaison negative -- signe deja verifie par le reseau).
  D6. BARREAUX NEGATIFS : INTERDITS (monotonie de l'echelle) -- aucun
      etat sous le sol d'alignement, aucune bande a X<2, aucun
      alignement -1/2 : les niveaux interdits du spectre.
  D7. ENTRELACS NOIRS n>=2 (hors les 6 stricts) : INSTABLES a
      l'echelle cosmologique (theoreme) -- pas de molecules sombres.
  D8. CHARGE NON ENTIERE : INTERDITE (fluxoide, Q = Sigma_chi + Lk
      entier) -- pas de fractionnaires libres.
  D9. NOEUD DE NEUTRINO / 4e MODE Z3 DU TREFLE : INTERDITS (les modes
      propres du trefle sont exactement 3).
  D10. ETATS TQNT-SEULS AU COLLISIONNEUR : INTERDITS en pratique
      (creation alpha^dc ~1e-11..1e-24) -- l'absence LHCb est une
      prediction, la fenetre est astro.
  Chaque interdit est FALSIFIABLE : une seule observation contraire
  (un Lambda_ss, une 4e famille, un sterile, un noeud stable c=20,
  un etat sous le sol...) refuterait la regle correspondante.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  OUI, l'espace des entrelacs est determinable : la taxonomie couvre
  ~98 % du catalogue (categories + statuts de detection), il reste
  TROIS familles sans categorie (hybrides de frontiere, doubles
  jonctions -- dont les dibaryons base-2 deja predits --, sommes
  connexes), TROIS categories nouvelles proposees pour les accueillir,
  et DIX interdits falsifiables qui delimitent l'existable. La
  prochaine etape naturelle : peupler B2 (le dibaryon base-2, seule
  categorie nouvelle avec une signature collisionneur).""")


if __name__ == "__main__":
    main()
