"""La vitesse de la lumiere a-t-elle evolue ? Statut, verrou, nuls (20/07).

[1] LE STATUT DU TRIO (ontologie du socle, V6) :
    h    = la dent (graduation en action)      -> INVARIANT ;
    c    = l'engrenement du socle              -> PREEXISTE avec le
           socle (niveau alpha^0 : c'est la structure causale du
           champ chi lui-meme -- toute equation du cadre l'a dans son
           terme cinetique ; AUCUN degre de liberte du cadre ne peut
           le faire varier) ;
    M_Pl = l'horloge/rigidite                  -> PREEXISTE (ancre) ;
    G    = l'amplification (besoin de puits)   -> NAIT en une marche.
    => c n'a PAS de marche : contrairement a G, il ne depend de rien
    qui se forme. Un c(t) exigerait un changement de dispersion du
    socle -- qui definirait un referentiel a la transition (ether),
    exclu par la graduation en action (S3).

[2] LE VERROU TOPOLOGIQUE (propre a la TQNT) :
    alpha_EM = 1/(k+2), k = 135 ENTIER topologique -> alpha ne peut
    pas deriver continument (il ne pourrait que SAUTER). Or alpha
    contient c : une derive de c a alpha fixe exigerait une co-derive
    de e^2/hbar -- h etant la dent invariante, c est verrouille par
    l'entier k. Confrontation aux mesures de derive d'alpha :
      - horloges atomiques : |d alpha/alpha| < ~1e-17 / an ;
      - Oklo (2 Gyr)       : |D alpha/alpha| < ~1e-7 ;
      - quasars (z~1-3)    : |D alpha/alpha| < ~1e-5 ;
      - CMB/BBN (z 1e3/1e9): ~% .
    ZERO derive observee -- coherent avec l'entier topologique.

[3] LES NULS DIRECTS SUR c(t) :
    - sursauts gamma multi-energies (z jusqu'a ~8) : arrivees
      simultanees -> pas de variation de c le long de ~Gyr de
      trajet (les memes bornes E_QG > 9,3 E_P) ;
    - BBN reussie (t ~ 1-10 min) aux taux standard : c(t_BBN) = c
      au niveau du % (les taux faibles/EM en dependent) ;
    - le socle predit l'exactitude de l'engrenement A TOUTE EPOQUE
      (v = lambda f = c par mode) : les nuls sont des predictions.

[4] AUDIT DE LA CHAINE OMEGA : ou c entre-t-il ?
    Partout comme STRUCTURE (unites naturelles) et nulle part comme
    levier : les criteres sont des rapports adimensionnes (E/T,
    Gamma/H, xi.T, S_d/Z) -- un rescaling GLOBAL de c se reabsorbe
    dans les unites ; seul un c DIFFERENTIEL (photons vs chi)
    laisserait une trace -> c'est la dispersion, nulle (predite et
    observee). Impact sur Omega : NUL par adimensionnement.
"""


def run():
    print(__doc__)
    print("BILAN : c n'a pas evolue -- par STATUT (constante du socle,")
    print("comme h : l'engrenement preexiste, pas de marche), par VERROU")
    print("(alpha = 1/(k+2), k entier topologique : pas de derive")
    print("continue possible ; zero derive observee), et par NULS directs")
    print("(GRB, BBN). Le trio final : h et c = constantes du SOCLE")
    print("(dent, engrenement -- eternelles) ; M_Pl = horloge (ancre,")
    print("preexiste) ; G = couplage de la MATIERE (une marche, a la")
    print("condensation). Impact sur Omega : nul par adimensionnement.")


if __name__ == "__main__":
    run()
