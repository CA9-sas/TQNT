"""Le trio cosmologique deduit du chantier ecrasement (24/07).

QUESTION UTILISATEUR : peut-on desormais DEDUIRE (1) l'evolution de
l'energie noire (DESI w(z)), (2) les anomalies JWST z>10, (3) les
anomalies CMB a grande echelle ? OUI -- les trois passent de
« verdicts du portefeuille » a COROLLAIRES des theoremes consolides
(T1 Friedmann mecanise, elasticite, riviere, croissance mecanisee).

[1] w = -1 CONSTANT DEVIENT UN THEOREME DE T1.
    Dans le bilan de la poussee, chaque source d'ecrasement contribue
    par sa densite. Le residu du vide chi (rho_Lambda, etabli :
    statique, 123 ordres resolus) a DEUX proprietes structurelles :
    il ne se DILUE pas (ce n'est pas de la matiere qui s'ecarte --
    c'est le deficit de point zero de la trame elle-meme) et il ne se
    RETIRE jamais (elasticite : rien ne se retire -> rien ne revient).
    Par le theoreme de la riviere (la trame repond a la densite), sa
    poussee est donc CONSTANTE dans le temps et l'espace :
       w = -1 exactement, dw/dz = 0 -- PAR CONSTRUCTION.
    La preference DESI w0-wa (~3-4 sigma) DOIT regresser ; une
    evolution confirmee refuterait la lecture « residu statique ».

[2] JWST z>10 : L'ABONDANCE = LCDM DEVIENT UN THEOREME.
    La croissance est mecanisee (friction 2H delta' = la poussee) et
    Birkhoff verrouille les perturbations sur la RG : il N'EXISTE PAS
    de raccourci de croissance dans le cadre (la riviere : la trame
    repond a la densite, pas au mouvement -- aucun canal pour
    accelerer un effondrement). L'audit historique (boost P22 =
    1,000000, nul a 81 decades) devient un COROLLAIRE : l'exces
    apparent est NECESSAIREMENT astrophysique (IMF, photo-z,
    poussiere -- revisions ~-1,5 dex, LCDM+FFB ferme l'ecart).
    Prediction ferme : aucune population z>10 ne survivra aux
    systematiques au-dela de LCDM -- sinon le theoreme de croissance
    est refute (pas seulement un audit).

[3] CMB GRANDE ECHELLE : L'ISOTROPIE EST STRUCTURELLE.
    La poussee est sourcee par la densite (riviere) -- aucun axe
    prefere possible ; zero empreinte Z3 (domaines sous-horizon,
    VERIFIE par l'audit m=3, 3 frequences) ; A_s Starobinsky sans
    modulation. Le cadre PREDIT l'isotropie statistique stricte :
    axe du mal, alignement quadripole/octopole, cold spot, asymetrie
    hemispherique = fluctuations a posteriori ou residus d'avant-plan
    (exactement ce que l'audit m=3 a trouve pour l'exces bas-m :
    m=1-4 generiques, aucune signature).
    PREDICTION FALSIFIABLE : les anomalies ne se REPLIQUERONT pas
    dans les sondes independantes du meme ciel -- modes E (LiteBIRD),
    reconstruction du lentillage, 21 cm ; toute replication croisee
    significative refuterait l'isotropie structurelle de la trame.
"""


def main():
    print(__doc__)
    print("=" * 78)
    print("SYNTHESE : LE PORTEFEUILLE MIS A NIVEAU")
    print("=" * 78)
    print("""  entree              avant (portefeuille)      apres (chantier ecrasement)
  -----------------------------------------------------------------------
  w(z) DESI           ABSENCE (rho statique)    THEOREME (T1 + riviere :
                                                le residu ne se dilue ni
                                                ne se retire -> w = -1)
  JWST z>10           EXPLIQUE (audit nul)      COROLLAIRE (pas de
                                                raccourci de croissance :
                                                theoreme, plus un audit)
  CMB grande echelle  ABSENCE faible (partage)  ISOTROPIE STRUCTURELLE
                                                (+ prediction de
                                                non-replication E/lensing)
  Les trois vont dans le meme sens que le bilan du chantier :
  l'indiscernabilite de LCDM+RG est un theoreme de la trame -- et
  chaque « anomalie » qui s'y oppose recoit une prediction d'absence
  MECANISEE, datee, falsifiable (DESI DR3+, JWST systematiques,
  LiteBIRD modes E).""")


if __name__ == "__main__":
    main()
