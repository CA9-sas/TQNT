"""La mesure comme reconnexion de feuille : le banc des six contraintes (20/07).

MECANISME PROPOSE : la mesure de A = reconnexion (P15) de la branche
W_A de la feuille de paire avec la toile de feuilles de l'appareil
(macroscopique). Deux ingredients structurels :
  - SUPERSELECTION TOPOLOGIQUE : les secteurs de topologie de feuille
    (enroulements, enlacements) sont superselectionnes -- une
    superposition de topologies distinctes est tranchee par la premiere
    reconnexion environnementale ;
  - la reconnexion est ENVIRONNEMENTALE (contact P15 a un croisement),
    jamais spontanee -> AUCUN effondrement intrinseque.

BANC : les six contraintes de tout mecanisme de mesure, avec le statut
honnete de chaque reponse (derive / herite / esquisse) ; les taux ; et
la prediction discriminante vs GRW/CSL.
"""


def run():
    print("=== La mesure comme reconnexion de feuille : six contraintes ===\n")

    rows = [
        ("1. Resultats definis",
         "superselection topologique : la topologie de feuille est un",
         "secteur (pas d'observable locale entre secteurs) ; la premiere",
         "reconnexion environnementale tranche -- ESQUISSE STRUCTURELLE",
         "(la superselection des charges topologiques est etablie en QFT ;",
         "son application a la feuille de paire est la proposition)"),
        ("2. Regle de Born",
         "taux de reconnexion proportionnel a la densite locale |Psi|^2",
         "(la danse chi) -> ponderation en |Psi|^2 mecanisee ; la forme",
         "QUADRATIQUE exacte reste HERITEE (statut Gleason/envariance --",
         "le meme que toutes les ontologies)"),
        ("3. Non-signalisation",
         "rien ne se propage a la mesure : la contrainte w_A + w_B = 0",
         "passe par le vertex de creation (PASSE commun, homotopie) ;",
         "les marginales de B sont intactes -- HERITE du formalisme,",
         "COHERENT avec l'ontologie (aucun canal a fermer)"),
        ("4. Base pointeur",
         "P15 est LOCAL (reconnexion a un croisement spatial) -> la base",
         "selectionnee est position-locale -- l'einselection par la",
         "localite du contact : DERIVE de la localite de P15"),
        ("5. Vitesse (Bell >= 1e4 c)",
         "satisfaite TRIVIALEMENT : correlation par histoire commune",
         "(homotopie de feuille), aucune influence ne voyage -- pas de",
         "vitesse a borner"),
        ("6. Irreversibilite",
         "reconnexion avec une toile macroscopique = production",
         "d'entropie au taux Gamma de la danse (formule maitresse de la",
         "dissipation) -- la fleche de la mesure EST la 2e loi : DERIVE",
         "(unification interne deja consignee)"),
    ]
    for r in rows:
        print(r[0])
        for line in r[1:]:
            print("   " + line)
        print()

    print("TAUX (identification Gamma_mesure = Gamma_decoherence, aucun")
    print("parametre nouveau -- ordres de grandeur standards Joos-Zeh) :")
    for sys_, tau in (("grain de poussiere 10 um, air", "~1e-31 s"),
                      ("nanosphere levitee, ultravide", "~1e-2 s"),
                      ("atome isole (fond cosmique seul)", "~1e+8 s")):
        print("   %-36s tau_dec %s" % (sys_, tau))
    print("   -> la hierarchie de decoherence standard est REPRISE telle")
    print("      quelle : la TQNT n'ajoute AUCUN taux.\n")

    print("PREDICTION DISCRIMINANTE (falsifiable) :")
    print("   la reconnexion exige un CONTACT environnemental -> effondrement")
    print("   intrinseque STRICTEMENT NUL : lambda_GRW = 0, lambda_CSL = 0.")
    print("   Etat de l'art : emission X spontanee et nanospheres poussent")
    print("   lambda_CSL vers le bas (une part de l'espace GRW/CSL initial")
    print("   est deja exclue) -- la TQNT est du cote de la mecanique")
    print("   quantique standard : TOUTE detection confirmee d'un")
    print("   effondrement spontane refuterait ce mecanisme.")
    print("   Corollaire : une superposition macroscopique topologiquement")
    print("   ISOLEE (zero reconnexion) survit indefiniment.\n")

    print("BILAN : 2 pieces DERIVEES (base pointeur par localite de P15 ;")
    print("irreversibilite par Gamma commun), 2 HERITEES (Born quadratique,")
    print("non-signalisation), 1 ESQUISSE STRUCTURELLE (superselection de")
    print("feuille -> resultats definis), 1 TRIVIALE (vitesse). Le coeur")
    print("dur restant : DEMONTRER la superselection de topologie de feuille")
    print("dans la theorie chi (le pas qui changerait le statut du chantier).")


if __name__ == "__main__":
    run()
