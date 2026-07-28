"""(m1) Designation du porteur de la barriere faible -- attaque topologique.

Suite de derive_mesure_sommet_v.py (2026-07-20). Question : POURQUOI la
barriere de relaxation E_relax vivrait-elle au barreau alpha^8 ?

[1] CONSOLIDATION (derivee) : le second ordre de Fermi = paire de
    reconnexions avec INTERMEDIAIRE FUSIONNE.
    - Lk(A,B) = (1/2) somme des croisements signes inter-composantes ;
    - un CHANGEMENT de croisement (brin qui traverse l'autre tube) donnerait
      dLk = +-1 au PREMIER ordre -- mais il est INTERDIT par
      l'impenetrabilite du tube de flux (document principal,
      derive_facteur_interpenetration.py : l'axe d'un tube ne peut pas
      traverser un autre tube) ;
    - l'operation elementaire AUTORISEE est la reconnexion P15 (couper-
      recoller au contact). UNE reconnexion inter-composantes FUSIONNE les
      deux composantes en une seule (lissage d'un croisement) ; il en faut
      une SECONDE pour les separer -- et la paire fusion/scission realise
      dLk = +-1.
    => Le << second ordre / deux echanges / denominateur au carre >> de la
    P15 quantitative n'est PAS un postulat : c'est un THEOREME (tube
    impenetrable + P15). L'etat intermediaire a un NOM : la configuration
    FUSIONNEE (les deux composantes reunies en un seul brin).
    Coherence : c'est la MEME structure de double reconnexion que le canal
    du serrage topologique T_c (pivot des jonctions, c_eff = 3,26).

[2] CANDIDAT << host-lifting >> (le hote souleve de 2 barreaux) : REFUTE.
    Idee : l'intermediaire = le nucleon avec la paire R2 ouverte
    (c_eff = 17-2 = 15 -> barreau carre 16 -> masse alpha^8) ; l'amplitude
    passe de sqrt(12/7) (partition coeur/compagnons) a sqrt(2pi) (mesure de
    sommet du paquet minimal). Nombres : EXACTS (-0,05 %).
    MAIS la barriere est UNIVERSELLE (le muon paie le meme G_F, tau_mu a
    -0,44 %). Sous host-lifting, le hote muon (3_1, c=3) donnerait
    c_eff = 1 -> barreau masse alpha^1 -> E_relax(mu) ~ 2e17 GeV ->
    suppression (E/v)^4 ~ 1e60 : tau_mu 60 ORDRES trop long. MORT.

[3] CANDIDAT << intermediaire fusionne du hote >> : REFUTE aussi.
    Le fusionne du nucleon (lissage d'un croisement : c_eff = 16, une
    composante) -> barreau carre 17 -> masse alpha^8,5 -> E ~ 21 GeV
    (x11,7 sous v) ; et le fusionne du muon differerait encore : meme mort
    par universalite (et demi-barreau non entier).

[4] CE QUI SURVIT -- la contrainte est AIGUISEE :
    TOUTE designation qui compte les croisements du HOTE meurt par
    l'universalite de G_F. La barriere est une propriete du VIDE chi /
    du socle seul (coherent avec << E_relax = vev, echelle du vide >>).
    Le barreau 8 doit venir d'une structure UNIVERSELLE du vide -- aucune
    n'est designee a ce jour. (m1) RESTE OUVERT, mieux borne :
      - la solution doit etre independante du hote ;
      - elle doit couvrir le mode-flip du muon (pas de dLk apparent) ET le
        dLk baryonique par le MEME objet ;
      - les lectures 7_2/8_3/paire de l'etape precedente sont toutes des
        comptages de hote -> ELIMINEES en bloc par [2]-[3].
"""
import math

M_PL_GEV = 1.220890e19
V_MEAS_GEV = 246.21965
ALPHA_MEAS = 1.0 / 137.035999
AMP = math.sqrt(2 * math.pi)


def run():
    print(__doc__)
    print("=" * 72)
    print("CHIFFRAGE DES REFUTATIONS")
    print("=" * 72)
    print("\n[2] Host-lifting (paire R2 ouverte, c_eff = c_tot - 2) :")
    for host, c_tot in [("nucleon (17)", 17), ("muon 3_1 (3)", 3)]:
        c_eff = c_tot - 2
        n_mass = (c_eff + 1) / 2.0
        e = AMP * ALPHA_MEAS**n_mass * M_PL_GEV
        supp = (e / V_MEAS_GEV)**4
        print("    %-14s c_eff=%2d -> barreau masse alpha^%.1f -> E = %.4g GeV"
              % (host, c_eff, n_mass, e))
        print("                   suppression tau vs v : x %.1e" % supp)
    print("    Le nucleon atterrit (-0,05 %) mais le muon est a ~1e60 :")
    print("    l'UNIVERSALITE de G_F tue toute designation par le hote. MORT.")

    print("\n[3] Intermediaire fusionne (lissage, c_eff = c_tot - 1) :")
    for host, c_tot in [("nucleon", 17), ("muon 3_1", 3)]:
        c_eff = c_tot - 1
        n_mass = (c_eff + 1) / 2.0
        e = AMP * ALPHA_MEAS**n_mass * M_PL_GEV
        print("    %-14s c_eff=%2d -> alpha^%.1f -> E = %.4g GeV (vs v : x %.3g)"
              % (host, c_eff, n_mass, e, e / V_MEAS_GEV))
    print("    Demi-barreaux + dependance au hote : MORT aussi.")

    print("\n[4] BILAN :")
    print("  DERIVE (consolidation) : second ordre de Fermi = paire")
    print("    fusion/scission (theoreme : impenetrabilite + P15) ;")
    print("    intermediaire NOMME = configuration fusionnee ; meme")
    print("    structure que la double reconnexion du serrage T_c.")
    print("  REFUTE : host-lifting ; fusionne-du-hote ; en bloc, TOUT")
    print("    comptage de croisements du hote (universalite de G_F).")
    print("  OUVERT : (m1) -- le barreau 8 doit etre une structure")
    print("    UNIVERSELLE du vide/socle ; aucune designee. Le candidat")
    print("    v = sqrt(2pi) alpha^8 M_Pl reste DOCUMENTE, non promu ;")
    print("    v reste au comptage (10+2 inchange).")


if __name__ == "__main__":
    run()
