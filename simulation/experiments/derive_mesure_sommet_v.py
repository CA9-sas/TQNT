"""Promotion du candidat v : mesure de sommet, regle d'Euler, signification.

Suite de vev_beta_sommets_socle.py (2026-07-20). Trois pas :

[1] DERIVER sqrt(2pi) -- la mesure de sommet du socle.
    Le socle ne transmet qu'aux sommets, par echantillons d'une dent (pas=h).
    Un paquet gaussien norme de largeur RMS sigma delivre sa norme par
    N_eff = sqrt(2pi) sigma / b dents (largeur effective d'une gaussienne :
    le rectangle de meme sommet et meme aire a pour largeur sqrt(2pi) sigma).
    Le paquet MINIMAL transmissible (la reconnexion P15 est LOCALE : un seul
    croisement, un seul sommet) a sigma = b (une dent -- il ne peut pas etre
    plus etroit que la graduation). Alors :
        E(paquet complet) / E(echantillon de sommet) = sqrt(2pi)  EXACTEMENT.
    DERIVATION CONDITIONNELLE (2 conditions declarees) :
      (c1) paquet gaussien (= paquet d'incertitude minimale) ;
      (c2) sigma = b (paquet minimal, une dent).
    Meme statut que epsilon = pi (hbar-ondelette) : forme close sous
    conditions d'ontologie declarees.

[2] TRANCHER l'exposant -- TENTATIVE, verdict honnete : ECHEC PARTIEL.
    La regle preexistante du raccord porte sur la hierarchie AU CARRE :
    (m_base/M_Pl)^2 = (12/7) alpha^18, 18 = c_tot+1 (Euler) -- le nucleon
    est a alpha^9 dans le rapport de MASSE. Le candidat v est a alpha^8
    (rapport de masse) soit alpha^16 au carre -> sous la regle carree il
    faudrait c_eff = 15 (lecture : paire 7_2+7b_2 [14] + le croisement
    reconnecte [1]) ; en unites de masse : 7_2+interface (7+1) ou 8_3 nu.
    TROIS lectures sur DEUX conventions d'unites : la degenerescence du
    porteur N'EST PAS tranchee -- elle est CARTOGRAPHIEE et declaree.

[3] QUANTIFIER face au null certifie.
    Amplitude pinnee (mesure de sommet) et M_Pl/alpha fixes ; MAIS la
    designation du porteur reste libre (8 lectures declarees) -> p ~ 5 %,
    gain modeste sur le null aveugle (~12 %). Le seul vrai gain est le pin
    de l'amplitude ; la cible unique restante est (m1), la designation.
"""
import math

M_PL_GEV = 1.220890e19
V_MEAS_GEV = 246.21965
ALPHA_MEAS = 1.0 / 137.035999
ALPHA_EXACT = 1.0 / 137.0
AMP = math.sqrt(2 * math.pi)


def step1_mesure_sommet():
    print("=" * 72)
    print("[1] MESURE DE SOMMET : sqrt(2pi) (derivation conditionnelle)")
    print("=" * 72)
    print("  Gaussienne normee : |psi|^2(0) = 1/(sigma sqrt(2pi)) ; integrale = 1")
    print("  -> largeur effective (aire/sommet) = sqrt(2pi) sigma.")
    print("  Paquet minimal du socle : sigma = b (une dent ; la reconnexion")
    print("  P15 est locale, un seul sommet) ->")
    print("     E(paquet) / E(sommet) = sqrt(2pi) = %.6f   EXACT" % AMP)
    print("  Conditions declarees : (c1) gaussien/minimal ; (c2) sigma = b.")
    print("  Statut : forme close conditionnelle (meme rang que epsilon = pi).")


def step2_regle_euler():
    print()
    print("=" * 72)
    print("[2] EXPOSANT : la regle d'Euler ne tranche PAS (verdict honnete)")
    print("=" * 72)
    m_base = math.sqrt(12.0 / 7.0) * ALPHA_MEAS**9 * M_PL_GEV * 1000  # MeV
    print("  Regle consignee (raccord 19/07), sur la hierarchie AU CARRE :")
    print("    (m_base/M_Pl)^2 = (12/7) alpha^18, 18 = 17+1 (Euler)")
    print("    -> en rapport de MASSE le nucleon est a alpha^9 :")
    print("       m_base = sqrt(12/7) alpha^9 M_Pl = %.2f MeV (-0,025 %%)" % m_base)
    print("  Le candidat v est a alpha^8 (masse), alpha^16 (carre).")
    print("\n  Lectures disponibles de l'entier (DEGENERESCENCE DECLAREE) :")
    print("    [carre, c_eff+1=16] paire 7_2+7b_2 (14) + croisement reconnecte")
    print("       (1) = 15 -- l'objet du dLk beta ; amplitude 2pi = une mesure")
    print("       de sommet par compagnon ;")
    print("    [masse, n=8]        7_2 + marche d'interface (7+1) ;")
    print("    [masse, n=8]        8_3 nu (c=8, noeud commun EW).")
    print("  AUCUNE regle preexistante ne selectionne une lecture unique :")
    print("  le tranchage par Euler ECHOUE -- c'est le manque (m1) qui")
    print("  gouverne (deriver la designation du porteur trancherait aussi")
    print("  la convention d'unites).")
    print("\n  Ce qui survit du scan des porteurs : l'echelle est si raide")
    print("  (x137 par barreau) qu'UN SEUL barreau peut atterrir -- tout le")
    print("  contenu discriminant est dans l'AMPLITUDE residuelle (etape 3).")


def step3_signification():
    print()
    print("=" * 72)
    print("[3] SIGNIFICATION FACE AU NULL CERTIFIE")
    print("=" * 72)
    # amplitude residuelle mesuree du barreau 7_2
    res_meas = V_MEAS_GEV / (ALPHA_MEAS**8 * M_PL_GEV)
    res_exact = V_MEAS_GEV / (ALPHA_EXACT**8 * M_PL_GEV)
    print("  Amplitude residuelle du barreau 7_2 (v / alpha^8 M_Pl) :")
    print("    alpha mesure : %.5f  vs sqrt(2pi) = %.5f  (%+.3f %%)"
          % (res_meas, AMP, 100 * (res_meas / AMP - 1)))
    print("    alpha 1/137  : %.5f                        (%+.3f %%)"
          % (res_exact, 100 * (res_exact / AMP - 1)))
    # p par hasard : residuelle log-uniforme sur un barreau (ln 137), fenetre
    # +-0,5 % autour d'une amplitude PINNEE (plus de scan d'amplitudes)
    window = 2 * 0.005
    p_one = window / math.log(137.0)
    n_readings = 8           # lectures porteur/unites declarees (etape 2)
    n_derivable = 3          # amplitudes derivables : sqrt(2pi) (sommet),
    #                          sqrt(12/7) (coeur 3/7), pi (epsilon)
    p = p_one * n_readings * n_derivable
    print("\n  p(hasard) une amplitude pinnee, fenetre +-0,5 %% : %.2e" % p_one)
    print("  x %d lectures porteur/unites x %d amplitudes derivables ->"
          % (n_readings, n_derivable))
    print("  p ~ %.0f %% -- gain MODESTE sur le null aveugle (~12 %%) :" % (100 * p))
    print("  le pin de l'amplitude (mesure de sommet) est le seul vrai gain ;")
    print("  la liberte du porteur mange le reste.")
    print("\n  CE QUI MANQUE ENCORE (declare) :")
    print("   (m1) mecanisme de la designation : POURQUOI la barriere de")
    print("        relaxation vit sur le barreau du compagnon reconnecte")
    print("        (E_relax = energie du paquet minimal au barreau du circuit")
    print("        dont Lk bascule) -- coherent mais non demontre ;")
    print("   (m2) les conditions (c1)-(c2) de la mesure de sommet ;")
    print("   (m3) une seconde observable portee par la meme mesure.")
    print("\n  VERDICT : candidat DOCUMENTE, PAS promu (la tentative de")
    print("  tranchage Euler a ECHOUE -- degenerescence cartographiee).")
    print("  Gains reels : sqrt(2pi) derivee conditionnellement (mesure de")
    print("  sommet, reutilisable) ; cible unique identifiee = (m1), la")
    print("  designation du porteur. v RESTE au comptage (10+2 inchange).")


def run():
    print(__doc__)
    step1_mesure_sommet()
    step2_regle_euler()
    step3_signification()


if __name__ == "__main__":
    run()
