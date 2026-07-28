"""Raccord metrique du fond d'ondes chi : la rigidite a travers T_c (19/07).

CHANTIER (brouillon, PAS V6) : le theoreme G est ancre sur le nucleon,
qui n'existe pas avant la condensation (T_c = 67 MeV). Que vaut la
reponse metrique du fond d'ondes chi pre-condensation, et raccorde-t-elle
au kappa_g nucleonique ?

STRUCTURE DU CHANTIER :
 [1] Lecture inversee : la RIGIDITE du vide chi (M_Pl) est primitive ;
     le theoreme G lu a l'envers PREDIT la tension du condensat :
     sqrt(sigma_chi) = (M_Pl/S_octet) * sqrt((12/7) alpha^18).
 [2] L'echelle kappa_g PAR CROISEMENT : h0(c) ~ P * alpha^(c+1) --
     le haut de l'echelle (alpha^0) est le fond d'ondes lui-meme
     (la densite chi non nouee EST la metrique, sans suppression) ;
     chaque croisement ferme le flux chi et ne laisse fuir que
     alpha = 1/(k+2) ; le +1 = la marche d'interface (porte).
 [3] Le pont k : le secteur gravitationnel PRE-condensation (V5.15)
     porte le MEME alpha -- a_R2 = (k+2)^5/6, M_S = M_Pl alpha^(5/2),
     A_s = N^2 6 alpha^5/(24 pi^2) : adimensionne en M_Pl (ne pince
     PAS la rigidite absolue, mais verrouille la structure alpha).
 [4] Epinglages de la rigidite ABSOLUE pre-T_c (honnete : faibles) ;
 [5] La marche G(t) : indefini -> G0 a t_c ~ 1,4e-4 s ;
 [6] Sonde dynamique : ondes gravitationnelles de la condensation.
"""
import math

# constantes
ALPHA = 7.2973525693e-3        # alpha_EM mesuree (celle de la verification V6)
K_CS = 135                     # niveau CS ; 1/(k+2) = 1/137
HBARC_SI = 3.1615268e-26       # J.m
G_CODATA = 6.6743e-11
G_TQNT = 6.6710e-11
MEV_KG = 1.7826619e-30
M_BASE = 938.240               # MeV (base octet V6)
S_OCTET = 2.242006             # base octet = sqrt(sigma_chi) x S_octet
SQRT_SIGMA = 418.47            # MeV (recalibree 2026-07-18)
C_SI = 2.99792458e8
T_C = 67.0                     # MeV
G_STAR = 15.2                  # g_* a T_c (derive 2026-07-17)


def run():
    print("=== Raccord metrique du fond d'ondes chi ===\n")

    # [1] lecture inversee : rigidite primitive -> sigma_chi PREDITE
    print("[1] Lecture inversee (rigidite du vide primitive) :")
    a18 = ALPHA ** 18
    m2_kg2 = (12.0 / 7.0) * a18 * HBARC_SI / G_CODATA   # kg^2
    m_pred = math.sqrt(m2_kg2) / MEV_KG
    sig_pred = m_pred / S_OCTET
    print("  m_base predite = M_Pl sqrt((12/7) a^18) = %.2f MeV" % m_pred)
    print("  sqrt(sigma_chi) predite = %.2f MeV  vs recalibree %.2f : %+.3f %%"
          % (sig_pred, SQRT_SIGMA, 100 * (sig_pred / SQRT_SIGMA - 1)))
    print("  -> le residu est EXACTEMENT celui de G (-0,05 %% <-> -0,025 %%")
    print("     en masse) : meme equation, fleche inversee. Sous cette")
    print("     lecture le raccord est STRUCTUREL (la rigidite ne change")
    print("     pas a T_c) et la condensation herite l'echelle.\n")

    # [2] kappa_g par croisement
    print("[2] L'echelle par croisement (h0(c) ~ P alpha^(c+1)) :")
    h0 = G_TQNT * (M_BASE * MEV_KG) ** 2 / (2 * HBARC_SI)
    alpha_per = (h0 * 7.0 / 6.0) ** (1.0 / 18.0)
    print("  h0(nucleon) = %.3e ; facteur par croisement extrait :" % h0)
    print("  (7 h0/6)^(1/18) = %.6e  vs 1/(k+2) = %.6e  (%+.2f %%)"
          % (alpha_per, 1.0 / (K_CS + 2), 100 * (alpha_per * (K_CS + 2) - 1)))
    print("  Echelle (interpretation raccord ; PAS un G par particule --")
    print("  l'universalite de G est etablie, cf. discriminant) :")
    print("    alpha^0  : le FOND lui-meme (la densite chi non nouee est la")
    print("               metrique : reponse pleine, rien a fermer)")
    print("    alpha^1  : la marche d'interface (porte) de toute structure")
    print("               localisee (le +1 de l'exposant c_n+1)")
    print("    alpha^18 : le nucleon (17 croisements + interface) -> la")
    print("               hierarchie 1e-36 nait AVEC la matiere.\n")

    # [3] le pont k : le secteur gravitationnel pre-condensation
    print("[3] Pont k (secteur pre-condensation, V5.15) :")
    a_v5 = (K_CS + 2) ** 5 / 6.0
    A_s = 60.0 ** 2 / (4 * math.pi ** 2 * (K_CS + 2) ** 5)
    print("  a_R2 = (k+2)^5/6 = %.3e M_Pl^-2 (vs calibration 7,94e9 : +1,3 %%)" % a_v5)
    print("  M_S = M_Pl alpha^(5/2) ; A_s = N^2/(4 pi^2 137^5) = %.3e" % A_s)
    print("  (Planck 2,099e-9 : %+.1f %%)" % (100 * (A_s / 2.099e-9 - 1)))
    print("  A_s est ADIMENSIONNEE en M_Pl : elle ne pince pas la rigidite")
    print("  absolue, mais VERROUILLE la structure -- le meme alpha = 1/(k+2)")
    print("  gouverne les exposants des deux cotes de T_c (5 pre / 18 post).\n")

    # [4] epinglages de la rigidite absolue pre-T_c (honnete)
    print("[4] Rigidite ABSOLUE pre-T_c -- epinglages empiriques (honnete) :")
    print("  T_c^- (chaine Omega, marges Gamma/H 9,3 et 1/0,067) :")
    print("    H libre a x9-15 -> G_eff(T_c^-) pince a un facteur ~1e2")
    print("  Rechauffage (marge T_reh/T_BBN ~ 13 ordres, T_reh ~ G^-1/4) :")
    print("    pince a des dizaines d'ordres -> sans objet")
    print("  Inflation : observables V5.14/15 adimensionnees -> AUCUN epinglage absolu")
    print("  Post-T_c : exact (BBN, CMB, aujourd'hui).")
    print("  -> la continuite absolue est une HYPOTHESE (fondee par [1]+[3]),")
    print("     testee a un facteur ~1e2 a l'entree de l'ere nouee.\n")

    # [5] la marche G(t)
    t_c = 2.42 / math.sqrt(G_STAR) * (T_C) ** (-2)
    print("[5] La marche G(t) :")
    print("  t(T_c = 67 MeV, g_* = %.1f) = %.2e s" % (G_STAR, t_c))
    print("  G(t) : INDEFINI (pas de puits localises) -> G0 = 6,671e-11,")
    print("  une seule marche a t ~ 1,4e-4 s, a la vitesse de la")
    print("  condensation (domaines/cordes de la chaine Omega) ; aucune")
    print("  cinetique de valeur (la rigidite preexiste, lecture [1]).\n")

    # [6] sonde dynamique
    f_gw = 1.65e-7 * (T_C * 1e-3 / 100.0) * (G_STAR / 100.0) ** (1.0 / 6.0)
    print("[6] Sonde dynamique de la transition :")
    print("  GW de la condensation : f_aujourd'hui ~ %.0e Hz" % f_gw)
    print("  (sous la bande PTA ~1e-9 Hz : hors de portee actuelle ; trace")
    print("  potentielle dans les distorsions mu du CMB -- a explorer).")

    # [7] arbitrage : quelle constante est fixe ?
    print("[7] Arbitrage (19/07) : quelle constante est fixe ?")
    eta = (2.0 - 1.4) / ((2.0 + 1.4) / 2)
    print("  Lecture 'kappa_g fixe par particule normalisee' : chaque")
    print("  particule source le meme puits -> champ ~ nombre de particules ;")
    print("  eta(H,Pt) ~ %.2f vs MICROSCOPE 2,3e-15 -> EXCLUE (~14 ordres)." % eta)
    rel_e = (ALPHA ** 4 / 0.511) / (ALPHA ** 18 / M_BASE)
    print("  Variante 'echelle par espece' (chaque K suit alpha^(c+1)) :")
    print("  puits/masse (e-)/(nucleon) = %.1e -> les electrons domineraient" % rel_e)
    print("  la gravite : contradiction immediate. EXCLUE aussi.")
    for lab, m in [("nucleon", M_BASE), ("electron", 0.511)]:
        kg = 8 * math.pi * G_TQNT * (m * MEV_KG) / C_SI ** 2
        print("  kappa_g(%s) = %.4e m" % (lab, kg))
    print("  -> l'objet FIXE est la RIGIDITE (kappa_g/m = 8 pi G/c^2, i.e. G) ;")
    print("  kappa_g(K), une longueur, VARIE avec l'espece (prop. a m_K).")
    print("  Le vrai de l'intuition : la VALEUR de G est ancree par le noeud")
    print("  dominant (contrefactuel : d'autres noeuds -> un autre G) ; dans")
    print("  NOTRE univers, un seul G pour tout (discriminant).\n")

    # [8] determination EXACTE de kappa_g (chaine auto-contenue)
    print("[8] Determination exacte de kappa_g :")
    lam_c = HBARC_SI / ((M_BASE * MEV_KG) * C_SI ** 2)   # m
    kg_auto = (96.0 * math.pi / 7.0) * a18 * lam_c
    l_pl = math.sqrt(6.62607015e-34 / (2 * math.pi) * G_CODATA / C_SI ** 3)
    kg_planck = 8 * math.pi * math.sqrt(12.0 / 7.0) * ALPHA ** 9 * l_pl
    print("  forme close (auto-contenue) : kappa_g = (96 pi/7) a^18 lambdabar_C")
    print("    = %.4e m" % kg_auto)
    print("  forme de Planck (lecture inversee, l_Pl = %.4e m) :" % l_pl)
    print("    kappa_g = 8 pi sqrt(12/7) a^9 l_Pl = %.4e m" % kg_planck)
    print("  valeur historique du code (KAPPA_PHYS, CODATA-ancree) : 3.1218e-53")
    print("    -> ecart %.2f pour mille = le residu de G ; la valeur EXACTE" % (
        1000 * (3.1218e-53 / kg_auto - 1)))
    print("    de la chaine auto-contenue est kappa_g = %.4e m." % kg_auto)
    h0_ex = (6.0 / 7.0) * a18
    print("  h0 exact = (6/7) a^18 = %.4e (perturbation par nucleon)" % h0_ex)
    print("  par espece, kappa_g(K) = kappa_g x m_K/m_base :")
    for lab, m in [("e-", 0.51099895), ("mu-", 105.6584), ("tau-", 1776.86),
                   ("nucleon (base)", M_BASE), ("neutron", M_BASE + 1.293)]:
        print("    %-15s %.4e m" % (lab, kg_auto * m / M_BASE))
    print("  echelle : le long de la construction, kappa_g(K) ~ m_K ~")
    print("  alpha^((c+1)/2) -> un facteur sqrt(alpha) = 1/11,7 par croisement")
    print("  (sens constructif seulement -- pas des couplages multiples).\n")

    print("\nRESTES OUVERTS : (i) la rigidite elle-meme (pourquoi M_Pl --")
    print("pendant UV de m_l, transmutation) ; (ii) le mecanisme microscopique")
    print("du alpha par croisement (tunneling du flux chi a un croisement --")
    print("cible solveur) ; (iii) le statut de la marche d'interface (+1).")


if __name__ == "__main__":
    run()
