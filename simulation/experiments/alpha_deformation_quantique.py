"""Si alpha_EM = 1/137,036 : evaluation precise des implications TQNT (19/07).

TENSION LATENTE du corpus : P1 identifie alpha = 1/(k+2) = 1/137 EXACT
(k = 135), mais le mesure vaut 1/137,035999084(21) (-0,026 %) -- et les
verifications numeriques (le theoreme G en tete) utilisent le MESURE.

BANC :
 [1] Le levier alpha^18 tranche : G(1/137 exact) vs G(mesure) vs CODATA.
     Retro-diction : le << 0,42 % historique V5 >> etait-il l'effet du
     choix d'alpha ?
 [2] Lecture (i) -- evolution du couplage (running) : a quelle echelle
     Q* a-t-on alpha(Q*) = 1/137 exact ? (1 boucle EXACTE, dispersion
     e+e- ; voisinages numeriques consignes sous reserve de test nul.)
 [3] Lecture (ii) -- DEFORMATION QUANTIQUE (pre-enregistree : d_1/2 porte
     deja delta_V5p31 = -4 ln d_1/2 et le hint sqrt2 = d_1/2(k=2) de
     Koide) : alpha = d_1/2(k)/(2(k+2)) = cos(pi/137)/137.
     Residu vs CODATA en ppm ; cote atteignable ou non par l'evolution.
 [4] Mini test nul DECLARE : 5 formes a un pas, fenetre du hit.
 [5] Consequences chiffrees par chaine si (ii) est adoptee.
 [6] Unicite de k : le mesure selectionne-t-il toujours k = 135 ?
"""
import math

from scipy.integrate import quad
from scipy.optimize import brentq

ALPHA_INV_MEAS = 137.035999084     # CODATA 2018 (21) -- 1,5e-10 relatif
ALPHA_MEAS = 1.0 / ALPHA_INV_MEAS
K = 135
N = K + 2                          # 137
M_E = 0.51099895                   # MeV
G_CODATA = 6.6743e-11
G_TQNT_MEAS = 6.6710e-11           # theoreme avec alpha mesure


def delta_alpha_inv_spacelike(Q):
    """Decalage de alpha^-1 a l'echelle spacelike Q (MeV), boucle e+e- exacte
    (relation de dispersion) : Delta(alpha^-1) = -(Q^2/3pi) I, I > 0."""
    def integrand(s):
        return math.sqrt(1 - 4 * M_E ** 2 / s) * (1 + 2 * M_E ** 2 / s) / (
            s * (s + Q ** 2))
    val, _ = quad(integrand, 4 * M_E ** 2, math.inf, limit=200)
    return (Q ** 2 / (3 * math.pi)) * val


def run():
    print("=== alpha_EM = 1/137,036 : implications TQNT ===\n")

    # [1] le levier alpha^18
    print("[1] Le levier alpha^18 (theoreme G) :")
    lever = (ALPHA_INV_MEAS / N) ** 18
    g_exact = G_TQNT_MEAS * lever
    print("  (137,036/137)^18 = %.5f" % lever)
    print("  G(alpha mesure)  = %.4e  -> %+.2f %% vs CODATA" % (
        G_TQNT_MEAS, 100 * (G_TQNT_MEAS / G_CODATA - 1)))
    print("  G(1/137 exact)   = %.4e  -> %+.2f %% vs CODATA" % (
        g_exact, 100 * (g_exact / G_CODATA - 1)))
    print("  -> les donnees SELECTIONNENT le mesure (22 sigma vs ~190 sigma) ;")
    print("  RETRO-DICTION : le << 0,42 % historique V5 >> = exactement")
    print("  l'effet du choix alpha = 1/137 exact. Explique.\n")

    # [2] lecture (i) : evolution du couplage
    print("[2] Lecture (i) -- evolution : alpha(Q*) = 1/137 exact pour Q* = ?")
    target = ALPHA_INV_MEAS - N     # 0.035999
    qstar = brentq(lambda q: delta_alpha_inv_spacelike(q) - target, 0.2, 20.0)
    print("  1 boucle exacte (e+e-, dispersion) : Q* = %.4f MeV" % qstar)
    print("  asymptotique log : %.4f MeV (controle)" % (
        M_E * math.exp(0.5 * (3 * math.pi * target + 5.0 / 3.0))))
    for lab, v in [("2 m_e", 2 * M_E), ("Delta m(n-p)", 1.293),
                   ("e x m_e", math.e * M_E)]:
        print("    voisinage : %s = %.4f MeV (%+.1f %%)" % (lab, v, 100 * (qstar / v - 1)))
    print("  (voisinages = curiosites SOUS test nul, aucun pre-enregistrement)")
    print("  Implication (i) : l'identification 1/(k+2) serait etiquetee a")
    print("  l'echelle Q* ; les chaines statiques (G : champ lointain)")
    print("  prennent alpha(0) = le mesure -- coherent mais l'etiquette")
    print("  d'echelle est un AJOUT au postulat.\n")

    # [3] lecture (ii) : deformation quantique
    print("[3] Lecture (ii) -- deformation quantique (PRE-ENREGISTREE) :")
    d_half = 2 * math.cos(math.pi / N)
    alpha_def = d_half / (2 * N)          # cos(pi/137)/137
    inv_def = 1.0 / alpha_def
    resid = alpha_def / ALPHA_MEAS - 1
    print("  alpha = d_1/2/(2(k+2)) = cos(pi/137)/137")
    print("  serie : alpha^-1 = 137 + pi^2/274 + ... = %.7f" % inv_def)
    print("  mesure :                                  %.7f" % ALPHA_INV_MEAS)
    print("  RESIDU : %+.2f ppm sur alpha (mesure a 1,5e-4 ppm)" % (1e6 * resid))
    print("  structure : d_1/2 porte DEJA delta_V5p31 = -4 ln d_1/2 et le")
    print("  hint sqrt2 = d_1/2(k=2) de Koide -- UNE forme, pas un balayage.")
    # cote atteignable ? l'evolution QED (spacelike ET timelike sous seuil)
    # ne fait qu'AUGMENTER alpha : alpha(Q) >= alpha(0). alpha_def < alpha(0) ?
    print("  signe : alpha_def %s alpha(0) -- l'evolution QED ne produit que" % (
        "<" if alpha_def < ALPHA_MEAS else ">"))
    print("  alpha(Q) > alpha(0) : le residu %.2f ppm est du cote NON" % abs(1e6 * resid))
    print("  atteignable par l'evolution a 1 boucle -> ecart intrinseque fin")
    print("  (2 boucles / raccordement complet : a comprendre), PAS du bruit.\n")

    # [4] mini test nul declare (5 formes a un pas)
    print("[4] Test nul declare (5 formes, N = 137) :")
    forms = [
        ("1/N (classique)", 1.0 / N),
        ("cos(pi/N)/N  [q-deformation]", alpha_def),
        ("sin(pi/N)/pi", math.sin(math.pi / N) / math.pi),
        ("(2/pi) sin(pi/2N)", 2 * math.sin(math.pi / (2 * N)) / math.pi),
        ("1/(N + 1/4pi)", 1.0 / (N + 1 / (4 * math.pi))),
    ]
    for lab, v in forms:
        print("    %-30s %+9.2f ppm" % (lab, 1e6 * (v / ALPHA_MEAS - 1)))
    print("  -> un seul hit (0,2 ppm) ; les autres a 175-318 ppm. Fenetre")
    print("  0,4 ppm sur une dispersion ~600 ppm x 5 formes : probabilite")
    print("  de hasard ~3e-3, PLUS le pre-enregistrement structurel.\n")

    # [5] consequences par chaine si (ii)
    print("[5] Consequences chiffrees (adoption de (ii)) :")
    print("  G : alpha_def = mesure a 0,2 ppm -> G decale de %.1e (%.4f %%)" % (
        18 * abs(resid), 100 * 18 * abs(resid)))
    print("      -> -0,05 % INCHANGE ; la tension exact/mesure est DISSOUTE")
    print("      (plus d'etiquette d'echelle a ajouter).")
    a_s_shift = (N / inv_def) ** 5 - 1
    print("  A_s (137^5 -> deforme) : %+.2f %% (vs ecart -10 %% : invisible)" % (
        100 * a_s_shift))
    print("  delta_V5p31 = -4 ln d_1/2 : DEJA deforme -- coherence interne.")
    n_flux_def = inv_def
    print("  quantification de flux (m_l Koide) : N vise %.3f au lieu de 137" % n_flux_def)
    l_pred = 3 * inv_def / (8 * math.pi)
    print("    -> ropelength l(3_1) = 3 alpha^-1/(8 pi) = %.4f" % l_pred)
    print("       vs Abraham 16,372 : %+.3f %% (AMELIORE, etait -0,12 %%)" % (
        100 * (l_pred / 16.372 - 1)))
    print("  Vassiliev, hierarchie, comptages : decalages < 3e-4, invisibles.\n")

    # [6] unicite de k
    print("[6] Unicite de k sous la deformation :")
    for kk in (134, 135, 136):
        nn = kk + 2
        inv_k = nn / math.cos(math.pi / nn)
        print("    k = %d : alpha^-1 = %.4f  (mesure : 137,0360)" % (kk, inv_k))
    print("  -> k = 135 reste UNIQUE (voisins a +-1 unite) ; l'identification")
    print("  du niveau par alpha est inchangee.\n")

    # [7] et sur les masses ?
    print("[7] Sur les masses (chaine par chaine) :")
    lever9 = (ALPHA_INV_MEAS / N) ** 9
    sig_meas, sig_recal = 418.38, 418.47
    sig_exact = sig_meas * lever9
    print("  loi maitresse m = sqrt(sigma) S : PAS de alpha -> e, mu, tau, p,")
    print("    n, pi, secteurs Koide/Regge/GMOR/EW : INCHANGES (0).")
    print("  prediction inversee sqrt(sigma) = (M_Pl/S) sqrt((12/7) a^18)")
    print("    (levier a^9) : deforme/mesure -> %.2f (-0,022 %%) ;" % sig_meas)
    print("    1/137 exact -> %.2f (%+.3f %% vs 418,47) : la prediction de" % (
        sig_exact, 100 * (sig_exact / sig_recal - 1)))
    print("    l'echelle de masse SELECTIONNE elle aussi la deformation --")
    print("    2e vote independant du levier G.")
    print("  a_e = alpha/2pi (ancre du g-2) : l'ecart de la derivation de")
    print("    alpha passe de 263 ppm (1/137) a 0,21 ppm (deforme) : x1250.")
    print("  delta oscillations (V5p29-31) : deja d_1/2 -- closure 0,0008 dex")
    print("    intacte. Cornell (K_alpha calibre) et EW (alpha_0 calibre) :")
    print("    absorbes par calibration, splittings inchanges.")
    print("  Delta m(n-p) fluxoide (echelle alpha hbar c/r_link) : decalage")
    print("    0,2 ppm (ou 263 ppm si l'on venait de 1/137) -- invisible")
    print("    devant le +6,6 %% du mecanisme.")
    print("  Vassiliev sigma_DM ~ alpha^(c-18) : c=30 -> +0,3 %% (1/137 vs")
    print("    mesure) -- invisible devant les ordres de grandeur.\n")

    print("VERDICT : << alpha = 1/137,036 >> n'est pas une menace mais un")
    print("RAFFINEMENT candidat de P1 : alpha_EM = d_1/2/(2(k+2)) -- la ligne")
    print("de Wilson spin-1/2 normalisee par sa dimension QUANTIQUE (pas la")
    print("classique 2). Accord 0,2 ppm (residu fin cote non atteignable par")
    print("l'evolution : reste ouvert) ; G et toutes les chaines inchangees ;")
    print("le 0,42 % historique V5 explique ; ropelength amelioree a -0,09 %.")


if __name__ == "__main__":
    run()
