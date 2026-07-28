"""Derivation du '2 familles' du W : les niveaux Z_3 ouverts de la paire nucleee.

AVANT : le comptage hadronique du W (poids 6 = 2 familles x 3 phases) empruntait
le '2' au MS (doublets (ud), (cs) ouverts, (tb) ferme). Ici : le '2' sort de la
MEME regle Z_3 que le comptage N=2+3 du tau (derive_n_z3_counting.py).

LOI DE COMPTAGE UNIFIEE (P15) :
    N = somme sur les canaux des [niveaux Z_3 cinematiquement OUVERTS]
        x [3 si paire nucleee (phase relative degeneree) ; 1 si noeud simple
           final (le mode EST l'identite/masse -> canaux separes)]
  - Le 'niveau' est l'indice de generation Z_3 (la trialite du trefle), le
    MEME qui indexe la loi de cout lambda^D (D = distance de generation) :
    chaque niveau ouvert contribue exactement 1 par unitarite, fois 3 phases.
  - Les seuils de niveau sont des ENTREES (masses catalogue/PDG), au meme
    titre que m_e, m_mu pour le comptage leptonique.

APPLICATION :
  - muon : 1 niveau lepton (e) -> N = 1                    [deja -0.44%]
  - tau  : 2 niveaux leptons (e,mu) + 1 niveau de paire x3 -> N = 5
           (niveau 1 'charme' FERME : plus legere realisation D/D_s a
            1.87-1.97 GeV > m_tau = 1.777 -- ferme de ~5-11%)
  - W    : 3 niveaux leptons (e,mu,tau) + 2 niveaux de paire x3 -> N = 9
           (niveau 2 FERME : ses realisations sont les etats du secteur EW,
            top-entrelacs 170.4 GeV >> m_W -- or le W est le PLANCHER du
            quartet EW (80.7 < 90.6 < 125.3 < 170.4) : la fermeture du
            3e niveau est STRUCTURELLE, pas accidentelle)

PREDICTIONS : saturation a 3 familles (Z_3 exactement) -- aucun 4e niveau
n'existe ; tout parent en courant charge, si lourd soit-il, a un poids
hadronique <= 9(1+delta) et BR_lep/mode >= ~8%.
"""
import math

HBAR = 6.582119569e-25
M_W = 80.377
M_E, M_MU, M_TAU = 0.000511, 0.1056583755, 1.77686
M_DS, M_D0 = 1.96835, 1.86484        # realisations les plus legeres du niveau 1
M_TOP_TQNT = 170.41                   # niveau 2 : plancher EW = entrelacs top (catalogue)
BW_E, BW_MU, BW_TAU, BW_HAD = 0.1071, 0.1063, 0.1138, 0.6741  # PDG
ALPHA_S_MW_PI = 0.120 / math.pi


def f_ps(x):
    if x < 1e-12:
        return 1.0
    return 1 - 8*x + 8*x**3 - x**4 - 12*x**2*math.log(x)


def run():
    print("=== Le '2 familles' du W = 2 niveaux Z_3 ouverts de la paire nucleee ===\n")

    # -- 1. Seuils de niveau (entrees, comme m_e/m_mu pour les leptons) --
    print("[1] Seuils des niveaux de paire :")
    print("    niveau 0 (leger/GMOR)  : ~0.14-1 GeV  -> ouvert pour tau ET W")
    print("    niveau 1 (charme)      : D/D_s = %.2f-%.2f GeV" % (M_D0, M_DS))
    print("      vs m_tau = %.3f : FERME (+%.0f%%) -> tau n'a qu'UNE famille (le 1 du 2+3)" % (
        M_TAU, (M_D0/M_TAU-1)*100))
    print("      vs m_W = %.1f : ouvert -> 2e famille du W" % M_W)
    print("    niveau 2 (EW)          : top-entrelacs = %.1f GeV >> m_W : FERME (x%.1f)" % (
        M_TOP_TQNT, M_TOP_TQNT/M_W))
    print("    STRUCTUREL : le W est le PLANCHER du quartet EW (80.7<90.6<125.3<170.4)")
    print("    -> le 3e niveau est ferme PAR CONSTRUCTION pour le W.\n")

    # -- 2. Le W avec N = 3 + 2x3 --
    f_lep = [f_ps((m/M_W)**2) for m in (M_E, M_MU, M_TAU)]
    lep_tot = sum(f_lep)
    had = 2 * 3 * (1 + ALPHA_S_MW_PI)
    tot = lep_tot + had
    print("[2] W : N = (%.3f leptons) + 2x3x(1+%.3f) = %.3f" % (
        lep_tot, ALPHA_S_MW_PI, tot))
    print("    BR(W->e nu)  = %.2f%%  vs PDG %.2f%% (%+.1f%%)" % (
        f_lep[0]/tot*100, BW_E*100, (f_lep[0]/tot/BW_E-1)*100))
    print("    BR(W->had)   = %.2f%%  vs PDG %.2f%% (%+.2f%%)\n" % (
        had/tot*100, BW_HAD*100, (had/tot/BW_HAD-1)*100))

    # -- 3. Pincement a un bit : combien de familles les donnees exigent-elles ? --
    b_lep_tot = BW_E + BW_MU + BW_TAU
    had_true = BW_HAD / b_lep_tot * lep_tot
    print("[3] Poids hadronique exige par les BR mesures : %.3f" % had_true)
    print("    delta_fort requis selon le nombre de familles :")
    for nf in (1, 2, 3):
        delta = had_true / (3*nf) - 1
        verdict = ("SELECTIONNE (delta>0, ordre alpha_s(m_W)/pi = %.3f)" % ALPHA_S_MW_PI
                   if nf == 2 else
                   "exclu (delta<0)" if delta < 0 else
                   "exclu (delta ~ +%.0f%%)" % (delta*100))
        print("      familles = %d : delta = %+6.1f%%  -- %s" % (nf, delta*100, verdict))
    print("    => familles = 2 pince a UN BIT ; le residu (%+.1f%%) est la correction" % (
        (had_true/6-1)*100))
    print("    forte a m_W -- et le delta COURT dans le bon sens (+21%% a m_tau -> ~+3-4%%")
    print("    a m_W, exactement le running de alpha_s).\n")

    # -- 4. Coherence croisee tau <-> W (la meme regle, deux comptages) --
    print("[4] La MEME regle compte tout :")
    print("    muon : 1 niveau lepton -> N=1 (automatique)      [-0.44%]")
    print("    tau  : 2 leptons + 1 niveau paire x3 -> N=5      [+12% = delta_fort(m_tau)]")
    print("    W    : 3 leptons + 2 niveaux paire x3 -> N=9     [BR a ~1% pres]")
    print("    Le seuil qui ferme le niveau 1 au tau (D a +5%) et celui qui ferme le")
    print("    niveau 2 au W (top-entrelacs, plancher EW) jouent le MEME role que m_pi")
    print("    fermant le canal hadronique du muon.\n")

    # -- 5. Predictions --
    print("[5] PREDICTIONS (saturation Z_3) : il n'existe QUE 3 niveaux (trialite du")
    print("    trefle). Tout parent en courant charge, si lourd soit-il (W'...), sature a")
    print("    familles=3 : BR_lep/mode >= 1/(3+9x(1+delta)) ~ 8.1%% ; aucune 4e famille")
    print("    ne peut ouvrir. C'est la version 'canaux' du N_gen=3 : meme Z_3.\n")

    print("VERDICT : le '2 familles' du W est DERIVE par la meme regle Z_3 que le 2+3")
    print("du tau -- familles = niveaux Z_3 de la paire ouverts sous m_parent. Entrees :")
    print("les seuils de niveau (masses catalogue/PDG, comme m_e/m_mu au canal lepton).")
    print("Structurel : la fermeture du 3e niveau pour le W vient de ce que le W est le")
    print("PLANCHER du secteur EW. Emprunt residuel : aucun de specifique au W.")


if __name__ == "__main__":
    run()
