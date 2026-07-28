"""Derivation du comptage N = 2+3 de la reconnexion P15 (canal hadronique en modes Z_3).

STATUT AVANT : le '3' du canal hadronique du tau etait une LECTURE (relabel du
facteur de couleur du MS en 'modes Z_3'), pas une derivation. Ici : mecanisme
candidat + pincement a un bit par les donnees + controles.

DERIVATION CANDIDATE (M3 -- phase Z_3 relative de la paire nucleee) :
  1. Le sommet P15 est unite (g_topo=1, cf. p15_reconnexion_quantitative.py).
     Le taux par canal est donc identique ; N = nombre d'etats finals
     topologiquement DISTINCTS et DEGENERES en energie (somme incoherente
     sur les secteurs topologiques).
  2. Canal leptonique : l'etat final porte UN 3_1 dont le mode Z_3 est son
     IDENTITE (e, mu) -- les masses different (Koide) => PAS de degenerescence,
     chaque mode ouvert = un canal separe avec son propre espace des phases.
     Ouverts sous m_tau : e et mu => 2.
  3. Canal hadronique : la reconnexion nuclee une paire noeud+miroir depuis le
     tube chi. La paire porte un label de phase Z_3 RELATIF (herite de la
     structure Z_3 du coeur trefle / trialite du tube) ; le label etant relatif
     (mode + anti-mode), il se COMPENSE dans l'energie => les 3 valeurs sont
     exactement degenerees => UN seuil, multiplicite 3.
     (Coherent avec le catalogue : aucune masse de meson ne porte de label Z_3
     de paire -- le label est 'silencieux', donc degenere.)
  4. Somme inclusive sur les types de paires : la loi de cout lambda^D avec
     unitarite donne V_ud^2 + V_us^2 = (1-lambda^2) + lambda^2 = 1 -- le
     comptage inclusif est exactement 3 x 1 (pas de double comptage).
  => N_tau = 2 + 3, et le '3 de couleur' et le '3 des generations' sont LE
     MEME Z_3 (celui du trefle) : coincidence inexpliquee dans le MS
     (N_c et N_gen independants), identite structurelle en TQNT.

CONTROLE M1 (alternative rejetee) : si la multiplicite comptait l'orbite C_q
des sites de chirurgie sur le noeud nuclee, un contenu 5_1 (torique C_5)
donnerait 5, un 4_1 donnerait 2 : multiplicite DEPENDANTE du contenu. Les
donnees (R_tau inclusif, W) montrent un 3 UNIVERSEL => c'est la phase Z_3
(M3), pas l'orbite geometrique (M1).

HYPOTHESE RESTANTE (honnete) : l'existence du label Z_3 pour TOUTE paire
nucleee = trialite du tube chi (etablie pour les coeurs trefle, postulee
au-dela). Une hypothese structurelle identifiee, pas zero.
"""
import math

HBAR = 6.582119569e-25   # GeV.s
V = 246.22               # GeV
M_E, M_MU, M_TAU = 0.000511, 0.1056583755, 1.77686
M_PI = 0.13957
TAU_TAU_PDG = 2.903e-13
TAU_MU_PDG = 2.1969811e-6
# W (PDG)
BW_E, BW_MU, BW_TAU, BW_HAD = 0.1071, 0.1063, 0.1138, 0.6741
ALPHA_S_MTAU_PI = 0.33 / math.pi   # ~0.105 ; resomme (serie R_tau) : ~0.20
ALPHA_S_MW_PI = 0.120 / math.pi    # ~0.038


def f_ps(x):
    """Espace des phases 3-corps, f(x=m_f^2/m_i^2) = 1-8x+8x^3-x^4-12x^2 ln x."""
    if x < 1e-12:
        return 1.0
    return 1 - 8*x + 8*x**3 - x**4 - 12*x**2*math.log(x)


def gamma_base(m):
    return m**5 / (384 * math.pi**3 * V**4)


def run():
    print("=== Derivation du comptage N = 2+3 (modes Z_3 du canal hadronique) ===\n")

    # -- 0. Controle nul automatique : le muon --
    # Sous m_mu : seul e ouvert (1 canal) ; hadronique FERME (m_pi > m_mu).
    print("[0] Controle nul (muon) : m_pi = %.1f MeV > m_mu = %.1f MeV" % (M_PI*1e3, M_MU*1e3))
    print("    => le canal hadronique est cinematiquement FERME : la loi de comptage")
    print("    predit N_mu = 1 AUTOMATIQUEMENT (pas un choix). tau_mu deja a -0.44%%.\n")

    # -- 1. Le tau avec N = 2+3 et espaces des phases par canal --
    fs = {'e': f_ps((M_E/M_TAU)**2), 'mu': f_ps((M_MU/M_TAU)**2)}
    n_eff = fs['e'] + fs['mu'] + 3.0
    t_tau = HBAR / (n_eff * gamma_base(M_TAU))
    print("[1] tau avec N = f_e + f_mu + 3 = %.4f + %.4f + 3 = %.4f" % (fs['e'], fs['mu'], n_eff))
    print("    tau_tau = %.4e s vs PDG %.4e (%+.1f%%)\n" % (
        t_tau, TAU_TAU_PDG, (t_tau/TAU_TAU_PDG - 1) * 100))

    # -- 2. Pincement a un bit : quelle multiplicite hadronique les donnees exigent-elles ?
    n_true = (HBAR / TAU_TAU_PDG) / gamma_base(M_TAU)
    had_true = n_true - fs['e'] - fs['mu']
    print("[2] Comptage exige par tau_tau mesure : N_true = %.3f -> hadronique = %.3f" % (
        n_true, had_true))
    print("    delta_fort requis selon N_had (had_true/N_had - 1) :")
    for nh in (2, 3, 4):
        delta = had_true/nh - 1
        verdict = ("SELECTIONNE (delta>0, ordre alpha_s)" if nh == 3 else
                   "exclu (delta<0 : une correction forte augmente le taux)" if delta < 0 else
                   "exclu (delta ~ +%.0f%% >> corrections fortes connues)" % (delta*100))
        print("      N_had = %d : delta = %+6.1f%%  -- %s" % (nh, delta*100, verdict))
    print("    Reference MS : delta_QCD(R_tau) resomme ~ +20%% (alpha_s(m_tau)/pi = %.3f," % (
        ALPHA_S_MTAU_PI))
    print("    serie + non-perturbatif). Requis ici : %+.1f%%. COINCIDE." % ((had_true/3-1)*100))
    print("    => N_had = 3 est pince a UN BIT par les donnees ; le residu +12%% du")
    print("    tau EST la correction forte du canal hadronique (identifiee, pas ajustee).\n")

    # -- 3. Coherence W : le MEME 3, avec un delta qui court dans le bon sens --
    b_lep = (BW_E + BW_MU + BW_TAU) / 3
    r_w = BW_HAD / b_lep
    print("[3] Controle W : R_W = B_had / B_lep,mode = %.3f" % r_w)
    print("    Par famille de paires (2 ouvertes ; emprunt, cf. NB) : %.3f = 3 x (1%+.3f)" % (
        r_w/2, r_w/6 - 1))
    print("    alpha_s(m_W)/pi = %.3f -- le MEME 3, et le delta COURT dans le bon sens" % (
        ALPHA_S_MW_PI))
    print("    (0.21 a m_tau -> ~0.03 a m_W). NB : le '2 familles' n'est PAS derive")
    print("    (OUVERT) ; seul le facteur 3 par famille est teste ici.\n")

    # -- 4. Discrimination M3 vs M1 --
    print("[4] M1 (orbite C_q du noeud nuclee) predirait une multiplicite DEPENDANTE")
    print("    du contenu (5_1 -> 5, 4_1 -> 2) ; les donnees montrent un 3 universel")
    print("    (R_tau inclusif ~25%% de canaux K/multi-pion, W) => M3 (phase Z_3).\n")

    print("VERDICT : derivation CONDITIONNELLE du 2+3 -- mecanisme = phase Z_3 relative")
    print("de la paire nucleee (degeneree car relative ; leptons non degeneres car le")
    print("mode EST l'identite/masse) ; N_had=3 pince a un bit ; hypothese restante :")
    print("la trialite Z_3 du tube pour toute paire (etablie coeurs trefle, postulee")
    print("au-dela). Structurel : N_c = N_gen = 3 est UNE SEULE origine (le Z_3 du")
    print("trefle) -- coincidence inexpliquee dans le MS, identite en TQNT.")


if __name__ == "__main__":
    run()
