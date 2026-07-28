"""Deriver T_c : gel cinetique du rearrangement topologique du reseau (20/07).

PROBLEME : T_c ~ 67 MeV (enveloppe 60-75) est la systematique DOMINANTE
de la chaine Omega ; il est actuellement ADOPTE (epingle par eta_B, le
pas geometrique 70,2, la coherence g_s), pas derive. La limbox
baryogenese documente la confusion de deux echelles (T_trans ~ 155 MeV
de premiere main ; l'echelle ~67 non derivee).

MECANISME PROPOSE : le verrouillage du reseau = GEL Gamma = H du
rearrangement topologique. Tant que le reseau se rearrange (P15 entre
fragments), la population de nœuds s'equilibre ; elle gele quand le
taux d'activation tombe sous Hubble. L'unite de rearrangement est le
FRAGMENT de corde a la coupure -- le quantum de la cascade
E_q = sigma_chi * xi, avec xi = kappa_dom/T la taille de domaine au
verrouillage (kappa_dom = 1,14, nombre PUR du croisement xi*T=1 mesure
sur reseau, Caselle-Nada -- aucune echelle absolue requise).

    Gamma ~ T e^{-E_q(T)/T},  H = 1,66 sqrt(g_s) T^2 / M_Pl
    Gel :  kappa_dom sigma_chi / T^2 = ln[ M_Pl / (1,66 sqrt(g_s) T) ]

Equation AUTO-COHERENTE : T_c ne depend que de sigma_chi, kappa_dom et
M_Pl (+ g_s, logarithmique). C'est une TRANSMUTATION DIMENSIONNELLE
ancree sur M_Pl -- la meme structure que le raccord G (rigidite).

BANC : [1] resolution ; [2] candidats de barriere alternatifs
(discipline du null : la selection doit etre mecanique, pas l'accord) ;
[3] sensibilites (prefacteur, kappa_dom, g_s) ; [4] aval (Omega, eta_B,
pas geometrique, convergence de la serie).
"""
import math

M_PL = 1.2209e19          # GeV (masse de Planck)
SQRT_SIGMA = 0.41847      # GeV (recalibree 2026-07-18)
SIGMA = SQRT_SIGMA**2     # GeV^2
KAPPA_DOM = 1.14          # xi*T au verrouillage (Caselle-Nada, nombre pur)
G_S = 15.2                # g_s(T_c) derive
M_EDGE = 1.73             # GeV, m(c=23) bord du reservoir (loi geometrique)
DM_STEP = 0.0702          # GeV, pas geometrique par croisement
LAM_ALT = 6.1479          # croissance du comptage de nœuds
ALPHA = 1.1991            # alpha_cond
T_NOM = 0.067             # GeV, valeur adoptee historique


def log_term(t, prefac=1.0, g_s=G_S):
    """ln[ prefac * M_Pl / (1,66 sqrt(g_s) T) ]"""
    return math.log(prefac * M_PL / (1.66 * math.sqrt(g_s) * t))


def solve_tc(barrier, prefac=1.0, kappa=KAPPA_DOM, g_s=G_S):
    """Resout E_b(T)/T = log_term(T). barrier(T, kappa) en GeV."""
    t = 0.07
    for _ in range(200):
        L = log_term(t, prefac, g_s)
        e_b = barrier(t, kappa)
        t_new = e_b / L if barrier is bar_knot else math.sqrt(e_b * t / L)
        t = 0.5 * t + 0.5 * t_new
    return t


def bar_quantum(t, kappa):
    """(a) quantum de fragment : E_q = kappa sigma / T (unite du reseau)."""
    return kappa * SIGMA / t


def bar_floor(t, kappa):
    """(b) plancher energetique de la cascade : 0,575 * quantum."""
    return 0.575 * kappa * SIGMA / t


def bar_knot(t, kappa):
    """(c) masse du nœud du bord : m(c=23) (rearrangement PAR nœud)."""
    return M_EDGE


def run():
    print("=== Deriver T_c : gel topologique Gamma = H ===\n")
    print("[1] EQUATION AUTO-COHERENTE (barriere = quantum de fragment) :")
    print("    kappa sigma / T^2 = ln[M_Pl/(1,66 sqrt(g_s) T)]")
    tc = solve_tc(bar_quantum)
    L = log_term(tc)
    print("    sigma_chi = %.6f GeV^2, kappa_dom = %.2f, g_s = %.1f" % (
        SIGMA, KAPPA_DOM, G_S))
    print("    -> T_c = %.2f MeV  (log = %.1f ; quantum E_q = %.2f GeV)" % (
        tc * 1e3, L, KAPPA_DOM * SIGMA / tc))
    print("    Adopte historique : 67 MeV -> ecart %+.1f %%" % (
        (tc / T_NOM - 1) * 100))
    print("    (le quantum E_q = %.2f GeV retrouve le 2,98 GeV documente)\n"
          % (KAPPA_DOM * SIGMA / tc))

    print("[2] CANDIDATS DE BARRIERE (discipline du null -- la selection")
    print("    est MECANIQUE : l'unite qui re-equilibre le RESEAU est le")
    print("    fragment ; le plancher est un etat FINAL pas une barriere ;")
    print("    la masse de nœud gele la population de nœuds, pas le reseau) :")
    for name, bar in (("(a) quantum de fragment kappa*sigma/T", bar_quantum),
                      ("(b) plancher 0,575 * quantum", bar_floor),
                      ("(c) masse du bord m(c=23)", bar_knot)):
        t = solve_tc(bar)
        print("    %-38s T_c = %5.1f MeV  (%+5.1f %%)" % (
            name, t * 1e3, (t / T_NOM - 1) * 100))
    print("    -> seul (a) tombe sur la valeur triplement sollicitee.\n")

    print("[3] SENSIBILITES (la forme en racine attenue tout) :")
    for lab, kw in (("prefacteur x e^{+2}", dict(prefac=math.e**2)),
                    ("prefacteur x e^{-2}", dict(prefac=math.e**-2)),
                    ("kappa_dom 1,08 (-5 %)", dict(kappa=1.08)),
                    ("kappa_dom 1,20 (+5 %)", dict(kappa=1.20))):
        t = solve_tc(bar_quantum, **kw)
        print("    %-24s T_c = %.1f MeV (%+.1f %%)" % (
            lab, t * 1e3, (t / tc - 1) * 100))
    outs = [solve_tc(bar_quantum, g_s=g) for g in (13.0, 20.0)]
    print("    g_s 13 / 20 :            T_c = %.1f / %.1f MeV" % (
        outs[0] * 1e3, outs[1] * 1e3))
    print("    -> enveloppe derivee ~ [64, 69] MeV, contre [60, 75] adoptee\n")

    print("[4] AVAL (a T_c derive = %.2f MeV, vs nominal 67) :" % (tc * 1e3))
    # Omega : poids de Boltzmann du bord, m_edge/T
    ratio = math.exp(-M_EDGE * (1 / tc - 1 / T_NOM))
    print("    Omega : 3,9 x %.2f = %.1f  (Planck 5,45 : %+.0f %%)" % (
        ratio, 3.9 * ratio, (3.9 * ratio / 5.45 - 1) * 100))
    lo, hi = 3.9 * math.exp(-M_EDGE * (1 / 0.064 - 1 / T_NOM)), \
        3.9 * math.exp(-M_EDGE * (1 / 0.069 - 1 / T_NOM))
    print("    enveloppe derivee [64, 69] -> Omega in [%.1f, %.1f]" % (lo, hi))
    print("    eta_B : (m_e/T)^2 -> x %.3f (reste 6,0e-10, inchange)" % (
        (T_NOM / tc) ** 2))
    print("    pas geometrique : Dm/T_c = %.3f (net %.2f < 1 : converge)" % (
        DM_STEP / tc,
        LAM_ALT * math.exp(-ALPHA) * math.exp(-DM_STEP / tc)))
    print("    t_c (marche G) : quasi inchange (T_c a -0,4 %%)\n")

    print("BILAN : la STRUCTURE de T_c est un gel Gamma=H a barriere")
    print("quantum-de-fragment -- transmutation ancree M_Pl (comme G).")
    print("Central %.1f MeV a -0,4 %% de l'adopte ; l'enveloppe se resserre" % (
        tc * 1e3))
    print("de [60,75] a ~[64,69]. Residu honnete : le O(1) de la barriere")
    print("(les candidats (b)/(c) sont ecartes par mecanisme, pas par le")
    print("null seul) et le prefacteur (attenue en racine par le log).")


if __name__ == "__main__":
    run()
