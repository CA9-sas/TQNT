"""La distribution de Boltzmann A TRAVERS la TQNT : fond thermique fini de domaine (16/07).

PROGRAMME (demande utilisateur) : ne pas IMPORTER e^{-m/T} -- le deriver des
ingredients TQNT, et voir ce que la TQNT y change.

INGREDIENTS TQNT :
  1. LE BAIN = les ondes chi libres (secteur BASE) : modes de radiation,
     entropie S(E,V) = (4/3)(aV)^{1/4} E^{3/4}, a = g_s pi^2/30.
  2. L'EQUIPROBABILITE = la danse du chi (formule maitresse de la
     dissipation) : Gamma > 0 <=> melange <=> 2e loi (DEJA document principal) --
     l'hypothese ergodique est remplacee par le point fixe du melange.
  3. T EMERGENTE = (dS/dE)^{-1} du fond thermique (coherente avec la definition
     cinetique du chapitre thermo).
  4. LE DOMAINE FINI (specifiquement TQNT) : le fond thermique accessible a une
     nucleation est le domaine de correlation kappa_dom/T_c (kappa_dom =
     1.14, verrouillage lu sur reseau). Boltzmann e^{-m/T} n'est que la
     LIMITE fond thermique-infini du vrai poids microcanonique :
        w_mc(m) = W(E_dom - m)/W(E_dom) = exp[S(E_dom-m) - S(E_dom)]
                = exp[(4/3)(E/T)((1 - m/E)^{3/4} - 1)],   0 si m > E_dom.

CONSEQUENCE : un PLAFOND energetique par domaine E_dom = a V T^4 --
remarquablement, E_dom tombe SUR les masses du reservoir sombre.

TROIS REGIMES DE FORMATION (la question de fond) :
  (EQ)  equilibre jusqu'au verrouillage (reconnexion active jusqu'a
        T_lock = 0.877 T_c, lu sur reseau) : w = N e^{-m/T_lock} --
        l'equilibre EFFACE la cinetique (pas de facteur alpha).
  (KZ)  trempe pure de Kibble-Zurek : l'energie du defaut est PIEGEE
        (gradients du condensat), pas payee par le fond thermique : w = N e^{-alpha c}
        (combinatoire/cinetique seulement, aucun Boltzmann).
  (KL)  nucleation cinetiquement limitee (les secteurs haut-c, reconnexion
        Vassiliev-ecrasee, n'equilibrent jamais ; les bas-c oui) :
        w = N e^{-alpha c} x [poids de fond thermique], le poids de fond thermique etant
        Boltzmann (fond thermique infini) ou microcanonique (domaine fini).
"""
import math
from simulation.experiments.omega_true_Nc import N_true, m_geo

T_C = 0.067
T_LOCK = 0.877 * T_C
ALPHA = 1.217
KAPPA = 1.14
ETA_B = 6.12e-10
S0_NG0 = 7.04
M_P = 0.93827
R_INF = 6.5
CMAXS = 150


def E_dom(g_s, tc=T_C, kappa=KAPPA):
    a = g_s * math.pi ** 2 / 30
    V = (4 * math.pi / 3) * (kappa / tc) ** 3
    return a * V * tc ** 4


def w_mc(m, Ed, tc):
    """Poids microcanonique du fond thermique fini (radiation)."""
    if m >= Ed:
        return 0.0
    return math.exp((4.0 / 3.0) * (Ed / tc) * ((1 - m / Ed) ** 0.75 - 1))


def omega(weight_fn, g_s, kappa=KAPPA):
    """Budget asymetrique avec un poids de formation arbitraire."""
    Z = sum(weight_fn(c) for c in range(3, CMAXS))
    Sd = sum(weight_fn(c) for c in range(23, CMAXS))
    if Sd == 0 or Z == 0:
        return 0.0, 0.0
    mbar = sum(weight_fn(c) * m_geo(c) for c in range(23, CMAXS)) / Sd
    n_dom_s = 135.0 / (8 * math.pi ** 3 * g_s * kappa ** 3)
    return n_dom_s * (Sd / Z) * S0_NG0 * mbar / (ETA_B * M_P), mbar


def run():
    print("=== Boltzmann a travers la TQNT : le domaine fini ===\n")

    print("[1] Le plafond energetique par domaine (E_dom = aVT^4) :")
    print("    g_s     E_dom(GeV)   c_plafond (m_geo = E_dom)")
    for g in (10.75, 12.0, 14.0, 15.5, 17.25):
        Ed = E_dom(g)
        c_ceil = 3 + (Ed * 1000 / 18.936 - 16.372) / 3.708
        print("    %5.2f     %5.2f        %5.1f" % (g, Ed, c_ceil))
    print("    -> le plafond d'UN domaine tombe SUR le reservoir (c_min=23,")
    print("       m=1.71 GeV) : la correction microcanonique n'est PAS un")
    print("       raffinement -- elle decide de l'existence meme du reservoir.\n")

    print("[2] Poids microcanonique vs Boltzmann (g_s = 14, E_dom = %.2f GeV) :" % E_dom(14))
    Ed = E_dom(14)
    for c in (3, 18, 23, 25, 26):
        m = m_geo(c)
        wb = math.exp(-m / T_C)
        wm = w_mc(m, Ed, T_C)
        print("    c=%2d  m=%.2f : Boltzmann %.2e ; microcanonique %.2e (x%.1e)" % (
            c, m, wb, wm, (wm / wb if wb > 0 else 0)))
    print()

    print("[3] Le budget asymetrique sous les trois regimes (et deux bains) :")
    rows = [
        ("(KZ) trempe pure  w = N e^-ac", lambda c: N_true(c, R_INF) * math.exp(-ALPHA * c), 14.0),
        ("(EQ) equilibre @T_lock  w = N e^-m/Tl", lambda c: N_true(c, R_INF) * math.exp(-m_geo(c) / T_LOCK), 14.0),
        ("(KL) cinetique x Boltzmann (budget 16/07)", lambda c: N_true(c, R_INF) * math.exp(-ALPHA * c - m_geo(c) / T_C), 14.0),
    ]
    for lab, fn, g in rows:
        om, mbar = omega(fn, g)
        # divergence check
        net = fn(60) / fn(59) * 1.0
        tag = "DIVERGE (net %.2f)" % net if net > 1 else ("Omega = %.3g (mbar %.2f GeV)" % (om, mbar))
        print("    %-42s %s" % (lab, tag))
    print()

    print("[4] (KL) avec fond thermique FINI (microcanonique, scan g_s) :")
    print("    g_s    E_dom   c_plafond   Omega      mbar(GeV)")
    for g in (10.75, 12.0, 13.0, 14.0, 15.0, 16.0, 17.25):
        Ed = E_dom(g)
        fn = lambda c: N_true(c, R_INF) * math.exp(-ALPHA * c) * w_mc(m_geo(c), Ed, T_C)
        om, mbar = omega(fn, g)
        c_ceil = 3 + (Ed * 1000 / 18.936 - 16.372) / 3.708
        print("    %5.2f   %5.2f    %5.1f     %9.3g    %s" % (
            g, Ed, c_ceil, om, ("%.2f" % mbar) if om > 0 else "--"))
    print()

    print("VERDICT (fondations) :")
    print(" - Boltzmann e^{-m/T} est DERIVABLE en TQNT (fond thermique d'ondes chi +")
    print("   melange par la danse [2e loi deja document principal] + T emergente) -- mais")
    print("   seulement comme LIMITE fond thermique-infini. La TQNT impose le fond thermique FINI")
    print("   du domaine de correlation : poids microcanonique + PLAFOND E_dom.")
    print(" - Coincidence structurelle : E_dom (1.5-2.4 GeV selon g_s) tombe")
    print("   exactement sur le bord du reservoir (m(23) = 1.71 GeV). Le sort")
    print("   du reservoir depend du g_s effectif du fond thermique chi a T_c -- a")
    print("   deriver proprement (quels modes comptent : photons seuls ?")
    print("   ondes chi ? neutrinos decouples ?).")
    print(" - Les regimes purs echouent : (KZ) diverge (1e21) ; (EQ) diverge")
    print("   aussi (l'equilibre efface alpha ; net 6.5*e^-1.19 = 2.0) --")
    print("   la reconnexion Vassiliev-ecrasee des hauts-c JUSTIFIE que")
    print("   l'equilibre ne s'etende pas au reservoir : le poids composite")
    print("   (KL) est le seul coherent avec le verrouillage lu sur reseau.")
    print(" - Selon le fond thermique (infini vs fini), Omega = 15.6 vs la table [4] :")
    print("   la question 'qui paie la masse du noeud' (fond thermique radiatif vs")
    print("   chaleur latente du condensat ~27x plus dense) est LA question")
    print("   ouverte des fondations.")


if __name__ == "__main__":
    run()
