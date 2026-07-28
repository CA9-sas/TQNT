"""Front T_form : l'etablissement de sigma(T) comme horloge (20/07).

L'IDEE : plus de T_form pose. Sous la transition (T_trans ~ 155 MeV,
premiere main : sigma_Cornell = sigma_chi => la transition chi EST le
croisement QCD), tout court :

  sigma(T)/sigma_chi = (1 - T/T_trans)^(2 nu)   [Ising 3D, nu=0,6301 ;
                                                 classe confirmee reseau]
  xi(T) * T = kappa_dom                          [etirement COMOBILE du
                                                 reseau gele : xi ~ a ~ 1/T
                                                 -> xi.T conserve -- la
                                                 relation du cadre, DERIVEE]
  m(T)      = m_edge * sqrt(sigma/sigma_chi)     [loi de masse geometrique]
  E_frag(T) = sigma(T) * L * xi(T)               [barriere de re-fonte du
                                                 fragment porteur, L mesure]

et le gel est CINETIQUE (equation de Boltzmann, cliquet radiatif) :
  dw/dt = -Gamma_d (w - w_eq),  Gamma_d = T e^{-E_frag/T},
  w_eq = e^{-m(T)/T},  H = 1,66 sqrt(g_s) T^2 / M_Pl.

AUCUN T_form : la seule echelle d'entree est T_trans (premiere main).
Sortie : T de gel EMERGENT par fragment + Omega.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.pin_x17_fragments_porteurs import batch  # noqa: E402

M_PL = 1.2209e19
G_S = 15.2
SIGMA = 0.41847 ** 2
KAPPA = 1.14
M_EDGE = 1.73
NU_ISING = 0.6301
T_REF = 0.067          # convention de normalisation de Omega = 3,9


def evolve(L, t_trans, expo):
    """Gel cinetique d'un fragment porteur de longueur L (xi) ;
    retourne (w_final, T ou le suivi decroche)."""
    t_start = 0.98 * t_trans
    t_stop = 0.015
    n = 3000
    du = math.log(t_start / t_stop) / n
    t = t_start
    x = 1 - t / t_trans
    w = math.exp(-M_EDGE * math.sqrt(x ** expo) / t)
    t_freeze = None
    for k in range(n):
        t = t_start * math.exp(-(k + 0.5) * du)
        x = max(1e-12, 1 - t / t_trans)
        s_ratio = x ** expo
        m_t = M_EDGE * math.sqrt(s_ratio)
        e_frag = SIGMA * s_ratio * L * KAPPA / t
        g_over_h = M_PL * math.exp(-e_frag / t) / (1.66 * math.sqrt(G_S) * t)
        w_eq = math.exp(-m_t / t)
        w_new = w_eq + (w - w_eq) * math.exp(-g_over_h * du)
        if t_freeze is None and g_over_h < 1.0:
            t_freeze = t
        w = w_new
    return w, (t_freeze if t_freeze else t_stop)


def run():
    print("=== sigma(T) comme horloge : gel dynamique, sans T_form ===\n")
    finals = batch((42, 43), 2026) + batch((44, 45), 3026)
    L_bear = np.array([L for L, nn in finals if nn >= 1])
    w_ref = math.exp(-M_EDGE / T_REF)

    print("Entrees : T_trans (premiere main), exposant d'Ising 2nu = 1,26,")
    print("etirement comobile (xi.T = kappa_dom), distribution L mesuree,")
    print("cliquet radiatif + cinetique. AUCUN T_form.\n")

    print("%-34s %-12s %-10s %s" % ("configuration", "T_gel median",
                                    "Omega", "(Planck 5,45)"))
    for lab, t_trans, expo in (
            ("T_trans = 155 MeV, Ising", 0.155, 2 * NU_ISING),
            ("T_trans = 145 MeV, Ising", 0.145, 2 * NU_ISING),
            ("T_trans = 165 MeV, Ising", 0.165, 2 * NU_ISING),
            ("T_trans = 155, champ moyen (2nu=1)", 0.155, 1.0),
            ("T_trans = 297 (SU(2) pur), Ising", 0.297, 2 * NU_ISING)):
        ws, tf = [], []
        for L in L_bear:
            w, t_f = evolve(L, t_trans, expo)
            ws.append(w)
            tf.append(t_f)
        om = 3.9 * float(np.mean(ws)) / w_ref
        print("%-34s %5.1f MeV    %8.2f   (%+.0f %%)" % (
            lab, np.median(tf) * 1e3, om, 100 * (om / 5.45 - 1)))

    print("\nINVERSION DU SERRAGE (le gel = l'allumage de la protection) :")
    print("   modele : barriere ~0 tant que le tube est gras ; au serrage")
    print("   (sigma/sigma_chi = x*), protection Vassiliev/CS -> gel")
    print("   immediat au poids w_eq(T*(x*)). Planck exige :")
    w_target = 5.45 / 3.9 * w_ref
    for x_star in (0.5, 0.55, 0.6, 0.65, 0.7):
        t_star = 0.155 * (1 - x_star ** (1 / (2 * NU_ISING)))
        m_t = M_EDGE * math.sqrt(x_star)
        om = 3.9 * math.exp(-m_t / t_star) / w_ref
        print("   x* = %.2f : T* = %4.1f MeV, D/D0 = %.2f -> Omega = %6.2f" % (
            x_star, t_star * 1e3, 1 / math.sqrt(x_star), om))
    print("   -> Planck pince x* ~ 0,60 (T* ~ 52 MeV, tube a 1,29 D0).")
    d0_sig = 0.1084 * SIGMA  # sigma_chi * D0 (GeV)
    x_p = 0.60
    t_p = 0.155 * (1 - x_p ** (1 / (2 * NU_ISING)))
    e_pass = d0_sig * math.sqrt(x_p)
    print("   Critere candidat : cout d'ouverture d'un passage ~")
    print("   c_p sqrt(sigma sigma_chi) D0 = c_p x %.1f MeV = T* ->" % (
        e_pass * 1e3))
    print("   c_p = %.1f requis (O(1) naturel, a deriver).\n" % (t_p / e_pass))

    print("LECTURE :")
    print(" - le gel EMERGE de la course entre le developpement de la")
    print("   tension (Ising sous T_trans) et le refroidissement : la")
    print("   barriere E_frag(T) = sigma(T) L kappa/T croit, la re-fonte")
    print("   decroche, le poids fige -- sans aucun T_form pose ;")
    print(" - l'etirement comobile DERIVE la relation xi.T = kappa_dom du")
    print("   cadre (elle etait employee, elle est maintenant fondee) ;")
    print(" - la seule echelle d'entree est T_trans, de premiere main.")


if __name__ == "__main__":
    run()
