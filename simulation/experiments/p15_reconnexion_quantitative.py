"""Reconnexion P15 quantitative : la structure de Fermi SANS constante de couplage.

DERIVATION DE STRUCTURE (TQNT) :
  1. Le sommet de reconnexion est la chirurgie de croisement (echange de brins) :
     operation LOCALE et SANS ECHELLE -> amplitude adimensionnee g_topo.
  2. La transition dLk=+-1 passe par la barriere de relaxation E_relax (2nd ordre) :
     amplitude effective A = g_topo^2 / E_relax^2  (deux sommets, un denominateur^2
     de barriere -- la forme de contact a 4 corps de Fermi).
  3. Espace des phases 3-corps relativiste (cinematique, pas dynamique) : m^5/(192 pi^3).
  => Gamma = g_topo^4 m^5 / (384 pi^3 E_relax^4)
FAIT REMARQUABLE (SM) : G_F = 1/(sqrt2 v^2) ne contient PAS g (il se simplifie via
m_W = g v/2) -- la constante de Fermi ne connait que v. Lecture TQNT : g_topo = 1
EXACTEMENT (pas de constante de couplage faible ; seule la barriere existe).
COMPTAGE DE CANAUX (tau) : N_eff = 2 (leptoniques) + 3 (hadronique en 3 modes Z_3
du trefle -- la lecture TQNT du comptage de 'couleur') = 5.
"""
import math

HBAR = 6.582119569e-25   # GeV.s
V = 246.22               # GeV (E_relax, identifie)
M_MU, M_TAU = 0.1056583755, 1.77686
TAU_MU_PDG, TAU_TAU_PDG = 2.1969811e-6, 2.903e-13


def gamma_p15(m, n_channels=1, g_topo=1.0):
    return g_topo**4 * n_channels * m**5 / (384 * math.pi**3 * V**4)


def run():
    print("=== Reconnexion P15 quantitative : Gamma = g^4 N m^5/(384 pi^3 E_relax^4) ===\n")
    t_mu = HBAR / gamma_p15(M_MU)
    print("muon (N=1, g_topo=1)   : tau = %.4e s vs PDG %.4e (%+.2f%% ; corr. rad. ~0.4%%)" % (
        t_mu, TAU_MU_PDG, (t_mu/TAU_MU_PDG-1)*100))
    t_tau = HBAR / gamma_p15(M_TAU, n_channels=5)
    print("tau  (N=2+3 [Z3], g=1) : tau = %.4e s vs PDG %.4e (%+.1f%%)" % (
        t_tau, TAU_TAU_PDG, (t_tau/TAU_TAU_PDG-1)*100))
    # sensibilite : quel g_topo exact pour le muon ? (mesure de l'unite du sommet)
    g4 = (HBAR / TAU_MU_PDG) / gamma_p15(M_MU)
    print("\nsommet mesure : g_topo = %.5f (unite a %.2f%% pres -- l'ecart = corrections" % (
        g4**0.25, (g4**0.25-1)*100))
    print("radiatives QED, pas un couplage faible)")
    print("\nLECTURE : la 'constante de couplage faible' N'EXISTE PAS en TQNT --")
    print("le sommet de chirurgie est unite ; seule la barriere E_relax (=v) entre.")
    print("Ce qui reste emprunte : (a) la VALEUR de v (mur, 3 voies fermees) ;")
    print("(b) l'espace des phases relativiste 192pi^3 (mur de Lorentz).")
    print("Le comptage N_tau = 2+3 : les 3 modes Z_3 du canal hadronique = la lecture")
    print("TQNT du '3 de couleur' pour ce comptage (lecture, pas derivation).")


if __name__ == "__main__":
    run()
