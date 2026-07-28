"""Premier assaut sur epsilon — les canaux EM du mode nodal
(27/07/2026, suite de la consolidation magnétisme-rails).

CIBLE : epsilon <=> m_e/m_l = 1,63e-3 — la masse résiduelle de
l'électron comme énergie EM du mode nodal (le nœud tue le canal
condensat, pas le canal charge).

ÉCHELLES (toutes ÉTABLIES, aucun choix) :
  - L = m_l / sigma_chi (la relation longueur-masse du chantier DM,
    L_phys = M/sigma_chi) ; m_l = mu^2 = 313,84 MeV ;
  - rayon de tube r = L / l_ACPR(3_1) (convention longueur/rayon) ;
    contrôle : D = 2r doit redonner le D_plateau = 0,0214 fm établi.

DEUX CANAUX pour une charge e sur le circuit (géométrie trèfle du
catalogue, coupure au rayon de tube — plongement NON idéal, déclaré):
 [1] COULOMB (charge répartie sur le circuit) :
        E_C = (alpha hbar c / 2 L^2) * J,   J = ∫∫' ds ds'/|r-r'| ;
 [2] MAGNÉTIQUE (courant persistant I = ec/L — le canal du SENS,
     celui de la lecture) :
        E_M = (alpha hbar c / 2 L^2) * N,   N = ∫∫' t·t' ds ds'/|r-r'|
     (même préfacteur : E_M/E_C = N/J, purement géométrique).

PLUS UN CANDIDAT DE FERMETURE à tester (et à juger au test nul) :
 [3] m_e = delta · alpha · m_l — dimensionnellement licite,
     mécaniquement obscur ; en système FERMÉ avec la forme de Koide
     (mu, A=sqrt(2), alpha seuls), il prédit delta SANS paramètre :
     à résoudre exactement et à confronter.

GARDE-FOUS : plongement de calcul non idéal (les intégrales J, N
portent un log doux — ordre de grandeur seulement) ; règle du test
nul sur [3].
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import brentq

from theory.omega_ts import _knot_curve      # helper de COURBE seulement
from theory.ropelength_acpr import ROPELENGTH_ACPR

M_E = 0.51099895                 # MeV
ALPHA = 1 / 137.035999
HBARC = 197.32698                # MeV fm
MU = 17.715562                   # MeV^1/2 (inversion Koide exacte)
M_L = MU ** 2                    # = 313,84 MeV = (3/4) sqrt(sigma_chi)
SQRT_SIGMA = 418.47              # MeV
D_PLATEAU = 0.0214               # fm (établi, chantier DM)
DELTA_MES = 0.222230             # rad (inversion exacte)


def geom_integrals(n_pts=900):
    """J/L (Coulomb) et N/L (Neumann) sur le trèfle, coupure au rayon."""
    P = np.asarray(_knot_curve("3_1", n_pts), float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    P = P - P.mean(0)
    dl = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dl, axis=1)
    t = dl / ds[:, None]
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    a = L / ROPELENGTH_ACPR["3_1"]          # rayon de tube
    R = P + 0.5 * dl
    J = N = 0.0
    for i in range(len(P)):
        dv = R[i] - R
        dist = np.linalg.norm(dv, axis=1)
        darc = np.abs(s - s[i])
        darc = np.minimum(darc, L - darc)
        ok = (darc > a) & (dist > 1e-12)
        J += (ds[i] * ds[ok] / dist[ok]).sum()
        N += ((t[i] @ t[ok].T) * ds[i] * ds[ok] / dist[ok]).sum()
    return J / L, N / L


def main():
    print(__doc__)
    print("=" * 78)
    print("[0] LES ÉCHELLES (établies — contrôle de cohérence)")
    print("=" * 78)
    sigma = SQRT_SIGMA ** 2                       # MeV^2
    L_mev = M_L / sigma                           # MeV^-1
    L_fm = L_mev * HBARC
    r_fm = L_fm / ROPELENGTH_ACPR["3_1"]
    print("  L = m_l/sigma_chi = %.4f fm ; r_tube = L/l_ACPR = %.5f fm"
          % (L_fm, r_fm))
    print("  contrôle : D = 2r = %.4f fm vs D_plateau établi %.4f fm (%+.1f %%)"
          % (2 * r_fm, D_PLATEAU, 100 * (2 * r_fm / D_PLATEAU - 1)))
    pref = ALPHA * HBARC / (2 * L_fm)             # MeV (x [intégrale/L])
    print("  préfacteur commun alpha·hbar c/(2L) = %.4f MeV" % pref)

    print("=" * 78)
    print("[1][2] LES DEUX CANAUX (géométrie trèfle, coupure au tube)")
    print("=" * 78)
    JL, NL = geom_integrals()
    E_C = pref * JL
    E_M = pref * NL
    print("  J/L (Coulomb)  = %.3f  ->  E_C = %.3f MeV  (m_e x %.1f)"
          % (JL, E_C, E_C / M_E))
    print("  N/L (Neumann)  = %.3f  ->  E_M = %.3f MeV  (m_e x %.2f)"
          % (NL, E_M, E_M / M_E))
    print("""
  LECTURE : le canal COULOMB nu sur-tire d'un ordre (x%.0f) — une
  charge statique répartie sur le circuit N'EST PAS l'électron
  (réfuté comme source seule ; l'écrantage/le partage doit le
  supprimer). Le canal MAGNÉTIQUE — celui du SENS, exactement la
  grandeur de la lecture consolidée — donne %.2f MeV : LE BON ORDRE
  (x%.1f, sur plongement non idéal et sans partage de canaux).
  L'ordre de grandeur soutient : la masse résiduelle de l'électron
  = l'énergie magnétique de sa propre circulation. La décimale
  demande la géométrie idéale + le partage Coulomb/magnétique +
  la participation du mode (cibles).""" % (E_C / M_E, E_M, E_M / M_E))

    print("=" * 78)
    print("[3] LE CANDIDAT DE FERMETURE m_e = delta·alpha·m_l (test nul)")
    print("=" * 78)
    f = lambda d: 1 + np.sqrt(2) * np.cos(2 * np.pi / 3 + d)

    def closure(d):
        return f(d) - np.sqrt(ALPHA * d)
    d_star = brentq(closure, 0.1, 0.26)
    me_star = ALPHA * d_star * M_L
    print("  système fermé {Koide(mu, sqrt2) ; m_e = delta alpha m_l} :")
    print("    delta* = %.6f  vs mesuré %.6f  (%+.2f %%)  vs 2/9 (%+.2f %%)"
          % (d_star, DELTA_MES, 100 * (d_star / DELTA_MES - 1),
             100 * (d_star / (2 / 9) - 1)))
    print("    m_e*   = %.5f MeV  vs %.5f  (%+.2f %%)" % (
        me_star, M_E, 100 * (me_star / M_E - 1)))
    print("""
  VERDICT (règle du test nul) : le candidat est auto-cohérent et
  tombe à 0,03 %% de delta — mais delta est une variable RAIDE
  (dm/m ~ -50·d delta près du nœud) : la vraie jauge est m_e, manqué
  de -0,4 %%. Et la relation « masse = angle x alpha x base » n'a
  AUCUN mécanisme (pourquoi la fraction EM par radian d'écart
  nodal ?) ; sa précision (10^-3) est très en deçà de la ligue Koide
  (61 ppm) : NON RETENU — consigné comme coïncidence remarquable à
  re-examiner seulement si un mécanisme la produit.""")

    print("=" * 78)
    print("BILAN DU PREMIER ASSAUT")
    print("=" * 78)
    print("""   - cohérence d'échelles gratuite : D = 2L/l_ACPR redonne le
     D_plateau établi (%+.1f %%) ;
   - Coulomb nu RÉFUTÉ comme source seule (x%.0f) ;
   - le canal du SENS (magnétique) donne l'ORDRE de m_e (x%.1f,
     %.2f MeV) — la cible epsilon est physiquement plausible dans
     le canal même que la lecture consolidée désigne ;
   - fermeture delta·alpha·m_l : coïncidence (m_e à -0,4 %%), sans
     mécanisme, non retenue ;
   - cibles nommées : géométrie idéale (SONO), partage des canaux,
     participation du mode nodal.""" % (
        100 * (2 * r_fm / D_PLATEAU - 1), E_C / M_E, E_M / M_E, E_M))


if __name__ == "__main__":
    main()
