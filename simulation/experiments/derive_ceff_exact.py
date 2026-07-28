"""Deriver c_eff exact : le canal de DOUBLE RECONNEXION (20/07).

CORRECTION DE CANAL (le vrai raffinement) : le modele tangente+arc
faisait CONTOURNER le partenaire sans le toucher (canal sans contact,
c_eff = 4,72). Mais le passage physique en TQNT est MEDIE PAR LA
DOUBLE RECONNEXION P15 -- la definition meme du pas DeltaLk = +-1 :
  (1) reconnexion au contact EXISTANT (gratuite : les cœurs se
      touchent deja) -> deux epingles a cheveux ;
  (2) les deux jonctions (pointes) PIVOTENT l'une autour de l'autre
      de phi = 0 a pi, en CONTACT GLISSANT (|T1-T2| = D constant :
      aucun cout de degagement) ;
  (3) re-reconnexion a phi = pi avec le croisement inverse.
La barriere = max sur phi de [surcout de longueur des bras +
travail de pression], minimisee sur l'epinglage s_p.

GEOMETRIE : tresse serree (2,q), angle d'inclinaison 45 deg ->
croisement a 2*alpha = 90 deg ; separation verticale des axes D au
croisement ; pression de contact q = 0,81 (zone-marche, mesuree).

SORTIE : c_eff_reco -> T_c = c_eff x 19,0 MeV -> Omega.
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from simulation.experiments.omega_budget_reconstruction import budget  # noqa: E402

SIGMA_D0 = 0.0190          # sigma_chi * D0 en GeV (19,0 MeV)
P_OCC_EFF = (8.0 / 9.0) * 0.304
Q = 0.81                   # pression (kappa_perp D, zone-marche)
NSAMP = 40                 # echantillons par bras (pression)


def dist_point_line(p, d_hat):
    """Distance du point p a la droite passant par l'origine, direction
    d_hat (les brins originaux passent par le croisement ~origine)."""
    return np.linalg.norm(p - (p @ d_hat) * d_hat)


def barrier(alpha_deg, s_p, q=Q, nphi=61):
    """Barriere du chemin de double reconnexion (unites D = 1)."""
    a = math.radians(alpha_deg)
    uA = np.array([math.cos(a), math.sin(a), 0.0])
    uB = np.array([math.cos(a), -math.sin(a), 0.0])
    zA, zB = 0.5, -0.5     # axes separes de D au croisement
    Am = -s_p * uA + np.array([0, 0, zA])
    Ap = +s_p * uA + np.array([0, 0, zA])
    Bm = -s_p * uB + np.array([0, 0, zB])
    Bp = +s_p * uB + np.array([0, 0, zB])
    L_orig = 4 * s_p

    def arm_cost(p_from, p_to, line_dir, line_z):
        """Longueur du bras + travail de pression (deplacement depuis la
        droite originale, echantillonne)."""
        L = np.linalg.norm(p_to - p_from)
        ts = np.linspace(0, 1, NSAMP)
        pts = p_from[None, :] + ts[:, None] * (p_to - p_from)[None, :]
        # deplacement : distance a la droite originale (axe a hauteur z)
        disp = np.array([dist_point_line(p - np.array([0, 0, line_z]),
                                         line_dir) for p in pts])
        w = q * np.trapezoid(disp, dx=L / (NSAMP - 1))
        return L, w

    worst = 0.0
    for phi in np.linspace(0, math.pi, nphi):
        e = np.array([0.0, math.sin(phi), math.cos(phi)])
        T1 = 0.5 * e          # relie les bras de A- et B+
        T2 = -0.5 * e         # relie B- et A+
        tot_L, tot_W = 0.0, 0.0
        for (pf, pt, ld, lz) in ((Am, T1, uA, zA), (T1, Bp, uB, zB),
                                 (Bm, T2, uB, zB), (T2, Ap, uA, zA)):
            L, w = arm_cost(pf, pt, ld, lz)
            tot_L += L
            tot_W += w
        cost = (tot_L - L_orig) + tot_W
        worst = max(worst, cost)
    return worst


def run():
    print("=== c_eff exact : canal de double reconnexion ===\n")
    print("[1] BARRIERE (q = %.2f) : max sur phi, optimisee sur s_p :" % Q)
    best = None
    for s_p in np.arange(0.6, 4.01, 0.2):
        c = barrier(45.0, s_p)
        if best is None or c < best[1]:
            best = (s_p, c)
    s_opt, c_reco = best
    print("   angle de croisement 90 deg : c_eff = %.2f (s_p* = %.1f D)" % (
        c_reco, s_opt))

    print("\n[2] SENSIBILITES :")
    for a_deg in (30.0, 45.0, 60.0):
        cs = min(barrier(a_deg, s) for s in np.arange(0.6, 4.01, 0.2))
        print("   angle 2a = %3.0f deg : c_eff = %.2f" % (2 * a_deg, cs))
    for qv in (0.69, 0.81, 0.94):
        cs = min(barrier(45.0, s, q=qv) for s in np.arange(0.6, 4.01, 0.2))
        print("   q = %.2f          : c_eff = %.2f" % (qv, cs))

    print("\n[3] PROPAGATION (T_c = c_eff x 19,0 MeV, chaine canonique) :")
    for lab, ce in (("double reconnexion (derive)", c_reco),
                    ("ex-canal sans contact", 4.72),
                    ("cible Planck", 3.5)):
        t_c = ce * SIGMA_D0
        b = budget(tc=t_c, g_s=15.2, p_occ=P_OCC_EFF)
        print("   %-28s c_eff = %.2f -> T_c = %5.1f MeV,"
              " Omega = %8.2f" % (lab, ce, t_c * 1e3, b["omega"]))

    print("\nLECTURE : le canal physique (P15, double reconnexion --")
    print("le pas DeltaLk = +-1 de la marche) est moins couteux que le")
    print("contournement sans contact : les pointes pivotent en contact")
    print("glissant, aucun degagement complet a payer. Les O(1) du")
    print("modele (famille quasi-statique, comptabilite de pression,")
    print("bras droits) sont declares.")


if __name__ == "__main__":
    run()
