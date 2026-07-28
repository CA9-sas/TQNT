"""Torsion propre du 5_2 ideal au bout d'un tube en rotation (SONO) (21/07).

CIBLE : le shift s = 0,3577 +- 0,0023 (SS18) -- candidat geometrique :
la capacite differentielle de writhe du noeud compagnon (5_2 vs 3_1).
MODELE : une revolution du tube injecte un tour de twist au bout ; le
noeud absorbe une part en WRITHE (Calugareanu Lk = Tw + Wr) -- l'exces
stocke par rapport au trefle est fixe par la geometrie IDEALE.

CALCUL : plongements ideaux par SONO (outillage valide du depot :
3_1 +0,8 %, Hopf +0,07 %) ; writhe par integrale de Gauss ; torsion
totale de Frenet en second candidat. Controle : ratio de ropelength
l(5_2)/l(3_1) vs le document principal 1,596 (= m_s/m_l DERIVE).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from simulation.ideal_embedding import sono, ropelength, resample
from simulation.experiments.repair_curves_braid import braid_curve

S_MES, S_SIG = 0.3577, 0.0023


def torus_trefoil(n=240):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R, r = 2.0, 1.0
    x = (R + r * np.cos(3 * t)) * np.cos(2 * t)
    y = (R + r * np.cos(3 * t)) * np.sin(2 * t)
    z = r * np.sin(3 * t)
    return np.column_stack([x, y, z]) * 3.0


def writhe(P):
    """Integrale de Gauss (approximation des segments par leurs milieux)."""
    n = len(P)
    T = np.roll(P, -1, 0) - P
    M = P + 0.5 * T
    W = 0.0
    for i in range(n):
        d = M[i] - M                       # r_ij
        cr = np.cross(T[i], T)             # t_i x t_j ds^2
        num = np.einsum('ij,ij->i', cr, d)
        r3 = np.linalg.norm(d, axis=1) ** 3
        mask = np.ones(n, bool)
        for k in (-2, -1, 0, 1, 2):
            mask[(i + k) % n] = False
        W += np.sum(num[mask] / r3[mask])
    return W / (4 * np.pi)


def total_torsion(P):
    """Torsion de Frenet integree / 2pi (courbe lissee)."""
    Q = resample(P, 4 * len(P))
    # lissage doux (moyenne mobile circulaire)
    for _ in range(8):
        Q = 0.25 * np.roll(Q, 1, 0) + 0.5 * Q + 0.25 * np.roll(Q, -1, 0)
    d1 = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0)) / 2
    d2 = np.roll(Q, -1, 0) - 2 * Q + np.roll(Q, 1, 0)
    d3 = (np.roll(Q, -2, 0) - 2 * np.roll(Q, -1, 0)
          + 2 * np.roll(Q, 1, 0) - np.roll(Q, 2, 0)) / 2
    c12 = np.cross(d1, d2)
    num = np.einsum('ij,ij->i', c12, d3)
    den = np.einsum('ij,ij->i', c12, c12)
    ok = den > 1e-12 * np.max(den)
    return float(np.sum(num[ok] / den[ok])) / (2 * np.pi)


print(__doc__)
print("=" * 76)
print("[A] PLONGEMENTS IDEAUX (SONO)")
print("=" * 76)
res = {}
for nom, P0 in [("3_1", torus_trefoil()),
                ("5_2", resample(braid_curve([1, 1, 1, 2, -1, 2], 3)[0], 240))]:
    P, info = sono(P0, iters=12000)
    rl = ropelength(P)
    wr = writhe(P)
    tt = total_torsion(P)
    res[nom] = (rl, wr, tt)
    print("  %s : ropelength = %.2f ; Wr = %.4f ; torsion totale/2pi = %.3f"
          % (nom, rl, wr, tt))
r_rl = res["5_2"][0] / res["3_1"][0]
print("  controle : l(5_2)/l(3_1) = %.3f (document principal m_s/m_l = 1,596 ; %+.1f %%)"
      % (r_rl, 100 * (r_rl / 1.596 - 1)))
print("  controle : Wr(3_1) vs quasi-quantisation 24/7 = %.3f ; Wr(5_2) vs"
      % (24 / 7))
print("  32/7 = %.3f (Stasiak)." % (32 / 7))

print("\n" + "=" * 76)
print("[B] CANDIDATS GEOMETRIQUES POUR s (mesure %.4f +- %.4f)" % (S_MES, S_SIG))
print("=" * 76)
# chiralites coherentes : |Wr| (le trefle du tore sort gaucher)
dwr = abs(res["5_2"][1]) - abs(res["3_1"][1])
# NB : l'estimateur de torsion de Frenet est NON FIABLE sur les courbes
# ideales (segments droits, contacts) -- retire des candidats.
for nom, v in [("dWr mesure / pi (5_2 non converge)", dwr / np.pi),
               ("(8/7)/pi [quasi-quant. Stasiak]", (8 / 7) / np.pi),
               ("dWr litterature (1,142)/pi", 1.142 / np.pi)]:
    print("  %-36s %.4f  (%+6.1f %%, %5.1f sigma)"
          % (nom, v, 100 * (S_MES / v - 1) if v else float('nan'),
             abs(S_MES - v) / S_SIG))
print("\n  LECTURE : candidat s = dWr_ideal/pi (Calugareanu : une revolution")
print("  injecte un tour de twist ; l'exces de writhe ideal du 5_2 vs 3_1")
print("  s'absorbe au taux 1/pi en demi-tours). Avec dWr ideal = 8/7 :")
print("  0,3638 (-1,7 %, 2,6 sigma) -- LE MEILLEUR candidat a mecanisme.")
