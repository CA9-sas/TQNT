"""Le noyau azimutal — le juge de la participation du mode nodal
(27/07/2026, réponse à « pourquoi ~1/5 de la circulation ? »).

MÉCANISME CANDIDAT : l'incompressibilité topologique (rho verrouillée
par le framing) impose par continuité d I_long/ds = 0 — le courant
LONGITUDINAL ne peut pas se moduler le long du circuit. Le mode de
génération ne peut vivre que dans la composante AZIMUTALE du courant
de framing (le pas hélicoïdal des lignes de White) :
    tan(theta) = 2 pi a |Tw| / L,  Tw = Lk - Wr = -2,42  (tout dérivé)
    => theta ~ 24,9 deg ; fraction d'énergie naïve tan²theta = 0,211
       vs participation requise m_e/E_complet = 0,224 (-4 %).
LE JUGE : la « fraction naïve » suppose que le noyau azimutal vaut le
noyau longitudinal. On calcule ICI le vrai noyau : l'énergie de la
modulation n=3 des ANNEAUX azimutaux sur le tube idéal —
    E_phi = (1/2) pref · s²(theta) · G_3,
    G_3 = (1/L) ∮∮ ds ds' cos(3sigma) cos(3sigma')
              <e_phi·e_phi' / |X - X'|>_(phi,phi'),
même normalisation que le calcul longitudinal (K_33 : remplacer
e_phi·e_phi' par t·t' redonne exactement K_33) ; s(theta) = tan theta
(réorientation à courant longitudinal fixé) ou sin theta (projection)
— les deux variantes chiffrées, la distinction déclarée.

CONTRÔLES : (a) contrôle de normalisation t·t' -> K_33 retrouvé ;
(b) convergence en résolution (n, m) ; (c) l'auto-terme des anneaux
(i=j exclu) tend vers zéro dans le continu (slab ~ ds²) — vérifié
par la convergence.

VERDICT : m_e-candidat azimutal vs m_e = 0,511 MeV. Règle du test
nul : rien n'est retenu au-delà de ce que les nombres montrent.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve
from simulation.experiments.derive_profil_tube_k33 import frames

M_E = 0.51099895
ALPHA = 1 / 137.035999
HBARC = 197.32698
MU = 17.715562
M_L = MU ** 2
SQRT_SIGMA = 418.47
WR_IDEAL = 3.4165
LK = 1


def kernels(P, a, m=12, mode="azimutal"):
    """G_3 (et G_0) : noyau azimutal (ou longitudinal en contrôle)."""
    P = np.asarray(P, float)
    P = P - P.mean(0)
    t, u, v, ds = frames(P)
    n = len(P)
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    sig = 2 * np.pi * s / L
    phi = 2 * np.pi * np.arange(m) / m
    # points de coquille et vecteurs azimutaux
    X = (P[:, None, :] + a * np.cos(phi)[None, :, None] * u[:, None, :]
         + a * np.sin(phi)[None, :, None] * v[:, None, :])       # (n, m, 3)
    E = (-np.sin(phi)[None, :, None] * u[:, None, :]
         + np.cos(phi)[None, :, None] * v[:, None, :])           # (n, m, 3)
    G3 = 0.0
    G0 = 0.0
    for i in range(n):
        diff = X[i][:, None, None, :] - X[None, :, :, :]          # (m,n,m,3)
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        if mode == "azimutal":
            dots = np.einsum('kc,jlc->kjl', E[i], E)              # (m,n,m)
        else:
            dots = np.broadcast_to((t[i] @ t.T)[None, :, None],
                                   (m, n, m))
        g = (dots / d).sum(axis=(0, 2)) / m ** 2                  # (n,)
        if mode == "azimutal":
            g[i] = 0.0                                            # i=j exclu
        c3 = np.cos(3 * (sig[i] - sig))
        w = ds[i] * ds
        G3 += (c3 * w * g).sum()
        G0 += (w * g).sum()
    return G3 / L, G0 / L, L


def main():
    print(__doc__)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]
    L_fm = M_L / SQRT_SIGMA ** 2 * HBARC
    pref = ALPHA * HBARC / (2 * L_fm)
    tw = abs(LK - WR_IDEAL)

    print("=" * 78)
    print("CONTRÔLES")
    print("=" * 78)
    P240 = resample(P_id, 240)
    dl = np.roll(P240, -1, 0) - P240
    a240 = np.linalg.norm(dl, axis=1).sum() / ell
    G3c, G0c, _ = kernels(P240, a240, 12, mode="controle")
    print("  [a] normalisation (t·t' au lieu de e_phi) : G_3 = %.4f"
          " (K_33 attendu ~2,09 coquille non torsadée : %+.1f %%)" % (
              G3c, 100 * (G3c / 2.0879 - 1)))
    res = {}
    for nres, m in [(160, 8), (240, 12), (320, 16)]:
        Pr = resample(P_id, nres)
        dlr = np.roll(Pr, -1, 0) - Pr
        ar = np.linalg.norm(dlr, axis=1).sum() / ell
        G3, G0, _ = kernels(Pr, ar, m)
        res[(nres, m)] = G3
        print("  [b] convergence n=%3d, m=%2d : G_3(azimutal) = %.4f ;"
              " G_0 = %.4f" % (nres, m, G3, G0))

    print("=" * 78)
    print("LE VERDICT")
    print("=" * 78)
    G3 = res[(320, 16)]
    theta = np.arctan(2 * np.pi * a240 * tw / (a240 * ell))
    for lab, s2 in [("tan²theta (courant long. fixé)", np.tan(theta) ** 2),
                    ("sin²theta (projection)", np.sin(theta) ** 2)]:
        me = 0.5 * pref * s2 * G3
        print("  %-32s = %.4f : m_e-candidat = %.4f MeV (%+.1f %%)" % (
            lab, s2, me, 100 * (me / M_E - 1)))
    print("  (rappel : participation requise 0,224 ; G_3/K_33_long = %.3f)"
          % (G3 / 2.2427))
    print("""
  LECTURE : le mécanisme (incompressibilité -> canal azimutal seul)
  fixe la STRUCTURE m_e = s²(theta)·(1/2)pref·G_3 avec s(theta)
  entièrement dérivé (White + charge + ropelength). Le verdict
  décimal est dans les chiffres ci-dessus — et la distinction
  tan/sin ainsi que le rapport G_3/K_33 sont désormais des questions
  GÉOMÉTRIQUES calculées, plus des suppositions. Règle du test nul :
  rien n'est retenu au-delà.""")


if __name__ == "__main__":
    main()
