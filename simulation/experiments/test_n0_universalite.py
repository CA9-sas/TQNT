"""Mécanisme de n0 ~ 1 : test d'universalité (écrantage de contact).
n0 = Neumann/longueur - terme azimutal clos 2pi^2(Lk/x)^2, pour les
nœuds IDÉAUX 3_1, 4_1, 5_1 + contrôle cercle. Si n0 ~ 1 partout ->
propriété du SERRAGE (mécanisme) ; sinon -> coïncidence du trèfle (null).
"""
import sys; sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
import numpy as np, os
from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve

SCR = '/tmp/claude-1000/-home-euk9-projets-physique-nouvelle/4281778e-628f-46db-80e6-4e8c179f6e45/scratchpad'

def frames(P):
    dP = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dP, axis=1)
    t = dP / ds[:, None]
    u = np.zeros_like(P)
    v0 = np.array([0., 0., 1.]); v0 -= (v0 @ t[0]) * t[0]
    u[0] = v0 / np.linalg.norm(v0)
    for i in range(1, len(P)):
        w = u[i-1] - (u[i-1] @ t[i]) * t[i]
        u[i] = w / np.linalg.norm(w)
    return t, u, np.cross(t, u), ds

def writhe(P):
    dP = np.roll(P, -1, 0) - P
    mid = P + 0.5 * dP
    W = 0.0
    for i in range(len(P)):
        r = mid[i] - mid
        d = np.linalg.norm(r, axis=1); d[i] = np.inf
        W += np.sum(np.einsum('j,ij->i', dP[i], np.cross(dP, r, axis=1)) / d**3)
    return W / (4 * np.pi)

def neumann(P, a, m, k_extra):
    t, u, v, ds = frames(P)
    n = len(P)
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    w_tr = u[-1] - (u[-1] @ t[0]) * t[0]
    w_tr /= np.linalg.norm(w_tr)
    th_H = np.arctan2(w_tr @ np.cross(t[0], u[0]), w_tr @ u[0])
    rot = 2 * np.pi * k_extra - th_H
    ph = 2 * np.pi * np.arange(m)[None, :] / m + rot * (s_arc / L)[:, None]
    X = (P[:, None, :] + a * np.cos(ph)[:, :, None] * u[:, None, :]
         + a * np.sin(ph)[:, :, None] * v[:, None, :])
    dl = np.roll(X, -1, axis=0) - X
    Mid = X + 0.5 * dl
    Nh = 0.0
    for i in range(n):
        diff = Mid[i][:, None, None, :] - Mid[None, :, :, :]
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        dot = np.einsum('kc,jlc->kjl', dl[i], dl)
        Nh += (dot / d).sum() / m**2
    return Nh / L, rot / (2 * np.pi)

print("=" * 72)
print("n0 — TEST D'UNIVERSALITÉ SUR LES NŒUDS IDÉAUX  (2026-08-28)")
print("=" * 72)

results = {}
for knot in ("3_1", "4_1", "5_1"):
    cache = f"{SCR}/{knot}_ideal_sono.npz"
    if os.path.exists(cache):
        z = np.load(cache); P_id, ell = z['P'], float(z['ell'])
    elif knot == "3_1":
        z = np.load(f"{SCR}/trefle_ideal_sono.npz"); P_id, ell = z['P'], float(z['ell'])
    else:
        P0 = np.asarray(_knot_curve(knot, 200), float)
        P_id, info = sono(P0, iters=20000)
        ell = info["ropelength"]
        np.savez(cache, P=P_id, ell=ell)
    for NN in (240, 480):
        P = resample(P_id, NN)
        L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
        a = L / ell
        Wr = writhe(P)
        # fermeture à rotation minimale -> Lk réalisé = Tw_appliqué + Wr
        best = None
        for k in range(-8, 9):
            nh, tours = neumann(P, a, 24, k) if False else (None, None)
        # une seule éval suffit : k minimisant |rot|
        t_, u_, v_, ds_ = frames(P)
        w_tr = u_[-1] - (u_[-1] @ t_[0]) * t_[0]; w_tr /= np.linalg.norm(w_tr)
        th_H = np.arctan2(w_tr @ np.cross(t_[0], u_[0]), w_tr @ u_[0])
        k_min = round(th_H / (2*np.pi))
        nh, tours = neumann(P, a, 24, k_min)
        Lk_real = tours + Wr
        n_azim = 2 * np.pi**2 * (Lk_real / ell)**2
        n0 = nh/2 - n_azim
        results[(knot, NN)] = (ell, Wr, Lk_real, nh/2, n_azim, n0)
        print(f"{knot} N={NN}: x={ell:.2f} Wr={Wr:+.3f} Lk_réalisé={Lk_real:+.2f}"
              f" n_brut={nh/2:.4f} azim(Lk²)={n_azim:.4f} -> n0 = {n0:.4f}")

# contact du trèfle : fraction, distance, angle
z = np.load(f"{SCR}/trefle_ideal_sono.npz"); P = resample(z['P'], 480)
L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum(); a = L/float(z['ell'])
t_, _, _, _ = frames(P)
Nn = len(P)
dmin = np.zeros(Nn); ang = np.zeros(Nn)
for i in range(Nn):
    d = np.linalg.norm(P[i] - P, axis=1)
    # exclure les voisins d'arc (|i-j| < Nn/10 modulaire)
    idx = np.arange(Nn); arc = np.minimum((idx-i) % Nn, (i-idx) % Nn)
    d[arc < Nn//10] = np.inf
    j = d.argmin(); dmin[i] = d[j]
    ang[i] = np.degrees(np.arccos(np.clip(abs(t_[i] @ t_[j]), 0, 1)))
print(f"\ncontact du trèfle idéal (480 pts) : distance mini axe-axe "
      f"médiane = {np.median(dmin)/(2*a):.3f} D ; fraction en contact "
      f"(< 1,05 D) = {np.mean(dmin < 1.05*2*a):.2f} ; angle des tangentes "
      f"au contact : médiane {np.median(ang):.1f} deg")
print(f"écrantage naïf : n0 = 2 ln(d_eff/a) -> d_eff = a·e^(n0/2) = "
      f"{np.exp(results[('3_1',480)][5]/2):.2f} a = "
      f"{np.exp(results[('3_1',480)][5]/2)/2:.2f} D")
