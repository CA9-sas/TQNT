"""TEST A UN BIT : le mecanisme 45deg avec geometrie entierement ancree.

Forme = plongement IDEAL du trefle (SONO, rayon de tube = 1 par construction,
axe C3 aligne sur z, defaut C3 = 0.053) ; epsilon = 1.0 = LE rayon de tube.
Plus aucune geometrie libre. Le bit : tan^2(theta*) ~ 2 (A = sqrt2) ou non.
Contexte : sur la courbe parametrique, tan^2 dependait d'epsilon (1.30 a 0.20 ;
2.42 a 0.25 ; cible entre les deux a eps*~0.23 en unites de CETTE courbe-la).

Usage : python3 koide45_ideal_anchor.py <n_coarse>
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem  # noqa: E402
from simulation.ideal_embedding import resample  # noqa: E402


def lumped_volumes(points, tets):
    tets = np.asarray(tets)
    p = points[tets]
    e = p[:, 1:] - p[:, :1]
    vols = np.abs(np.einsum('ni,ni->n', e[:, 0], np.cross(e[:, 1], e[:, 2]))) / 24.0
    V = np.zeros(len(points))
    np.add.at(V, tets.ravel(), np.repeat(vols, 4))
    return V


def run(n_coarse):
    curve = resample(np.load('/tmp/ideal_31_aligned.npy'), 500)
    eps = 1.0                       # = rayon de tube ideal, DERIVE (pas un choix)
    tree = cKDTree(curve)
    pts, tets, boundary = generate_mesh(curve, n_coarse=n_coarse, epsilon=eps, refine=True)
    print(f"maillage : {len(pts)} pts", flush=True)
    H, M = assemble_system(pts, tets, boundary, "3_1", tree, energy=0.0, epsilon=eps)
    evals, ev_free = solve_eigenproblem(H, M, n_eigenvalues=6)
    evecs = np.zeros((len(pts), ev_free.shape[1]))
    evecs[~boundary, :] = ev_free
    V = lumped_volumes(pts, tets)

    def nrm(v):
        return v / np.sqrt(np.sum(V * v * v))

    modes = [nrm(evecs[:, k]) for k in range(evecs.shape[1])]
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ptree = cKDTree(pts)
    _, idx = ptree.query(pts @ R.T)
    chars = [float(np.sum(V * v[idx] * v)) for v in modes]
    print("chars :", ["%.2f" % ch for ch in chars], flush=True)
    pair = None
    for k in range(1, len(modes) - 1):
        if (abs(evals[k + 1] - evals[k]) / abs(evals[k]) < 0.05
                and chars[k] < -0.15 and chars[k + 1] < -0.15):
            pair = (k, k + 1)
            break
    if pair is None:
        print(f"RESULT ideal nc={n_coarse} : PAS de doublet", flush=True)
        return
    a, b = pair
    psi0, za = modes[0], modes[a]
    zb = modes[b] - np.sum(V * za * modes[b]) * za
    zb = nrm(zb)
    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    I = np.zeros((5, len(phis)))
    for j, ph in enumerate(phis):
        z = np.cos(ph) * za + np.sin(ph) * zb
        for m in range(5):
            I[m, j] = np.sum(V * psi0**(4 - m) * z**m)
    dl = np.linspace(0, 2 * np.pi / 3, 121)
    best = None
    for d0 in dl:
        Im = [np.mean([np.interp((d0 + 2 * np.pi * k / 3) % (2 * np.pi), phis, I[m],
                                 period=2 * np.pi) for k in range(3)]) for m in range(5)]

        def negE(th):
            ct, st = np.cos(th), np.sin(th)
            return -(ct**4 * Im[0] + 4 * ct**3 * st * Im[1] + 6 * ct**2 * st**2 * Im[2]
                     + 4 * ct * st**3 * Im[3] + st**4 * Im[4])

        r = minimize_scalar(negE, bounds=(0.05, np.pi / 2 - 0.05), method="bounded")
        if best is None or -r.fun > best[0]:
            best = (-r.fun, np.degrees(r.x))
    print(f"RESULT ideal nc={n_coarse} : doublet {a},{b} "
          f"(split {abs(evals[b]-evals[a])/abs(evals[a])*100:.1f}%, chars {chars[a]:+.2f}/{chars[b]:+.2f}) ; "
          f"theta* = {best[1]:.2f} deg ; tan2 = {np.tan(np.radians(best[1]))**2:.3f} "
          f"(BIT : ~2 = OUI, sinon NON)", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]))
