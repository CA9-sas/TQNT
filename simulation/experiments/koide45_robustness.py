"""Robustesse du mecanisme 45deg/arctan(sqrt2) : maillages, epsilon, controles negatifs.

Usage : python3 koide45_robustness.py <knot> <n_coarse> <eps>
Pipeline : modes FEM -> doublet C_q (caractere de rotation 2pi/q, q=3 pour 3_1, 5 pour 5_1)
-> integrales quartiques I_m(phi) -> moyenne d'orbite Z_q -> extremum (theta*, delta*)
avec resolution fine de theta* (compare a 45.0000 et arctan(sqrt2)=54.7356).
RESULT sur une ligne.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem  # noqa: E402
from theory.knot_catalog import knot_curve  # noqa: E402

Q_SYM = {"3_1": 3, "5_1": 5, "7_1": 7}


def lumped_volumes(points, tets):
    tets = np.asarray(tets)
    p = points[tets]                       # (n, 4, 3)
    e = p[:, 1:] - p[:, :1]                # (n, 3, 3)
    vols = np.abs(np.einsum('ni,ni->n', e[:, 0], np.cross(e[:, 1], e[:, 2]))) / 24.0
    V = np.zeros(len(points))
    np.add.at(V, tets.ravel(), np.repeat(vols, 4))
    return V


def run(knot, n_coarse, eps):
    q = Q_SYM[knot]
    curve = knot_curve(knot, 500)
    tree = cKDTree(curve)
    pts, tets, boundary = generate_mesh(curve, n_coarse=n_coarse, epsilon=eps, refine=True)
    H, M = assemble_system(pts, tets, boundary, knot, tree, energy=0.0, epsilon=eps)
    evals, ev_free = solve_eigenproblem(H, M, n_eigenvalues=6)
    free = ~boundary
    evecs = np.zeros((len(pts), ev_free.shape[1]))
    evecs[free, :] = ev_free
    V = lumped_volumes(pts, tets)

    def nrm(v):
        return v / np.sqrt(np.sum(V * v * v))

    modes = [nrm(evecs[:, k]) for k in range(evecs.shape[1])]

    c, s = np.cos(2 * np.pi / q), np.sin(2 * np.pi / q)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ptree = cKDTree(pts)
    _, idx = ptree.query(pts @ R.T)
    chars = [float(np.sum(V * v[idx] * v)) for v in modes]
    target_char = np.cos(2 * np.pi / q)
    pair = None
    for k in range(1, len(modes) - 1):
        if (abs(evals[k + 1] - evals[k]) / abs(evals[k]) < 0.05
                and abs(chars[k] - target_char) < 0.35
                and abs(chars[k + 1] - target_char) < 0.35):
            pair = (k, k + 1)
            break
    if pair is None:
        print(f"RESULT {knot} nc={n_coarse} eps={eps} : PAS de doublet C_{q} "
              f"(chars {['%.2f' % c_ for c_ in chars]})", flush=True)
        return
    a, b = pair
    psi0 = modes[0]
    za = modes[a]
    zb = modes[b] - np.sum(V * za * modes[b]) * za
    zb = nrm(zb)

    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    I = np.zeros((5, len(phis)))
    for j, ph in enumerate(phis):
        z = np.cos(ph) * za + np.sin(ph) * zb
        for m in range(5):
            I[m, j] = np.sum(V * psi0**(4 - m) * z**m)

    # moyenne d'orbite Z_q + extremum (theta, delta)
    dl = np.linspace(0, 2 * np.pi / q, 121)
    Imean = np.zeros((5, len(dl)))
    for m in range(5):
        for j, d0 in enumerate(dl):
            vals = [np.interp((d0 + 2 * np.pi * k / q) % (2 * np.pi), phis, I[m],
                              period=2 * np.pi) for k in range(q)]
            Imean[m, j] = np.mean(vals)

    best = None
    for j in range(len(dl)):
        Im = Imean[:, j]

        def negE(th):
            ct, st = np.cos(th), np.sin(th)
            return -(ct**4 * Im[0] + 4 * ct**3 * st * Im[1] + 6 * ct**2 * st**2 * Im[2]
                     + 4 * ct * st**3 * Im[3] + st**4 * Im[4])

        r = minimize_scalar(negE, bounds=(0.05, np.pi / 2 - 0.05), method="bounded")
        if best is None or -r.fun > best[0]:
            best = (-r.fun, np.degrees(r.x), dl[j])
    _, th_star, dl_star = best
    print(f"RESULT {knot} nc={n_coarse} eps={eps} : doublet modes {a},{b} "
          f"(split {abs(evals[b]-evals[a])/abs(evals[a])*100:.1f}%, chars {chars[a]:+.2f}/{chars[b]:+.2f}) ; "
          f"theta* = {th_star:.3f} deg (45.000 ? arctan(sqrt2)=54.736 ?) ; "
          f"delta* = {np.degrees(dl_star):.1f} deg ; tan2(theta*) = {np.tan(np.radians(th_star))**2:.3f}",
          flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
