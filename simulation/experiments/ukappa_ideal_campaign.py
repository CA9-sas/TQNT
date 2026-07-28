"""Campagne U_kappa intrinseque : re-mesure du catalogue sur plongements IDEAUX.

CONTEXTE : 14/19 courbes de theory/knot_catalog.py sont topologiquement FAUSSES
(realisent trefle/unknot) ; les U_kappa/U_tau qui alimentaient Omega_ts etaient donc
la geometrie du MAUVAIS noeud. L'outillage de plongement ideal (SONO, valide trefle
+0.8% / Hopf +0.07%) fournit desormais L'embedding principiel pour mesurer les
fonctionnelles geometriques.

MESURES (invariantes d'echelle, sur la courbe ideale R=1) :
  - rho        : ropelength (vs publie Cantarella-Kusner-Sullivan/Pieranski, en rayons)
  - TC = int kappa ds        : courbure totale [rad] (invariante d'echelle)
  - LU = L * int kappa^2 ds  : U_kappa adimensionnee (invariante d'echelle)
  - CS = (int kappa ds)^2    : proxy Cauchy-Schwarz (borne inf de LU ; validee <16%)
  - Wr : writhe (Gauss self-integral)
  + comparaison : memes mesures sur l'ANCIENNE courbe du catalogue (fausse pour 14/19).

AMORCES : torus_knot (3_1,5_1,7_1,9_1) ; twist_knot_seed (4_1,5_2,6_1,7_2 — echelle
(n_half,x_eye) validee par Alexander) ; braid_curve (6_2,8_1,8_2,8_3,9_2,10_1 — risque
de plateau type 7_2 : rho vs publie FLAGGE). Garde d'Alexander sur l'ideal final.

Usage : python3 ukappa_ideal_campaign.py <knot>   (RESULT sur une ligne)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono, torus_knot, twist_knot_seed, resample, curve_length,
)
from repair_curves_braid import braid_curve, alexander, BRAID_WORDS  # noqa: E402
from theory.knot_catalog import knot_curve  # noqa: E402

ROPE_PUB = {"3_1": 32.744, "4_1": 41.98, "5_1": 47.10, "5_2": 52.26, "6_1": 56.62,
            "6_2": 56.26, "7_1": 61.38, "7_2": 65.92, "8_1": 74.98, "8_2": 74.64,
            "8_3": 74.60, "9_1": 76.0, "9_2": 80.0, "10_1": 90.0}   # unites de RAYON
ALEX = {"3_1": (1, -1, 1), "4_1": (1, -3, 1), "5_1": (1, -1, 1, -1, 1),
        "5_2": (2, -3, 2), "6_1": (2, -5, 2), "6_2": (1, -3, 3, -3, 1),
        "7_1": (1, -1, 1, -1, 1, -1, 1), "7_2": (3, -5, 3)}
TWIST = {"4_1": (5, -0.35), "5_2": (5, 0.10), "6_1": (7, 0.10), "7_2": (9, 0.10),
         "8_1": (11, 0.10), "9_2": (13, 0.10), "10_1": (15, 0.10)}
TORUS = {"3_1": 3, "5_1": 5, "7_1": 7, "9_1": 9}


def seed_for(name):
    if name in TORUS:
        return torus_knot(2, TORUS[name], n=80 + 20 * TORUS[name])
    if name in TWIST:
        nh, xe = TWIST[name]
        return resample(twist_knot_seed(nh, hand=+1, thread=+1, x_eye=xe), 1200)
    word, ns, target = BRAID_WORDS[name]
    P, _, _ = braid_curve(word, ns, pts_per_seg=40)
    P = resample(P, 4096)
    F = [np.fft.rfft(P[:, c]) for c in range(3)]
    for f in F:
        f[49:] = 0
    S = np.stack([np.fft.irfft(f, len(P)) for f in F], 1)
    # garde d'AMORCE : le lissage peut denouer (lecon 9_2) -> repli sans lissage
    try:
        ok = tuple(int(x) for x in alexander(S)[0]) == tuple(target)
    except Exception:
        ok = False
    if not ok:
        print(f"[seed {name} : lissage denoue -> repli courbe brute]", flush=True)
        S = P
    return S


def measures(P):
    n = len(P)
    prev, nxt = np.roll(P, 1, 0), np.roll(P, -1, 0)
    e1, e2 = P - prev, nxt - P
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)
    cosang = np.clip((e1 * e2).sum(1) / (l1 * l2), -1, 1)
    theta = np.arccos(cosang)
    ds = 0.5 * (l1 + l2)
    L = curve_length(P)
    TC = float(theta.sum())                       # int kappa ds
    LU = float(L * (theta**2 / ds).sum())         # L * int kappa^2 ds
    # writhe (Gauss)
    dl = nxt - P
    mid = P + 0.5 * dl
    wr = 0.0
    for i in range(n):
        r = mid - mid[i]
        d3 = np.linalg.norm(r, axis=1)**3
        d3[i] = np.inf
        wr += float(((np.cross(dl[i], dl) * r).sum(1) / d3).sum())
    wr /= 4 * np.pi
    return L, TC, LU, wr


def run(name):
    npts = 200 + 10 * int(name.split("_")[0])
    P, info = sono(seed_for(name), n=min(npts, 320), iters=30000)
    try:
        al = tuple(int(x) for x in alexander(P)[0])
        guard = "OK" if (name not in ALEX or al == ALEX[name]) else f"KO{al}"
    except Exception:
        guard = "?"
    L, TC, LU, wr = measures(P)
    pub = ROPE_PUB.get(name)
    drho = f"{(info['ropelength']/pub-1)*100:+.0f}%" if pub else "?"
    # ancienne courbe du catalogue (fausse pour 14/19)
    try:
        Q = resample(np.asarray(knot_curve(name, n_points=1200)), 600)
        _, TCo, LUo, wro = measures(Q)
        old = f"TC={TCo:.1f} LU={LUo:.0f} Wr={wro:+.1f}"
    except Exception as e:
        old = f"indisponible ({type(e).__name__})"
    print(f"RESULT {name} rho={info['ropelength']:.2f} ({drho} vs pub) gap={info['min_gap']:.3f} "
          f"alex={guard} TC={TC:.2f} LU={LU:.0f} CS={TC**2:.0f} Wr={wr:+.2f} | ancien: {old}",
          flush=True)
    np.save(f"/tmp/ideal_{name}.npy", P)


if __name__ == "__main__":
    run(sys.argv[1])
