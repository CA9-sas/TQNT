"""Nucleon repoli (polish_gradient_link) -- vers le pincement de beta_otimes.

ETAT : le circuit d'enlacement mesure sur le nucleon relaxe (sono_link seul)
donne E = 1.302 MeV (+0.7% -- artefact de plateau) et, en robustesse, 8 minima
quasi degeneres rho in [204.5, 208] -> Delta m(n-p) a +3+-3%. Le bruit vient
du minimiseur (plateaux SONO).

ICI : meme pipeline (construction C3 du motif neutron, memes gardes) + le
POLISSAGE DE GRADIENT du lien apres sono_link, puis re-mesure de l'inductance
du coeur 3_1 -> E -> beta_otimes = E / 6.15.

Si le polissage resserre la dispersion des minima, la precision de la
reproduction sans bouton s'ameliore d'autant (cible : mieux que +-3%).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono_link, polish_gradient_link, torus_knot, twist_knot_seed, resample,
    linking_number, link_ropelength, curve_length, squeeze,
)
from repair_curves_braid import alexander  # noqa: E402
from derive_inductance_factor_test import L_gmd, energy_MeV, D_TUBE  # noqa: E402

MU0 = 4e-7 * np.pi
FM = 1e-15
TARGET = 1.293
GEOM_MEAN_CJ = 6.15


def build_neutron():
    K = torus_knot(2, 3, n=240)
    s7 = 2.5
    S = resample(twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10), 300)
    S = (S - S.mean(0)) * s7
    W0 = np.array([-1.1, 0.0, 0.0]) * s7
    T = np.roll(K, -1, 0) - np.roll(K, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    i0 = int(np.argmax(np.abs(T[:, 1])))
    off = np.array([0.40, -0.75, 1.20])
    S1 = S + (K[i0] - W0 - off)
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    S2 = S1 @ Rz.T
    return [K, S1, S2]


def guards_ok(comps):
    lks = (linking_number(comps[0], comps[1]), linking_number(comps[0], comps[2]),
           linking_number(comps[1], comps[2]))
    try:
        als = [tuple(int(x) for x in alexander(C)[0]) for C in comps]
    except Exception:
        return False, lks, None
    ok = (abs(lks[0]) == 1 and abs(lks[1]) == 1 and lks[2] == 0 and
          als[0] == (1, -1, 1) and als[1] == (3, -5, 3) and als[2] == (3, -5, 3))
    return ok, lks, als


def measure_core(comps, tag):
    C = comps[0]
    X = C * D_TUBE * FM
    L, var = L_gmd(X, D_TUBE * FM)
    E = energy_MeV(L)
    beta = E / GEOM_MEAN_CJ
    print("   %s : rho_tot = %.2f ; L(coeur) = %.3f mu0.fm (var %.0f%%) ; "
          "E = %.3f MeV (%+.1f%%) ; beta_otimes = %.4f (vs 0.210)" % (
              tag, link_ropelength(comps), L / MU0 / FM, var, E,
              (E / TARGET - 1) * 100, beta), flush=True)
    return E


def run():
    print("=== Nucleon repoli : sono_link + polish_gradient_link ===", flush=True)
    rng = np.random.default_rng(7)
    results = []
    for trial in range(4):
        comps0 = build_neutron()
        if trial:  # depart perturbe (squeeze homeomorphe global)
            u = rng.normal(size=3)
            sfac = 0.72 + 0.1 * rng.random()
            comps0 = [squeeze(C, u, sfac) for C in comps0]
        lk0 = (linking_number(comps0[0], comps0[1]),
               linking_number(comps0[0], comps0[2]))
        if abs(lk0[0]) != 1 or abs(lk0[1]) != 1:
            print(" essai %d : placement Lk invalide apres squeeze, saute" % trial,
                  flush=True)
            continue
        comps, info = sono_link(comps0, iters=25000,
                                loosen_start=(trial == 0))
        ok, lks, als = guards_ok(comps)
        print(" essai %d apres sono_link : rho = %.2f, gap = %.3f, Lk = %s, "
              "gardes %s" % (trial, info["ropelength"], info["min_gap"],
                             lks, "OK" if ok else "KO"), flush=True)
        if not ok:
            continue
        E_before = measure_core(comps, "avant polissage")
        comps, rho = polish_gradient_link(comps, iters=800)
        ok, lks, als = guards_ok(comps)
        if not ok:
            print("   polissage : gardes KO (rejet)", flush=True)
            continue
        E_after = measure_core(comps, "APRES polissage ")
        results.append((rho, E_after))
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        for i, C in enumerate(comps):
            np.save(os.path.join(out, f"nucleon_polished_t{trial}_{i}.npy"), C)
    if results:
        Es = [e for _, e in results]
        rhos = [r for r, _ in results]
        print("\nBILAN (%d essais valides) :" % len(results), flush=True)
        print("  rho : [%.2f, %.2f] (dispersion %.1f%%)" % (
            min(rhos), max(rhos), (max(rhos) / min(rhos) - 1) * 100), flush=True)
        print("  E   : [%.3f, %.3f] MeV -> ecart cible [%+.1f%%, %+.1f%%]" % (
            min(Es), max(Es), (min(Es) / TARGET - 1) * 100,
            (max(Es) / TARGET - 1) * 100), flush=True)
        print("  (reference avant chantier : 8 minima rho in [204.5, 208], "
              "+3+-3%)", flush=True)


if __name__ == "__main__":
    run()
