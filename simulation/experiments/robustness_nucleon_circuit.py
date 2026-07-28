"""Campagne de robustesse du resultat nucleon (+0.7%) — beta_otimes.

Usage : python3 robustness_nucleon_circuit.py <config>
Configs :
  seed2  : neutron C3, placement alternatif (2e clairance)
  seed3  : neutron, S2 = Rz(240) S1 (autre triangle C3)
  seed4  : neutron, compagnons s7=2.2 (propre scan de placement)
  proton : motif geometrique |Lk|=(1,1,1) (somme signee -1, cf. nota), depuis
           /tmp/proton_*.npy (fenetre du 2e compagnon autour du faisceau K+S1)
  long   : configuration originale, 40000 iterations (derive de convergence)
  hires  : configuration originale, resolutions K=300 / S=360

Chaque run : sono_link -> gardes (Lk Gauss + Alexander) -> L_gmd/E du coeur 3_1
a a=0.209 ET a=lambdabar_C(p)=0.2103. Ligne finale : 'RESULT <config> ...'.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ideal_embedding import (  # noqa: E402
    sono_link, torus_knot, twist_knot_seed, resample, linking_number, curve_length,
)
from derive_inductance_factor_test import L_gmd, energy_MeV, TARGET, FM, MU0  # noqa: E402
from repair_curves_braid import alexander  # noqa: E402


def build_neutron(s7=2.5, rank=0, c3_angle=2 * np.pi / 3, nK=240, nS=300):
    """Construction C3 du motif neutron ; rank = index (par clairance) du placement."""
    K = torus_knot(2, 3, n=nK)
    S = resample(twist_knot_seed(9, hand=+1, thread=+1, x_eye=0.10), nS)
    S = (S - S.mean(0)) * s7
    W0 = np.array([-1.1, 0.0, 0.0]) * s7
    T = np.roll(K, -1, 0) - np.roll(K, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    i0 = int(np.argmax(np.abs(T[:, 1])))
    found = []
    for dx in np.linspace(-1.2, 1.2, 7):
        for dz in np.linspace(-1.2, 1.2, 7):
            for dy in np.linspace(-1.5, 1.5, 5):
                Kp = K - K[i0] + W0 + np.array([dx, dy, dz])
                d = np.linalg.norm(Kp[:, None, :] - S[None, :, :], axis=2).min()
                if d < 0.35:
                    continue
                if abs(linking_number(Kp, S)) == 1:
                    found.append((d, dx, dy, dz))
    found.sort(reverse=True)
    d, dx, dy, dz = found[min(rank, len(found) - 1)]
    off = np.array([dx, dy, dz])
    S1 = S + (K[i0] - W0 - off)          # K reste a l'origine
    c, s = np.cos(c3_angle), np.sin(c3_angle)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    S2 = S1 @ Rz.T
    return [K, S1, S2]


def measure(config, comps, iters):
    comps, info = sono_link(comps, iters=iters, log_every=5000)
    lks = (linking_number(comps[0], comps[1]), linking_number(comps[0], comps[2]),
           linking_number(comps[1], comps[2]))
    alex = [tuple(int(x) for x in alexander(C)[0]) for C in comps]
    ok = alex[0] == (1, -1, 1) and alex[1] == (3, -5, 3) and alex[2] == (3, -5, 3)
    print(f"{config}: rho = {info['ropelength']:.2f} gap = {info['min_gap']:.3f} "
          f"Lk = {lks} alexander {'OK' if ok else 'KO ' + str(alex)}", flush=True)
    Es = []
    for a in (0.209, 0.2103):
        X = comps[0] * a * FM
        L, var = L_gmd(X, a * FM)
        Es.append(energy_MeV(L))
    L, var = L_gmd(comps[0] * 0.209 * FM, 0.209 * FM)
    print(f"RESULT {config} rho={info['ropelength']:.2f} gap={info['min_gap']:.3f} "
          f"lk={lks[0]}/{lks[1]}/{lks[2]} alex={'OK' if ok else 'KO'} "
          f"L={L/MU0/FM:.3f} var={var:.0f} E209={Es[0]:.3f} ({(Es[0]/TARGET-1)*100:+.1f}%) "
          f"E2103={Es[1]:.3f} ({(Es[1]/TARGET-1)*100:+.1f}%)", flush=True)


def run(config):
    if config == "seed2":
        measure(config, build_neutron(rank=1), 25000)
    elif config == "seed3":
        measure(config, build_neutron(c3_angle=4 * np.pi / 3), 25000)
    elif config == "seed4":
        measure(config, build_neutron(s7=2.2), 25000)
    elif config == "proton":
        comps = [np.load(f"/tmp/proton_{t}.npy") for t in ("K", "S1", "S2")]
        measure(config, comps, 25000)
    elif config == "long":
        measure(config, build_neutron(rank=0), 40000)
    elif config == "hires":
        measure(config, build_neutron(rank=0, nK=300, nS=360), 25000)
    else:
        raise SystemExit(f"config inconnue : {config}")


if __name__ == "__main__":
    run(sys.argv[1])
