"""Convergence haute resolution du 5_2 (N=400) + etalonnage 3_1 (22/07).

Objectif : trancher la prediction Wr(5_2)_ideal = 4,5381 +- 0,0072
(taux 1/pi, SS20). Trois ameliorations vs le run N=200 :
  (1) N = 400 (biais de discretisation du writhe divise ~4) ;
  (2) ETALONNAGE : le 3_1 relaxe a N=400, son Wr calcule vs l'ideal
      connu 3,4144 -> correction multiplicative appliquee au 5_2 ;
  (3) run long (plafond ~100 min), etat repris de p52_ideal_partial.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from simulation.ideal_embedding import (sono, ropelength, resample,
                                        polish_gradient, anneal)
from simulation.experiments.repair_curves_braid import alexander
from simulation.experiments.calc_torsion_52_sono import writhe, torus_trefoil

STATE = "output/p52_ideal_hires.npy"
LOG = "output/p52_hires_log.txt"
T_MAX = 100 * 60
TARGET = 48.5   # cible relachee 22/07 : trouver le VRAI minimum (l'arret a 52,6 etait impose)
WR3_IDEAL = 3.4144
S_MES, S_SIG = 0.3577, 0.0023
guard = lambda Q: tuple(alexander(Q)[0]) == (2, -3, 2)
t0 = time.time()


def log(line):
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# --- (2) etalonnage : 3_1 a N=400 --------------------------------------
P3 = resample(torus_trefoil(400), 400)
P3, _ = sono(P3, iters=15000)
P3, _ = polish_gradient(P3, iters=600)
rl3, wr3 = ropelength(P3), abs(writhe(P3))
CAL = WR3_IDEAL / wr3
log("etalonnage 3_1 (N=400) : RL = %.3f ; |Wr| = %.4f ; ideal 3,4144"
    % (rl3, wr3))
log("  -> facteur d'etalonnage du writhe : %.5f (biais %+.2f %%)"
    % (CAL, 100 * (wr3 / WR3_IDEAL - 1)))

# --- (1)+(3) : 5_2 a N=400, reprise ------------------------------------
if os.path.exists(STATE):
    P = np.load(STATE)
else:
    P = resample(np.load("output/p52_ideal_partial.npy"), 400)
rl = ropelength(P)
log("depart 5_2 (N=400) : RL = %.3f ; Wr = %.4f" % (rl, writhe(P)))
rounds, stall = 0, 0
while time.time() - t0 < T_MAX and rl > TARGET and rounds < 40:
    rounds += 1
    best = rl
    if rounds % 4 == 0:
        try:
            P2, _, _ = anneal(P, guard=guard, target=TARGET,
                              rungs=(0.55, 0.7), trials_per_rung=2,
                              iters=6000, sweeps=1,
                              rng=np.random.default_rng(200 + rounds))
            P = P2
            tag = "recuit%d" % rounds
        except ValueError:
            tag = "recuit%d-echec" % rounds
    else:
        P, _ = sono(P, iters=8000, shrink=0.9988)
        tag = "sono%d" % rounds
    P, _ = polish_gradient(P, iters=400)
    if not guard(P):
        log("GARDE VIOLEE (%s) -- retour" % tag)
        P = np.load(STATE) if os.path.exists(STATE) else P
        continue
    np.save(STATE, P)
    rl = ropelength(P)
    wr = writhe(P)
    log("%7.1fs  %-12s RL = %.3f  Wr = %.4f  (etalonne %.4f)"
        % (time.time() - t0, tag, rl, wr, wr * CAL))
    stall = stall + 1 if best - rl < 0.02 else 0
    if stall >= 12:
        log("plateau persistant -- arret")
        break

wr = writhe(P)
wr_cal = wr * CAL
dwr = wr_cal - WR3_IDEAL
s_geo = dwr / np.pi
log("\n=== BILAN N=400 (%.0f s, %d tours) ===" % (time.time() - t0, rounds))
log("RL(5_2) = %.3f ; ratio l(5_2)/l(3_1) = %.4f (document principal 1,596)"
    % (ropelength(P), ropelength(P) / rl3))
log("Wr brut = %.4f ; Wr ETALONNE = %.4f  (prediction 1/pi : 4,5381 +- 0,0072)"
    % (wr, wr_cal))
log("dWr_etalonne/pi = %.4f  vs  s = %.4f +- %.4f  (%+.1f %%, %.1f sigma)"
    % (s_geo, S_MES, S_SIG, 100 * (S_MES / s_geo - 1) if s_geo else 0,
       abs(S_MES - s_geo) / S_SIG))
