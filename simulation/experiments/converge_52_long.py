"""Convergence longue du 5_2 ideal (suite du SS19) : SONO + recuits (21/07).

Reprend output/p52_ideal_partial.npy ; alterne relaxation (sono+polish)
et recuits gardes (invariant d'Alexander (2,-3,2)) ; journalise RL/Wr ;
s'arrete a RL <= 52,5, au plateau persistant, ou au plafond de temps.
Cible : Wr(5_2) ideal a +-0,02 pour trancher s = dWr/pi (2,6 sigma).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from simulation.ideal_embedding import sono, ropelength, polish_gradient, anneal
from simulation.experiments.repair_curves_braid import alexander
from simulation.experiments.calc_torsion_52_sono import writhe

STATE = "output/p52_ideal_partial.npy"
LOG = "output/p52_convergence_log.txt"
T_MAX = 80 * 60          # plafond 80 min
TARGET = 52.5
guard = lambda Q: tuple(alexander(Q)[0]) == (2, -3, 2)

P = np.load(STATE)
t0 = time.time()
hist = []


def report(tag, P):
    rl, wr = ropelength(P), writhe(P)
    hist.append((rl, wr))
    line = "%7.1fs  %-10s RL = %.3f  Wr = %.4f" % (time.time() - t0, tag, rl, wr)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    return rl


with open(LOG, "a") as f:
    f.write("=== reprise %s ===\n" % time.strftime("%H:%M:%S"))
rl = report("depart", P)
rounds = 0
stall = 0
while time.time() - t0 < T_MAX and rl > TARGET and rounds < 30:
    rounds += 1
    best_rl = rl
    if rounds % 3 == 0:
        Pb, rho, _ = anneal(P, guard=guard, target=TARGET,
                            rungs=(0.45, 0.6, 0.75), trials_per_rung=2,
                            iters=8000, sweeps=1,
                            rng=np.random.default_rng(100 + rounds))
        P = Pb
        tag = "recuit%d" % rounds
    else:
        P, _ = sono(P, iters=12000, shrink=0.9985)
        tag = "sono%d" % rounds
    P, _ = polish_gradient(P, iters=500)
    if not guard(P):
        print("GARDE VIOLEE apres %s -- retour a l'etat precedent" % tag,
              flush=True)
        P = np.load(STATE)
        continue
    np.save(STATE, P)
    rl = report(tag, P)
    stall = stall + 1 if best_rl - rl < 0.03 else 0
    if stall >= 6:
        print("plateau persistant (6 tours < 0,03) -- arret", flush=True)
        break

rl, wr = ropelength(P), writhe(P)
dwr = abs(wr) - 3.4516                      # |Wr(3_1)| de notre pipeline
print("\n=== BILAN (%.0f s, %d tours) ===" % (time.time() - t0, rounds))
print("RL = %.3f (cible ~52,3) ; Wr = %.4f (ideal litt. 4,556 ; 32/7 = 4,571)"
      % (rl, wr))
print("dWr/pi = %.4f  vs  s mesure 0,3577 +- 0,0023  (%+.1f %%, %.1f sigma)"
      % (dwr / np.pi, 100 * (0.3577 / (dwr / np.pi) - 1),
         abs(0.3577 - dwr / np.pi) / 0.0023))
print("(8/7)/pi = 0,3638 ; litt. 1,142/pi = 0,3635.")
