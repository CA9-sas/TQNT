"""Resserrer la borne du 5_2 : tester le 3/2 exact (23/07).

CIBLE : l(5_2). ACPR (borne superieure, plongement reel) : 49,4704 ;
hypothese 3/2 exact : 1,5 x l(3_1) = 49,1172 (-0,71 %). Le 3_1 ACPR
est converge a <0,01 % -> le 3/2 exige que la borne 5_2 ait ~0,7 % de
mou. Notre etat SONO plafonne a 50,987 (+3,1 %) alors que le 3_1 au
meme protocole est a +0,3 % : plateau local, marge reelle.

PROTOCOLE (deux chaines, garde Alexander (2,-3,2), cap temps) :
  A) reprise de l'etat N=400 : recuits PROFONDS (squeezes 0,35-0,8),
     polish_gradient entre les vagues, re-echantillonnages 300<->400 ;
  B) redemarrage frais depuis la fermeture de tresse (autre bassin).
  + 3_1 au meme protocole final (le biais de methode s'annule dans le
    RATIO -- c'est le ratio qui teste le 3/2).
VERDICTS POSSIBLES : descendre sous 49,47 = mou PROUVE (le 3/2
respire) ; plateau au-dessus = notre borne ne discrimine pas (le test
reste chez ACPR/ridgerunner) ; ratio meme-protocole en appui.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from simulation.ideal_embedding import (sono, ropelength, resample,
                                        polish_gradient, anneal)
from simulation.experiments.repair_curves_braid import alexander, braid_curve

STATE = "output/p52_ideal_hires.npy"
OUT = "output/p52_resserre.npy"
LOG = "output/p52_resserre_log.txt"
T_MAX = 44 * 60
ACPR3, ACPR5 = 32.7448, 49.4704
TARGET = 1.5 * ACPR3
t0 = time.time()
guard = lambda Q: tuple(alexander(Q)[0]) == (2, -3, 2)


def log(line):
    print(f"{time.time() - t0:7.1f}s  {line}", flush=True)
    with open(LOG, "a") as f:
        f.write(f"{time.time() - t0:7.1f}s  {line}\n")


def polish_cycle(P, tag):
    P2 = polish_gradient(P, iters=800)
    if isinstance(P2, tuple):
        P2 = P2[0]
    if guard(P2) and ropelength(P2) < ropelength(P):
        P = P2
    log(f"{tag:14s} RL = {ropelength(P):.4f}")
    return P


def main():
    print(__doc__)
    rng = np.random.default_rng(11)
    best, best_rho = None, np.inf

    def keep(P, tag):
        nonlocal best, best_rho
        rho = ropelength(P)
        if guard(P) and rho < best_rho:
            best, best_rho = P.copy(), rho
            np.save(OUT, best)
            log(f"** MEILLEUR {tag} : RL = {rho:.4f} "
                f"(ACPR {100 * (rho / ACPR5 - 1):+.2f} %, "
                f"3/2 {100 * (rho / TARGET - 1):+.2f} %)")

    # --- chaine A : reprise de l'etat converge ---
    if os.path.exists(OUT):
        STATE_EFF = OUT                     # reprendre le meilleur du run long
    else:
        STATE_EFF = STATE
    P = np.load(STATE_EFF)
    log(f"depart A : RL = {ropelength(P):.4f} (N={len(P)})")
    keep(P, "A0")
    sweep = 0
    while time.time() - t0 < T_MAX * 0.05:      # run court : B + 3_1 surtout
        sweep += 1
        n_work = 300 if sweep % 2 else 400
        P = resample(best, n_work)
        Pb, rho, _ = anneal(P, guard=guard, target=TARGET,
                            rungs=(0.35, 0.45, 0.55, 0.65, 0.8),
                            trials_per_rung=2, iters=6000, sweeps=1,
                            rng=rng, log=lambda m: log("  " + m))
        Pb = polish_cycle(Pb, f"A.polish{sweep}")
        Pb = resample(Pb, 400)
        Pb, info = sono(Pb, iters=4000, shrink=0.9990)
        keep(Pb, f"A.sweep{sweep}")
        if best_rho <= TARGET:
            break

    # --- chaine B : bassin frais (fermeture de tresse) ---
    if best_rho > TARGET:
        P0, _, _ = braid_curve([1, 1, 1, 2, -1, 2], 3)  # mot du 5_2 (repo)
        log(f"depart B : RL brut = {ropelength(resample(P0, 360)):.1f}")
        Pb, info = sono(P0, n=360, iters=12000)
        keep(Pb, "B.sono")
        Pb, rho, _ = anneal(Pb, guard=guard, target=TARGET,
                            rungs=(0.4, 0.55, 0.7), trials_per_rung=2,
                            iters=6000, sweeps=1, rng=rng,
                            log=lambda m: log("  " + m))
        Pb = polish_cycle(Pb, "B.polish")
        keep(Pb, "B.final")

    # --- 3_1 au meme protocole (ratio) ---
    from simulation.experiments.calc_torsion_52_sono import torus_trefoil
    P3 = torus_trefoil(400)
    P3, _ = sono(P3, iters=10000)
    P3p = polish_gradient(P3, iters=800)
    if isinstance(P3p, tuple):
        P3p = P3p[0]
    if ropelength(P3p) < ropelength(P3):
        P3 = P3p
    rho3 = ropelength(P3)
    log(f"3_1 meme protocole : RL = {rho3:.4f} "
        f"(ACPR {100 * (rho3 / ACPR3 - 1):+.2f} %)")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    ratio = best_rho / rho3
    corr5 = best_rho / (rho3 / ACPR3)
    print(f"""  Meilleur 5_2 : RL = {best_rho:.4f}
     vs ACPR 49,4704 : {100 * (best_rho / ACPR5 - 1):+.2f} %
     vs 3/2 exact 49,1172 : {100 * (best_rho / TARGET - 1):+.2f} %
  3_1 meme protocole : {rho3:.4f} (biais {100 * (rho3 / ACPR3 - 1):+.2f} %)
  RATIO meme protocole : {ratio:.4f}  (3/2 = 1,5000 ; ACPR = 1,5108)
  5_2 corrige du biais 3_1 : {corr5:.3f}
  VERDICT : {'SOUS ACPR -- mou prouve, le 3/2 respire' if best_rho < ACPR5 else
             'au-dessus d ACPR -- notre borne ne discrimine pas encore'}""")


if __name__ == "__main__":
    main()
