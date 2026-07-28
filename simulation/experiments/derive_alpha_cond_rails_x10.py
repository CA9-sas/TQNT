"""alpha_cond depuis les rails -- statistique x10, estimateur corrige (23/07).

Reprise du S46 avec :
 - STATISTIQUE x10 : ~78 000 fermetures (parallelise, 12 coeurs) ;
 - ESTIMATEUR SANS BIAIS DE SATURATION : dans l'image de Poisson des
   evenements de nouage (le nouage premier = evenements independants
   de taux 1/N0(K) par pas), on a EXACTEMENT
      P(K | N) / P(non-noue | N) = (N - N_off) / N0(K)
   -- le ratio au non-noue MESURE annule la saturation (le facteur
   e^{-N/N0_tot} commun se divise) : la pente du ratio donne N0(K)
   sans biais, y compris pour 3_1.
 - alpha_rails = d ln N0/dc (regression 7 types, bootstrap).
Cibles : alpha_cond utilise 1,1991 ; fenetre cinetique/loi
geometrique [1,214, 1,220].
"""
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.experiments.approfondit_rails_e_p_n import (  # noqa: E402
    closed_lattice_walk, identify)

TYPES = {"3_1": 3, "4_1": 4, "5_1": 5, "5_2": 5, "6_1": 6, "6_2": 6,
         "6_3": 6}
PLAN = [(40, 24000), (56, 22000), (72, 18000), (96, 14000)]
N_OFF = 22.0


def chunk(args):
    n, trials, seed = args
    rng = np.random.default_rng(seed)
    cnt = Counter()
    for _ in range(trials):
        w = closed_lattice_walk(n, rng)
        name, _ = identify(w, rng)
        cnt[name] += 1
    return n, cnt


def main():
    print(__doc__)
    jobs = []
    seed = 1000
    for n, trials in PLAN:
        per = trials // 12 + 1
        for k in range(12):
            jobs.append((n, per, seed))
            seed += 1
    with Pool(12) as pool:
        results = pool.map(chunk, jobs)

    counts = {n: Counter() for n, _ in PLAN}
    totals = Counter()
    for n, cnt in results:
        counts[n].update(cnt)
        totals[n] += sum(cnt.values())

    print("=" * 78)
    print("[A] COMPTAGES (x10, estimateur ratio-au-non-noue)")
    print("=" * 78)
    for n, _ in PLAN:
        c = counts[n]
        row = "  ".join(f"{k} {c[k]:4d}" for k in TYPES)
        print(f"  N = {n:3d} ({totals[n]:6d}, non-noue {c['0_1']:6d}) : {row}")

    print("=" * 78)
    print("[B] N0(K) PAR LE RATIO (sans biais de saturation)")
    print("=" * 78)
    ns = np.array([n for n, _ in PLAN], float)
    n0s, cs, errs, names = [], [], [], []
    for k, c in TYPES.items():
        ratios, werr = [], []
        for n, _ in PLAN:
            nk = counts[n][k]
            nu = counts[n]["0_1"]
            ratios.append(nk / nu)
            werr.append(max(nk, 1))
        ratios = np.array(ratios)
        tot = sum(counts[n][k] for n, _ in PLAN)
        # fit ratio = (N - N_OFF)/N0 -> pente 1/N0 (poids Poisson)
        x = ns - N_OFF
        wgt = np.array(werr, float) / np.maximum(ratios, 1e-12)**2
        slope = float(np.sum(wgt * x * ratios) / np.sum(wgt * x * x))
        n0 = 1.0 / max(slope, 1e-12)
        err = n0 / max(np.sqrt(tot), 1.0)
        n0s.append(n0)
        cs.append(c)
        errs.append(err)
        names.append(k)
        print(f"  {k} (c={c}) : {tot:5d} evts, N0 = {n0:8.0f} (+- {err:.0f})")

    print("=" * 78)
    print("[C] alpha_rails PINCE")
    print("=" * 78)
    cs_a, n0_a, er_a = map(np.array, (cs, n0s, errs))
    sig = er_a / n0_a
    A = np.vstack([cs_a, np.ones_like(cs_a, float)]).T
    W = np.diag(1.0 / sig**2)
    coef, *_ = np.linalg.lstsq(W @ A, W @ np.log(n0_a), rcond=None)
    alpha = float(coef[0])
    rngb = np.random.default_rng(7)
    boots = []
    for _ in range(4000):
        ii = rngb.integers(0, len(cs_a), len(cs_a))
        if len(set(cs_a[ii])) < 2:
            continue
        y = np.log(n0_a[ii]) + rngb.normal(0, sig[ii])
        Ab = np.vstack([cs_a[ii], np.ones(len(ii))]).T
        cb, *_ = np.linalg.lstsq(Ab, y, rcond=None)
        boots.append(cb[0])
    d_alpha = float(np.std(boots))
    print(f"""  Regression ln N0 vs c (7 types, bootstrap types+Poisson) :
     alpha_rails = {alpha:.3f} +- {d_alpha:.3f}
  vs alpha_cond utilise 1,1991 : {abs(alpha - 1.1991) / d_alpha:.1f} sigma
  vs fenetre cinetique [1,214, 1,220] : {abs(alpha - 1.217) / d_alpha:.1f} sigma""")
    pairs = []
    for i in range(len(cs_a)):
        for j in range(len(cs_a)):
            if cs_a[j] == cs_a[i] + 1:
                pairs.append((f"{names[i]}->{names[j]}",
                              float(np.log(n0_a[j] / n0_a[i]))))
    print("  Pas individuels : " +
          ", ".join(f"{t} {v:+.2f}" for t, v in pairs))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    within = abs(alpha - 1.1991) <= 2 * d_alpha or \
        abs(alpha - 1.217) <= 2 * d_alpha
    print(f"""  alpha_rails = {alpha:.3f} +- {d_alpha:.3f}
  ({'COMPATIBLE' if within else 'EN TENSION'} avec la cible ; dispersion
  des pas individuels = entropie par type, incluse dans le bootstrap).
  Statut : voir verdict -- si l'erreur est descendue sous ~0,05 et que
  le central tient la fenetre, alpha_cond est PINCE par la
  combinatoire du socle ; sinon, soutenu a la precision atteinte.""")


if __name__ == "__main__":
    main()
