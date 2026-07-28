"""alpha_cond depuis la combinatoire des rails (ponderation en action, 23/07).

CIBLE : alpha_cond -- le poids de formation e^{-alpha c} de la chaine
Omega (utilise 1,1991 ; la derivation cinetique etablie donne
1,214-1,220 sous la loi de masse geometrique : fenetre cible
[1,20, 1,22] a ~1 %). Identite d'arbre etablie : passages/scissions
= 2e^{-alpha} = 0,59-0,60.

LE MECANISME (force par l'etabli, zero hypothese nouvelle) :
[1] Au gel, le spectre des fragments est EXPONENTIEL en longueur
    (la cascade etablie : fragments ~ e^{-N/lambda}, lambda en pas de
    rail >> minima des types) -- C'EST la ponderation en action :
    les chemins longs coutent leur longueur.
[2] L'abondance d'un type K est l'integrale du regime d'amorce :
       A(K) = int e^{-N/lambda} P(K|N) dN,   P(K|N) ~ (N-Nmin)/N0(K)
            ~ lambda^2 e^{-Nmin/lambda} / N0(K)  ->  1/N0(K)
    (e^{-Nmin/lambda} -> 1 car lambda >> Nmin : la dependance en la
    longueur MINIMALE disparait, seule reste la LONGUEUR DE NOUAGE
    N0(K) -- la grandeur combinatoire pure des rails).
[3] DONC le poids par croisement e^{-alpha c} EMERGE avec
       alpha_rails = d ln N0(K) / dc
    -- la croissance des longueurs de nouage avec le nombre de
    croisements. A MESURER par Monte Carlo (P(K|N) au regime
    d'amorce, 4 longueurs, 7 types identifies exactement).

CONDITIONS DECLAREES : regime d'amorce (P petit -- vrai au gel :
~0,3 noeud/fragment) ; mesure d'equiprobabilite a N fixe (la mesure
d'onde) ; identification det(-1) + |Delta(2)| impair (Bareiss exact).
"""
import sys, os
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.experiments.approfondit_rails_e_p_n import (  # noqa: E402
    closed_lattice_walk, identify)

TYPES = {"3_1": 3, "4_1": 4, "5_1": 5, "5_2": 5, "6_1": 6, "6_2": 6,
         "6_3": 6}
TARGET_LO, TARGET_HI, TARGET_USED = 1.214, 1.220, 1.1991


def main():
    print(__doc__)
    rng = np.random.default_rng(42)
    plan = [(40, 2400), (56, 2200), (72, 1800), (96, 1400)]

    print("=" * 78)
    print("[A] P(K | N) AU REGIME D'AMORCE (MC, identification exacte)")
    print("=" * 78)
    counts = {}
    for n, trials in plan:
        cnt = Counter()
        for _ in range(trials):
            w = closed_lattice_walk(n, rng)
            name, _ = identify(w, rng)
            cnt[name] += 1
        counts[n] = (cnt, trials)
        row = "  ".join(f"{k} {cnt[k]:3d}" for k in TYPES)
        print(f"  N = {n:3d} ({trials}) : {row}")

    print("=" * 78)
    print("[B] LONGUEURS DE NOUAGE N0(K) (pente de l'amorce)")
    print("=" * 78)
    ns = np.array([n for n, _ in plan], float)
    n0s, cs, errs = [], [], []
    for k, c in TYPES.items():
        ps = np.array([counts[n][0][k] / counts[n][1] for n, _ in plan])
        tot = sum(counts[n][0][k] for n, _ in plan)
        if tot < 4:
            print(f"  {k} : {tot} evts -- trop mince, ecarte")
            continue
        # fit lineaire P = (N - b)/N0  (moindres carres, poids Poisson)
        w = np.array([counts[n][1] for n, _ in plan], float)
        A = np.vstack([ns, np.ones_like(ns)]).T
        W = np.diag(w)
        coef, *_ = np.linalg.lstsq(W @ A, W @ ps, rcond=None)
        slope = max(coef[0], 1e-9)
        n0 = 1.0 / slope
        err = n0 / max(np.sqrt(tot), 1.0)
        n0s.append(n0)
        cs.append(c)
        errs.append(err)
        print(f"  {k} (c={c}) : {tot:4d} evts, N0 = {n0:7.0f} "
              f"(+- {err:.0f})")

    print("=" * 78)
    print("[C] alpha_rails = d ln N0 / dc")
    print("=" * 78)
    cs, n0s, errs = map(np.array, (cs, n0s, errs))
    lw = 1.0 / (errs / n0s)**2
    A = np.vstack([cs, np.ones_like(cs, dtype=float)]).T
    W = np.diag(lw)
    coef, *_ = np.linalg.lstsq(W @ A, W @ np.log(n0s), rcond=None)
    alpha = float(coef[0])
    # erreur par bootstrap des types
    boots = []
    rngb = np.random.default_rng(5)
    for _ in range(2000):
        ii = rngb.integers(0, len(cs), len(cs))
        if len(set(cs[ii])) < 2:
            continue
        Ab = np.vstack([cs[ii], np.ones(len(ii))]).T
        cb, *_ = np.linalg.lstsq(Ab, np.log(n0s[ii])
                                 + rngb.normal(0, errs[ii] / n0s[ii]),
                                 rcond=None)
        boots.append(cb[0])
    d_alpha = float(np.std(boots))
    print(f"""  Regression ln N0 vs c ({len(cs)} types) :
     alpha_rails = {alpha:.3f} +- {d_alpha:.3f}
  Cibles : alpha_cond utilise = {TARGET_USED} ; derivation cinetique
  sous loi geometrique = [{TARGET_LO}, {TARGET_HI}].
  Ecart au centre de la fenetre : {alpha - 0.5 * (TARGET_LO + TARGET_HI):+.3f}
  ({abs(alpha - 0.5 * (TARGET_LO + TARGET_HI)) / d_alpha:.1f} sigma).""")
    pairs = []
    for i in range(len(cs)):
        for j in range(len(cs)):
            if cs[j] == cs[i] + 1:
                pairs.append((f"c{cs[i]}->c{cs[j]}",
                              float(np.log(n0s[j] / n0s[i]))))
    if pairs:
        print("  Pas individuels : " +
              ", ".join(f"{t} {v:+.2f}" for t, v in pairs))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    verdict = ("DANS la fenetre cible" if TARGET_LO - d_alpha <= alpha
               <= TARGET_HI + d_alpha else
               "hors fenetre" if abs(alpha - 1.217) > 2 * d_alpha else
               "compatible a ~1 sigma")
    print(f"""  alpha_rails = {alpha:.3f} +- {d_alpha:.3f} -- {verdict}.
  LE MECANISME : la ponderation en action (spectre exponentiel des
  fragments, la cascade etablie) transforme les probabilites d'amorce
  en abondances ~ 1/N0(K) ; le poids e^{{-alpha c}} EMERGE avec
  alpha = d ln N0/dc, une grandeur PUREMENT combinatoire des rails
  (aucun parametre). Statut : derivation conditionnelle (amorce ;
  equiprobabilite a N fixe ; lambda >> minima). Si l'accord se
  confirme a plus forte statistique, alpha_cond cesse d'etre un
  produit de la seule cinetique : il est ANCRE dans la combinatoire
  du socle -- la jonction rails <-> chaine Omega serait complete.""")


if __name__ == "__main__":
    main()
