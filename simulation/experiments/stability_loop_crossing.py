r"""
Probabilité de stabilité par BOUCLE (enlacement Lk) et CROISEMENT — schéma V4.

Facteur de stabilité combinant les deux ingrédients discrets :
  - BOUCLES / enlacement : écart d'enlacement D = (Lk − Lk_opt)²,
    Lk_opt = C(n,2) = un Hopf par paire de boucles (CLAUDE.md).
    D = 0  <=> configuration au minimum d'enlacement => STABLE.
  - CROISEMENTS : c_tot = Σ c_i (complexité topologique, échelle de masse).

  Probabilité de stabilité :  P_stab = exp(-[β·D + α·c_tot/c_0])
  (D domine la stabilité ; c_tot module faiblement.)

On calcule pour les particules « élémentaires » + p + n (schéma V4), on compare
le verdict stable/instable de D à la réalité, et on regarde la durée de vie.

Usage :  python3 -m simulation.experiments.stability_loop_crossing
"""
from __future__ import annotations

import numpy as np

# nom, config (schéma V4), Lk (tableau V4 ; None=nœud simple -> 0), τ_exp(s), note
# τ_exp : None=stable ; "conf"=quark confiné (pas de libre) ; sinon valeur PDG
SET = [
    ("e-",   ["3_1"],                   0,    None,   "stable"),
    ("nu",   ["6_2", "9_1"],            1,    None,   "~stable"),   # ν : Lk=Lk_opt=1
    ("u",    ["3_1"],                   0,    None,   "stable (confiné)"),
    ("d",    ["3_1"],                   0,    "conf", "confiné"),
    ("mu-",  ["3_1", "6_1"],            0,    2.197e-6,  ""),
    ("s",    ["5_2", "5_2"],            0,    "conf", "confiné (~K,Λ)"),
    ("c",    ["7_3", "7_7"],            0,    "conf", "confiné (~D)"),
    ("tau-", ["3_1", "5_1", "7_1"],     0,    2.903e-13, ""),
    ("b",    ["4_1", "5_2", "7_6"],     0,    "conf", "confiné (~B)"),
    ("t",    ["7_4", "7_6", "8_3"],     0,    5.0e-25,   ""),
    ("W",    ["4_1", "7_3", "8_3"],     0,    3.157e-25, ""),
    ("Z0",   ["5_1", "7_2", "8_3"],     0,    2.638e-25, ""),
    ("H",    ["6_1", "7_4", "8_3"],     0,    1.6e-22,   ""),
    ("p",    ["3_1", "7_2", "7_2"],     3,    None,   "stable"),
    ("n",    ["3_1", "7_2", "7_2"],     2,    878.4,  "métastable"),
]

BETA = 1.0     # poids de l'écart d'enlacement
ALPHA = 0.05   # poids (faible) des croisements
C0 = 17.0      # croisements de référence (nucléon)


def crossings(name):
    return int(name.split("_")[0])


def main():
    print("=" * 96)
    print("  P_stab par BOUCLE (Lk) et CROISEMENT — élémentaires + p + n (schéma V4)")
    print("=" * 96)
    print(f"\n  D = (Lk − Lk_opt)², Lk_opt = C(n,2) ;  P_stab = exp(−[{BETA}·D + {ALPHA}·c_tot/{C0:.0f}])\n")
    print(f"  {'part.':>6} {'config':>16} {'c_tot':>6} {'n':>2} {'Lk':>4} "
          f"{'Lk_opt':>6} {'D':>3} {'P_stab':>7} {'verdict':>10} {'τ_exp(s)':>11}")
    print("  " + "-" * 92)
    rows = []
    for name, knots, lk, tau, note in SET:
        c_tot = sum(crossings(k) for k in knots)
        n = len(knots)
        lk_opt = n * (n - 1) // 2
        D = (lk - lk_opt) ** 2
        P = np.exp(-(BETA * D + ALPHA * c_tot / C0))
        verdict = "STABLE" if D == 0 else "instable"
        cfg = "+".join(knots)
        te = ("stable" if tau is None else ("confiné" if tau == "conf" else f"{tau:.2e}"))
        print(f"  {name:>6} {cfg:>16} {c_tot:>6} {n:>2} {lk:>4} {lk_opt:>6} "
              f"{D:>3} {P:>7.3f} {verdict:>10} {te:>11}")
        rows.append({"name": name, "c": c_tot, "n": n, "D": D, "P": P,
                     "tau": tau if isinstance(tau, (int, float)) else None,
                     "note": note})

    # verdict D vs réalité
    print("\n  VERDICT BOUCLE (D) vs réalité :")
    print("    D = 0 -> STABLE :", ", ".join(r["name"] for r in rows if r["D"] == 0),
          "  (réels stables : e, ν, u, p ✓ ; d confiné)")
    print("    D > 0 -> instable :", ", ".join(r["name"] for r in rows if r["D"] > 0))
    print("    -> le proton (Lk=Lk_opt=3, D=0) est STABLE ; le neutron (D=1) DÉCAÎT.")

    # la durée de vie : D fixe-t-il l'échelle ? (réponse : non, il faut Q)
    unst = [r for r in rows if r["tau"]]
    print("\n  Durée de vie des instables (D vs τ) :")
    for r in sorted(unst, key=lambda r: r["tau"], reverse=True):
        print(f"    {r['name']:>6} : D={r['D']}  c_tot={r['c']:>2}  τ={r['tau']:.2e}")
    lt = np.log10([r["tau"] for r in unst]); D = np.array([r["D"] for r in unst])
    c = np.array([r["c"] for r in unst])
    print(f"\n    corr(log τ, D)     = {np.corrcoef(D, lt)[0,1]:+.3f}")
    print(f"    corr(log τ, c_tot) = {np.corrcoef(c, lt)[0,1]:+.3f}")
    print("    -> à D fixé (ex. n & μ : D=1) τ varie de ~9 ordres : D donne la")
    print("       STABILITÉ (oui/non), pas l'échelle de durée de vie (= Q, énergie).")
    print("=" * 96)


if __name__ == "__main__":
    main()
