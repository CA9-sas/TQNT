r"""
Couplage entre GROUPE n (ondes entrelacées), Q (énergie libérée) et SPIN.

Trois pistes physiques :
  1. spin ↔ n : le spin vient du type torique des composantes T(2,q)→½,
     T(3,q)→1, T(4,q)→3/2... et de leur combinaison dans l'entrelacs de n
     ondes. Y a-t-il une contrainte spin_max(n) ?
  2. spin ↔ canal : le spin sélectionne-t-il le mode de désintégration
     (spin 3/2 baryons -> résonance forte rapide ; spin ½ -> faible) ?
  3. spin ↔ Q : moment angulaire emporté L -> barrière centrifuge
     Γ ∝ Q^(2L+1). Le spin du parent/changement de spin modifie-t-il
     l'exposant d'énergie ?

On tabule (n, Q, 2J, canal, τ) et on mesure les corrélations/structure.

Usage :  python3 -m simulation.experiments.coupling_n_Q_spin
"""
from __future__ import annotations

import numpy as np

# nom, n, 2J (spin×2), Q(MeV)|None, τ(s)|None|"conf", canal
P = [
    ("e-",     1, 1, None,  None,   "stable"),
    ("u",      1, 1, None,  None,   "stable"),
    ("pi+-",   1, 0, 33.9,  2.603e-8,  "faible-2c"),
    ("pi0",    1, 0, 135.,  8.43e-17,  "EM"),
    ("mu-",    2, 1, 105.7, 2.197e-6,  "faible-lept"),
    ("Lambda", 2, 1, 37.8,  2.617e-10, "faible-nonlept"),
    ("Sigma+", 2, 1, 113.,  8.018e-11, "faible-nonlept"),
    ("Sigma-", 2, 1, 118.,  1.479e-10, "faible-nonlept"),
    ("D0",     2, 0, 1270., 4.10e-13,  "faible-charm"),
    ("D+",     2, 0, 1270., 1.033e-12, "faible-charm"),
    ("Lambdac",2, 1, 1270., 2.03e-13,  "faible-charm"),
    ("B",      2, 0, 4180., 1.519e-12, "faible-bottom"),
    ("rho_om", 2, 2, 440.,  7.75e-23,  "fort"),
    ("tau-",   3, 1, 1777., 2.903e-13, "faible-lept"),
    ("p",      3, 1, None,  None,   "stable"),
    ("n",      3, 1, 0.78,  878.4,  "faible-beta"),
    ("K+-",    3, 0, 388.,  1.238e-8,  "faible-2c"),
    ("K0_S",   3, 0, 218.,  8.954e-11, "faible-nonlept"),
    ("eta",    3, 0, 548.,  5.02e-19,  "EM"),
    ("Xi0",    3, 1, 64.2,  2.90e-10,  "faible-nonlept"),
    ("Xi-",    3, 1, 66.5,  1.639e-10, "faible-nonlept"),
    ("Omega-", 3, 3, 63.0,  8.21e-11,  "faible-nonlept"),
    ("Delta",  3, 3, 156.,  5.63e-24,  "fort"),
    ("Sigma*", 3, 3, 130.,  1.83e-23,  "fort"),
    ("Xi*",    3, 3, 79.,   7.23e-23,  "fort"),
    ("phi",    3, 2, 32.,   1.55e-22,  "fort"),
    ("Jpsi",   3, 2, 3097., 7.11e-21,  "fort"),
    ("Omegac", 3, 1, 1270., 2.68e-13,  "faible-charm"),
    ("Lambdab",3, 1, 4180., 1.471e-12, "faible-bottom"),
    ("W",      3, 2, None,  3.157e-25, "EW"),
    ("Z0",     3, 2, None,  2.638e-25, "EW"),
    ("H",      3, 0, None,  1.6e-22,   "EW"),
    ("t",      3, 1, None,  5.0e-25,   "EW"),
]


def main():
    print("=" * 86)
    print("  COUPLAGE  groupe n  ×  Q  ×  spin (2J)")
    print("=" * 86)

    # (1) spin vs n : spin_max par groupe
    print("\n  (1) SPIN × GROUPE n  —  spin_max(n) ?")
    for n in (1, 2, 3):
        js = sorted(set(p[2] for p in P if p[1] == n))
        jmax = max(js)
        print(f"      n={n} : 2J présents = {js}  ->  spin_max = {jmax}/2")
    print("      -> 2J_max(n) : n=1->1, n=2->2, n=3->3 ?  (spin_max = n/2 ?)")

    # (2) spin vs canal
    print("\n  (2) SPIN × CANAL de désintégration")
    for j in (0, 1, 2, 3):
        chans = sorted(set(p[5] for p in P if p[2] == j and p[5] not in ("stable",)))
        names = [p[0] for p in P if p[2] == j]
        print(f"      2J={j} ({j/2:g}) : {', '.join(names)}")
        print(f"               canaux : {', '.join(chans)}")

    # (3) spin vs Q et effet sur la durée de vie
    print("\n  (3) SPIN × Q × τ  —  le spin module-t-il l'exposant d'énergie ?")
    # comparer les FORTES (résonances) : Γ ∝ Q^(2L+1) ; L lié au spin
    forte = [p for p in P if p[5] == "fort"]
    print("      Désintégrations FORTES (résonances) — Γ ∝ Q^(2L+1) attendu :")
    for nm, n, j2, Q, tau, _ in sorted(forte, key=lambda z: z[3]):
        print(f"        {nm:>8} 2J={j2} Q={Q:>7.1f}  τ={tau:.2e}")
    lQ = np.log10([p[3] for p in forte]); lt = np.log10([p[4] for p in forte])
    j2 = np.array([p[2] for p in forte])
    A = np.column_stack([np.ones(len(forte)), -lQ])
    c, *_ = np.linalg.lstsq(A, lt, rcond=None)
    print(f"        fit τ∝Q^ν (fortes) : ν={c[1]:.2f}  (multipôle 2L+1 ; L=1 p-wave -> 3)")
    print(f"        corr(2J, log τ) chez les fortes = {np.corrcoef(j2, lt)[0,1]:+.2f}")

    # spin global vs lifetime (toutes instables)
    inst = [p for p in P if isinstance(p[4], float)]
    j2a = np.array([p[2] for p in inst]); lta = np.log10([p[4] for p in inst])
    na = np.array([p[1] for p in inst])
    print(f"\n      Toutes instables : corr(2J, log τ) = {np.corrcoef(j2a, lta)[0,1]:+.2f}")
    print(f"                         corr(2J, n)     = {np.corrcoef(j2a, na)[0,1]:+.2f}")
    print("=" * 86)


if __name__ == "__main__":
    main()
