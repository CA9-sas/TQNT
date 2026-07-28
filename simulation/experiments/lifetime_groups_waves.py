r"""
Durée de vie par GROUPE D'ONDES ENTRELACÉES (n composantes) — exploration.

Idée (L. Cariou) : organiser par groupe = nombre d'ondes entrelacées.
  Groupe 1 : nœud d'une onde (n=1)
  Groupe 2 : entrelacs de 2 ondes (n=2)
  Groupe 3 : entrelacs de 3 ondes (n=3)
  ...

Question : à n fixé, la durée de vie suit-elle une loi propre au groupe
(p.ex. τ ∝ Q^ν avec ν, préfacteur dépendant de n) ? Ou bien le canal de
désintégration (leptonique / non-leptonique / charm / bottom) TRAVERSE-t-il
les groupes, rendant le regroupement par n peu informatif à lui seul ?

On classe les particules (schéma V4) par n, on ajuste τ ∝ Q^ν par groupe, et on
croise n × canal pour voir la structure.

Usage :  python3 -m simulation.experiments.lifetime_groups_waves
"""
from __future__ import annotations

import numpy as np

# nom, n (nb d'ondes), Q (MeV) ou None, τ_exp(s) ou None(stable)/"conf", canal
P = [
    # groupe 1 : nœud d'une onde
    ("e-",     1, None,  None,   "stable"),
    ("u",      1, None,  None,   "stable"),
    ("d",      1, None,  "conf", "confiné"),
    ("pi+-",   1, 33.9,  2.603e-8,  "2corps"),
    ("pi0",    1, 135.,  8.43e-17,  "EM"),
    # groupe 2 : entrelacs de 2 ondes
    ("nu",     2, None,  None,   "stable"),
    ("mu-",    2, 105.7, 2.197e-6,  "leptonique"),
    ("s",      2, None,  "conf", "confiné"),
    ("c",      2, None,  "conf", "confiné"),
    ("Lambda", 2, 37.8,  2.617e-10, "non-lept"),
    ("Sigma+", 2, 113.,  8.018e-11, "non-lept"),
    ("Sigma-", 2, 118.,  1.479e-10, "non-lept"),
    ("D0",     2, 1270., 4.10e-13,  "charm"),
    ("D+",     2, 1270., 1.033e-12, "charm"),
    ("Lambdac",2, 1270., 2.03e-13,  "charm"),
    ("B",      2, 4180., 1.519e-12, "bottom"),
    # groupe 3 : entrelacs de 3 ondes
    ("tau-",   3, 1777., 2.903e-13, "leptonique"),
    ("p",      3, None,  None,   "stable"),
    ("n",      3, 0.78,  878.4,  "beta"),
    ("K+-",    3, 388.,  1.238e-8,  "2corps"),
    ("K0_S",   3, 218.,  8.954e-11, "non-lept"),
    ("Xi0",    3, 64.2,  2.90e-10,  "non-lept"),
    ("Xi-",    3, 66.5,  1.639e-10, "non-lept"),
    ("Omega-", 3, 63.0,  8.21e-11,  "non-lept"),
    ("Omegac", 3, 1270., 2.68e-13,  "charm"),
    ("Lambdab",3, 4180., 1.471e-12, "bottom"),
    ("W",      3, None,  3.157e-25, "EW"),
    ("Z0",     3, None,  2.638e-25, "EW"),
    ("H",      3, None,  1.6e-22,   "EW"),
    ("t",      3, None,  5.0e-25,   "EW"),
]


def main():
    print("=" * 84)
    print("  DURÉE DE VIE PAR GROUPE D'ONDES ENTRELACÉES (n) — exploration")
    print("=" * 84)

    for n in (1, 2, 3):
        grp = [p for p in P if p[1] == n]
        print(f"\n  ===== GROUPE {n} (entrelacs de {n} onde{'s' if n>1 else ''}) — "
              f"{len(grp)} particules =====")
        for nm, _, Q, tau, ch in grp:
            qs = f"{Q:>7.1f}" if Q else "   --- "
            ts = ("stable" if tau is None else ("confiné" if tau == "conf"
                  else f"{tau:.2e}"))
            print(f"    {nm:>8} Q={qs}  τ={ts:>10}  [{ch}]")

        # fit τ ∝ Q^ν sur les instables avec Q de ce groupe
        wk = [p for p in grp if isinstance(p[3], float) and p[2]]
        if len(wk) >= 3:
            lQ = np.log10([p[2] for p in wk]); lt = np.log10([p[3] for p in wk])
            A = np.column_stack([np.ones(len(wk)), -lQ])
            c, *_ = np.linalg.lstsq(A, lt, rcond=None)
            rms = np.sqrt(np.mean((lt - A @ c) ** 2))
            cr = np.corrcoef(lQ, lt)[0, 1]
            print(f"    -> fit τ∝Q^ν (groupe entier) : ν={c[1]:.2f}, "
                  f"RMS={rms:.2f} déc, corr={cr:+.2f}  ({len(wk)} pts)")
            # canaux présents dans ce groupe
            chans = sorted(set(p[4] for p in wk))
            print(f"       canaux mélangés dans le groupe : {', '.join(chans)}")

    # croisement n × canal : le canal traverse-t-il les groupes ?
    print("\n" + "=" * 84)
    print("  CROISEMENT  groupe(n) × canal de désintégration")
    print("=" * 84)
    chans = ["leptonique", "2corps", "non-lept", "charm", "bottom", "beta", "EW", "EM"]
    print(f"  {'canal':>12} | " + " ".join(f"n={n}" for n in (1, 2, 3)))
    for ch in chans:
        row = [sum(1 for p in P if p[1] == n and p[4] == ch) for n in (1, 2, 3)]
        if sum(row):
            print(f"  {ch:>12} | " + "   ".join(str(x) for x in row))
    print("\n  Si un canal apparaît sur PLUSIEURS colonnes -> il traverse les groupes")
    print("  -> le regroupement par n seul ne sépare pas les mécanismes.")
    print("=" * 84)


if __name__ == "__main__":
    main()
