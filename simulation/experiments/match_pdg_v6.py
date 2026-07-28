"""Confrontation du catalogue V6 unifié aux particules PDG mesurées.

Les expériences mesurent les observables AU REPOS (basse énergie) : masse, spin,
charge, durée de vie. On cherche, pour chaque particule PDG, l'état (config, Lk)
du catalogue dont les observables au repos collent, avec la CHARGE EXACTE
(Q = Σχ + Lk entier) comme filtre dur, plus le spin (parité) et la durée de vie.
"""
from __future__ import annotations
import openpyxl

CAT = "docs/TQNT_catalog_V6_unifie.xlsx"

# (nom, m_exp MeV, spin, charge, τ_exp s)
PDG = [
    ("e-", 0.511, 0.5, -1, 1e40), ("mu-", 105.66, 0.5, -1, 2.197e-6), ("tau-", 1776.9, 0.5, -1, 2.9e-13),
    ("pi+", 139.57, 0, 1, 2.6e-8), ("K+", 493.68, 0, 1, 1.24e-8), ("eta", 547.86, 0, 0, 5e-19),
    ("rho+", 775.26, 1, 1, 4.5e-24), ("omega", 782.66, 1, 0, 7.8e-23), ("phi", 1019.46, 1, 0, 1.5e-22),
    ("D+", 1869.66, 0, 1, 1.0e-12), ("D0", 1864.84, 0, 0, 4.1e-13), ("J/psi", 3096.9, 1, 0, 7.1e-21),
    ("B+", 5279.34, 0, 1, 1.6e-12), ("Ups", 9460.3, 1, 0, 1.2e-20),
    ("p", 938.27, 0.5, 1, 1e40), ("n", 939.57, 0.5, 0, 879.0), ("Lambda", 1115.68, 0.5, 0, 2.6e-10),
    ("Sigma+", 1189.37, 0.5, 1, 8.0e-11), ("Sigma-", 1197.45, 0.5, -1, 1.5e-10),
    ("Xi-", 1321.71, 0.5, -1, 1.6e-10), ("Omega-", 1672.45, 1.5, -1, 8.2e-11),
    ("Lambda_c", 2286.46, 0.5, 1, 2.0e-13), ("Delta++", 1232.0, 1.5, 2, 5.6e-24),
    ("W", 80377.0, 1, 1, 3.2e-25), ("Z", 91187.6, 1, 0, 2.6e-25), ("H", 125250.0, 0, 0, 1.6e-22),
]


def parity(spin_set):
    """parité du spin (half-int=fermion / int=boson) d'un ensemble '0.5, 1.5'."""
    try:
        v = float(str(spin_set).split(",")[0].strip())
    except (ValueError, AttributeError):
        return None
    return int(2 * v) % 2  # 1 = demi-entier (fermion), 0 = entier (boson)


def main():
    wb = openpyxl.load_workbook(CAT, data_only=True, read_only=True)
    ws = wb["Catalogue V6 unifié"]
    states = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[1] is None or r[6] is None:
            continue
        # 1 cfg, 4 spin, 5 Q, 6 m_repos, 8 τ_repos
        states.append({"cfg": r[1], "Lk": r[4 - 0] if False else None, "spin": r[4],
                       "Q": r[5], "m": r[6], "tau": r[8]})

    print(f"{'PDG':>9} | {'m_exp':>8} | {'best (config)':>20} | {'m_TQNT':>8} | {'Δm%':>6} | {'#cand':>5}")
    nuniq = 0; deg = []
    for nm, mx, sp, q, tx in PDG:
        par = int(2 * sp) % 2
        tol = 0.03 if mx < 3000 else 0.08
        cand = [s for s in states if s["Q"] == q and parity(s["spin"]) == par
                and abs(s["m"] / mx - 1) <= tol]
        deg.append(len(cand))
        if cand:
            best = min(cand, key=lambda s: abs(s["m"] / mx - 1))
            if len(cand) == 1:
                nuniq += 1
            print(f"{nm:>9} | {mx:>8.1f} | {best['cfg']:>20} | {best['m']:>8.1f} | "
                  f"{(best['m']/mx-1)*100:>+5.1f} | {len(cand):>5}")
        else:
            print(f"{nm:>9} | {mx:>8.1f} | {'AUCUN (charge exacte)':>20} | {'':>8} | {'':>6} | {0:>5}")
    import numpy as np
    print(f"\n  {sum(1 for d in deg if d>0)}/{len(PDG)} appariés ; uniques : {nuniq} ; "
          f"degré médian {int(np.median(deg))}, max {max(deg)}")


if __name__ == "__main__":
    main()
