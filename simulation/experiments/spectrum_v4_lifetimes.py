r"""
Spectre élargi en SCHÉMA V4 (petites configs, validé masses) + durées de vie.

But : unifier sur les assignations topologiques du tableau « Spectre 46 » de
TQNT_detaillee_V4_FR (p = 3_1+7_2+7_2, μ = 3_1+6_1, Λ = 4_1+7_4, Ω = 5_2+5_2+7_2,
...), RECALIBRER la loi de durée de vie dans ce schéma, et tester sur le spectre
élargi (canal faible : ~17 particules au lieu de 6) si l'agrégation du facteur
de stabilité par MAILLON FAIBLE (max 1/S) bat la SOMME (Σ 1/S).

Loi :  τ = τ_P · M^{-ν} · e^{-λ n} · exp(-γ · agg_i(U_i/ρ_{J,i}²))
  - M = masse TQNT (MeV) du tableau V4 ;
  - n = nombre de composantes ;  agg = Σ (somme) ou max (maillon faible).
On ajuste (τ_P, ν, λ, γ) par canal (régression log-linéaire).

Durées de vie : PDG (résonances : τ = ħ/Γ, ħ = 6,582e-22 MeV·s).

Usage :  python3 -m simulation.experiments.spectrum_v4_lifetimes
"""
from __future__ import annotations

import numpy as np

from theory.abraham import geometric_integrals
from theory.retraction import retraction_factor
from theory.jones import jones_polynomial

HBAR = 6.582e-22  # MeV·s

# nom, knots (schéma V4), m_TQNT (MeV), tau_exp (s) ou None si stable, canal
# canal : "faible", "EM", "fort" (résonances + EW rapides)
SPECTRUM = [
    # leptons
    ("e-",   ["3_1"],                     0.51,   None,      "stable"),
    ("mu-",  ["3_1", "6_1"],              102.0,  2.197e-6,  "faible"),
    ("tau-", ["3_1", "5_1", "7_1"],       1781.0, 2.903e-13, "faible"),
    # mésons légers
    ("pi0",  ["8_1"],                     134.0,  8.43e-17,  "EM"),
    ("pi+-", ["8_1"],                     134.0,  2.603e-8,  "faible"),
    ("K+-",  ["4_1", "4_1", "5_2"],       494.0,  1.238e-8,  "faible"),
    ("K0_S", ["4_1", "4_1", "5_2"],       494.0,  8.954e-11, "faible"),
    ("eta",  ["3_1", "4_1", "5_1"],       545.0,  5.02e-19,  "EM"),
    ("etap", ["5_2", "5_2", "6_1"],       955.0,  3.50e-21,  "fort"),
    ("omega",["5_2", "7_7"],              782.0,  7.75e-23,  "fort"),
    ("phi",  ["3_1", "4_1", "6_2"],       1015.0, 1.55e-22,  "fort"),
    ("Jpsi", ["4_1", "5_2", "9_1"],       3097.0, 7.11e-21,  "fort"),
    ("Ups",  ["3_1", "7_4", "9_1"],       9428.0, 1.22e-20,  "fort"),
    # mésons lourds
    ("D0",   ["6_2", "7_7"],              1869.0, 4.10e-13,  "faible"),
    ("D+",   ["6_2", "7_7"],              1869.0, 1.033e-12, "faible"),
    ("B",    ["4_1", "10_1"],             5277.0, 1.519e-12, "faible"),
    # baryons légers
    ("p",    ["3_1", "7_2", "7_2"],       940.0,  None,      "stable"),
    ("n",    ["3_1", "7_2", "7_2"],       941.0,  878.4,     "faible"),
    ("Lambda",["4_1", "7_4"],             1123.0, 2.617e-10, "faible"),
    ("Sigma+",["6_2", "7_2"],             1193.0, 8.018e-11, "faible"),
    ("Sigma0",["6_2", "7_2"],             1193.0, 7.4e-20,   "EM"),
    ("Sigma-",["6_2", "7_2"],             1199.0, 1.479e-10, "faible"),
    ("Xi0",  ["4_1", "4_1", "7_2"],       1307.0, 2.90e-10,  "faible"),
    ("Xi-",  ["4_1", "4_1", "7_2"],       1307.0, 1.639e-10, "faible"),
    ("Omega-",["5_2", "5_2", "7_2"],      1672.0, 8.21e-11,  "faible"),
    # résonances baryoniques (fort)
    ("Delta",["3_1", "5_2", "6_2"],       1232.0, 5.63e-24,  "fort"),
    ("Sigma*",["5_2", "6_2", "7_2"],      1384.0, 1.83e-23,  "fort"),
    ("Xi*",  ["4_1", "5_2", "7_2"],       1532.0, 7.23e-23,  "fort"),
    # baryons charmés/bottom
    ("Lambdac",["6_3", "8_1"],            2285.0, 2.03e-13,  "faible"),
    ("Omegac",["3_1", "6_2", "6_2"],      2697.0, 2.68e-13,  "faible"),
    ("Lambdab",["5_2", "5_2", "8_2"],     5613.0, 1.471e-12, "faible"),
    # bosons (EW rapide)
    ("W",    ["4_1", "7_3", "8_3"],       80503.0, 3.157e-25, "fort"),
    ("Z0",   ["5_1", "7_2", "8_3"],       91079.0, 2.638e-25, "fort"),
    ("H",    ["6_1", "7_4", "8_3"],       124956.0, 1.6e-22,  "fort"),
    ("t",    ["7_4", "7_6", "8_3"],       173468.0, 5.0e-25,  "fort"),
]

_G: dict[str, float] = {}


def inv_S(name: str) -> float:
    """1/S = U/ρ_J²  (fragilité d'un nœud)."""
    if name not in _G:
        gi = geometric_integrals(name, 10000)
        rho = retraction_factor(jones_polynomial(name))
        _G[name] = gi["U"] / rho ** 2
    return _G[name]


def features():
    rows = []
    for name, knots, m, tau, chan in SPECTRUM:
        phis = [inv_S(k) for k in knots]
        rows.append({
            "name": name, "knots": knots, "m": m, "tau": tau, "chan": chan,
            "n": float(len(knots)),
            "sum_invS": float(sum(phis)),
            "max_invS": float(max(phis)),
        })
    return rows


def fit_channel(rows, chan, agg):
    """Régression log10 τ = c - ν logM - (λ/ln10) n - (γ/ln10) X(agg). lstsq."""
    memb = [r for r in rows if r["chan"] == chan and r["tau"]]
    if len(memb) < 4:
        return None
    y = np.array([np.log10(r["tau"]) for r in memb])
    X = np.column_stack([
        np.ones(len(memb)),
        -np.log10([r["m"] for r in memb]),
        -np.array([r["n"] for r in memb]),
        -np.array([r[agg] for r in memb]),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"coef": coef, "rms": float(np.sqrt(np.mean(resid ** 2))),
            "n": len(memb), "resid": {r["name"]: rr for r, rr in zip(memb, resid)},
            "memb": [r["name"] for r in memb]}


def main():
    rows = features()
    print("=" * 96)
    print("  SPECTRE ÉLARGI (schéma V4) — recalibration durée de vie ; SOMME vs MAILLON FAIBLE")
    print("=" * 96)
    nf = sum(1 for r in rows if r["chan"] == "faible" and r["tau"])
    ns = sum(1 for r in rows if r["chan"] == "fort" and r["tau"])
    ne = sum(1 for r in rows if r["chan"] == "EM" and r["tau"])
    print(f"\n  Particules avec τ connu : faible {nf}, fort {ns}, EM {ne}.\n")

    for chan in ["faible", "fort"]:
        print(f"  --- canal {chan} ---")
        for agg, lab in [("sum_invS", "SOMME Σ1/S"), ("max_invS", "MAILLON FAIBLE max1/S")]:
            f = fit_channel(rows, chan, agg)
            if f is None:
                print(f"    {lab:>22} : trop peu de points")
                continue
            c = f["coef"]
            print(f"    {lab:>22} : ν={c[1]:.2f} λ={c[2]*np.log(10):.2f} "
                  f"γ={c[3]*np.log(10):.4f}  RMS={f['rms']:.3f} déc ({f['n']} part.)")
        print()

    # discrimination : Δ(RMS) entre somme et maillon faible, canal faible
    fs = fit_channel(rows, "faible", "sum_invS")
    fm = fit_channel(rows, "faible", "max_invS")
    print("  DISCRIMINATION (canal faible, le plus peuplé) :")
    print(f"    SOMME        : RMS {fs['rms']:.3f} déc")
    print(f"    MAILLON FAIBLE: RMS {fm['rms']:.3f} déc")
    win = "MAILLON FAIBLE" if fm["rms"] < fs["rms"] else "SOMME"
    print(f"    -> {win} l'emporte (Δ={abs(fs['rms']-fm['rms']):.3f} déc sur {fs['n']} particules)")
    print("=" * 96)


if __name__ == "__main__":
    main()
