"""
Exploration : le résidu de la durée de vie P15 corrèle-t-il avec le TENSEUR
du nœud/entrelacs (courbure U_κ, torsion-magnétique U_τ) ?

Principe de dégradation (P15) :  τ = τ_P · exp(-λ n) / M_raw^ν
  — n = nb de nœuds (canaux de cassure), M_raw = masse topologique brute.
Cette loi SCALAIRE ignore la STRUCTURE GÉOMÉTRIQUE (tensorielle) de chaque
composante. Or chaque nœud porte un tenseur de Frenet (κ courbure, τ torsion) :
  U_κ = ∮κ²ds  (énergie de flexion),  U_τ = ∮τ²ds  (énergie magnétique).

Hypothèse à tester : le résidu log10(τ_P15/τ_exp) de la famille FAIBLE
corrèle avec une grandeur tensorielle (p.ex. U_τ/U_κ, ou U_τ²/(L U_κ)),
ce qui justifierait d'intégrer le tenseur dans la loi de dégradation.

Usage :  python3 -m simulation.experiments.lifetime_tensor_explore
"""
from __future__ import annotations

import numpy as np

from theory.abraham import geometric_integrals
from theory.retraction import retraction_factor
from theory.jones import jones_polynomial
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import sm_lifetime_table_algebraic, TAU_EXP_DATA

# famille faible (le canal "d" porte la β du neutron, τ_exp = 878,4 s)
WEAK = {"d", "mu-", "s", "c", "tau-", "b"}


def link_tensor(knots: list[str]) -> dict:
    """Contractions tensorielles d'un entrelacs : somme des U_κ, U_τ, L, ρ_J."""
    U_k = U_t = L = rho = 0.0
    for name in knots:
        gi = geometric_integrals(name, 10000)
        U_k += gi["U_kappa"]
        U_t += gi["U_tau"]
        L += gi["L"]
        rho += retraction_factor(jones_polynomial(name))
    return {"U_kappa": U_k, "U_tau": U_t, "L": L, "rho": rho,
            "ratio_tk": U_t / U_k if U_k else 0.0}


def main():
    print("=" * 100)
    print("  EXPLORATION — résidu durée de vie P15 vs TENSEUR (U_κ, U_τ) — famille faible")
    print("=" * 100)
    rows = {r["name"]: r for r in sm_lifetime_table_algebraic()}

    hdr = (f"{'part.':>6} {'n':>2} {'M_raw':>9} {'ΣU_κ':>8} {'ΣU_τ':>8} "
           f"{'U_τ/U_κ':>8} {'τ_P15':>11} {'τ_exp':>11} {'résidu':>8}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr)))

    data = []
    for p in SM_ASSIGNMENTS:
        name = p["name"]
        if name not in WEAK:
            continue
        eq = cornell_equilibrium(p["knots"], p["k"])
        lt = link_tensor(p["knots"])
        r = rows[name]
        tau_p15 = r["tau_p15"]
        tau_exp = TAU_EXP_DATA[name]
        resid = np.log10(tau_p15 / tau_exp) if tau_exp else None
        n = len(p["knots"])
        print(f"{name:>6} {n:>2} {eq['mass_raw']:>9.3f} {lt['U_kappa']:>8.1f} "
              f"{lt['U_tau']:>8.1f} {lt['ratio_tk']:>8.3f} {tau_p15:>11.3e} "
              f"{tau_exp:>11.3e} {resid:>+8.3f}")
        data.append({"name": name, "n": n, "M_raw": eq["mass_raw"],
                     "U_k": lt["U_kappa"], "U_t": lt["U_tau"],
                     "ratio": lt["ratio_tk"], "resid": resid,
                     "L": lt["L"], "rho": lt["rho"]})

    # corrélations résidu vs grandeurs tensorielles candidates
    res = np.array([d["resid"] for d in data])
    print("\n  Corrélation (Pearson) du RÉSIDU P15 avec :")
    cands = {
        "U_τ/U_κ":           np.array([d["ratio"] for d in data]),
        "log ΣU_τ":          np.log([d["U_t"] for d in data]),
        "log ΣU_κ":          np.log([d["U_k"] for d in data]),
        "log U_τ²/(L·U_κ)":  np.log([d["U_t"]**2 / (d["L"] * d["U_k"]) for d in data]),
        "log(ΣU_τ/n)":       np.log([d["U_t"] / d["n"] for d in data]),
        "log ρ_J/√U":        np.log([d["rho"] / np.sqrt(d["U_k"] + d["U_t"]) for d in data]),
    }
    for label, x in cands.items():
        c = np.corrcoef(x, res)[0, 1]
        print(f"    {label:>18} : r = {c:+.3f}")

    print("\n  Lecture : un |r| élevé => cette grandeur tensorielle explique une")
    print("  part du résidu P15 et mérite d'entrer dans la loi de dégradation.")
    print("=" * 100)


if __name__ == "__main__":
    main()
