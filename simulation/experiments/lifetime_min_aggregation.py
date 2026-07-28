r"""
« Min des facteurs » : la durée de vie fixée par le maillon FAIBLE (canal dominant).

Au lieu de SOMMER les fragilités des nœuds (Σ_i 1/S_i) dans le facteur de
stabilité, on prend le nœud le plus fragile : max_i(1/S_i) (= min_i S_i).
Image de taux : les canaux parallèles ajoutent leurs taux, le plus rapide
domine -> τ ≈ min_i τ_i, soit le MAX de la fragilité. C'est l'idée « min des
facteurs ».

On compare, combiné à P15 (ν=5π/2 fixe, fit intercept+λ+γ), trois agrégations
du facteur de stabilité φ_i = U_i/ρ_{J,i}² :
    somme   Σ_i φ_i      (le modèle actuel, RMS 0,233)
    max     max_i φ_i    (maillon faible = « min des facteurs »)
    moyenne (Σφ)/n
et idem pour la proximité géométrique δ (maillon faible = min_i δ_i).

Usage :  python3 -m simulation.experiments.lifetime_min_aggregation
"""
from __future__ import annotations

import numpy as np

from theory.abraham import geometric_integrals
from theory.retraction import retraction_factor
from theory.jones import jones_polynomial
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA
from simulation.experiments.lifetime_stability_prob import min_strand_distance

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
NU = 5 * np.pi / 2


def rows_data():
    rows = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        phi, delt = [], []
        for kn in p["knots"]:
            gi = geometric_integrals(kn, 10000)
            rho = retraction_factor(jones_polynomial(kn))
            phi.append(gi["U"] / rho ** 2)
            delt.append(min_strand_distance(kn)["delta"])
        phi = np.array(phi); delt = np.array(delt)
        rows.append({
            "name": p["name"], "n": float(len(p["knots"])),
            "logM": np.log10(cornell_equilibrium(p["knots"], p["k"])["mass_raw"]),
            "logtau": np.log10(TAU_EXP_DATA[p["name"]]),
            "phi_sum": phi.sum(), "phi_max": phi.max(), "phi_mean": phi.mean(),
            "ln_dmin": np.log(delt.min()),       # maillon faible (proximité)
            "ln_dsum": np.log(delt).sum(),
        })
    return rows


def fit(rows, key):
    """log10 τ = c - ν logM - (λ')n + γ·X(key) ; ν fixe. Renvoie RMS, γ, résidus."""
    y = np.array([r["logtau"] + NU * r["logM"] for r in rows])   # retire -ν logM
    cols = [np.ones(len(rows)), [r["n"] for r in rows]]
    if key is not None:
        cols.append([r[key] for r in rows])
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    g = coef[2] if key is not None else 0.0
    return {"rms": float(np.sqrt(np.mean(resid ** 2))), "gamma": g,
            "resid": dict(zip([r["name"] for r in rows], resid))}


def main():
    rows = rows_data()
    print("=" * 84)
    print("  « MIN DES FACTEURS » — maillon faible (max fragilité) vs somme")
    print("=" * 84)
    print(f"\n  Combiné à P15 (ν=5π/2 fixe). Réf P15 seul : RMS 0,303.\n")

    base = fit(rows, None)
    print(f"  {'agrégation du facteur de stabilité':>40} {'γ':>9} {'RMS':>7}  neutron")
    print("  " + "-" * 68)
    print(f"  {'P15 seul (pas de facteur)':>40} {'—':>9} {base['rms']:>7.3f}"
          f"  {base['resid']['d']:>+7.3f}")
    tests = [
        ("× exp(-γ Σ φ)   tenseur, SOMME", "phi_sum"),
        ("× exp(-γ max φ) tenseur, MAILLON FAIBLE", "phi_max"),
        ("× exp(-γ moy φ) tenseur, MOYENNE", "phi_mean"),
        ("× (min δ)^γ     proximité, MAILLON FAIBLE", "ln_dmin"),
        ("× (Π δ)^γ       proximité, PRODUIT", "ln_dsum"),
    ]
    best = None
    for lab, key in tests:
        m = fit(rows, key)
        if best is None or m["rms"] < best[1]["rms"]:
            best = (lab, m)
        flag = "  <-- bat somme(0,233)" if m["rms"] < 0.233 - 1e-3 else (
            "  <-- bat P15(0,303)" if m["rms"] < 0.303 - 1e-3 else "")
        print(f"  {lab:>40} {m['gamma']:>+9.4f} {m['rms']:>7.3f}"
              f"  {m['resid']['d']:>+7.3f}{flag}")

    print(f"\n  Meilleure agrégation : {best[0]}  (RMS {best[1]['rms']:.3f})")
    print("\n  Lecture : si 'max φ' (maillon faible) bat 'Σ φ' (somme), le canal")
    print("  dominant fixe la durée de vie -> ton idée 'min des facteurs' gagne.")
    print("  (réserve : 'min δ' utilise les courbes catalogue, artefacts possibles.)")
    print("=" * 84)


if __name__ == "__main__":
    main()
