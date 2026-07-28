"""
COMBINER les deux facteurs :  τ = (P15) × (probabilité de stabilité).

P15 (dégradation) :        exp(-λ n) / M_raw^ν   (ν = 5π/2 fixe, canal faible)
Probabilité de stabilité :  P_stab(géométrie)     (les brins ne se touchent pas)

  τ = τ_P · exp(-λ n) / M_raw^ν · P_stab.

On garde TOUT P15 et on ajoute le facteur de stabilité comme un SECOND facteur
(1 paramètre de plus). On teste plusieurs définitions de P_stab et on compare le
RMS à P15 seul (0,303 sur la famille faible, ν fixe).

P_stab testés (chacun via un paramètre γ) :
  exp(-γ Σ U_i/ρ_i²)   : crispation continue (= 1/S, robuste, sans artefact)
  (min δ)^γ            : maillon faible (rapprochement le plus serré, littéral)
  exp(γ Σ ln δ_i)      : produit des stabilités par composante (probabilités ×)
  exp(-γ Σ 1/δ_i)      : crispation ∝ 1/distance

Usage :  python3 -m simulation.experiments.lifetime_p15_x_stability
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


def particle_vars():
    rows = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        invS = 0.0          # Σ U_i/ρ_i²
        lnd, invd = 0.0, 0.0
        mind = np.inf
        for kn in p["knots"]:
            gi = geometric_integrals(kn, 10000)
            rho = retraction_factor(jones_polynomial(kn))
            invS += gi["U"] / rho ** 2
            d = min_strand_distance(kn)["delta"]
            lnd += np.log(d + 1e-9)
            invd += 1.0 / (d + 1e-9)
            mind = min(mind, d)
        rows.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "logM": np.log10(cornell_equilibrium(p["knots"], p["k"])["mass_raw"]),
            "logtau": np.log10(TAU_EXP_DATA[p["name"]]),
            "invS": invS, "lnd": lnd, "invd": invd,
            "logmind": np.log10(mind + 1e-9),
        })
    return rows


def fit(rows, stab_key=None):
    """Régression log10 τ = c - ν·logM - (λ')·n + γ·X (ν fixe). X = stab_key.
    Si stab_key None : P15 seul (c, λ')."""
    y = np.array([r["logtau"] + NU * r["logM"] for r in rows])   # retire pente -ν
    cols = [np.ones(len(rows)), np.array([r["n"] for r in rows])]
    if stab_key is not None:
        cols.append(np.array([r[stab_key] for r in rows]))
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"coef": coef, "rms": float(np.sqrt(np.mean(resid ** 2))),
            "npar": X.shape[1] + 0,  # + ν fixe (non compté)
            "resid": dict(zip([r["name"] for r in rows], resid))}


def main():
    rows = particle_vars()
    print("=" * 90)
    print("  τ = P15 × probabilité de stabilité  —  combinaison des deux facteurs")
    print("=" * 90)
    print(f"\n  ν = 5π/2 fixe. P15 seul : λ' + intercept (2 param).")
    print("  Chaque facteur de stabilité ajoute 1 paramètre (γ). Réf P15 = 0,303.\n")

    p15 = fit(rows, None)
    print(f"  {'modèle':>34} {'#par':>5} {'γ':>9} {'RMS':>7}  neutron")
    print("  " + "-" * 62)
    print(f"  {'P15 seul (exp(-λn)/M^ν)':>34} {p15['npar']:>5} {'—':>9} "
          f"{p15['rms']:>7.3f}  {p15['resid']['d']:>+7.3f}")

    facteurs = [
        ("× exp(-γ Σ U/ρ²)  (1/S continu)", "invS"),
        ("× (min δ)^γ        (maillon faible)", "logmind"),
        ("× exp(γ Σ ln δ)    (produit stab.)", "lnd"),
        ("× exp(-γ Σ 1/δ)    (crispation)", "invd"),
    ]
    best = None
    for lab, key in facteurs:
        m = fit(rows, key)
        gamma = m["coef"][2]
        flag = "  <-- bat P15" if m["rms"] < p15["rms"] - 1e-3 else ""
        if best is None or m["rms"] < best[1]["rms"]:
            best = (lab, m)
        print(f"  {lab:>34} {m['npar']:>5} {gamma:>+9.4f} {m['rms']:>7.3f}"
              f"  {m['resid']['d']:>+7.3f}{flag}")

    print(f"\n  Meilleure combinaison : {best[0]}")
    print(f"    RMS {best[1]['rms']:.3f} (vs P15 seul {p15['rms']:.3f}), "
          f"neutron {best[1]['resid']['d']:+.3f}")
    print("\n  Note : tous ces modèles ont 3 param (P15:2 + stabilité:1) sur 6 pts.")
    print("  Un gain net ET un γ stable => le 2e facteur est réel ; sinon, surajustement.")
    print("=" * 90)


if __name__ == "__main__":
    main()
