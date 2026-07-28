"""
Croisements comme STRESS (contrainte locale), pas comme multiplicité de canaux.

Image (L. Cariou) : le fil de flux casse là où il est le plus SERRÉ — aux
croisements où le pliage concentre la contrainte. Le croisement n'ajoute pas un
canal ; il mesure la TENSION locale d'une jonction. On garde donc n canaux
(nœuds) et on module la dégradation par un stress de croisement :

    τ = τ_P · exp[-(λ n + γ·STRESS)] / M_raw^ν

avec STRESS construit sur les croisements c_i et longueurs L_i des composantes :
  densité  c_i/L_i,  brut c_i,  c_i²/L_i ...
et deux contractions :
  SOMME   Σ_i s_i        (toutes les jonctions contribuent)
  MAX     max_i s_i      (maillon faible : le fil casse au point le plus serré)

Réf. : P15 scalaire RMS 0,303 ; tenseur nœud 1/S_K RMS 0,238.

Usage :  python3 -m simulation.experiments.lifetime_crossing_stress
"""
from __future__ import annotations

import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA, NU_WEAK_ALG
from simulation.experiments.lifetime_tensor_weight import node_geo

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
NU = NU_WEAK_ALG
LN10 = np.log(10.0)


def crossings(name: str) -> int:
    return int(name.split("_")[0])


def node_stress(name: str) -> dict[str, float]:
    """Stress de croisement d'un nœud (plusieurs définitions)."""
    g = node_geo(name)
    c = crossings(name)
    L = g["L"]
    return {
        "c/L (densité)":  c / L,
        "c (brut)":       float(c),
        "c²/L":           c * c / L,
        "c·√(U_κ)":       c * np.sqrt(g["U_kappa"]),
    }


def contractions(knots: list[str], form: str) -> dict[str, float]:
    s = [node_stress(kn)[form] for kn in knots]
    return {"Σ": float(sum(s)), "max": float(max(s)), "moy": float(np.mean(s))}


def fit_additive(form: str, contr: str) -> dict:
    """τ = τ_P exp(-(λ n + γ T))/M^ν ; T = contraction du stress ; calibre muon."""
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        rec.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "T": contractions(p["knots"], form)[contr],
            "M": cornell_equilibrium(p["knots"], p["k"])["mass_raw"],
            "tau_exp": TAU_EXP_DATA[p["name"]],
        })
    mu = next(r for r in rec if r["name"] == "mu-")
    lt_mu = np.log10(mu["tau_exp"])
    A, Bn, Bt, names = [], [], [], []
    for r in rec:
        A.append(lt_mu + NU * np.log10(mu["M"] / r["M"]) - np.log10(r["tau_exp"]))
        Bn.append((r["n"] - mu["n"]) / LN10)
        Bt.append((r["T"] - mu["T"]) / LN10)
        names.append(r["name"])
    A = np.array(A); X = np.column_stack([Bn, Bt])
    coef, *_ = np.linalg.lstsq(X, A, rcond=None)
    resid = A - X @ coef
    return {"lambda": float(coef[0]), "gamma": float(coef[1]),
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip(names, resid))}


def main():
    print("=" * 94)
    print("  CROISEMENTS COMME STRESS — le fil casse où il est le plus serré")
    print("=" * 94)
    forms = ["c/L (densité)", "c (brut)", "c²/L", "c·√(U_κ)"]

    print("\n  Stress par particule (contraction SOMME et MAX) :")
    print(f"  {'part.':>6} " + " ".join(f"{f[:9]:>10}" for f in forms))
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        vals = []
        for f in forms:
            c = contractions(p["knots"], f)
            vals.append(f"{c['Σ']:.2f}/{c['max']:.2f}")
        print(f"  {p['name']:>6} " + " ".join(f"{v:>10}" for v in vals))
    print("  (format Σ/max)")

    print(f"\n  Ajustement additif (2 param λ,γ), réf. P15=0,303 / tenseur 1/S=0,238 :")
    print(f"  {'stress':>16} {'contr.':>6} {'λ':>7} {'γ':>9} {'RMS':>7}  neutron")
    print("  " + "-" * 60)
    best = None
    for f in forms:
        for contr in ["Σ", "max", "moy"]:
            r = fit_additive(f, contr)
            if best is None or r["rms"] < best["rms"]:
                best = {**r, "form": f, "contr": contr}
            flag = ""
            if r["rms"] < 0.238 - 1e-3:
                flag = "  <-- bat tenseur 1/S"
            elif r["rms"] < 0.303 - 1e-3:
                flag = "  <-- bat P15"
            print(f"  {f:>16} {contr:>6} {r['lambda']:>7.3f} {r['gamma']:>+9.4f}"
                  f" {r['rms']:>7.3f}  {r['resid']['d']:>+7.3f}{flag}")

    print(f"\n  Meilleur stress de croisement : {best['form']} ({best['contr']}), "
          f"RMS {best['rms']:.3f}, neutron {best['resid']['d']:+.3f}")
    print("=" * 94)


if __name__ == "__main__":
    main()
