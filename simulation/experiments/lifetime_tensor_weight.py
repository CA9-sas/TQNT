"""
Loi de dégradation TENSORIELLE : pondération par nœud (canaux non identiques).

P15 scalaire :  τ = τ_P · exp(-λ n) / M_raw^ν   (les n nœuds = n canaux IDENTIQUES)
Tensorielle   :  τ = τ_P · exp(-λ W_K) / M_raw^ν,  W_K = Σ_i w_i

où w_i est le POIDS TENSORIEL du nœud i (son activité comme canal de cassure).
Intuition (confirmée par corrélation ρ_J/√U ↔ résidu = -0,47) : un nœud TENDU
(grand √U = courbure+torsion) se casse plus vite -> w_i grand ; un nœud RIGIDE
(grand ρ_J de Jones) résiste -> w_i petit. D'où w_i ∝ √U_i / ρ_J,i.

On calibre τ_P sur le muon, on garde ν = 5π/2 (canal faible), et on ajuste le
SEUL paramètre λ par moindres carrés sur la famille faible — exactement le même
budget de paramètres que P15. On compare le RMS au P15 scalaire (0,24 déc).

Usage :  python3 -m simulation.experiments.lifetime_tensor_weight
"""
from __future__ import annotations

import numpy as np

from theory.abraham import geometric_integrals
from theory.retraction import retraction_factor
from theory.jones import jones_polynomial
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA, NU_WEAK_ALG

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]   # d = canal β du neutron
NU = NU_WEAK_ALG                              # 5π/2, canal faible
LN10 = np.log(10.0)

# cache géométrique par nœud
_GEO: dict[str, dict] = {}


def node_geo(name: str) -> dict:
    if name not in _GEO:
        gi = geometric_integrals(name, 10000)
        rho = retraction_factor(jones_polynomial(name))
        _GEO[name] = {**gi, "rho": rho}
    return _GEO[name]


def weight_forms(name: str) -> dict[str, float]:
    """Différentes définitions candidates du poids tensoriel w_i d'un nœud."""
    g = node_geo(name)
    Uk, Ut, U, rho = g["U_kappa"], g["U_tau"], g["U"], g["rho"]
    return {
        "P15 (w=1)":          1.0,
        "√U/ρ_J":             np.sqrt(U) / rho,
        "√U_κ/ρ_J":           np.sqrt(Uk) / rho,
        "√U_τ/ρ_J":           np.sqrt(Ut) / rho,
        "U/ρ_J²  (1/S_K)":    U / rho ** 2,
        "√(U_κU_τ)/ρ_J":      (Uk * Ut) ** 0.25 / rho,
    }


def fit_and_score(form: str) -> dict:
    """Pour une définition de poids : W_K par particule, calibre sur muon,
    ajuste λ par moindres carrés, renvoie résidus et RMS."""
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        W = sum(weight_forms(kn)[form] for kn in p["knots"])
        M = cornell_equilibrium(p["knots"], p["k"])["mass_raw"]
        rec.append({"name": p["name"], "W": W, "M": M,
                    "tau_exp": TAU_EXP_DATA[p["name"]]})

    mu = next(r for r in rec if r["name"] == "mu-")
    lt_mu = np.log10(mu["tau_exp"])

    # log10 τ_pred(i) = lt_mu + ν log10(M_mu/M_i) - λ (W_i - W_mu)/ln10
    # résidu = A_i - λ B_i,  A_i = [lt_mu + ν log10(M_mu/M_i)] - log10 τ_exp,i
    A, B, names = [], [], []
    for r in rec:
        A_i = lt_mu + NU * np.log10(mu["M"] / r["M"]) - np.log10(r["tau_exp"])
        B_i = (r["W"] - mu["W"]) / LN10
        A.append(A_i); B.append(B_i); names.append(r["name"])
    A = np.array(A); B = np.array(B)
    lam = float(np.sum(A * B) / np.sum(B * B)) if np.sum(B * B) > 0 else 0.0

    resid = A - lam * B
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {"form": form, "lambda": lam, "rms": rms,
            "resid": dict(zip(names, resid)),
            "W": {r["name"]: r["W"] for r in rec}}


def fit_additive(form: str) -> dict:
    """Modèle 2-param : log τ = C - (λ n + γ T_K)/ln10 - ν log M, T_K = Σ tensor_i.
    Teste si le tenseur porte un signal INDÉPENDANT du comptage n.
    Renvoie λ, γ, RMS et résidus."""
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        n = float(len(p["knots"]))
        T = sum(weight_forms(kn)[form] for kn in p["knots"])
        M = cornell_equilibrium(p["knots"], p["k"])["mass_raw"]
        rec.append({"name": p["name"], "n": n, "T": T, "M": M,
                    "tau_exp": TAU_EXP_DATA[p["name"]]})
    mu = next(r for r in rec if r["name"] == "mu-")
    lt_mu = np.log10(mu["tau_exp"])
    # résidu_i = A_i - λ Bn_i - γ Bt_i
    A, Bn, Bt, names = [], [], [], []
    for r in rec:
        A.append(lt_mu + NU * np.log10(mu["M"] / r["M"]) - np.log10(r["tau_exp"]))
        Bn.append((r["n"] - mu["n"]) / LN10)
        Bt.append((r["T"] - mu["T"]) / LN10)
        names.append(r["name"])
    A = np.array(A); X = np.column_stack([Bn, Bt])
    coef, *_ = np.linalg.lstsq(X, A, rcond=None)
    lam, gam = float(coef[0]), float(coef[1])
    resid = A - X @ coef
    return {"form": form, "lambda": lam, "gamma": gam,
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip(names, resid))}


def main():
    print("=" * 92)
    print("  LOI DE DÉGRADATION TENSORIELLE — pondération par nœud  τ = τ_P e^{-λ ΣW_i}/M^ν")
    print("=" * 92)
    print(f"\n  ν = 5π/2 = {NU:.3f} (canal faible), τ_P calibré sur le muon, λ ajusté (1 param).")
    print("  Comparaison du RMS (décades) sur la famille faible : d(neutron), s, c, τ, b (+μ=0).\n")

    forms = list(weight_forms("3_1").keys())
    results = [fit_and_score(f) for f in forms]
    results.sort(key=lambda r: r["rms"])

    print(f"  {'forme du poids w_i':>20} {'λ':>8} {'RMS(déc)':>9}   résidu neutron(d)")
    print("  " + "-" * 70)
    for r in results:
        tag = "  <-- P15 scalaire" if r["form"].startswith("P15") else ""
        print(f"  {r['form']:>20} {r['lambda']:>8.3f} {r['rms']:>9.3f}"
              f"   {r['resid']['d']:>+7.3f}{tag}")

    best = results[0]
    print(f"\n  Meilleure forme : {best['form']}  (RMS {best['rms']:.3f} déc, λ={best['lambda']:.3f})")
    print("  Résidus par particule (log10 τ_calc/τ_exp) :")
    for nm in WEAK:
        wtot = best["W"][nm]
        print(f"    {nm:>6} : W_K={wtot:>7.2f}  résidu={best['resid'][nm]:>+7.3f}")
    p15 = next(r for r in results if r["form"].startswith("P15"))
    print(f"\n  P15 scalaire (w=1) : RMS {p15['rms']:.3f} déc, neutron {p15['resid']['d']:+.3f}")
    gain = (1 - best["rms"] / p15["rms"]) * 100 if p15["rms"] else 0
    print(f"  Gain (remplacement) : {gain:+.0f}% sur le RMS, neutron {p15['resid']['d']:+.3f} -> {best['resid']['d']:+.3f}")

    print("\n" + "=" * 92)
    print("  TEST DÉCISIF — tenseur comme CORRECTION ADDITIVE (2 param λ,γ) : signal indépendant de n ?")
    print("=" * 92)
    print(f"  {'forme tensorielle T':>20} {'λ':>7} {'γ':>8} {'RMS(déc)':>9}  neutron")
    print("  " + "-" * 64)
    add = [fit_additive(f) for f in forms if not f.startswith("P15")]
    add.sort(key=lambda r: r["rms"])
    for r in add:
        flag = "  <-- bat P15" if r["rms"] < p15["rms"] - 1e-3 else ""
        print(f"  {r['form']:>20} {r['lambda']:>7.3f} {r['gamma']:>+8.4f}"
              f" {r['rms']:>9.3f}  {r['resid']['d']:>+7.3f}{flag}")
    print(f"\n  (Référence P15 1-param : RMS {p15['rms']:.3f}. γ≈0 ou RMS≥0,30 => le tenseur n'aide pas.)")
    print("=" * 92)


if __name__ == "__main__":
    main()
