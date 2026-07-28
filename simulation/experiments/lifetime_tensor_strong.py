"""
Validation du terme tensoriel 1/S_K = U/ρ_J² sur le CANAL FORT (W, Z, H, t).

Le canal faible donnait : w_i = 1 + γ·(U_i/ρ_J,i²), γ ≈ +0,072, RMS 0,30 -> 0,24.
Question : ce signal est-il RÉEL ou un surajustement de la famille faible ?

Deux tests :
  (1) Ajustement LIBRE (λ, γ) sur le canal fort (ν=7/4, ancré sur W).
      -> le signe et l'ordre de γ_fort doivent rejoindre γ_faible.
  (2) VALIDATION CROISÉE : on FIXE γ = γ_faible et on l'applique au canal fort
      (on n'ajuste que λ). Si le RMS baisse vs γ=0, le tenseur généralise
      hors de la famille qui l'a appris -> signal réel, pas artefact.

Usage :  python3 -m simulation.experiments.lifetime_tensor_strong
"""
from __future__ import annotations

import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA, NU_STRONG_ALG
from simulation.experiments.lifetime_tensor_weight import weight_forms, fit_additive

STRONG = ["W+", "Z0", "H", "t"]
NU = float(NU_STRONG_ALG)   # 7/4
LN10 = np.log(10.0)
FORM = "U/ρ_J²  (1/S_K)"
ANCHOR = "W+"


def strong_records():
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in STRONG:
            continue
        rec.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "T": sum(weight_forms(kn)[FORM] for kn in p["knots"]),
            "M": cornell_equilibrium(p["knots"], p["k"])["mass_raw"],
            "tau_exp": TAU_EXP_DATA[p["name"]],
        })
    return rec


def fit(rec, gamma_fixed=None):
    """Ancre sur W ; ν=7/4 fixe. Si gamma_fixed None -> fit (λ,γ), sinon fit λ seul."""
    a = next(r for r in rec if r["name"] == ANCHOR)
    lt_a = np.log10(a["tau_exp"])
    A, Bn, Bt, names = [], [], [], []
    for r in rec:
        A.append(lt_a + NU * np.log10(a["M"] / r["M"]) - np.log10(r["tau_exp"]))
        Bn.append((r["n"] - a["n"]) / LN10)
        Bt.append((r["T"] - a["T"]) / LN10)
        names.append(r["name"])
    A = np.array(A); Bn = np.array(Bn); Bt = np.array(Bt)
    if gamma_fixed is None:
        X = np.column_stack([Bn, Bt])
        coef, *_ = np.linalg.lstsq(X, A, rcond=None)
        lam, gam = float(coef[0]), float(coef[1])
        resid = A - X @ coef
    else:
        gam = float(gamma_fixed)
        A2 = A - gam * Bt
        lam = float(np.sum(A2 * Bn) / np.sum(Bn * Bn)) if np.sum(Bn * Bn) else 0.0
        resid = A2 - lam * Bn
    return {"lambda": lam, "gamma": gam,
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip(names, resid))}


def main():
    print("=" * 88)
    print("  VALIDATION CANAL FORT — terme tensoriel 1/S_K = U/ρ_J²  (W, Z, H, t ; ν=7/4)")
    print("=" * 88)

    gamma_weak = fit_additive(FORM)["gamma"]
    print(f"\n  γ appris sur le canal FAIBLE (forme {FORM}) : γ_faible = {gamma_weak:+.4f}")

    rec = strong_records()
    print(f"\n  {'part.':>6} {'n':>3} {'T=Σ1/S_K':>9} {'M_raw':>10} {'τ_exp':>11}")
    print("  " + "-" * 46)
    for r in rec:
        print(f"  {r['name']:>6} {r['n']:>3.0f} {r['T']:>9.3f} {r['M']:>10.2f} {r['tau_exp']:>11.3e}")

    base = fit(rec, gamma_fixed=0.0)
    free = fit(rec, gamma_fixed=None)
    cross = fit(rec, gamma_fixed=gamma_weak)

    print("\n  (1) AJUSTEMENT LIBRE sur le canal fort :")
    print(f"      λ = {free['lambda']:+.3f},  γ_fort = {free['gamma']:+.4f},  RMS = {free['rms']:.3f} déc")
    same_sign = (free["gamma"] > 0) == (gamma_weak > 0)
    print(f"      -> γ_fort {'MÊME signe' if same_sign else 'signe OPPOSÉ'} que γ_faible "
          f"({free['gamma']:+.4f} vs {gamma_weak:+.4f})")

    print("\n  (2) VALIDATION CROISÉE (γ = γ_faible imposé, on n'ajuste que λ) :")
    print(f"      sans tenseur (γ=0)        : RMS = {base['rms']:.3f} déc")
    print(f"      avec γ_faible={gamma_weak:+.4f}    : RMS = {cross['rms']:.3f} déc")
    verdict = "AMÉLIORE" if cross["rms"] < base["rms"] - 1e-3 else (
        "n'aide pas" if abs(cross["rms"] - base["rms"]) <= 1e-3 else "DÉGRADE")
    print(f"      -> le tenseur appris sur le faible {verdict} le canal fort.")

    print("\n  VERDICT :")
    if same_sign and cross["rms"] < base["rms"]:
        print("  ✓ Signal tensoriel CONFIRMÉ : même signe en libre + amélioration croisée.")
    elif same_sign:
        print("  ~ Partiel : signe cohérent, mais pas d'amélioration croisée nette.")
    else:
        print("  ✗ Non confirmé : γ change de signe -> probable surajustement du faible.")
    print("=" * 88)


if __name__ == "__main__":
    main()
