"""
Tenseur d'ENLACEMENT de l'entrelacs dans la loi de dégradation.

Idée (proche de l'intuition initiale "tenseur des entrelacs comme le neutron") :
l'entrelacs ne se dégrade pas par ses n NŒUDS isolés, mais par ses LIENS entre
composantes. Chaque paire (i,j) de composantes est un lien (un Hopf, Lk_opt =
C(n,2) selon CLAUDE.md). Le canal de cassure naturel serait donc le LIEN, pas le
nœud, et sa "force" dépend des deux composantes liées.

On teste, en remplacement de n et en correction additive à n, des contractions
PAR PAIRE :
  C(n,2)            = nombre de liens (= Lk_opt, fct de n seul)
  Σ √(ρ_i ρ_j)      = liens pondérés par la rigidité topologique (Jones)
  Σ √(U_i U_j)      = liens pondérés par la tension géométrique
  Σ 1/√(S_i S_j)    = liens pondérés par 1/stabilité structurelle (S=ρ²/U)

Référence : P15 scalaire RMS 0,303 déc ; meilleur nœud (1/S_K additif) 0,238.

Usage :  python3 -m simulation.experiments.lifetime_linking
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA, NU_WEAK_ALG
from simulation.experiments.lifetime_tensor_weight import node_geo

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
NU = NU_WEAK_ALG
LN10 = np.log(10.0)


def pair_contractions(knots: list[str]) -> dict[str, float]:
    """Contractions tensorielles par paire de composantes de l'entrelacs."""
    g = {kn: node_geo(kn) for kn in set(knots)}
    rho = {kn: g[kn]["rho"] for kn in g}
    U = {kn: g[kn]["U"] for kn in g}
    S = {kn: rho[kn] ** 2 / U[kn] for kn in g}
    n = len(knots)
    pairs = list(combinations(knots, 2))
    return {
        "C(n,2) liens":      float(len(pairs)),               # = n(n-1)/2
        "Σ√(ρ_iρ_j)":        sum(np.sqrt(rho[a] * rho[b]) for a, b in pairs),
        "Σ√(U_iU_j)":        sum(np.sqrt(U[a] * U[b]) for a, b in pairs),
        "Σ1/√(S_iS_j)":      sum(1.0 / np.sqrt(S[a] * S[b]) for a, b in pairs),
        "n (réf.)":          float(n),
    }


def _records(form: str):
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        rec.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "W": pair_contractions(p["knots"])[form],
            "M": cornell_equilibrium(p["knots"], p["k"])["mass_raw"],
            "tau_exp": TAU_EXP_DATA[p["name"]],
        })
    return rec


def fit_replace(form: str) -> dict:
    """τ = τ_P exp(-λ W)/M^ν, W = contraction d'enlacement ; calibre muon, fit λ."""
    rec = _records(form)
    mu = next(r for r in rec if r["name"] == "mu-")
    lt_mu = np.log10(mu["tau_exp"])
    A, B, names = [], [], []
    for r in rec:
        A.append(lt_mu + NU * np.log10(mu["M"] / r["M"]) - np.log10(r["tau_exp"]))
        B.append((r["W"] - mu["W"]) / LN10)
        names.append(r["name"])
    A = np.array(A); B = np.array(B)
    lam = float(np.sum(A * B) / np.sum(B * B)) if np.sum(B * B) else 0.0
    resid = A - lam * B
    return {"lambda": lam, "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip(names, resid))}


def fit_additive(form: str) -> dict:
    """τ = τ_P exp(-(λ n + γ W))/M^ν ; calibre muon, fit (λ,γ)."""
    rec = _records(form)
    mu = next(r for r in rec if r["name"] == "mu-")
    lt_mu = np.log10(mu["tau_exp"])
    A, Bn, Bw, names = [], [], [], []
    for r in rec:
        A.append(lt_mu + NU * np.log10(mu["M"] / r["M"]) - np.log10(r["tau_exp"]))
        Bn.append((r["n"] - mu["n"]) / LN10)
        Bw.append((r["W"] - mu["W"]) / LN10)
        names.append(r["name"])
    A = np.array(A); X = np.column_stack([Bn, Bw])
    coef, *_ = np.linalg.lstsq(X, A, rcond=None)
    resid = A - X @ coef
    return {"lambda": float(coef[0]), "gamma": float(coef[1]),
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip(names, resid))}


def main():
    print("=" * 90)
    print("  TENSEUR D'ENLACEMENT — liens entre composantes comme canaux de dégradation")
    print("=" * 90)
    forms = ["C(n,2) liens", "Σ√(ρ_iρ_j)", "Σ√(U_iU_j)", "Σ1/√(S_iS_j)"]

    print("\n  Valeurs des contractions d'enlacement par particule :")
    print(f"  {'part.':>6} {'n':>3} " + " ".join(f"{f:>13}" for f in forms))
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        pc = pair_contractions(p["knots"])
        print(f"  {p['name']:>6} {len(p['knots']):>3} "
              + " ".join(f"{pc[f]:>13.2f}" for f in forms))

    print("\n  (A) REMPLACEMENT n -> W (1 param λ) :")
    print(f"  {'enlacement W':>16} {'λ':>8} {'RMS':>7}  neutron")
    print("  " + "-" * 46)
    for f in forms:
        r = fit_replace(f)
        flag = "  <-- bat P15(0,303)" if r["rms"] < 0.303 - 1e-3 else ""
        print(f"  {f:>16} {r['lambda']:>8.3f} {r['rms']:>7.3f}  {r['resid']['d']:>+7.3f}{flag}")

    print("\n  (B) ADDITIF  exp(-(λ n + γ W))  (2 param) :")
    print(f"  {'enlacement W':>16} {'λ':>7} {'γ':>9} {'RMS':>7}  neutron")
    print("  " + "-" * 52)
    best = None
    for f in forms:
        r = fit_additive(f)
        if best is None or r["rms"] < best[1]["rms"]:
            best = (f, r)
        flag = "  <-- bat 1/S_K(0,238)" if r["rms"] < 0.238 - 1e-3 else (
            "  <-- bat P15(0,303)" if r["rms"] < 0.303 - 1e-3 else "")
        print(f"  {f:>16} {r['lambda']:>7.3f} {r['gamma']:>+9.4f} {r['rms']:>7.3f}"
              f"  {r['resid']['d']:>+7.3f}{flag}")

    print(f"\n  Meilleur enlacement (additif) : {best[0]}  RMS {best[1]['rms']:.3f}, "
          f"neutron {best[1]['resid']['d']:+.3f}")
    print("  Repères : P15 scalaire 0,303 ; tenseur nœud 1/S_K 0,238.")
    print("=" * 90)


if __name__ == "__main__":
    main()
