r"""
Reconstruction de la durée de vie sur l'ÉNERGIE LIBÉRÉE Q (schéma V4).

On a montré que M_raw du code était un proxy de Q (corr +0,95) et que Q est le
vrai moteur (corr -0,99). On refonde donc la loi du canal FAIBLE sur Q :

    τ = τ_P · Q^{-ν} · e^{-λ n} · exp(-γ · agg_i(1/S_i))
        \__ espace des phases (Sargent) __/  \__ facteurs topologiques __/

Q = m_parent − Σ m_produits (canal dominant), masses PDG (MeV).
agg = Σ (somme) ou max (maillon faible). On ajuste (τ_P, ν, λ, γ) par moindres
carrés et on compare somme vs maillon faible sur ~17 hadrons faibles.

NB : le canal faible suit Sargent (phase-space ∝ Q^5). Les canaux fort/EM ne
sont PAS limités par Q de la même façon (couplage fort) ; on les traite à part.

Usage :  python3 -m simulation.experiments.lifetime_Q_rebuild
"""
from __future__ import annotations

import numpy as np

from simulation.experiments.spectrum_v4_lifetimes import SPECTRUM, inv_S

# Q = énergie libérée du canal dominant (MeV). Multi-corps (D,B,τ,charm/bottom)
# = estimations (m_parent − système de produits dominant) -> incertains.
Q_RELEASE = {
    "mu-": 105.15, "tau-": 1300.0, "pi+-": 33.91, "K+-": 388.0,
    "K0_S": 218.5, "n": 0.79, "Lambda": 37.84, "Sigma+": 113.0,
    "Sigma-": 118.31, "Xi0": 64.20, "Xi-": 66.46, "Omega-": 63.0,
    "D0": 1000.0, "D+": 1000.0, "Lambdac": 1000.0, "Omegac": 1200.0,
    "B": 2300.0, "Lambdab": 2300.0,
}


def weak_rows():
    rows = []
    for name, knots, m, tau, chan in SPECTRUM:
        if chan != "faible" or not tau or name not in Q_RELEASE:
            continue
        phis = [inv_S(k) for k in knots]
        rows.append({
            "name": name, "n": float(len(knots)), "Q": Q_RELEASE[name],
            "tau": tau, "sum": float(sum(phis)), "max": float(max(phis)),
        })
    return rows


def fit(rows, terms):
    """log10 τ = c + Σ coef·col. terms: liste de (label, vecteur). RMS + résidus."""
    y = np.array([np.log10(r["tau"]) for r in rows])
    X = np.column_stack([np.ones(len(rows))] + [t[1] for t in terms])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"coef": coef, "labels": ["c"] + [t[0] for t in terms],
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "resid": dict(zip([r["name"] for r in rows], resid)),
            "npar": X.shape[1]}


def main():
    rows = weak_rows()
    lQ = np.log10([r["Q"] for r in rows])
    n = np.array([r["n"] for r in rows])
    ssum = np.array([r["sum"] for r in rows])
    smax = np.array([r["max"] for r in rows])
    lt = np.array([np.log10(r["tau"]) for r in rows])

    print("=" * 92)
    print("  RECONSTRUCTION SUR Q (énergie libérée) — canal faible, schéma V4")
    print("=" * 92)
    print(f"\n  {len(rows)} hadrons/leptons faibles. corr(log τ, log Q) = "
          f"{np.corrcoef(lQ, lt)[0,1]:+.3f}\n")

    print(f"  {'modèle':>34} {'#par':>5} {'ν(Q)':>6} {'RMS':>7}  neutron")
    print("  " + "-" * 64)
    models = [
        ("Q seul : τ∝Q^-ν", [("-logQ", -lQ)]),
        ("Q + n", [("-logQ", -lQ), ("-n", -n)]),
        ("Q + n + Σ1/S (somme)", [("-logQ", -lQ), ("-n", -n), ("-sum", -ssum)]),
        ("Q + n + max1/S (maillon faible)", [("-logQ", -lQ), ("-n", -n), ("-max", -smax)]),
    ]
    for lab, terms in models:
        f = fit(rows, terms)
        nu = f["coef"][1]   # coef de -logQ = ν
        print(f"  {lab:>34} {f['npar']:>5} {nu:>6.2f} {f['rms']:>7.3f}"
              f"  {f['resid']['n']:>+7.3f}")

    # détail du meilleur (Q+n+somme)
    f = fit(rows, [("-logQ", -lQ), ("-n", -n), ("-sum", -ssum)])
    print(f"\n  Détail (Q + n + Σ1/S) : ν={f['coef'][1]:.2f}, "
          f"λ={f['coef'][2]*np.log(10):.2f}, γ={f['coef'][3]*np.log(10):.4f}")
    print("  résidus par particule (log10 τ_calc/τ_exp) :")
    for r in sorted(rows, key=lambda r: r["Q"]):
        print(f"    {r['name']:>8} Q={r['Q']:>7.1f}  résidu={f['resid'][r['name']]:>+6.2f}")
    print("\n  Comparaison : masse totale -> RMS 2,9 déc ; Q -> voir ci-dessus.")
    print("  (charm/bottom/τ : Q multi-corps grossier, principal contributeur au RMS.)")
    print("=" * 92)


if __name__ == "__main__":
    main()
