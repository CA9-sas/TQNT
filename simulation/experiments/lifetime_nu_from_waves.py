"""
Dériver l'exposant de masse ν du STRESS ONDULATOIRE (le fil qui casse).

Mécanisme (L. Cariou) : le nœud est une onde stationnaire χ qui oscille à la
fréquence de Compton ω = m c²/ħ ∝ M. Le fil de flux subit une contrainte
cyclique σ ∝ M (l'inertie de l'onde charge le fil). Le fil rompt par FATIGUE :
nombre de cycles à la rupture N ∝ σ^{-p} (loi de Basquin). La durée de vie est

    τ = N × (période) = N/ω ∝ M^{-p} / M = M^{-(p+1)}    =>   ν = p + 1.

Variante "espace des phases" : Γ ∝ ω × ρ_f, ω ∝ M (Compton, exposant 1),
ρ_f ∝ M^5 (espace des phases à 3 corps, type Sargent) => ν ≈ 6.

Ce script DÉTERMINE quel ν les données préfèrent (ajustement libre de ν, à
canal fixé, ancré sur le calibrant), pour le confronter à :
  - 5π/2 ≈ 7,854 (valeur P15 actuelle, canal faible),
  - la prédiction du mécanisme (ν = p+1 ; phase space ≈ 6).

Usage :  python3 -m simulation.experiments.lifetime_nu_from_waves
"""
from __future__ import annotations

import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA

LN10 = np.log(10.0)
CHAN = {"faible": ["d", "s", "mu-", "c", "tau-", "b"],
        "fort":   ["W+", "Z0", "H", "t"]}
ANCHOR = {"faible": "mu-", "fort": "W+"}


def fit_nu_lambda(channel: str):
    """Ancre sur le calibrant ; ajuste LIBREMENT (ν, λ) :
       log10 τ = log10 τ_a + ν log10(M_a/M) - λ(n - n_a)/ln10.
    Renvoie ν, λ, RMS."""
    members = CHAN[channel]
    rec = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in members:
            continue
        rec.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "logM": np.log10(cornell_equilibrium(p["knots"], p["k"])["mass_raw"]),
            "logtau": np.log10(TAU_EXP_DATA[p["name"]]),
        })
    a = next(r for r in rec if r["name"] == ANCHOR[channel])
    # y = log10 τ - log10 τ_a ; modèle = ν·(logM_a - logM) - λ·(n - n_a)/ln10
    y, Xnu, Xlam = [], [], []
    for r in rec:
        y.append(r["logtau"] - a["logtau"])
        Xnu.append(a["logM"] - r["logM"])
        Xlam.append(-(r["n"] - a["n"]) / LN10)
    y = np.array(y); X = np.column_stack([Xnu, Xlam])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    nu, lam = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    return nu, lam, float(np.sqrt(np.mean(resid ** 2))), len(rec)


def main():
    print("=" * 88)
    print("  ν DEPUIS LE STRESS ONDULATOIRE — quel exposant de masse les données préfèrent ?")
    print("=" * 88)
    print("\n  Mécanisme : τ ∝ M^{-ν},  ν = p+1 (fatigue) ;  prédiction phase-space ν≈6.")
    print("  P15 actuel : ν_faible = 5π/2 ≈ 7,854 ; ν_fort = 7/4 = 1,75.\n")

    print(f"  {'canal':>8} {'#pts':>5} {'ν ajusté':>10} {'λ ajusté':>10} {'RMS':>7}")
    print("  " + "-" * 46)
    for ch in ["faible", "fort"]:
        nu, lam, rms, npts = fit_nu_lambda(ch)
        print(f"  {ch:>8} {npts:>5} {nu:>10.3f} {lam:>10.3f} {rms:>7.3f}")

    nu_w = fit_nu_lambda("faible")[0]
    print("\n  Confrontation (canal faible) :")
    print(f"    ν ajusté libre        = {nu_w:.3f}")
    print(f"    5π/2 (P15)            = {5*np.pi/2:.3f}   écart {abs(nu_w-5*np.pi/2):.2f}")
    print(f"    phase-space (Compton×Sargent : 1+5) = 6      écart {abs(nu_w-6):.2f}")
    print(f"    2π (fatigue+courbure ?) = {2*np.pi:.3f}   écart {abs(nu_w-2*np.pi):.2f}")
    print(f"    -> p = ν-1 (exposant de fatigue Basquin) ≈ {nu_w-1:.2f}")
    print("\n  Lecture : si ν_ajusté ≈ 6, le mécanisme Compton×phase-space le DÉRIVE ;")
    print("  l'écart à 5π/2 dit si le π de la courbure est nécessaire ou cosmétique.")

    # relation ν = λ θ
    nuw, lamw, _, _ = fit_nu_lambda("faible")
    nus, lams, _, _ = fit_nu_lambda("fort")
    print("\n  Relation ν = λ·θ (angle géométrique du mode de rupture) :")
    print(f"    faible : ν/λ = {nuw/lamw:.3f}  (π/2 = {np.pi/2:.3f})  -> reconnexion = quart de tour")
    print(f"    fort   : ν/λ = {nus/lams:.3f}  (1)               -> effilochage direct")

    # --- figure : log τ vs log M, pentes ν par canal ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        colors = {"faible": "darkred", "fort": "navy"}
        for ch in ["faible", "fort"]:
            xs, ys, labs = [], [], []
            for p in SM_ASSIGNMENTS:
                if p["name"] not in CHAN[ch]:
                    continue
                M = cornell_equilibrium(p["knots"], p["k"])["mass_raw"]
                xs.append(np.log10(M)); ys.append(np.log10(TAU_EXP_DATA[p["name"]]))
                labs.append(p["name"])
            xs = np.array(xs); ys = np.array(ys)
            ax.scatter(xs, ys, c=colors[ch], s=45, zorder=3, label=f"canal {ch}")
            for x, y, l in zip(xs, ys, labs):
                ax.annotate(l, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 3))
            nu = fit_nu_lambda(ch)[0]
            xr = np.array([xs.min(), xs.max()])
            # droite de pente -ν (à n constant ; indicative)
            b = ys.mean() + nu * xs.mean()
            ax.plot(xr, -nu * xr + b, "--", color=colors[ch], alpha=0.6,
                    label=f"pente $-\\nu_{{{ch}}}={-nu:.2f}$")
        ax.set_xlabel(r"$\log_{10} M_{\rm raw}$ (masse topologique)")
        ax.set_ylabel(r"$\log_{10}\tau_{\rm exp}$ (s)")
        ax.set_title("Durée de vie vs masse : le fil lourd casse plus vite\n"
                     r"$\nu_{\rm faible}\approx 5\pi/2$, $\nu_{\rm fort}\approx 7/4$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = "output/lifetime_nu_mass.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure -> {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")
    print("=" * 88)


if __name__ == "__main__":
    main()
