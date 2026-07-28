"""
Durée de vie par BARRIÈRE TOPOLOGIQUE (col d'énergie) — modèle ALTERNATIF à P15.

Cadre (validé avec L. Cariou) : barrière INTERNE à la TQNT, PAS le tunnel SM.
La désintégration = franchissement d'un col d'énergie entre la topologie
instable et la topologie stable. Forme de Kramers/Arrhenius :

    τ = τ_0 · exp( ΔE‡ / E_onde )

  - ΔE‡ = énergie d'activation (topologie de transition) — plusieurs proxys ;
  - E_onde ∝ M_raw : l'énergie de l'onde qui « chauffe » le franchissement.
    => lourd = onde chaude = barrière vite franchie = vie courte (bon signe).

On compare HONNÊTEMENT, à nombre de paramètres affiché, le RMS de plusieurs
variantes de barrière à la loi de puissance P15 (τ ∝ M^{-ν} e^{-λn}, RMS 0,303
sur la famille faible). Régression log-linéaire non contrainte sur les 6
particules faibles.

Usage :  python3 -m simulation.experiments.lifetime_barrier
"""
from __future__ import annotations

import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium
from theory.topological_lifetime import TAU_EXP_DATA

WEAK = ["d", "s", "mu-", "c", "tau-", "b"]
LN10 = np.log(10.0)
NU_P15 = 5 * np.pi / 2


def data():
    rows = []
    for p in SM_ASSIGNMENTS:
        if p["name"] not in WEAK:
            continue
        eq = cornell_equilibrium(p["knots"], p["k"])
        rows.append({
            "name": p["name"],
            "n": float(len(p["knots"])),
            "M": eq["mass_raw"],
            "Mbase": eq["M_base"],
            "rho": eq["rho_total"],
            "logtau": np.log10(TAU_EXP_DATA[p["name"]]),
        })
    return rows


def fit(rows, design, labels):
    """Régression log10 τ ~ design (colonnes), intercept inclus. RMS + résidus."""
    y = np.array([r["logtau"] for r in rows])
    X = np.column_stack([np.ones(len(rows))] + design)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    npar = X.shape[1]
    res = dict(zip([r["name"] for r in rows], resid))
    return {"coef": coef, "labels": ["1"] + labels, "rms": rms,
            "npar": npar, "resid": res}


def main():
    rows = data()
    M = np.array([r["M"] for r in rows])
    logM = np.log10(M)
    n = np.array([r["n"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    Mbase = np.array([r["Mbase"] for r in rows])
    invM = 1.0 / (M * LN10)   # pour le terme exp(B/M) -> +B·invM dans log10

    print("=" * 92)
    print("  DURÉE DE VIE PAR BARRIÈRE TOPOLOGIQUE (Arrhenius) vs LOI DE PUISSANCE (P15)")
    print("=" * 92)
    print("\n  Modèle : log10 τ = cste + [termes]. Barrière -> terme +B/(M ln10).")
    print("  Comparaison à nombre de paramètres égal. Réf. P15 (ν=5π/2 fixe) :\n")

    models = []
    # --- P15 : puissance, ν fixé, λ libre (2 param : intercept, λ) ---
    y = np.array([r["logtau"] for r in rows])
    Xp15 = np.column_stack([np.ones(len(rows)), n])
    yp = y + NU_P15 * logM        # on retire la pente fixe -ν logM
    cp, *_ = np.linalg.lstsq(Xp15, yp, rcond=None)
    resid = yp - Xp15 @ cp
    rms_p15 = float(np.sqrt(np.mean(resid ** 2)))
    res_p15 = dict(zip([r["name"] for r in rows], resid))
    models.append(("P15 puissance (ν=5π/2 fixe, λ)", rms_p15, 2, res_p15["d"]))

    # --- Barrières (Arrhenius) ---
    # B1 : barrière constante, freq d'essai fixe :  log τ = c + B·invM           (2 par)
    m = fit(rows, [invM], ["B/M"]); models.append(("Arrhenius B const", m["rms"], m["npar"], m["resid"]["d"]))
    # B2 : + fréquence d'essai ∝ M :  log τ = c - a logM + B·invM                 (3 par)
    m = fit(rows, [logM, invM], ["logM", "B/M"]); models.append(("Arrhenius + M^-a", m["rms"], m["npar"], m["resid"]["d"]))
    # B3 : barrière ∝ ρ_total :  log τ = c + B·(ρ/M)/ln10                          (2 par)
    m = fit(rows, [rho * invM], ["ρ/M"]); models.append(("Arrhenius B∝ρ_tot", m["rms"], m["npar"], m["resid"]["d"]))
    # B4 : barrière ∝ M_base :  log τ = c + B·(Mbase/M)/ln10                        (2 par)
    m = fit(rows, [Mbase * invM], ["Mbase/M"]); models.append(("Arrhenius B∝M_base", m["rms"], m["npar"], m["resid"]["d"]))
    # B5 : Arrhenius ρ/M + n (canaux) :                                            (3 par)
    m = fit(rows, [rho * invM, n], ["ρ/M", "n"]); models.append(("Arrhenius B∝ρ + n", m["rms"], m["npar"], m["resid"]["d"]))

    print(f"  {'modèle':>34} {'#par':>5} {'RMS':>7}  neutron(d)")
    print("  " + "-" * 60)
    for name, rms, npar, rd in models:
        flag = "  <-- bat P15" if (rms < rms_p15 - 1e-3 and not name.startswith("P15")) else ""
        print(f"  {name:>34} {npar:>5} {rms:>7.3f}  {rd:>+7.3f}{flag}")

    print("\n  Lecture :")
    print("  - Si une barrière atteint RMS ≲ 0,30 à #par égal -> alternative viable.")
    print("  - Forme Arrhenius (exp(B/M)) vs puissance (M^-ν) : laquelle colle aux données ?")
    print("=" * 92)


if __name__ == "__main__":
    main()
