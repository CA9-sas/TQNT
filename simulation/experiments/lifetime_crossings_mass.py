"""
Durée de vie = rupture du FIL aux points de CROISEMENT sous mouvement ondulatoire.

Image (L. Cariou) : un entrelacs de nœuds est un fil de fer. L'onde χ le fait
osciller ; le fil casse aux POINTS DE CROISEMENT (concentrateurs de contrainte).
Plus l'entrelacs est LOURD, plus l'oscillation charge le fil -> il casse plus
vite -> durée de vie plus courte.

Conséquence testable : la durée de vie dépend de
  - la MASSE M (inertie chargée par l'oscillation),
  - le nombre de CROISEMENTS c_tot (nombre de points de rupture),
plutôt que du seul nombre de nœuds n (P15 actuel).

c_tot = somme des croisements des composantes (nom "X_Y" -> X croisements).
Ex. nucléon 3_1+7_2+7_2 = 3+7+7 = 17 croisements.

On explore, sur tout le spectre SM, comment log τ_exp dépend de (log M, c_tot, n),
puis on teste des formes physiques contre la baseline P15.

Usage :  python3 -m simulation.experiments.lifetime_crossings_mass
"""
from __future__ import annotations

import numpy as np

from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium, ETA
from theory.topological_lifetime import TAU_EXP_DATA

STRONG = {"W+", "Z0", "H", "t"}


def crossings(name: str) -> int:
    """Nombre de croisements d'un nœud à partir de son nom 'X_Y'."""
    return int(name.split("_")[0])


def records():
    rows = []
    for p in SM_ASSIGNMENTS:
        te = TAU_EXP_DATA.get(p["name"])
        if te is None:
            continue
        eq = cornell_equilibrium(p["knots"], p["k"])
        c_tot = sum(crossings(kn) for kn in p["knots"])
        rows.append({
            "name": p["name"],
            "n": len(p["knots"]),
            "c_tot": c_tot,
            "M_raw": eq["mass_raw"],
            "m_mev": ETA * eq["mass_raw"],
            "logtau": np.log10(te),
            "channel": "fort" if p["name"] in STRONG else "faible",
        })
    return rows


def lstsq_model(rows, cols):
    """Régression linéaire log10 τ ~ [1] + cols ; renvoie coeffs et RMS."""
    y = np.array([r["logtau"] for r in rows])
    X = np.column_stack([np.ones(len(rows))] + [np.array([r[c] for r in rows]) for c in cols])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt(np.mean(resid ** 2)))


def main():
    rows = records()
    print("=" * 96)
    print("  DURÉE DE VIE vs MASSE et CROISEMENTS — le fil casse aux croisements (mouvement ondulatoire)")
    print("=" * 96)

    print(f"\n  {'part.':>6} {'canal':>7} {'n':>3} {'c_tot':>6} {'M_raw':>10} "
          f"{'m(MeV)':>11} {'log10 τ':>9}")
    print("  " + "-" * 60)
    for r in sorted(rows, key=lambda r: r["m_mev"]):
        print(f"  {r['name']:>6} {r['channel']:>7} {r['n']:>3} {r['c_tot']:>6} "
              f"{r['M_raw']:>10.1f} {r['m_mev']:>11.2f} {r['logtau']:>+9.2f}")

    # corrélations brutes (tout le spectre)
    y = np.array([r["logtau"] for r in rows])
    print("\n  Corrélations (Pearson) de log10 τ avec :")
    for lab, x in [("log10 M", np.log10([r["M_raw"] for r in rows])),
                   ("c_tot (croisements)", np.array([r["c_tot"] for r in rows], float)),
                   ("n (nœuds)", np.array([r["n"] for r in rows], float)),
                   ("c_tot/M", np.array([r["c_tot"]/r["M_raw"] for r in rows]))]:
        print(f"    {lab:>22} : r = {np.corrcoef(x, y)[0,1]:+.3f}")

    # modèles (régression libre sur tout le spectre, descriptif)
    print("\n  Régressions log10 τ = a0 + Σ a_i x_i  (tout le spectre, 13 pts) :")
    for cols in [["n", "M_raw"], ["c_tot", "M_raw"], ["n"], ["c_tot"],
                 ["c_tot", "n", "M_raw"]]:
        # M_raw entre en log
        rr = [{**r, "M_raw": np.log10(r["M_raw"])} for r in rows]
        coef, rms = lstsq_model(rr, cols)
        terms = " + ".join(f"{coef[i+1]:+.3f}·{c}" for i, c in enumerate(cols))
        lab = "[" + ", ".join("logM" if c == "M_raw" else c for c in cols) + "]"
        print(f"    {lab:>22} : {coef[0]:+.2f} {terms:>40}   RMS={rms:.3f}")

    print("\n  (Comparaison : ces régressions LIBRES n'ont pas la contrainte de")
    print("   calibration de P15 ; elles servent à voir quelles variables comptent.)")

    # ---- Test calibré à CANAL FIXÉ (famille faible) : croisements vs nœuds ----
    print("\n" + "=" * 96)
    print("  TEST CALIBRÉ — canal faible, ν=5π/2, ancré sur le muon : multiplicité de rupture")
    print("=" * 96)
    NU = 5 * np.pi / 2
    LN10 = np.log(10.0)
    weak = [r for r in rows if r["channel"] == "faible"]
    mu = next(r for r in weak if r["name"] == "mu-")
    lt_mu = mu["logtau"]

    def calib_fit(varname, mode):
        """mode 'exp' : exp(-λ X)/M^ν ; mode 'pow' : /(M^ν X^μ). 1 param, ancré muon."""
        A, B, names = [], [], []
        for r in weak:
            A.append(lt_mu + NU * np.log10(mu["M_raw"] / r["M_raw"]) - r["logtau"])
            if mode == "exp":
                B.append((r[varname] - mu[varname]) / LN10)
            else:  # pow : -μ (log X - log X_mu)
                B.append(np.log10(r[varname] / mu[varname]))
            names.append(r["name"])
        A = np.array(A); B = np.array(B)
        coef = float(np.sum(A * B) / np.sum(B * B)) if np.sum(B * B) else 0.0
        resid = A - coef * B
        return coef, float(np.sqrt(np.mean(resid ** 2))), dict(zip(names, resid))

    print(f"\n  {'multiplicité':>26} {'coef':>8} {'RMS':>7}  neutron(d)")
    print("  " + "-" * 56)
    tests = [
        ("nœuds n      (P15, exp)", "n", "exp"),
        ("croisements c_tot (exp)", "c_tot", "exp"),
        ("nœuds n      (puissance)", "n", "pow"),
        ("croisements c_tot (puiss.)", "c_tot", "pow"),
    ]
    for lab, var, mode in tests:
        c, rms, res = calib_fit(var, mode)
        print(f"  {lab:>26} {c:>8.3f} {rms:>7.3f}  {res['d']:>+7.3f}")
    print("\n  (coef = λ pour 'exp', = μ pour 'puissance'. Réf. P15 nœuds exp : RMS 0,303.)")
    print("=" * 96)


if __name__ == "__main__":
    main()
