r"""
Tableau récapitulatif : TOUTES les particules — valeurs connues vs prévision TQNT.

Masse :  m_TQNT = η · M_raw   (formule de masse topologique)
Durée de vie (loi à DEUX FACTEURS de la session) :
    τ = τ_P · M_raw^{-ν} · e^{-λ n} · exp(-γ Σ_i U_i/ρ_{J,i}²)
        \________ P15 (dégradation) ______/  \__ probabilité de stabilité __/
  - canal faible : ν = 5π/2 ;  canal fort : ν = 7/4 ;
  - (λ, γ, τ_P) calibrés par canal (ν fixe).

Sort un tableau console + le bloc LaTeX (booktabs) prêt à coller dans le brouillon.

Usage :  python3 -m simulation.experiments.lifetime_full_table
"""
from __future__ import annotations

import numpy as np

from theory.abraham import geometric_integrals
from theory.retraction import retraction_factor
from theory.jones import jones_polynomial
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium, ETA
from theory.topological_lifetime import TAU_EXP_DATA

STRONG = {"W+", "Z0", "H", "t"}
NU = {"faible": 5 * np.pi / 2, "fort": 7.0 / 4.0}
LN10 = np.log(10.0)
# noms d'affichage
DISP = {"e-": "e⁻", "u": "u", "d": "d (→n,β)", "nu_tau": "ν_τ", "s": "s",
        "mu-": "μ⁻", "c": "c", "tau-": "τ⁻", "b": "b", "W+": "W±",
        "Z0": "Z⁰", "H": "H⁰", "t": "t"}


def invS_sum(knots):
    s = 0.0
    for kn in knots:
        gi = geometric_integrals(kn, 10000)
        rho = retraction_factor(jones_polynomial(kn))
        s += gi["U"] / rho ** 2
    return s


def build():
    rows = []
    for p in SM_ASSIGNMENTS:
        eq = cornell_equilibrium(p["knots"], p["k"])
        rows.append({
            "name": p["name"],
            "chan": "fort" if p["name"] in STRONG else "faible",
            "n": len(p["knots"]),
            "M": eq["mass_raw"],
            "invS": invS_sum(p["knots"]),
            "m_tqnt": ETA * eq["mass_raw"],
            "m_exp": p["mass_sm"],
            "tau_exp": TAU_EXP_DATA.get(p["name"]),
        })
    # fit deux-facteurs par canal (sur les membres avec τ connu)
    for chan in ("faible", "fort"):
        memb = [r for r in rows if r["chan"] == chan and r["tau_exp"]]
        nu = NU[chan]
        y = np.array([np.log10(r["tau_exp"]) + nu * np.log10(r["M"]) for r in memb])
        X = np.column_stack([np.ones(len(memb)),
                             [r["n"] for r in memb],
                             [r["invS"] for r in memb]])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        for r in rows:
            if r["chan"] != chan:
                continue
            logtau = (coef[0] + coef[1] * r["n"] + coef[2] * r["invS"]
                      - nu * np.log10(r["M"]))
            r["tau_tqnt"] = 10.0 ** logtau
            r["dlog"] = (logtau - np.log10(r["tau_exp"])) if r["tau_exp"] else None
    return rows


def fmt_tau(t):
    if t is None:
        return "stable"
    return f"{t:.2e}"


def main():
    rows = build()
    order = ["e-", "u", "nu_tau", "d", "s", "mu-", "c", "tau-", "b",
             "W+", "Z0", "H", "t"]
    rows.sort(key=lambda r: order.index(r["name"]))

    print("=" * 110)
    print("  TOUTES LES PARTICULES — valeurs connues vs prévision TQNT (masse + durée de vie)")
    print("=" * 110)
    print(f"\n  {'part.':>9} {'canal':>7} {'n':>2} {'m_TQNT':>9} {'m_exp':>9} "
          f"{'τ_TQNT(s)':>11} {'τ_exp(s)':>11} {'Δlog τ':>7}")
    print("  " + "-" * 78)
    for r in rows:
        m1 = f"{r['m_tqnt']:.1f}" if r['m_tqnt'] < 1e4 else f"{r['m_tqnt']/1000:.1f}G"
        m2 = f"{r['m_exp']:.1f}" if r['m_exp'] < 1e4 else f"{r['m_exp']/1000:.1f}G"
        dl = f"{r['dlog']:+.2f}" if r.get("dlog") is not None else "—"
        te = fmt_tau(r["tau_exp"])
        tt = fmt_tau(r.get("tau_tqnt")) if r["tau_exp"] else "stable"
        print(f"  {DISP[r['name']]:>9} {r['chan']:>7} {r['n']:>2} {m1:>9} {m2:>9} "
              f"{tt:>11} {te:>11} {dl:>7}")

    # RMS
    for chan in ("faible", "fort"):
        d = [abs(r["dlog"]) for r in rows if r["chan"] == chan and r.get("dlog") is not None]
        print(f"\n  RMS Δlog τ ({chan}) : {np.sqrt(np.mean(np.square(d))):.3f} déc  ({len(d)} part.)")
    alld = [abs(r["dlog"]) for r in rows if r.get("dlog") is not None]
    print(f"  RMS Δlog τ (global) : {np.sqrt(np.mean(np.square(alld))):.3f} déc  ({len(alld)} part.)")

    # --- bloc LaTeX ---
    print("\n" + "=" * 110)
    print("  BLOC LATEX (à coller) :")
    print("=" * 110)
    tex = []
    tex.append(r"\begin{tabular}{lcc rr rr c}")
    tex.append(r"\toprule")
    tex.append(r"Particule & canal & $n$ & $m_{\mathrm{TQNT}}$ & $m_{\mathrm{exp}}$ "
               r"& $\tau_{\mathrm{TQNT}}$ & $\tau_{\mathrm{exp}}$ & $\Delta\log_{10}\tau$ \\")
    tex.append(r"\midrule")
    disp_tex = {"e-": r"$e^-$", "u": r"$u$", "nu_tau": r"$\nu_\tau$",
                "d": r"$d$ ($n,\beta$)", "s": r"$s$", "mu-": r"$\mu^-$",
                "c": r"$c$", "tau-": r"$\tau^-$", "b": r"$b$", "W+": r"$W^\pm$",
                "Z0": r"$Z^0$", "H": r"$H^0$", "t": r"$t$"}
    def texnum(x):
        if x is None:
            return "stable"
        e = int(np.floor(np.log10(x)))
        m = x / 10 ** e
        return f"${m:.2f}\\!\\times\\!10^{{{e}}}$"
    for r in rows:
        m1 = f"{r['m_tqnt']:.1f}" if r['m_tqnt'] < 1e4 else f"{r['m_tqnt']/1000:.2f}$\\,$G"
        m2 = f"{r['m_exp']:.1f}" if r['m_exp'] < 1e4 else f"{r['m_exp']/1000:.2f}$\\,$G"
        dl = f"${r['dlog']:+.2f}$" if r.get("dlog") is not None else "---"
        te = texnum(r["tau_exp"])
        tt = texnum(r.get("tau_tqnt")) if r["tau_exp"] else "stable"
        tex.append(f"{disp_tex[r['name']]} & {r['chan']} & {r['n']} & {m1} & {m2} "
                   f"& {tt} & {te} & {dl} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
