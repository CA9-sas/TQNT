r"""
Tableau unifié des durées de vie faibles par TAXONOMIE (schéma V4).

Le canal faible n'est pas une loi unique mais une taxonomie selon la topologie
de l'état final et le changement de génération :
  beta3  (n, μ, τ)            : 3 ondes libres -> Sargent τ ∝ Q^-ν (ν≈4,5)
  lep2   (π±, K± → μν)        : 2 ondes libres, hélicité -> ~Q-plat
  nonlep (Λ,Σ,Ξ,Ω,K0_S)       : produits noués -> reconnexion, Q-indépendant
  charm  (D,Λc,Ωc ; c→s)      : Sargent au quark, Q=m_c
  bottom (B,Λb ; b→c)         : Sargent + suppression CKM

On ajuste τ ∝ Q^-ν dans chaque sous-canal (calibration interne) et on tabule
τ_TQNT vs τ_exp. RMS par sous-canal + sortie LaTeX.

Usage :  python3 -m simulation.experiments.lifetime_taxonomy_table
"""
from __future__ import annotations

import numpy as np

from simulation.experiments.lifetime_Q_refined import DATA

TYPES = ["beta3", "lep2", "nonlep", "charm", "bottom"]
TLAB = {"beta3": "β 3-corps (ondes libres)", "lep2": "2-corps lept. (π,K→μν)",
        "nonlep": "non-lept. (noué)", "charm": "charm (c→s)",
        "bottom": "bottom (b→c, CKM)"}
DISP = {"n": "n", "mu-": "μ⁻", "tau-": "τ⁻", "pi+-": "π±", "K+-": "K±",
        "D0": "D⁰", "D+": "D±", "Lambdac": "Λc", "Omegac": "Ωc",
        "B": "B", "Lambdab": "Λb", "Lambda": "Λ", "Sigma+": "Σ⁺",
        "Sigma-": "Σ⁻", "Xi0": "Ξ⁰", "Xi-": "Ξ⁻", "Omega-": "Ω⁻", "K0_S": "K⁰_S"}


def predict(items):
    """τ_pred par sous-canal via ajustement τ=10^(c)·Q^-ν (intercept+ν)."""
    lQ = np.log10([q for _, q, _, _ in items])
    lt = np.log10([t for _, _, t, _ in items])
    if len(items) >= 2 and np.ptp(lQ) > 1e-6:
        A = np.column_stack([np.ones(len(items)), -lQ])
        c, *_ = np.linalg.lstsq(A, lt, rcond=None)
        pred = A @ c
        nu = c[1]
    else:  # Q constant (charm/bottom) : prédiction = moyenne log
        pred = np.full(len(items), lt.mean())
        nu = float("nan")
    return pred, lt, nu


def main():
    print("=" * 84)
    print("  TABLEAU UNIFIÉ — durées de vie faibles par taxonomie (schéma V4)")
    print("=" * 84)
    print(f"\n  {'part.':>7} {'type':>26} {'Q(MeV)':>8} {'τ_TQNT':>10} "
          f"{'τ_exp':>10} {'Δlog':>6}")
    print("  " + "-" * 74)
    all_res = []
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Particule & sous-canal & $Q$ (MeV) & $\tau_{\mathrm{TQNT}}$ & "
           r"$\tau_{\mathrm{exp}}$ & $\Delta\log_{10}$ \\", r"\midrule"]
    for ty in TYPES:
        items = [d for d in DATA if d[3] == ty]
        pred, lt, nu = predict(items)
        for (nm, q, t, _), lp, le in zip(items, pred, lt):
            dl = lp - le
            all_res.append(dl)
            tt = 10 ** lp
            print(f"  {DISP[nm]:>7} {TLAB[ty]:>26} {q:>8.1f} {tt:>10.2e} "
                  f"{t:>10.2e} {dl:>+6.2f}")
            e = int(np.floor(lp)); m = 10 ** (lp - e)
            ee = int(np.floor(le)); mm = 10 ** (le - ee)
            tex.append(f"{DISP[nm]} & {TLAB[ty]} & {q:.1f} & "
                       f"${m:.1f}\\!\\times\\!10^{{{e}}}$ & "
                       f"${mm:.1f}\\!\\times\\!10^{{{ee}}}$ & ${dl:+.2f}$ \\\\")
        d = [lp - le for lp, le in zip(pred, lt)]
        rms = np.sqrt(np.mean(np.square(d)))
        nutxt = f"ν={nu:.2f}" if not np.isnan(nu) else "Q const"
        print(f"  {'':>7} {'  -> '+nutxt+f', RMS={rms:.3f} déc':>50}")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")

    glob = np.sqrt(np.mean(np.square(all_res)))
    print(f"\n  RMS global (taxonomie, {len(all_res)} particules faibles) : {glob:.3f} déc")
    print("  (vs masse totale 2,9 ; Q sans taxonomie 1,8)")
    print("\n" + "=" * 84 + "\n  BLOC LATEX :\n" + "=" * 84)
    print("\n".join(tex))


if __name__ == "__main__":
    main()
