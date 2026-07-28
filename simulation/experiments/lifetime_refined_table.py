r"""
Régénère le tableau des 18 durées de vie faibles à partir de la FORMULE
RAFFINÉE (helicité 2-corps + N_ch + (1+3g_A²) + CKM mesurés), classes A/B/C de
lifetime_unified_refined. Sort τ_TQNT prédit par particule + le bloc à coller.

Usage :  python3 -m simulation.experiments.lifetime_refined_table
"""
from __future__ import annotations

import numpy as np

from simulation.experiments.lifetime_unified_refined import (
    A, B, C, M_MU)

DISP = {"n": "n", "mu-": r"\mu^-", "tau-": r"\tau^-", "D0": "D^0",
        "D+": "D^+", "Lambdac": r"\Lambda_c", "Omegac": r"\Omega_c",
        "B": "B", "Lambdab": r"\Lambda_b", "pi+-": r"\pi^\pm", "K+-": "K^\\pm",
        "Lambda": r"\Lambda", "Sigma+": r"\Sigma^+", "Sigma-": r"\Sigma^-",
        "Xi0": r"\Xi^0", "Xi-": r"\Xi^-", "Omega-": r"\Omega^-", "K0_S": "K^0_S"}
CLS = {}  # name -> classe label


def fitA():
    lt = np.array([np.log10(r[4]) for r in A])
    lth = np.array([np.log10(r[2]) for r in A])
    lw = np.array([np.log10(r[3]) for r in A])
    lQ = np.array([np.log10(r[1]) for r in A])
    X = np.column_stack([np.ones(len(A)), -lQ])
    y = lt + lth + lw
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    ltpred = (X @ c) - lth - lw           # log10 τ prédit
    return {r[0]: (r[1], r[4], 10**lp) for r, lp in zip(A, ltpred)}  # name->(Q,τexp,τpred)


def fitB():
    def rate(f, mP, V2):
        return V2 * f**2 * M_MU**2 * mP * (1 - M_MU**2/mP**2)**2
    lt = np.array([np.log10(r[4]) for r in B])
    base = np.array([-np.log10(rate(r[1], r[2], r[3])) for r in B])
    cB = np.mean(lt - base)
    ltpred = cB + base
    # Q "affiché" = m_P - m_mu (énergie 2-corps)
    return {r[0]: (r[2]-M_MU, r[4], 10**lp) for r, lp in zip(B, ltpred)}


def fitC():
    lt = np.array([np.log10(r[2]) for r in C])
    base = np.array([-np.log10(r[1]) for r in C])
    cC = np.mean(lt - base)
    ltpred = cC + base
    return {r[0]: (None, r[2], 10**lp) for r, lp in zip(C, ltpred)}


def texnum(x):
    e = int(np.floor(np.log10(x))); m = x / 10**e
    return f"${m:.1f}\\!\\times\\!10^{{{e}}}$"


def main():
    res = {}
    for nm, v in fitA().items(): res[nm] = (v, "A")
    for nm, v in fitB().items(): res[nm] = (v, "B")
    for nm, v in fitC().items(): res[nm] = (v, "C")
    CLAB = {"A": r"$\beta$/semi-lept.", "B": "2-corps", "C": "non-lept."}

    order = ["n", "mu-", "tau-", "pi+-", "K+-", "Lambda", "Sigma+", "Sigma-",
             "Xi0", "Xi-", "Omega-", "K0_S", "D0", "D+", "Lambdac", "Omegac",
             "B", "Lambdab"]
    print("=" * 76)
    print("  TABLEAU RAFFINÉ — τ_TQNT (formule helicité+N_ch+g_A+CKM) vs τ_exp")
    print("=" * 76)
    print(f"  {'part.':>9} {'cl':>3} {'Q(MeV)':>8} {'τ_TQNT':>11} {'τ_exp':>11} {'Δlog':>6}")
    resid = []
    for nm in order:
        (Q, te, tp), cl = res[nm]
        dl = np.log10(tp/te); resid.append(dl)
        qs = f"{Q:>8.1f}" if Q else "    --- "
        print(f"  {nm:>9} {cl:>3} {qs} {tp:>11.2e} {te:>11.2e} {dl:>+6.2f}")
    print(f"\n  RMS global = {np.sqrt(np.mean(np.square(resid))):.3f} déc")

    print("\n  --- LIGNES LATEX (Particule & sous-canal & Q & τ_TQNT & τ_exp) ---")
    for nm in order:
        (Q, te, tp), cl = res[nm]
        qs = f"${Q:.1f}$" if Q else "---"
        print(f"${DISP[nm]}$ & {CLAB[cl]} & {qs} & {texnum(tp)} & {texnum(te)} \\\\")


if __name__ == "__main__":
    main()
