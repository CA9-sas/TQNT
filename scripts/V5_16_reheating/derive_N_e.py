#!/usr/bin/env python3
"""
V5.16 — Derive N_e from the V5 chi-field reheating dynamics.

Produces:
    output/v5p16_reheating_N_e.png

Compares:
    V5.13/V5.14 convention : N_e = 60 (Planck pivot, external)
    V5.16 derivation       : N_e from Liddle-Leach + V5 chi-field decay
                             with topological suppression alpha_EM^{c_n_eff}.

Honest verdict (printed at the end):
    - With c_n_eff = 18 (V5 nucleon) and w_reh = 0 (Starobinsky matter-shape):
        T_reh ~ 0.15 MeV   < BBN bound 1 MeV  -> TENSION
        N_e   ~ 35.8       -> n_s ~ 0.944, off Planck by ~5 sigma
    - With c_n_eff = 18 and w_reh = 1/3 (instantaneous radiation-like):
        N_e   ~ 50.6       -> n_s ~ 0.960, compatible Planck at 1.2 sigma
    - For c_n_eff <= 12 with w_reh = 0:
        T_reh > BBN bound; N_e ~ 40.7, n_s ~ 0.951.

Launch :
    PYTHONPATH=. python3 scripts/V5_16_reheating/derive_N_e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.coupling_kg import ALPHA_EM, C_N_NUCLEON_EFF
from theory.inflation_chi_field import (
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    R_PLANCK_BICEP,
)
from theory.reheating_chi import (
    G_STAR_SM,
    T_REH_BBN_MIN_GEV,
    H_end_V5p14,
    N_e_V5,
    T_reh_V5,
    n_s_starobinsky_at_Ne,
    predict_reheating_V5,
    r_starobinsky_at_Ne,
    validate_against_Planck,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("V5.16 - Reheating chi-field derivation of N_e")
    print("=" * 72)

    # ---- Main V5.16 prediction (c_n=18, w_reh=0)
    pred_w0 = predict_reheating_V5(c_n_eff=18, w_reh=0.0)
    pred_w13 = predict_reheating_V5(c_n_eff=18, w_reh=1.0 / 3.0)

    print()
    print("DEFAULT V5.14/V5.16 inputs:")
    print(f"  alpha_EM            = {ALPHA_EM:.7e}")
    print(f"  c_n_eff (nucleon)   = {C_N_NUCLEON_EFF}")
    print(f"  alpha_EM^c_n        = {ALPHA_EM**C_N_NUCLEON_EFF:.3e}")
    print(f"  H_end (V5.14)       = {pred_w0.H_end_GeV:.3e} GeV")
    print(f"  Gamma_reh           = {pred_w0.Gamma_reh_GeV:.3e} GeV")

    print()
    print("V5.16 prediction (w_reh = 0, Starobinsky matter-shape):")
    print(f"  T_reh               = {pred_w0.T_reh_GeV:.3e} GeV "
          f"(BBN bound: {T_REH_BBN_MIN_GEV:.0e} GeV)")
    print(f"  N_e^V5              = {pred_w0.N_e_V5:.3f}")
    print(f"  n_s @ N_e^V5        = {pred_w0.n_s_at_Ne_V5:.4f} "
          f"(Planck: {N_S_PLANCK:.4f} +- {N_S_PLANCK_SIGMA:.4f}, "
          f"deviation = {pred_w0.n_s_deviation_sigma:+.2f} sigma)")
    print(f"  r   @ N_e^V5        = {pred_w0.r_at_Ne_V5:.3e} "
          f"(BICEP upper: {R_PLANCK_BICEP:.2f})")
    print(f"  BBN compatible      = {pred_w0.BBN_compatible}")
    print(f"  Planck compatible   = {pred_w0.Planck_compatible} (3 sigma)")
    print(f"  BICEP compatible    = {pred_w0.BICEP_compatible}")

    print()
    print("V5.16 prediction (w_reh = 1/3, instantaneous radiation):")
    print(f"  N_e^V5              = {pred_w13.N_e_V5:.3f}")
    print(f"  n_s @ N_e^V5        = {pred_w13.n_s_at_Ne_V5:.4f} "
          f"(deviation = {pred_w13.n_s_deviation_sigma:+.2f} sigma)")
    print(f"  r   @ N_e^V5        = {pred_w13.r_at_Ne_V5:.3e}")
    print(f"  Planck compatible   = {pred_w13.Planck_compatible} (3 sigma)")

    # ---- Scan over c_n_eff
    cn_grid = np.arange(0, 21)
    N_e_w0 = np.array([N_e_V5(c_n_eff=int(c), w_reh=0.0) for c in cn_grid])
    N_e_w13 = np.array([N_e_V5(c_n_eff=int(c), w_reh=1.0 / 3.0) for c in cn_grid])
    T_reh_GeV = np.array([T_reh_V5(c_n_eff=int(c)) for c in cn_grid])
    n_s_w0 = np.array([n_s_starobinsky_at_Ne(n) for n in N_e_w0])
    n_s_w13 = np.array([n_s_starobinsky_at_Ne(n) for n in N_e_w13])
    r_w0 = np.array([r_starobinsky_at_Ne(n) for n in N_e_w0])

    # ---- Figure
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))

    # (a) T_reh vs c_n_eff with BBN bound
    ax = axes[0, 0]
    ax.semilogy(cn_grid, T_reh_GeV, "-o", color="C0", label="T_reh V5")
    ax.axhline(T_REH_BBN_MIN_GEV, ls="--", color="red",
               label=f"BBN bound 1 MeV")
    ax.axvline(C_N_NUCLEON_EFF, ls=":", color="black",
               label=f"V5 nucleon c_n={C_N_NUCLEON_EFF}")
    ax.set_xlabel("c_n^eff (effective active crossings)")
    ax.set_ylabel("T_reh [GeV]")
    ax.set_title("(a) Reheating temperature vs topological suppression")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # (b) N_e vs c_n_eff for both EoS
    ax = axes[0, 1]
    ax.plot(cn_grid, N_e_w0, "-o", color="C0", label="V5.16  w_reh=0 (matter)")
    ax.plot(cn_grid, N_e_w13, "-s", color="C2",
            label="V5.16  w_reh=1/3 (radiation)")
    ax.axhline(N_EFOLDS_PIVOT, ls="--", color="black",
               label="V5.13/V5.14 pivot N=60")
    ax.axhspan(50, 65, alpha=0.15, color="gray",
               label="Planck-favored 50-65")
    ax.axvline(C_N_NUCLEON_EFF, ls=":", color="black")
    ax.set_xlabel("c_n^eff")
    ax.set_ylabel("N_e (derived)")
    ax.set_title("(b) e-folds N_e derived from V5 reheating")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # (c) n_s recomputed at N_e_V5
    ax = axes[1, 0]
    ax.plot(cn_grid, n_s_w0, "-o", color="C0", label="V5.16 w_reh=0")
    ax.plot(cn_grid, n_s_w13, "-s", color="C2", label="V5.16 w_reh=1/3")
    ax.axhline(N_S_PLANCK, color="red", label=f"Planck {N_S_PLANCK:.4f}")
    ax.axhspan(N_S_PLANCK - N_S_PLANCK_SIGMA, N_S_PLANCK + N_S_PLANCK_SIGMA,
               color="red", alpha=0.15, label="Planck +- 1 sigma")
    ax.axvline(C_N_NUCLEON_EFF, ls=":", color="black")
    ax.set_xlabel("c_n^eff")
    ax.set_ylabel("n_s")
    ax.set_title("(c) Scalar spectral tilt at N_e^V5")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # (d) Honest summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["V5 input", "value"],
        ["alpha_EM", f"{ALPHA_EM:.5e}"],
        ["c_n_eff (nucleon)", f"{C_N_NUCLEON_EFF}"],
        ["alpha_EM^c_n", f"{ALPHA_EM**C_N_NUCLEON_EFF:.2e}"],
        ["H_end [GeV]", f"{pred_w0.H_end_GeV:.2e}"],
        ["Gamma_reh [GeV]", f"{pred_w0.Gamma_reh_GeV:.2e}"],
        ["T_reh (w=0) [GeV]", f"{pred_w0.T_reh_GeV:.2e}"],
        ["T_reh (w=1/3) [GeV]", f"{pred_w13.T_reh_GeV:.2e}"],
        ["", ""],
        ["N_e (w=0)", f"{pred_w0.N_e_V5:.2f}"],
        ["N_e (w=1/3)", f"{pred_w13.N_e_V5:.2f}"],
        ["n_s (w=0)", f"{pred_w0.n_s_at_Ne_V5:.4f} "
                     f"(Dn={pred_w0.n_s_deviation_sigma:+.1f}s)"],
        ["n_s (w=1/3)", f"{pred_w13.n_s_at_Ne_V5:.4f} "
                       f"(Dn={pred_w13.n_s_deviation_sigma:+.1f}s)"],
        ["r (w=0)", f"{pred_w0.r_at_Ne_V5:.2e}"],
        ["r (w=1/3)", f"{pred_w13.r_at_Ne_V5:.2e}"],
        ["", ""],
        ["BBN comp. (w=0)", "NO" if not pred_w0.BBN_compatible else "YES"],
        ["Planck comp. (w=0)",
         "NO" if not pred_w0.Planck_compatible else "YES"],
        ["BICEP comp. (w=0)",
         "YES" if pred_w0.BICEP_compatible else "NO"],
    ]
    table = ax.table(cellText=rows, loc="center",
                     cellLoc="left", colWidths=[0.55, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    ax.set_title("(d) V5.16 reheating summary (default c_n=18)", pad=10)

    fig.suptitle("V5.16 — chi-field reheating and N_e derivation\n"
                 "(honest verdict: default c_n=18 yields T_reh < BBN; "
                 "the topological suppression is too strong)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = OUTPUT_DIR / "v5p16_reheating_N_e.png"
    fig.savefig(out, dpi=150)
    print()
    print(f"Figure saved to {out}")
    print()
    print("=" * 72)
    print("HONEST VERDICT (V5.16):")
    print("=" * 72)
    print("  V5 chi-field decay rate Gamma = alpha_EM^c_n * H_end is")
    print("  the SAME topological suppression V5 uses for G_F and kappa_g")
    print("  (consistent across V5).")
    print()
    print("  With c_n=18 (V5 nucleon) and the Starobinsky matter-shape")
    print("  (w_reh=0), the predicted T_reh ~ 0.15 MeV is BELOW the BBN")
    print("  bound (1 MeV). The chi-field decay is too suppressed.")
    print()
    print("  Two ways out:")
    print("   (i) instant-radiation EoS w=1/3 -> N_e~50.6, n_s~0.960")
    print("       (compatible Planck @ 1.2 sigma, BICEP OK).")
    print("  (ii) lower effective c_n_eff during reheating (the active")
    print("       crossings of a *photon* knot 0_1 are zero, etc).")
    print()
    print("  Neither is derived rigorously in V5.16; the proper resolution")
    print("  requires the chi -> primordial Standard-Model vertex from")
    print("  Chern-Simons (left as V5.17).")
    print("=" * 72)


if __name__ == "__main__":
    main()
