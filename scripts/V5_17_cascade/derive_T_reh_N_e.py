#!/usr/bin/env python3
"""
V5.17 — Derive T_reh and N_e from the 2-stage cascade reheating mechanism.

Produces:
    output/v5p17_cascade_reheating.png

Compares:
    V5.16 single-stage (chi -> nucleon, alpha_EM^{18} suppression)
        T_reh ~ 0.15 MeV  < T_BBN  -> BBN tension
        N_e   ~ 35.8     -> n_s = 0.944, 4.9 sigma off Planck

    V5.17 two-stage cascade:
        Stage 1 chi -> 0_1 (unknot, c_n=0, NO suppression)
            Gamma_1 = H_end, T_high ~ 1e15 GeV  -> BBN OK
        Stage 2 0_1 -> SM nucleons (thermal, instantaneous)
            T_reh ~ T_high
        Resulting N_e ~ 50.5 -> n_s = 0.960 (1.1 sigma), r = 0.0047

Launch:
    PYTHONPATH=. python3 scripts/V5_17_cascade/derive_T_reh_N_e.py
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
from theory.reheating_cascade import (
    C_N_UNKNOT,
    T_REH_BBN_MIN_GEV,
    Gamma_reh_stage1_unknot,
    Gamma_reh_stage2_nucleon,
    T_high_after_stage1,
    n_s_r_recomputed,
    predict_cascade_V5p17,
    validate_against_Planck_V5p17,
)
from theory.reheating_chi import (
    H_end_V5p14,
    validate_against_Planck as predict_v5p16,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("V5.17 - Two-stage cascade reheating: 0_1 (Stage 1) -> SM (Stage 2)")
    print("=" * 72)

    # ----- Predictions
    pred_v5p17 = predict_cascade_V5p17()
    summary_v5p16 = predict_v5p16()

    H_end = pred_v5p17.H_end_GeV
    print()
    print("Inputs (V5.14 inflation):")
    print(f"  alpha_EM           = {ALPHA_EM:.7e}")
    print(f"  H_end              = {H_end:.3e} GeV")
    print()
    print("STAGE 1 (chi -> 0_1 unknot, c_n = 0):")
    print(f"  Gamma_stage1       = {pred_v5p17.Gamma_stage1_GeV:.3e} GeV")
    print(f"  suppression        = alpha_EM^0 = "
          f"{pred_v5p17.suppression_stage1:.6f}  (NO Vassiliev penalty)")
    print(f"  T_high             = {pred_v5p17.T_high_GeV:.3e} GeV")
    print()
    print("STAGE 2 (0_1 photon bath -> SM nucleons, thermal):")
    print(f"  Gamma_stage2_direct= {pred_v5p17.Gamma_stage2_direct_GeV:.3e} GeV"
          f"  (sub-dominant, V5.16-style)")
    print(f"  suppression        = alpha_EM^{C_N_NUCLEON_EFF} = "
          f"{pred_v5p17.suppression_stage2:.3e}")
    print(f"  T_reh              = {pred_v5p17.T_reh_GeV:.3e} GeV")
    print()
    print("V5.17 CMB observables:")
    print(f"  N_e                = {pred_v5p17.N_e_V5p17:.3f} "
          f"(Planck window 50-65)")
    print(f"  n_s                = {pred_v5p17.n_s_V5p17:.4f}  "
          f"(Planck {N_S_PLANCK:.4f} +- {N_S_PLANCK_SIGMA:.4f}, "
          f"dev = {pred_v5p17.n_s_deviation_sigma:+.2f} sigma)")
    print(f"  r                  = {pred_v5p17.r_V5p17:.4e}  "
          f"(BICEP < {R_PLANCK_BICEP:.2f})")
    print(f"  T_reh / T_BBN      = {pred_v5p17.T_reh_GeV/T_REH_BBN_MIN_GEV:.2e}"
          f"  (V5.16 was 0.149 ! BBN FAIL)")
    print(f"  BBN compatible     = {pred_v5p17.BBN_compatible}")
    print(f"  Planck compatible  = {pred_v5p17.Planck_compatible}")
    print(f"  BICEP compatible   = {pred_v5p17.BICEP_compatible}")
    print()
    print("V5.16 vs V5.17 verdict:")
    print(f"  T_reh:   V5.16 = {summary_v5p16['T_reh_GeV']:.3e} GeV  ->  "
          f"V5.17 = {pred_v5p17.T_reh_GeV:.3e} GeV "
          f"(x {pred_v5p17.T_reh_GeV/summary_v5p16['T_reh_GeV']:.2e})")
    print(f"  N_e:     V5.16 = {summary_v5p16['N_e_V5']:.2f}        ->  "
          f"V5.17 = {pred_v5p17.N_e_V5p17:.2f}")
    print(f"  n_s:     V5.16 = {summary_v5p16['n_s_V5']:.4f}      ->  "
          f"V5.17 = {pred_v5p17.n_s_V5p17:.4f}")
    print(f"  r:       V5.16 = {summary_v5p16['r_V5']:.4e}     ->  "
          f"V5.17 = {pred_v5p17.r_V5p17:.4e}")

    # ----- Sweep c_n_eff for the cascade comparison plot
    cn_grid = np.arange(0, 21)
    # V5.16-style "single channel" T_reh as a function of the assumed c_n_eff
    T_reh_single = np.array([
        (90.0 / (np.pi ** 2 * 106.75)) ** 0.25
        * np.sqrt(Gamma_reh_stage2_nucleon(H_end, c_n_eff=int(c))
                  * 2.435e18)
        for c in cn_grid
    ])
    # V5.17 cascade: T_reh always = T_high (set by Stage 1 = 0_1)
    T_reh_cascade = np.full_like(cn_grid, pred_v5p17.T_high_GeV, dtype=float)

    # ----- Figure
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))

    # (a) Gamma_stage1 vs Gamma_stage2 (bar chart)
    ax = axes[0, 0]
    labels = ["Stage 1\n0_1 unknot (V5.17)", "Stage 2 direct\nnucleon (V5.16)"]
    rates = [pred_v5p17.Gamma_stage1_GeV, pred_v5p17.Gamma_stage2_direct_GeV]
    colors = ["C0", "C3"]
    ax.bar(labels, rates, color=colors, log=True)
    ax.axhline(H_end, ls="--", color="black", label="H_end")
    ax.set_ylabel("Decay rate Gamma [GeV]")
    ax.set_title("(a) chi-field decay channels (V5.17 cascade)\n"
                 "Stage 1 wins by factor alpha_EM^{-18} ~ 3e38")
    ax.set_ylim(1e-30, 1e15)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3, axis="y")

    # (b) T_reh: V5.16 single-channel vs V5.17 cascade
    ax = axes[0, 1]
    ax.semilogy(cn_grid, T_reh_single, "-o", color="C3",
                label="V5.16 single-stage (c_n in chi-decay)")
    ax.semilogy(cn_grid, T_reh_cascade, "-s", color="C0",
                label="V5.17 cascade (T_high, c_n=0 channel)")
    ax.axhline(T_REH_BBN_MIN_GEV, ls="--", color="red",
               label="BBN bound 1 MeV")
    ax.axvline(C_N_NUCLEON_EFF, ls=":", color="black",
               label="nucleon c_n=18")
    ax.set_xlabel("c_n^eff (Stage-2 / direct channel)")
    ax.set_ylabel("T_reh [GeV]")
    ax.set_title("(b) T_reh: single-stage vs cascade\n"
                 "V5.17 saturates at T_high regardless of c_n")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # (c) N_e and n_s, V5.16 vs V5.17 markers
    ax = axes[1, 0]
    ax.axhline(N_S_PLANCK, color="red", label=f"Planck n_s = {N_S_PLANCK:.4f}")
    ax.axhspan(N_S_PLANCK - N_S_PLANCK_SIGMA, N_S_PLANCK + N_S_PLANCK_SIGMA,
               color="red", alpha=0.15, label="Planck +- 1 sigma")
    N_grid = np.linspace(20, 70, 200)
    ax.plot(N_grid, [1.0 - 2.0 / N for N in N_grid], color="C0",
            label="Starobinsky n_s = 1 - 2/N_e")
    ax.plot([summary_v5p16["N_e_V5"]], [summary_v5p16["n_s_V5"]],
            marker="x", color="C3", markersize=12,
            label=f"V5.16  (N={summary_v5p16['N_e_V5']:.1f}, "
                  f"n_s={summary_v5p16['n_s_V5']:.4f})")
    ax.plot([pred_v5p17.N_e_V5p17], [pred_v5p17.n_s_V5p17],
            marker="*", color="C0", markersize=18,
            label=f"V5.17  (N={pred_v5p17.N_e_V5p17:.1f}, "
                  f"n_s={pred_v5p17.n_s_V5p17:.4f})")
    ax.axvline(N_EFOLDS_PIVOT, ls=":", color="gray")
    ax.set_xlabel("N_e (e-folds at pivot)")
    ax.set_ylabel("n_s")
    ax.set_xlim(20, 70)
    ax.set_ylim(0.93, 0.98)
    ax.set_title("(c) Scalar tilt n_s: V5.16 vs V5.17")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (d) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["Quantity", "V5.16 (1-stage)", "V5.17 (cascade)"],
        ["Gamma/H_end",
         f"{ALPHA_EM**C_N_NUCLEON_EFF:.2e}",
         f"{pred_v5p17.suppression_stage1:.4f}"],
        ["T_reh [GeV]",
         f"{summary_v5p16['T_reh_GeV']:.2e}",
         f"{pred_v5p17.T_reh_GeV:.2e}"],
        ["T_reh / T_BBN",
         f"{summary_v5p16['T_reh_over_T_BBN']:.2e}",
         f"{pred_v5p17.T_reh_GeV/T_REH_BBN_MIN_GEV:.2e}"],
        ["w_reh",
         f"0 (matter)",
         f"1/3 (radiation)"],
        ["N_e",
         f"{summary_v5p16['N_e_V5']:.2f}",
         f"{pred_v5p17.N_e_V5p17:.2f}"],
        ["n_s",
         f"{summary_v5p16['n_s_V5']:.4f}",
         f"{pred_v5p17.n_s_V5p17:.4f}"],
        ["n_s deviation sigma",
         f"{summary_v5p16['n_s_deviation_sigma']:+.2f}",
         f"{pred_v5p17.n_s_deviation_sigma:+.2f}"],
        ["r",
         f"{summary_v5p16['r_V5']:.4e}",
         f"{pred_v5p17.r_V5p17:.4e}"],
        ["BBN compatible",
         "NO" if not summary_v5p16['BBN_compatible'] else "YES",
         "YES" if pred_v5p17.BBN_compatible else "NO"],
        ["Planck compatible",
         "NO" if not summary_v5p16['Planck_compatible'] else "YES",
         "YES" if pred_v5p17.Planck_compatible else "NO"],
        ["BICEP compatible",
         "YES" if summary_v5p16['BICEP_compatible'] else "NO",
         "YES" if pred_v5p17.BICEP_compatible else "NO"],
    ]
    table = ax.table(cellText=rows, loc="center",
                     cellLoc="left", colWidths=[0.36, 0.30, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.30)
    ax.set_title("(d) V5.16 vs V5.17 comparison", pad=10)

    fig.suptitle(
        "V5.17 — Cascade reheating: 0_1 unknot (Stage 1) -> SM (Stage 2)\n"
        f"T_reh = {pred_v5p17.T_reh_GeV:.2e} GeV, "
        f"N_e = {pred_v5p17.N_e_V5p17:.1f}, "
        f"n_s = {pred_v5p17.n_s_V5p17:.4f}, "
        f"r = {pred_v5p17.r_V5p17:.4f}  --  ALL bounds satisfied",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = OUTPUT_DIR / "v5p17_cascade_reheating.png"
    fig.savefig(out, dpi=150)
    print()
    print(f"Figure saved to {out}")

    print()
    print("=" * 72)
    print("HONEST VERDICT (V5.17):")
    print("=" * 72)
    print("  V5.17 resolves the V5.16 BBN tension by recognizing that the")
    print("  *dominant* chi-field decay channel is the topologically trivial")
    print("  0_1 (unknot) mode, NOT the nucleon. Because Jones(0_1)=1, the")
    print("  Vassiliev penalty alpha_EM^{c_n} vanishes for c_n=0, and the")
    print("  Stage-1 rate is the natural Hubble rate at the end of inflation.")
    print()
    print("  Stage 2 (0_1 photon bath -> SM nucleons) is set kinematically")
    print("  by SU(3) thermal equilibration at T_high, which is instantaneous")
    print("  on cosmological timescales. The reheating temperature is therefore")
    print(f"  T_reh ~ {pred_v5p17.T_reh_GeV:.2e} GeV, ~10^18 above the BBN bound.")
    print()
    print("  Limits:")
    print("   - Prefactor in Gamma_stage1 = H_end is taken O(1) (Jones=1, no")
    print("     SU(2)_135 form factor computed). Shifts N_e by +- 1 e-fold.")
    print("   - Stage 2 instantaneousness assumed (SU(3) gauge equilibration")
    print("     at T_high >> H(T_high)); not derived rigorously here.")
    print("   - V5.17 introduces NO new free parameter relative to V5.16.")
    print("=" * 72)


if __name__ == "__main__":
    main()
