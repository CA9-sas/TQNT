"""V5.38 — Test de la derivation rigoureuse delta = -4 ln(d_{1/2}(k=135)).

Compare V5.35d avec delta first-principles (sans calibration) vs delta
calibre raffine. Genere output/v5p38_delta.png.

Reference : theory/v5p38_delta_derivation.py
            theory/v5p35d_3body_multiplicative.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p38_delta_derivation import (
    DELTA_D1_AMPHI,
    compare_calibrated_vs_exact,
    decompose_4lnd,
    delta_first_principles,
    enumerate_interpretations,
    retained_interpretation,
    summary_report,
)
from theory.su2_modular import V5_LEVEL


def print_full_report() -> None:
    print(summary_report())


def print_interpretation_table() -> None:
    print()
    print("=" * 88)
    print("V5.38 — Sept lectures candidates pour le facteur 4")
    print("=" * 88)
    for itp in enumerate_interpretations():
        marker = "[RETENU]" if itp.is_retained else "[rejete]"
        print(f"  {marker:<10} {itp.label}")
        print(f"             {itp.description[:130]}...")
    rt = retained_interpretation()
    print()
    print(f"=> RETENU : {rt.label}")
    print(f"   {rt.description}")
    print()


def make_plot(out_path: str = "output/v5p38_delta.png") -> None:
    cmp = compare_calibrated_vs_exact()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # Panneau 1 : per-meson log_ratio (delta_exact vs delta_calibrated)
    mesons = ("K0", "D0", "B0", "Bs0")
    log_r_exact = [cmp["per_meson"][m]["log_r_exact"] for m in mesons]
    log_r_calib = [cmp["per_meson"][m]["log_r_calib"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35
    ax1.bar(
        x - w / 2,
        log_r_exact,
        w,
        color="#2ca02c",
        label=(
            r"$\delta_{\rm exact} = -4\ln d_{1/2}(k\!=\!135)$ "
            f"= {cmp['delta_exact']:+.4f}"
        ),
    )
    ax1.bar(
        x + w / 2,
        log_r_calib,
        w,
        color="#1f77b4",
        label=(
            r"$\delta_{\rm calib}$ "
            f"= {cmp['delta_calibrated']:+.4f}"
        ),
    )
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.axhline(+0.01, color="g", ls=":", lw=0.8, alpha=0.7)
    ax1.axhline(-0.01, color="g", ls=":", lw=0.8, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r"$K^0$", r"$D^0$", r"$B^0$", r"$B_s^0$"])
    ax1.set_ylabel(r"$\log_{10}(\Delta m^{\rm V5.35d} / \Delta m^{\rm PDG})$")
    ax1.set_title(
        f"V5.38 — Closure {chr(963)}|log| : exact={cmp['sum_abs_log_exact']:.4f} dex, "
        f"calib={cmp['sum_abs_log_calib']:.4f} dex"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panneau 2 : decomposition Taylor de -4 ln(d_{1/2}(k)) avec k variable
    ax2.set_title(
        r"V5.38 — convergence $-4\ln d_{1/2}(k) \to -\ln 16$ "
        r"quand $k\to\infty$"
    )
    ks = np.array([10, 20, 30, 50, 80, 135, 200, 500, 1000, 5000])
    vals = []
    for k in ks:
        import math
        d = math.sin(math.pi * 2 / (k + 2)) / math.sin(math.pi / (k + 2))
        vals.append(-4 * math.log(d))
    ax2.plot(ks, vals, "o-", color="#2ca02c", label=r"$-4\ln d_{1/2}(k)$")
    ax2.axhline(
        DELTA_D1_AMPHI, color="#1f77b4", ls="--",
        label=r"$-\ln 16 = -2.7726$ (limite $k\to\infty$, D1)",
    )
    ax2.axhline(
        cmp["delta_calibrated"], color="r", ls=":",
        label=f"calibrated = {cmp['delta_calibrated']:+.4f}",
    )
    ax2.axvline(V5_LEVEL, color="k", ls=":", alpha=0.5, label=f"k = {V5_LEVEL}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Chern-Simons level k")
    ax2.set_ylabel(r"$\delta(k) = -4\ln d_{1/2}(k)$")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegarde : {out_path}")


def main() -> None:
    print()
    print(
        "V5.38 — derivation rigoureuse de delta depuis SU(2)_135\n"
        "    delta_exact = -4 ln(d_{1/2}(k=135)) = -4 ln(2 cos(pi/137))\n"
    )
    print_full_report()
    print_interpretation_table()
    decomp = decompose_4lnd()
    print(
        f"Decomposition Taylor : -4 ln 2 + 2(pi/137)^2 + (1/3)(pi/137)^4 + ...\n"
        f"  D1 = {decomp.leading_minus_4_ln2:+.10f}\n"
        f"  2nd= {decomp.second_order:+.10f}\n"
        f"  4th= {decomp.fourth_order:+.10f}\n"
        f"  sum= {decomp.leading_minus_4_ln2 + decomp.second_order + decomp.fourth_order:+.10f}\n"
        f"  D2 = {delta_first_principles():+.10f}\n"
    )
    make_plot()


if __name__ == "__main__":
    main()
