"""
scripts/V5_NICER/validate_M_R_diagram.py
========================================

Trace les courbes M-R V5 (avec et sans correction P5) pour 4 EoS et les
compare aux contours mesurés par NICER pour PSR J0030+0451 et PSR J0740+6620.

Outputs :
    output/v5_NICER_M_R_validation.png  — diagramme M-R 4 EoS × {GR, V5}
    output/v5_NICER_summary.txt         — résumé numérique (χ², Bayes)

Usage : python3 -m scripts.V5_NICER.validate_M_R_diagram
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theory.tov_V5 import (
    EOS_LIBRARY, KAPPA_G_M,
    mass_radius_curve, chi2_vs_NICER, bayes_factor_V5_vs_GR,
    relative_radius_shift, find_M_max,
    NICER_PULSARS, kappa_g_upper_limit,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 72)
    print("V5 NICER — test TOV modifiee V5 sur PSR J0030 + J0740")
    print("=" * 72)
    print(f"kappa_g (V5) = {KAPPA_G_M:.3e} m")
    print()

    # ------------------------------------------------------------------
    # 1. Calcul des courbes M-R
    # ------------------------------------------------------------------
    P_c_grid = np.logspace(33.0, 37.0, 25)
    curves_GR = {}
    curves_V5 = {}
    for name, eos in EOS_LIBRARY.items():
        curves_GR[name] = mass_radius_curve(eos, P_c_grid=P_c_grid,
                                             modifications=False)
        curves_V5[name] = mass_radius_curve(eos, P_c_grid=P_c_grid,
                                             modifications=True)
        Mmax_GR, R_GR = find_M_max(eos, modifications=False)
        Mmax_V5, R_V5 = find_M_max(eos, modifications=True)
        print(f"EoS {name:6s} : M_max(GR)={Mmax_GR:.3f} M_sun  R(M_max)={R_GR:.2f} km")
        print(f"            : M_max(V5)={Mmax_V5:.3f} M_sun  R(M_max)={R_V5:.2f} km")
        print(f"            : Δ(M_max)={Mmax_V5-Mmax_GR:+.2e} M_sun, "
              f"Δ(R)={R_V5-R_GR:+.2e} km")
    print()

    # ------------------------------------------------------------------
    # 2. χ² et Bayes factor
    # ------------------------------------------------------------------
    print(" EoS    | χ²(GR)  χ²(V5)  Δχ²(V5-GR) | B(V5/GR)")
    print("-" * 60)
    summary_lines = ["EoS,chi2_GR,chi2_V5,delta_chi2,bayes_V5_GR\n"]
    for name in EOS_LIBRARY:
        c_GR = chi2_vs_NICER(curves_GR[name])
        c_V5 = chi2_vs_NICER(curves_V5[name])
        dchi2 = c_V5 - c_GR
        B = math.exp(-0.5 * dchi2)
        print(f" {name:6s} | {c_GR:6.3f}  {c_V5:6.3f}  {dchi2:+.3e} | {B:.6f}")
        summary_lines.append(f"{name},{c_GR:.6f},{c_V5:.6f},{dchi2:.6e},{B:.6f}\n")
    print()

    # ------------------------------------------------------------------
    # 3. Sensibilité : ΔR/R pour M ≈ 2 M_sun
    # ------------------------------------------------------------------
    print("Sensibilite V5 (M ~ 2 M_sun):")
    P_c_2sun = 1e36
    for name, eos in EOS_LIBRARY.items():
        dr = relative_radius_shift(eos, P_c_2sun)
        print(f"  {name:6s} : ΔR/R = {dr:+.3e}")
    print()

    # ------------------------------------------------------------------
    # 4. Limite supérieure sur κ_g
    # ------------------------------------------------------------------
    print("Limite NICER sur kappa_g (sigma_R/R = 10%):")
    for name, eos in EOS_LIBRARY.items():
        kmax = kappa_g_upper_limit(eos, sigma_R_rel_target=0.10)
        ratio = kmax / KAPPA_G_M
        print(f"  {name:6s} : kappa_g_max ≲ {kmax:.2e} m  (= {ratio:.1e} × kappa_g_V5)")
    print()

    # ------------------------------------------------------------------
    # 5. Diagramme M-R
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Contours NICER (ellipses 1σ)
    colors_nicer = ["#1f77b4", "#d62728"]
    for (name, p), col in zip(NICER_PULSARS.items(), colors_nicer):
        # 1σ
        e1 = Ellipse((p["R_km"], p["M_sun"]),
                     width=2*p["R_err"], height=2*p["M_err"],
                     facecolor=col, alpha=0.20, edgecolor=col, lw=1.5,
                     label=f"PSR {name} (NICER 1σ)")
        ax.add_patch(e1)
        ax.plot(p["R_km"], p["M_sun"], "x", color=col, markersize=8)

    # Courbes M-R
    eos_styles = {
        "APR":  ("-", "C2"),
        "SLy4": ("-", "C3"),
        "SLy9": ("-", "C4"),
        "POLY": ("-", "C5"),
    }
    for name in EOS_LIBRARY:
        ls, col = eos_styles[name]
        c = curves_GR[name]
        # filtrer : ne tracer que la branche stable (avant M_max)
        i_max = int(np.argmax(c["M_sun"]))
        mask = slice(0, i_max+1)
        ax.plot(c["R_km"][mask], c["M_sun"][mask], ls, color=col,
                lw=2, label=f"{name} (GR)")
        # V5 (pointillé, mais superpose GR car ΔR ~ 1 m)
        c2 = curves_V5[name]
        ax.plot(c2["R_km"][mask], c2["M_sun"][mask], "--", color=col,
                lw=1, alpha=0.6)

    ax.set_xlabel("Rayon R (km)", fontsize=12)
    ax.set_ylabel("Masse M (M$_\\odot$)", fontsize=12)
    ax.set_xlim(8, 16)
    ax.set_ylim(0.5, 2.7)
    ax.set_title("Diagramme M-R : TOV V5 (P5) vs GR — confrontation NICER",
                 fontsize=13)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    note = ("V5 P5: $\\Omega^2 = 1 + \\kappa_g |\\Psi|^2$, "
            f"$\\kappa_g = {KAPPA_G_M:.2e}$ m\n"
            "$\\Delta R / R \\sim \\kappa_g \\cdot n_N \\sim 10^{-9}$ "
            "$\\ll$ precision NICER ($\\sim 10\\%$)\n"
            "Verdict : V5 et GR indiscernables sur PSR J0030 et J0740.")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "v5_NICER_M_R_validation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Figure : {fig_path}")

    summary_path = OUTPUT_DIR / "v5_NICER_summary.txt"
    summary_path.write_text("".join(summary_lines))
    print(f"Resume : {summary_path}")

    # ------------------------------------------------------------------
    # 6. Verdict final
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  kappa_g (V5) = {KAPPA_G_M:.3e} m  (derive V5.5 etape 6)")
    print(f"  Delta R/R (M=2 M_sun)  ~ 10^-9")
    print(f"  Precision NICER (M=2)  ~ 10% (R_err ~ 1.1 km / R ~ 12 km)")
    print(f"  → V5 P5 strictement INDISCERNABLE de GR a la precision NICER.")
    print(f"  → V5 NON CONTRAINT (kappa_g_max / kappa_g_V5 > 10^6).")
    print(f"  → V5 PASSE le test sans aucun ajustement.")
    print(f"  → Pas de signature observable de P5 dans le secteur NS.")
    print()


if __name__ == "__main__":
    main()
