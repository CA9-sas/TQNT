#!/usr/bin/env python3
"""
V5.12 — Validation Planck 2018 du spectre C_ℓ TT.

Compare le spectre C_ℓ V5 (pics acoustiques + plateau Sachs-Wolfe +
damping de Silk) au best-fit Planck 2018 :
  - Positions des pics ℓ_1 ≈ 220, ℓ_2 ≈ 540, ℓ_3 ≈ 815
  - Amplitude au premier pic ~ 5755 µK²
  - Rapports d'harmoniques ℓ_n/ℓ_1

Produit output/v5p12_planck_C_l_validation.png (3 panneaux).

Lancement :
    PYTHONPATH=. python3 scripts/V5_12_cmb/validate_planck_C_l.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.cmb_perturbations import (
    C_L_PEAK1_PLANCK_MUK2,
    L_PEAK1_PLANCK,
    L_PEAK2_PLANCK,
    L_PEAK3_PLANCK,
    C_l_spectrum_V5,
    summary,
    validate_planck_peaks,
)
from theory.flrw_emergence import cosmo_state_V5

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    state = cosmo_state_V5()
    val = validate_planck_peaks(state)
    s = summary()

    l_arr = np.arange(2, 2501)
    spec = C_l_spectrum_V5(l_arr, state)
    Dl_V5 = spec["D_l_muK2"]

    # ---- Synthétiser un Planck 2018 best fit semi-réaliste pour comparaison
    # (gaussiennes aux ℓ_n Planck, mêmes amplitudes relatives).
    # NB : pas un C_ℓ Planck publié — c'est un substitut visuel calibré.
    def planck_like_Cl(l):
        sigma = 80.0
        amp1 = C_L_PEAK1_PLANCK_MUK2
        Dl = (
            amp1 * np.exp(-((l - L_PEAK1_PLANCK) / sigma) ** 2)
            + amp1 * 0.45 * np.exp(-((l - L_PEAK2_PLANCK) / sigma) ** 2)
            + amp1 * 0.42 * np.exp(-((l - L_PEAK3_PLANCK) / sigma) ** 2)
            + amp1 * 0.27 * np.exp(-((l - 1130.0) / sigma) ** 2)
        )
        plateau = 1100.0 / (1.0 + (l / 30.0) ** 2) ** 0.5
        return plateau + Dl * np.exp(-(l / 1300.0) ** 2)

    Dl_planck = planck_like_Cl(l_arr)

    # =====================================================================
    # FIGURE 3 panneaux
    # =====================================================================
    fig = plt.figure(figsize=(15, 11))

    # Panneau 1 : spectre TT complet
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(l_arr, Dl_V5, "b-", lw=2.0, label="V5.12 (chi-field perturbe)")
    ax1.plot(l_arr, Dl_planck, "r--", lw=1.3, alpha=0.75,
             label="Planck 2018 best fit (proxy)")
    for lp, c in [(L_PEAK1_PLANCK, "g"), (L_PEAK2_PLANCK, "g"),
                  (L_PEAK3_PLANCK, "g")]:
        ax1.axvline(lp, color=c, ls=":", alpha=0.4)
    ax1.set_xlabel(r"Multipole $\ell$")
    ax1.set_ylabel(r"$\ell(\ell+1) C_\ell^{TT} / 2\pi$  [$\mu K^2$]")
    ax1.set_title("V5.12 - Spectre TT CMB depuis perturbations chi-field")
    ax1.set_xlim(2, 2500)
    ax1.set_ylim(0, 7000)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Panneau 2 : zoom basse-ℓ (Sachs-Wolfe)
    ax2 = plt.subplot(2, 2, 2)
    mask_low = l_arr <= 60
    ax2.plot(l_arr[mask_low], Dl_V5[mask_low], "b-", lw=2.0,
             label="V5.12 SW + ISW")
    ax2.plot(l_arr[mask_low], Dl_planck[mask_low], "r--", lw=1.3,
             alpha=0.75, label="Planck 2018 proxy")
    ax2.set_xlabel(r"Multipole $\ell$")
    ax2.set_ylabel(r"$\ell(\ell+1) C_\ell / 2\pi$ [$\mu K^2$]")
    ax2.set_title("Zoom Sachs-Wolfe + ISW (plateau bas-l)")
    ax2.set_xlim(2, 60)
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Panneau 3 : positions des pics
    ax3 = plt.subplot(2, 2, 3)
    bars_planck = [L_PEAK1_PLANCK, L_PEAK2_PLANCK, L_PEAK3_PLANCK]
    bars_V5 = [val.l_1_V5, val.l_2_V5, val.l_3_V5]
    x = np.arange(3)
    w = 0.35
    ax3.bar(x - w/2, bars_planck, w, color="red", alpha=0.7,
            label="Planck 2018")
    ax3.bar(x + w/2, bars_V5, w, color="blue", alpha=0.7, label="V5.12")
    ax3.set_xticks(x)
    ax3.set_xticklabels([r"$\ell_1$", r"$\ell_2$", r"$\ell_3$"])
    ax3.set_ylabel("Position multipole")
    ax3.set_title("Pics acoustiques V5 vs Planck 2018")
    ax3.grid(alpha=0.3, axis="y")
    ax3.legend()
    for i, (lp, lv) in enumerate(zip(bars_planck, bars_V5)):
        ax3.text(i - w/2, lp + 15, f"{lp:.0f}", ha="center", fontsize=9)
        ax3.text(i + w/2, lv + 15, f"{lv:.0f}", ha="center", fontsize=9)
        err = (lv - lp) / lp * 100.0
        ax3.text(i, lp / 2.0, f"{err:+.1f} %",
                 ha="center", fontsize=11, fontweight="bold",
                 color="purple")

    # Panneau 4 : table résumé
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    rows = [
        ["Quantite", "V5.12", "Planck", "Ecart"],
        [r"$\ell_A$ (echelle acoustique)",
         f"{val.l_A_V5:.1f}", "~302",
         f"{(val.l_A_V5 - 302) / 302 * 100:+.1f} %"],
        [r"$\ell_1$ (1er pic)",
         f"{val.l_1_V5:.1f}", "220.0",
         f"{val.l_1_error_pct:+.2f} %"],
        [r"$\ell_2$ (2e pic)",
         f"{val.l_2_V5:.1f}", "540.0",
         f"{val.l_2_error_pct:+.2f} %"],
        [r"$\ell_3$ (3e pic)",
         f"{val.l_3_V5:.1f}", "815.0",
         f"{val.l_3_error_pct:+.2f} %"],
        [r"$\ell_2 / \ell_1$",
         f"{val.ratio_l2_l1_V5:.3f}",
         f"{val.ratio_l2_l1_planck:.3f}", "--"],
        [r"$\ell_3 / \ell_1$",
         f"{val.ratio_l3_l1_V5:.3f}",
         f"{val.ratio_l3_l1_planck:.3f}", "--"],
        [r"$D_\ell(\ell_1)$ $\mu K^2$",
         f"{s['Dl_peak1_V5_muK2']:.0f}",
         f"{C_L_PEAK1_PLANCK_MUK2:.0f}",
         f"{(s['Dl_peak1_V5_muK2'] - C_L_PEAK1_PLANCK_MUK2) / C_L_PEAK1_PLANCK_MUK2 * 100:+.1f} %"],
        [r"$r_s$ drag (Mpc)",
         f"{val.r_s_Mpc:.2f}", "147.05",
         f"{(val.r_s_Mpc - 147.05) / 147.05 * 100:+.2f} %"],
        [r"$D_A(z_*)$ (Mpc)",
         f"{val.D_A_LSS_Mpc:.2f}", "~13.0",
         "--"],
    ]
    table = ax4.table(cellText=rows, cellLoc="center", loc="center",
                      colWidths=[0.42, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.6)
    # En-tête en gras
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor("#e0e0e0")
        cell.set_text_props(weight="bold")
    ax4.set_title("V5.12 - Validation Planck 2018 (semi-analytique)",
                  fontsize=12)

    fig.suptitle(
        "V5.12 - Perturbations CMB depuis chi-field V5.11 : "
        "spectre $C_\\ell^{TT}$ vs Planck 2018",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUTPUT_DIR / "v5p12_planck_C_l_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistree : {out}")

    # Print summary terminal
    print("=" * 72)
    print(" V5.12 - Validation Planck 2018 - resume")
    print("=" * 72)
    print(f"  l_A  V5  = {val.l_A_V5:.2f}")
    print(f"  l_1  V5  = {val.l_1_V5:7.2f} | Planck {val.l_1_planck:7.2f}"
          f" | {val.l_1_error_pct:+5.2f} %")
    print(f"  l_2  V5  = {val.l_2_V5:7.2f} | Planck {val.l_2_planck:7.2f}"
          f" | {val.l_2_error_pct:+5.2f} %")
    print(f"  l_3  V5  = {val.l_3_V5:7.2f} | Planck {val.l_3_planck:7.2f}"
          f" | {val.l_3_error_pct:+5.2f} %")
    print(f"  D_l(l_1) V5 = {s['Dl_peak1_V5_muK2']:.1f} uK^2 |"
          f" Planck {C_L_PEAK1_PLANCK_MUK2:.1f} uK^2")
    print("=" * 72)


if __name__ == "__main__":
    main()
