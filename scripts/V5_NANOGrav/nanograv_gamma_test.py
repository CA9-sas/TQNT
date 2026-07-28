#!/usr/bin/env python3
"""
V5 — Test du fond stochastique nHz contre NANOGrav 15-yr (Agazie+ 2023).

Produit :
  - output/v5_nanograv_gamma_test.png : 2 panneaux
      (1) strain h_c(f) : observations vs SMBHB pur vs V5 full mixture
      (2) pente locale alpha_eff(f) et flèche du gamma observé
  - Affichage console des chi^2 et du ln B (Bayes BIC).

Lancement :
    PYTHONPATH=. python3 scripts/V5_NANOGrav/nanograv_gamma_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.nanograv_sgwb import (
    ALPHA_SMBHB,
    ALPHA_STRING,
    ALPHA_V5_MEMBRANE,
    ALPHA_V5_POINT,
    F_YR_HZ,
    GAMMA_SMBHB,
    GAMMA_STRING,
    GAMMA_V5_MEMBRANE,
    GAMMA_V5_POINT,
    NANOGRAV_A_YR_HD,
    NANOGRAV_GAMMA_HD,
    NANOGRAV_GAMMA_HD_SIGMA,
    alpha_to_gamma,
    effective_alpha,
    h_c_SMBHB,
    h_c_V5_combined,
    h_c_powerlaw,
    nanograv_15yr_bins,
    run_full_test,
)


def main():
    print("=" * 72)
    print(" V5 NANOGrav gamma_SGWB test (vs Agazie+ 2023, 15-yr)")
    print("=" * 72)

    f_obs, h_obs, sigma_log10 = nanograv_15yr_bins()
    r = run_full_test()

    print("\n  --- gamma theorique ---")
    print(f"  SMBHB           : gamma = {GAMMA_SMBHB:.3f}")
    print(f"  cordes cusps    : gamma = {GAMMA_STRING:.3f}")
    print(f"  V5 ponctuel     : gamma = {GAMMA_V5_POINT:.3f}")
    print(f"  V5 membrane     : gamma = {GAMMA_V5_MEMBRANE:.3f}")
    print(f"\n  gamma OBSERVE NANOGrav 15yr = {NANOGRAV_GAMMA_HD:.2f} +/- "
          f"{NANOGRAV_GAMMA_HD_SIGMA:.2f}")

    print("\n  --- fits ---")
    fS = r.fit_SMBHB
    print(f"  SMBHB pur  : A_yr = {fS['A_yr']:.3e}, "
          f"chi^2 = {fS['chi2']:.2f} ({fS['dof']} dof)")
    fV = r.fit_V5_full
    print(f"  V5 full    : A_SMBHB = {fV['A_SMBHB']:.3e}, "
          f"A_string = {fV['A_string']:.3e}, A_mem = {fV['A_mem']:.3e}")
    print(f"               chi^2 = {fV['chi2']:.2f} ({fV['dof']} dof), "
          f"gamma_eff(10 nHz) = {fV['gamma_eff_10nHz']:.2f}")
    fMem = r.fit_V5_membrane
    print(f"  V5 membrane: A_SMBHB = {fMem['A_SMBHB']:.3e}, "
          f"A_mem = {fMem['A_mem']:.3e}, chi^2 = {fMem['chi2']:.2f}")

    print("\n  --- Bayes (BIC) ---")
    print(f"  ln B(V5 full / SMBHB)      = "
          f"{r.bayes_full['ln_Bayes_V5_over_SMBHB']:+.2f}"
          f"  -> {r.bayes_full['verdict']}")
    print(f"  ln B(V5 membrane / SMBHB)  = "
          f"{r.bayes_membrane['ln_Bayes_V5_over_SMBHB']:+.2f}"
          f"  -> {r.bayes_membrane['verdict']}")

    # ---------------------------------------------------------------
    # Plot.
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- Panel 1 : strain h_c(f) ---
    ax = axes[0]
    f_grid = np.logspace(np.log10(1e-9), np.log10(1e-7), 200)

    ax.plot(f_grid, h_c_SMBHB(f_grid, A_yr=fS["A_yr"]),
            color="C0", lw=2.2,
            label=fr"SMBHB pur ($\gamma=13/3\approx 4.33$), $\chi^2={fS['chi2']:.2f}$")
    ax.plot(f_grid, h_c_V5_combined(f_grid, fV["A_SMBHB"],
                                      fV["A_string"], fV["A_mem"]),
            color="C3", lw=2.2, ls="--",
            label=fr"V5 mixture (SMBHB+strings+membrane), $\chi^2={fV['chi2']:.2f}$")
    ax.plot(f_grid,
            h_c_powerlaw(f_grid, fV["A_mem"], ALPHA_V5_MEMBRANE),
            color="C2", lw=1.2, ls=":",
            label=fr"V5 membrane seule ($\gamma=7/3$)")
    ax.plot(f_grid,
            h_c_powerlaw(f_grid, fV["A_string"], ALPHA_STRING),
            color="C1", lw=1.2, ls=":",
            label=fr"V5 string seule ($\gamma=16/3$)")

    # Données : barres en log10
    h_lo = h_obs * 10.0 ** (-sigma_log10)
    h_hi = h_obs * 10.0 ** (+sigma_log10)
    ax.errorbar(f_obs, h_obs,
                yerr=[h_obs - h_lo, h_hi - h_obs],
                fmt="o", color="k", capsize=4, ms=8, lw=1.5,
                label="NANOGrav 15-yr (Agazie+ 2023)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequence f [Hz]", fontsize=11)
    ax.set_ylabel(r"Strain caracteristique $h_c(f)$", fontsize=11)
    ax.set_title("SGWB nHz : modeles vs NANOGrav 15-yr", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # --- Panel 2 : pente alpha_eff(f) et gamma observe ---
    ax = axes[1]
    f_g = np.logspace(np.log10(1.5e-9), np.log10(7e-8), 100)
    alpha_eff_V5 = effective_alpha(f_g, fV["A_SMBHB"],
                                     fV["A_string"], fV["A_mem"])
    gamma_eff_V5 = alpha_to_gamma(alpha_eff_V5)
    gamma_smbhb = np.full_like(f_g, GAMMA_SMBHB)

    ax.plot(f_g, gamma_smbhb, color="C0", lw=2.2,
            label=r"$\gamma_{\rm SMBHB}=13/3$")
    ax.plot(f_g, gamma_eff_V5, color="C3", lw=2.2, ls="--",
            label=r"$\gamma_{\rm eff}^{V5}(f)$ (mixture)")
    ax.axhline(GAMMA_V5_POINT, color="C4", lw=1.2, ls=":",
               label=r"$\gamma_{V5,\rm pt}=5$")
    ax.axhline(GAMMA_V5_MEMBRANE, color="C2", lw=1.2, ls=":",
               label=r"$\gamma_{V5,\rm mem}=7/3$")

    # NANOGrav band
    ax.axhspan(NANOGRAV_GAMMA_HD - NANOGRAV_GAMMA_HD_SIGMA,
               NANOGRAV_GAMMA_HD + NANOGRAV_GAMMA_HD_SIGMA,
               color="gray", alpha=0.25,
               label=fr"NANOGrav HD : $\gamma = {NANOGRAV_GAMMA_HD:.1f}\pm{NANOGRAV_GAMMA_HD_SIGMA}$")
    ax.axhline(NANOGRAV_GAMMA_HD, color="k", lw=1.0)

    ax.set_xscale("log")
    ax.set_xlabel("Frequence f [Hz]", fontsize=11)
    ax.set_ylabel(r"Index spectral PSD $\gamma$", fontsize=11)
    ax.set_title(r"Pente $\gamma$ : theorie vs observation", fontsize=12)
    ax.set_ylim(1.0, 6.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle("V5 — Test gamma SGWB nHz vs NANOGrav 15-yr (Agazie+ 2023)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = ROOT / "output" / "v5_nanograv_gamma_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  Figure sauvee : {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
