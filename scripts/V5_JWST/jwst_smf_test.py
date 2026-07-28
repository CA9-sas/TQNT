#!/usr/bin/env python3
"""
V5 JWST SMF test — confront V5 P22 boost prediction with JWST high-z data.

Computes:
  - SMF(M_star, z) in LambdaCDM and V5 P22 for z = 6, 8, 10, 12
  - chi^2 over JADES/CEERS/COSMOS-Web SMF measurements
  - Bayes factor B(V5/LCDM)
  - Best-fit alpha_P22 (NOT the V5 §11.5 nominal 0.10)
  - "Impossible early galaxies" rho_*(z) comparison vs Labbe+2023

Produces output/v5_jwst_smf_comparison.png (4 panels).

Run:
    PYTHONPATH=. python3 scripts/V5_JWST/jwst_smf_test.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.smf_V5 import (
    SMF_LambdaCDM, SMF_V5, boost_P22, rho_star, rho_star_massive,
    bayes_factor_V5_vs_LCDM, chi2_smf, impossible_early_galaxies_test,
    JWST_SMF_DATA, JWST_Z_MID, LABBE_2023,
)


OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def scan_alpha_P22(alphas):
    """Scan alpha_P22 over a grid, return (alphas, chi2_V5_array, chi2_LCDM)."""
    chi2_lcdm = sum(chi2_smf(b, "LambdaCDM")[0] for b in JWST_SMF_DATA)
    chi2s = []
    for a in alphas:
        chi2_v = sum(chi2_smf(b, "V5", alpha_P22=a)[0] for b in JWST_SMF_DATA)
        chi2s.append(chi2_v)
    return np.array(alphas), np.array(chi2s), chi2_lcdm


def main():
    # ---------------------------------------------------------------
    # 1. Bayes factor with V5 nominal (alpha_P22 = 0.10)
    # ---------------------------------------------------------------
    bayes_nom = bayes_factor_V5_vs_LCDM(alpha_P22=0.10)

    # ---------------------------------------------------------------
    # 2. Best-fit alpha_P22 scan
    # ---------------------------------------------------------------
    alphas = np.logspace(-2, 3, 41)
    a_grid, chi2_grid, chi2_lcdm = scan_alpha_P22(alphas)
    i_best = int(np.argmin(chi2_grid))
    alpha_best = a_grid[i_best]
    chi2_best = chi2_grid[i_best]

    bayes_best = bayes_factor_V5_vs_LCDM(alpha_P22=alpha_best)

    # ---------------------------------------------------------------
    # 3. Impossible early galaxies
    # ---------------------------------------------------------------
    ie_results = [
        impossible_early_galaxies_test(z, alpha_P22=alpha_best)
        for z in (7.0, 8.0, 9.0, 10.0)
    ]

    # ---------------------------------------------------------------
    # 4. Print summary
    # ---------------------------------------------------------------
    print("=" * 78)
    print("V5 §11 SMF — JWST high-z test")
    print("=" * 78)
    print()
    print(f"V5 nominal alpha_P22 = 0.10 (V5 §11.5 quote: boost(z=10)=1.5)")
    print(f"  boost(z=8)  = {boost_P22(8.0):.3f}")
    print(f"  boost(z=10) = {boost_P22(10.0):.3f}")
    print(f"  boost(z=12) = {boost_P22(12.0):.3f}")
    print()
    print(f"chi^2 over {bayes_nom['N_data']} JWST data points:")
    print(f"  chi^2(LambdaCDM)        = {bayes_nom['chi2_LCDM']:.2f}")
    print(f"  chi^2(V5, alpha=0.10)   = {bayes_nom['chi2_V5']:.2f}")
    print(f"  Delta chi^2             = {bayes_nom['delta_chi2']:.2f}")
    print(f"  Bayes raw (V5/LCDM)     = {bayes_nom['bayes_raw']:.3e}")
    print(f"  Bayes BIC (V5/LCDM)     = {bayes_nom['bayes_BIC']:.3e}")
    print()
    print(f"Best-fit alpha_P22 (free parameter scan):")
    print(f"  alpha_best  = {alpha_best:.3f}")
    print(f"  chi^2_best  = {chi2_best:.2f}")
    print(f"  boost(z=10) at best = {boost_P22(10.0, alpha_P22=alpha_best):.3f}")
    print(f"  boost(z=12) at best = {boost_P22(12.0, alpha_P22=alpha_best):.3f}")
    print(f"  Delta chi^2 (best)  = {chi2_lcdm - chi2_best:.2f}")
    print(f"  Bayes raw (best)    = {bayes_best['bayes_raw']:.3e}")
    print()
    print(f"Impossible early galaxies (Labbe+2023) at best-fit alpha_P22:")
    for r in ie_results:
        print(
            f"  z={r['z_test']:.0f}: rho_LCDM={r['rho_star_LCDM_Msun_Mpc3']:.2e}, "
            f"rho_V5={r['rho_star_V5_Msun_Mpc3']:.2e}, "
            f"obs={r['obs_lower']:.1e}-{r['obs_upper']:.1e}, "
            f"V5_explains={r['V5_explains_observation']}"
        )

    # ---------------------------------------------------------------
    # 5. Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel (a): SMF at z=8 and z=10
    ax = axes[0, 0]
    log_M = np.linspace(8.0, 11.0, 60)
    for z, color in [(8.5, "tab:blue"), (10.5, "tab:red")]:
        smf_l = [SMF_LambdaCDM(10 ** lm, z) for lm in log_M]
        smf_v = [SMF_V5(10 ** lm, z, alpha_P22=alpha_best) for lm in log_M]
        smf_v_nom = [SMF_V5(10 ** lm, z, alpha_P22=0.10) for lm in log_M]
        ax.plot(log_M, np.log10(smf_l), color=color, ls="--",
                label=fr"$\Lambda$CDM, $z={z:.1f}$")
        ax.plot(log_M, np.log10(smf_v_nom), color=color, ls=":",
                label=fr"V5 nominal ($\alpha=0.10$), $z={z:.1f}$")
        ax.plot(log_M, np.log10(smf_v), color=color, ls="-", lw=2,
                label=fr"V5 best ($\alpha={alpha_best:.1f}$), $z={z:.1f}$")
    # Overplot JWST data
    for z_bin, marker in [("z=8-9", "o"), ("z=10-12", "s")]:
        data = JWST_SMF_DATA[z_bin]
        Ms = [d[0] for d in data]
        phis = [d[1] for d in data]
        errs = [d[2] for d in data]
        z_label = z_bin.replace("z=", "")
        color = "tab:blue" if z_bin == "z=8-9" else "tab:red"
        ax.errorbar(Ms, phis, yerr=errs, fmt=marker, color=color,
                    capsize=3, label=f"JWST {z_label}", markersize=8,
                    markerfacecolor="white", markeredgecolor=color, mew=1.5)
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(\Phi /$ Mpc$^{-3}$ dex$^{-1})$")
    ax.set_title("(a) SMF: $\\Lambda$CDM vs V5 vs JWST")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-9.0, -2.5)
    ax.grid(alpha=0.3)

    # Panel (b): boost factor vs z
    ax = axes[0, 1]
    zs = np.linspace(0, 15, 100)
    for alpha, color, label in [
        (0.10, "tab:green", "V5 nominal, $\\alpha=0.10$"),
        (1.0, "tab:orange", "$\\alpha=1$"),
        (alpha_best, "tab:red", f"V5 best-fit, $\\alpha={alpha_best:.1f}$"),
    ]:
        boosts = [boost_P22(z, alpha_P22=alpha) for z in zs]
        ax.plot(zs, boosts, color=color, lw=2, label=label)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.axhline(1.5, color="tab:green", ls="--", alpha=0.5,
               label="V5 §11.5 target $z=10$")
    ax.set_yscale("log")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$\Phi_{\rm V5} / \Phi_{\Lambda \rm CDM}$")
    ax.set_title("(b) P22 boost factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel (c): rho_star(z) cosmic stellar mass density
    ax = axes[1, 0]
    z_grid = np.linspace(0, 14, 30)
    rho_l = [rho_star(z, "LambdaCDM") for z in z_grid]
    rho_v_nom = [rho_star(z, "V5", alpha_P22=0.10) for z in z_grid]
    rho_v_best = [rho_star(z, "V5", alpha_P22=alpha_best) for z in z_grid]
    ax.semilogy(z_grid, rho_l, "k--", lw=2, label=r"$\Lambda$CDM")
    ax.semilogy(z_grid, rho_v_nom, "tab:green", lw=1.5, ls=":",
                label=r"V5 nominal ($\alpha=0.10$)")
    ax.semilogy(z_grid, rho_v_best, "tab:red", lw=2,
                label=fr"V5 best ($\alpha={alpha_best:.1f}$)")
    # Labbe+2023 box
    z_lab = LABBE_2023["z_range"]
    ax.fill_between(
        z_lab,
        [LABBE_2023["rho_star_lower_Msun_Mpc3"]] * 2,
        [LABBE_2023["rho_star_upper_Msun_Mpc3"]] * 2,
        alpha=0.25, color="tab:purple", label="Labbe+2023 (M*>10^{10})",
    )
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$\rho_\star\ (M_\odot/$Mpc$^3)$")
    ax.set_title(r"(c) Cosmic stellar mass density $\rho_\star(z)$")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(1e0, 1e9)
    ax.grid(alpha=0.3, which="both")

    # Panel (d): chi^2 vs alpha_P22
    ax = axes[1, 1]
    ax.semilogx(a_grid, chi2_grid, "tab:red", lw=2, label=r"$\chi^2$(V5)")
    ax.axhline(chi2_lcdm, color="k", ls="--", lw=1.5,
               label=fr"$\chi^2(\Lambda CDM) = {chi2_lcdm:.1f}$")
    ax.axvline(0.10, color="tab:green", ls=":", alpha=0.7,
               label=r"V5 §11.5 nominal ($\alpha=0.10$)")
    ax.axvline(alpha_best, color="tab:purple", ls="-.", alpha=0.7,
               label=fr"Best-fit ($\alpha={alpha_best:.1f}$)")
    ax.set_xlabel(r"$\alpha_{\rm P22}$")
    ax.set_ylabel(r"$\chi^2$ over all JWST bins")
    ax.set_title(r"(d) $\chi^2$ vs $\alpha_{\rm P22}$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "TQNT-V5 §11 — JWST high-$z$ SMF: P22 boost vs $\\Lambda$CDM",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    outfile = OUTPUT_DIR / "v5_jwst_smf_comparison.png"
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {outfile}")

    # ---------------------------------------------------------------
    # 6. Return verdict
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    B_nom = bayes_nom["bayes_BIC"]
    B_best = bayes_best["bayes_BIC"]
    print(f"  V5 nominal (alpha=0.10) -- Bayes(BIC) = {B_nom:.2e}")
    print(f"  V5 best-fit            -- Bayes(BIC) = {B_best:.2e}")
    if B_nom > 10:
        v_nom = "V5 (nominal) STRONGLY preferred (B>10)"
    elif B_nom > 3:
        v_nom = "V5 (nominal) marginally preferred (B>3)"
    elif B_nom > 1/3:
        v_nom = "V5 (nominal) and LambdaCDM INDISTINGUISHABLE"
    else:
        v_nom = "V5 (nominal) DISFAVOURED (B<1/3)"
    print(f"  V5 nominal verdict     : {v_nom}")

    explained = ie_results[1]["V5_explains_observation"]  # z=8
    print(f"  Impossible early gal.  : V5 best explains? {explained}")
    return {
        "alpha_best": alpha_best,
        "chi2_lcdm": chi2_lcdm,
        "chi2_v5_nominal": bayes_nom["chi2_V5"],
        "chi2_v5_best": chi2_best,
        "bayes_nom": bayes_nom,
        "bayes_best": bayes_best,
        "impossible_early": ie_results,
    }


if __name__ == "__main__":
    main()
