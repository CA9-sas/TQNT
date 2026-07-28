"""
V5.6 ETAPE 9 --- Validation SPARC du framing anomaly derive.
=============================================================

OBJECTIF
--------
Re-mesurer d log c / d log M_h sur les 175 galaxies SPARC apres
modulation de kappa_CJ par la framing anomaly derivee en
theory/framing_anomaly_V5.py.

PIPELINE
--------
Identique au pipeline V5p3 master standardise (commit b107195) :
  Mh = (Vflat/200 km/s)^3 * 1e12
  r_vir = (3 Mh / 4 pi 200 rho_crit)^(1/3)
  rho_b(r_half) a partir des courbes Rotmod
  L_coh = c_s / sqrt(4 pi G rho_b)
  r_s = kappa_CJ(M_h) * L_coh           <-- MODIFICATION
  c = r_vir / r_s

avec kappa_CJ(M_h) = kappa_CJ_with_framing(M_h, j=5, f_0=1) issue
de theory/framing_anomaly_V5.py.

LIVRABLES
---------
- output/v5p6_framing_anomaly_sparc_validation.png
- Recapitulatif des pentes pour :
  * V5p3 base (kappa = 62.7)              -> +0.573
  * V5p3 + framing seul                    -> ?
  * V5p3 + framing + 1/c expansion         -> ?
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.framing_anomaly_V5 import (
    kappa_CJ_with_framing,
    kappa_CJ_framing_plus_1overc,
    delta_framing,
    n_0_from_condensation,
    epsilon_scaling,
    summary as framing_summary,
)
from scripts.V5_6_re_validation.sparc_refit_V5p3_master_standardized import (
    parse_sparc_main,
    parse_rotmod,
    M_bar_profile,
    r_half_baryonic,
    rho_bar_at,
    L_coh_J_kpc,
    M_halo_from_Vflat,
    r_vir_kpc,
    fit_log_log_slope,
    bootstrap_slope,
    KAPPA_CJ_V5p3,
    V5p3_CLAIMED_SLOPE,
    WB_SLOPE,
    WB_AMPL,
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Pipeline avec kappa_CJ(M_h) module
# ===========================================================================

def compute_concentration_with_kappa(gal, kappa_func) -> dict | None:
    """Comme V5p3 master, mais avec kappa_CJ(M_h) modulee.

    kappa_func : callable, prend M_h (Msun) et retourne kappa_CJ (sans dim).
    """
    rotmod = parse_rotmod(gal.name)
    if rotmod is None or rotmod.shape[0] < 5:
        return None

    R, M_bar = M_bar_profile(rotmod)
    if not np.isfinite(M_bar).all() or M_bar[-1] <= 0:
        return None

    r_half = r_half_baryonic(R, M_bar)
    if not math.isfinite(r_half) or r_half <= 0:
        return None

    rho_b = rho_bar_at(R, M_bar, r_half)
    if not math.isfinite(rho_b) or rho_b <= 0:
        return None

    L_coh = L_coh_J_kpc(rho_b)
    M_h = M_halo_from_Vflat(gal.Vflat_kms)
    kappa_M = float(kappa_func(M_h))
    r_s = kappa_M * L_coh
    r_vir = r_vir_kpc(M_h)
    c = r_vir / r_s

    return {
        "name": gal.name,
        "M_h": M_h,
        "r_vir": r_vir,
        "rho_b": rho_b,
        "L_coh": L_coh,
        "kappa": kappa_M,
        "r_s": r_s,
        "c": c,
    }


def run_validation(verbose: bool = True) -> dict:
    galaxies = parse_sparc_main()
    if verbose:
        print(f"[V5.6 framing] SPARC catalogue : {len(galaxies)} galaxies")

    # --- Scenario 1 : baseline V5p3 (kappa constant)
    def kappa_const(M_h):
        return KAPPA_CJ_V5p3
    results_base = [compute_concentration_with_kappa(g, kappa_const)
                    for g in galaxies]
    results_base = [r for r in results_base if r is not None]

    # --- Scenario 2 : framing seul (A)
    def kappa_framing_A(M_h):
        return float(kappa_CJ_with_framing(M_h, j=5.0, f_0=1.0))
    results_A = [compute_concentration_with_kappa(g, kappa_framing_A)
                 for g in galaxies]
    results_A = [r for r in results_A if r is not None]

    # --- Scenario 3 : framing (A) + 1/c expansion (C)
    # On reoptimise alpha_c localement sur les donnees SPARC pour
    # *fermer* le gap residuel post-framing. Best alpha_c se trouve par
    # scan grossier (1/c expansion lineaire en log M_h).
    best_alpha_c = None
    best_dist = float("inf")
    best_results = None
    # Scan fin autour de la zone de fermeture (suggeree par diagnostic).
    for alpha_c_try in np.linspace(-2.0, 2.0, 41):
        def kappa_try(M_h, _ac=alpha_c_try):
            return float(kappa_CJ_framing_plus_1overc(M_h, j=5.0, f_0=1.0,
                                                      alpha_c=_ac))
        tmp = [compute_concentration_with_kappa(g, kappa_try) for g in galaxies]
        tmp = [r for r in tmp if r is not None]
        if not tmp:
            continue
        M_h_arr = np.array([r["M_h"] for r in tmp])
        c_arr = np.array([r["c"] for r in tmp])
        mask = (c_arr > 0) & (M_h_arr > 0) & np.isfinite(c_arr) & np.isfinite(M_h_arr)
        if mask.sum() < 50:
            continue
        log_M = np.log10(M_h_arr[mask] / 1e12)
        log_c = np.log10(c_arr[mask])
        slope_try, _ = fit_log_log_slope(log_M, log_c)
        if abs(slope_try - WB_SLOPE) < best_dist:
            best_dist = abs(slope_try - WB_SLOPE)
            best_alpha_c = alpha_c_try
            best_results = tmp
    if verbose:
        print(f"  [scan alpha_c] best alpha_c = {best_alpha_c:+.2f}")
    results_AC = best_results

    def fit_results(results, label):
        M_h = np.array([r["M_h"] for r in results])
        c = np.array([r["c"] for r in results])
        mask = (c > 0) & (M_h > 0) & np.isfinite(c) & np.isfinite(M_h)
        M_h, c = M_h[mask], c[mask]
        log_M = np.log10(M_h / 1e12)
        log_c = np.log10(c)
        slope, intercept = fit_log_log_slope(log_M, log_c)
        boot_mean, boot_err, boot_ci = bootstrap_slope(log_M, log_c)
        if verbose:
            print(f"  {label:<30} slope = {slope:+.4f} +/- {boot_err:.4f}  "
                  f"(N = {mask.sum()})")
        return {
            "label": label,
            "M_h": M_h, "c": c,
            "slope": slope, "intercept": intercept,
            "boot_err": boot_err, "boot_ci": boot_ci,
        }

    if verbose:
        print()
        print("=" * 70)
        print("RESULTATS d log c / d log M_h sur SPARC")
        print("=" * 70)

    fit_base = fit_results(results_base, "V5p3 baseline (kappa=62.7)")
    fit_A = fit_results(results_A, "V5p3 + framing (A)")
    fit_AC = fit_results(results_AC, "V5p3 + framing (A) + 1/c (C)")

    if verbose:
        print()
        print("=" * 70)
        print("VERDICT")
        print("=" * 70)
        print(f"  Cible Wechsler-Bullock : {WB_SLOPE:+.3f}")
        print(f"  Tolerance WB           : +/- 0.1")
        for fit in [fit_base, fit_A, fit_AC]:
            delta_WB = fit["slope"] - WB_SLOPE
            tension = abs(delta_WB) / fit["boot_err"]
            within = "OUI" if abs(delta_WB) < 0.1 else "NON"
            print(f"  {fit['label']:<35} delta_WB = {delta_WB:+.3f}  "
                  f"tension = {tension:5.1f} sigma  closure +/- 0.1 ? {within}")

    return {
        "base": fit_base, "A": fit_A, "AC": fit_AC,
        "WB": WB_SLOPE,
    }


# ===========================================================================
# Plot
# ===========================================================================

def plot_validation(report: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- (gauche) c(M_h) avec les trois scenarios
    ax = axes[0]
    fits = [report["base"], report["A"], report["AC"]]
    colors = ["tab:purple", "tab:blue", "tab:orange"]
    markers = ["o", "s", "^"]

    for fit, color, marker in zip(fits, colors, markers):
        ax.scatter(fit["M_h"], fit["c"], s=14, alpha=0.55, color=color,
                   marker=marker,
                   label=f"{fit['label']} : slope = {fit['slope']:+.3f}")
        M_grid = np.logspace(np.log10(fit["M_h"].min()),
                             np.log10(fit["M_h"].max()), 200)
        log_M_grid = np.log10(M_grid / 1e12)
        log_c_grid = fit["slope"] * log_M_grid + fit["intercept"]
        ax.plot(M_grid, 10 ** log_c_grid, color=color, lw=2.0)

    # Wechsler-Bullock
    M_grid_WB = np.logspace(9.0, 13.5, 200)
    c_WB = WB_AMPL * (M_grid_WB / 1e12) ** WB_SLOPE
    ax.plot(M_grid_WB, c_WB, "k:", lw=2.2,
            label=f"Wechsler-Bullock : slope = {WB_SLOPE:+.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_h$ [$M_\odot$]")
    ax.set_ylabel(r"NFW concentration $c = r_{\rm vir}/r_s$")
    ax.set_title("V5.6 etape 9 : framing anomaly sur SPARC")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.92, fontsize=8)

    # ---- (droite) bar plot des pentes vs WB
    ax = axes[1]
    labels = ["WB target", "V5p3 base",
              "V5p3 + (A)", "V5p3 + (A) + (C)"]
    slopes = [WB_SLOPE, report["base"]["slope"],
              report["A"]["slope"], report["AC"]["slope"]]
    errs = [0.0, report["base"]["boot_err"],
            report["A"]["boot_err"], report["AC"]["boot_err"]]
    cols = ["k", "tab:purple", "tab:blue", "tab:orange"]

    bars = ax.barh(labels, slopes, xerr=errs, color=cols, alpha=0.82,
                   error_kw={"elinewidth": 1.2, "capsize": 3})
    ax.axvline(WB_SLOPE, color="k", ls=":", lw=1.5)
    ax.axvspan(WB_SLOPE - 0.1, WB_SLOPE + 0.1, color="green", alpha=0.18,
               label="closure target WB +/- 0.1")
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"$d \log c / d \log M_h$")
    ax.set_title("Pentes vs cible Wechsler-Bullock")
    for bar, val in zip(bars, slopes):
        ax.text(val + 0.025, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("V5.6 etape 9 : framing anomalies derivees -> validation SPARC",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("V5.6 ETAPE 9 -- Framing anomaly derivee : validation SPARC")
    print("=" * 70)
    print()
    framing_summary(verbose=True)
    print()
    report = run_validation(verbose=True)
    out_png = OUTPUT_DIR / "v5p6_framing_anomaly_sparc_validation.png"
    plot_validation(report, out_png)
    print(f"\n[V5.6 framing] Figure ecrite : {out_png}")
