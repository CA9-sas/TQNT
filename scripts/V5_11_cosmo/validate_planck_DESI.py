#!/usr/bin/env python3
"""
V5.11 — Validation Planck 2018 + DESI 2024.

Calcule depuis V5.11 :
  - H_0, Ω_m, Ω_Λ, ρ_crit
  - Ω_DM/Ω_b depuis α = 1,1991 (P18)
  - D_M(z=0.5)/r_s pour DESI DR1

Produit output/v5p11_planck_DESI_validation.png (3 panneaux).

Lancement :
    PYTHONPATH=. python3 scripts/V5_11_cosmo/validate_planck_DESI.py
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

from theory.flrw_emergence import (
    ALPHA_CONDENSATION,
    DM_OVER_RS_Z05_DESI,
    H0_PLANCK_KM_S_MPC,
    H0_SH0ES_KM_S_MPC,
    OMEGA_B_OVER_M_PLANCK,
    OMEGA_B_PLANCK,
    OMEGA_L_PLANCK,
    OMEGA_M_PLANCK,
    R_DRAG_MPC,
    SIGMA8_PLANCK,
    comoving_distance_flat,
    cosmo_state_V5,
    omega_dm_over_b_from_alpha,
    validate_desi_2024,
    validate_planck_2018,
)


def main():
    print("=" * 72)
    print(" V5.11 — Validation Planck 2018 + DESI 2024")
    print("=" * 72)

    planck = validate_planck_2018()
    state = cosmo_state_V5()
    desi = validate_desi_2024(state)

    # Sortie console
    print(f"\n  G_V5                       = {planck.G_V5:.5e} SI")
    print(f"  G_CODATA                   = {planck.G_codata:.5e} SI")
    print(f"\n  H0 V5 (= input Planck)     = {planck.H0_V5_km_s_Mpc:.3f} km/s/Mpc")
    print(f"  H0 Planck 2018             = {planck.H0_planck_km_s_Mpc:.3f} km/s/Mpc")
    print(f"  ρ_crit V5                  = {planck.rho_crit_V5:.4e} kg/m³")
    print(f"  ρ_m V5                     = {planck.rho_m_V5:.4e} kg/m³")
    print(f"  ρ_b V5                     = {planck.rho_b_V5:.4e} kg/m³")
    print(f"  ρ_DM V5                    = {planck.rho_dm_V5:.4e} kg/m³")
    print(f"  ρ_Λ V5                     = {planck.rho_lambda_V5:.4e} kg/m³")
    print(f"\n  Ω_b/Ω_m V5                 = {planck.Omega_b_over_m_V5:.4f}")
    print(f"  Ω_b/Ω_m Planck             = {planck.Omega_b_over_m_target:.4f}")
    print(f"\n  Ω_DM/Ω_b V5                = {planck.Omega_dm_over_b_V5:.3f}")
    print(f"  Ω_DM/Ω_b Planck            = {planck.Omega_dm_over_b_target:.3f}")

    # P18 cohérence
    # α = 1.20 ; nombre effectif de croisements pour reproduire 5.45 :
    n_eff = np.log(planck.Omega_dm_over_b_target) / np.log(ALPHA_CONDENSATION)
    print(f"\n  α (P18)                    = {ALPHA_CONDENSATION:.4f}")
    print(f"  n_eff (croisements)        = {n_eff:.2f}")
    print(f"  Ω_DM/Ω_b depuis α^n_eff    = {omega_dm_over_b_from_alpha(ALPHA_CONDENSATION, n_eff):.3f}")

    print(f"\n  --- DESI BAO 2024 ---")
    print(f"  r_drag (Mpc)               = {desi.r_drag_used_Mpc:.2f}")
    print(f"  D_M(z=0.5)/r_s V5          = {desi.DM_over_rs_z05_V5:.3f}")
    print(f"  D_M(z=0.5)/r_s DESI        = {desi.DM_over_rs_z05_target:.3f}")
    print(f"  Écart                      = {desi.DM_over_rs_z05_relative_error*100:+.2f} %")
    print(f"  Compatible Λ ?             = {desi.consistent_with_lambda}")

    # Plot
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_png = output_dir / "v5p11_planck_DESI_validation.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Densités cosmologiques
    ax = axes[0]
    labels = ["ρ_b", "ρ_DM", "ρ_Λ"]
    values = [planck.rho_b_V5, planck.rho_dm_V5, planck.rho_lambda_V5]
    colors = ["#377eb8", "#984ea3", "#e41a1c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.axhline(planck.rho_crit_V5, color="black", linestyle="--", lw=1,
               label=f"ρ_crit = {planck.rho_crit_V5:.2e} kg/m³")
    ax.set_ylabel(r"Densité [kg/m³]")
    ax.set_title("V5.11 — Densités cosmologiques (Planck 2018)")
    ax.legend()
    ax.set_yscale("linear")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.05,
                f"{val:.2e}", ha="center", fontsize=8)

    # Panel 2: D_M(z)/r_s vs DESI
    ax = axes[1]
    z_arr = np.linspace(0.05, 1.2, 60)
    dm_over_rs = np.array([comoving_distance_flat(z, state) / R_DRAG_MPC
                            for z in z_arr])
    ax.plot(z_arr, dm_over_rs, lw=2, color="#377eb8",
            label="V5.11 (ΛCDM, H₀=67.4)")
    ax.errorbar([0.5], [DM_OVER_RS_Z05_DESI], yerr=[0.16],
                fmt="o", color="#e41a1c", markersize=10,
                label=f"DESI DR1 z=0.5 : {DM_OVER_RS_Z05_DESI:.2f}±0.16")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$D_M(z) / r_s$")
    ax.set_title("V5.11 — DESI BAO 2024")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Ω_DM/Ω_b vs α^n
    ax = axes[2]
    n_arr = np.linspace(0, 15, 200)
    pred = ALPHA_CONDENSATION ** n_arr
    ax.plot(n_arr, pred, lw=2, color="#984ea3",
            label=fr"$\alpha^n$, $\alpha={ALPHA_CONDENSATION}$ (P18)")
    ax.axhline(planck.Omega_dm_over_b_target, color="#e41a1c",
                linestyle="--", lw=1.5,
                label=f"Ω_DM/Ω_b Planck = {planck.Omega_dm_over_b_target:.2f}")
    ax.axvline(n_eff, color="black", linestyle=":", lw=1,
                label=f"n_eff = {n_eff:.2f}")
    ax.set_xlabel("n (croisements cinétiques)")
    ax.set_ylabel(r"$\Omega_{DM}/\Omega_b$ prédit")
    ax.set_title("V5.11 — P18 condensation Big Bang")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot sauvegardé : {out_png}")
    print("=" * 72)


if __name__ == "__main__":
    main()
