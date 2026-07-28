#!/usr/bin/env python3
"""
V5.11 — Tension de Hubble : proposition P22 cosmologique.

Compare la valeur CMB H_0 = 67.4 km/s/Mpc (Planck 2018) à la valeur
locale H_0 = 73.04 km/s/Mpc (SH0ES 2022) et propose une résolution
via la nucléation topologique P22 à la frontière matière/vide
cosmologique.

Modèle :
    H_local = H_CMB × (1 + η · ξ_local)
    ξ_local = gradient effectif de Ω² au bord local

Produit output/v5p11_hubble_tension.png

Lancement :
    PYTHONPATH=. python3 scripts/V5_11_cosmo/hubble_tension_resolution.py
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

from theory.flrw_emergence import (
    H0_PLANCK_KM_S_MPC,
    H0_SH0ES_KM_S_MPC,
    hubble_tension_P22,
)


def main():
    print("=" * 72)
    print(" V5.11 — Tension de Hubble : résolution P22 ?")
    print("=" * 72)

    # Cas par défaut : η = 1, ξ_local = 0.080
    base = hubble_tension_P22(eta=1.0, xi_local=0.080)
    print(f"\n  H₀ CMB (Planck 2018)         = {base.H0_CMB_km_s_Mpc:.2f} km/s/Mpc")
    print(f"  H₀ local (SH0ES 2022)        = {base.H0_local_km_s_Mpc:.2f} km/s/Mpc")
    print(f"  Δ H₀ observée                = {base.delta_H0_observed_km_s_Mpc:+.2f} km/s/Mpc")
    print(f"  Δ H₀ observée                = {base.delta_H0_observed_pct:+.2f} %")
    print(f"\n  η utilisé                    = {base.eta_used:.3f}")
    print(f"  ξ_local utilisé              = {base.xi_local_used:.4f}")
    print(f"  Δ H₀ prédit par P22          = {base.delta_H0_P22_predicted_km_s_Mpc:+.2f} km/s/Mpc")
    print(f"  Δ H₀ prédit par P22          = {base.delta_H0_P22_predicted_pct:+.2f} %")
    print(f"  Fraction résolue par P22     = {base.fraction_resolved*100:.1f} %")
    print(f"\n  Note : {base.note}")

    # Scan ξ_local
    print(f"\n  --- Sensibilité (η = 1) ---")
    xi_arr = np.linspace(0.0, 0.20, 9)
    for xi in xi_arr:
        t = hubble_tension_P22(eta=1.0, xi_local=xi)
        print(f"  ξ = {xi:.3f} → H_local = "
               f"{t.H0_CMB_km_s_Mpc * (1 + t.delta_H0_P22_predicted_pct/100):.2f} "
               f"km/s/Mpc, fraction = {t.fraction_resolved:.2f}")

    # Plot : H_local vs ξ_local
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_png = output_dir / "v5p11_hubble_tension.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 : H_local en fonction de ξ pour quelques η
    ax = axes[0]
    xi_grid = np.linspace(0.0, 0.20, 200)
    for eta in [0.5, 1.0, 1.5]:
        H_pred = np.array([
            H0_PLANCK_KM_S_MPC * (1 + hubble_tension_P22(eta, xi)
                                        .delta_H0_P22_predicted_pct/100)
            for xi in xi_grid
        ])
        ax.plot(xi_grid, H_pred, lw=2, label=fr"$\eta={eta}$")
    ax.axhline(H0_PLANCK_KM_S_MPC, color="black", linestyle="--", lw=1,
                label=f"H₀ Planck = {H0_PLANCK_KM_S_MPC}")
    ax.axhline(H0_SH0ES_KM_S_MPC, color="red", linestyle="--", lw=1,
                label=f"H₀ SH0ES = {H0_SH0ES_KM_S_MPC}")
    ax.fill_between(xi_grid, H0_SH0ES_KM_S_MPC - 1.04,
                     H0_SH0ES_KM_S_MPC + 1.04,
                     color="red", alpha=0.15)
    ax.set_xlabel(r"$\xi_{local}$ (gradient $\Omega^2$ matière/vide)")
    ax.set_ylabel(r"$H_0$ local prédit (km/s/Mpc)")
    ax.set_title("V5.11 — Tension Hubble via P22 cosmologique")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 : fraction résolue
    ax = axes[1]
    for eta in [0.5, 1.0, 1.5]:
        frac = np.array([
            hubble_tension_P22(eta, xi).fraction_resolved
            for xi in xi_grid
        ])
        ax.plot(xi_grid, 100*frac, lw=2, label=fr"$\eta={eta}$")
    ax.axhline(100.0, color="green", linestyle="--", lw=1, label="100 %")
    ax.set_xlabel(r"$\xi_{local}$")
    ax.set_ylabel("Fraction de la tension résolue (%)")
    ax.set_title("V5.11 — Fraction de la tension expliquée")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot sauvegardé : {out_png}")
    print("=" * 72)


if __name__ == "__main__":
    main()
