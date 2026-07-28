#!/usr/bin/env python3
"""
V5.13 — Validation Planck 2018 du spectre primordial inflation χ-field.

Compare les prédictions V5 (V(φ) ∝ φ^{-q}, q = 2α_cond) aux observations
Planck 2018 + BICEP/Keck 2021 :

  - A_s prédit (calibré par construction sur Planck = 2.10e-9)
  - n_s prédit vs Planck 0.9649 ± 0.0042
  - r prédit vs BICEP/Keck upper limit < 0.06
  - Comparaison avec Starobinsky R² (référence Planck-favorite)

Produit output/v5p13_inflation_planck_validation.png (3 panneaux).

Lancement :
    PYTHONPATH=. python3 scripts/V5_13_inflation/validate_planck_inflation.py
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

from theory.inflation_chi_field import (
    A_S_PLANCK,
    ALPHA_COND_V5,
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    Q_INFLATION_V5,
    R_PLANCK_BICEP,
    V_potential_phi,
    find_psi_pivot,
    n_s_V5,
    phi_end_inflation,
    predict_inflation_V5,
    predict_starobinsky,
    r_V5,
    slow_roll_epsilon,
    slow_roll_eta,
    summary,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("V5.13 — Validation inflation topologique chi-field vs Planck 2018")
    print("=" * 70)

    # ---- Résumé prédictions V5
    pred = predict_inflation_V5(N_e=60.0)
    s = summary(N_e=60.0)
    staro = predict_starobinsky(N_e=60.0)

    print(f"\nParamètres dérivés V5 :")
    print(f"  alpha_cond  = {pred.alpha_cond:.4f}  (P18 / V5 §13.6.4)")
    print(f"  q = 2 alpha_cond = {pred.q:.4f}    (exposant V phi^{{-q}})")
    print(f"  N_e (pivot) = {pred.N_e:.1f}")
    print(f"  phi_end     = {pred.phi_end:.4f} M_pl   (epsilon = 1)")
    print(f"  phi_star    = {pred.phi_star:.4f} M_pl  (horizon-crossing)")
    print(f"  epsilon_*   = {pred.epsilon:.5f}")
    print(f"  eta_*       = {pred.eta:.5f}")

    print(f"\nObservables CMB V5 :")
    print(f"  A_s  V5 (calibré) = {pred.A_s:.3e}")
    print(f"  A_s  Planck       = {A_S_PLANCK:.3e}     [V_0 paramètre libre]")
    print(f"  n_s  V5           = {pred.n_s:.4f}")
    print(f"  n_s  Planck       = {N_S_PLANCK:.4f} +/- {N_S_PLANCK_SIGMA:.4f}")
    print(f"  ecart n_s         = {pred.n_s_deviation_sigma:+.2f} sigma")
    print(f"  r    V5           = {pred.r:.4f}")
    print(f"  r    upper BICEP  = {R_PLANCK_BICEP:.3f}  -> {'OK' if pred.r_passes_upper else 'TENSION'}")

    print(f"\nRéférence Starobinsky R² (modèle Planck-favori) :")
    print(f"  n_s = {staro['n_s']:.4f}, r = {staro['r']:.5f}")
    print(f"  compatible n_s : {staro['compatible_planck_n_s']}")
    print(f"  compatible r   : {staro['compatible_BICEP_r']}")

    # ---- Balayage N_e ∈ [40, 70]
    Ns = np.linspace(40.0, 70.0, 61)
    n_s_arr = np.array([n_s_V5(N_e=N) for N in Ns])
    r_arr = np.array([r_V5(N_e=N) for N in Ns])
    n_s_staro = 1.0 - 2.0 / Ns
    r_staro = 12.0 / Ns ** 2

    # ---- Potentiel
    phi_grid = np.linspace(0.5, 25.0, 400)
    V_pl = V_potential_phi(phi_grid, phi_0=1.0, V0=1.0)
    V_pl_norm = V_pl / V_pl[np.argmin(np.abs(phi_grid - pred.phi_star))]
    eps_grid = slow_roll_epsilon(phi_grid)
    et_grid = slow_roll_eta(phi_grid)

    # =====================================================================
    # FIGURE 3 panneaux
    # =====================================================================
    fig = plt.figure(figsize=(15, 11))

    # Panneau 1 : potentiel et slow-roll
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(phi_grid, V_pl_norm, "b-", lw=2.0, label=r"$V(\phi) \propto \phi^{-q}$, $q=2\alpha_{\rm cond}$")
    ax1.axvline(pred.phi_end, color="r", ls=":", lw=1.5,
                label=fr"$\phi_{{end}} = {pred.phi_end:.2f}\,M_{{pl}}$")
    ax1.axvline(pred.phi_star, color="g", ls="--", lw=1.5,
                label=fr"$\phi_\star = {pred.phi_star:.2f}\,M_{{pl}}$ ($N_e=60$)")
    ax1.set_xlabel(r"$\phi$ [$M_{pl}$]")
    ax1.set_ylabel(r"$V(\phi)/V(\phi_\star)$")
    ax1.set_yscale("log")
    ax1.set_xlim(0.5, 25)
    ax1.set_title("Potentiel topologique V5.13")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panneau 2 : epsilon, eta
    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogy(phi_grid, eps_grid, "b-", lw=2, label=r"$\epsilon(\phi) = q^2/(2\phi^2)$")
    ax2.semilogy(phi_grid, et_grid, "r-", lw=2, label=r"$\eta(\phi) = q(q+1)/\phi^2$")
    ax2.axhline(1.0, color="k", ls=":", lw=1, label=r"$\epsilon = 1$ fin")
    ax2.axvline(pred.phi_end, color="grey", ls=":", lw=1)
    ax2.axvline(pred.phi_star, color="g", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel(r"$\phi$ [$M_{pl}$]")
    ax2.set_ylabel("slow-roll")
    ax2.set_xlim(0.5, 25)
    ax2.set_ylim(1e-3, 5)
    ax2.set_title(r"Paramètres slow-roll $\epsilon, \eta$")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    # Panneau 3 : n_s vs N_e (V5 vs Starobinsky vs Planck)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(Ns, n_s_arr, "b-", lw=2.0, label=r"V5.13 $V \propto \phi^{-q}$")
    ax3.plot(Ns, n_s_staro, "g--", lw=2.0, label=r"Starobinsky $R^2$ (réf)")
    ax3.axhspan(N_S_PLANCK - 3 * N_S_PLANCK_SIGMA,
                N_S_PLANCK + 3 * N_S_PLANCK_SIGMA,
                color="red", alpha=0.15,
                label=fr"Planck $3\sigma$ ({N_S_PLANCK:.4f}$\pm${3*N_S_PLANCK_SIGMA:.4f})")
    ax3.axhline(N_S_PLANCK, color="red", ls="-", lw=1.5, alpha=0.7)
    ax3.axvline(60.0, color="grey", ls=":", lw=1.5, label=r"$N_e = 60$ (pivot)")
    ax3.set_xlabel(r"$N_e$ (e-folds)")
    ax3.set_ylabel(r"$n_s$")
    ax3.set_title(r"Tilt scalaire $n_s$ : V5 vs Starobinsky vs Planck 2018")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(40, 70)
    ax3.set_ylim(0.94, 1.005)

    # Panneau 4 : r vs N_e (V5 vs Starobinsky vs BICEP upper)
    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogy(Ns, r_arr, "b-", lw=2.0, label=r"V5.13 $V \propto \phi^{-q}$")
    ax4.semilogy(Ns, r_staro, "g--", lw=2.0, label=r"Starobinsky $R^2$ (réf)")
    ax4.axhline(R_PLANCK_BICEP, color="red", ls="-", lw=1.5,
                label=fr"BICEP/Keck upper $r<{R_PLANCK_BICEP:.2f}$")
    ax4.axhspan(R_PLANCK_BICEP, 1.0, color="red", alpha=0.15)
    ax4.axvline(60.0, color="grey", ls=":", lw=1.5)
    ax4.set_xlabel(r"$N_e$ (e-folds)")
    ax4.set_ylabel(r"$r$ (tensor-to-scalar)")
    ax4.set_title(r"Ratio $r$ : V5 (tension) vs Starobinsky (OK)")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(alpha=0.3, which="both")
    ax4.set_xlim(40, 70)
    ax4.set_ylim(1e-3, 1)

    fig.suptitle(
        f"V5.13 — Inflation topologique chi-field vs Planck 2018 + BICEP/Keck\n"
        f"alpha_cond = {ALPHA_COND_V5:.4f}, q = {Q_INFLATION_V5:.4f}, "
        f"N_e = 60 -> n_s_V5 = {pred.n_s:.4f}, r_V5 = {pred.r:.3f}",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "v5p13_inflation_planck_validation.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nFigure : {out_path}")

    print("\n" + "=" * 70)
    print("VERDICT V5.13 (loi-de-puissance pure)")
    print("=" * 70)
    print(f"  A_s  : 2.10e-9 reproduit par calibration V_0 (paramètre libre)")
    print(f"  n_s  : prédit {pred.n_s:.4f} vs Planck {N_S_PLANCK:.4f}  -> "
          f"TENSION ({pred.n_s_deviation_sigma:+.1f} sigma)")
    print(f"  r    : prédit {pred.r:.3f} vs BICEP < {R_PLANCK_BICEP:.2f}    -> "
          f"VIOLATION (facteur {pred.r/R_PLANCK_BICEP:.1f}x)")
    print()
    print("  -> La V5 pure-loi-de-puissance est en tension forte avec Planck/BICEP.")
    print("     Voir docs/V5p13_topological_inflation.tex pour discussion :")
    print("     une correction Chern-Simons R²-like à bas champ est requise.")


if __name__ == "__main__":
    main()
