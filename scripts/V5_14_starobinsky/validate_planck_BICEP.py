#!/usr/bin/env python3
"""
V5.14 — Validation Planck 2018 + BICEP/Keck 2021 du potentiel Starobinsky + R².

Montre que la correction R² type Starobinsky ajoutée au potentiel V5.13
(résidu topologique multiplicatif) ramène n_s et r dans la fenêtre permise
par Planck + BICEP.

Comparatif :
  V5.13 (loi de puissance pure)   : n_s = 0.9967 (+7.6σ), r = 0.158  → FALSIFIÉ
  V5.14 (Starobinsky + résidu)    : n_s ≈ 0.968 (~0.7σ),  r ≈ 0.003  → COMPATIBLE
  Starobinsky standard (référence) : n_s = 0.967,         r = 0.0033 → COMPATIBLE

Produit output/v5p14_starobinsky_planck_BICEP.png (4 panneaux).

Lancement :
    PYTHONPATH=. python3 scripts/V5_14_starobinsky/validate_planck_BICEP.py
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

from theory.inflation_chi_field import (
    A_S_PLANCK,
    ALPHA_COND_V5,
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    Q_INFLATION_V5,
    R_PLANCK_BICEP,
    V_potential_phi,
    n_s_V5,
    r_V5,
)
from theory.inflation_starobinsky_R2 import (
    BETA_TOP_DEFAULT,
    EPS_TOP_DEFAULT,
    N_S_STAROBINSKY_NE60,
    PHI_MAX_V5p13,
    R_STAROBINSKY_NE60,
    V_starobinsky_R2,
    V_total_V5p14,
    n_s_V5p14,
    phi_end_inflation_V5p14,
    predict_inflation_V5p14,
    r_V5p14,
    slow_roll_epsilon_V5p14,
    slow_roll_eta_V5p14,
    summary_V5p14,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("V5.14 — Validation Starobinsky R² + résidu V5.13 vs Planck/BICEP")
    print("=" * 72)

    # ---- Prédictions principales V5.14
    eps_top = EPS_TOP_DEFAULT
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=eps_top)
    s = summary_V5p14(N_e=60.0, eps_top=eps_top)

    # ---- Limite Starobinsky pur (ε_top = 0)
    pred_star = predict_inflation_V5p14(N_e=60.0, eps_top=0.0)

    # ---- V5.13 historique
    n_s_v5p13 = n_s_V5(N_e=60.0)
    r_v5p13 = r_V5(N_e=60.0)

    print()
    print(f"Paramètres V5.14 :")
    print(f"  ε_top      = {eps_top:.1e}     (fraction résidu topologique)")
    print(f"  β          = {BETA_TOP_DEFAULT:.4f}    (exposant résidu)")
    print(f"  φ_max      = {PHI_MAX_V5p13:.3f} M_pl  (= φ_★ V5.13 à N=60)")
    print()
    print(f"Géométrie slow-roll V5.14 :")
    print(f"  φ_end      = {pred.phi_end:.4f} M_pl   (ε = 1)")
    print(f"  φ_★        = {pred.phi_star:.4f} M_pl   (horizon-crossing N=60)")
    print(f"  ε_★        = {pred.epsilon:.4e}")
    print(f"  η_★        = {pred.eta:.4e}")
    print()
    print(f"Observables CMB V5.14 :")
    print(f"  A_s V5.14 (calibré V_0)   = {pred.A_s:.3e}")
    print(f"  V_0/M_pl⁴ (calibré)        = {pred.V0_over_Mpl4:.3e}")
    print(f"  n_s V5.14                  = {pred.n_s:.5f}")
    print(f"  n_s Planck                 = {N_S_PLANCK:.4f} ± {N_S_PLANCK_SIGMA:.4f}")
    print(f"  écart n_s                  = {pred.n_s_deviation_sigma:+.2f} σ")
    print(f"  compatible Planck (<3σ)   : {pred.compatible_planck_n_s}")
    print(f"  r V5.14                    = {pred.r:.5f}")
    print(f"  r BICEP upper              = {R_PLANCK_BICEP:.3f}")
    print(f"  compatible BICEP (r<0.06) : {pred.compatible_BICEP_r}")

    print()
    print(f"Référence — Starobinsky pur (ε_top = 0) :")
    print(f"  n_s = {pred_star.n_s:.5f},  r = {pred_star.r:.5f}")
    print(f"  cible théorique Starobinsky (N=60) :")
    print(f"    n_s = 1 - 2/N_e   = {N_S_STAROBINSKY_NE60:.5f}")
    print(f"    r   = 12/N_e²     = {R_STAROBINSKY_NE60:.5f}")

    print()
    print(f"Rappel — V5.13 historique (loi de puissance pure) :")
    print(f"  n_s V5.13 = {n_s_v5p13:.4f}  → écart Planck "
          f"{(n_s_v5p13-N_S_PLANCK)/N_S_PLANCK_SIGMA:+.1f} σ")
    print(f"  r   V5.13 = {r_v5p13:.4f}  → "
          f"{'OK' if r_v5p13 < R_PLANCK_BICEP else 'VIOLE BICEP ×' + f'{r_v5p13/R_PLANCK_BICEP:.1f}'}")

    # ---- Balayage N_e ∈ [40, 70]
    Ns = np.linspace(40.0, 70.0, 31)
    n_s_v5p14_arr = np.array([n_s_V5p14(N_e=N, eps_top=eps_top) for N in Ns])
    r_v5p14_arr = np.array([r_V5p14(N_e=N, eps_top=eps_top) for N in Ns])
    n_s_v5p13_arr = np.array([n_s_V5(N_e=N) for N in Ns])
    r_v5p13_arr = np.array([r_V5(N_e=N) for N in Ns])
    n_s_staro = 1.0 - 2.0 / Ns
    r_staro = 12.0 / Ns ** 2

    # ---- Potentiels V5.13, V5.14, Starobinsky pur
    phi_grid = np.linspace(0.1, 25.0, 400)
    V_v5p13 = V_potential_phi(phi_grid, phi_0=1.0, V0=1.0,
                              alpha_cond=ALPHA_COND_V5)
    V_star = V_starobinsky_R2(phi_grid, V0=1.0)
    V_v5p14 = V_total_V5p14(phi_grid, V0=1.0, eps_top=eps_top)
    # Normalisation à la même valeur au pivot V5.14 pour comparaison visuelle
    pivot_idx = np.argmin(np.abs(phi_grid - pred.phi_star))
    V_v5p13_norm = V_v5p13 / V_v5p13[pivot_idx]
    V_star_norm = V_star / V_star[pivot_idx]
    V_v5p14_norm = V_v5p14 / V_v5p14[pivot_idx]

    # ---- Balayage ε_top pour sensibilité
    eps_top_grid = np.logspace(-5, -0.5, 30)
    n_s_vs_eps = np.array([n_s_V5p14(N_e=60.0, eps_top=e) for e in eps_top_grid])
    r_vs_eps = np.array([r_V5p14(N_e=60.0, eps_top=e) for e in eps_top_grid])

    # =====================================================================
    # FIGURE 4 panneaux
    # =====================================================================
    fig = plt.figure(figsize=(15, 11))

    # --- Panneau 1 : Potentiels comparés
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(phi_grid, V_v5p13_norm, "b-", lw=2.0,
             label=r"V5.13 $V\propto\phi^{-q}$ (FALSIFIÉ)")
    ax1.plot(phi_grid, V_star_norm, "g--", lw=2.0,
             label=r"Starobinsky pur $V_*\propto(1-e^{-c\phi})^2$")
    ax1.plot(phi_grid, V_v5p14_norm, "r-", lw=2.2,
             label=fr"V5.14 ($\epsilon_{{\rm top}}={eps_top:.0e}$)")
    ax1.axvline(pred.phi_star, color="k", ls=":", lw=1.2,
                label=fr"$\phi_\star = {pred.phi_star:.2f}\,M_{{pl}}$")
    ax1.axvline(pred.phi_end, color="grey", ls=":", lw=1.0,
                label=fr"$\phi_{{end}} = {pred.phi_end:.2f}\,M_{{pl}}$")
    ax1.set_xlabel(r"$\phi$ [$M_{pl}$]")
    ax1.set_ylabel(r"$V(\phi)/V(\phi_\star)$")
    ax1.set_yscale("log")
    ax1.set_xlim(0.5, 25)
    ax1.set_ylim(1e-3, 1e2)
    ax1.set_title("Potentiels V5.13 / Starobinsky / V5.14")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # --- Panneau 2 : n_s vs N_e
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(Ns, n_s_v5p13_arr, "b-", lw=2.0, label="V5.13 (loi puissance)")
    ax2.plot(Ns, n_s_staro, "g--", lw=2.0, label="Starobinsky $R^2$ réf.")
    ax2.plot(Ns, n_s_v5p14_arr, "r-", lw=2.2,
             label=fr"V5.14 ($\epsilon_{{\rm top}}={eps_top:.0e}$)")
    ax2.axhspan(N_S_PLANCK - 3 * N_S_PLANCK_SIGMA,
                N_S_PLANCK + 3 * N_S_PLANCK_SIGMA,
                color="orange", alpha=0.2,
                label=fr"Planck $3\sigma$")
    ax2.axhline(N_S_PLANCK, color="orange", ls="-", lw=1.2)
    ax2.axvline(60.0, color="grey", ls=":", lw=1.2, label=r"$N_e=60$ pivot")
    ax2.set_xlabel(r"$N_e$ (e-folds)")
    ax2.set_ylabel(r"$n_s$")
    ax2.set_title(r"Tilt scalaire $n_s$ : V5.13 vs V5.14 vs Planck")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(40, 70)
    ax2.set_ylim(0.94, 1.005)

    # --- Panneau 3 : r vs N_e
    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogy(Ns, r_v5p13_arr, "b-", lw=2.0, label="V5.13 (loi puissance)")
    ax3.semilogy(Ns, r_staro, "g--", lw=2.0, label="Starobinsky $R^2$ réf.")
    ax3.semilogy(Ns, r_v5p14_arr, "r-", lw=2.2,
                 label=fr"V5.14 ($\epsilon_{{\rm top}}={eps_top:.0e}$)")
    ax3.axhline(R_PLANCK_BICEP, color="orange", ls="-", lw=1.5,
                label=fr"BICEP upper $r<{R_PLANCK_BICEP}$")
    ax3.axhspan(R_PLANCK_BICEP, 1.0, color="orange", alpha=0.15)
    ax3.axvline(60.0, color="grey", ls=":", lw=1.2)
    ax3.set_xlabel(r"$N_e$ (e-folds)")
    ax3.set_ylabel(r"$r$ (tensor-to-scalar)")
    ax3.set_title(r"Ratio $r$ : V5.14 entre dans la fenêtre BICEP")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.3, which="both")
    ax3.set_xlim(40, 70)
    ax3.set_ylim(1e-3, 1)

    # --- Panneau 4 : sensibilité à ε_top
    ax4 = plt.subplot(2, 2, 4)
    color_ns = "tab:red"
    color_r = "tab:blue"
    ax4.set_xscale("log")
    ax4.plot(eps_top_grid, n_s_vs_eps, "o-", color=color_ns, lw=2.0,
             label=r"$n_s(\epsilon_{\rm top})$")
    ax4.axhline(N_S_PLANCK, color=color_ns, ls=":", lw=1.2,
                label=r"Planck $n_s=0.965$")
    ax4.axhspan(N_S_PLANCK - 3 * N_S_PLANCK_SIGMA,
                N_S_PLANCK + 3 * N_S_PLANCK_SIGMA,
                color="orange", alpha=0.15)
    ax4.set_xlabel(r"$\epsilon_{\rm top}$ (résidu topologique)")
    ax4.set_ylabel(r"$n_s$", color=color_ns)
    ax4.tick_params(axis="y", labelcolor=color_ns)
    ax4.grid(alpha=0.3, which="both")
    ax4.set_title(r"Sensibilité de V5.14 au résidu $\epsilon_{\rm top}$ (N=60)")

    ax4b = ax4.twinx()
    ax4b.set_yscale("log")
    ax4b.plot(eps_top_grid, r_vs_eps, "s-", color=color_r, lw=2.0,
              label=r"$r(\epsilon_{\rm top})$")
    ax4b.axhline(R_PLANCK_BICEP, color=color_r, ls=":", lw=1.2)
    ax4b.set_ylabel(r"$r$", color=color_r)
    ax4b.tick_params(axis="y", labelcolor=color_r)
    ax4b.set_ylim(1e-3, 1)

    # légende combinée
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=8)

    fig.suptitle(
        f"V5.14 — Correction $R^2$ Starobinsky du χ-field vs Planck 2018 + BICEP/Keck 2021\n"
        f"$\\epsilon_{{\\rm top}}={eps_top:.0e}$, $\\beta={BETA_TOP_DEFAULT:.3f}$, "
        f"$\\phi_{{\\max}}={PHI_MAX_V5p13:.2f}\\,M_{{pl}}$  →  "
        f"$n_s={pred.n_s:.4f}$ ({pred.n_s_deviation_sigma:+.1f}σ), $r={pred.r:.4f}$",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "v5p14_starobinsky_planck_BICEP.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nFigure : {out_path}")

    print()
    print("=" * 72)
    print("VERDICT V5.14")
    print("=" * 72)
    print(f"  A_s  : 2.10e-9 reproduit (V_0 calibré comme V5.13)")
    print(f"  n_s  : V5.14 → {pred.n_s:.4f} ({pred.n_s_deviation_sigma:+.1f}σ Planck) "
          f"vs V5.13 → {n_s_v5p13:.4f} ({(n_s_v5p13-N_S_PLANCK)/N_S_PLANCK_SIGMA:+.1f}σ)")
    print(f"  r    : V5.14 → {pred.r:.4f} (< BICEP {R_PLANCK_BICEP}) "
          f"vs V5.13 → {r_v5p13:.3f} (×{r_v5p13/R_PLANCK_BICEP:.1f} BICEP)")
    print()
    if pred.compatible_planck_n_s and pred.compatible_BICEP_r:
        print("  ✓ V5.14 COMPATIBLE Planck 2018 + BICEP/Keck 2021.")
    else:
        print("  ✗ V5.14 incompatible (à investiguer).")
    print()
    print("  Limite HONNÊTE : le coefficient `a` du terme R² n'est PAS")
    print("  encore dérivé depuis la structure SU(2)_135 de V5. À ε_top=0,")
    print("  V_0 est calibré sur A_s ⇒ a = 3 M_pl²/(4 V_0) = libre. Le")
    print("  passage de V5.13 (loi puissance) à V5.14 (R² + résidu) est")
    print("  une CONJECTURE phénoménologique compatible avec les données ;")
    print("  la dérivation de `a` depuis l'action TQNT-V5 reste un travail")
    print("  futur (V5.15).")


if __name__ == "__main__":
    main()
