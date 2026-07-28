# simulation/experiments/jones_spectrum.py
"""
Experiment: Compare fixed-energy vs iterated Jones-based Schrödinger spectra.
Solve for all 15 knots <= 7 crossings with both methods.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from simulation.solver import solve_knot, solve_knot_jones
from simulation.visualization import (
    plot_energy_spectrum,
    plot_correlation_E0_rho,
)
from theory.jones import JONES_KNOT_NAMES
from theory.chi_field import jones_rho

OUTPUT_DIR = "output/jones_spectrum"
KNOTS_LE_7 = [k for k in JONES_KNOT_NAMES if int(k.split("_")[0]) <= 7]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_fixed = []
    results_jones = []

    print(f"Comparing fixed-energy vs iterated Jones for {len(KNOTS_LE_7)} knots...")
    print(f"{'Knot':<8} {'rho_J':<8} {'E0(fix)':<12} {'E0(iter)':<12} {'rho_final':<8} {'iter':<5}")
    print("-" * 60)

    for knot_name in KNOTS_LE_7:
        try:
            # Fixed energy (E=0, no iteration)
            r_f = solve_knot(knot_name, n_coarse=20, epsilon=0.2, n_eigenvalues=5, refine=True)
            results_fixed.append(r_f)

            # Iterated (fixed-point)
            r_j = solve_knot_jones(knot_name, n_coarse=20, epsilon=0.2, n_eigenvalues=5, refine=True)
            results_jones.append(r_j)

            rho_J = r_j["retraction_factor_jones"]
            rho_final = r_j["retraction_factor_final"]
            E0_f = r_f["eigenvalues"][0]
            E0_J = r_j["eigenvalues"][0]

            print(f"{knot_name:<8} {rho_J:<8.3f} "
                  f"{E0_f:<12.4f} {E0_J:<12.4f} {rho_final:<8.3f} {r_j['iterations']:<5d}")

        except Exception as e:
            print(f"{knot_name:<8} FAILED: {e}")

    # Comparison plots
    if len(results_jones) >= 2:
        import matplotlib.pyplot as plt

        rhos = [r["retraction_factor_jones"] for r in results_jones]
        E0s_iter = [r["eigenvalues"][0] for r in results_jones]
        E0s_fix = [r["eigenvalues"][0] for r in results_fixed]
        names = [r["knot_name"] for r in results_jones]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Iterated
        axes[0].scatter(rhos, E0s_iter, s=60, zorder=5, color='blue')
        for name, rho, e0 in zip(names, rhos, E0s_iter):
            axes[0].annotate(name, (rho, e0), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
        axes[0].set_xlabel("rho_J(K)")
        axes[0].set_ylabel("E0")
        axes[0].set_title("Jones iterated: E0 vs rho_J")

        # Panel 2: Fixed energy
        rhos_f = [r["retraction_factor"] for r in results_fixed]
        axes[1].scatter(rhos_f, E0s_fix, s=60, zorder=5, color='red')
        for name, rho, e0 in zip(names, rhos_f, E0s_fix):
            axes[1].annotate(name, (rho, e0), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
        axes[1].set_xlabel("rho_J(K)")
        axes[1].set_ylabel("E0")
        axes[1].set_title("Jones fixed (E=0): E0 vs rho_J")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "comparison_E0_rho.png"), dpi=150)
        plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("RESUME: Fixed-energy vs Iterated Jones")
    print(f"{'Knot':<8} {'rho_J':<8} {'E0(fix)':<12} {'E0(iter)':<12} {'d_rho':<8}")
    print("-" * 48)
    for r_f, r_j in zip(results_fixed, results_jones):
        delta_rho = r_j["retraction_factor_jones"] - r_j["retraction_factor_final"]
        print(f"{r_f['knot_name']:<8} {r_j['retraction_factor_jones']:<8.3f} "
              f"{r_f['eigenvalues'][0]:<12.4f} {r_j['eigenvalues'][0]:<12.4f} "
              f"{delta_rho:<8.3f}")

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
