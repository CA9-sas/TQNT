# simulation/experiments/schrodinger_spectrum.py
"""
Experiment: Solve Schrödinger for all 15 knots <= 7 crossings.
Produce energy spectra, density cuts, 3D isosurfaces, and correlation analysis.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from simulation.solver import solve_knot
from simulation.visualization import (
    plot_density_cut,
    plot_energy_spectrum,
    plot_3d_isosurface,
    plot_correlation_E0_rho,
)
from theory.jones import JONES_KNOT_NAMES as KNOT_NAMES

OUTPUT_DIR = "output/schrodinger"
KNOTS_LE_7 = [k for k in KNOT_NAMES if int(k.split("_")[0]) <= 7]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    print(f"Solving Schrödinger for {len(KNOTS_LE_7)} knots...")
    print(f"{'Knot':<8} {'rho(K)':<8} {'M(K)':<8} {'E_0':<12} {'E_1':<12} {'E_2':<12}")
    print("-" * 68)

    for knot_name in KNOTS_LE_7:
        try:
            result = solve_knot(
                knot_name, n_coarse=25, epsilon=0.2,
                n_eigenvalues=10, refine=True
            )
            results.append(result)

            eigs = result["eigenvalues"]
            rho = result["retraction_factor"]
            mass = result["retraction_mass"]
            print(f"{knot_name:<8} {rho:<8.3f} {mass:<8.3f} "
                  f"{eigs[0]:<12.4f} {eigs[1]:<12.4f} {eigs[2]:<12.4f}")

            # Per-knot plots
            knot_dir = os.path.join(OUTPUT_DIR, knot_name)
            os.makedirs(knot_dir, exist_ok=True)

            for s in range(min(3, len(eigs))):
                plot_density_cut(result, state_index=s,
                                 output_path=os.path.join(knot_dir, f"density_state{s}.png"))

            plot_energy_spectrum(result,
                                 output_path=os.path.join(knot_dir, "spectrum.png"))

            plot_3d_isosurface(result, state_index=0,
                                output_path=os.path.join(knot_dir, "isosurface_ground.png"))

        except Exception as e:
            print(f"{knot_name:<8} FAILED: {e}")

    # Correlation plot
    if len(results) >= 2:
        plot_correlation_E0_rho(
            results,
            output_path=os.path.join(OUTPUT_DIR, "correlation_E0_rho.png")
        )

    # Summary table
    print("\n" + "=" * 68)
    print("COMPARAISON E_0 vs M(K) = alpha * rho(K)")
    print(f"{'Knot':<8} {'rho(K)':<8} {'M(K)':<8} {'E_0':<12} {'E_0/M(K)':<10}")
    print("-" * 46)
    for r in results:
        ratio = r["eigenvalues"][0] / r["retraction_mass"] if r["retraction_mass"] > 0 else float('nan')
        print(f"{r['knot_name']:<8} {r['retraction_factor']:<8.3f} "
              f"{r['retraction_mass']:<8.3f} {r['eigenvalues'][0]:<12.4f} {ratio:<10.4f}")

    print(f"\nPlots sauvegardes dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
