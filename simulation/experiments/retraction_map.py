# simulation/experiments/retraction_map.py
"""
Experiment: Visualize the retraction potential around different knots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from theory.knot_catalog import knot_curve, distance_to_knot
from theory.jones import jones_polynomial
from theory.retraction import retraction_factor

OUTPUT_DIR = "output/retraction_map"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    knots = ["0_1", "3_1", "4_1", "5_1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, knot_name in zip(axes.flat, knots):
        curve = knot_curve(knot_name, n_points=300)
        poly = jones_polynomial(knot_name)
        rho = retraction_factor(poly)

        grid_size = 80
        x_range = np.linspace(-5, 5, grid_size)
        y_range = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        V = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([X[i, j], Y[i, j], 0.0])
                d = distance_to_knot(point, curve)
                d = max(d, 0.1)
                V[i, j] = -rho / d**2

        im = ax.pcolormesh(X, Y, V, shading='auto', cmap='inferno',
                           vmin=-50, vmax=0)
        ax.plot(curve[:, 0], curve[:, 1], 'w-', linewidth=1.5, alpha=0.8)
        ax.set_title(f"{knot_name} -- rho={rho:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label="V_retract")

    fig.suptitle("Carte de retractation de l'espace-temps (coupe z=0)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/retraction_maps.png", dpi=150)
    plt.close(fig)

    print(f"Carte de retractation sauvegardee dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
