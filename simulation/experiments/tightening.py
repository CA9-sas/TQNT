# simulation/experiments/tightening.py
"""
Experiment: Visualize the tightening of a knot under increasing energy.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from theory.retraction import tightening_radius
from theory.knot_catalog import knot_curve

OUTPUT_DIR = "output/tightening"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    energies = np.linspace(0, 50, 500)
    radii = np.array([tightening_radius(E) for E in energies])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, radii, 'b-', linewidth=2)
    ax.set_xlabel("Energie E")
    ax.set_ylabel("Rayon de serrage r_s")
    ax.set_title("P8 -- Serrage Energetique : Dualite Onde-Particule")
    ax.axhline(y=radii[0], color='green', linestyle='--', alpha=0.5, label=f"r_0 = {radii[0]:.1f} (onde)")
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="r_s -> 0 (particule)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tightening_curve.png", dpi=150)
    plt.close(fig)

    knot_name = "3_1"
    base_curve = knot_curve(knot_name, n_points=500)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                              subplot_kw={"projection": "3d"})
    energy_levels = [0.0, 1.0, 5.0, 10.0, 50.0, 200.0]

    for ax, E in zip(axes.flat, energy_levels):
        r_s = tightening_radius(E)
        scale = r_s / tightening_radius(0.0)
        scaled_curve = base_curve * scale

        ax.plot(scaled_curve[:, 0], scaled_curve[:, 1], scaled_curve[:, 2],
                linewidth=2, color='blue')
        ax.set_title(f"E = {E:.0f}  |  r_s = {r_s:.2f}")
        max_range = max(np.ptp(scaled_curve[:, i]) for i in range(3)) / 2
        if max_range < 0.1:
            max_range = 0.1
        center = scaled_curve.mean(axis=0)
        for i, label in enumerate(["x", "y", "z"]):
            getattr(ax, f"set_{label}lim")(center[i] - max_range, center[i] + max_range)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"Serrage du noeud {knot_name} (trefle) sous energie croissante",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/trefoil_tightening.png", dpi=150)
    plt.close(fig)

    print(f"Graphiques sauvegardes dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
