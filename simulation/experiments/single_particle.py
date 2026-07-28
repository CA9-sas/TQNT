# simulation/experiments/single_particle.py
"""
Experiment: Visualize single nodal particles.

Generates plots for particles with different (n, l, m) triplets,
showing how nodes determine particle identity.
"""
import matplotlib
matplotlib.use("Agg")

from simulation.visualization import (
    plot_radial_wave,
    plot_angular_distribution,
    plot_emergent_spacetime,
)
import os

OUTPUT_DIR = "output/single_particle"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Radial waves for increasing n (more nodes = heavier particles)
    for n in [1, 2, 3]:
        plot_radial_wave(n=n, l=0, save_path=f"{OUTPUT_DIR}/radial_n{n}_l0.png")

    # Angular distributions for different l, m
    for l, m in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]:
        plot_angular_distribution(l=l, m=m, save_path=f"{OUTPUT_DIR}/angular_l{l}_m{m}.png")

    # Emergent spacetime around different particles
    for n, l, m in [(1, 0, 0), (1, 1, -1), (2, 0, 0), (3, 1, 0)]:
        plot_emergent_spacetime(n=n, l=l, m=m, save_path=f"{OUTPUT_DIR}/spacetime_{n}{l}{m}.png")

    print(f"All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
