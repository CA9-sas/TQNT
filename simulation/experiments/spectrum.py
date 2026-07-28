# simulation/experiments/spectrum.py
"""
Experiment: Compare predicted nodal mass spectrum with known particles.

Generates a mass spectrum chart and prints the classification table.
"""
import matplotlib
matplotlib.use("Agg")

from theory.nodal_classification import PARTICLE_CATALOG, mass_spectrum
from simulation.visualization import plot_mass_spectrum
import os

OUTPUT_DIR = "output/spectrum"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print classification table
    print("=" * 65)
    print("THÉORIE NODALE — Classification des Particules")
    print("=" * 65)
    print(f"{'Nom':<20} {'(n,l,m)':<12} {'Masse':<10} {'Charge':<10}")
    print("-" * 65)
    for name, mass in mass_spectrum():
        particle = next(p for p in PARTICLE_CATALOG if name.endswith(p.name))
        print(f"{particle.name:<20} {particle.nodal_signature:<12} {particle.mass:<10.2f} {particle.charge:<+10.1f}")
    print("=" * 65)

    # Plot mass spectrum
    plot_mass_spectrum(save_path=f"{OUTPUT_DIR}/mass_spectrum.png")
    print(f"\nSpectre de masse sauvegardé dans {OUTPUT_DIR}/mass_spectrum.png")


if __name__ == "__main__":
    run()
