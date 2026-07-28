# simulation/experiments/knot_spectrum.py
"""
Experiment: Compute and visualize the knot mass spectrum.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from theory.knot_catalog import knot_info_table
from theory.particle_classification import (
    STANDARD_MODEL_PARTICLES,
    knot_mass_spectrum,
    find_best_knot_mapping,
)

OUTPUT_DIR = "output/knot_spectrum"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    table = knot_info_table()
    print("=" * 105)
    print("THEORIE NODALE v3 -- Spectre de Masse par Noeud Topologique")
    print("=" * 105)
    print(f"{'Noeud':<8} {'Polynome de Jones':<35} {'Span':<5} {'Det':<5} {'Rho':<8} {'Masse':<8} {'Spin':<6} {'Charge':<8}")
    print("-" * 105)
    for k in table:
        print(f"{k.name:<8} {k.polynomial_str:<35} {k.degree:<5} {k.determinant:<5} "
              f"{k.retraction:<8.2f} {k.mass:<8.2f} {k.spin:<6.1f} {k.charge:<8.4f}")
    print("=" * 105)

    fig, ax = plt.subplots(figsize=(14, 8))
    names = [k.name for k in table]
    masses = [k.mass for k in table]
    degrees = [k.degree for k in table]

    colors = plt.cm.viridis(np.array(degrees) / max(max(degrees), 1))
    ax.barh(range(len(names)), masses, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Masse predite (unites naturelles)")
    ax.set_title("Spectre de masse -- Theorie Nodale v3 (Polynomes de Jones)")
    ax.grid(True, axis='x', alpha=0.3)

    for i, (mass, k) in enumerate(zip(masses, table)):
        ax.text(mass + 0.05, i, f"rho={k.retraction:.1f}", va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/knot_mass_spectrum.png", dpi=150)
    plt.close(fig)

    print("\n")
    print("=" * 85)
    print("MAPPING NOEUDS -> PARTICULES (par ordre de masse)")
    print("=" * 85)
    mapping = find_best_knot_mapping()
    spectrum = knot_mass_spectrum()
    for knot_name, particles in mapping.items():
        knot_mass_val = next((m for n, m in spectrum if n == knot_name), "?")
        p_names = ", ".join(p["name"] for p in particles)
        print(f"  {knot_name:<10} (M={knot_mass_val}) -> {p_names}")
    print("=" * 85)

    print(f"\nGraphiques sauvegardes dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
