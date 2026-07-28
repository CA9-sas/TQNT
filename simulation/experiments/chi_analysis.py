# simulation/experiments/chi_analysis.py
"""
Experiment: Analyze the chi field properties and verify AJ conjecture consistency.
- Plot rho(K, E) curves for different knots
- Check A-polynomial consistency
- Compare characteristic energies
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from theory.chi_field import jones_rho, chi_retraction, characteristic_energy
from theory.jones import JONES_KNOT_NAMES, jones_polynomial, jones_span, jones_determinant
from theory.jones import is_amphicheiral
from theory.a_polynomial import A_POLY_KNOT_NAMES, evaluate_a_polynomial

OUTPUT_DIR = "output/chi_analysis"
KNOTS_LE_7 = [k for k in JONES_KNOT_NAMES if int(k.split("_")[0]) <= 7]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. rho(K, E) curves
    fig, ax = plt.subplots(figsize=(10, 7))
    E_range = np.linspace(0, 50, 200)

    for knot_name in KNOTS_LE_7:
        rho_curve = [chi_retraction(knot_name, E) for E in E_range]
        ax.plot(E_range, rho_curve, label=knot_name, linewidth=1.5)

    ax.set_xlabel("Energy E")
    ax.set_ylabel("rho(K, E)")
    ax.set_title("Retraction factor vs energy - all knots <= 7 crossings")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rho_vs_energy.png"), dpi=150)
    plt.close(fig)

    # 2. Jones polynomial properties table
    print("JONES POLYNOMIAL PROPERTIES")
    print(f"{'Knot':<8} {'span':<6} {'det':<6} {'rho_J':<8} {'E_K':<8} {'chiral':<8}")
    print("-" * 50)
    for name in JONES_KNOT_NAMES:
        poly = jones_polynomial(name)
        print(f"{name:<8} {jones_span(poly):<6} {jones_determinant(poly):<6} "
              f"{jones_rho(name):<8.2f} {characteristic_energy(name):<8.2f} "
              f"{'no' if is_amphicheiral(poly) else 'yes':<8}")

    # 3. A-polynomial verification
    print("\nA-POLYNOMIAL VERIFICATION")
    print(f"{'Knot':<8} {'A(1, 2)':<12} {'A(1, -1)':<12} {'OK?':<5}")
    print("-" * 40)
    for name in A_POLY_KNOT_NAMES:
        a_1_2 = evaluate_a_polynomial(name, 1.0, 2.0)
        a_1_m1 = evaluate_a_polynomial(name, 1.0, -1.0)
        ok = abs(a_1_2) < 1e-10 and abs(a_1_m1) < 1e-10
        print(f"{name:<8} {a_1_2:<12.6f} {a_1_m1:<12.6f} {'OK' if ok else 'FAIL':<5}")

    # 4. Characteristic energy bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    names = KNOTS_LE_7
    E_Ks = [characteristic_energy(n) for n in names]
    ax.bar(range(len(names)), E_Ks, color='steelblue')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45)
    ax.set_ylabel("E_K = E_0 * rho_J(K)")
    ax.set_title("Characteristic energy scale per knot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "characteristic_energies.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
