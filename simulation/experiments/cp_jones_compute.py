r"""
Compute ε_CP(K) = 2·Im[V_K(q)] / |V_K(q)| at q = e^(2πi/137).

Reproduces Table 1 of:
    L. Cariou, "A CP-violating phase from the Jones polynomial
    at q = e^{2πi/137}", arXiv:XXXX.XXXXX (2026).

Usage:
    python3 -m simulation.experiments.cp_jones_compute
"""
from __future__ import annotations
import numpy as np

from theory.jones import jones_polynomial


# Root of unity for SU(2) Chern-Simons at level k = 135
K_LEVEL = 135
Q = np.exp(2j * np.pi / (K_LEVEL + 2))


def epsilon_cp(knot_name: str) -> dict:
    """Compute ε_CP for a knot from its Jones polynomial.

    Returns a dict with V_K(q), |V_K(q)|, Im V_K(q), ε_CP.
    """
    poly = jones_polynomial(knot_name)  # dict {power: coefficient}
    V = sum(c * Q ** p for p, c in poly.items())
    mod = abs(V)
    eps = 2 * V.imag / mod if mod > 1e-12 else 0.0
    return {
        "knot": knot_name,
        "V_real": V.real,
        "V_imag": V.imag,
        "V_abs": mod,
        "epsilon_cp": eps,
    }


# Knots tabulated in the paper (Section 4, Table 1)
KNOTS = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_3", "7_1", "8_3"]


def main():
    print(f"q = exp(2πi/137) = {Q:.6f}")
    print(f"|q| = {abs(Q):.6f}\n")
    print(f"{'Knot':<6} {'Re V(q)':>12} {'Im V(q)':>14} "
          f"{'|V(q)|':>10} {'ε_CP':>14}")
    print("-" * 60)
    for k in KNOTS:
        r = epsilon_cp(k)
        print(f"{r['knot']:<6} {r['V_real']:>12.6f} "
              f"{r['V_imag']:>14.6e} {r['V_abs']:>10.6f} "
              f"{r['epsilon_cp']:>14.6e}")

    # Additional check: hierarchy ratio for torus knots
    print()
    refs = {k: epsilon_cp(k)["epsilon_cp"] for k in ("3_1", "5_1", "7_1")}
    base = refs["3_1"]
    ratios = {k: v / base for k, v in refs.items()}
    print(f"Torus T(2, 2n+1) ratio 3_1:5_1:7_1 = "
          f"{ratios['3_1']:.2f} : {ratios['5_1']:.2f} : {ratios['7_1']:.2f}")

    # PMNS prediction: 1/3 sum of the three generation phases
    delta_pmns = (refs["3_1"] + refs["5_1"] + refs["7_1"]) / 3
    print(f"\nPredicted δ_CP^PMNS = (1/3)·Σ ε_CP(K_gen) = "
          f"{delta_pmns:.4e} rad = {np.degrees(delta_pmns):.4f}°")


if __name__ == "__main__":
    main()
