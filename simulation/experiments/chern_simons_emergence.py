# simulation/experiments/chern_simons_emergence.py
"""
Experiment: Visualize the full Chern-Simons emergence chain.
F_A = 0 -> flat connections -> colored Jones -> AJ -> Schrodinger
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from theory.chern_simons import (
    cs_q, colored_jones, verify_aj_conjecture,
    representation_variety, classical_limit_check,
    effective_potential_from_a_poly, compare_potentials,
    COLORED_JONES_KNOTS, quantum_integer,
)
from theory.a_polynomial import A_POLY_KNOT_NAMES

OUTPUT_DIR = "output/chern_simons"

# Knots with both colored Jones and A-polynomial
KNOTS_BOTH = [k for k in COLORED_JONES_KNOTS if k in A_POLY_KNOT_NAMES and k != "0_1"]


def plot_representation_variety():
    """Plot A_K(L, M) = 0 curves for each knot."""
    fig, axes = plt.subplots(1, len(KNOTS_BOTH), figsize=(4 * len(KNOTS_BOTH), 4))
    if len(KNOTS_BOTH) == 1:
        axes = [axes]

    for ax, name in zip(axes, KNOTS_BOTH):
        M_vals, L_vals = representation_variety(name, n_points=300)
        ax.scatter(M_vals.real, L_vals.real, s=1, alpha=0.5)
        ax.set_xlabel("M")
        ax.set_ylabel("L")
        ax.set_title(f"{name}: A_na(L,M) = 0")
        ax.set_xlim(0, 5)
        ax.set_ylim(-10, 10)
        ax.axhline(y=0, color="k", linewidth=0.5)

    fig.suptitle("Representation Variety (Flat Connections)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/representation_variety.png", dpi=150)
    plt.close(fig)
    print("  -> representation_variety.png")


def print_colored_jones_table():
    """Print J_n(K; q) for n=1..6 for each knot at k=50."""
    q = cs_q(k=50)
    print("\nColored Jones Table (k=50, q = exp(2*pi*i/52)):")
    header = f"{'Knot':<8}" + "".join(f"{'J_'+str(n):<22}" for n in range(1, 7))
    print(header)
    print("-" * len(header))

    for name in COLORED_JONES_KNOTS:
        row = f"{name:<8}"
        for n in range(1, 7):
            j = colored_jones(name, n, q)
            row += f"{j.real:8.3f}{j.imag:+8.3f}i  "
        print(row)


def verify_aj():
    """Verify AJ conjecture for all knots via classical limit convergence."""
    print("\nAJ Conjecture Verification (classical limit convergence):")
    print(f"{'Knot':<8} {'k':<8} {'Max Residual':<15} {'Status'}")
    print("-" * 45)

    for name in KNOTS_BOTH:
        result = verify_aj_conjecture(name, k=200, n_max=5)
        status = "PASS" if result["passed"] else "CHECK"
        print(f"{name:<8} {result['k']:<8} {result['max_residual']:<15.6f} {status}")


def plot_classical_limit():
    """Show [n]_q -> n as k -> inf."""
    fig, ax = plt.subplots(figsize=(8, 5))
    k_values = [5, 10, 20, 50, 100, 500]

    for n in [2, 3, 5, 7]:
        qn_at_k = []
        for k in k_values:
            q = cs_q(k)
            qn_at_k.append(quantum_integer(n, q).real)
        ax.plot(k_values, qn_at_k, "o-", label=f"[{n}]_q")
        ax.axhline(y=n, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Level k")
    ax.set_ylabel("[n]_q (real part)")
    ax.set_title("Classical Limit: [n]_q -> n as k -> inf")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/classical_limit.png", dpi=150)
    plt.close(fig)
    print("  -> classical_limit.png")


def plot_potential_comparison():
    """A-polynomial V_eff vs retraction V for each knot."""
    fig, axes = plt.subplots(1, len(KNOTS_BOTH), figsize=(4 * len(KNOTS_BOTH), 4))
    if len(KNOTS_BOTH) == 1:
        axes = [axes]

    for ax, name in zip(axes, KNOTS_BOTH):
        result = compare_potentials(name, n_points=200)
        x = result["x_vals"]
        v_cs = result["V_cs"]
        v_ret = result["V_retraction"]
        valid_cs = ~np.isnan(v_cs) & np.isfinite(v_cs)
        valid_ret = ~np.isnan(v_ret) & np.isfinite(v_ret)

        if np.any(valid_cs):
            ax.plot(x[valid_cs], v_cs[valid_cs], label="V_CS (A-poly)")
        if np.any(valid_ret):
            ax.plot(x[valid_ret], v_ret[valid_ret], "--", label="V_ret")

        ax.set_xlabel("M")
        ax.set_ylabel("V(M)")
        ax.set_title(f"{name} (corr={result['correlation']:.2f})")
        ax.set_ylim(-5, 1)
        ax.legend(fontsize=8)

    fig.suptitle("Potential Comparison: CS vs Retraction")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/potential_comparison.png", dpi=150)
    plt.close(fig)
    print("  -> potential_comparison.png")


def print_emergence_summary():
    """Full emergence chain summary table."""
    print("\n" + "=" * 70)
    print("CHERN-SIMONS EMERGENCE CHAIN SUMMARY")
    print("=" * 70)
    print("""
Chain: F_A = 0 -> flat connections -> colored Jones -> AJ -> Schrodinger

Layer 1: Quantum Algebra      [n]_q, [n]_q!, [n choose k]_q
Layer 2: Colored Jones         J_n(K; q) for unknot, torus, figure-eight
Layer 3: AJ Conjecture         A_K(L,M)=0 encodes J_n asymptotics
Layer 4: Flat Connections      representation_variety(K) solves A_na=0
Layer 5: CS Action             S_CS = (1/2) int log(L) d(log(M))
Layer 6: Semi-Classical        V_eff = -p^2/2 from WKB on A-polynomial
    """)

    print(f"{'Knot':<8} {'rho_J':<8} {'V_CS depth':<12} {'V_ret depth':<12} {'Correlation':<12}")
    print("-" * 52)

    from theory.chi_field import jones_rho
    for name in KNOTS_BOTH:
        result = compare_potentials(name, n_points=100)
        rho = jones_rho(name)
        v_cs = result["V_cs"]
        v_ret = result["V_retraction"]
        v_cs_min = np.nanmin(v_cs) if np.any(~np.isnan(v_cs)) else float("nan")
        v_ret_min = np.nanmin(v_ret) if np.any(~np.isnan(v_ret)) else float("nan")
        print(f"{name:<8} {rho:<8.3f} {v_cs_min:<12.3f} {v_ret_min:<12.3f} {result['correlation']:<12.3f}")


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chern-Simons Emergence Experiment")
    print("=" * 50)

    # 1. Representation variety
    print("\n1. Representation variety (flat connections)...")
    plot_representation_variety()

    # 2. Colored Jones table
    print("\n2. Colored Jones polynomials...")
    print_colored_jones_table()

    # 3. AJ conjecture
    print("\n3. AJ conjecture verification...")
    verify_aj()

    # 4. Classical limit
    print("\n4. Classical limit convergence...")
    plot_classical_limit()

    # 5. Potential comparison
    print("\n5. Potential comparison (CS vs retraction)...")
    plot_potential_comparison()

    # 6. Summary
    print_emergence_summary()

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
