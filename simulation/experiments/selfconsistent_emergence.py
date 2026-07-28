"""
Théorie Nodale — Self-Consistent Metric Backreaction Experiment

Demonstrates the convergence of the coupled Schrödinger + metric system:
  Ω² = 1 + κ|Ψ|²  feeds back into the FEM weak form.

For trefoil (3_1) and figure-eight (4_1):
  1. Per-iteration convergence table
  2. Final comparison: eigenvalues with/without backreaction
"""
import numpy as np
from scipy.spatial import cKDTree
from theory.knot_catalog import knot_curve
from theory.chi_field import jones_rho, chi_retraction
from theory.postulates import KAPPA
from theory.emergent_spacetime import emergent_metric_factor
from theory.schrodinger import (
    generate_mesh, assemble_system, solve_eigenproblem,
)
from simulation.solver import solve_knot_jones, solve_knot_selfconsistent


def run_with_trace(knot_name, n_coarse=12, n_eigenvalues=5, max_iterations=30,
                   tol=1e-4):
    """Run self-consistent iteration with per-step diagnostics."""
    curve = knot_curve(knot_name, 500)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=0.2, refine=True
    )

    n_pts = len(points)
    free_mask = ~boundary
    energy = 0.0
    omega_sq = np.ones(n_pts)
    damping = 0.5
    trace = []

    for it in range(max_iterations):
        H, M = assemble_system(
            points, tets, boundary, knot_name, knot_tree,
            energy=energy, epsilon=0.2, omega_sq=omega_sq,
        )
        eigenvalues, eigenvectors = solve_eigenproblem(
            H, M, n_eigenvalues=n_eigenvalues
        )

        new_energy = abs(eigenvalues[0])
        blended_energy = damping * new_energy + (1 - damping) * energy
        delta_energy = abs(blended_energy - energy)

        psi_full = np.zeros(n_pts)
        psi_full[free_mask] = eigenvectors[:, 0]
        new_omega_sq = emergent_metric_factor(psi_full)
        blended_omega_sq = damping * new_omega_sq + (1 - damping) * omega_sq
        delta_metric = np.max(np.abs(blended_omega_sq - omega_sq))

        max_kappa_psi2 = KAPPA * np.max(np.abs(psi_full) ** 2)

        trace.append({
            "iteration": it + 1,
            "energy": blended_energy,
            "max_kappa_psi2": max_kappa_psi2,
            "delta_metric": delta_metric,
            "delta_energy": delta_energy,
        })

        energy = blended_energy
        omega_sq = blended_omega_sq

        energy_scale = max(abs(energy), 1e-10)
        if delta_energy < tol * energy_scale and delta_metric < tol:
            break

    return trace, eigenvalues


def main():
    knots = ["3_1", "4_1"]

    print("=" * 72)
    print("  SELF-CONSISTENT NONLINEAR SCHRÖDINGER WITH METRIC BACKREACTION")
    print("  Ω² = 1 + κ|Ψ|²  ←→  FEM weak form on conformal metric")
    print("=" * 72)

    results_sc = {}
    results_lin = {}

    for knot_name in knots:
        print(f"\n{'─' * 72}")
        print(f"  Knot: {knot_name}   (ρ_J = {jones_rho(knot_name):.4f})")
        print(f"{'─' * 72}")

        # Per-iteration convergence table
        print(f"\n  {'Iter':>4s}  {'Energy':>12s}  {'max κ|Ψ|²':>12s}  "
              f"{'ΔMetric':>12s}  {'ΔEnergy':>12s}")
        print(f"  {'─' * 4}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")

        trace, eigenvalues_sc = run_with_trace(knot_name, n_coarse=12)
        for step in trace:
            print(f"  {step['iteration']:4d}  {step['energy']:12.6f}  "
                  f"{step['max_kappa_psi2']:12.6e}  "
                  f"{step['delta_metric']:12.6e}  "
                  f"{step['delta_energy']:12.6e}")

        converged = len(trace) < 30
        print(f"\n  → Converged: {converged} ({len(trace)} iterations)")

        results_sc[knot_name] = eigenvalues_sc

        # Linear (no backreaction) for comparison
        result_lin = solve_knot_jones(
            knot_name, n_coarse=12, n_eigenvalues=5, max_iterations=20
        )
        results_lin[knot_name] = result_lin["eigenvalues"]

    # Final comparison table
    print(f"\n\n{'=' * 72}")
    print("  EIGENVALUE COMPARISON: Self-Consistent vs Linear")
    print(f"{'=' * 72}")
    print(f"\n  {'Knot':>6s}  {'State':>5s}  {'E (linear)':>12s}  "
          f"{'E (self-con)':>12s}  {'ΔE/|E|':>12s}")
    print(f"  {'─' * 6}  {'─' * 5}  {'─' * 12}  {'─' * 12}  {'─' * 12}")

    for knot_name in knots:
        n_show = min(len(results_sc[knot_name]), len(results_lin[knot_name]))
        for i in range(n_show):
            E_lin = results_lin[knot_name][i]
            E_sc = results_sc[knot_name][i]
            rel = abs(E_sc - E_lin) / max(abs(E_lin), 1e-10)
            print(f"  {knot_name:>6s}  {i:5d}  {E_lin:12.6f}  "
                  f"{E_sc:12.6f}  {rel:12.6e}")

    print(f"\n  κ = {KAPPA} (spacetime coupling constant)")
    print("  Self-consistency closes the emergence loop:")
    print("  Chern-Simons → A-polynomial → Schrödinger ←→ Metric")
    print()


if __name__ == "__main__":
    main()
