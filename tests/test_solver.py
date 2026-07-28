# tests/test_solver.py
import numpy as np
from simulation.solver import solve_knot


def test_solve_knot_returns_result():
    result = solve_knot("0_1", n_coarse=8, n_eigenvalues=3)
    assert "eigenvalues" in result
    assert "eigenvectors" in result
    assert "points" in result
    assert "boundary" in result
    assert "knot_name" in result


def test_solve_knot_eigenvalues_sorted():
    result = solve_knot("0_1", n_coarse=8, n_eigenvalues=3)
    eigs = result["eigenvalues"]
    assert len(eigs) == 3
    assert eigs[0] <= eigs[1] <= eigs[2]


def test_solve_knot_trefoil():
    result = solve_knot("3_1", n_coarse=8, n_eigenvalues=3)
    assert len(result["eigenvalues"]) == 3
    assert result["retraction_mass"] > 0


def test_solve_knot_different_knots_different_spectra():
    r1 = solve_knot("0_1", n_coarse=8, n_eigenvalues=3)
    r2 = solve_knot("3_1", n_coarse=8, n_eigenvalues=3)
    assert not np.allclose(r1["eigenvalues"], r2["eigenvalues"])


# --- Phase 3: Jones-based solver ---
from simulation.solver import solve_knot_jones


def test_solve_knot_jones_returns_result():
    result = solve_knot_jones("0_1", n_coarse=8, n_eigenvalues=3)
    assert "eigenvalues" in result
    assert "retraction_factor_jones" in result
    assert "retraction_factor_final" in result
    assert "converged" in result


def test_solve_knot_jones_converges():
    result = solve_knot_jones("3_1", n_coarse=8, n_eigenvalues=3)
    assert result["converged"]
    assert result["iterations"] <= 20


def test_solve_knot_jones_final_rho_less_than_initial():
    result = solve_knot_jones("3_1", n_coarse=8, n_eigenvalues=3)
    assert result["retraction_factor_final"] <= result["retraction_factor_jones"]


# --- Phase 4: Self-consistent metric backreaction ---
from simulation.solver import solve_knot_selfconsistent


def test_solve_knot_selfconsistent_returns_result():
    result = solve_knot_selfconsistent("0_1", n_coarse=8, n_eigenvalues=3)
    for key in ("eigenvalues", "eigenvectors", "omega_sq", "metric_correction",
                "iterations", "converged"):
        assert key in result, f"Missing key: {key}"


def test_solve_knot_selfconsistent_converges():
    result = solve_knot_selfconsistent("3_1", n_coarse=8, n_eigenvalues=3)
    assert result["converged"]
    assert result["iterations"] <= 30


def test_solve_knot_selfconsistent_metric_near_one():
    """In the weak-field regime, κ|Ψ|² should be small (< 0.5)."""
    result = solve_knot_selfconsistent("3_1", n_coarse=8, n_eigenvalues=3)
    max_correction = np.max(np.abs(result["metric_correction"]))
    assert max_correction < 0.5


def test_solve_knot_selfconsistent_matches_linear():
    """Eigenvalues should be close to the linear (no backreaction) case."""
    result_sc = solve_knot_selfconsistent("3_1", n_coarse=8, n_eigenvalues=3)
    result_lin = solve_knot_jones("3_1", n_coarse=8, n_eigenvalues=3)
    # Ground state energies should agree within 50% (weak field)
    E_sc = result_sc["eigenvalues"][0]
    E_lin = result_lin["eigenvalues"][0]
    assert abs(E_sc - E_lin) < 0.5 * max(abs(E_lin), 1e-10)
