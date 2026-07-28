"""Tests pour le solveur FEM multi-noeuds."""
import numpy as np
import pytest


def test_solve_link_returns_result():
    from simulation.solver_multi import solve_link
    result = solve_link(
        knot_names=["3_1", "3_1"],
        distance=8.0,
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    assert "eigenvalues" in result
    assert "eigenvectors" in result
    assert "points" in result
    assert len(result["eigenvalues"]) == 5


def test_solve_link_eigenvalues_finite():
    from simulation.solver_multi import solve_link
    result = solve_link(
        knot_names=["3_1", "3_1"],
        distance=8.0,
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    assert np.all(np.isfinite(result["eigenvalues"]))


def test_solve_link_single_knot_similar_to_solve_knot():
    """Single-component link should give similar eigenvalues to solve_knot."""
    from simulation.solver_multi import solve_link
    from simulation.solver import solve_knot
    result_link = solve_link(
        knot_names=["3_1"],
        distance=10.0,
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    result_single = solve_knot(
        "3_1",
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    # Ground state should be in the same ballpark (not exact due to mesh differences)
    ratio = result_link["eigenvalues"][0] / result_single["eigenvalues"][0]
    assert 0.5 < ratio < 2.0


def test_solve_link_closer_knots_lower_ground_state():
    """Closer knots => deeper potential well => lower ground state energy."""
    from simulation.solver_multi import solve_link
    result_far = solve_link(
        knot_names=["3_1", "3_1"],
        distance=10.0,
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    result_near = solve_link(
        knot_names=["3_1", "3_1"],
        distance=5.0,
        n_coarse=10,
        n_eigenvalues=5,
        refine=False,
    )
    # Closer knots have overlapping wells => lower (more negative) ground state
    assert result_near["eigenvalues"][0] < result_far["eigenvalues"][0]
