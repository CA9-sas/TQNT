"""
Théorie Nodale v2 — Interface Haut-Niveau du Solveur de Schrödinger Nodal
"""
import numpy as np
from scipy.spatial import cKDTree
from theory.knot_catalog import knot_curve
from theory.chi_field import jones_rho, chi_retraction, characteristic_energy
from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem
from theory.emergent_spacetime import emergent_metric_factor


def solve_knot(
    knot_name: str,
    n_coarse: int = 30,
    epsilon: float = 0.2,
    n_eigenvalues: int = 10,
    refine: bool = True,
    n_curve_points: int = 500,
    use_gpu: bool = False,
) -> dict:
    """Solve the Schrödinger equation for a given knot.

    Parameters
    ----------
    knot_name : str — Rolfsen table name (e.g. "3_1")
    n_coarse : int — grid resolution
    epsilon : float — exclusion tube radius
    n_eigenvalues : int — number of eigenvalues to compute
    refine : bool — whether to refine mesh near knot
    n_curve_points : int — number of points for the knot curve

    Returns
    -------
    dict with keys:
        knot_name, eigenvalues, eigenvectors, points, boundary,
        retraction_factor, retraction_mass, free_mask
    """
    curve = knot_curve(knot_name, n_curve_points)
    knot_tree = cKDTree(curve)

    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=epsilon, refine=refine
    )

    H, M = assemble_system(
        points, tets, boundary, knot_name, knot_tree, energy=0.0, epsilon=epsilon
    )

    eigenvalues, eigenvectors = solve_eigenproblem(
        H, M, n_eigenvalues=n_eigenvalues, use_gpu=use_gpu
    )

    rho = jones_rho(knot_name)

    return {
        "knot_name": knot_name,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "points": points,
        "boundary": boundary,
        "free_mask": ~boundary,
        "retraction_factor": rho,
        "retraction_mass": rho,
    }


def solve_knot_jones(
    knot_name: str,
    n_coarse: int = 30,
    epsilon: float = 0.2,
    n_eigenvalues: int = 10,
    refine: bool = True,
    n_curve_points: int = 500,
    max_iterations: int = 20,
    tol: float = 0.01,
    use_gpu: bool = False,
) -> dict:
    """Solve with Jones-based rho(K, E) using fixed-point iteration.

    The eigenvalue E depends on rho which depends on E:
    1. Start with E = 0 -> rho = rho_J
    2. Solve -> get E_0
    3. Update rho(K, E_0)
    4. Re-solve -> get new E_0
    5. Repeat until convergence
    """
    curve = knot_curve(knot_name, n_curve_points)
    knot_tree_obj = cKDTree(curve)

    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=epsilon, refine=refine
    )

    rho_J = jones_rho(knot_name)
    energy = 0.0  # initial guess
    damping = 0.5  # under-relaxation to stabilize fixed-point iteration

    for iteration in range(max_iterations):
        H, M = assemble_system(
            points, tets, boundary, knot_name, knot_tree_obj,
            energy=energy, epsilon=epsilon
        )
        eigenvalues, eigenvectors = solve_eigenproblem(
            H, M, n_eigenvalues=n_eigenvalues, use_gpu=use_gpu
        )
        new_energy = abs(eigenvalues[0])

        # Damped update: E_{n+1} = alpha * E_new + (1-alpha) * E_old
        blended = damping * new_energy + (1 - damping) * energy

        if abs(blended - energy) < tol * max(abs(energy), 1e-10):
            energy = blended
            break
        energy = blended

    rho_final = chi_retraction(knot_name, energy)
    E_K = characteristic_energy(knot_name)

    return {
        "knot_name": knot_name,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "points": points,
        "boundary": boundary,
        "free_mask": ~boundary,
        "retraction_factor_jones": rho_J,
        "retraction_factor_final": rho_final,
        "retraction_mass": rho_J,
        "retraction_factor": rho_J,
        "characteristic_energy": E_K,
        "iterations": iteration + 1,
        "converged": iteration < max_iterations - 1,
    }


def solve_knot_selfconsistent(
    knot_name: str,
    n_coarse: int = 30,
    epsilon: float = 0.2,
    n_eigenvalues: int = 10,
    refine: bool = True,
    n_curve_points: int = 500,
    max_iterations: int = 30,
    tol: float = 1e-4,
    use_gpu: bool = False,
    metric_mode: str = "conformal",
) -> dict:
    """Solve with self-consistent metric backreaction.

    Couples the Jones-based Schrödinger equation with the emergent
    conformal metric Ω² = 1 + κ|Ψ|². The wave shapes spacetime and
    spacetime shapes the wave, iterated to a fixed point.

    metric_mode:
      - "conformal": K×Ω, V×Ω³, M×Ω³ (standard, reduces |E|)
      - "potential": K unchanged, V×Ω², M unchanged (amplifies |E|)

    Convergence requires both energy and metric changes below tol.
    """
    from theory.postulates import KAPPA

    curve = knot_curve(knot_name, n_curve_points)
    knot_tree_obj = cKDTree(curve)

    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=epsilon, refine=refine
    )

    n_pts = len(points)
    free_mask = ~boundary
    rho_J = jones_rho(knot_name)

    energy = 0.0
    omega_sq = np.ones(n_pts)
    damping = 0.5

    converged = False
    for iteration in range(max_iterations):
        H, M = assemble_system(
            points, tets, boundary, knot_name, knot_tree_obj,
            energy=energy, epsilon=epsilon, omega_sq=omega_sq,
            metric_mode=metric_mode,
        )
        eigenvalues, eigenvectors = solve_eigenproblem(
            H, M, n_eigenvalues=n_eigenvalues, use_gpu=use_gpu
        )

        # Damped energy update
        new_energy = abs(eigenvalues[0])
        blended_energy = damping * new_energy + (1 - damping) * energy
        delta_energy = abs(blended_energy - energy)

        # Extract ground state Ψ on free DOFs → full array
        psi_free = eigenvectors[:, 0]
        psi_full = np.zeros(n_pts)
        psi_full[free_mask] = psi_free

        # New metric: Ω² = 1 + κ|Ψ|²
        new_omega_sq = emergent_metric_factor(psi_full)

        # Damped metric update
        blended_omega_sq = damping * new_omega_sq + (1 - damping) * omega_sq
        delta_metric = np.max(np.abs(blended_omega_sq - omega_sq))

        # Update state
        energy = blended_energy
        omega_sq = blended_omega_sq

        # Convergence check
        energy_scale = max(abs(energy), 1e-10)
        if delta_energy < tol * energy_scale and delta_metric < tol:
            converged = True
            break

    rho_final = chi_retraction(knot_name, energy)
    metric_correction = omega_sq - 1.0  # κ|Ψ|² contribution

    return {
        "knot_name": knot_name,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "points": points,
        "boundary": boundary,
        "free_mask": free_mask,
        "omega_sq": omega_sq,
        "metric_correction": metric_correction,
        "retraction_factor_jones": rho_J,
        "retraction_factor_final": rho_final,
        "retraction_factor": rho_J,
        "retraction_mass": rho_J,
        "characteristic_energy": characteristic_energy(knot_name),
        "iterations": iteration + 1,
        "converged": converged,
    }
