"""Theorie Nodale v4 -- Solveur FEM Multi-Noeuds

Resout l'equation de Schrodinger sur le complement commun
S^3 \\ (K_1 U K_2 U ... U K_n) avec potentiel V(r) = Sum_i V(r, K_i).
"""
import numpy as np
from theory.knot_catalog import knot_curve
from theory.schrodinger import (
    generate_mesh_multi,
    assemble_system_multi,
    solve_eigenproblem,
)


def _place_curves(
    knot_names: list[str],
    distance: float,
    n_curve_points: int = 500,
    auto_scale: bool = True,
    knot_scale: float = 0.3,
) -> tuple[list[np.ndarray], float]:
    """Place knot curves evenly spaced along the x-axis.

    For n knots with inter-knot distance d, centers are at:
        x_i = (i - (n-1)/2) * d   for i = 0, ..., n-1

    Parameters
    ----------
    auto_scale : bool -- if True, rescale each knot curve so its extent
        is knot_scale * distance, preventing overlap at small d.
    knot_scale : float -- fraction of d used as knot radius (default 0.3).

    Returns
    -------
    curves : list of (M, 3) arrays
    scale_ratio : float -- ratio of scaled extent to original extent
        (1.0 if no scaling applied)
    """
    n = len(knot_names)
    curves = []
    scale_ratio = 1.0
    for i, name in enumerate(knot_names):
        curve = knot_curve(name, n_curve_points)
        if auto_scale and n > 1:
            center = (curve.max(axis=0) + curve.min(axis=0)) / 2
            curve = curve - center
            max_ext = np.abs(curve).max()
            if max_ext > 0:
                target_radius = knot_scale * distance / 2
                scale_ratio = target_radius / max_ext
                curve = curve * scale_ratio
        offset = np.array([(i - (n - 1) / 2) * distance, 0.0, 0.0])
        curves.append(curve + offset)
    return curves, scale_ratio


def solve_link(
    knot_names: list[str],
    distance: float,
    n_coarse: int = 30,
    epsilon: float = 0.2,
    n_eigenvalues: int = 10,
    refine: bool = True,
    n_curve_points: int = 500,
    use_gpu: bool = False,
    auto_scale: bool = True,
    knot_scale: float = 0.3,
) -> dict:
    """Solve Schrodinger on the common complement of multiple knots.

    Parameters
    ----------
    knot_names : list of knot identifiers
    distance : float -- inter-knot distance d
    n_coarse, epsilon, n_eigenvalues, refine : same as solve_knot
    use_gpu : bool -- use GPU for eigensolver
    auto_scale : bool -- rescale knot curves to fit within distance
    knot_scale : float -- fraction of d used as knot radius

    Returns
    -------
    dict with eigenvalues, eigenvectors, points, boundary, free_mask,
         knot_names, distance, n_components, scale_ratio
    """
    curves, scale_ratio = _place_curves(
        knot_names, distance, n_curve_points,
        auto_scale=auto_scale, knot_scale=knot_scale,
    )

    # Scale epsilon proportionally when knots are rescaled
    if auto_scale and len(knot_names) > 1:
        effective_epsilon = epsilon * scale_ratio
        # Ensure epsilon is at least resolvable by the mesh
        grid_spacing = 2 * np.max(np.abs(np.vstack(curves))) * 1.5 / n_coarse
        effective_epsilon = max(effective_epsilon, grid_spacing * 0.5)
    else:
        effective_epsilon = epsilon

    points, tets, boundary = generate_mesh_multi(
        curves, n_coarse=n_coarse, epsilon=effective_epsilon, refine=refine
    )

    H, M = assemble_system_multi(
        points, tets, boundary,
        knot_names=knot_names,
        curves=curves,
        energy=0.0,
        epsilon=effective_epsilon,
    )

    eigenvalues, eigenvectors = solve_eigenproblem(
        H, M, n_eigenvalues=n_eigenvalues, use_gpu=use_gpu
    )

    return {
        "knot_names": knot_names,
        "distance": distance,
        "n_components": len(knot_names),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "points": points,
        "boundary": boundary,
        "free_mask": ~boundary,
        "scale_ratio": scale_ratio,
        "effective_epsilon": effective_epsilon,
    }
