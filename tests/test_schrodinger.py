# tests/test_schrodinger.py
import numpy as np
from theory.schrodinger import (
    generate_mesh,
    tet_volume, tet_gradients, local_stiffness, local_mass,
    gauss_points_tet, local_potential_matrix,
    assemble_system, solve_eigenproblem,
)



def test_generate_mesh_returns_points_and_tets():
    from theory.knot_catalog import knot_curve
    curve = knot_curve("0_1", 200)
    points, tets, boundary = generate_mesh(curve, n_coarse=10, epsilon=0.2)
    assert points.ndim == 2 and points.shape[1] == 3
    assert tets.ndim == 2 and tets.shape[1] == 4
    assert boundary.shape == (len(points),)
    assert boundary.dtype == bool


def test_mesh_excludes_knot_tube():
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    curve = knot_curve("3_1", 300)
    epsilon = 0.3
    points, tets, boundary = generate_mesh(curve, n_coarse=15, epsilon=epsilon)
    tree = cKDTree(curve)
    dists, _ = tree.query(points)
    assert np.all(dists >= epsilon * 0.9)  # all points outside tube


def test_mesh_has_refinement_near_knot():
    from theory.knot_catalog import knot_curve
    curve = knot_curve("3_1", 300)
    epsilon = 0.3
    points_coarse, _, _ = generate_mesh(curve, n_coarse=15, epsilon=epsilon,
                                         refine=False)
    points_refined, _, _ = generate_mesh(curve, n_coarse=15, epsilon=epsilon,
                                          refine=True)
    assert len(points_refined) > len(points_coarse)


def test_mesh_boundary_includes_box_faces():
    from theory.knot_catalog import knot_curve
    curve = knot_curve("0_1", 200)
    points, tets, boundary = generate_mesh(curve, n_coarse=10, epsilon=0.2)
    L = np.max(np.abs(points))
    on_face = (np.abs(np.abs(points) - L) < 1e-10).any(axis=1)
    assert np.all(boundary[on_face])


# --- Task 2: FEM element computations ---

def test_tet_volume_unit_tet():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    vol = tet_volume(verts)
    assert np.isclose(vol, 1.0 / 6.0)


def test_tet_volume_positive():
    verts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    vol = tet_volume(verts)
    assert vol > 0
    assert np.isclose(vol, 8.0 / 6.0)


def test_tet_gradients_shape():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    B = tet_gradients(verts)
    assert B.shape == (3, 4)


def test_tet_gradients_sum_to_zero():
    """Gradients of P1 shape functions sum to zero (partition of unity)."""
    verts = np.array([[1, 2, 0], [4, 1, 0], [2, 5, 0], [2, 2, 3]], dtype=float)
    B = tet_gradients(verts)
    assert np.allclose(B.sum(axis=1), 0.0)


def test_local_stiffness_symmetric():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    K = local_stiffness(verts, hbar=1.0, mass=1.0)
    assert K.shape == (4, 4)
    assert np.allclose(K, K.T)


def test_local_mass_symmetric():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    M = local_mass(verts)
    assert M.shape == (4, 4)
    assert np.allclose(M, M.T)


def test_local_mass_unit_tet_values():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    M = local_mass(verts)
    V = 1.0 / 6.0
    assert np.isclose(M[0, 0], V * 2 / 20)  # diagonal: V*(1+1)/20
    assert np.isclose(M[0, 1], V * 1 / 20)  # off-diagonal: V*1/20


# --- Task 3: Gauss quadrature and potential matrix ---

def test_gauss_points_tet_shape():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pts, weights = gauss_points_tet(verts)
    assert pts.shape == (4, 3)
    assert weights.shape == (4,)


def test_gauss_points_inside_tet():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pts, _ = gauss_points_tet(verts)
    for pt in pts:
        assert np.all(pt >= -1e-10)
        assert pt.sum() <= 1.0 + 1e-10


def test_gauss_weights_sum():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    _, weights = gauss_points_tet(verts)
    V = 1.0 / 6.0
    assert np.isclose(weights.sum(), V)


def test_local_potential_symmetric():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    V_vals = np.array([-1.0, -1.0, -1.0, -1.0])
    P = local_potential_matrix(verts, V_vals)
    assert P.shape == (4, 4)
    assert np.allclose(P, P.T)


def test_local_potential_attractive():
    """Negative potential should give negative matrix entries on diagonal."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    V_vals = np.array([-10.0, -10.0, -10.0, -10.0])
    P = local_potential_matrix(verts, V_vals)
    assert np.all(np.diag(P) < 0)


# --- Task 6 & 7: Visualization ---

def test_plot_3d_isosurface_creates_file(tmp_path):
    from simulation.solver import solve_knot
    from simulation.visualization import plot_3d_isosurface
    result = solve_knot("0_1", n_coarse=8, n_eigenvalues=3)
    outfile = str(tmp_path / "isosurface.png")
    plot_3d_isosurface(result, state_index=0, output_path=outfile)
    import os
    assert os.path.exists(outfile)


def test_plot_density_cut_creates_file(tmp_path):
    from simulation.solver import solve_knot
    from simulation.visualization import plot_density_cut
    result = solve_knot("0_1", n_coarse=8, n_eigenvalues=3)
    outfile = str(tmp_path / "density.png")
    plot_density_cut(result, state_index=0, output_path=outfile)
    import os
    assert os.path.exists(outfile)


def test_plot_energy_spectrum_creates_file(tmp_path):
    from simulation.solver import solve_knot
    from simulation.visualization import plot_energy_spectrum
    result = solve_knot("0_1", n_coarse=8, n_eigenvalues=5)
    outfile = str(tmp_path / "spectrum.png")
    plot_energy_spectrum(result, output_path=outfile)
    import os
    assert os.path.exists(outfile)


def test_plot_correlation_creates_file(tmp_path):
    from simulation.visualization import plot_correlation_E0_rho
    results = [
        {"knot_name": "0_1", "eigenvalues": np.array([-1.0]), "retraction_factor": 0.1},
        {"knot_name": "3_1", "eigenvalues": np.array([-2.0]), "retraction_factor": 1.3},
    ]
    outfile = str(tmp_path / "correlation.png")
    plot_correlation_E0_rho(results, output_path=outfile)
    import os
    assert os.path.exists(outfile)


# --- Task 4: Global assembly and eigensolver ---

def test_assemble_system_shapes():
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    curve = knot_curve("0_1", 200)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(curve, n_coarse=8, epsilon=0.2)
    H, M = assemble_system(points, tets, boundary, "0_1", knot_tree, epsilon=0.2)
    n_free = int((~boundary).sum())
    assert H.shape == (n_free, n_free)
    assert M.shape == (n_free, n_free)


def test_assemble_system_symmetric():
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    curve = knot_curve("0_1", 200)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(curve, n_coarse=8, epsilon=0.2)
    H, M = assemble_system(points, tets, boundary, "0_1", knot_tree, epsilon=0.2)
    diff_H = H - H.T
    diff_M = M - M.T
    assert diff_H.nnz == 0 or np.max(np.abs(diff_H.data)) < 1e-12
    assert diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-12


def test_solve_eigenproblem_returns_eigenvalues():
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    curve = knot_curve("0_1", 200)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(curve, n_coarse=8, epsilon=0.2)
    H, M = assemble_system(points, tets, boundary, "0_1", knot_tree, epsilon=0.2)
    eigenvalues, eigenvectors = solve_eigenproblem(H, M, n_eigenvalues=5)
    assert len(eigenvalues) == 5
    assert eigenvalues[0] <= eigenvalues[1]  # sorted


def test_solve_eigenproblem_eigenvectors_shape():
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    curve = knot_curve("0_1", 200)
    knot_tree = cKDTree(curve)
    points, tets, boundary = generate_mesh(curve, n_coarse=8, epsilon=0.2)
    H, M = assemble_system(points, tets, boundary, "0_1", knot_tree, epsilon=0.2)
    eigenvalues, eigenvectors = solve_eigenproblem(H, M, n_eigenvalues=5)
    n_free = H.shape[0]
    assert eigenvectors.shape == (n_free, 5)


# --- Task 4b: Multi-curve mesh generation ---

def test_generate_mesh_multi_returns_points_and_tets():
    """Multi-curve mesh has same output structure as single-curve."""
    from theory.schrodinger import generate_mesh_multi
    from theory.knot_catalog import knot_curve
    import numpy as np
    curve1 = knot_curve("3_1", 200)
    curve2 = knot_curve("3_1", 200) + np.array([8.0, 0.0, 0.0])
    points, tets, boundary = generate_mesh_multi(
        [curve1, curve2], n_coarse=10, epsilon=0.2, refine=False
    )
    assert points.ndim == 2 and points.shape[1] == 3
    assert tets.ndim == 2 and tets.shape[1] == 4
    assert boundary.shape == (len(points),)

def test_generate_mesh_multi_excludes_both_tubes():
    """Points are excluded from BOTH knot tubes."""
    from theory.schrodinger import generate_mesh_multi
    from theory.knot_catalog import knot_curve
    from scipy.spatial import cKDTree
    import numpy as np
    curve1 = knot_curve("0_1", 200)
    curve2 = knot_curve("0_1", 200) + np.array([5.0, 0.0, 0.0])
    epsilon = 0.3
    points, tets, boundary = generate_mesh_multi(
        [curve1, curve2], n_coarse=12, epsilon=epsilon, refine=False
    )
    tree1 = cKDTree(curve1)
    tree2 = cKDTree(curve2)
    d1, _ = tree1.query(points)
    d2, _ = tree2.query(points)
    min_dist = np.minimum(d1, d2)
    assert np.all(min_dist >= epsilon * 0.99)

def test_generate_mesh_multi_single_curve_matches():
    """Single-curve multi mesh should be similar to regular mesh."""
    from theory.schrodinger import generate_mesh, generate_mesh_multi
    from theory.knot_catalog import knot_curve
    curve = knot_curve("3_1", 200)
    pts1, _, _ = generate_mesh(curve, n_coarse=10, epsilon=0.2, refine=False)
    pts2, _, _ = generate_mesh_multi([curve], n_coarse=10, epsilon=0.2, refine=False)
    # Same number of points (same algorithm)
    assert len(pts1) == len(pts2)

def test_assemble_system_multi_shapes():
    """Multi-knot assembly produces valid sparse matrices."""
    from theory.schrodinger import generate_mesh_multi, assemble_system_multi
    from theory.knot_catalog import knot_curve
    import numpy as np
    curve1 = knot_curve("3_1", 200)
    curve2 = knot_curve("3_1", 200) + np.array([8.0, 0.0, 0.0])
    points, tets, boundary = generate_mesh_multi(
        [curve1, curve2], n_coarse=10, epsilon=0.2, refine=False
    )
    H, M = assemble_system_multi(
        points, tets, boundary,
        knot_names=["3_1", "3_1"],
        curves=[curve1, curve2],
        energy=0.0, epsilon=0.2,
    )
    n_free = int((~boundary).sum())
    assert H.shape == (n_free, n_free)
    assert M.shape == (n_free, n_free)

def test_assemble_system_multi_symmetric():
    from theory.schrodinger import generate_mesh_multi, assemble_system_multi
    from theory.knot_catalog import knot_curve
    import numpy as np
    curve1 = knot_curve("0_1", 200)
    curve2 = knot_curve("0_1", 200) + np.array([5.0, 0.0, 0.0])
    points, tets, boundary = generate_mesh_multi(
        [curve1, curve2], n_coarse=10, epsilon=0.2, refine=False
    )
    H, M = assemble_system_multi(
        points, tets, boundary,
        knot_names=["0_1", "0_1"],
        curves=[curve1, curve2],
        energy=0.0, epsilon=0.2,
    )
    assert abs(H - H.T).max() < 1e-12
    assert abs(M - M.T).max() < 1e-12
