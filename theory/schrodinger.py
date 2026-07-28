"""
Théorie Nodale v2 — Solveur de Schrödinger Nodal (FEM semi-structuré)

Résout [-ℏ²/(2m) ∇² + V_retract(r, K)] Ψ = E Ψ dans R³ \\ K
avec V_retract = -λ·ρ(K) / d(r,K)²

Self-consistent mode (omega_sq != None):
  The conformal metric g_ij = Ω² δ_ij with Ω² = 1 + κ|Ψ|²
  modifies the FEM weak form:
    ∫ ℏ²/(2m) Ω ∇Ψ·∇φ d³x + ∫ V Ω³ Ψ φ d³x = E ∫ Ω³ Ψ φ d³x
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def generate_mesh(
    curve: np.ndarray,
    n_coarse: int = 30,
    epsilon: float = 0.2,
    refine: bool = True,
    box_margin: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a semi-structured tetrahedral mesh around a knot curve.

    Parameters
    ----------
    curve : (M, 3) array — parametric knot curve points
    n_coarse : int — grid points per axis for the coarse grid
    epsilon : float — exclusion tube radius around the knot
    refine : bool — whether to add refined points near the knot
    box_margin : float — multiplier for bounding box size

    Returns
    -------
    points : (N, 3) array — mesh vertices
    tetrahedra : (T, 4) array — tetrahedron vertex indices
    boundary : (N,) bool array — True for boundary nodes
    """
    # 1. Bounding box
    max_extent = np.max(np.abs(curve))
    L = max_extent * box_margin
    knot_tree = cKDTree(curve)

    # 2. Coarse grid
    axis = np.linspace(-L, L, n_coarse)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
    coarse_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # 3. Compute distances, remove exclusion zone
    dists, _ = knot_tree.query(coarse_pts)
    outside = dists >= epsilon
    coarse_pts = coarse_pts[outside]

    # 4. Refinement: add fine grid points in the refinement zone
    if refine:
        n_fine = 2 * n_coarse - 1
        axis_fine = np.linspace(-L, L, n_fine)
        gx_f, gy_f, gz_f = np.meshgrid(axis_fine, axis_fine, axis_fine,
                                         indexing='ij')
        fine_pts = np.column_stack([gx_f.ravel(), gy_f.ravel(), gz_f.ravel()])
        dists_fine, _ = knot_tree.query(fine_pts)
        # Keep only fine points in refinement zone (epsilon <= d < 3*epsilon)
        in_refine = (dists_fine >= epsilon) & (dists_fine < 3 * epsilon)
        refine_pts = fine_pts[in_refine]
        all_pts = np.vstack([coarse_pts, refine_pts])
        # Remove duplicate points (within h/4 tolerance)
        h = 2 * L / n_coarse
        unique_tree = cKDTree(all_pts)
        pairs = unique_tree.query_pairs(r=h / 4)
        remove = set()
        for i, j in pairs:
            remove.add(max(i, j))  # keep the lower index
        keep = np.array([i for i in range(len(all_pts)) if i not in remove])
        all_pts = all_pts[keep]
    else:
        all_pts = coarse_pts

    # 5. Delaunay tetrahedralization
    tri = Delaunay(all_pts)
    tets = tri.simplices

    # 6. Remove tetrahedra whose barycentre is inside exclusion zone
    barycentres = all_pts[tets].mean(axis=1)  # (T, 3)
    bary_dists, _ = knot_tree.query(barycentres)
    valid = bary_dists >= epsilon
    tets = tets[valid]

    # 7. Boundary mask: box faces + near-exclusion-zone surface
    dists_final, _ = knot_tree.query(all_pts)
    on_box = np.any(np.abs(np.abs(all_pts) - L) < 1e-10, axis=1)
    near_tube = dists_final < epsilon * 1.5
    boundary = on_box | near_tube

    return all_pts, tets, boundary


def generate_mesh_multi(
    curves: list[np.ndarray],
    n_coarse: int = 30,
    epsilon: float = 0.2,
    refine: bool = True,
    box_margin: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a tetrahedral mesh around multiple knot curves.

    Excludes the epsilon-tube around ALL curves. Refinement zone
    applies near all curves. Bounding box encloses all curves.

    Parameters
    ----------
    curves : list of (M_i, 3) arrays -- parametric knot curves
    """
    all_curve_pts = np.vstack(curves)
    max_extent = np.max(np.abs(all_curve_pts))
    L = max_extent * box_margin
    knot_trees = [cKDTree(c) for c in curves]

    # Coarse grid
    axis = np.linspace(-L, L, n_coarse)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
    coarse_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Min distance to any curve
    def min_dist_to_curves(pts):
        dists = np.full(len(pts), np.inf)
        for tree in knot_trees:
            d, _ = tree.query(pts)
            dists = np.minimum(dists, d)
        return dists

    dists = min_dist_to_curves(coarse_pts)
    outside = dists >= epsilon
    coarse_pts = coarse_pts[outside]

    # Refinement
    if refine:
        n_fine = 2 * n_coarse - 1
        axis_fine = np.linspace(-L, L, n_fine)
        gx_f, gy_f, gz_f = np.meshgrid(axis_fine, axis_fine, axis_fine,
                                         indexing='ij')
        fine_pts = np.column_stack([gx_f.ravel(), gy_f.ravel(), gz_f.ravel()])
        dists_fine = min_dist_to_curves(fine_pts)
        in_refine = (dists_fine >= epsilon) & (dists_fine < 3 * epsilon)
        refine_pts = fine_pts[in_refine]
        all_pts = np.vstack([coarse_pts, refine_pts])
        h = 2 * L / n_coarse
        unique_tree = cKDTree(all_pts)
        pairs = unique_tree.query_pairs(r=h / 4)
        remove = set()
        for i, j in pairs:
            remove.add(max(i, j))
        keep = np.array([i for i in range(len(all_pts)) if i not in remove])
        all_pts = all_pts[keep]
    else:
        all_pts = coarse_pts

    # Delaunay
    tri = Delaunay(all_pts)
    tets = tri.simplices

    # Remove tets inside any exclusion zone
    barycentres = all_pts[tets].mean(axis=1)
    bary_dists = min_dist_to_curves(barycentres)
    tets = tets[bary_dists >= epsilon]

    # Boundary
    dists_final = min_dist_to_curves(all_pts)
    on_box = np.any(np.abs(np.abs(all_pts) - L) < 1e-10, axis=1)
    near_tube = dists_final < epsilon * 1.5
    boundary = on_box | near_tube

    return all_pts, tets, boundary


# --- FEM element computations ---

def tet_volume(verts: np.ndarray) -> float:
    """Volume of a tetrahedron with vertices (4, 3)."""
    J = (verts[1:] - verts[0]).T  # 3×3, columns = edge vectors
    return abs(np.linalg.det(J)) / 6.0


def tet_gradients(verts: np.ndarray) -> np.ndarray:
    """Gradient matrix B (3×4) for P1 shape functions on a tetrahedron.

    B[:, i] = gradient of the i-th shape function in physical space.
    """
    J = (verts[1:] - verts[0]).T  # 3×3
    J_inv_T = np.linalg.inv(J).T  # 3×3
    # Reference gradients: ∇_ξ N_0 = [-1,-1,-1], N_1=[1,0,0], N_2=[0,1,0], N_3=[0,0,1]
    grad_ref = np.array([
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
    ], dtype=float)
    return J_inv_T @ grad_ref  # (3, 4)


def local_stiffness(verts: np.ndarray, hbar: float = 1.0,
                     mass: float = 1.0) -> np.ndarray:
    """Local stiffness matrix (4×4) for the kinetic energy term.

    K_ij = (ℏ²/2m) * V_e * B[:,i] · B[:,j]
    """
    V = tet_volume(verts)
    B = tet_gradients(verts)
    coeff = hbar**2 / (2.0 * mass)
    return coeff * V * (B.T @ B)  # (4, 4)


def local_mass(verts: np.ndarray) -> np.ndarray:
    """Local mass matrix (4×4) for P1 elements.

    M_ij = V_e * (1 + delta_ij) / 20
    """
    V = tet_volume(verts)
    M = np.ones((4, 4)) * V / 20.0
    np.fill_diagonal(M, 2.0 * V / 20.0)
    return M


# --- Gauss quadrature ---

def gauss_points_tet(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-point Gauss quadrature on a tetrahedron (degree 2 exact).

    Returns
    -------
    points : (4, 3) array — quadrature points in physical space
    weights : (4,) array — quadrature weights (sum = volume)
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b, a, a, a],
        [a, b, a, a],
        [a, a, b, a],
        [a, a, a, b],
    ])
    points = bary @ verts  # (4, 3)
    V = tet_volume(verts)
    weights = np.full(4, V / 4.0)
    return points, weights


def local_potential_matrix(verts: np.ndarray,
                            V_at_quad: np.ndarray) -> np.ndarray:
    """Local potential matrix (4×4) using 4-point Gauss quadrature.

    V_pot_ij = sum_q w_q * V(r_q) * N_i(r_q) * N_j(r_q)
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b, a, a, a],
        [a, b, a, a],
        [a, a, b, a],
        [a, a, a, b],
    ])
    V_e = tet_volume(verts)
    P = np.zeros((4, 4))
    for q in range(4):
        N = bary[q]  # shape function values = barycentric coords
        P += (V_e / 4.0) * V_at_quad[q] * np.outer(N, N)
    return P


# --- Global assembly and eigenvalue solver ---

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from theory.postulates import HBAR, LAMBDA


def solve_eigenproblem(
    H: csr_matrix,
    M: csr_matrix,
    n_eigenvalues: int = 10,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenvalue problem H phi = E M phi.

    Returns the n smallest eigenvalues and corresponding eigenvectors.
    Falls back to shift-invert mode if SA mode fails.

    Parameters
    ----------
    use_gpu : bool — si True, utilise CuPy/GPU via simulation.gpu
    """
    from simulation.gpu import sparse_eigsh
    return sparse_eigsh(H, M, k=n_eigenvalues, use_gpu=use_gpu)


from theory.chi_field import chi_retraction


def assemble_system(
    points: np.ndarray,
    tets: np.ndarray,
    boundary: np.ndarray,
    knot_name: str,
    knot_tree: cKDTree,
    energy: float = 0.0,
    epsilon: float = 0.2,
    omega_sq: np.ndarray | None = None,
    metric_mode: str = "conformal",
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble Hamiltonian using Jones-based energy-dependent rho(K, E).

    Same FEM assembly as assemble_system but with chi_retraction for rho.

    When omega_sq is provided (shape (n_pts,)), the metric modifies
    the weak form according to metric_mode:
      - "conformal": K×Ω, V×Ω³, M×Ω³ (métrique conforme standard)
      - "potential": K inchangé, V×Ω², M inchangé (amplification potentiel)
    """
    n_pts = len(points)
    rho = chi_retraction(knot_name, energy)
    coeff_kinetic = HBAR**2 / 2.0

    free_mask = ~boundary
    free_indices = np.where(free_mask)[0]
    n_free = len(free_indices)
    global_to_free = -np.ones(n_pts, dtype=int)
    global_to_free[free_indices] = np.arange(n_free)

    a_g = (5.0 - np.sqrt(5.0)) / 20.0
    b_g = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b_g, a_g, a_g, a_g],
        [a_g, b_g, a_g, a_g],
        [a_g, a_g, b_g, a_g],
        [a_g, a_g, a_g, b_g],
    ])

    grad_ref = np.array([
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
    ], dtype=float)

    all_verts = points[tets]
    edges = all_verts[:, 1:, :] - all_verts[:, 0:1, :]
    J = np.transpose(edges, (0, 2, 1))
    volumes = np.abs(np.linalg.det(J)) / 6.0

    valid_mask = volumes > 1e-15
    valid_tets = tets[valid_mask]
    valid_verts = all_verts[valid_mask]
    valid_J = J[valid_mask]
    valid_volumes = volumes[valid_mask]
    n_valid = len(valid_tets)

    J_inv_T = np.linalg.inv(valid_J).transpose(0, 2, 1)
    B = J_inv_T @ grad_ref

    K_local = coeff_kinetic * valid_volumes[:, None, None] * np.einsum(
        'eji,ejk->eik', B, B
    )

    # --- Metric modifications ---
    omega_scale_V = None
    if omega_sq is not None:
        omega_sq_nodes = omega_sq[valid_tets]  # (n_valid, 4)
        omega_nodes = np.sqrt(omega_sq_nodes)   # (n_valid, 4)

        if metric_mode == "conformal":
            # Mode A (standard): K×Ω, V×Ω³, M×Ω³
            omega3_nodes = omega_sq_nodes * omega_nodes
            omega_avg = omega_nodes.mean(axis=1)
            K_local *= omega_avg[:, None, None]
            omega_scale_V = np.einsum('qi,ei->eq', bary, omega3_nodes)
        elif metric_mode == "potential":
            # Mode C: K inchangé, V×Ω², M inchangé
            omega_scale_V = np.einsum('qi,ei->eq', bary, omega_sq_nodes)
        else:
            raise ValueError(f"metric_mode inconnu: {metric_mode}")

    # --- Potential at quadrature points ---
    quad_pts = np.einsum('qi,eij->eqj', bary, valid_verts)
    quad_pts_flat = quad_pts.reshape(-1, 3)
    dists_flat, _ = knot_tree.query(quad_pts_flat)
    dists_flat = np.maximum(dists_flat, epsilon)
    V_flat = -LAMBDA * rho / dists_flat**2
    V_quad = V_flat.reshape(n_valid, 4)

    if omega_scale_V is not None:
        V_quad = V_quad * omega_scale_V

    P_local = np.zeros((n_valid, 4, 4))
    for qq in range(4):
        N_q = bary[qq]
        outer_N = np.outer(N_q, N_q)
        P_local += (valid_volumes[:, None, None] / 4.0) * V_quad[:, qq:qq+1, None] * outer_N

    H_local = K_local + P_local

    # --- Mass matrix ---
    if omega_sq is not None and metric_mode == "conformal":
        # Mode conformal: Quadrature-based mass with Ω³ weights
        omega3_nodes_m = omega_sq[valid_tets] * np.sqrt(omega_sq[valid_tets])
        omega3_quad_m = np.einsum('qi,ei->eq', bary, omega3_nodes_m)
        M_local = np.zeros((n_valid, 4, 4))
        for qq in range(4):
            N_q = bary[qq]
            outer_N = np.outer(N_q, N_q)
            M_local += (valid_volumes[:, None, None] / 4.0) * omega3_quad_m[:, qq:qq+1, None] * outer_N
    else:
        # Analytic mass (original formula)
        M_local = np.ones((n_valid, 4, 4)) * valid_volumes[:, None, None] / 20.0
        idx = np.arange(4)
        M_local[:, idx, idx] = 2.0 * valid_volumes[:, None] / 20.0

    gi_all = global_to_free[valid_tets]
    ii = gi_all[:, np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])]
    jj = gi_all[:, np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])]
    hv = H_local.reshape(n_valid, 16)
    mv = M_local.reshape(n_valid, 16)

    mask = (ii >= 0) & (jj >= 0)
    rows = ii[mask]
    cols = jj[mask]
    h_vals = hv[mask]
    m_vals = mv[mask]

    H = coo_matrix((h_vals, (rows, cols)), shape=(n_free, n_free)).tocsr()
    M_mat = coo_matrix((m_vals, (rows, cols)), shape=(n_free, n_free)).tocsr()
    H = (H + H.T) / 2
    M_mat = (M_mat + M_mat.T) / 2

    return H, M_mat


def assemble_system_multi(
    points: np.ndarray,
    tets: np.ndarray,
    boundary: np.ndarray,
    knot_names: list[str],
    curves: list[np.ndarray],
    energy: float = 0.0,
    epsilon: float = 0.2,
    omega_sq: np.ndarray | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble Hamiltonian for multi-knot link (complement commun).

    The potential is the sum of individual knot potentials:
        V(r) = Sum_i V(r, K_i) = Sum_i -lambda * rho(K_i, E) / d(r, K_i)^2

    Parameters
    ----------
    knot_names : list of knot identifiers for each component
    curves : list of (M_i, 3) arrays for each knot curve
    """
    n_pts = len(points)
    coeff_kinetic = HBAR**2 / 2.0

    free_mask = ~boundary
    free_indices = np.where(free_mask)[0]
    n_free = len(free_indices)
    global_to_free = -np.ones(n_pts, dtype=int)
    global_to_free[free_indices] = np.arange(n_free)

    a_g = (5.0 - np.sqrt(5.0)) / 20.0
    b_g = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    bary = np.array([
        [b_g, a_g, a_g, a_g],
        [a_g, b_g, a_g, a_g],
        [a_g, a_g, b_g, a_g],
        [a_g, a_g, a_g, b_g],
    ])

    grad_ref = np.array([
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
    ], dtype=float)

    all_verts = points[tets]
    edges = all_verts[:, 1:, :] - all_verts[:, 0:1, :]
    J = np.transpose(edges, (0, 2, 1))
    volumes = np.abs(np.linalg.det(J)) / 6.0

    valid_mask = volumes > 1e-15
    valid_tets = tets[valid_mask]
    valid_verts = all_verts[valid_mask]
    valid_J = J[valid_mask]
    valid_volumes = volumes[valid_mask]
    n_valid = len(valid_tets)

    J_inv_T = np.linalg.inv(valid_J).transpose(0, 2, 1)
    B = J_inv_T @ grad_ref

    K_local = coeff_kinetic * valid_volumes[:, None, None] * np.einsum(
        'eji,ejk->eik', B, B
    )

    # Conformal metric (same as assemble_system)
    if omega_sq is not None:
        omega_sq_nodes = omega_sq[valid_tets]
        omega_nodes = np.sqrt(omega_sq_nodes)
        omega3_nodes = omega_sq_nodes * omega_nodes
        omega_avg = omega_nodes.mean(axis=1)
        K_local *= omega_avg[:, None, None]
        omega3_quad = np.einsum('qi,ei->eq', bary, omega3_nodes)

    # Potential: sum over all knot components
    quad_pts = np.einsum('qi,eij->eqj', bary, valid_verts)
    quad_pts_flat = quad_pts.reshape(-1, 3)

    V_flat = np.zeros(len(quad_pts_flat))
    knot_trees = [cKDTree(c) for c in curves]
    for name, tree in zip(knot_names, knot_trees):
        rho = chi_retraction(name, energy)
        dists, _ = tree.query(quad_pts_flat)
        dists = np.maximum(dists, epsilon)
        V_flat += -LAMBDA * rho / dists**2

    V_quad = V_flat.reshape(n_valid, 4)

    if omega_sq is not None:
        V_quad = V_quad * omega3_quad

    P_local = np.zeros((n_valid, 4, 4))
    for qq in range(4):
        N_q = bary[qq]
        outer_N = np.outer(N_q, N_q)
        P_local += (valid_volumes[:, None, None] / 4.0) * V_quad[:, qq:qq+1, None] * outer_N

    H_local = K_local + P_local

    # Mass matrix (same as assemble_system)
    if omega_sq is not None:
        M_local = np.zeros((n_valid, 4, 4))
        for qq in range(4):
            N_q = bary[qq]
            outer_N = np.outer(N_q, N_q)
            M_local += (valid_volumes[:, None, None] / 4.0) * omega3_quad[:, qq:qq+1, None] * outer_N
    else:
        M_local = np.ones((n_valid, 4, 4)) * valid_volumes[:, None, None] / 20.0
        idx = np.arange(4)
        M_local[:, idx, idx] = 2.0 * valid_volumes[:, None] / 20.0

    gi_all = global_to_free[valid_tets]
    ii = gi_all[:, np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])]
    jj = gi_all[:, np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])]
    hv = H_local.reshape(n_valid, 16)
    mv = M_local.reshape(n_valid, 16)

    mask = (ii >= 0) & (jj >= 0)
    rows = ii[mask]
    cols = jj[mask]
    h_vals = hv[mask]
    m_vals = mv[mask]

    H = coo_matrix((h_vals, (rows, cols)), shape=(n_free, n_free)).tocsr()
    M_mat = coo_matrix((m_vals, (rows, cols)), shape=(n_free, n_free)).tocsr()
    H = (H + H.T) / 2
    M_mat = (M_mat + M_mat.T) / 2

    return H, M_mat
