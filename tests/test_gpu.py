"""Tests pour l'accélération GPU du solveur de Schrödinger.

Tous les tests GPU sont skipés si CuPy n'est pas disponible.
"""
import time
import numpy as np
import pytest
from scipy.sparse import random as sparse_random, eye as sparse_eye

from simulation.gpu import GPU_AVAILABLE, gpu_info, sparse_eigsh

gpu_required = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="CuPy non disponible (pas de GPU ou CUDA non installé)"
)


def test_gpu_info():
    """gpu_info() retourne un dict avec 'available' key."""
    info = gpu_info()
    assert "available" in info
    if not GPU_AVAILABLE:
        assert info["available"] is False


def test_sparse_eigsh_cpu():
    """sparse_eigsh fonctionne en mode CPU (use_gpu=False)."""
    n = 100
    np.random.seed(42)
    A = sparse_random(n, n, density=0.1, format='csr')
    H = (A + A.T) / 2 + 5 * sparse_eye(n, format='csr')
    M = sparse_eye(n, format='csr')
    eigenvalues, eigenvectors = sparse_eigsh(H, M, k=5, use_gpu=False)
    assert eigenvalues.shape == (5,)
    assert eigenvectors.shape == (n, 5)
    assert np.all(np.diff(eigenvalues) >= -1e-10)


@gpu_required
def test_gpu_available():
    """Vérifie que CuPy est importable et un GPU détecté."""
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() >= 1


@gpu_required
def test_gpu_eigsh_matches_cpu():
    """Les eigenvalues GPU doivent correspondre aux CPU (tolérance 1e-6)."""
    n = 200
    np.random.seed(42)
    A = sparse_random(n, n, density=0.1, format='csr')
    H = (A + A.T) / 2 + 10 * sparse_eye(n, format='csr')
    M = sparse_eye(n, format='csr')

    evals_cpu, _ = sparse_eigsh(H, M, k=5, use_gpu=False)
    evals_gpu, _ = sparse_eigsh(H, M, k=5, use_gpu=True)

    np.testing.assert_allclose(evals_cpu, evals_gpu, atol=1e-6)


@gpu_required
def test_gpu_solve_knot_trefoil():
    """solve_knot('3_1', use_gpu=True) produit des résultats valides."""
    from simulation.solver import solve_knot
    result = solve_knot("3_1", n_coarse=10, n_eigenvalues=5, refine=False, use_gpu=True)
    assert "eigenvalues" in result
    assert len(result["eigenvalues"]) == 5
    assert np.all(np.isfinite(result["eigenvalues"]))


@gpu_required
def test_gpu_speedup():
    """Benchmark GPU vs CPU (informatif, pas d'assertion de speedup)."""
    from simulation.solver import solve_knot

    # Warmup GPU
    solve_knot("3_1", n_coarse=10, n_eigenvalues=5, refine=False, use_gpu=True)

    t0 = time.time()
    solve_knot("3_1", n_coarse=15, n_eigenvalues=5, refine=False, use_gpu=False)
    cpu_time = time.time() - t0

    t0 = time.time()
    solve_knot("3_1", n_coarse=15, n_eigenvalues=5, refine=False, use_gpu=True)
    gpu_time = time.time() - t0

    print(f"\nCPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s, "
          f"speedup: {cpu_time / max(gpu_time, 1e-6):.1f}x")
