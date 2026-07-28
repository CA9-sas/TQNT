"""Backend GPU/CPU transparent pour le solveur FEM.

Fournit des fonctions qui utilisent CuPy (GPU) quand disponible,
avec fallback automatique sur NumPy/SciPy (CPU).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def gpu_info() -> dict:
    """Retourne les informations sur le GPU disponible."""
    if not GPU_AVAILABLE:
        return {"available": False, "reason": "CuPy non installé"}
    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        free, total = device.mem_info
        return {
            "available": True,
            "name": props["name"].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "total_memory_mb": total // (1024 * 1024),
            "free_memory_mb": free // (1024 * 1024),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def to_gpu(array: np.ndarray) -> "cp.ndarray":
    """Transfère un array NumPy vers le GPU."""
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy non disponible")
    return cp.asarray(array)


def to_cpu(array) -> np.ndarray:
    """Transfère un array CuPy vers le CPU (no-op si déjà NumPy)."""
    if isinstance(array, np.ndarray):
        return array
    return cp.asnumpy(array)


def sparse_eigsh(
    H: csr_matrix,
    M: csr_matrix,
    k: int = 10,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Résout le problème aux valeurs propres H phi = E M phi.

    Utilise CuPy sur GPU si use_gpu=True et GPU disponible,
    sinon SciPy sur CPU.

    Returns
    -------
    eigenvalues : (k,) array trié par ordre croissant
    eigenvectors : (n, k) array
    """
    if use_gpu and GPU_AVAILABLE:
        H_gpu = cp_sparse.csr_matrix(H)
        M_gpu = cp_sparse.csr_matrix(M)
        try:
            eigenvalues, eigenvectors = cp_splinalg.eigsh(
                H_gpu, k=k, M=M_gpu, which='SA'
            )
        except Exception:
            eigenvalues, eigenvectors = cp_splinalg.eigsh(
                H_gpu, k=k, M=M_gpu, sigma=-100.0, which='LM'
            )
        order = cp.argsort(eigenvalues)
        return cp.asnumpy(eigenvalues[order]), cp.asnumpy(eigenvectors[:, order])
    else:
        try:
            eigenvalues, eigenvectors = eigsh(H, k=k, M=M, which='SA')
        except Exception:
            eigenvalues, eigenvectors = eigsh(H, k=k, M=M, sigma=-100.0, which='LM')
        order = np.argsort(eigenvalues)
        return eigenvalues[order], eigenvectors[:, order]
