"""
Theorie Nodale -- Fonctions d'Onde

Solutions stationnaires en coordonnees spheriques:
Psi(r, theta, phi, t) = R_n(r) * Y_l^m(theta, phi) * exp(-i*omega*t)

The radial function R_n uses spherical Bessel functions j_l(k*r)
with k chosen so that R_n has exactly n radial nodes.
"""
import numpy as np
from scipy.special import spherical_jn, sph_harm_y
from theory.postulates import nodal_energy, HBAR


def _find_bessel_zeros(l: int, n: int) -> float:
    """Find the n-th zero of j_l(x) by bracketing and bisection."""
    x = np.linspace(0.5, 50.0 + n * 10, 50000)
    vals = spherical_jn(l, x)
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) < n:
        raise ValueError(f"Could not find {n} zeros of j_{l}")
    idx = sign_changes[n - 1]
    a, b = x[idx], x[idx + 1]
    for _ in range(100):
        mid = (a + b) / 2
        if spherical_jn(l, mid) * spherical_jn(l, a) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2


def radial_function(r: np.ndarray, n: int, l: int, r_max: float = 20.0) -> np.ndarray:
    """Compute radial wave function R_n,l(r) with n radial nodes.

    The (n+1)-th Bessel zero is placed at r_max (confinement boundary),
    leaving exactly n interior nodes where the function crosses zero.
    """
    zero_n = _find_bessel_zeros(l, n + 1)
    k = zero_n / r_max
    R = spherical_jn(l, k * r)
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    norm = np.sqrt(np.sum(np.abs(R) ** 2 * r**2) * dr)
    if norm > 0:
        R = R / norm
    return R


def spherical_harmonic(theta: np.ndarray, phi: np.ndarray, l: int, m: int) -> np.ndarray:
    """Compute spherical harmonic Y_l^m(theta, phi)."""
    return sph_harm_y(l, m, theta, phi)


def nodal_wave(r, theta, phi, t, n, l, m, r_max=20.0):
    """Compute full nodal wave function Psi(r, theta, phi, t)."""
    R = radial_function(r, n, l, r_max)
    Y = spherical_harmonic(theta, phi, l, m)
    E = nodal_energy(n, l)
    omega = E / HBAR
    return R * Y * np.exp(-1j * omega * t)


def count_radial_nodes(r: np.ndarray, R: np.ndarray) -> int:
    """Count radial nodes (zero crossings) of R(r)."""
    signs = np.sign(R)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0]
    return len(crossings)
