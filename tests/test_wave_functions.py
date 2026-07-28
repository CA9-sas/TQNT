# tests/test_wave_functions.py
import numpy as np
from theory.wave_functions import (
    radial_function,
    spherical_harmonic,
    nodal_wave,
    count_radial_nodes,
)


def test_radial_function_shape():
    r = np.linspace(0.01, 10.0, 100)
    R = radial_function(r, n=1, l=0)
    assert R.shape == (100,)


def test_radial_function_has_n_nodes():
    """R_n should have exactly n zeros (nodes) in the radial domain."""
    r = np.linspace(0.01, 20.0, 5000)
    for n in [1, 2, 3]:
        R = radial_function(r, n=n, l=0)
        nodes = count_radial_nodes(r, R)
        assert nodes == n, f"Expected {n} nodes, got {nodes}"


def test_spherical_harmonic_normalization():
    """Y_l^m should be normalized over the unit sphere."""
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    Y = spherical_harmonic(THETA, PHI, l=1, m=0)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    integral = np.sum(np.abs(Y) ** 2 * np.sin(THETA)) * dtheta * dphi
    assert np.isclose(integral, 1.0, atol=0.05)


def test_nodal_wave_returns_complex():
    r = np.array([1.0, 2.0, 3.0])
    theta = np.array([0.5, 1.0, 1.5])
    phi = np.array([0.0, 0.5, 1.0])
    t = 0.0
    psi = nodal_wave(r, theta, phi, t, n=1, l=0, m=0)
    assert psi.dtype == np.complex128 or np.issubdtype(psi.dtype, np.complexfloating)


def test_nodal_wave_time_evolution():
    """Wave at t=0 vs t>0 should differ by a phase factor."""
    r = np.array([1.0, 2.0])
    theta = np.array([0.5, 1.0])
    phi = np.array([0.0, 0.5])
    psi0 = nodal_wave(r, theta, phi, t=0.0, n=1, l=0, m=0)
    psi1 = nodal_wave(r, theta, phi, t=1.0, n=1, l=0, m=0)
    np.testing.assert_allclose(np.abs(psi0), np.abs(psi1), atol=1e-10)
    assert not np.allclose(psi0, psi1)
