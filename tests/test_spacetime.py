# tests/test_spacetime.py
import numpy as np
from theory.emergent_spacetime import (
    spacetime_density,
    emergent_metric_factor,
    gravitational_potential,
)


def test_spacetime_density_from_wave():
    psi = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 0j])
    rho = spacetime_density(psi)
    np.testing.assert_allclose(rho, np.array([1.0, 0.5, 0.0]))


def test_spacetime_density_zero_for_nodes():
    psi = np.array([0.0 + 0j, 0.0 + 0j])
    rho = spacetime_density(psi)
    np.testing.assert_allclose(rho, np.array([0.0, 0.0]))


def test_metric_factor_is_one_plus_correction():
    psi = np.array([0.0 + 0j])
    factor = emergent_metric_factor(psi)
    np.testing.assert_allclose(factor, np.array([1.0]))


def test_metric_factor_increases_with_density():
    psi_weak = np.array([0.1 + 0j])
    psi_strong = np.array([10.0 + 0j])
    f_weak = emergent_metric_factor(psi_weak)
    f_strong = emergent_metric_factor(psi_strong)
    assert f_strong[0] > f_weak[0]


def test_gravitational_potential_radial():
    r = np.linspace(0.1, 10.0, 100)
    psi = np.exp(-r) + 0j
    phi_grav = gravitational_potential(r, psi)
    assert phi_grav[0] < phi_grav[-1]
