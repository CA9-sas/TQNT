# tests/test_chi_field.py
import numpy as np
from theory.chi_field import (
    chi_retraction,
    jones_rho,
    characteristic_energy,
    chi_potential,
)


def test_jones_rho_unknot():
    rho = jones_rho("0_1")
    assert rho < 0.5


def test_jones_rho_trefoil():
    rho = jones_rho("3_1")
    assert rho > 1.0


def test_jones_rho_increases_with_complexity():
    rho_3 = jones_rho("3_1")
    rho_7 = jones_rho("7_1")
    assert rho_7 > rho_3


def test_chi_retraction_at_zero_energy():
    rho_J = jones_rho("3_1")
    rho = chi_retraction("3_1", energy=0.0)
    assert np.isclose(rho, rho_J)


def test_chi_retraction_decreases_with_energy():
    rho_0 = chi_retraction("3_1", energy=0.0)
    rho_1 = chi_retraction("3_1", energy=1.0)
    rho_10 = chi_retraction("3_1", energy=10.0)
    assert rho_0 > rho_1 > rho_10


def test_chi_retraction_approaches_zero():
    rho = chi_retraction("3_1", energy=1e6)
    assert rho < 0.01


def test_chi_retraction_complex_knot_resists_more():
    """Complex knots need more energy to reduce rho."""
    rho_3_at_5 = chi_retraction("3_1", energy=5.0)
    rho_7_at_5 = chi_retraction("7_1", energy=5.0)
    ratio_3 = rho_3_at_5 / jones_rho("3_1")
    ratio_7 = rho_7_at_5 / jones_rho("7_1")
    assert ratio_7 > ratio_3


def test_characteristic_energy():
    from theory.postulates import E_0
    rho_J = jones_rho("3_1")
    E_K = characteristic_energy("3_1")
    assert np.isclose(E_K, E_0 * rho_J)


def test_chi_retraction_at_characteristic_energy():
    """At E = E_K, rho should be rho_J / 2."""
    E_K = characteristic_energy("3_1")
    rho_J = jones_rho("3_1")
    rho = chi_retraction("3_1", energy=E_K)
    assert np.isclose(rho, rho_J / 2.0)


def test_chi_potential_negative():
    from scipy.spatial import cKDTree
    from theory.knot_catalog import knot_curve
    curve = knot_curve("3_1", 200)
    tree = cKDTree(curve)
    r = np.array([[5.0, 0.0, 0.0]])
    V = chi_potential(r, "3_1", energy=0.0, knot_tree=tree, epsilon=0.2)
    assert V[0] < 0


def test_chi_potential_weaker_at_high_energy():
    from scipy.spatial import cKDTree
    from theory.knot_catalog import knot_curve
    curve = knot_curve("3_1", 200)
    tree = cKDTree(curve)
    r = np.array([[5.0, 0.0, 0.0]])
    V_low = chi_potential(r, "3_1", energy=0.0, knot_tree=tree, epsilon=0.2)
    V_high = chi_potential(r, "3_1", energy=100.0, knot_tree=tree, epsilon=0.2)
    assert abs(V_high[0]) < abs(V_low[0])
