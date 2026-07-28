# tests/test_retraction.py
import numpy as np
from theory.jones import jones_polynomial
from theory.retraction import (
    retraction_factor,
    knot_mass,
    tightening_radius,
    energy_dependent_retraction,
)


def test_retraction_factor_unknot():
    poly = jones_polynomial("0_1")
    rho = retraction_factor(poly)
    assert rho < 0.5


def test_retraction_factor_trefoil():
    poly = jones_polynomial("3_1")
    rho = retraction_factor(poly)
    assert rho > 1.0


def test_retraction_increases_with_complexity():
    rho_trefoil = retraction_factor(jones_polynomial("3_1"))
    rho_5_1 = retraction_factor(jones_polynomial("5_1"))
    rho_7_1 = retraction_factor(jones_polynomial("7_1"))
    assert rho_5_1 > rho_trefoil
    assert rho_7_1 > rho_5_1


def test_knot_mass_from_retraction():
    poly = jones_polynomial("3_1")
    m = knot_mass(poly)
    assert m > 0
    rho = retraction_factor(poly)
    assert np.isclose(m, 1.0 * rho)


def test_knot_mass_unknot_nearly_zero():
    m = knot_mass(jones_polynomial("0_1"))
    assert m < 0.5


def test_tightening_radius_decreases_with_energy():
    r1 = tightening_radius(E=0.0)
    r2 = tightening_radius(E=1.0)
    r3 = tightening_radius(E=100.0)
    assert r1 > r2 > r3


def test_tightening_radius_base():
    from theory.postulates import R_0
    r = tightening_radius(E=0.0)
    assert np.isclose(r, R_0)


def test_energy_dependent_retraction_zero_energy():
    poly = jones_polynomial("3_1")
    rho_J = retraction_factor(poly)
    rho = energy_dependent_retraction(poly, 0.0)
    assert abs(rho - rho_J) < 1e-10


def test_energy_dependent_retraction_decreasing():
    poly = jones_polynomial("3_1")
    rho_0 = energy_dependent_retraction(poly, 0.0)
    rho_10 = energy_dependent_retraction(poly, 10.0)
    assert rho_0 > rho_10
