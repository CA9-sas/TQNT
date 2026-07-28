# tests/test_postulates.py
import numpy as np
from theory.postulates import (
    ALPHA, BETA, KAPPA, C, HBAR,
    GAMMA, LAMBDA, R_0, E_0,
    nodal_mass, nodal_energy, nodal_charge,
    retraction_mass,
)


def test_constants_are_positive():
    assert ALPHA > 0
    assert BETA > 0
    assert KAPPA > 0
    assert C > 0
    assert HBAR > 0


def test_nodal_mass_increases_with_n():
    m1 = nodal_mass(n=1, l=0)
    m2 = nodal_mass(n=2, l=0)
    m3 = nodal_mass(n=3, l=0)
    assert m2 > m1
    assert m3 > m2


def test_nodal_mass_increases_with_l():
    m0 = nodal_mass(n=1, l=0)
    m1 = nodal_mass(n=1, l=1)
    m2 = nodal_mass(n=1, l=2)
    assert m1 > m0
    assert m2 > m1


def test_nodal_mass_formula():
    m = nodal_mass(n=2, l=3)
    expected = ALPHA * 2 + BETA * 3 * 4
    assert np.isclose(m, expected)


def test_nodal_energy_from_mass():
    m = nodal_mass(n=1, l=0)
    e = nodal_energy(n=1, l=0)
    assert np.isclose(e, m * C**2)


def test_nodal_charge_from_m_quantum_number():
    assert nodal_charge(m=1) > 0
    assert nodal_charge(m=-1) < 0
    assert nodal_charge(m=0) == 0.0
    assert nodal_charge(m=1) == -nodal_charge(m=-1)


def test_v2_constants_exist():
    assert GAMMA > 0
    assert LAMBDA > 0
    assert R_0 > 0
    assert E_0 > 0


def test_retraction_mass():
    m = retraction_mass(rho=1.5)
    assert np.isclose(m, ALPHA * 1.5)


def test_retraction_mass_zero_for_massless():
    m = retraction_mass(rho=0.0)
    assert m == 0.0


def test_retraction_mass_increases_with_rho():
    m1 = retraction_mass(rho=1.0)
    m2 = retraction_mass(rho=3.0)
    assert m2 > m1


def test_link_constants_exist():
    from theory.postulates import LAMBDA_LINK, BETA_LINK
    assert LAMBDA_LINK > 0
    assert BETA_LINK > 0

def test_postulate_p11_exists():
    from theory.postulates import POSTULATE_P11
    assert "multi" in POSTULATE_P11.lower() or "entrelacs" in POSTULATE_P11.lower()
