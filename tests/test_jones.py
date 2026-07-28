# tests/test_jones.py
import numpy as np
from theory.jones import (
    JonesPolynomial,
    jones_polynomial,
    evaluate_jones,
    jones_span,
    jones_determinant,
    jones_retraction_factor,
    mirror_jones,
    is_amphicheiral,
    JONES_KNOT_NAMES,
)


def test_jones_polynomial_unknot():
    poly = jones_polynomial("0_1")
    assert poly == {0: 1}


def test_jones_polynomial_trefoil():
    poly = jones_polynomial("3_1")
    assert 1 in poly or -4 in poly


def test_jones_polynomial_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        jones_polynomial("99_99")


def test_evaluate_jones_unknot():
    poly = jones_polynomial("0_1")
    assert evaluate_jones(poly, 2.0) == 1.0


def test_evaluate_jones_at_1_is_1():
    """V_K(1) = 1 for all knots (normalization)."""
    for name in JONES_KNOT_NAMES:
        poly = jones_polynomial(name)
        val = evaluate_jones(poly, 1.0)
        assert abs(val - 1.0) < 1e-10, f"{name}: V(1) = {val} != 1"


def test_jones_span_unknot():
    poly = jones_polynomial("0_1")
    assert jones_span(poly) == 0


def test_jones_span_trefoil():
    poly = jones_polynomial("3_1")
    assert jones_span(poly) == 3


def test_jones_span_figure_eight():
    poly = jones_polynomial("4_1")
    assert jones_span(poly) == 4


def test_jones_span_equals_crossing_number():
    """For alternating knots, span = crossing number."""
    alternating = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
                   "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7"]
    for name in alternating:
        poly = jones_polynomial(name)
        crossing = int(name.split("_")[0])
        assert jones_span(poly) == crossing, f"{name}: span={jones_span(poly)} != {crossing}"


def test_jones_determinant_trefoil():
    poly = jones_polynomial("3_1")
    assert jones_determinant(poly) == 3


def test_jones_determinant_figure_eight():
    poly = jones_polynomial("4_1")
    assert jones_determinant(poly) == 5


def test_jones_retraction_factor_unknot():
    poly = jones_polynomial("0_1")
    rho = jones_retraction_factor(poly)
    assert rho < 0.5  # quasi-massless


def test_jones_retraction_factor_positive():
    for name in JONES_KNOT_NAMES:
        poly = jones_polynomial(name)
        assert jones_retraction_factor(poly) > 0


def test_mirror_jones_trefoil():
    poly = jones_polynomial("3_1")
    mirror = mirror_jones(poly)
    assert poly != mirror  # trefoil is chiral


def test_is_amphicheiral_figure_eight():
    poly = jones_polynomial("4_1")
    assert is_amphicheiral(poly)


def test_is_not_amphicheiral_trefoil():
    poly = jones_polynomial("3_1")
    assert not is_amphicheiral(poly)


def test_jones_knot_names_has_20():
    assert len(JONES_KNOT_NAMES) == 20
