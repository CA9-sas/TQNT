# tests/test_a_polynomial.py
import sympy as sp
from theory.a_polynomial import (
    a_polynomial,
    a_polynomial_nonabelian,
    evaluate_a_polynomial,
    quantize_a_polynomial,
    A_POLY_KNOT_NAMES,
)


def test_a_polynomial_unknot():
    L, M = sp.symbols('L M')
    A = a_polynomial("0_1")
    assert A == L - 1


def test_a_polynomial_trefoil():
    A = a_polynomial("3_1")
    assert A.subs(sp.Symbol('L'), 1) == 0


def test_a_polynomial_nonabelian_trefoil():
    L, M = sp.symbols('L M')
    A_na = a_polynomial_nonabelian("3_1")
    assert A_na == L + M**6


def test_a_polynomial_figure_eight():
    A = a_polynomial("4_1")
    assert A.subs(sp.Symbol('L'), 1) == 0


def test_evaluate_a_polynomial_abelian():
    """At L=1, A_K(1, M) = 0 for all knots."""
    for name in A_POLY_KNOT_NAMES:
        val = evaluate_a_polynomial(name, L_val=1.0, M_val=2.0)
        assert abs(val) < 1e-10, f"{name}: A(1, 2) = {val} != 0"


def test_quantize_a_polynomial_trefoil():
    A_hat = quantize_a_polynomial("3_1")
    assert A_hat is not None


def test_a_poly_knot_names():
    assert "0_1" in A_POLY_KNOT_NAMES
    assert "3_1" in A_POLY_KNOT_NAMES
    assert "4_1" in A_POLY_KNOT_NAMES


def test_a_polynomial_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        a_polynomial("99_99")
