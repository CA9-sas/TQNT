# tests/test_chern_simons.py
import numpy as np
from theory.chern_simons import (
    cs_q, quantum_integer, quantum_factorial, quantum_binomial,
    colored_jones, colored_jones_unknot,
    COLORED_JONES_KNOTS,
    verify_aj_conjecture,
    representation_variety,
    cs_action_neumann_zagier,
    effective_potential_from_a_poly,
    classical_limit_check,
    compare_potentials,
)


def test_cs_q_is_root_of_unity():
    q = cs_q(k=3)
    assert np.isclose(abs(q), 1.0)
    # q^(k+2) = q^5 = 1
    assert np.isclose(abs(q**5 - 1), 0.0, atol=1e-10)


def test_cs_q_level_1():
    q = cs_q(k=1)
    # q = exp(2*pi*i / 3) => q^3 = 1
    assert np.isclose(abs(q**3 - 1), 0.0, atol=1e-10)


def test_quantum_integer_classical_limit():
    """[n]_q -> n as q -> 1."""
    q = cs_q(k=10000)  # q very close to 1
    for n in range(1, 8):
        assert np.isclose(quantum_integer(n, q).real, n, atol=0.01)


def test_quantum_integer_zero():
    q = cs_q(k=10)
    assert quantum_integer(0, q) == 0


def test_quantum_integer_one():
    q = cs_q(k=10)
    assert np.isclose(abs(quantum_integer(1, q) - 1), 0.0, atol=1e-10)


def test_quantum_factorial_zero():
    q = cs_q(k=10)
    assert np.isclose(abs(quantum_factorial(0, q) - 1), 0.0, atol=1e-10)


def test_quantum_factorial_classical_limit():
    q = cs_q(k=10000)
    # [4]! -> 24
    assert np.isclose(quantum_factorial(4, q).real, 24.0, atol=0.5)


def test_quantum_binomial_classical_limit():
    q = cs_q(k=10000)
    # [4 choose 2] -> 6
    assert np.isclose(quantum_binomial(4, 2, q).real, 6.0, atol=0.5)


# ============================================================
# Layer 2: Colored Jones Polynomials
# ============================================================

def test_colored_jones_unknot_j1():
    q = cs_q(k=10)
    assert np.isclose(abs(colored_jones("0_1", 1, q) - 1), 0.0, atol=1e-10)


def test_colored_jones_unknot_j2():
    q = cs_q(k=10)
    j2 = colored_jones("0_1", 2, q)
    expected = quantum_integer(2, q)
    assert np.isclose(abs(j2 - expected), 0.0, atol=1e-10)


def test_colored_jones_unknot_jn_equals_quantum_n():
    q = cs_q(k=50)
    for n in range(1, 8):
        jn = colored_jones("0_1", n, q)
        qn = quantum_integer(n, q)
        assert np.isclose(abs(jn - qn), 0.0, atol=1e-10)


def test_colored_jones_j1_is_one():
    """J_1(K; q) = 1 for all knots."""
    q = cs_q(k=20)
    for name in COLORED_JONES_KNOTS:
        assert np.isclose(abs(colored_jones(name, 1, q) - 1), 0.0, atol=1e-10)


def test_colored_jones_trefoil_j2_nonzero():
    q = cs_q(k=50)
    j2 = colored_jones("3_1", 2, q)
    assert abs(j2) > 0.1


def test_colored_jones_figure_eight_j2_nonzero():
    q = cs_q(k=50)
    j2 = colored_jones("4_1", 2, q)
    assert abs(j2) > 0.1


def test_colored_jones_classical_limit_unknot():
    """As k -> inf (q -> 1), J_n(unknot; q) -> n."""
    q = cs_q(k=10000)
    for n in [2, 3, 5]:
        assert np.isclose(colored_jones("0_1", n, q).real, n, atol=0.1)


def test_colored_jones_unknown_raises():
    import pytest
    q = cs_q(k=10)
    with pytest.raises(ValueError):
        colored_jones("99_99", 2, q)


def test_colored_jones_knots_list():
    assert "0_1" in COLORED_JONES_KNOTS
    assert "3_1" in COLORED_JONES_KNOTS
    assert "4_1" in COLORED_JONES_KNOTS
    assert len(COLORED_JONES_KNOTS) >= 6


# ============================================================
# Layer 3: AJ Conjecture
# ============================================================

def test_aj_trefoil_convergence():
    """Torus knot AJ: ratio J_{n+1}/J_n converges as k grows."""
    result = verify_aj_conjecture("3_1", k=200, n_max=6)
    assert result["max_residual"] < 0.25


def test_aj_5_1_convergence():
    result = verify_aj_conjecture("5_1", k=200, n_max=4)
    assert result["max_residual"] < 0.25


def test_aj_figure_eight_convergence():
    """Figure-eight AJ: classical limit converges well at k=200."""
    result = verify_aj_conjecture("4_1", k=200, n_max=5)
    assert result["max_residual"] < 0.05


def test_aj_all_knots_converge():
    """All knots converge via classical limit at k=200."""
    for name in ["3_1", "4_1", "5_1"]:
        result = verify_aj_conjecture(name, k=200, n_max=4)
        assert result["max_residual"] < 0.25, f"{name} failed: {result['max_residual']}"


def test_aj_figure_eight_method():
    """Figure-eight uses classical_limit method."""
    result = verify_aj_conjecture("4_1", k=50, n_max=3)
    assert result["method"] == "classical_limit"


def test_aj_returns_expected_keys():
    result = verify_aj_conjecture("3_1", k=20, n_max=4)
    assert "knot_name" in result
    assert "residuals" in result
    assert "max_residual" in result
    assert "passed" in result


# ============================================================
# Layer 4: Representation Variety
# ============================================================

def test_representation_variety_trefoil():
    M_vals, L_vals = representation_variety("3_1", n_points=50)
    assert len(M_vals) > 0
    assert len(L_vals) == len(M_vals)


def test_representation_variety_satisfies_a_polynomial():
    """All (L, M) on the variety should satisfy A_na(L, M) ≈ 0."""
    from theory.a_polynomial import evaluate_a_polynomial
    M_vals, L_vals = representation_variety("3_1", n_points=30)
    for m, l in zip(M_vals[:10], L_vals[:10]):
        val = evaluate_a_polynomial("3_1", complex(l), complex(m))
        assert abs(val) < 1e-4 or abs(complex(l) - 1) < 1e-4


def test_representation_variety_figure_eight():
    M_vals, L_vals = representation_variety("4_1", n_points=50)
    assert len(M_vals) > 0


# ============================================================
# Layers 5-6: CS Action and Semi-Classical Analysis
# ============================================================

def test_cs_action_trefoil_computable():
    M_vals, S_cs = cs_action_neumann_zagier("3_1", n_points=50)
    assert len(M_vals) == 50
    assert len(S_cs) == 50


def test_effective_potential_trefoil():
    x_vals, V_eff = effective_potential_from_a_poly("3_1", n_points=50)
    valid = ~np.isnan(V_eff)
    assert np.sum(valid) > 10


def test_classical_limit_unknot():
    result = classical_limit_check("0_1")
    # As k grows, [n]_q -> n
    qn_values = result["quantum_n"]
    assert np.isclose(qn_values[-1].real, result["classical_n"], atol=0.01)


def test_compare_potentials_trefoil():
    result = compare_potentials("3_1", n_points=50)
    assert "V_cs" in result
    assert "V_retraction" in result
    assert "correlation" in result
