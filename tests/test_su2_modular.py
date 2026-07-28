"""Tests de la structure modulaire SU(2)_135 (V5 §10.4 et V5.5)."""
import math
import pytest

from theory.su2_modular import (
    V5_LEVEL, quantum_dim, total_dim_squared, central_charge,
    conformal_weight, modular_S, q_parameter, wilson_loop_unknot,
    summary,
)


def test_v5_level_is_135():
    assert V5_LEVEL == 135


def test_alpha_EM_emerges():
    """V5 §10.4 : α_EM = 1/(k+2). À k=135 : 1/137 ≈ α_PDG à 0.026 %."""
    alpha_V5 = 1.0 / (V5_LEVEL + 2)
    alpha_PDG = 1.0 / 137.036
    assert abs(alpha_V5 - alpha_PDG) / alpha_PDG < 3e-4


def test_quantum_dim_unknot():
    """d_0 = 1 (vacuum)."""
    assert quantum_dim(0.0) == pytest.approx(1.0)


def test_quantum_dim_half():
    """d_{1/2} = 2 cos(π/(k+2)) ≈ 2 pour k=135."""
    d = quantum_dim(0.5)
    expected = 2 * math.cos(math.pi / (V5_LEVEL + 2))
    assert d == pytest.approx(expected)
    assert d > 1.999


def test_quantum_dim_self_duality():
    """d_j = d_{k/2 - j} pour SU(2)_k unitaire."""
    for j2 in [1, 3, 5, 10, 50, 100]:
        j = j2 / 2
        j_dual = V5_LEVEL / 2 - j
        d_j = quantum_dim(j)
        d_dual = quantum_dim(j_dual)
        assert d_j == pytest.approx(d_dual, rel=1e-9), (
            f"d_j({j}) = {d_j} != d_j({j_dual}) = {d_dual}"
        )


def test_total_dim_squared_formula():
    """D² = (k+2) / (2 sin²(π/(k+2))).

    Également vérifie : Σ_j d_j² = D².
    """
    D2 = total_dim_squared()
    expected = (V5_LEVEL + 2) / (2 * math.sin(math.pi / (V5_LEVEL + 2)) ** 2)
    assert D2 == pytest.approx(expected)

    # Vérification numérique de Σ d_j² ≈ D²
    sum_d2 = sum(quantum_dim(j / 2) ** 2 for j in range(V5_LEVEL + 1))
    assert sum_d2 == pytest.approx(D2, rel=1e-6)


def test_central_charge():
    """c = 3k/(k+2). À k=135 : c = 405/137 ≈ 2.956."""
    c = central_charge()
    assert c == pytest.approx(405 / 137)
    assert 2.95 < c < 2.96


def test_conformal_weight():
    """h_j = j(j+1)/(k+2). h_0 = 0, h_{1/2} = (3/8)·(1/137) ≈ 0.00274."""
    assert conformal_weight(0) == 0
    expected_half = 0.5 * 1.5 / 137
    assert conformal_weight(0.5) == pytest.approx(expected_half)


def test_modular_S_symmetry():
    """S_{jl} = S_{lj} (symétrie)."""
    for j2, l2 in [(0, 1), (1, 3), (5, 10), (50, 100)]:
        j, l = j2 / 2, l2 / 2
        assert modular_S(j, l) == pytest.approx(modular_S(l, j))


def test_q_parameter():
    """q = exp(2πi/(k+2)) ; |q| = 1."""
    q = q_parameter()
    assert abs(q) == pytest.approx(1.0)
    # Phase = 2π/137
    assert q.imag > 0  # premier quadrant pour k=135


def test_wilson_loop_unknot():
    """⟨W_unknot⟩ = d_{1/2} en convention CS."""
    assert wilson_loop_unknot() == pytest.approx(quantum_dim(0.5))


def test_summary_keys():
    s = summary()
    expected_keys = {"k", "k_plus_2", "alpha_EM", "central_charge_c",
                     "D_squared", "d_half", "h_half", "n_primaries",
                     "S_00", "S_half_half"}
    assert expected_keys.issubset(s.keys())
    assert s["n_primaries"] == 136  # j ∈ {0, 1/2, ..., 67.5}
