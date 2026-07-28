"""Tests V5.15 — dérivation du coefficient *a* du terme R² depuis SU(2)_135."""
from __future__ import annotations

import math

import pytest

from theory.coefficient_a_R2 import (
    A_S_PLANCK,
    A_S_PLANCK_SIGMA,
    A_s_predicted_V5p15,
    K_V5,
    N_EFOLDS_PIVOT,
    V5p15CoefficientResult,
    a_V5p15_from_SU2_135,
    effective_action_1loop_SU2,
    n_s_V5p15,
    r_V5p15,
    scalaron_mass_GeV,
    scalaron_mass_over_Mpl,
    summary_V5p15,
    validate_against_V5p14_calibration,
)
from theory.su2_modular import V5_LEVEL


# ---------------------------------------------------------------------------
# Test 1 — Constantes fondamentales
# ---------------------------------------------------------------------------
def test_K_V5_is_135():
    """K_V5 = 135 (niveau Chern-Simons unique V5)."""
    assert K_V5 == V5_LEVEL == 135


# ---------------------------------------------------------------------------
# Test 2 — Formule a = (k+2)^5/6
# ---------------------------------------------------------------------------
def test_a_V5p15_formula():
    """a_V5.15(k) = (k+2)^5 / 6 exact."""
    k = 135
    expected = (k + 2) ** 5 / 6.0
    assert a_V5p15_from_SU2_135(k) == pytest.approx(expected, rel=1e-15)


def test_a_V5p15_numerical_value():
    """a_V5.15(135) ≈ 8,04·10⁹ M_pl⁻²."""
    assert a_V5p15_from_SU2_135(135) == pytest.approx(8.044e9, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 3 — Masse du scalaron
# ---------------------------------------------------------------------------
def test_scalaron_mass_relation():
    """M_S/M_pl = α_EM^(5/2) ≡ 1/(k+2)^(5/2)."""
    k = 135
    expected = (k + 2) ** (-2.5)
    assert scalaron_mass_over_Mpl(k) == pytest.approx(expected, rel=1e-15)


def test_scalaron_mass_GeV_order():
    """M_S(135) ≈ 1,1·10¹³ GeV (compatible Starobinsky standard ~10¹³ GeV)."""
    M_S = scalaron_mass_GeV(135)
    assert 1.0e13 < M_S < 1.5e13


# ---------------------------------------------------------------------------
# Test 4 — Cohérence M_S² = M_pl²/(6 a)
# ---------------------------------------------------------------------------
def test_scalaron_mass_consistency_with_a():
    """M_S²/M_pl² = 1/(6 a). Cohérence interne V5.15."""
    k = 135
    a = a_V5p15_from_SU2_135(k)
    M_S_over_Mpl = scalaron_mass_over_Mpl(k)
    assert M_S_over_Mpl ** 2 == pytest.approx(1.0 / (6.0 * a), rel=1e-12)


# ---------------------------------------------------------------------------
# Test 5 — Validation contre V5.14 (tolérance 20 %)
# ---------------------------------------------------------------------------
def test_validate_against_V5p14_passes_20pct():
    """a_V5.15 reproduit la calibration V5.14 (Planck A_s) à ±20 %."""
    valid = validate_against_V5p14_calibration(tolerance=0.20)
    assert valid["passes_tolerance"]
    assert abs(valid["relative_error"]) < 0.20


def test_validate_relative_error_around_11pct():
    """Écart relatif a_V5.15 vs a_calib ≈ +11 % (signature du modèle)."""
    valid = validate_against_V5p14_calibration()
    assert 0.05 < valid["relative_error"] < 0.20


# ---------------------------------------------------------------------------
# Test 6 — Prédiction A_s
# ---------------------------------------------------------------------------
def test_A_s_predicted_order_of_magnitude():
    """A_s prédit ≈ 1,89·10⁻⁹ (à -9,9 % de Planck 2,10·10⁻⁹)."""
    A_s = A_s_predicted_V5p15()
    assert 1.7e-9 < A_s < 2.0e-9


def test_A_s_predicted_formula():
    """A_s = N_e² / (24 π² · a_V5.15) exact."""
    k, N = 135, 60.0
    a = a_V5p15_from_SU2_135(k)
    expected = N ** 2 / (24.0 * math.pi ** 2 * a)
    assert A_s_predicted_V5p15(N_e=N, k=k) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 7 — n_s, r invariants par V5.15 (Starobinsky asymptotique)
# ---------------------------------------------------------------------------
def test_n_s_invariant_Starobinsky():
    """n_s(N=60) = 1 - 2/60 ≈ 0,9667, compatible Planck à ~0,4 σ."""
    n_s = n_s_V5p15(60.0)
    assert n_s == pytest.approx(0.96667, abs=1e-4)
    # Compatible Planck (0,9649 ± 0,0042)
    assert abs(n_s - 0.9649) < 3 * 0.0042


def test_r_invariant_Starobinsky():
    """r(N=60) = 12/60² ≈ 0,00333, bien sous BICEP 0,06."""
    r = r_V5p15(60.0)
    assert r == pytest.approx(0.00333, abs=1e-4)
    assert r < 0.06


# ---------------------------------------------------------------------------
# Test 8 — Variation avec k
# ---------------------------------------------------------------------------
def test_a_monotonic_in_k():
    """a(k) strictement croissant en k."""
    assert a_V5p15_from_SU2_135(100) < a_V5p15_from_SU2_135(135)
    assert a_V5p15_from_SU2_135(135) < a_V5p15_from_SU2_135(150)


def test_A_s_decreasing_in_k():
    """A_s ∝ 1/a ∝ 1/(k+2)^5, donc décroissant en k."""
    assert A_s_predicted_V5p15(k=100) > A_s_predicted_V5p15(k=135)
    assert A_s_predicted_V5p15(k=135) > A_s_predicted_V5p15(k=150)


# ---------------------------------------------------------------------------
# Test 9 — Action effective 1-boucle SU(2)_k (structure)
# ---------------------------------------------------------------------------
def test_effective_action_keys_present():
    """effective_action_1loop_SU2 expose les clés structurelles attendues."""
    eff = effective_action_1loop_SU2()
    for k in ("k", "k_plus_2", "alpha_EM_V5", "central_charge", "D_squared",
              "n_primaries", "term_R_squared_b1", "scalaron_mass_GeV",
              "a_V5p15"):
        assert k in eff


def test_effective_action_b1_matches_a_V5p15():
    """Le terme b₁ R² de l'action effective coïncide avec a_V5.15."""
    eff = effective_action_1loop_SU2()
    assert eff["term_R_squared_b1"].coefficient == pytest.approx(
        a_V5p15_from_SU2_135(K_V5), rel=1e-15
    )


# ---------------------------------------------------------------------------
# Test 10 — Synthèse summary_V5p15
# ---------------------------------------------------------------------------
def test_summary_V5p15_dataclass():
    """summary_V5p15 renvoie un V5p15CoefficientResult avec champs cohérents."""
    res = summary_V5p15()
    assert isinstance(res, V5p15CoefficientResult)
    assert res.k == 135
    assert res.alpha_EM == pytest.approx(1.0 / 137.0, rel=1e-12)
    assert res.a_V5p15 > 0
    assert res.passes_tolerance
    assert res.M_S_GeV > 0
    assert 0.94 < res.n_s < 0.97
    assert 0.001 < res.r < 0.01
    assert 1.7e-9 < res.A_s_predicted < 2.1e-9
