"""Tests de la closure V5.5 étape 8 (Ω² saturé sigmoïdal)."""
import math
import pytest

from theory.cgq_envelope_saturated import (
    sigmoid, h_saturated, omega_squared, beta_at_M, delta_CGQ_at_M,
    Lc_correction_factor, summary,
    M_STAR_MSUN, H_MAX, DELTA_LOG_M,
    BETA_TARGET, DELTA_CGQ_TARGET,
)


def test_sigmoid_properties():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(100) == pytest.approx(1.0)
    assert sigmoid(-100) == pytest.approx(0.0)


def test_h_at_M_star():
    """h(M_*) = h_max / 2 = 0.15 par construction."""
    h = h_saturated(M_STAR_MSUN)
    assert h == pytest.approx(H_MAX / 2)
    assert h == pytest.approx(0.15)


def test_h_saturates_high_M():
    """h → h_max à grande masse."""
    h = h_saturated(1e15)
    assert h == pytest.approx(H_MAX, abs=0.001)


def test_h_negligible_low_M():
    """h → 0 à petite masse."""
    h = h_saturated(1e9)
    assert h < 0.001


def test_omega_squared_at_M_star():
    """Ω²(M_*) = 1 + h(M_*) = 1.15."""
    assert omega_squared(M_STAR_MSUN) == pytest.approx(1.15)


def test_beta_at_M_star_equals_target():
    """β à M_* doit être 6.29 par construction."""
    beta = beta_at_M(M_STAR_MSUN)
    assert beta == pytest.approx(BETA_TARGET, rel=0.01)


def test_delta_CGQ_closure():
    """V5.5 ÉTAPE 8 CLOSURE : Δ_CGQ à M_* = 0.41 ± 5 %."""
    dcgq = delta_CGQ_at_M(M_STAR_MSUN)
    assert dcgq == pytest.approx(DELTA_CGQ_TARGET, rel=0.05)
    # En fait < 0.1 % d'écart
    assert abs(dcgq - DELTA_CGQ_TARGET) / DELTA_CGQ_TARGET < 0.001


def test_Lcoh_correction_factor():
    """L_coh correction selon convention V5.5 étape 8b.

    Convention par défaut "multiply" (révision après agent 2 SPARC re-fit) :
      L_coh_eff = L_coh × sqrt(Ω²)  -> stretche L_coh aux hautes masses
                                      → c plus petit (sens WB)
    """
    f = Lc_correction_factor(M_STAR_MSUN)
    # Convention par défaut "multiply"
    assert f == pytest.approx(math.sqrt(1.15))
    # Convention alternative "divide" reste accessible
    f_div = Lc_correction_factor(M_STAR_MSUN, sign="divide")
    assert f_div == pytest.approx(1.0 / math.sqrt(1.15))


def test_summary_consistency():
    s = summary()
    assert s["h"] == pytest.approx(0.15)
    assert s["omega_squared"] == pytest.approx(1.15)
    assert s["beta"] == pytest.approx(BETA_TARGET, rel=0.01)
    assert s["delta_CGQ"] == pytest.approx(DELTA_CGQ_TARGET, rel=0.005)


def test_three_anchors():
    """Trois ancrages du modèle (M_*, h_max, Δ_log_M) :
       1. h(M_*) = 0.15 (calibration V5.5 étape 6)
       2. β(M_*) = 6.29 (V5.5 étape 5)
       3. M_* = 10^12 M_sun (transition Milky Way scale, V5 §11.4)
    Zéro paramètre libre additionnel."""
    s = summary(M_STAR_MSUN)
    assert s["h"] == pytest.approx(0.15)
    assert s["beta"] == pytest.approx(6.29, rel=0.001)
    assert s["M_star_Msun"] == 1e12
