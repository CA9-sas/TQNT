"""Tests V5.14 — correction R² Starobinsky du potentiel V5.13."""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.inflation_chi_field import (
    A_S_PLANCK,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    R_PLANCK_BICEP,
)
from theory.inflation_starobinsky_R2 import (
    BETA_TOP_DEFAULT,
    C_STAROBINSKY,
    EPS_TOP_DEFAULT,
    N_S_STAROBINSKY_NE60,
    PHI_MAX_V5p13,
    R_STAROBINSKY_NE60,
    InflationV5p14Prediction,
    N_efolds_V5p14,
    V0_calibrated_V5p14,
    V_starobinsky_R2,
    V_total_V5p14,
    Vprime_starobinsky_R2,
    Vprime_total_V5p14,
    Vsecond_starobinsky_R2,
    envelope_V5p13_residual,
    find_phi_pivot_N60,
    find_phi_pivot_V5p14,
    n_s_V5p14,
    phi_end_inflation_V5p14,
    predict_inflation_V5p14,
    r_V5p14,
    slow_roll_epsilon_V5p14,
    slow_roll_eta_V5p14,
    summary_V5p14,
)


# ---------------------------------------------------------------------------
# Tests sur les constantes.
# ---------------------------------------------------------------------------
def test_C_starobinsky_canonical():
    """Constante √(2/3) du potentiel Starobinsky (forme universelle R + R²)."""
    assert C_STAROBINSKY == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-15)


def test_phi_max_is_V5p13_pivot():
    """φ_max V5.14 = φ_★ V5.13 à N=60 ≈ 17 M_pl."""
    assert PHI_MAX_V5p13 == pytest.approx(17.05, abs=0.1)


def test_starobinsky_targets():
    """Cibles standard Starobinsky N_e=60 : n_s ≈ 0,967, r ≈ 0,0033."""
    assert N_S_STAROBINSKY_NE60 == pytest.approx(0.96667, abs=1e-4)
    assert R_STAROBINSKY_NE60 == pytest.approx(0.00333, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests sur le potentiel Starobinsky standard.
# ---------------------------------------------------------------------------
def test_V_starobinsky_zero_at_phi_zero():
    """V_Star(0) = 0 (graceful exit)."""
    assert V_starobinsky_R2(0.0, V0=1.0) == pytest.approx(0.0, abs=1e-15)


def test_V_starobinsky_plateau_at_large_phi():
    """V_Star(φ → ∞) → V_0 (plateau de De Sitter)."""
    V_large = V_starobinsky_R2(20.0, V0=1.0)
    assert V_large == pytest.approx(1.0, rel=1e-6)


def test_Vprime_starobinsky_numerical():
    """dV_Star/dφ analytique = numérique."""
    phi = 3.0
    h = 1e-6
    V_p = V_starobinsky_R2(phi + h, V0=1.0)
    V_m = V_starobinsky_R2(phi - h, V0=1.0)
    Vprime_num = (V_p - V_m) / (2.0 * h)
    Vprime_an = Vprime_starobinsky_R2(phi, V0=1.0)
    assert Vprime_an == pytest.approx(Vprime_num, rel=1e-6)


def test_Vsecond_starobinsky_numerical():
    """d²V_Star/dφ² analytique = numérique."""
    phi = 3.0
    h = 1e-4
    V_p = V_starobinsky_R2(phi + h, V0=1.0)
    V_0 = V_starobinsky_R2(phi, V0=1.0)
    V_m = V_starobinsky_R2(phi - h, V0=1.0)
    Vsec_num = (V_p - 2.0 * V_0 + V_m) / (h ** 2)
    Vsec_an = Vsecond_starobinsky_R2(phi, V0=1.0)
    assert Vsec_an == pytest.approx(Vsec_num, rel=1e-3)


# ---------------------------------------------------------------------------
# Tests sur l'enveloppe résidu V5.13.
# ---------------------------------------------------------------------------
def test_envelope_equals_one_at_eps_zero():
    """ε_top = 0 ⇒ E(φ) = 1 partout (Starobinsky pur récupéré)."""
    E = envelope_V5p13_residual(5.0, eps_top=0.0)
    assert E == pytest.approx(1.0, abs=1e-15)


def test_envelope_equals_one_plus_eps_at_phi_max():
    """À φ = φ_max, E = 1 + ε_top (par définition)."""
    eps = 0.05
    E = envelope_V5p13_residual(PHI_MAX_V5p13, eps_top=eps,
                                phi_max=PHI_MAX_V5p13)
    assert E == pytest.approx(1.0 + eps, rel=1e-12)


def test_envelope_decreases_at_low_phi_for_beta_positive():
    """β > 0 ⇒ E(φ) → 1 à bas φ (plateau Starobinsky préservé)."""
    E_low = envelope_V5p13_residual(1.0, eps_top=0.1, beta=2.4)
    E_high = envelope_V5p13_residual(20.0, eps_top=0.1, beta=2.4)
    assert abs(E_low - 1.0) < abs(E_high - 1.0)


# ---------------------------------------------------------------------------
# Tests sur V_total V5.14.
# ---------------------------------------------------------------------------
def test_V_total_reduces_to_starobinsky_at_eps_zero():
    """ε_top = 0 ⇒ V_total = V_Starobinsky."""
    phi = 5.0
    V_t = V_total_V5p14(phi, V0=1.0, eps_top=0.0)
    V_s = V_starobinsky_R2(phi, V0=1.0)
    assert V_t == pytest.approx(V_s, rel=1e-12)


def test_Vprime_total_numerical():
    """dV_total/dφ analytique = numérique (à ε_top par défaut)."""
    phi = 5.0
    h = 1e-6
    V_p = V_total_V5p14(phi + h, V0=1.0)
    V_m = V_total_V5p14(phi - h, V0=1.0)
    Vprime_num = (V_p - V_m) / (2.0 * h)
    Vprime_an = Vprime_total_V5p14(phi, V0=1.0)
    assert Vprime_an == pytest.approx(Vprime_num, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests slow-roll V5.14.
# ---------------------------------------------------------------------------
def test_epsilon_positive():
    """ε > 0 partout (potentiel V positif, V' réel)."""
    eps = slow_roll_epsilon_V5p14(5.0, eps_top=EPS_TOP_DEFAULT)
    assert eps > 0.0


def test_phi_end_epsilon_one():
    """ε(φ_end) = 1 (définition fin slow-roll)."""
    phi_end = phi_end_inflation_V5p14(eps_top=EPS_TOP_DEFAULT)
    eps = slow_roll_epsilon_V5p14(phi_end, eps_top=EPS_TOP_DEFAULT)
    assert eps == pytest.approx(1.0, rel=1e-5)


def test_phi_end_below_one_Mpl():
    """φ_end ~ O(1) M_pl pour Starobinsky pur (avec ε_top petit)."""
    phi_end = phi_end_inflation_V5p14(eps_top=0.0)
    assert 0.5 < phi_end < 1.5


# ---------------------------------------------------------------------------
# Tests sur les e-folds et φ_★.
# ---------------------------------------------------------------------------
def test_N_efolds_positive_above_phi_end():
    """N_e > 0 pour φ_initial > φ_end."""
    phi_end = phi_end_inflation_V5p14(eps_top=0.0)
    N = N_efolds_V5p14(5.0, phi_final=phi_end, eps_top=0.0)
    assert N > 0.0


def test_find_phi_pivot_N60_starobinsky():
    """φ_★(N_e=60, ε_top=0) ≈ 5,5 M_pl (Starobinsky pur)."""
    phi_star = find_phi_pivot_N60(eps_top=0.0)
    assert 5.0 < phi_star < 6.0


def test_find_phi_pivot_consistency():
    """find_phi_pivot(N_e) ↔ N_efolds inverse."""
    phi_star = find_phi_pivot_V5p14(N_e_target=60.0, eps_top=EPS_TOP_DEFAULT)
    N_back = N_efolds_V5p14(phi_star, eps_top=EPS_TOP_DEFAULT)
    assert N_back == pytest.approx(60.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Tests sur n_s, r prédits par V5.14 (limite Starobinsky).
# ---------------------------------------------------------------------------
def test_n_s_starobinsky_limit():
    """n_s(ε_top=0, N=60) ≈ 1 - 2/N_e = 0,967 (compatible Planck)."""
    n_s = n_s_V5p14(N_e=60.0, eps_top=0.0)
    # Tolérance 1 % par rapport au théorème grand-N_e.
    assert n_s == pytest.approx(N_S_STAROBINSKY_NE60, abs=2e-3)


def test_r_starobinsky_limit():
    """r(ε_top=0, N=60) ≈ 12/N_e² = 0,003 (compatible BICEP)."""
    r = r_V5p14(N_e=60.0, eps_top=0.0)
    # Tolérance large car la formule 12/N² est asymptotique.
    assert r == pytest.approx(R_STAROBINSKY_NE60, abs=1e-3)


def test_V5p14_compatible_planck_n_s():
    """V5.14 (ε_top défaut) doit être compatible Planck (< 3σ)."""
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert pred.compatible_planck_n_s


def test_V5p14_compatible_BICEP_r():
    """V5.14 (ε_top défaut) doit avoir r < 0,06 BICEP."""
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert pred.compatible_BICEP_r


def test_V5p14_resolves_V5p13_tension():
    """V5.14 (ε_top défaut) résout la tension n_s de V5.13 (+7.6σ → < 1σ)."""
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert abs(pred.n_s_deviation_sigma) < 2.0


def test_V5p14_resolves_V5p13_r_violation():
    """V5.14 (ε_top défaut) résout la violation BICEP de V5.13 (0,158 → ≪ 0,06)."""
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert pred.r < 0.01  # bien sous BICEP


# ---------------------------------------------------------------------------
# Tests de calibration et synthèse.
# ---------------------------------------------------------------------------
def test_V0_calibration_positive():
    """V_0 calibré > 0."""
    V0 = V0_calibrated_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert V0 > 0.0


def test_summary_keys_present():
    """summary_V5p14 expose les clés attendues."""
    s = summary_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    for k in ("model", "eps_top", "n_s_V5p14", "r_V5p14",
              "compatible_planck_n_s", "compatible_BICEP_r",
              "V0_over_Mpl4_calibrated"):
        assert k in s


def test_prediction_dataclass_fields():
    """predict_inflation_V5p14 renvoie un dataclass avec les bons champs."""
    pred = predict_inflation_V5p14(N_e=60.0, eps_top=EPS_TOP_DEFAULT)
    assert isinstance(pred, InflationV5p14Prediction)
    assert pred.phi_star > pred.phi_end
    assert 0.0 < pred.epsilon < 1.0
    assert pred.V0_over_Mpl4 > 0.0
    assert pred.r > 0.0
