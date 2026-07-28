"""Tests V5.16 — chi-field reheating, derivation of N_e."""
from __future__ import annotations

import math

import pytest

from theory.coupling_kg import ALPHA_EM, C_N_NUCLEON_EFF
from theory.inflation_chi_field import (
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    R_PLANCK_BICEP,
    N_EFOLDS_PIVOT,
)
from theory.inflation_starobinsky_R2 import EPS_TOP_DEFAULT
from theory.reheating_chi import (
    G_STAR_SM,
    M_PL_REDUCED_GEV,
    N_PIVOT_LL,
    T_REH_BBN_MIN_GEV,
    W_REH_V5,
    H_end_V5p14,
    N_e_V5,
    ReheatingV5Prediction,
    T_end_V5,
    T_reh_V5,
    chi_field_decay_rate,
    chi_field_decay_suppression,
    n_s_starobinsky_at_Ne,
    predict_reheating_V5,
    r_starobinsky_at_Ne,
    rho_end_V5p14,
    summary,
    validate_against_Planck,
)


# ---------------------------------------------------------------------------
# 1. Constants and identifications.
# ---------------------------------------------------------------------------
def test_alpha_EM_topological_suppression():
    """alpha_EM^{c_n_eff} = (1/137)^18 ~ 5e-39."""
    sup = chi_field_decay_suppression(c_n_eff=18)
    assert sup == pytest.approx(ALPHA_EM ** 18, rel=1e-12)
    assert 1e-40 < sup < 1e-38


def test_g_star_SM_full():
    """g_star_SM = 106.75 (full Standard Model above EW scale)."""
    assert G_STAR_SM == pytest.approx(106.75, rel=1e-3)


def test_N_pivot_LL_value():
    """Liddle-Leach pivot constant 56.12 for k_* = 0.05 Mpc^-1."""
    assert 55.0 < N_PIVOT_LL < 57.0


def test_M_pl_reduced_value():
    """Reduced Planck mass ~ 2.435e18 GeV."""
    assert M_PL_REDUCED_GEV == pytest.approx(2.435e18, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. chi-field decay rate.
# ---------------------------------------------------------------------------
def test_decay_rate_linear_in_H():
    """Gamma_reh is linear in H_end."""
    g1 = chi_field_decay_rate(1.0)
    g2 = chi_field_decay_rate(2.0)
    assert g2 / g1 == pytest.approx(2.0, rel=1e-12)


def test_decay_rate_topological_factor():
    """Gamma_reh = alpha_EM^{c_n} * H_end (prefactor 1)."""
    H_end = 1e13
    Gamma = chi_field_decay_rate(H_end, c_n_eff=18, prefactor=1.0)
    expected = ALPHA_EM ** 18 * H_end
    assert Gamma == pytest.approx(expected, rel=1e-12)


def test_decay_rate_prefactor_scaling():
    """Linear scaling with the O(1) prefactor."""
    H_end = 1e13
    g1 = chi_field_decay_rate(H_end, prefactor=1.0)
    g10 = chi_field_decay_rate(H_end, prefactor=10.0)
    assert g10 / g1 == pytest.approx(10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Hubble and energy density at end of inflation.
# ---------------------------------------------------------------------------
def test_H_end_positive_and_bounded():
    """H_end is in the GUT-ish range 1e12 - 1e14 GeV for V5.14 calibration."""
    H_end = H_end_V5p14()
    assert 1e12 < H_end < 1e14


def test_rho_end_consistent_with_H_end():
    """rho_end = 3 H_end^2 M_pl^2 (Friedmann in reduced Planck units)."""
    H_end = H_end_V5p14()
    rho_end_GeV4 = 3.0 * (H_end * M_PL_REDUCED_GEV) ** 2
    # rho_end_V5p14 returns in M_pl^4 units; convert
    rho_mpl4 = rho_end_V5p14()
    rho_GeV4 = rho_mpl4 * M_PL_REDUCED_GEV ** 4
    assert rho_GeV4 == pytest.approx(rho_end_GeV4, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Reheating temperature.
# ---------------------------------------------------------------------------
def test_T_reh_below_BBN_for_cn_18():
    """HONEST: With c_n=18 the V5 chi-field T_reh ~ 0.15 MeV is BELOW BBN.

    This is a genuine tension of V5.16. The c_n=18 suppression of the
    chi -> baryon vertex is too strong for the standard Starobinsky V_0.
    """
    T_reh = T_reh_V5(c_n_eff=18)
    assert T_reh < T_REH_BBN_MIN_GEV   # documented tension
    assert T_reh > 1e-6                # not absurdly small


def test_T_reh_safe_for_lower_cn():
    """For c_n <= 12 the V5 chi-field T_reh exceeds the BBN bound."""
    T_reh = T_reh_V5(c_n_eff=12)
    assert T_reh > T_REH_BBN_MIN_GEV


def test_T_reh_scales_with_suppression():
    """T_reh ~ sqrt(alpha_EM^{c_n}). Lower c_n -> exponentially higher T_reh."""
    T6 = T_reh_V5(c_n_eff=6)
    T18 = T_reh_V5(c_n_eff=18)
    # Ratio expected = sqrt(alpha^{-12}) = 137^6
    ratio_expected = math.sqrt(ALPHA_EM ** (6 - 18))
    assert T6 / T18 == pytest.approx(ratio_expected, rel=1e-6)


def test_T_end_above_T_reh():
    """T_end >> T_reh: the universe redshifts between inflation end and
    completion of reheating."""
    T_end = T_end_V5()
    T_reh = T_reh_V5(c_n_eff=18)
    assert T_end > T_reh


# ---------------------------------------------------------------------------
# 5. N_e derivation.
# ---------------------------------------------------------------------------
def test_N_e_in_finite_range():
    """N_e should be a finite positive number, even if outside the
    Planck-favored 50-65 window."""
    N_e = N_e_V5()
    assert math.isfinite(N_e)
    assert 20.0 < N_e < 80.0


def test_N_e_with_radiation_EoS_is_higher():
    """w_reh = 1/3 kills the reheating term -> higher N_e than w=0."""
    N_w0 = N_e_V5(w_reh=0.0)
    N_w13 = N_e_V5(w_reh=1.0 / 3.0)
    assert N_w13 > N_w0


def test_N_e_independent_of_prefactor_logarithmically():
    """Changing the O(1) prefactor moves N_e by less than 1 e-fold."""
    N1 = N_e_V5(prefactor=1.0)
    N10 = N_e_V5(prefactor=10.0)
    assert abs(N_e_V5(prefactor=100.0) - N1) < 2.0
    assert abs(N10 - N1) < 1.0


def test_N_e_decreases_with_higher_cn():
    """Higher topological suppression -> lower T_reh -> lower N_e (w=0)."""
    N_low = N_e_V5(c_n_eff=6)
    N_high = N_e_V5(c_n_eff=18)
    assert N_high < N_low


# ---------------------------------------------------------------------------
# 6. CMB observables at N_e_V5.
# ---------------------------------------------------------------------------
def test_n_s_at_Ne_formula():
    """n_s = 1 - 2/N_e (Starobinsky)."""
    assert n_s_starobinsky_at_Ne(60.0) == pytest.approx(0.96667, abs=1e-4)
    assert n_s_starobinsky_at_Ne(50.0) == pytest.approx(0.96, abs=1e-4)


def test_r_at_Ne_formula():
    """r = 12/N_e^2 (Starobinsky)."""
    assert r_starobinsky_at_Ne(60.0) == pytest.approx(0.003333, abs=1e-4)
    assert r_starobinsky_at_Ne(50.0) == pytest.approx(0.0048, abs=1e-4)


def test_n_s_at_N_e_V5_within_3sigma_for_radiation_EoS():
    """For w_reh=1/3 the predicted n_s is compatible with Planck at <3sigma."""
    pred = predict_reheating_V5(w_reh=1.0 / 3.0)
    assert abs(pred.n_s_deviation_sigma) < 3.0


def test_BICEP_bound_satisfied():
    """r < 0.06 in all V5 scenarios considered (Starobinsky shape)."""
    pred = predict_reheating_V5()
    assert pred.r_at_Ne_V5 < R_PLANCK_BICEP


# ---------------------------------------------------------------------------
# 7. Synthesis dataclass and validation API.
# ---------------------------------------------------------------------------
def test_predict_returns_dataclass():
    """predict_reheating_V5 returns a ReheatingV5Prediction."""
    pred = predict_reheating_V5()
    assert isinstance(pred, ReheatingV5Prediction)


def test_validate_summary_keys():
    """validate_against_Planck returns the expected keys."""
    v = validate_against_Planck()
    required = {"N_e_V5", "n_s_V5", "r_V5", "T_reh_GeV", "BBN_compatible",
                "Planck_compatible", "BICEP_compatible", "Delta_N_e"}
    assert required.issubset(set(v.keys()))


def test_summary_alias():
    """summary() and validate_against_Planck() agree."""
    a = summary()
    b = validate_against_Planck()
    assert a == b


def test_honest_tension_documented():
    """The honest V5.16 verdict: with c_n=18 and w=0, the model is in
    tension with both Planck (n_s) and BBN (T_reh)."""
    pred = predict_reheating_V5()    # c_n=18, w=0 defaults
    assert not pred.BBN_compatible
    assert not pred.Planck_compatible
    # but BICEP r-upper is still satisfied
    assert pred.BICEP_compatible
