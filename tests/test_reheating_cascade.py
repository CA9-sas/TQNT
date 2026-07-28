"""Unit tests for V5.17 cascade reheating (`theory.reheating_cascade`)."""
from __future__ import annotations

import math

import pytest

from theory.coupling_kg import ALPHA_EM, C_N_NUCLEON_EFF
from theory.inflation_chi_field import (
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    R_PLANCK_BICEP,
)
from theory.reheating_cascade import (
    C_N_UNKNOT,
    G_STAR_SM,
    PREFACTOR_STAGE1,
    T_REH_BBN_MIN_GEV,
    W_REH_V5p17,
    CascadeReheatingV5p17,
    Gamma_reh_stage1_unknot,
    Gamma_reh_stage2_nucleon,
    N_e_V5p17,
    T_high_after_stage1,
    T_reh_after_thermalization,
    n_s_r_recomputed,
    predict_cascade_V5p17,
    validate_against_Planck_V5p17,
)
from theory.reheating_chi import validate_against_Planck as v_V5p16


# ---------------------------------------------------------------------------
# Section 1: physical constants / consistency
# ---------------------------------------------------------------------------
def test_c_n_unknot_is_zero():
    """The unknot has zero crossings by definition."""
    assert C_N_UNKNOT == 0


def test_w_reh_is_radiation():
    """Stage-1 instantaneous reheating gives radiation EoS w = 1/3."""
    assert W_REH_V5p17 == pytest.approx(1.0 / 3.0)


def test_prefactor_stage1_O1():
    """Stage-1 prefactor must be of order unity (Witten Jones(0_1)=1)."""
    assert 0.1 < PREFACTOR_STAGE1 < 10.0


# ---------------------------------------------------------------------------
# Section 2: Stage-1 decay rate
# ---------------------------------------------------------------------------
def test_gamma_stage1_no_suppression():
    """V5.17 key: Gamma_stage1 must equal H_end (no alpha_EM^{c_n})."""
    H_end = 1e13  # GeV
    Gamma1 = Gamma_reh_stage1_unknot(H_end)
    assert Gamma1 == pytest.approx(H_end)
    # Check it is NOT suppressed by alpha_EM^{18}
    assert Gamma1 > 1e30 * (ALPHA_EM ** C_N_NUCLEON_EFF) * H_end


def test_gamma_stage1_scales_with_H_end():
    """Gamma_stage1 linear in H_end."""
    H1 = 1e12
    H2 = 1e14
    assert Gamma_reh_stage1_unknot(H2) / Gamma_reh_stage1_unknot(H1) == \
        pytest.approx(H2 / H1)


def test_gamma_stage2_recovers_v5p16():
    """V5.17 Stage-2 direct rate must equal V5.16 single-stage rate."""
    H_end = 1e13
    Gamma2 = Gamma_reh_stage2_nucleon(H_end)
    expected_v5p16 = (ALPHA_EM ** C_N_NUCLEON_EFF) * H_end
    assert Gamma2 == pytest.approx(expected_v5p16, rel=1e-12)


def test_gamma_stage1_dominates_stage2():
    """Stage-1 / Stage-2 ratio is the Vassiliev factor alpha_EM^{-18}."""
    H_end = 1e13
    ratio = (Gamma_reh_stage1_unknot(H_end)
             / Gamma_reh_stage2_nucleon(H_end))
    expected = ALPHA_EM ** (-C_N_NUCLEON_EFF)
    assert ratio == pytest.approx(expected, rel=1e-12)
    # Numerically: alpha_EM^{-18} ~ 137^{18} ~ 3e38
    assert ratio > 1e30


# ---------------------------------------------------------------------------
# Section 3: Temperatures
# ---------------------------------------------------------------------------
def test_T_high_above_BBN():
    """V5.17 must satisfy T_reh > T_BBN = 1 MeV."""
    T_high = T_high_after_stage1()
    assert T_high > T_REH_BBN_MIN_GEV, (
        f"T_high = {T_high:.3e} GeV is below BBN bound "
        f"{T_REH_BBN_MIN_GEV} GeV")
    # Actually orders of magnitude above:
    assert T_high > 1e10  # GeV (very large margin)


def test_T_reh_equals_T_high():
    """Stage-2 thermalization is instantaneous: T_reh == T_high."""
    assert T_reh_after_thermalization() == pytest.approx(
        T_high_after_stage1(), rel=1e-12)


def test_T_high_scales_as_sqrt_H_end():
    """T_high ~ sqrt(H_end M_pl): doubling H_end multiplies T by sqrt(2)."""
    T1 = T_high_after_stage1(H_end_GeV=1e12)
    T2 = T_high_after_stage1(H_end_GeV=4e12)
    # T ~ sqrt(H_end) -> 4x in H_end -> 2x in T
    assert T2 / T1 == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Section 4: e-folds and CMB
# ---------------------------------------------------------------------------
def test_N_e_in_planck_window():
    """V5.17 N_e must lie in the Planck-favored 50-65 window."""
    N_e = N_e_V5p17()
    assert 50.0 <= N_e <= 65.0, f"N_e = {N_e:.3f} outside Planck window"


def test_n_s_compatible_planck():
    """V5.17 n_s within 3 sigma of Planck."""
    N_e = N_e_V5p17()
    n_s, _r = n_s_r_recomputed(N_e)
    deviation = abs(n_s - N_S_PLANCK) / N_S_PLANCK_SIGMA
    assert deviation < 3.0, (
        f"n_s = {n_s:.4f} deviates {deviation:.2f} sigma from Planck"
    )


def test_r_below_BICEP():
    """V5.17 r must be below BICEP/Keck 95% CL upper bound."""
    N_e = N_e_V5p17()
    _n_s, r = n_s_r_recomputed(N_e)
    assert r < R_PLANCK_BICEP, f"r = {r:.4f} violates BICEP < {R_PLANCK_BICEP}"


def test_n_s_r_recomputed_formulas():
    """n_s and r from the Starobinsky asymptotic formulas."""
    N_e = 55.0
    n_s, r = n_s_r_recomputed(N_e)
    assert n_s == pytest.approx(1.0 - 2.0 / 55.0)
    assert r == pytest.approx(12.0 / 55.0 ** 2)


def test_n_s_r_recomputed_rejects_invalid():
    """N_e <= 0 must raise."""
    with pytest.raises(ValueError):
        n_s_r_recomputed(0.0)
    with pytest.raises(ValueError):
        n_s_r_recomputed(-10.0)


# ---------------------------------------------------------------------------
# Section 5: Full cascade prediction
# ---------------------------------------------------------------------------
def test_predict_cascade_returns_dataclass():
    """predict_cascade_V5p17 returns a CascadeReheatingV5p17."""
    pred = predict_cascade_V5p17()
    assert isinstance(pred, CascadeReheatingV5p17)


def test_cascade_all_compatibilities_pass():
    """The 3 compatibility flags must all be True in V5.17."""
    pred = predict_cascade_V5p17()
    assert pred.BBN_compatible, f"BBN failed: T_reh = {pred.T_reh_GeV:.3e}"
    assert pred.Planck_compatible, (
        f"Planck failed: deviation = {pred.n_s_deviation_sigma:.2f} sigma"
    )
    assert pred.BICEP_compatible, f"BICEP failed: r = {pred.r_V5p17:.4f}"


def test_cascade_resolves_v5p16_tension():
    """V5.17 T_reh must beat the V5.16 T_reh by > 10^{15}."""
    pred_v5p17 = predict_cascade_V5p17()
    v5p16 = v_V5p16()
    ratio = pred_v5p17.T_reh_GeV / v5p16["T_reh_GeV"]
    # alpha_EM^{-18/2} ~ 137^{9} ~ 1e19; we expect a similarly enormous gain
    assert ratio > 1e15, f"V5.17/V5.16 T_reh ratio only {ratio:.2e}"


def test_cascade_N_e_better_than_v5p16():
    """V5.17 N_e must be closer to 60 than V5.16 N_e."""
    pred_v5p17 = predict_cascade_V5p17()
    v5p16 = v_V5p16()
    assert abs(pred_v5p17.N_e_V5p17 - 60.0) \
        < abs(v5p16["N_e_V5"] - 60.0)


# ---------------------------------------------------------------------------
# Section 6: validate_against_Planck_V5p17 dict
# ---------------------------------------------------------------------------
def test_validate_dict_has_all_keys():
    """The validation dict must expose all useful quantities."""
    d = validate_against_Planck_V5p17()
    required = {
        "model", "H_end_GeV", "Gamma_stage1_GeV", "T_high_GeV",
        "T_reh_GeV", "T_BBN_min_GeV", "T_reh_over_T_BBN", "w_reh",
        "N_e_V5p17", "n_s_V5p17", "r_V5p17", "BBN_compatible",
        "Planck_compatible", "BICEP_compatible",
    }
    assert required.issubset(d.keys())


def test_validate_dict_w_reh_radiation():
    """w_reh in the summary must be 1/3."""
    d = validate_against_Planck_V5p17()
    assert d["w_reh"] == pytest.approx(1.0 / 3.0)
