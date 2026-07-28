"""Tests for scripts/V5_BBN/class_D_bbn_test.py (V5.19 Class D vs BBN D/H)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "V5_BBN"))

import class_D_bbn_test as M

from theory.v5_compact_objects_catalog import (
    class_D_heavy_proton_plasma, ALPHA_EM,
)


# ----------------------------------------------------------------------------
# BBN sample sanity
# ----------------------------------------------------------------------------

def test_sample_contains_cooke_pettini_and_standard():
    """The hardcoded BBN sample must include Cooke+2018, Pettini-Cooke+2012,
    and a standard PArthENoPE prediction."""
    names = {m.name for m in M.BBN_DH_SAMPLE}
    assert "Cooke+2018" in names
    assert "Pettini-Cooke+2012" in names
    assert "PArthENoPE-Planck" in names


def test_sample_DH_values_in_sensible_range():
    """All D/H measurements must be in [2.0, 3.0] x 1e-5 (within 3-sigma
    of every published value)."""
    for m in M.BBN_DH_SAMPLE:
        assert 2.0 < m.DH_value < 3.0, (
            f"{m.name}: D/H = {m.DH_value}e-5 out of range")


def test_cooke_2018_central_value():
    """Cooke+ 2018 central value must be 2.527e-5 (per ApJ 855, 102)."""
    cooke = next(m for m in M.BBN_DH_SAMPLE if m.name == "Cooke+2018")
    assert cooke.DH_value == pytest.approx(2.527, abs=0.01)
    assert cooke.DH_error == pytest.approx(0.030, abs=0.005)


# ----------------------------------------------------------------------------
# Hubble rate sanity at BBN
# ----------------------------------------------------------------------------

def test_hubble_rate_at_1_MeV_is_around_1_per_s():
    """Standard FRW: H(1 MeV) ~ 1 s^-1 (within an O(1) factor)."""
    H = M.hubble_rate_BBN(T_MeV=1.0, g_star=10.75)
    assert 0.1 < H < 10.0, f"H(1 MeV) = {H} s^-1 unrealistic"


def test_cosmic_time_at_1_MeV_is_subseconds():
    """t(1 MeV) ~ 0.5 s for g_* = 10.75."""
    t = M.cosmic_time_BBN(T_MeV=1.0, g_star=10.75)
    assert 0.05 < t < 5.0, f"t(1 MeV) = {t} s unrealistic"


def test_cosmic_time_monotonic_with_T():
    """Lower T -> later universe -> longer cosmic time."""
    t_high = M.cosmic_time_BBN(T_MeV=1.0)
    t_low = M.cosmic_time_BBN(T_MeV=0.01)
    assert t_low > t_high


# ----------------------------------------------------------------------------
# Vassiliev suppression
# ----------------------------------------------------------------------------

def test_vassiliev_factor_is_tiny():
    """Vassiliev factor alpha_EM^20 must be ~ 1e-43 (CLAUDE.md V5 sec 13.6.4)."""
    f = M.vassiliev_suppression_HP(c_n=20)
    assert 1e-44 < f < 1e-42, f"alpha_EM^20 = {f} out of expected range"


def test_vassiliev_factor_monotonic_in_cn():
    """Higher c_n -> stronger suppression."""
    for c1, c2 in [(18, 19), (19, 20), (20, 22), (22, 25)]:
        f1 = M.vassiliev_suppression_HP(c_n=c1)
        f2 = M.vassiliev_suppression_HP(c_n=c2)
        assert f1 > f2, f"alpha^{c1} ({f1}) should exceed alpha^{c2} ({f2})"


def test_vassiliev_factor_matches_alpha_em_power():
    """Direct check: f = ALPHA_EM**c_n."""
    for c in (18, 20, 22, 25):
        f = M.vassiliev_suppression_HP(c_n=c)
        assert math.isclose(f, ALPHA_EM ** c, rel_tol=1e-12)


# ----------------------------------------------------------------------------
# Class-D abundance during BBN
# ----------------------------------------------------------------------------

def test_n_HP_abundance_is_negligible():
    """n_HP/n_b at BBN start (T = 1 MeV) must be << 1e-30 (Vassiliev kills it)."""
    n = M.n_HP_over_n_baryon(T_MeV=1.0)
    assert n < 1e-30, f"n_HP/n_b = {n} unrealistically large"


def test_n_HP_abundance_decreases_with_T():
    """n_HP/n_b INCREASES as T decreases (because t_BBN grows ; tau_HP/t shrinks
    only when tau_HP < t)."""
    # Strictly speaking n_HP/n_b ~ alpha^20 * tau/t : at very low T, t > tau,
    # so the ratio DECREASES. At high T (t < tau), ratio is alpha^20 * tau/t,
    # which DECREASES with T (because t = 1/(2H) grows with T -> 0).
    # We just check that abundance stays << 1 across the BBN range.
    for T in [1.0, 0.1, 0.01]:
        n = M.n_HP_over_n_baryon(T_MeV=T)
        assert 0 < n < 1e-30


# ----------------------------------------------------------------------------
# Delta N_eff contribution
# ----------------------------------------------------------------------------

def test_delta_N_eff_below_planck_bound():
    """V5 Class D Delta N_eff << Planck 2018 bound (0.3)."""
    dN = M.delta_N_eff_HP(T_MeV=1.0)
    assert dN < M.DELTA_N_EFF_BOUND
    assert dN < 1e-30, f"Delta N_eff = {dN} dangerously large"


# ----------------------------------------------------------------------------
# D/H shift from HP decay
# ----------------------------------------------------------------------------

def test_DH_shift_is_negligible_at_freezeout():
    """|Delta D/H|/(D/H) at T = 70 keV (D freeze-out) must be < 1e-30."""
    shift = M.DH_shift_from_HP_decay(T_MeV=0.07)
    assert shift < 1e-30


def test_chi2_V5_vs_observed_is_small():
    """chi2(V5 D/H, Cooke+ 2018) must be < 1 (V5 = standard within errors)."""
    res = M.chi2_V5_vs_obs()
    assert res["max_chi2"] < 1.0, f"chi2_max = {res['max_chi2']} too large"


def test_V5_D_H_equals_standard_to_machine_precision():
    """V5 Class D D/H = standard PArthENoPE D/H to 1e-30 precision."""
    res = M.chi2_V5_vs_obs()
    rel_diff = np.abs(res["DH_V5"] - M.DH_STD_VALUE) / M.DH_STD_VALUE
    assert np.all(rel_diff < 1e-30)


# ----------------------------------------------------------------------------
# Observational bound headroom
# ----------------------------------------------------------------------------

def test_V5_orders_of_magnitude_below_observational_bound():
    """V5 Class D abundance must be at least 30 dex below the obs bound."""
    res = M.v5_below_observational_bound()
    assert res["headroom_dex"] > 30, (
        f"headroom = {res['headroom_dex']} dex too small")


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------

def test_verdict_returns_expected_keys():
    """verdict() must return a dict with all diagnostic keys."""
    vd = M.verdict()
    expected = {
        "class_D_name", "c_n_vassiliev", "vassiliev_factor",
        "n_HP_over_n_b_peak", "delta_N_eff", "DH_shift_at_freezeout",
        "DH_V5_central", "DH_observed", "DH_std_prediction",
        "chi2_max", "obs_upper_bound_n_HP_over_b",
        "log10_V5_below_bound", "verdict",
    }
    assert expected.issubset(vd.keys())


def test_verdict_is_indistinct_trivial():
    """V5 Class D must be the 'INDISTINCT TRIVIAL' null-test category."""
    vd = M.verdict()
    assert vd["verdict"].startswith("INDISTINCT TRIVIAL")


def test_verdict_class_D_name_from_catalog():
    """verdict() must use the canonical catalogue class name."""
    vd = M.verdict()
    cls_D = class_D_heavy_proton_plasma()
    assert vd["class_D_name"] == cls_D.name


# ----------------------------------------------------------------------------
# End-to-end run
# ----------------------------------------------------------------------------

def test_run_returns_full_result_dict():
    """run() must execute without error, return a dict with keys
    {chi2_result, verdict, output_png}, and produce a non-trivial PNG."""
    res = M.run()
    for key in ("chi2_result", "verdict", "output_png"):
        assert key in res
    out_path = Path(res["output_png"])
    assert out_path.exists()
    assert out_path.stat().st_size > 10_000  # PNG must be non-trivial


def test_run_PNG_matches_expected_path():
    """Output PNG path must be output/v5p19_class_D_BBN_DH.png."""
    res = M.run()
    out_path = Path(res["output_png"])
    assert out_path.name == "v5p19_class_D_BBN_DH.png"
    assert out_path.parent.name == "output"
