"""Tests for scripts/V5_GW/class_F_imbh_test.py
(V5.19 Class F vs LIGO-Virgo-KAGRA IMBH events).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "V5_GW"))

import class_F_imbh_test as M

from theory.v5_compact_objects_catalog import (
    class_F_unknot_defects,
    OMEGA_DM,
)


# ----------------------------------------------------------------------------
# LVK sample sanity
# ----------------------------------------------------------------------------

def test_sample_contains_GW190521():
    """GW190521 (flagship IMBH-formation event) must be in the LVK sample."""
    names = {ev.name for ev in M.GW_IMBH_SAMPLE}
    assert "GW190521" in names


def test_sample_GW190521_is_imbh_candidate():
    """GW190521 final-remnant mass = 142 M_sun => IMBH candidate by definition."""
    gw190521 = next(ev for ev in M.GW_IMBH_SAMPLE if ev.name == "GW190521")
    assert gw190521.is_imbh_candidate
    assert gw190521.Mf_Msun == pytest.approx(142.0, abs=5.0)
    assert gw190521.Mf_Msun >= 100.0


def test_sample_GWTC4_events_present():
    """At least one GWTC-4 partial event must be in the sample."""
    catalogs = {ev.catalog for ev in M.GW_IMBH_SAMPLE}
    assert "GWTC-4" in catalogs


def test_sample_masses_in_realistic_range():
    """All events must have M_f, M1, M2 within physically sensible ranges."""
    for ev in M.GW_IMBH_SAMPLE:
        assert 1.0 < ev.M1_Msun < 1e5, f"{ev.name}: M1 = {ev.M1_Msun} unrealistic"
        assert 1.0 < ev.M2_Msun < 1e5, f"{ev.name}: M2 = {ev.M2_Msun} unrealistic"
        assert ev.M2_Msun <= ev.M1_Msun + 1e-3, (
            f"{ev.name}: secondary heavier than primary")
        assert ev.Mf_Msun > 0
        assert ev.z > 0


# ----------------------------------------------------------------------------
# Class F catalogue values
# ----------------------------------------------------------------------------

def test_class_F_mass_is_IMBH_range():
    """V5 Class F M_typical must lie in the IMBH range [100, 1e5] M_sun."""
    M_F = M.class_F_typical_mass_Msun()
    assert 100 < M_F < 1e5, f"M_F = {M_F} not in IMBH range"


def test_class_F_M_typical_matches_brief():
    """Catalogue M_typical must be ~ 1.6e4 M_sun (within factor 10)."""
    M_F = M.class_F_typical_mass_Msun()
    assert 1e3 < M_F < 1e5
    # The catalogue value should be close to the 1.6e4 mentioned in the brief
    # within an order of magnitude.
    assert abs(math.log10(M_F / 1.6e4)) < 1.0


def test_class_F_omega_fraction():
    """V5 Class F f_F = 1e-4 (catalogue default)."""
    f_F = M.class_F_omega_fraction()
    assert f_F == pytest.approx(1e-4, rel=1e-9)


def test_class_F_below_observational_abundance_bound():
    """V5 f_F < LIGO IMBH bound (1e-3 from Coen+ 2023)."""
    assert M.is_below_abundance_constraint()
    assert M.class_F_omega_fraction() < M.F_IMBH_OBS_UPPER_BOUND


def test_class_F_headroom_factor_of_10():
    """f_F = 1e-4 is exactly 10x below f_IMBH_bound = 1e-3."""
    h = M.abundance_headroom_factor()
    assert h == pytest.approx(10.0, rel=1e-9)


# ----------------------------------------------------------------------------
# Merger rate prediction
# ----------------------------------------------------------------------------

def test_merger_rate_positive():
    """Class F merger rate must be > 0."""
    R = M.class_F_merger_rate_density()
    assert R > 0


def test_merger_rate_scales_with_f_F():
    """R_F ~ f_F^{53/37} : doubling f_F gives factor 2^{53/37} ~ 2.69."""
    R1 = M.class_F_merger_rate_density(f_F=1e-4, M_F_Msun=1.6e4)
    R2 = M.class_F_merger_rate_density(f_F=2e-4, M_F_Msun=1.6e4)
    ratio = R2 / R1
    expected = 2.0 ** (53.0 / 37.0)
    assert ratio == pytest.approx(expected, rel=1e-6)


def test_predictions_increase_O3_to_O5():
    """N_O5 > N_O4 > N_O3 (LIGO sensitivity grows)."""
    preds = M.predict_detections_all_runs()
    assert preds["N_O4"] > preds["N_O3"]
    assert preds["N_O5"] > preds["N_O4"]


def test_predicted_N_O3_is_below_observed():
    """V5 predicts << 1 detection at O3 -- sub-dominant to observed BBH IMBH."""
    preds = M.predict_detections_all_runs()
    n_obs = M.observed_imbh_count("GWTC-3")
    assert preds["N_O3"] < n_obs
    # Should be subdominant by at least factor 100
    assert preds["N_O3"] < 0.01 * max(n_obs, 1)


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------

def test_verdict_returns_expected_keys():
    """verdict() must return a dict with all diagnostic keys."""
    vd = M.verdict()
    expected = {
        "class_F_name", "f_F", "M_F_Msun", "M_F_max_Msun",
        "Omega_F", "n_F_Mpc3", "n_F_Gpc3", "R_F_Gpc3_yr",
        "f_IMBH_bound", "abundance_headroom",
        "N_O3_pred", "N_O4_pred", "N_O5_pred",
        "n_obs_imbh_GWTC3", "n_obs_imbh_GWTC4", "n_obs_total",
        "verdict",
    }
    assert expected.issubset(vd.keys())


def test_verdict_is_survives():
    """Given f_F = 1e-4 < 1e-3 and R_F << observed, verdict must be SURVIVES."""
    vd = M.verdict()
    assert "SURVIVES" in vd["verdict"]
    assert "EXCLUDED" not in vd["verdict"]


def test_verdict_class_F_name_from_catalog():
    """verdict() must use the canonical catalogue class name."""
    vd = M.verdict()
    cls_F = class_F_unknot_defects()
    assert vd["class_F_name"] == cls_F.name


# ----------------------------------------------------------------------------
# End-to-end run
# ----------------------------------------------------------------------------

def test_run_returns_full_result_dict():
    """run() must execute, return {verdict, output_png}, and produce a PNG."""
    res = M.run()
    for key in ("verdict", "output_png"):
        assert key in res
    out_path = Path(res["output_png"])
    assert out_path.exists()
    assert out_path.stat().st_size > 10_000  # PNG must be non-trivial


def test_run_PNG_matches_expected_path():
    """Output PNG path must be output/v5p19_class_F_imbh_ligo.png."""
    res = M.run()
    out_path = Path(res["output_png"])
    assert out_path.name == "v5p19_class_F_imbh_ligo.png"
    assert out_path.parent.name == "output"
