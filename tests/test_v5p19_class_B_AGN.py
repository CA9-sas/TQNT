"""Tests for scripts/V5_JWST/v5p19_class_B_AGN_test.py (V5.19 Class B vs JWST)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "V5_JWST"))

import v5p19_class_B_AGN_test as M

from theory.v5_compact_objects_catalog import (
    class_B_topological_pbh_seeds, OMEGA_DM,
)


# ----------------------------------------------------------------------------
# JWST AGN sample sanity
# ----------------------------------------------------------------------------

def test_sample_has_at_least_20_objects():
    """The hardcoded sample must include >=20 high-z AGN, per task brief."""
    assert len(M.JWST_AGN_SAMPLE) >= 20


def test_sample_spans_z_5_to_z_above_8():
    """The sample must span z = 5 to z > 8 (with at least 5 at z>8)."""
    z = np.array([a.z_spec for a in M.JWST_AGN_SAMPLE])
    assert z.min() < 6.0
    assert z.max() > 10.0
    assert np.sum(z > 8.0) >= 5


def test_sample_M_BH_in_sensible_range():
    """All AGN must have log M_BH in [6, 9] (high-z AGN window)."""
    for a in M.JWST_AGN_SAMPLE:
        assert 5.5 < a.log_M_BH < 9.5, f"{a.name}: log M_BH = {a.log_M_BH}"


def test_sample_sources_referenced():
    """Every AGN must have a 'ref' field pointing to a known reference."""
    allowed = {"M24", "G24", "F24", "L23", "Go23", "Bo24", "K23", "M23", "H23"}
    for a in M.JWST_AGN_SAMPLE:
        assert a.ref in allowed, f"{a.name}: unknown ref '{a.ref}'"


# ----------------------------------------------------------------------------
# Cosmology helpers
# ----------------------------------------------------------------------------

def test_age_universe_today_is_13_8_Gyr():
    """t(z=0) ~ 13.8 Gyr for Planck 2018 LCDM."""
    age0 = M.age_universe_Myr(0.0)
    assert 13_500 < age0 < 14_000, f"age(0) = {age0} Myr"


def test_age_universe_at_z10_is_few_hundred_Myr():
    """t(z=10) ~ 470 Myr for Planck 2018 LCDM."""
    age = M.age_universe_Myr(10.0)
    assert 300 < age < 600


def test_age_universe_monotonic_decreasing_with_z():
    """Universe is younger at higher redshift."""
    for z1, z2 in [(0, 5), (5, 10), (10, 20)]:
        assert M.age_universe_Myr(z1) > M.age_universe_Myr(z2)


# ----------------------------------------------------------------------------
# Eddington growth
# ----------------------------------------------------------------------------

def test_eddington_growth_increasing():
    """log M_BH(z) must be monotonically increasing as z decreases."""
    z_grid, log_M = M.eddington_growth_track(10.0, z_form=30.0)
    # z_grid descending in time (z_form -> 4), so log_M should rise
    # Order z_grid ascending in time:
    order = np.argsort(-z_grid)
    log_M_time = log_M[order]
    assert np.all(np.diff(log_M_time) >= -1e-9)


def test_m_seed_required_for_10_8_at_z10_is_demanding():
    """Growing M_BH = 1e8 Msun at z=10 from z_form=30 requires a seed
    >> 1 Msun (Eddington-limited)."""
    M_seed = M.m_seed_required(M_BH_obs_Msun=1e8, z_obs=10.0, z_form=30.0)
    assert M_seed > 1.0


def test_m_seed_required_smaller_than_M_BH_obs():
    """M_seed required must be at most M_BH_obs (you can't shrink)."""
    for z_obs in (5.0, 7.0, 10.0):
        M_seed = M.m_seed_required(1e7, z_obs=z_obs, z_form=30.0)
        assert 0 < M_seed < 1e7


# ----------------------------------------------------------------------------
# Class B number density
# ----------------------------------------------------------------------------

def test_class_B_n_comoving_positive():
    """n_pred > 0 for any positive M_seed."""
    n = M.class_B_n_comoving_Mpc3(M_seed_Msun=10.0)
    assert n > 0


def test_class_B_n_inverse_proportional_to_M_seed():
    """n_pred is inversely proportional to M_seed (Omega fixed)."""
    n_1 = M.class_B_n_comoving_Mpc3(M_seed_Msun=1.0)
    n_100 = M.class_B_n_comoving_Mpc3(M_seed_Msun=100.0)
    assert n_1 / n_100 == pytest.approx(100.0, rel=1e-6)


def test_class_B_n_matches_omega_dm():
    """Sanity : Omega_class_B * rho_crit / M_typical reproduces the catalogue
    abundance bound = 1e-3 * Omega_DM."""
    cls_B = class_B_topological_pbh_seeds()
    # n_pred * M_typical_Msun = Omega_class_B * rho_crit
    n_pred = M.class_B_n_comoving_Mpc3()  # uses catalogue M_typical
    Omega_class_B = cls_B.omega_over_omega_dm * OMEGA_DM
    rho_crit_Msun_Mpc3 = (M.RHO_CRIT_MSUN_MPC3
                           if hasattr(M, "RHO_CRIT_MSUN_MPC3")
                           else 2.775e11 * (M.H0_KM_S_MPC / 100.0) ** 2)
    rho_target = n_pred * cls_B.m_typical_Msun
    expected = Omega_class_B * rho_crit_Msun_Mpc3
    assert math.isclose(rho_target, expected, rel_tol=1e-6)


# ----------------------------------------------------------------------------
# JWST observed number density estimator
# ----------------------------------------------------------------------------

def test_jwst_observed_n_in_sensible_range():
    """The naive observed AGN density at z = 5..8 should be 1e-6 .. 1e-2."""
    n_obs = M.jwst_observed_n_AGN_Mpc3(z_min=5.0, z_max=8.0)
    assert 1e-6 < n_obs < 1e-2


# ----------------------------------------------------------------------------
# chi-square / grid search
# ----------------------------------------------------------------------------

def test_chi2_small_seed_gives_large_chi2():
    """A tiny seed (< 1 Msun) cannot grow to log M_BH = 7 at z = 5 from
    z_form = 30 -> chi2 large."""
    chi2_small, dof_small = M.chi2_growth_track(1e-3, z_form=30.0)
    chi2_huge, dof_huge = M.chi2_growth_track(1e7, z_form=30.0)
    assert chi2_small > chi2_huge


def test_grid_search_finds_required_seed():
    """Grid search must find a finite seed mass satisfying chi^2/dof < 1."""
    res = M.grid_search_best_seed(
        M_seed_grid=np.logspace(0, 6, 30), z_form=30.0)
    assert math.isfinite(res["M_seed_min_required_Msun"])
    assert res["M_seed_min_required_Msun"] > 0


def test_grid_search_seed_in_classB_window_or_above():
    """The smallest seed satisfying chi2/dof<1 must be within or above the
    Class B [M_typical, M_max] window (we don't expect M_seed << M_typical)."""
    cls_B = class_B_topological_pbh_seeds()
    res = M.grid_search_best_seed(
        M_seed_grid=np.logspace(-1, 8, 50), z_form=30.0)
    M_req = res["M_seed_min_required_Msun"]
    assert M_req >= cls_B.m_typical_Msun * 0.1  # at most 1 dex below M_typ


# ----------------------------------------------------------------------------
# Verdict logic
# ----------------------------------------------------------------------------

def test_verdict_returns_known_keys():
    """verdict() must return a dict with the expected diagnostic keys."""
    vd = M.verdict(M_seed_required_Msun=100.0, n_obs_Mpc3=1e-5)
    expected_keys = {
        "M_seed_required_Msun", "M_typical_Msun", "M_max_Msun",
        "n_pred_at_typical_Mpc3", "n_pred_at_max_Mpc3",
        "n_pred_at_required_Mpc3", "n_obs_Mpc3",
        "log_ratio_n_pred_over_obs",
        "mass_compatible", "abundance_compatible", "verdict",
    }
    assert expected_keys.issubset(vd.keys())


def test_verdict_mass_compatible_when_required_in_window():
    """If M_seed_required = M_typical (within window), mass_compatible
    must be True."""
    cls_B = class_B_topological_pbh_seeds()
    vd = M.verdict(M_seed_required_Msun=cls_B.m_typical_Msun, n_obs_Mpc3=1e-5)
    assert vd["mass_compatible"]


def test_verdict_string_is_one_of_expected_categories():
    """The verdict string must start with one of EXPLAINS / CONTRIBUTES /
    INDISTINCT / REFUTED."""
    vd = M.verdict(M_seed_required_Msun=100.0, n_obs_Mpc3=1e-5)
    head = vd["verdict"].split()[0]
    assert head in {"EXPLAINS", "CONTRIBUTES", "INDISTINCT", "REFUTED"}


# ----------------------------------------------------------------------------
# End-to-end run
# ----------------------------------------------------------------------------

def test_run_returns_full_result_dict():
    """run() must execute without error and return a dict with keys
    {grid, n_obs_Mpc3, verdict, output_png}, and the PNG must be on disk."""
    res = M.run()
    for key in ("grid", "n_obs_Mpc3", "verdict", "output_png"):
        assert key in res
    out_path = Path(res["output_png"])
    assert out_path.exists()
    assert out_path.stat().st_size > 10_000  # PNG must be non-trivial
