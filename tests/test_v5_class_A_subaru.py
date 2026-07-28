"""Tests for scripts/V5_microlensing/class_A_subaru_hsc_test.py.

Tests V5.19 Class A "DM Pearls" against Subaru HSC microlensing limits
(Niikura+ 2019, Smyth+ 2020, OGLE/EROS).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "V5_microlensing"))

import class_A_subaru_hsc_test as M  # noqa: E402

from theory.v5_compact_objects_catalog import class_A_dm_pearls  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Hardcoded HSC limit table sanity
# ----------------------------------------------------------------------------

def test_hsc_limits_table_well_formed():
    """Limit table is non-empty, sorted in log10(M), and bounded in (0, 1]."""
    table = M.HSC_LIMITS
    assert len(table) >= 10, "expected at least 10 hardcoded data points"
    log_M = [r[0] for r in table]
    assert log_M == sorted(log_M), "table must be sorted by log10(M)"
    for _, f, _ in table:
        assert 0 < f <= 1.0


def test_hsc_limit_at_2em3_msun_close_to_0p3():
    """At M = 2e-3 Msun (Class A typical), HSC limit ~ 0.3 (Smyth+ 2020)."""
    f = float(M.f_hsc_limit(2.0e-3)[0])
    # The interpolation in log10 should land near 0.3 (allow 0.2..0.5).
    assert 0.2 < f < 0.5, f"unexpected limit at 2e-3 Msun: {f}"


def test_hsc_limit_best_constraint_near_1em8_msun():
    """Tightest Subaru HSC bound (~3%) near M ~ 1e-8 Msun (Niikura+ 2019)."""
    f = float(M.f_hsc_limit(1.0e-8)[0])
    assert 0.01 < f < 0.1, f"expected ~3% limit, got {f}"


def test_hsc_limit_saturates_at_macho_scale():
    """At M ~ 0.1-1 Msun (MACHO), the bound saturates at ~1 (no constraint)."""
    f_high = float(M.f_hsc_limit(1.0)[0])
    assert f_high >= 0.5


# ----------------------------------------------------------------------------
# 2. V5 Class A parameters match the catalogue
# ----------------------------------------------------------------------------

def test_v5_class_A_parameters_from_catalogue():
    """The Class A parameters used in the test come from the V5 catalogue."""
    params = M.v5_class_A_parameters()
    cls_A = class_A_dm_pearls()
    assert math.isclose(params.f_total, cls_A.omega_over_omega_dm)
    assert math.isclose(
        params.log_M_typical, math.log10(cls_A.m_typical_Msun)
    )
    assert math.isclose(params.M_max_Msun, cls_A.m_max_Msun)
    # M_typical should be ~ 2e-3 Msun per spec
    assert 1e-3 < cls_A.m_typical_Msun < 5e-3


def test_v5_f_total_is_5_percent():
    """V5.19 Class A predicts Omega_A / Omega_DM = 0.05."""
    params = M.v5_class_A_parameters()
    assert math.isclose(params.f_total, 0.05, rel_tol=1e-6)


# ----------------------------------------------------------------------------
# 3. V5 prediction shape
# ----------------------------------------------------------------------------

def test_v5_f_peak_near_M_typical():
    """The log-normal V5 f(M) peaks near M_typical = 2e-3 Msun."""
    params = M.v5_class_A_parameters()
    M_grid = np.logspace(-11, 0, 2200)
    f_v5 = M.f_v5_class_A(M_grid, params=params)
    M_peak = float(M_grid[int(np.argmax(f_v5))])
    M_typ = 10.0 ** params.log_M_typical
    # peak should be within 1 dex of M_typical
    assert abs(math.log10(M_peak / M_typ)) < 0.5


def test_v5_f_truncated_above_M_max():
    """V5 f(M) is exactly zero above the topological Chandrasekhar cutoff."""
    params = M.v5_class_A_parameters()
    f_above = M.f_v5_class_A(2.0 * params.M_max_Msun, params=params)
    assert float(f_above[0]) == 0.0


def test_v5_f_integral_consistent_with_total():
    """Integrating dN/dlog10M over the log-normal recovers f_total."""
    params = M.v5_class_A_parameters()
    # Use a fine grid spanning many sigma around M_typical
    log_M = np.linspace(
        params.log_M_typical - 6 * params.sigma_log_M,
        params.log_M_typical + 6 * params.sigma_log_M,
        4000,
    )
    M_grid = 10.0 ** log_M
    f = M.f_v5_class_A(M_grid, params=params)
    # trapz integration in log10(M)
    dlog = log_M[1] - log_M[0]
    integ = float(np.sum(f) * dlog)
    # For a log-normal with truncation, integral may be slightly <= f_total
    assert 0.7 * params.f_total < integ <= 1.05 * params.f_total


# ----------------------------------------------------------------------------
# 4. Tension / verdict
# ----------------------------------------------------------------------------

def test_ratio_at_M_typical_below_1():
    """At M_typical the V5 prediction is below the HSC limit (SURVIVES)."""
    params = M.v5_class_A_parameters()
    M_typ = 10.0 ** params.log_M_typical
    r = float(M.tension_ratio(M_typ, params=params)[0])
    assert 0.0 < r < 1.0, f"unexpected tension at M_typical: {r}"
    # Spec expects ratio of order 0.1-0.3 (sub-limit by ~3-10x)
    assert r < 0.5


def test_global_verdict_survives():
    """The Class A prediction is not excluded anywhere in [1e-11, 1e0] Msun."""
    v = M.verdict()
    assert v["max_tension"] < 1.0
    assert v["n_excluded_decades"] == 0.0
    assert v["verdict"] == "SURVIVES"


def test_sensitivity_factor_reasonable():
    """Detecting Class A needs a sensitivity boost factor in [2, 20]."""
    s = M.sensitivity_factor_to_detect()
    assert 2.0 < s < 20.0, f"unexpected sensitivity factor: {s}"


# ----------------------------------------------------------------------------
# 5. Summary smoke test
# ----------------------------------------------------------------------------

def test_summary_has_expected_keys():
    """summary() exposes all the quantities the CLI / report expects."""
    s = M.summary()
    keys = {
        "M_typical_Msun", "f_total", "sigma_log_M", "M_max_Msun",
        "f_HSC_limit_at_M_typical", "f_V5_at_M_typical",
        "ratio_at_M_typical", "max_tension", "M_max_tension_Msun",
        "n_excluded_decades", "sensitivity_factor_to_detect", "verdict",
    }
    assert keys.issubset(set(s.keys()))
    assert s["verdict"] in {"SURVIVES", "MARGINAL", "EXCLUDED"}


# ----------------------------------------------------------------------------
# 6. Plot generation
# ----------------------------------------------------------------------------

def test_make_plot_writes_png(tmp_path):
    """make_plot writes a non-empty PNG file."""
    out = tmp_path / "v5_class_A.png"
    path = M.make_plot(path=out)
    assert path.exists()
    assert path.stat().st_size > 5000  # at least 5 kB, real plot
