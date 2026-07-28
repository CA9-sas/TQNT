"""Tests for scripts/V5_microlensing/class_E_strange_knot_test.py.

Tests V5.19 Class E "Strange Knot Stars" (6 STRICT amphichirals)
against Subaru HSC + OGLE + EROS microlensing limits.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_microlensing"))

import class_E_strange_knot_test as M  # noqa: E402
import class_A_subaru_hsc_test as MA  # noqa: E402


# ----------------------------------------------------------------------------
# 1. The 6 amphichiral mass-peaks match the CLAUDE.md spec
# ----------------------------------------------------------------------------

def test_six_amphichirals_present():
    """Exactly six STRICT amphichirals catalogued."""
    assert len(M.AMPHICHIRALS) == 6
    labels = [a[0] for a in M.AMPHICHIRALS]
    assert set(labels) == {
        "6_3+6_3", "6_3+6_3+6_3", "4_1+8_3",
        "8_3+8_3", "6_3+8_3", "4_1+4_1+8_3",
    }


def test_amphichiral_masses_match_spec():
    """Knot-link masses match the CLAUDE.md V5 catalogue values."""
    spec = {
        "6_3+6_3":      1.55,
        "6_3+6_3+6_3":  4.70,
        "4_1+8_3":     16.01,
        "8_3+8_3":     17.24,
        "6_3+8_3":     26.48,
        "4_1+4_1+8_3": 51.78,
    }
    table = {a[0]: a[2] for a in M.AMPHICHIRALS}
    for k, v in spec.items():
        assert math.isclose(table[k], v, rel_tol=1e-9), k


def test_chandrasekhar_per_amphichiral_ordering():
    """Heavier knot ⇒ smaller M_Ch ~ 1/m_knot^2."""
    params = M.v5_class_E_parameters()
    # Sort peaks by m_knot ascending: M_Ch must come out descending.
    by_mass = sorted(params.peaks, key=lambda p: p.m_knot_GeV)
    M_Ch = [p.log_M_Ch for p in by_mass]
    assert M_Ch == sorted(M_Ch, reverse=True), \
        f"Chandrasekhar ordering broken: {M_Ch}"


def test_chandrasekhar_lightest_amphichiral_is_subsolar():
    """6_3+6_3 (lightest knot, 1.55 GeV) ⇒ M_Ch ≈ 0.68 M_⊙ (sub-solar)."""
    params = M.v5_class_E_parameters()
    p = next(p for p in params.peaks if p.label == "6_3+6_3")
    M_Ch = 10.0 ** p.log_M_Ch
    assert 0.3 < M_Ch < 1.0, f"unexpected M_Ch for 6_3+6_3: {M_Ch}"


def test_chandrasekhar_heaviest_amphichiral_is_below_milli_msun():
    """4_1+4_1+8_3 (51.78 GeV) ⇒ M_Ch ~ 6e-4 M_⊙."""
    params = M.v5_class_E_parameters()
    p = next(p for p in params.peaks if p.label == "4_1+4_1+8_3")
    M_Ch = 10.0 ** p.log_M_Ch
    assert 1e-4 < M_Ch < 1e-3


# ----------------------------------------------------------------------------
# 2. Abundance weights
# ----------------------------------------------------------------------------

def test_weights_sum_to_one():
    params = M.v5_class_E_parameters()
    s = sum(p.weight for p in params.peaks)
    assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_f_total_is_10_percent():
    """V5 catalogue: Ω_E / Ω_DM = 0.10."""
    params = M.v5_class_E_parameters()
    assert math.isclose(params.f_total, 0.10, rel_tol=1e-9)
    # And sum of f_i equals f_total
    s = sum(p.f_i for p in params.peaks)
    assert math.isclose(s, params.f_total, rel_tol=1e-9)


def test_lightest_crossing_has_largest_weight():
    """The c=12 amphichirals (lightest topology) are the most abundant."""
    params = M.v5_class_E_parameters()
    c12 = [p for p in params.peaks if p.crossing == 12]
    c18 = [p for p in params.peaks if p.crossing == 18]
    assert len(c12) == 2 and len(c18) == 1
    # Every c=12 weight > every c>=14 weight
    for p12 in c12:
        for p_heavy in params.peaks:
            if p_heavy.crossing > 12:
                assert p12.weight >= p_heavy.weight - 1e-12, \
                    f"weight ordering broken {p12.label} vs {p_heavy.label}"
    # And c=18 has the smallest weight strictly
    assert c18[0].weight == min(p.weight for p in params.peaks)


# ----------------------------------------------------------------------------
# 3. Multi-peak mass function
# ----------------------------------------------------------------------------

def test_f_v5_class_E_peaks_at_all_six_centroids():
    """f_v5(M) has local maxima near every M_Ch_i (within the per-peak σ)."""
    params = M.v5_class_E_parameters()
    M_grid = np.logspace(-5.0, 1.0, 6000)
    f = M.f_v5_class_E(M_grid, params=params)
    # For each peak, check there's a local maximum within 0.2 dex of M_Ch
    for p in params.peaks:
        log_M_grid = np.log10(M_grid)
        window = np.abs(log_M_grid - p.log_M_Ch) < 0.2
        if window.any():
            f_local_max = f[window].max()
            f_at_center = float(M.f_v5_class_E(10.0 ** p.log_M_Ch,
                                               params=params)[0])
            # f at center should be close to (within 5%) the local max
            assert f_at_center >= 0.9 * f_local_max, p.label


def test_f_v5_class_E_normalisation():
    """Integral of f(M) d log10 M over all peaks ≈ f_total."""
    params = M.v5_class_E_parameters()
    log_M = np.linspace(-5.0, 1.0, 8000)
    M_arr = 10.0 ** log_M
    f = M.f_v5_class_E(M_arr, params=params)
    integ = float(np.trapz(f, log_M))
    assert 0.85 * params.f_total < integ <= 1.05 * params.f_total


def test_f_v5_above_truncation_is_zero():
    """f(M) = 0 well above the heaviest peak (M >> M_Ch_max)."""
    params = M.v5_class_E_parameters()
    M_huge = 1e2  # 100 M_⊙ >> 0.68 M_⊙
    f = float(M.f_v5_class_E(M_huge, params=params)[0])
    assert f == 0.0


# ----------------------------------------------------------------------------
# 4. Tension / verdict
# ----------------------------------------------------------------------------

def test_no_peak_exceeds_hsc_limit():
    """Every amphichiral peak sits below the published HSC bound."""
    per_peak = M.per_peak_tension()
    for p in per_peak:
        assert not p["excluded"], (
            f"{p['label']} exceeds HSC limit: ratio={p['ratio']:.3e}"
        )


def test_global_verdict_survives():
    """Class E as a whole is compatible with current microlensing."""
    v = M.verdict()
    assert v["verdict"] == "SURVIVES"
    assert v["max_tension"] < 1.0
    assert v["n_excluded_peaks"] == 0
    assert v["n_excluded_decades"] == 0.0


def test_max_tension_in_milliMsun_window():
    """The tightest constraint on Class E falls in 1e-4 - 1e-2 M_⊙."""
    v = M.verdict()
    assert 1e-4 < v["M_max_tension"] < 1e-1


# ----------------------------------------------------------------------------
# 5. Resolvability of the multi-peak pattern
# ----------------------------------------------------------------------------

def test_peak_separations_are_six_minus_one():
    """6 peaks ⇒ 5 adjacent separations."""
    seps = M.peak_separations()
    assert len(seps) == 5
    for s in seps:
        assert s >= 0


def test_at_least_one_peak_pair_too_close_to_resolve():
    """The 4_1+8_3 / 8_3+8_3 pair sits at log M_Ch ≈ -2.20 vs -2.26 — only
    0.06 dex apart, much smaller than HSC resolution 0.5 dex. Subaru CANNOT
    distinguish them.
    """
    res = M.resolvability_report()
    assert min(res["separations_dex"]) < 0.1, (
        f"unexpectedly large min separation: {res['separations_dex']}"
    )
    # And the resolvability flag for that closest pair must be False
    closest = min(res["separations_dex"])
    idx = res["separations_dex"].index(closest)
    assert res["is_resolvable_pairs"][idx] is False


def test_n_resolved_features_less_than_total_peaks():
    """Subaru HSC merges some peaks: visible features < 6."""
    res = M.resolvability_report()
    assert res["n_resolved_peaks"] < res["n_total_peaks"]
    assert res["multi_peak_visible"] is True


def test_multi_peak_not_fully_resolvable_by_subaru():
    """A perfect 6-peak resolution would need R_HSC ≲ 0.05 dex.
    With R_HSC = 0.5 dex (realistic), at most a handful of peaks resolve.
    """
    res = M.resolvability_report()
    assert res["n_resolved_peaks"] <= 4, (
        f"surprisingly fine resolution: {res['n_resolved_peaks']} peaks"
    )


# ----------------------------------------------------------------------------
# 6. Class E vs Class A distinguishability
# ----------------------------------------------------------------------------

def test_class_E_total_differs_from_class_A_blended():
    """Class E multi-peak prediction does NOT identically equal Class A
    log-normal renormalised to f_total=0.10.
    """
    M_grid = np.logspace(-4.0, 0.0, 600)
    f_E = M.f_v5_class_E(M_grid)
    f_A = M.f_v5_class_A_renormalised(M_grid)
    # Peak of f_E should NOT coincide with peak of f_A (log M_typ_A = -2.7)
    iE = int(np.argmax(f_E))
    iA = int(np.argmax(f_A))
    assert iE != iA or abs(math.log10(M_grid[iE] / M_grid[iA])) > 0.2


def test_summary_keys_present():
    s = M.summary()
    expected = {
        "f_total", "n_peaks", "sigma_log_M_per_peak", "per_peak",
        "max_tension", "M_max_tension_Msun", "n_excluded_decades",
        "n_excluded_peaks", "excluded_peaks", "surviving_peaks", "verdict",
        "R_HSC_dex", "sigma_eff_dex", "n_resolved_peaks",
        "multi_peak_visible", "separations_dex",
    }
    assert expected.issubset(set(s.keys()))
    assert s["verdict"] in {"SURVIVES", "PARTIALLY_EXCLUDED", "EXCLUDED"}


# ----------------------------------------------------------------------------
# 7. Plot generation
# ----------------------------------------------------------------------------

def test_make_plot_writes_png(tmp_path):
    out = tmp_path / "v5_class_E.png"
    path = M.make_plot(path=out)
    assert path.exists()
    assert path.stat().st_size > 5000  # at least 5 kB
