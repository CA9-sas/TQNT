"""Tests for scripts/V5_Gaia/class_C_hybrid_stars_test.py.

Tests the V5.19 Class C "Knot-Baryon Hybrid Stars" predictions versus the
Gaia DR3 (2023) and projected DR4 (2026) population statistics.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "V5_Gaia"))

import class_C_hybrid_stars_test as M  # noqa: E402

from theory.v5_compact_objects_catalog import (  # noqa: E402
    class_C_hybrid_stars,
    OMEGA_B,
    OMEGA_DM,
)


# ----------------------------------------------------------------------------
# 1. Gaia population stats sanity
# ----------------------------------------------------------------------------

def test_gaia_dr3_realistic_population():
    """Gaia DR3 advertises ~ 1.8 billion sources."""
    assert 1.0e9 < M.GAIA_DR3.N_stars < 3.0e9
    assert M.GAIA_DR3.sigma_BP_RP_mag > 0
    assert M.GAIA_DR3.sigma_RV_kms == pytest.approx(1.0, rel=0.5)


def test_gaia_dr4_better_than_dr3():
    """All DR4 precisions strictly improve on DR3."""
    assert M.GAIA_DR4.sigma_BP_RP_mag < M.GAIA_DR3.sigma_BP_RP_mag
    assert M.GAIA_DR4.sigma_RV_kms < M.GAIA_DR3.sigma_RV_kms
    assert M.GAIA_DR4.sigma_parallax_mas < M.GAIA_DR3.sigma_parallax_mas
    assert M.GAIA_DR4.sigma_Teff_K < M.GAIA_DR3.sigma_Teff_K


# ----------------------------------------------------------------------------
# 2. V5 Class C parameters match the catalogue
# ----------------------------------------------------------------------------

def test_class_C_number_fraction_from_catalogue():
    """f_C = Omega_C / Omega_b = 1e-3 per V5.19 catalogue."""
    f = M.class_C_number_fraction()
    cls_C = class_C_hybrid_stars()
    assert math.isclose(f, cls_C.omega_over_omega_b, rel_tol=1e-9)
    assert math.isclose(f, 1e-3, rel_tol=1e-9)


def test_predicted_N_class_C_dr3_in_millions():
    """N_C in DR3 should be ~ 1.5-2 million (1e-3 of 1.8 G)."""
    N = M.predicted_N_class_C(M.GAIA_DR3)
    assert 1.0e6 < N < 3.0e6


def test_predicted_N_class_C_dr4_larger_than_dr3():
    """DR4 catalogues more sources, so more Class C."""
    assert M.predicted_N_class_C(M.GAIA_DR4) > M.predicted_N_class_C(M.GAIA_DR3)


# ----------------------------------------------------------------------------
# 3. Class C physical signature: per-star magnitude
# ----------------------------------------------------------------------------

def test_HR_color_offset_is_negative_and_tiny():
    """A more compact star is BLUER (negative BP-RP shift); magnitude ~ 1e-5."""
    d = M.HR_offset_BP_RP_mag(M_core_Msun=1e-4, M_env_Msun=1.0)
    assert d < 0  # bluer
    assert 1e-6 < abs(d) < 1e-3   # very small, below per-star DR3 sigma


def test_HR_color_scales_linearly_with_M_core():
    """At small M_core/M_env, |Delta(BP-RP)| ~ M_core/M_env."""
    a = abs(M.HR_offset_BP_RP_mag(M_core_Msun=1e-4))
    b = abs(M.HR_offset_BP_RP_mag(M_core_Msun=1e-3))
    # b should be ~ 10x larger
    assert 5.0 < b / a < 20.0


def test_RV_signal_is_subkms_for_1em4_core():
    """RV asymmetry for M_core = 1e-4 Msun is below 1 km/s."""
    v = M.spectral_velocity_shift_kms(M_core_Msun=1e-4, M_env_Msun=1.0)
    # v_circ_solar ~ 437 km/s ; v = 1e-4 * 437 ~ 0.04 km/s
    assert 0.01 < v < 0.2


def test_central_density_local_boost_huge_but_localized():
    """The local pearl boost is enormous (~1e10) but limited to ~km^3."""
    boost = M.central_density_boost_factor(
        M_core_Msun=1e-4, R_env_Rsun=1.0,
        R_core_AU=1e-7,
    )
    assert boost > 1e8


def test_bulk_density_offset_is_M_core_over_M_env():
    """Bulk density offset = M_core / M_env (no contrast amplification)."""
    off = M.bulk_density_offset(M_core_Msun=1e-4, M_env_Msun=1.0)
    assert math.isclose(off, 1e-4, rel_tol=1e-9)


def test_astrometric_wobble_zero_when_centered():
    """Pearl AT the centre produces NO photocentre wobble."""
    w = M.astrometric_wobble_mas(M_core_Msun=1e-4, a_au=0.0)
    assert w == 0.0


# ----------------------------------------------------------------------------
# 4. Detectability ratios
# ----------------------------------------------------------------------------

def test_per_star_color_below_DR3_noise():
    """Per-star colour signal < DR3 noise (ratio < 1)."""
    r = M.detectability_ratio_color(M.GAIA_DR3)
    assert r < 1.0, f"per-star colour detectable in DR3 (ratio={r})"


def test_per_star_RV_below_DR3_noise():
    """Per-star RV signal < DR3 RVS noise (ratio < 1)."""
    r = M.detectability_ratio_RV(M.GAIA_DR3)
    assert r < 1.0, f"per-star RV detectable in DR3 (ratio={r})"


def test_ensemble_significance_DR4_gt_DR3():
    """DR4 ensemble significance must exceed DR3 (more sources + smaller sigma)."""
    s3 = M.ensemble_detection_significance(M.GAIA_DR3)
    s4 = M.ensemble_detection_significance(M.GAIA_DR4)
    assert s4 > s3
    # Quantitative: with N x ~ 1.1 and sigma / 5, expect ~ sqrt(1.1) * 5
    assert s4 / s3 > 3.0


def test_ensemble_significance_DR4_above_5sigma():
    """V5 Class C predicts a > 5 sigma stacked anomaly in DR4 colour."""
    s = M.ensemble_detection_significance(M.GAIA_DR4)
    assert s > 5.0


# ----------------------------------------------------------------------------
# 5. Anomaly comparison
# ----------------------------------------------------------------------------

def test_anomalies_table_well_formed():
    """The known-anomaly table is non-empty and consistent."""
    anoms = M.anomaly_explained_by_class_C(M.GAIA_DR3)
    assert len(anoms) >= 3
    for name, d in anoms.items():
        assert d["f_anomaly"] > 0
        assert d["f_class_C"] > 0
        assert math.isfinite(d["ratio_anomaly_over_classC"])


def test_astrometric_BH_rate_far_below_classC():
    """Gaia astrometric BH detection rate is 1e-7, far below f_C = 1e-3.

    Class C is NOT identified by Gaia astrometric BHs (which require
    a non-zero offset pearl, not a centered pearl).
    """
    anoms = M.anomaly_explained_by_class_C(M.GAIA_DR3)
    bh_key = next(k for k in anoms if "astrometric BHs" in k)
    f_bh = anoms[bh_key]["f_anomaly"]
    f_C = anoms[bh_key]["f_class_C"]
    assert f_bh < f_C * 1e-2  # at least 100x rarer than f_C


# ----------------------------------------------------------------------------
# 6. Verdict
# ----------------------------------------------------------------------------

def test_verdict_is_well_formed():
    """verdict() returns the expected keys and a sensible label."""
    v = M.verdict()
    keys = {
        "M_core_Msun", "f_C", "N_C_DR3", "N_C_DR4",
        "signal_color_mag", "signal_RV_kms",
        "ratio_color_DR3", "ratio_color_DR4",
        "ratio_RV_DR3", "ratio_RV_DR4",
        "ensemble_significance_DR3_sigma",
        "ensemble_significance_DR4_sigma",
        "verdict",
    }
    assert keys.issubset(set(v.keys()))
    assert any(label in v["verdict"]
               for label in ("SURVIVES", "MARGINAL", "EXCLUDED"))


def test_verdict_marginal_or_survives_not_excluded():
    """The current V5.19 Class C is NOT excluded by Gaia DR3 (per-star noise
    too high). It should land on MARGINAL or SURVIVES."""
    v = M.verdict()
    assert "EXCLUDED" not in v["verdict"]


# ----------------------------------------------------------------------------
# 7. Summary smoke test
# ----------------------------------------------------------------------------

def test_summary_has_expected_keys():
    """summary() exposes the quantities the CLI / report expects."""
    s = M.summary()
    keys = {
        "f_C", "N_C_DR3", "N_C_DR4", "M_core_Msun",
        "signal_color_mag", "signal_RV_kms",
        "bulk_density_offset", "central_boost_local",
        "ratio_color_DR3", "ratio_color_DR4",
        "ratio_RV_DR3", "ratio_RV_DR4",
        "ensemble_significance_DR3_sigma",
        "ensemble_significance_DR4_sigma",
        "verdict",
    }
    assert keys.issubset(set(s.keys()))


# ----------------------------------------------------------------------------
# 8. Plot generation
# ----------------------------------------------------------------------------

def test_make_plot_writes_png(tmp_path):
    """make_plot writes a non-empty PNG file."""
    out = tmp_path / "v5_class_C.png"
    path = M.make_plot(path=out)
    assert path.exists()
    assert path.stat().st_size > 5000  # at least 5 kB
