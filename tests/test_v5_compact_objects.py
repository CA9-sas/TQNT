"""Tests for theory/v5_compact_objects_catalog.py (V5.19)."""
from __future__ import annotations

import math

import pytest

from theory.v5_compact_objects_catalog import (
    CompactObjectClass,
    MPL_KG, MSUN_KG, OMEGA_B, OMEGA_DM, RHO_CRIT_MSUN_MPC3,
    DM_STRICT_AMPHI_GEV,
    chandrasekhar_mass_topological,
    pbh_seed_mass_at_sound_horizon,
    heavy_proton_chandrasekhar,
    class_A_dm_pearls,
    class_B_topological_pbh_seeds,
    class_C_hybrid_stars,
    class_D_heavy_proton_plasma,
    class_E_strange_knot_stars,
    class_F_unknot_defects,
    build_catalog,
    jwst_smf_contribution,
    total_jwst_contribution,
    residual_gap_analysis,
    LOG_PHI_OBS_CORRECTED,
    LOG_PHI_LCDM_NOMINAL,
    summary,
)


# ----------------------------------------------------------------------------
# Constants sanity
# ----------------------------------------------------------------------------

def test_mpl_value():
    """Planck mass should be ~ 2.18e-8 kg."""
    assert 2.0e-8 < MPL_KG < 2.4e-8


def test_dm_strict_amphi_count():
    """V5 catalogue lists exactly 6 STRICT amphichirals."""
    assert len(DM_STRICT_AMPHI_GEV) == 6


def test_dm_strict_amphi_range():
    """Lightest DM strict knot at 1.55 GeV, heaviest at ~52 GeV."""
    assert min(DM_STRICT_AMPHI_GEV) == pytest.approx(1.55, abs=0.05)
    assert max(DM_STRICT_AMPHI_GEV) == pytest.approx(51.78, abs=0.05)


# ----------------------------------------------------------------------------
# Chandrasekhar mass formula
# ----------------------------------------------------------------------------

def test_chandrasekhar_topological_scaling():
    """M_Ch ~ 1/m^2: doubling m_chi should give factor 4 decrease."""
    M1 = chandrasekhar_mass_topological(1.0)
    M2 = chandrasekhar_mass_topological(2.0)
    assert M1 / M2 == pytest.approx(4.0, rel=1e-6)


def test_chandrasekhar_at_canonical_wimp():
    """At m_chi = 20 GeV: bare M_Pl^3/m^2 = 8e27 kg.

    The dm_pearls paper (eq 6) quotes 1.85e30 kg which includes the
    standard Landau prefactor ~230. We test the BARE formula here.
    """
    M_Ch = chandrasekhar_mass_topological(20.0)
    assert 1e27 < M_Ch < 1e29


def test_heavy_proton_chandrasekhar_in_solar_mass_range():
    """Heavy-proton Chandrasekhar ~ M_Pl^3/(3 GeV)^2 ~ 8e28 kg ~ 0.04 M_sun."""
    M = heavy_proton_chandrasekhar(3.1)
    assert 0.01 * MSUN_KG < M < 1.0 * MSUN_KG


# ----------------------------------------------------------------------------
# PBH seed at sound horizon
# ----------------------------------------------------------------------------

def test_pbh_seed_at_100MeV_solar_mass():
    """At T = 100 MeV (QCD epoch), horizon mass ~ 1 M_sun (canonical)."""
    M = pbh_seed_mass_at_sound_horizon(T_MeV=100.0)
    assert 0.1 * MSUN_KG < M < 1e3 * MSUN_KG


def test_pbh_seed_at_1MeV_larger():
    """At lower T, horizon mass grows (M_H ~ 1/T^2)."""
    M_hot = pbh_seed_mass_at_sound_horizon(T_MeV=100.0)
    M_cold = pbh_seed_mass_at_sound_horizon(T_MeV=1.0)
    assert M_cold > M_hot * 1e3


# ----------------------------------------------------------------------------
# Catalog: 6 classes, each well-formed
# ----------------------------------------------------------------------------

def test_catalog_has_six_classes():
    """Catalog must contain exactly 6 classes A..F."""
    cat = build_catalog()
    assert set(cat.keys()) == {"A", "B", "C", "D", "E", "F"}


def test_class_A_is_dark():
    """Class A (DM pearls): EM-invisible -> not JWST-observable."""
    c = class_A_dm_pearls()
    assert c.short_id == "A"
    assert c.jwst_observable is False
    assert "dark" in c.spectral_signature.lower()


def test_class_B_is_jwst_visible_as_agn():
    """Class B (PBH seeds) is JWST-visible (AGN-like)."""
    c = class_B_topological_pbh_seeds()
    assert c.short_id == "B"
    assert c.jwst_observable is True
    assert c.transient is False
    assert "AGN" in c.spectral_signature


def test_class_D_is_transient():
    """Class D (heavy-proton plasma) is transient (decays before z<10)."""
    c = class_D_heavy_proton_plasma()
    assert c.short_id == "D"
    assert c.transient is True
    assert c.jwst_observable is False


def test_class_E_is_dark_and_amphichiral():
    """Class E (strange knot stars) is EM-invisible (Q=0 exactly)."""
    c = class_E_strange_knot_stars()
    assert c.short_id == "E"
    assert c.jwst_observable is False


def test_class_F_intermediate_mass_bh():
    """Class F (unknot defects) is IMBH-class, JWST-visible."""
    c = class_F_unknot_defects()
    assert c.short_id == "F"
    assert c.jwst_observable is True
    assert "IMBH" in c.spectral_signature


# ----------------------------------------------------------------------------
# Mass scales: each class has plausible M_typical / M_max
# ----------------------------------------------------------------------------

def test_all_classes_have_positive_mass():
    """Every class must have M_typical > 0 and M_max >= M_typical."""
    for sid, c in build_catalog().items():
        assert c.m_typical_kg > 0, f"Class {sid} has M_typical = 0"
        assert c.m_max_kg >= c.m_typical_kg, f"Class {sid}: M_max < M_typical"


def test_class_B_seed_mass_in_pbh_window():
    """Class B PBH seeds: 1 M_sun .. 10^8 M_sun."""
    c = class_B_topological_pbh_seeds()
    assert 0.1 < c.m_typical_Msun < 1e3
    assert 1.0 < c.m_max_Msun < 1e10


def test_dm_pearls_subgalactic():
    """Class A pearls: 10^-12 .. 1 M_sun (sub-galactic, MACHO scale)."""
    c = class_A_dm_pearls()
    assert c.m_typical_Msun < 1.0


# ----------------------------------------------------------------------------
# Abundance constraints: each class fraction <= 1
# ----------------------------------------------------------------------------

def test_abundances_below_total():
    """Each class abundance (Omega/Omega_X) must be in [0,1]."""
    for sid, c in build_catalog().items():
        assert 0 <= c.omega_over_omega_b <= 1.0
        assert 0 <= c.omega_over_omega_dm <= 1.0


def test_class_B_pbh_below_microlensing_bound():
    """PBH f_DM in 10..1e9 M_sun window must be <= 1e-2 (OGLE bound)."""
    c = class_B_topological_pbh_seeds()
    assert c.omega_over_omega_dm <= 1e-2


# ----------------------------------------------------------------------------
# JWST contribution: dark/transient classes contribute nothing
# ----------------------------------------------------------------------------

def test_dark_classes_have_zero_contribution():
    """Classes A, D, E must give zero JWST SMF contribution."""
    for sid in ("A", "D", "E"):
        cls = build_catalog()[sid]
        contrib = jwst_smf_contribution(cls)
        assert contrib["phi_apparent_dex"] == 0.0


def test_visible_classes_have_nonzero_contribution():
    """Classes B, C, F (jwst_observable=True) must give phi > 0 at peak."""
    for sid in ("B", "C", "F"):
        cls = build_catalog()[sid]
        contrib = jwst_smf_contribution(cls)
        assert contrib["phi_apparent_dex"] > 0
        assert contrib["jwst_relevant"]


# ----------------------------------------------------------------------------
# Residual gap analysis
# ----------------------------------------------------------------------------

def test_residual_gap_positive():
    """V5 compact-objects do NOT fully close the +5 dex residual gap.

    This is the HONEST verdict: V5 compact objects contribute very little
    at log M_* = 10 because their M_apparent peaks at 1 or 10^5-10^6 M_sun,
    not at the JWST target of 10^10 M_sun.
    """
    rg = residual_gap_analysis(z=8.0, log_M_star=10.0)
    assert rg["residual_gap_dex"] > 1.0   # gap remains > 1 dex
    assert not rg["V5_compact_closes_gap"]


def test_residual_gap_target_consistent():
    """Sanity: LOG_PHI_OBS_CORRECTED - LOG_PHI_LCDM_NOMINAL ~ +5.0 dex."""
    gap = LOG_PHI_OBS_CORRECTED - LOG_PHI_LCDM_NOMINAL
    assert gap == pytest.approx(5.0, abs=0.5)


# ----------------------------------------------------------------------------
# Summary function: must contain all expected keys
# ----------------------------------------------------------------------------

def test_summary_contains_six_classes():
    """summary()['classes'] must have all 6 short_id keys."""
    s = summary()
    assert set(s["classes"].keys()) == {"A", "B", "C", "D", "E", "F"}


def test_summary_residual_gap_present():
    """summary() must include the residual gap analysis at z=8."""
    s = summary()
    assert "residual_gap_z8" in s
    assert "residual_gap_dex" in s["residual_gap_z8"]
