"""Tests for V5.20 Fermi-LAT diffuse gamma test
(scripts/V5_Fermi_LAT/v5p20_gamma_diffuse_test.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))

from v5p20_gamma_diffuse_test import (
    E_CASCADE_GEV,
    N_PHOTONS_PER_CASCADE,
    cascade_spectrum_dN_dE,
    cascade_photons_total,
    n_events_per_volume_per_time,
    intensity_diffuse_v5p20,
    edgrb_fermi,
    gce_hooper,
    ratio_v5_fermi,
    verdict_fermi_test,
    gce_comparison,
    make_plot,
)


# ---------------------------------------------------------------------------
# 1. Spectre cascade : forme et normalisation
# ---------------------------------------------------------------------------

def test_cascade_spectrum_normalized():
    """Le spectre photons par cascade integre a 1 sur [1e-3, 1e3] GeV."""
    norm = cascade_photons_total(E_min_GeV=1e-3, E_max_GeV=1e3, n_pts=10000)
    # Le pic log-normal et le continuum sont tous deux normalises ; somme
    # ponderee F_PEAK + F_CONT = 1.
    assert 0.95 < norm < 1.05


def test_cascade_spectrum_peak_at_1p5_GeV():
    """Le pic d'energie E^2 dN/dE est proche de 1.5 GeV (cascade caracteristique).

    Le continuum E^-2 domine dN/dE a basse energie, mais E^2 dN/dE
    reveille le pic log-normal centre a E_CASCADE_GEV.
    """
    E = np.logspace(-1, 1, 2000)
    spec = cascade_spectrum_dN_dE(E)
    E2_spec = E ** 2 * spec
    E_peak = E[np.argmax(E2_spec)]
    # Pic E^2 dN/dE proche de E_CASCADE_GEV = 1.5 GeV
    # (asymetrie continuum E^-2 cutoff a 10 GeV deplace legerement vers le haut)
    assert 0.8 < E_peak < 8.0


def test_cascade_spectrum_positive_and_finite():
    E = np.logspace(-2, 3, 500)
    spec = cascade_spectrum_dN_dE(E)
    assert np.all(np.isfinite(spec))
    assert np.all(spec >= 0)


def test_cascade_spectrum_zero_outside_continuum():
    """Hors [E_MIN_cont, E_MAX_cont] le continuum E^-2 s'eteint ;
    seul reste le pic log-normal qui meurt loin de 1.5 GeV."""
    E_far = np.array([1e-6, 1e6])
    spec = cascade_spectrum_dN_dE(E_far)
    # Tres faible (log-normal evanescent loin du pic)
    assert np.all(spec < 1e-6)


# ---------------------------------------------------------------------------
# 2. EDGRB Fermi : valeurs de reference Ackermann+ 2015
# ---------------------------------------------------------------------------

def test_edgrb_at_100MeV_matches_normalisation():
    """A E=100 MeV (0.1 GeV) sans cutoff, dI/dE = I_0 * 1e3 (MeV->GeV)."""
    E = np.array([0.1])
    flux = edgrb_fermi(E, with_cutoff=False)
    expected = 1.12e-7 * 1e3   # cm^-2 s^-1 sr^-1 GeV^-1
    assert abs(flux[0] / expected - 1.0) < 1e-6


def test_edgrb_power_law_slope():
    """Sans cutoff, log d(I)/dlog(E) = -alpha = -2.32."""
    E = np.array([1.0, 10.0])
    flux = edgrb_fermi(E, with_cutoff=False)
    slope = math.log(flux[1] / flux[0]) / math.log(E[1] / E[0])
    assert abs(slope + 2.32) < 0.01


def test_edgrb_cutoff_reduces_flux_at_high_E():
    """Cutoff diminue le flux au-dessus de E_cut ~ 279 GeV."""
    E = np.array([500.0])
    flux_no = edgrb_fermi(E, with_cutoff=False)[0]
    flux_yes = edgrb_fermi(E, with_cutoff=True)[0]
    assert flux_yes < flux_no
    assert flux_yes / flux_no < math.exp(-500.0 / 279.0) * 1.01


# ---------------------------------------------------------------------------
# 3. Intensite V5.20 : ordre de grandeur
# ---------------------------------------------------------------------------

def test_n_events_per_volume_delta_c_10():
    """Pour Delta_c=10 : ~ 1.9e92 events sur V_Hubble * t_Hubble."""
    rate = n_events_per_volume_per_time(delta_c=10)
    # rate * V_H * t_H = N_total_universe ~ 1.9e92
    from theory.dm_pearl_internal import V_HUBBLE_M3, T_HUBBLE_S
    N_tot = rate * V_HUBBLE_M3 * T_HUBBLE_S
    assert 1e91 < N_tot < 1e93


def test_intensity_increases_when_dc_decreases():
    """Suppression Vassiliev : Delta_c bas -> intensite haute."""
    E = np.array([1.0])
    I_10 = intensity_diffuse_v5p20(E, delta_c=10)[0]
    I_30 = intensity_diffuse_v5p20(E, delta_c=30)[0]
    I_50 = intensity_diffuse_v5p20(E, delta_c=50)[0]
    assert I_10 > I_30 > I_50


def test_intensity_finite_and_positive():
    E = np.logspace(-1, 2, 100)
    for dc in [10, 30, 50]:
        I = intensity_diffuse_v5p20(E, delta_c=dc)
        assert np.all(np.isfinite(I))
        assert np.all(I >= 0)


# ---------------------------------------------------------------------------
# 4. Verdict : V5.20 dc=10 EXCLU, dc>=22 compatible
# ---------------------------------------------------------------------------

def test_verdict_dc10_strictly_excluded():
    """Delta_c=10 : V5.20 sur-predit EDGRB par > 1e20 ordres de magnitude."""
    v = verdict_fermi_test(delta_c=10)
    assert v["status"] in ("STRICTLY EXCLUDED", "EXCLUDED")
    assert v["ratio_integrated"] > 1e10


def test_verdict_dc_excluded_above_around_22():
    """Le Delta_c critique d'exclusion est dans [18, 28]."""
    v = verdict_fermi_test(delta_c=10)
    assert v["delta_c_excluded_above"] is not None
    assert 18 <= v["delta_c_excluded_above"] <= 28


def test_verdict_dc50_sub_dominant():
    """Pour Delta_c=50 : V5.20 sous-domine EDGRB par >> ordres de magnitude."""
    v = verdict_fermi_test(delta_c=50)
    assert v["status"] == "SUB-DOMINANT"
    assert v["ratio_integrated"] < 1e-3


# ---------------------------------------------------------------------------
# 5. GCE comparison
# ---------------------------------------------------------------------------

def test_gce_dc10_overpredicts():
    """V5.20 Delta_c=10 sur-predit massivement le GCE."""
    g = gce_comparison(delta_c=10)
    assert g["ratio"] > 10.0
    assert g["verdict"] == "V5.20 sur-predit GCE"


def test_gce_dc30_subdominant():
    g = gce_comparison(delta_c=30)
    assert g["ratio"] < 0.1
    assert g["verdict"] == "V5.20 sous-domine GCE (autre origine)"


# ---------------------------------------------------------------------------
# 6. Plot generation (smoke test)
# ---------------------------------------------------------------------------

def test_make_plot_creates_png(tmp_path):
    out = tmp_path / "v5p20_fermi_lat_test.png"
    p = make_plot(delta_c_values=(10, 22, 30), out_path=out)
    assert p.exists()
    # Verify it's a non-trivial PNG
    assert p.stat().st_size > 10000


# ---------------------------------------------------------------------------
# 7. Coherence ratio
# ---------------------------------------------------------------------------

def test_ratio_consistent_with_individual_intensities():
    """ratio_v5_fermi == I_V5/I_Fermi point par point."""
    E = np.array([0.5, 1.0, 10.0, 100.0])
    I_V5 = intensity_diffuse_v5p20(E, delta_c=10)
    I_F = edgrb_fermi(E)
    r = ratio_v5_fermi(E, delta_c=10)
    np.testing.assert_allclose(r, I_V5 / I_F, rtol=1e-10)
