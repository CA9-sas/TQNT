"""Tests V5.20 Class A continuum vs Fermi-LAT (V5.21 Delta c >= 22).

scripts/V5_Fermi_LAT/class_A_continuum_test.py
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

from class_A_continuum_test import (
    DELTA_C_DEFAULT,
    M_MAX_GEV,
    M_MIN_GEV,
    MU_GEV_DEFAULT,
    N_MASS_BINS,
    SIGMA_DEX_DEFAULT,
    cascade_dN_dE_bin,
    intensity_class_A_continuum,
    make_plot,
    mass_grid_and_pdf,
    mass_pdf,
    n_events_rate_per_m3_s,
    ratio_continuum_vs_fermi,
    verdict_continuum,
)


# ---------------------------------------------------------------------------
# 1. Mass function : log-normale normalisee
# ---------------------------------------------------------------------------

def test_mass_pdf_positive_and_finite():
    m = np.logspace(-1, 2, 50)
    p = mass_pdf(m)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)


def test_mass_pdf_peak_at_mu():
    """Le pic de m * pdf (log-uniforme) est proche de mu = 10 GeV."""
    m = np.logspace(-0.5, 1.7, 1000)
    p = mass_pdf(m)
    weighted = m * p   # equivalent du pic en m
    m_peak = m[int(np.argmax(weighted))]
    assert 7.0 < m_peak < 14.0


def test_mass_grid_pdf_normalised():
    """Integration sur la grille tronquee : doit etre proche de 1 (~0.99)."""
    m, p = mass_grid_and_pdf()
    integral = float(np.trapezoid(p, m))
    assert 0.95 < integral < 1.05


# ---------------------------------------------------------------------------
# 2. Cascade spectrum centre sur m_chi
# ---------------------------------------------------------------------------

def test_cascade_bin_normalised():
    """Integration cascade log-normale = 1 sur large bande."""
    E = np.logspace(-3, 3, 5000)
    spec = cascade_dN_dE_bin(E, m_chi_GeV=10.0)
    integ = float(np.trapezoid(spec, E))
    assert 0.95 < integ < 1.05


def test_cascade_bin_peaks_at_m_chi():
    """Le pic E*dN/dE est proche de m_chi."""
    m_chi = 5.0
    E = np.logspace(-2, 2, 2000)
    spec = cascade_dN_dE_bin(E, m_chi_GeV=m_chi)
    E_peak = E[int(np.argmax(E * spec))]
    assert 0.5 * m_chi < E_peak < 2.0 * m_chi


# ---------------------------------------------------------------------------
# 3. Rate per bin : monotone croissant en m_chi (saturation Vassiliev fixe)
# ---------------------------------------------------------------------------

def test_rate_monotone_in_m_chi():
    """A Delta_c fixe (>= 22) la densite n_F augmente avec m_chi, donc le
    taux d'evenements grossit pour m_chi croissant (tant que TOV converge)."""
    dc = 22
    r1 = n_events_rate_per_m3_s(1.55, dc)
    r10 = n_events_rate_per_m3_s(10.0, dc)
    r30 = n_events_rate_per_m3_s(30.0, dc)
    assert r10 > r1
    assert r30 > r10


def test_rate_decreases_with_delta_c():
    """Suppression Vassiliev : Delta_c +1 -> taux divise par 1/alpha^1 ~ 137."""
    m = 10.0
    r22 = n_events_rate_per_m3_s(m, 22)
    r30 = n_events_rate_per_m3_s(m, 30)
    r50 = n_events_rate_per_m3_s(m, 50)
    assert r22 > r30 > r50
    # ratio ~ alpha^(30-22) = 1/137^8 ~ 1.6e-17
    assert r30 / r22 < 1e-15


# ---------------------------------------------------------------------------
# 4. Intensite continuum finite et positive
# ---------------------------------------------------------------------------

def test_intensity_finite_positive_dc22():
    E = np.logspace(-1, 2, 30)
    I = intensity_class_A_continuum(E, delta_c=22)
    assert np.all(np.isfinite(I))
    assert np.all(I >= 0)
    # Au-dessus du peak du mass function (~10 GeV) le flux est non nul
    assert I[int(np.argmin(np.abs(E - 10.0)))] > 0


def test_intensity_decreases_with_delta_c():
    """A E fixe, intensite continuum monotone decroissante en Delta_c."""
    E = np.array([1.0, 10.0])
    I22 = intensity_class_A_continuum(E, delta_c=22)
    I30 = intensity_class_A_continuum(E, delta_c=30)
    I50 = intensity_class_A_continuum(E, delta_c=50)
    assert np.all(I22 > I30)
    assert np.all(I30 > I50)


# ---------------------------------------------------------------------------
# 5. Verdict V5.21 (Delta c=22) : Class A continuum EXCEDE EDGRB (F3 OPEN)
# ---------------------------------------------------------------------------

def test_verdict_dc22_overpredicts_EDGRB():
    """Critique : Class A continuum mass function complete + Delta_c=22
    sur-predit largement EDGRB."""
    v = verdict_continuum(delta_c=22)
    # Ratio integre 1-100 GeV largement > 1
    assert v["ratio_integrated_1_100GeV"] > 1.0
    # Ratio @ 1 GeV grand
    assert v["ratio_1GeV"] > 100.0
    # Status = exclusion
    assert v["status"] in ("STRICTLY EXCLUDED", "EXCLUDED")


def test_verdict_dc22_overpredicts_GCE():
    """Class A continuum dc=22 sur-predit aussi le GCE."""
    v = verdict_continuum(delta_c=22)
    assert v["ratio_gce_1_10GeV"] > 1.0


def test_verdict_dc22_overpredicts_line_search():
    """Class A continuum dc=22 viole egalement Fermi line search."""
    v = verdict_continuum(delta_c=22)
    assert v["line_search_ratio_max"] > 1.0


def test_verdict_dc30_compatible():
    """A Delta_c=30 : suppression Vassiliev suffit pour passer sous Fermi."""
    v = verdict_continuum(delta_c=30)
    assert v["ratio_integrated_1_100GeV"] < 1.0
    assert v["status"] in ("COMPATIBLE", "DEEPLY SUB-DOMINANT", "MARGINAL")


# ---------------------------------------------------------------------------
# 6. Ratio coherence
# ---------------------------------------------------------------------------

def test_ratio_consistent_with_intensity():
    """ratio_continuum_vs_fermi == I_V5 / I_Fermi point par point."""
    from class_A_continuum_test import intensity_class_A_continuum
    from v5p20_gamma_diffuse_test import edgrb_fermi
    E = np.array([0.5, 1.0, 10.0, 50.0])
    r = ratio_continuum_vs_fermi(E, delta_c=22)
    expected = intensity_class_A_continuum(E, delta_c=22) / edgrb_fermi(E)
    np.testing.assert_allclose(r, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 7. Critical Delta_c pour Class A continuum
# ---------------------------------------------------------------------------

def test_critical_delta_c_above_22():
    """Le Delta_c necessaire pour Class A continuum > 22 (V5.21 insuffisant)."""
    # Test : iterer dc=22,24,...,40 et trouver le premier qui passe
    dc_pass = None
    for dc in range(22, 50, 2):
        v = verdict_continuum(delta_c=dc)
        if v["ratio_integrated_1_100GeV"] < 1.0:
            dc_pass = dc
            break
    assert dc_pass is not None
    # On exige strictement Delta_c > 22 pour Class A continuum complete
    assert dc_pass > 22


# ---------------------------------------------------------------------------
# 8. Smoke test : plot
# ---------------------------------------------------------------------------

def test_make_plot_creates_png(tmp_path):
    out = tmp_path / "v5p20_class_A_continuum_fermi.png"
    p = make_plot(delta_c_values=(22, 30, 50), out_path=out)
    assert p.exists()
    assert p.stat().st_size > 10000
