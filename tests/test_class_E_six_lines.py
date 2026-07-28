"""Tests for V5.20 Class E six gamma-ray line predictions vs Fermi-LAT + CTA."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))

import class_E_six_lines_predictions as SL
from theory.dm_pearl_internal import DM_STRICT_AMPHI_GEV


# ----------------------------------------------------------------------------
# 1. Position des 6 raies
# ----------------------------------------------------------------------------

def test_six_lines_at_amphichiral_masses():
    """Les 6 raies sont aux masses des amphichiraux V5 catalogue xlsx."""
    lines = SL.relative_intensity_per_line()
    masses = {L["m_chi_GeV"] for L in lines}
    assert masses == set(DM_STRICT_AMPHI_GEV)
    assert len(lines) == 6


def test_six_lines_native_delta_c_4_or_6():
    """delta_c_E natif = 2*n_brick -> 4 (n=2) ou 6 (n=3)."""
    lines = SL.relative_intensity_per_line(use_v5p21_min_dc=False)
    for L in lines:
        assert L["delta_c_E_native"] in (4, 6)
        assert L["delta_c_used"] == L["delta_c_E_native"]


# ----------------------------------------------------------------------------
# 2. V5.21 strict (dc >= 22)
# ----------------------------------------------------------------------------

def test_v5p21_imposes_min_delta_c_22():
    """Sous condition V5.21, delta_c_used >= 22 pour TOUTES les raies."""
    lines = SL.relative_intensity_per_line(use_v5p21_min_dc=True)
    for L in lines:
        assert L["delta_c_used"] >= 22


def test_v5p21_strict_suppresses_flux_vs_baseline():
    """V5.21 (dc=22) doit donner des flux STRICTEMENT inferieurs a V5.20
    baseline (dc=4-6), puisque sigma ~ alpha^dc."""
    base = SL.relative_intensity_per_line(use_v5p21_min_dc=False)
    strict = SL.relative_intensity_per_line(use_v5p21_min_dc=True)
    for Lb, Ls in zip(base, strict):
        assert Ls["phi_line_ph_cm2_s"] < Lb["phi_line_ph_cm2_s"]


# ----------------------------------------------------------------------------
# 3. Largeur de raie naturelle ~ 0, instrumentale domine
# ----------------------------------------------------------------------------

def test_natural_width_is_zero():
    """Transition topologique discrete -> largeur naturelle = 0."""
    w = SL.line_widths_GeV(10.0)
    assert w["natural_GeV"] == 0.0


def test_cta_better_resolution_than_fermi():
    """CTA ~7%/Fermi ~12% -> raies mieux resolues par CTA."""
    for E in [2.0, 16.0, 50.0]:
        w = SL.line_widths_GeV(E)
        assert w["cta_FWHM_GeV"] < w["fermi_FWHM_GeV"]
        assert w["fermi_DE_E"] == pytest.approx(0.12, rel=1e-6)
        assert w["cta_DE_E"] == pytest.approx(0.07, rel=1e-6)


# ----------------------------------------------------------------------------
# 4. Fermi-LAT line search limits (interpolation)
# ----------------------------------------------------------------------------

def test_fermi_limit_monotonic_in_amplitude():
    """Les limites Fermi-LAT line search sont ~ 1e-10 ph cm^-2 s^-1
    autour de 10-100 GeV (Ackermann+ 2015)."""
    phi10 = SL.fermi_limit_at_E(10.0)
    phi100 = SL.fermi_limit_at_E(100.0)
    # Les deux dans la bonne fenetre
    assert 1e-12 < phi10 < 1e-8
    assert 1e-12 < phi100 < 1e-8


def test_cta_below_25GeV_is_out_of_band():
    """CTA ne couvre pas E < 25 GeV (sub-array North limit)."""
    assert SL.cta_limit_at_E(10.0) == float('inf')
    assert SL.cta_limit_at_E(5.0) == float('inf')
    # Mais OK >= 25 GeV
    assert SL.cta_limit_at_E(30.0) < float('inf')


def test_cta_more_sensitive_than_fermi_in_overlap():
    """Dans la zone d'overlap (>25 GeV), CTA forecast < Fermi limit."""
    for E in [30.0, 50.0, 100.0]:
        assert SL.cta_limit_at_E(E) < SL.fermi_limit_at_E(E)


# ----------------------------------------------------------------------------
# 5. Verdict global -- honnetete sur l'exclusion
# ----------------------------------------------------------------------------

def test_baseline_v5p20_excluded_by_fermi():
    """V5.20 baseline (dc_E natif ~4-6) est EXCLU par Fermi-LAT line
    search (sur-prediction massive)."""
    v = SL.global_verdict(use_v5p21_min_dc=False)
    assert v["n_excluded_fermi"] == 6
    assert "EXCLUDED" in v["status"]


def test_v5p21_strict_still_excluded_by_fermi():
    """Meme avec V5.21 (dc=22), les 6 raies sur-predisent Fermi.
    Verdict honnete : Class E EXCLU sans suppression Vassiliev
    supplementaire (dc_crit > 22)."""
    v = SL.global_verdict(use_v5p21_min_dc=True)
    assert v["n_excluded_fermi"] == 6


def test_critical_delta_c_per_line_above_22():
    """Le dc_crit pour passer sous Fermi est > 22 pour les 6 raies
    -> meme V5.21 n'est pas suffisamment supprime."""
    crit = SL.critical_delta_c_per_line()
    assert len(crit) == 6
    for c in crit:
        assert c["delta_c_critical_Fermi"] > 22, (
            f"{c['name']}: dc_crit={c['delta_c_critical_Fermi']} "
            "expected > 22"
        )


def test_critical_delta_c_in_realistic_range():
    """dc_crit doit etre dans la fenetre 23-35 (verifiable par alpha^N
    avec N modeste)."""
    crit = SL.critical_delta_c_per_line()
    for c in crit:
        assert 23 <= c["delta_c_critical_Fermi"] <= 35


# ----------------------------------------------------------------------------
# 6. CTA detectabilite (2027+)
# ----------------------------------------------------------------------------

def test_cta_only_covers_4_lines_in_band():
    """CTA couvre 25 GeV-100 TeV -> 4 raies sur 6 in-band (16, 17, 26, 52)."""
    lines = SL.relative_intensity_per_line(use_v5p21_min_dc=False)
    in_band = [L for L in lines if L["m_chi_GeV"] >= 25.0]
    out_band = [L for L in lines if L["m_chi_GeV"] < 25.0]
    # 2 in band : 26.48 et 51.78
    assert len(in_band) == 2
    # 4 out of band : 1.55, 4.70, 16.01, 17.24
    assert len(out_band) == 4


def test_cta_detectability_count_matches_verdict():
    """Le verdict global doit compter coherentement les n_detectable_cta."""
    v = SL.global_verdict(use_v5p21_min_dc=False)
    explicit = sum(1 for L in v["per_line"]
                   if L["cta_status"] == "DETECTABLE")
    assert v["n_detectable_cta"] == explicit


# ----------------------------------------------------------------------------
# 7. Plot smoke test
# ----------------------------------------------------------------------------

def test_plot_generation(tmp_path):
    """make_plot produit un PNG non-vide."""
    p = SL.make_plot(path=tmp_path / "six_lines.png")
    assert p.exists()
    assert p.stat().st_size > 5000
