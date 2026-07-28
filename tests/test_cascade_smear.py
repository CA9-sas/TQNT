"""Tests for V5.20 Class E cascade smear analysis.

Vérifie que :
  1. Le modèle de cascade produit un spectre physique (multiplicité,
     ⟨E_γ⟩, pic SED dans les bonnes plages).
  2. FWHM_cascade ≫ FWHM_Fermi pour les 6 amphichiraux (raies smeared).
  3. Le verdict "line search ne s'applique pas" tient pour 6/6.
  4. Le test EDGRB continuum (cascade smeared) donne un dc_min cohérent.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))

import cascade_smear_analysis as CS  # noqa: E402
from theory.dm_pearl_internal import DM_STRICT_AMPHI_GEV  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Forme du spectre de cascade
# ----------------------------------------------------------------------------

def test_cascade_spectrum_normalized_multiplicity_in_range():
    """M_γ par link doit être dans [5, 20] (multiplicité hadronique typique
    pour m_DM ~ 1-50 GeV ; Cirelli+ 2011, Bhattacherjee+ 2018)."""
    for m in DM_STRICT_AMPHI_GEV:
        M = CS.cascade_multiplicity(m)
        assert 5.0 <= M <= 20.0, (
            f"m={m:.2f}: M_gamma={M:.2f} hors plage physique [5, 20]"
        )


def test_cascade_mean_energy_fraction():
    """⟨E_γ⟩ / m_link ∈ [0.01, 0.30] (énergie moyenne ~ m/5 à m/100,
    cohérent avec PYTHIA bb-like soft frag à 1-50 GeV ; Cirelli+ 2011)."""
    for m in DM_STRICT_AMPHI_GEV:
        E_mean = CS.cascade_mean_energy(m)
        ratio = E_mean / m
        assert 0.01 <= ratio <= 0.30, (
            f"m={m:.2f}: ⟨E_γ⟩/m = {ratio:.3f} hors [0.01, 0.30]"
        )


def test_cascade_peak_SED_below_mlink():
    """Pic SED E^2 dN/dE doit être bien sous m_link (cascade soft)."""
    for m in DM_STRICT_AMPHI_GEV:
        Epeak = CS.cascade_peak_energy(m)
        assert 0 < Epeak < 0.6 * m, (
            f"m={m:.2f}: E_peak={Epeak:.3f} pas dans (0, 0.6*m)"
        )


def test_cascade_spectrum_nonnegative():
    """dN/dE >= 0 partout."""
    m = 16.01
    E = np.logspace(-4, 2, 500)
    dNdE = CS.cascade_spectrum_per_link(E, m)
    assert np.all(dNdE >= 0)


# ----------------------------------------------------------------------------
# 2. FWHM cascade vs FWHM Fermi
# ----------------------------------------------------------------------------

def test_fermi_FWHM_is_12pct():
    """ΔE/E Fermi line search ~ 12%."""
    assert CS.FERMI_DE_OVER_E == pytest.approx(0.12)
    assert CS.fermi_FWHM_at_E(10.0) == pytest.approx(1.2)
    assert CS.fermi_FWHM_at_E(50.0) == pytest.approx(6.0)


def test_cascade_FWHM_positive_for_all_amphis():
    """FWHM_cascade > 0 pour chacun des 6 amphichiraux."""
    for m in DM_STRICT_AMPHI_GEV:
        _, _, fwhm = CS.cascade_FWHM(m, mode="ESED")
        assert fwhm > 0, f"m={m}: FWHM nul"


def test_cascade_FWHM_scales_with_m_link():
    """FWHM cascade scale ~ m_link (cascade hadronique self-similar)."""
    fw_low = CS.cascade_FWHM(1.55, mode="ESED")[2]
    fw_hi = CS.cascade_FWHM(51.78, mode="ESED")[2]
    # Scaling roughly linear : fw_hi / fw_low ~ 51.78/1.55 = 33
    ratio = fw_hi / fw_low
    expected = 51.78 / 1.55
    assert 0.5 * expected < ratio < 2.0 * expected, (
        f"FWHM scaling: ratio={ratio:.1f} vs expected~{expected:.1f}"
    )


# ----------------------------------------------------------------------------
# 3. Smear ratio : 6/6 doivent être smeared
# ----------------------------------------------------------------------------

def test_all_six_amphis_are_smeared():
    """6/6 amphichiraux ont FWHM_cascade > k * FWHM_Fermi (k=1.5)
    -> line search Fermi ne s'applique PAS."""
    table = CS.class_E_cascade_smear_table()
    n_smeared = sum(1 for r in table if r["is_smeared_continuum"])
    assert n_smeared == 6, (
        f"Seulement {n_smeared}/6 smeared : line search applicable"
    )


def test_smear_ratio_well_above_threshold():
    """ratio FWHM_casc / FWHM_Fermi est >> 1.5 (au moins 5x) pour les 6."""
    table = CS.class_E_cascade_smear_table()
    for r in table:
        assert r["ratio_FWHM_casc_over_Fermi"] > 5.0, (
            f"{r['m_link_GeV']}: ratio={r['ratio_FWHM_casc_over_Fermi']:.2f} "
            "trop proche du seuil (< 5x)"
        )


# ----------------------------------------------------------------------------
# 4. EDGRB continuum verdict (cascade smeared)
# ----------------------------------------------------------------------------

def test_edgrb_continuum_excluded_at_dc22():
    """À Δc=22 (V5.21), même en cascade smeared, EDGRB exclut encore
    (over-prediction par >>1)."""
    v = CS.verdict_class_E_cascade_vs_EDGRB(delta_c=22)
    assert v["status"] == "EXCLUDED EDGRB"
    assert v["ratio_peak_V5_over_EDGRB"] > 1.0


def test_edgrb_compatible_at_dc30():
    """À Δc=30, V5 Class E (cascade smeared) sous EDGRB."""
    v = CS.verdict_class_E_cascade_vs_EDGRB(delta_c=30)
    assert v["status"] == "COMPATIBLE EDGRB"
    assert v["ratio_peak_V5_over_EDGRB"] < 1.0


def test_dc_min_compatibility_in_realistic_range():
    """Le Δc_min pour passer EDGRB diffus doit être < 30 (test honnête:
    cascade smear assouplit la borne line-search 30 → continuum ~25-28)."""
    v = CS.verdict_class_E_cascade_vs_EDGRB(delta_c=22)
    assert v["delta_c_min_for_compatibility"] is not None
    dc_min = v["delta_c_min_for_compatibility"]
    assert 23 <= dc_min <= 30, (
        f"dc_min={dc_min} hors plage [23, 30] (cascade smear vs EDGRB)"
    )


def test_dc_min_strictly_less_than_line_search_30():
    """Cascade smear DOIT relâcher la borne : dc_min(EDGRB continuum) <
    dc_min(line search Fermi)=30. Sinon le mécanisme n'apporte rien."""
    v = CS.verdict_class_E_cascade_vs_EDGRB(delta_c=22)
    assert v["delta_c_min_for_compatibility"] < 30, (
        "Cascade smear ne relâche pas la borne line search Fermi (dc=30)"
    )


# ----------------------------------------------------------------------------
# 5. Plot smoke test
# ----------------------------------------------------------------------------

def test_plot_generation(tmp_path):
    """make_plot produit un PNG non vide."""
    p = CS.make_plot(path=tmp_path / "cascade_smear.png", delta_c=22)
    assert p.exists()
    assert p.stat().st_size > 5000


# ----------------------------------------------------------------------------
# 6. Cohérence avec class_E_six_lines_predictions (raies discrètes natives)
# ----------------------------------------------------------------------------

def test_partial_redeem_status_honesty():
    """Verdict global doit refléter honnêtement :
    - Line search est neutralisé (6/6 smeared)
    - Mais EDGRB continuum à Δc=22 reste excluant
    -> 'PARTIAL: smeared mais EXCLUDED EDGRB'
    """
    s = CS.global_summary(delta_c=22)
    assert s["n_smeared"] == 6
    assert "PARTIAL" in s["global_status"]
    assert "EXCLUDED EDGRB" in s["global_status"]


def test_full_redeem_at_higher_dc():
    """À Δc=26 (compatible EDGRB selon CLI run), verdict global =
    REDEEMED (cascade smear + continuum compat.)."""
    s = CS.global_summary(delta_c=26)
    assert s["n_smeared"] == 6
    assert "REDEEMED" in s["global_status"]
