"""Tests for V5.20 Class E "Strange Knot Stars" entanglement (amphichiral)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))

import class_E_entanglement_test as CE
from theory.dm_pearl_internal import (
    ALPHA_EM,
    DM_STRICT_AMPHI_GEV,
    M_PEARL_TYPICAL_MSUN,
    sigma_entanglement,
)


# ----------------------------------------------------------------------------
# 1. Catalog: 6 amphichirals
# ----------------------------------------------------------------------------

def test_catalog_has_six_amphichirals():
    """Le catalogue Class E contient exactement les 6 STRICT amphichiraux."""
    assert len(CE.CLASS_E_CATALOG) == 6
    masses = [m for (_, m, _, _) in CE.CLASS_E_CATALOG]
    assert set(masses) == set(DM_STRICT_AMPHI_GEV)


def test_catalog_consistent_with_claude_md():
    """Vérifie les masses exactes du catalogue V5 (xlsx)."""
    expected = {
        "6_3+6_3": 1.55,
        "6_3+6_3+6_3": 4.70,
        "4_1+8_3": 16.01,
        "8_3+8_3": 17.24,
        "6_3+8_3": 26.48,
        "4_1+4_1+8_3": 51.78,
    }
    for name, m_chi, _, _ in CE.CLASS_E_CATALOG:
        assert m_chi == pytest.approx(expected[name], abs=0.01)


# ----------------------------------------------------------------------------
# 2. Delta c amphichiral
# ----------------------------------------------------------------------------

def test_delta_c_amphi_lower_than_class_A():
    """Pour chaque amphichiral Class E, Delta c_E < Delta c_A = 10."""
    for name, m, c_brick, n_brick in CE.CLASS_E_CATALOG:
        dc_E = CE.delta_c_amphichiral(c_brick, n_brick)
        assert dc_E < CE.DELTA_C_CLASS_A, (
            f"{name}: Delta c_E={dc_E} >= Class A reference (=10) — "
            "amphichiralité censée réduire suppression Vassiliev"
        )


def test_delta_c_amphi_scales_with_n_brick():
    """Delta c = 2 * n_brick (formule V5 §13.6.4)."""
    for name, m, c_brick, n_brick in CE.CLASS_E_CATALOG:
        dc = CE.delta_c_amphichiral(c_brick, n_brick)
        assert dc == 2 * n_brick


def test_delta_c_amphi_two_components_is_four():
    """n_brick = 2 (4 sur les 6 compositions) -> Delta c = 4."""
    assert CE.delta_c_amphichiral(12, 2) == 4
    assert CE.delta_c_amphichiral(14, 2) == 4
    assert CE.delta_c_amphichiral(16, 2) == 4


def test_delta_c_amphi_three_components_is_six():
    """n_brick = 3 -> Delta c = 6."""
    assert CE.delta_c_amphichiral(18, 3) == 6
    assert CE.delta_c_amphichiral(16, 3) == 6


# ----------------------------------------------------------------------------
# 3. Gain factor Gamma_E / Gamma_A
# ----------------------------------------------------------------------------

def test_gain_factor_strictly_greater_than_one():
    """Pour Delta c_E < Delta c_A, le gain est > 1."""
    for name, m, c, n in CE.CLASS_E_CATALOG:
        dc_E = CE.delta_c_amphichiral(c, n)
        gain = CE.gain_factor_class_E(dc_E)
        assert gain > 1.0


def test_gain_factor_at_delta_c_E_4():
    """Gain(dc_E=4) = alpha^{4-10} = alpha^{-6} = 137^6 ~ 6.6e12."""
    g = CE.gain_factor_class_E(4)
    expected = ALPHA_EM ** (4 - 10)
    assert math.isclose(g, expected, rel_tol=1e-6)
    assert 1e12 < g < 1e13


# ----------------------------------------------------------------------------
# 4. Sigma (cross-section)
# ----------------------------------------------------------------------------

def test_sigma_E_matches_underlying_module():
    """Le sigma_E par perle utilise bien la formule centrale."""
    for name, m, c, n in CE.CLASS_E_CATALOG:
        dc_E = CE.delta_c_amphichiral(c, n)
        s_expected_m2 = sigma_entanglement(dc_E)
        # rates list returns sigma_E_cm2 (cm^2)
        rates = CE.class_E_entanglement_rates()
        item = next(r for r in rates if r["name"] == name)
        assert math.isclose(item["sigma_E_cm2"], s_expected_m2 * 1e4,
                            rel_tol=1e-6)


# ----------------------------------------------------------------------------
# 5. Mass-Radius
# ----------------------------------------------------------------------------

def test_mass_radius_has_six_entries():
    table = CE.class_E_mass_radius()
    assert len(table) == 6
    for t in table:
        assert t["R_pearl_m"] > 0
        assert t["rho_central"] > 0
        assert t["N_knots"] > 0


def test_M_Ch_decreases_with_m_chi_squared():
    """M_Ch ~ 1/m^2 : amphichiral le + lourd -> M_Ch le + petit."""
    table = CE.class_E_mass_radius()
    sorted_by_mass = sorted(table, key=lambda r: r["m_chi_GeV"])
    M_Ch_seq = [r["M_Ch_Msun"] for r in sorted_by_mass]
    # Strictement décroissant
    for i in range(len(M_Ch_seq) - 1):
        assert M_Ch_seq[i] > M_Ch_seq[i+1]


# ----------------------------------------------------------------------------
# 6. Gamma γ signature discrète
# ----------------------------------------------------------------------------

def test_gamma_signature_six_primary_lines():
    """Le spectre Class E comporte 6 raies primaires distinctes."""
    sig = CE.gamma_signature_class_E()
    primary = sig["primary_lines_GeV"]
    assert len(primary) == 6
    assert len(set(primary)) == 6   # toutes distinctes


def test_gamma_signature_cascade_positive():
    """Les Delta_E de cascade sont positives (de grand vers petit)."""
    sig = CE.gamma_signature_class_E()
    for dE in sig["cascade_lines_GeV"]:
        assert dE > 0


def test_gamma_signature_spans_GeV_range():
    """Le spectre primaire couvre 1.5 -> 52 GeV (fenêtre Fermi-LAT/CTA)."""
    sig = CE.gamma_signature_class_E()
    p = sig["primary_lines_GeV"]
    assert min(p) == pytest.approx(1.55, abs=0.01)
    assert max(p) == pytest.approx(51.78, abs=0.01)


# ----------------------------------------------------------------------------
# 7. Summary / verdict
# ----------------------------------------------------------------------------

def test_summary_keys_present():
    s = CE.summary()
    needed = {"rates", "mass_radius", "gamma_signature",
              "gain_mean_class_E_over_A", "verdict", "comment"}
    assert needed.issubset(s.keys())


def test_verdict_is_distinctive():
    """Avec Delta c_E ~ 4-6 et Delta c_A = 10, gain ~ 1e8-1e12 -> distinctive."""
    s = CE.summary()
    assert s["verdict"] == "SIGNATURE DISTINCTIVE"
    assert s["gain_mean_class_E_over_A"] > 1e6


# ----------------------------------------------------------------------------
# 8. Plot smoke test
# ----------------------------------------------------------------------------

def test_plot_generation(tmp_path):
    """Le script principal produit un PNG non-vide."""
    p = CE.make_plot(path=tmp_path / "v5p20_class_E.png")
    assert p.exists()
    assert p.stat().st_size > 5000
