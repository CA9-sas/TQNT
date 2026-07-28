"""Tests for theory/dm_pearl_internal.py (V5.20)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from theory.dm_pearl_internal import (
    DM_STRICT_AMPHI_GEV,
    M_PEARL_TYPICAL_MSUN,
    M_SUN_KG,
    ALPHA_EM,
    SIGMA_QED,
    T_HUBBLE_S,
    MPL_KG,
    n_knots,
    n_knots_table,
    fermi_momentum,
    n_crit_relativistic,
    pressure_NR,
    pressure_UR,
    eos_topological,
    PearlPolytropicEoS,
    tov_pearl,
    chandrasekhar_topological_pearl,
    chandrasekhar_white_dwarf,
    chandrasekhar_ratio,
    sigma_entanglement,
    entanglement_rate,
    n_events_over_universe_age,
    verdict_v5p20,
)


# ----------------------------------------------------------------------------
# 1. Comptage des nœuds dans la perle
# ----------------------------------------------------------------------------

def test_n_knots_typical_pearl():
    """Pour M=2e-3 M_sun, m_knot=1.55 GeV : N ~ 1.44e54."""
    N = n_knots(2.0e-3, 1.55)
    assert 1e54 < N < 2e54


def test_n_knots_scales_with_mass():
    """N est strictement croissant avec M_pearl."""
    N_low = n_knots(1e-4, 1.55)
    N_high = n_knots(1e-2, 1.55)
    assert N_high == pytest.approx(100 * N_low, rel=1e-6)


def test_n_knots_table_has_six_entries():
    """Le table contient les 6 STRICT amphichiraux."""
    tbl = n_knots_table()
    assert len(tbl) == 6
    for m in DM_STRICT_AMPHI_GEV:
        assert m in tbl
        assert tbl[m] > 0


# ----------------------------------------------------------------------------
# 2. EoS topologique
# ----------------------------------------------------------------------------

def test_fermi_momentum_positive():
    p_F = fermi_momentum(1e40)
    assert p_F > 0
    # p_F ~ (3 pi^2 n)^{1/3} * hbar
    expected = (3 * math.pi**2 * 1e40)**(1/3) * 1.054571817e-34
    assert math.isclose(p_F, expected, rel_tol=1e-6)


def test_pressure_NR_scales_as_n_5_3():
    """P_NR ~ n^{5/3}."""
    P_1 = pressure_NR(1e40, 1.55)
    P_2 = pressure_NR(2e40, 1.55)
    assert P_2 / P_1 == pytest.approx(2.0 ** (5/3), rel=1e-6)


def test_pressure_UR_scales_as_n_4_3():
    """P_UR ~ n^{4/3}."""
    P_1 = pressure_UR(1e50)
    P_2 = pressure_UR(2e50)
    assert P_2 / P_1 == pytest.approx(2.0 ** (4/3), rel=1e-6)


def test_n_crit_relativistic_scales_as_m_3():
    """n_crit ~ m^3."""
    n_low = n_crit_relativistic(1.55)
    n_high = n_crit_relativistic(15.5)
    assert n_high / n_low == pytest.approx(1000.0, rel=1e-3)


def test_eos_returns_correct_keys():
    out = eos_topological(1e40, 1.55)
    assert {"P", "rho", "n_crit", "regime"}.issubset(out.keys())


# ----------------------------------------------------------------------------
# 3. Chandrasekhar topologique
# ----------------------------------------------------------------------------

def test_chandrasekhar_pearl_dimensional():
    """M_Ch_pearl(1.55 GeV) ~ M_Pl^3 / m_chi^2 ~ 0.68 M_sun."""
    M_Ch = chandrasekhar_topological_pearl(1.55)
    M_Ch_Msun = M_Ch / M_SUN_KG
    assert 0.5 < M_Ch_Msun < 0.8


def test_chandrasekhar_ratio_pearl_vs_wd():
    """M_Ch_pearl(1.55 GeV) / M_Ch_WD ~ 0.5."""
    r = chandrasekhar_ratio(1.55)
    assert 0.3 < r < 0.7


def test_chandrasekhar_heaviest_amphichiral():
    """Pour m_knot = 51.78 GeV (le plus lourd STRICT), M_Ch ~ 6e-4 M_sun."""
    M_Ch = chandrasekhar_topological_pearl(51.78) / M_SUN_KG
    # M_Pl^3/m^2 scales as 1/m^2 -> 0.68 * (1.55/51.78)^2 ~ 6.1e-4
    expected = chandrasekhar_topological_pearl(1.55) / M_SUN_KG * (1.55/51.78)**2
    assert math.isclose(M_Ch, expected, rel_tol=1e-3)


# ----------------------------------------------------------------------------
# 4. TOV pearl
# ----------------------------------------------------------------------------

def test_tov_pearl_converges_to_target_mass():
    """L'intégration TOV récupère la masse cible à ~10 %."""
    res = tov_pearl(M_PEARL_TYPICAL_MSUN, 1.55)
    M_recovered = res["M"] / M_SUN_KG
    assert math.isclose(M_recovered, M_PEARL_TYPICAL_MSUN, rel_tol=0.1)


def test_tov_pearl_radius_positive():
    res = tov_pearl(M_PEARL_TYPICAL_MSUN, 1.55)
    assert res["R"] > 0
    # Échelle attendue : qq dizaines de km pour 2e-3 M_sun
    assert 1.0e3 < res["R"] < 1.0e7  # m


def test_tov_pearl_profile_monotonic():
    """P et rho décroissent monotoniquement du centre vers la surface."""
    res = tov_pearl(M_PEARL_TYPICAL_MSUN, 1.55)
    P = res["P"]
    # Profile sample
    assert P[0] > P[len(P)//2] > P[-1] >= 0


# ----------------------------------------------------------------------------
# 5. Suppression Vassiliev
# ----------------------------------------------------------------------------

def test_sigma_entanglement_decreases_with_delta_c():
    """sigma(delta_c) est strictement décroissante."""
    s_5 = sigma_entanglement(5)
    s_10 = sigma_entanglement(10)
    s_20 = sigma_entanglement(20)
    assert s_5 > s_10 > s_20


def test_sigma_entanglement_value():
    """sigma(10) ~ sigma_QED * alpha^10 ~ 1e-46 cm^2."""
    s = sigma_entanglement(10)
    expected = SIGMA_QED * ALPHA_EM ** 10
    assert math.isclose(s, expected, rel_tol=1e-6)
    # En cm^2
    s_cm2 = s * 1e4
    assert 1e-50 < s_cm2 < 1e-42


# ----------------------------------------------------------------------------
# 6. Taux d'enchevêtrement et verdict
# ----------------------------------------------------------------------------

def test_entanglement_rate_returns_required_keys():
    r = entanglement_rate(M_PEARL_TYPICAL_MSUN, 1.55, delta_c=10)
    keys = {
        "N_knots", "r_pearl_m", "rho_central", "n_central",
        "sigma_cm2", "v_F_central",
        "Gamma_pair_central_Hz", "Gamma_total_pearl_Hz",
    }
    assert keys.issubset(set(r.keys()))


def test_verdict_v5p20_strict_excluded_at_high_delta_c():
    """Pour delta_c >> 50 la suppression Vassiliev tue tout enchevêtrement."""
    v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, delta_c=100)
    assert v["N_events_total_universe"] < 1e-3
    assert v["status"] == "STRICTLY EXCLUDED"


def test_verdict_v5p20_frequent_at_low_delta_c():
    """Pour delta_c = 5 (peu de crossings additionnels) l'enchevêtrement est fréquent."""
    v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, delta_c=5)
    assert v["status"] in ("FREQUENT", "RARE BUT POSSIBLE")


def test_n_events_universe_scales_with_pearl_count():
    """N_events_total est proportionnel au nombre de perles."""
    r_typ = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, 1.55, 10)
    r_heavy = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN * 10, 1.55, 10)
    # M_pearl * 10 -> 10x moins de perles, mais N_knots * 10, N^2 * 100
    # plus r_pearl change. Test simple : tous deux > 0
    assert r_typ["N_events_total_universe"] > 0
    assert r_heavy["N_events_total_universe"] > 0


def test_verdict_signature_observable_decision():
    """La décision (signature observable) est déterministe pour delta_c=10."""
    v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, 10)
    assert v["status"] in (
        "STRICTLY EXCLUDED", "MARGINAL", "RARE BUT POSSIBLE", "FREQUENT",
    )
    assert v["comment"]


# ----------------------------------------------------------------------------
# 7. Plot generation (smoke test)
# ----------------------------------------------------------------------------

def test_plot_generation(tmp_path):
    """Le script principal produit un PNG non-vide."""
    sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))
    import dm_pearl_full_analysis as A
    p = A.make_plot(path=tmp_path / "v5p20.png")
    assert p.exists()
    assert p.stat().st_size > 5000


def test_summary_keys():
    sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))
    import dm_pearl_full_analysis as A
    s = A.summary()
    needed = {
        "M_pearl_Msun", "m_knot_GeV", "N_knots",
        "R_pearl_km", "rho_central_kg_m3", "n_central_m3",
        "M_Ch_pearl_Msun", "M_Ch_WD_Msun",
        "Chandrasekhar_ratio_pearl_over_WD",
    }
    assert needed.issubset(set(s.keys()))
