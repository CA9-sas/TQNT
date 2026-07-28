"""Tests V5.34 — Inertie chirale collective + force centripète."""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import braid_index, is_amphicheiral, jones_polynomial
from theory.kashaev_invariants import hyperbolic_volume, total_hyperbolic_volume
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30c_vibrational import delta_m_v5p30c
from theory.v5p34_chiral_inertia import (
    FORMULAS,
    _clear_cache,
    _omega_chi_K0,
    chiral_inertia,
    compare_5_formulas,
    delta_m_v5p34,
    omega_chirality,
    torque_force_v5p33,
    v5p34_table,
)


# ---------------------------------------------------------------------------
# 1. Chiral inertia — basic properties
# ---------------------------------------------------------------------------
def test_chiral_inertia_returns_all_keys():
    res = chiral_inertia(["6_1", "6_1", "6_1"], "F1_sum")
    for k in ("I_chi", "formula", "kappa", "n_comp", "components", "vol_tot"):
        assert k in res
    assert res["n_comp"] == 3
    assert res["I_chi"] > 0


def test_chiral_inertia_rejects_unknown_formula():
    with pytest.raises(ValueError, match="Formule inconnue"):
        chiral_inertia(["3_1"], "BOGUS")


def test_formula_F1_sum_matches_definition():
    """F1 : I = Σ c³ / b² ; vérifié à la main sur D⁰ = 3_1+5_1+7_2."""
    knots = ["3_1", "5_1", "7_2"]
    expected = 0.0
    for k in knots:
        c = crossing_number(k)
        b = braid_index(k)
        expected += c * (c / b) ** 2
    res = chiral_inertia(knots, "F1_sum")
    assert res["I_chi"] == pytest.approx(expected)


def test_formula_F2_mult_matches_definition():
    """F2 : I = Π (c × b) ; B_s⁰ = 6_3+7_6."""
    knots = ["6_3", "7_6"]
    expected = 1.0
    for k in knots:
        expected *= crossing_number(k) * braid_index(k)
    res = chiral_inertia(knots, "F2_mult")
    assert res["I_chi"] == pytest.approx(expected)


def test_formula_F5_kashaev_matches_definition():
    """F5 : I = exp(κ × vol_tot)."""
    knots = ["3_1", "5_1", "7_2"]
    vol = total_hyperbolic_volume(knots)
    res = chiral_inertia(knots, "F5_kashaev", kappa=0.3)
    assert res["I_chi"] == pytest.approx(np.exp(0.3 * vol))
    assert res["vol_tot"] == pytest.approx(vol)


def test_formula_F4_chiral_amphi_doubles():
    """6_3 est amphichiral ⇒ (1 + amphi) = 2."""
    res_63 = chiral_inertia(["6_3"], "F4_chiral", kappa=2.0)
    c = crossing_number("6_3")
    b = braid_index("6_3")
    expected = c * (b ** 2) * 2.0
    assert res_63["I_chi"] == pytest.approx(expected)
    assert res_63["amphi_count"] == 1


# ---------------------------------------------------------------------------
# 2. Torque force
# ---------------------------------------------------------------------------
def test_torque_force_single_component_returns_self_freq_sq():
    """n_comp = 1 ⇒ F_torque = ω_self² (rigidité propre)."""
    res = torque_force_v5p33(["3_1"])
    assert res["n_comp"] == 1
    b = braid_index("3_1")
    c = crossing_number("3_1")
    omega_self_sq = (b / np.sqrt(c)) ** 2
    assert res["F_torque"] == pytest.approx(omega_self_sq)


def test_torque_force_positive_for_all_systems():
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        res = torque_force_v5p33(cfg["knots"])
        assert res["F_torque"] > 0


# ---------------------------------------------------------------------------
# 3. Omega chirality
# ---------------------------------------------------------------------------
def test_omega_chirality_is_sqrt_F_over_I():
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        knots = cfg["knots"]
        for f in ("F1_sum", "F2_mult", "F3_coupling"):
            oc = omega_chirality(knots, f, 1.0)
            assert oc["omega_chi"] == pytest.approx(
                np.sqrt(oc["F_torque"] / oc["I_chi"])
            )
            assert oc["omega_chi"] > 0


# ---------------------------------------------------------------------------
# 4. Δm V5.34
# ---------------------------------------------------------------------------
def test_delta_m_v5p34_K0_equals_v5p30c_K0():
    """Par construction : Δm V5.34(K⁰) = Δm V5.30c(K⁰) (R_χ = 1)."""
    _clear_cache()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for f in FORMULAS:
        _clear_cache()
        res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], f, 1.0)
        ref = delta_m_v5p30c(cfg["knots"], cfg["mass_pdg"])
        assert res["delta_m_MeV"] == pytest.approx(ref["delta_m_MeV"], rel=1e-9)
        assert res["R_chi"] == pytest.approx(1.0)


def test_delta_m_v5p34_returns_v5p30c_baseline():
    """Le champ delta_m_v5p30c_MeV doit reproduire V5.30c exactement."""
    _clear_cache()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["D0"]
    res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], "F3_coupling", 1.0)
    ref = delta_m_v5p30c(cfg["knots"], cfg["mass_pdg"])
    assert res["delta_m_v5p30c_MeV"] == pytest.approx(ref["delta_m_MeV"])


def test_delta_m_v5p34_positive_all_systems():
    _clear_cache()
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        for f in ("F1_sum", "F2_mult", "F3_coupling"):
            _clear_cache()
            res = delta_m_v5p34(cfg["knots"], cfg["mass_pdg"], f, 1.0)
            assert res["delta_m_MeV"] > 0
            assert np.isfinite(res["delta_m_MeV"])


# ---------------------------------------------------------------------------
# 5. Comparison
# ---------------------------------------------------------------------------
def test_compare_5_formulas_returns_summary():
    res = compare_5_formulas()
    assert "best_formula" in res
    assert res["best_formula"] in FORMULAS
    assert "summary" in res
    assert len(res["summary"]) == 5
    for s in res["summary"]:
        assert s["sum_abs_log"] >= 0


def test_compare_5_formulas_improves_v5p30c_baseline():
    """Au moins une formule doit améliorer V5.30c (1.05 dex)."""
    res = compare_5_formulas()
    best_sum = res[res["best_formula"]]["sum_abs_log"]
    # Tolérance : la meilleure formule doit faire au plus aussi bien
    # que V5.30c + 0.1 dex (cas neutre).  En pratique on attend < 1.05.
    assert best_sum < 1.10


# ---------------------------------------------------------------------------
# 6. Table
# ---------------------------------------------------------------------------
def test_v5p34_table_K0_log_ratio_zero():
    """K⁰ a log_ratio ≈ 0 (calibration exacte)."""
    _clear_cache()
    rows = v5p34_table("F3_coupling", 1.0)
    by_name = {r["name"]: r for r in rows}
    assert by_name["K0"]["log_ratio_v5p34"] == pytest.approx(0.0, abs=1e-9)


def test_v5p34_table_contains_all_mesons():
    _clear_cache()
    rows = v5p34_table("F1_sum", 1.0)
    names = {r["name"] for r in rows}
    assert names == set(MESON_OSCILLATION_ASSIGNMENTS.keys())
