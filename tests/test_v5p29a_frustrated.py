"""Tests V5.29a — Chemins frustrés amphichirales.

Vérifie :
1. Le facteur F_i = 1 pour les composantes non amphichirales
2. Le facteur F_i = c_i! × b_i pour les composantes amphichirales (4_1, 6_3, 8_3)
3. N_paths V5.29a = N_paths V5.28 × Π F_i
4. K⁰, D⁰, B⁰ inchangés (aucune composante amphichirale)
5. Bs⁰ boosté ×2880 (6_3 amphichiral)
6. K⁰ reste calibré exactement (F_total=1 ne perturbe pas le préfacteur)
7. La hiérarchie Bs⁰ > B⁰ est rétablie (V5.28 inversait)
8. La somme |log_ratio| est strictement améliorée vs V5.28
"""
from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from theory.v5p29_frustrated_amphichiral import (
    frustrated_factor,
    n_paths_with_frustration,
    matrix_element_topo_v5p29a,
    delta_m_v5p29a,
    v5p29a_table,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    count_paths_K_to_Kstar,
    delta_m_chiral_oscillation,
)


# --------------------------------------------------------------------------
# 1. Facteur de frustration par composante
# --------------------------------------------------------------------------
def test_frustrated_factor_nonamphi_is_one():
    """F_i = 1 pour les composantes non amphichirales."""
    for k in ["3_1", "5_1", "6_1", "6_2", "7_2", "7_6", "7_7"]:
        f = frustrated_factor(k)
        assert f["F"] == 1, f"{k} ne devrait pas être amphichiral"
        assert f["amphichiral"] is False


def test_frustrated_factor_amphi_is_c_factorial_times_b():
    """F_i = c_i! × b_i pour les amphichiraux 4_1, 6_3, 8_3."""
    expected = {
        "4_1": factorial(4) * 3,    # 24 × 3 = 72
        "6_3": factorial(6) * 4,    # 720 × 4 = 2880
        "8_3": factorial(8) * 4,    # 40320 × 4 = 161280
    }
    for k, F_exp in expected.items():
        f = frustrated_factor(k)
        assert f["amphichiral"] is True
        assert f["F"] == F_exp, f"F({k}) = {f['F']}, attendu {F_exp}"


# --------------------------------------------------------------------------
# 2. N_paths V5.29a vs V5.28
# --------------------------------------------------------------------------
def test_n_paths_v5p29a_equals_v528_for_no_amphi():
    """N_paths inchangé pour K⁰, D⁰, B⁰ (aucun amphichiral)."""
    for name in ["K0", "D0", "B0"]:
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        v528 = count_paths_K_to_Kstar(cfg["knots"])
        v29a = n_paths_with_frustration(cfg["knots"])
        assert v29a["N_paths"] == v528["N_paths"]
        assert v29a["F_total"] == 1
        assert v29a["n_amphichiral"] == 0


def test_n_paths_v5p29a_boost_Bs0():
    """Bs⁰ = 6_3 + 7_6 : N_paths V5.29a = N_paths V5.28 × 2880."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    v528 = count_paths_K_to_Kstar(cfg["knots"])
    v29a = n_paths_with_frustration(cfg["knots"])
    assert v29a["n_amphichiral"] == 1
    assert v29a["F_total"] == factorial(6) * 4  # 2880
    assert v29a["N_paths"] == v528["N_paths"] * 2880


def test_matrix_element_factorisation_v5p29a():
    """⟨K|H|K*⟩_29a = N_paths^V5.29a × exp(-S_min) (saddle-point)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    me = matrix_element_topo_v5p29a(cfg["knots"], cfg["mass_pdg"])
    expected = me["N_paths"] * np.exp(-me["S_min"])
    assert abs(me["matrix_element"] - expected) / expected < 1e-10


# --------------------------------------------------------------------------
# 3. Δm V5.29a : conservation calibration et inchangé pour non-amphi
# --------------------------------------------------------------------------
def test_K0_still_calibrated_exact():
    """K⁰ : Δm V5.29a = Δm PDG par construction (calibration intacte)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    rel = abs(res["delta_m_MeV"] - DELTA_M_PDG["K0"]) / DELTA_M_PDG["K0"]
    assert rel < 1e-9, f"K0 calibration drift V5.29a : {rel}"


def test_D0_unchanged_vs_v528():
    """D⁰ : Δm V5.29a = Δm V5.28 (aucun amphichiral)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["D0"]
    res_29a = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    res_528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
    rel = abs(res_29a["delta_m_MeV"] - res_528["delta_m_MeV"]) / res_528[
        "delta_m_MeV"
    ]
    assert rel < 1e-9, f"D0 drift V5.29a vs V5.28 : {rel}"


def test_B0_unchanged_vs_v528():
    """B⁰ : Δm V5.29a = Δm V5.28 (aucun amphichiral)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["B0"]
    res_29a = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    res_528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
    rel = abs(res_29a["delta_m_MeV"] - res_528["delta_m_MeV"]) / res_528[
        "delta_m_MeV"
    ]
    assert rel < 1e-9, f"B0 drift V5.29a vs V5.28 : {rel}"


# --------------------------------------------------------------------------
# 4. Δm Bs⁰ : boost ×2880, gap fermé
# --------------------------------------------------------------------------
def test_Bs0_boosted_by_2880_vs_v528():
    """Δm(Bs⁰) V5.29a = Δm(Bs⁰) V5.28 × 2880 (boost combinatoire pur)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    res_29a = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    res_528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
    ratio = res_29a["delta_m_MeV"] / res_528["delta_m_MeV"]
    assert abs(ratio - 2880) / 2880 < 1e-6, (
        f"Boost Bs⁰ V5.29a/V5.28 = {ratio}, attendu 2880"
    )


def test_Bs0_log_ratio_within_half_dex():
    """Bs⁰ : V5.29a doit fermer le gap sous ±0.5 dex (V5.28 était à -3.30)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    res = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["Bs0"])
    assert abs(log_ratio) < 0.5, (
        f"Bs⁰ V5.29a hors fenêtre ±0.5 dex : log_ratio = {log_ratio:+.3f}"
    )


# --------------------------------------------------------------------------
# 5. Hiérarchie Bs > B rétablie
# --------------------------------------------------------------------------
def test_Bs0_greater_than_B0_v5p29a():
    """V5.29a doit prédire Δm(Bs⁰) > Δm(B⁰), comme dans PDG.

    En V5.28, on avait Bs⁰ ~ 0.09 × B⁰ (inverse).  V5.29a doit
    rétablir le bon ordre, même si le ratio absolu reste imparfait.
    """
    rows = v5p29a_table()
    by_name = {r["name"]: r["delta_m_v5p29a_MeV"] for r in rows}
    assert by_name["Bs0"] > by_name["B0"], (
        f"V5.29a doit rétablir Bs⁰ > B⁰ ; "
        f"obtenu Bs⁰={by_name['Bs0']:.2e}, B⁰={by_name['B0']:.2e}"
    )


# --------------------------------------------------------------------------
# 6. Cohérence formelle et globale
# --------------------------------------------------------------------------
def test_delta_m_v5p29a_equals_two_prefactor_matrix_element():
    """Δm = 2 × prefactor × matrix_element (cohérence du facteur 2)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    res = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
    expected = 2.0 * res["prefactor_MeV"] * res["matrix_element"]
    assert abs(res["delta_m_MeV"] - expected) / expected < 1e-12


def test_v5p29a_global_log_ratio_improved():
    """La somme cumulée |log_ratio| est strictement améliorée vs V5.28."""
    rows = v5p29a_table()
    sum_528 = sum(abs(r["log_ratio_v528"]) for r in rows)
    sum_29a = sum(abs(r["log_ratio_v5p29a"]) for r in rows)
    assert sum_29a < sum_528, (
        f"V5.29a devrait améliorer Σ|log_ratio| : "
        f"V5.28={sum_528:.3f}, V5.29a={sum_29a:.3f}"
    )


def test_v5p29a_table_complete():
    """La table V5.29a contient les 4 mésons."""
    rows = v5p29a_table()
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}
    for r in rows:
        assert r["log_ratio_v5p29a"] is not None
        assert np.isfinite(r["log_ratio_v5p29a"])
