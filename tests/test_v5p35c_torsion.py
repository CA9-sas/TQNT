"""Tests V5.35c — Torsion writhe vs linking dans l'inertie chirale."""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import braid_index
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30c_vibrational import delta_m_v5p30c
from theory.v5p35c_torsion_writhe import (
    _clear_cache,
    _WRITHE_TABLE,
    chiral_inertia_v5p35c,
    delta_m_v5p35c,
    is_chiral,
    knot_writhe,
    omega_chirality_v5p35c,
    summary_v5p35c,
    v5p35c_table,
    writhe_inter_components,
)


# ---------------------------------------------------------------------------
# 1. Writhe table — standard values
# ---------------------------------------------------------------------------
def test_writhe_table_chiral_knots_match_standard():
    """Valeurs standards Rolfsen pour les chiraux clés."""
    assert knot_writhe("3_1") == 3
    assert knot_writhe("5_1") == 5
    assert knot_writhe("6_1") == 2
    assert knot_writhe("6_2") == 4
    assert knot_writhe("7_2") == 7
    assert knot_writhe("7_6") == 5
    assert knot_writhe("7_7") == 3


def test_writhe_amphichiral_knots_are_zero():
    """Amphichiraux : 4_1, 6_3, 8_3 → writhe = 0."""
    assert knot_writhe("4_1") == 0
    assert knot_writhe("6_3") == 0
    assert knot_writhe("8_3") == 0


def test_is_chiral_consistent_with_writhe():
    assert is_chiral("3_1") is True
    assert is_chiral("6_1") is True
    assert is_chiral("6_3") is False
    assert is_chiral("4_1") is False


# ---------------------------------------------------------------------------
# 2. Writhe inter-composantes
# ---------------------------------------------------------------------------
def test_writhe_inter_same_chirality_is_plus_one():
    """Deux chiraux alignés (+1, +1) → w_ij = +1."""
    res = writhe_inter_components("3_1", "5_1", +1, +1)
    assert res["w_ij_signed"] == +1
    assert res["is_chiral_i"] and res["is_chiral_j"]


def test_writhe_inter_opposite_chirality_is_minus_one():
    res = writhe_inter_components("3_1", "5_1", +1, -1)
    assert res["w_ij_signed"] == -1


def test_writhe_inter_with_amphi_is_zero():
    """Une composante amphi → w_ij = 0 quelle que soit la chiralité."""
    res = writhe_inter_components("3_1", "6_3", +1, +1)
    assert res["w_ij_signed"] == 0
    res2 = writhe_inter_components("6_3", "6_3", +1, +1)
    assert res2["w_ij_signed"] == 0


# ---------------------------------------------------------------------------
# 3. Chiral inertia V5.35c
# ---------------------------------------------------------------------------
def test_chiral_inertia_K0_only_self_term():
    """K⁰ = 3×6_1 : 6_1 chiral, alignement +1 → tous les inter sont +.

    Vérifie l'égalité analytique : I_self = 3 × c³/b² avec c=6, b=3.
    """
    res = chiral_inertia_v5p35c(["6_1", "6_1", "6_1"])
    c, b = 6, 3
    I_self_expected = 3 * c * (c / b) ** 2
    assert res["I_self"] == pytest.approx(I_self_expected)
    # 3 paires (i<j) toutes chiral+chiral, signe +1 (alignement)
    # J_ij = √(c c) × b × b × 1 = 6 × 9 = 54.  3 paires → 162.
    assert res["I_inter"] == pytest.approx(3 * (6 * 3 * 3))
    assert res["I_chi"] == pytest.approx(I_self_expected + 162.0)


def test_chiral_inertia_Bs0_amphi_kills_inter():
    """B_s⁰ = 6_3 + 7_6 : 6_3 amphi → w_ij = 0 → I_inter = 0."""
    res = chiral_inertia_v5p35c(["6_3", "7_6"])
    assert res["I_inter"] == pytest.approx(0.0)
    # I_self purement somme c³/b²
    c1, b1 = crossing_number("6_3"), braid_index("6_3")
    c2, b2 = crossing_number("7_6"), braid_index("7_6")
    expected_self = c1 * (c1 / b1) ** 2 + c2 * (c2 / b2) ** 2
    assert res["I_self"] == pytest.approx(expected_self)
    assert res["I_chi"] == pytest.approx(expected_self)


def test_chiral_inertia_B0_three_chirals_all_inter_active():
    """B⁰ = 3_1 + 6_2 + 7_7 : 3 chiraux alignés → 3 inter actifs."""
    res = chiral_inertia_v5p35c(["3_1", "6_2", "7_7"])
    # 3 inter (3 paires) tous +1
    actives = [c for c in res["inter_couplings"] if c["w_ij_signed"] != 0]
    assert len(actives) == 3
    assert all(c["w_ij_signed"] == +1 for c in actives)


def test_chiral_inertia_B0_opposite_chirality_negates_inter():
    """Si on inverse la chiralité d'un nœud, l'inter correspondant flip."""
    res_pos = chiral_inertia_v5p35c(["3_1", "6_2", "7_7"], [+1, +1, +1])
    res_neg = chiral_inertia_v5p35c(["3_1", "6_2", "7_7"], [+1, -1, +1])
    # paires concernées : (3_1,6_2) et (6_2,7_7) flippent ; (3_1,7_7) reste +
    by_pair_pos = {(c["i"], c["j"]): c["w_ij_signed"]
                   for c in res_pos["inter_couplings"]}
    by_pair_neg = {(c["i"], c["j"]): c["w_ij_signed"]
                   for c in res_neg["inter_couplings"]}
    assert by_pair_pos[(0, 1)] == +1 and by_pair_neg[(0, 1)] == -1
    assert by_pair_pos[(1, 2)] == +1 and by_pair_neg[(1, 2)] == -1
    assert by_pair_pos[(0, 2)] == +1 and by_pair_neg[(0, 2)] == +1


# ---------------------------------------------------------------------------
# 4. ω_χ et Δm V5.35c
# ---------------------------------------------------------------------------
def test_omega_chirality_v5p35c_positive_for_all_mesons():
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        oc = omega_chirality_v5p35c(cfg["knots"])
        assert oc["omega_chi"] > 0
        assert np.isfinite(oc["omega_chi"])


def test_delta_m_v5p35c_K0_equals_v5p30c_K0():
    """Par construction Δm V5.35c(K⁰) = Δm V5.30c(K⁰) (R_χ = 1)."""
    _clear_cache()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p35c(cfg["knots"], cfg["mass_pdg"])
    ref = delta_m_v5p30c(cfg["knots"], cfg["mass_pdg"])
    assert res["R_chi"] == pytest.approx(1.0)
    assert res["delta_m_MeV"] == pytest.approx(ref["delta_m_MeV"], rel=1e-9)


def test_delta_m_v5p35c_positive_all_systems():
    _clear_cache()
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        res = delta_m_v5p35c(cfg["knots"], cfg["mass_pdg"])
        assert res["delta_m_MeV"] > 0
        assert np.isfinite(res["delta_m_MeV"])


# ---------------------------------------------------------------------------
# 5. Table V5.35c
# ---------------------------------------------------------------------------
def test_v5p35c_table_contains_all_4_mesons():
    rows = v5p35c_table()
    names = {r["name"] for r in rows}
    assert names == set(MESON_OSCILLATION_ASSIGNMENTS.keys())


def test_v5p35c_summary_returns_sum_abs_log_and_ratio():
    s = summary_v5p35c()
    assert "sum_abs_log" in s and s["sum_abs_log"] >= 0
    assert "ratio_Bs_B" in s and s["ratio_Bs_B"] > 0
