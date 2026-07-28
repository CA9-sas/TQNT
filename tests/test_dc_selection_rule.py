"""Tests V5.21 : règle de sélection Δc ≥ 22 pour perles DM."""
from __future__ import annotations

import pytest

from theory.dc_selection_rule import (
    AMPHICHIRAL_DATA,
    DELTA_C_MIN_OVERALL,
    Amphichiral,
    all_amphichiral_pairs_delta_c_min,
    compute_delta_c_min,
    delta_c_Lk,
    delta_c_Q_phys,
    delta_c_SU2_135,
    delta_c_vassiliev,
    su2_135_fusion_admissibility,
    vassiliev_first_non_zero,
)


# ---------------------------------------------------------------------------
# 1. Les 6 STRICT amphichiraux
# ---------------------------------------------------------------------------

def test_six_amphichirals_loaded():
    assert len(AMPHICHIRAL_DATA) == 6


def test_amphichiral_masses_match_V5():
    expected = {
        "6_3+6_3": 1.55,
        "6_3+6_3+6_3": 4.70,
        "4_1+8_3": 16.01,
        "8_3+8_3": 17.24,
        "6_3+8_3": 26.48,
        "4_1+4_1+8_3": 51.78,
    }
    actual = {a.name: a.mass_GeV for a in AMPHICHIRAL_DATA}
    assert actual == expected


def test_amphichiral_crossings_are_sums():
    expected_c = {
        "6_3+6_3": 12,
        "6_3+6_3+6_3": 18,
        "4_1+8_3": 12,
        "8_3+8_3": 16,
        "6_3+8_3": 14,
        "4_1+4_1+8_3": 16,
    }
    for a in AMPHICHIRAL_DATA:
        assert a.c == expected_c[a.name], f"Mismatch for {a.name}: c = {a.c}"


def test_all_Q_phys_zero():
    for a in AMPHICHIRAL_DATA:
        assert a.Q_phys == 0


# ---------------------------------------------------------------------------
# 2. compute_delta_c_min API
# ---------------------------------------------------------------------------

def test_compute_delta_c_min_returns_four_mechanisms_and_total():
    a1 = AMPHICHIRAL_DATA[0]
    a2 = AMPHICHIRAL_DATA[1]
    d = compute_delta_c_min(a1, a2)
    for key in ("Q_phys_min_crossings", "Lk_min_crossings",
                "vassiliev_min", "su2_135_min",
                "delta_c_min_total", "binding_mechanism"):
        assert key in d, f"missing key {key}"


def test_delta_c_min_total_is_max_of_four():
    a1 = AMPHICHIRAL_DATA[0]
    a2 = AMPHICHIRAL_DATA[3]
    d = compute_delta_c_min(a1, a2)
    parts = [d["Q_phys_min_crossings"], d["Lk_min_crossings"],
             d["vassiliev_min"], d["su2_135_min"]]
    assert d["delta_c_min_total"] == max(parts)


# ---------------------------------------------------------------------------
# 3. Mécanisme Q_phys
# ---------------------------------------------------------------------------

def test_Q_phys_non_negative():
    for c_i in (6, 12, 16, 18, 20):
        for c_j in (6, 12, 16, 18, 20):
            assert delta_c_Q_phys(c_i, c_j) >= 0


def test_Q_phys_minimum_floor_is_two():
    # Even for very small c the χ-balance always requires at least 2.
    assert delta_c_Q_phys(0, 0) >= 2
    assert delta_c_Q_phys(1, 1) >= 2


# ---------------------------------------------------------------------------
# 4. Mécanisme Lk (Kanenobu-Murakami constant = 4)
# ---------------------------------------------------------------------------

def test_Lk_mechanism_is_constant_four():
    for c_i in (4, 6, 8, 12, 16):
        for c_j in (4, 6, 8, 12, 16):
            assert delta_c_Lk(c_i, c_j) == 4


# ---------------------------------------------------------------------------
# 5. Mécanisme Vassiliev
# ---------------------------------------------------------------------------

def test_vassiliev_monotonic_in_sum():
    # N(c_i, c_j) = c_i + c_j − 2 doit être croissant en (c_i + c_j)
    prev = -1
    sums = sorted({c_i + c_j for c_i in (6, 8, 12, 14, 16, 18)
                              for c_j in (6, 8, 12, 14, 16, 18)})
    for s in sums:
        # take any decomposition c_i=s//2, c_j=s-s//2
        c_i = s // 2
        c_j = s - c_i
        v = vassiliev_first_non_zero(c_i, c_j)
        assert v == s - 2
        assert v >= prev
        prev = v


def test_vassiliev_formula_examples():
    assert vassiliev_first_non_zero(6, 6) == 10
    assert vassiliev_first_non_zero(8, 8) == 14
    assert vassiliev_first_non_zero(12, 12) == 22


# ---------------------------------------------------------------------------
# 6. Mécanisme SU(2)_135
# ---------------------------------------------------------------------------

def test_su2_135_admissibility_h_equal_h():
    # h_init = h_final = 0 (amphichiraux): admissible
    assert su2_135_fusion_admissibility(0.0, 0.0) is True
    # arbitrary same h ≪ k/2
    assert su2_135_fusion_admissibility(0.5, 0.5) is True


def test_su2_135_admissibility_excessive_h_excluded():
    # h = j(j+1)/(k+2). À k = 135, j_max = 67.5, h_max ≈ 67.5·68.5/137 ≈ 33.7
    assert su2_135_fusion_admissibility(0.0, 100.0) is False


def test_su2_135_mechanism_is_16_at_k_135():
    # 2 · ceil(log2(137)) = 2 · 8 = 16
    assert delta_c_SU2_135(0, 0) == 16
    assert delta_c_SU2_135(100, 100) == 16


# ---------------------------------------------------------------------------
# 7. Énumération paires
# ---------------------------------------------------------------------------

def test_all_pairs_returns_15_cross_and_6_self():
    pairs = all_amphichiral_pairs_delta_c_min()
    assert "cross_pairs" in pairs
    assert "self_pairs" in pairs
    assert len(pairs["cross_pairs"]) == 15
    assert len(pairs["self_pairs"]) == 6


def test_all_pairs_entries_have_required_keys():
    pairs = all_amphichiral_pairs_delta_c_min()
    required = {"name_i", "name_j", "c_i", "c_j",
                "Q_phys_min_crossings", "Lk_min_crossings",
                "vassiliev_min", "su2_135_min",
                "delta_c_min_total", "binding_mechanism"}
    for e in pairs["cross_pairs"] + pairs["self_pairs"]:
        assert required.issubset(e.keys()), f"missing keys: {required - e.keys()}"


# ---------------------------------------------------------------------------
# 8. Constante globale (reproductibilité)
# ---------------------------------------------------------------------------

def test_DELTA_C_MIN_OVERALL_reproducible():
    pairs = all_amphichiral_pairs_delta_c_min()
    all_entries = pairs["cross_pairs"] + pairs["self_pairs"]
    overall = min(e["delta_c_min_total"] for e in all_entries)
    assert overall == DELTA_C_MIN_OVERALL


def test_DELTA_C_MIN_OVERALL_is_22():
    # Verdict V5.21 : Δc_min = 22 (binding by Vassiliev on the lightest
    # pair 6_3+6_3 self-link OR 6_3+6_3 + 4_1+8_3)
    assert DELTA_C_MIN_OVERALL == 22


# ---------------------------------------------------------------------------
# 9. Cas particuliers explicites
# ---------------------------------------------------------------------------

def test_self_pair_6_3_plus_6_3():
    a = next(a for a in AMPHICHIRAL_DATA if a.name == "6_3+6_3")
    d = compute_delta_c_min(a, a)
    assert d["delta_c_min_total"] == 22
    assert d["binding_mechanism"] == "vassiliev_min"


def test_cross_pair_6_3_with_4_1_4_1_8_3():
    a = next(a for a in AMPHICHIRAL_DATA if a.name == "6_3+6_3")
    b = next(a for a in AMPHICHIRAL_DATA if a.name == "4_1+4_1+8_3")
    assert a.c == 12
    assert b.c == 16
    d = compute_delta_c_min(a, b)
    # Vassiliev : 12 + 16 − 2 = 26
    assert d["vassiliev_min"] == 26
    assert d["delta_c_min_total"] == 26
