"""Tests V5.22 : tentative derivation of Δc_min ≥ 30 (Fermi line search).

Honesty-first: we test the API, the formulas, and the HONEST verdict that
V5.22 falls short of 30 with the current axiomatic mechanisms.
"""
from __future__ import annotations

import pytest

from theory.dc_higher_order_selection import (
    DELTA_C_MIN_V5P22_HONEST,
    DELTA_C_MIN_V5P22_UPPER,
    FERMI_LINE_SEARCH_THRESHOLD,
    V5p22MechanismResult,
    all_amphichiral_pairs_v5p22,
    color_decomposition_steps,
    delta_c_min_total,
    evaluate_all_mechanisms,
    framing_anomaly_addition_crossings,
    higher_vassiliev_primitive_min,
    one_loop_conformal_weight_correction,
    quadrupole_topological_conservation,
)
from theory.dc_selection_rule import (
    AMPHICHIRAL_DATA,
    DELTA_C_MIN_OVERALL,
    delta_c_vassiliev,
)


# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

def test_fermi_threshold_is_30():
    assert FERMI_LINE_SEARCH_THRESHOLD == 30


def test_v5p21_baseline_is_22():
    # Sanity: V5.21 baseline didn't drift.
    assert DELTA_C_MIN_OVERALL == 22


# ---------------------------------------------------------------------------
# Mechanism 1 : primitive Vassiliev
# ---------------------------------------------------------------------------

def test_primitive_vassiliev_exceeds_first_nonzero_by_two():
    # Bar-Natan: lowest primitive ord = c_i + c_j = (c_i + c_j − 2) + 2.
    for c_i, c_j in [(6, 6), (12, 12), (12, 16), (8, 8)]:
        n_v21 = delta_c_vassiliev(c_i, c_j)
        n_prim = higher_vassiliev_primitive_min(c_i, c_j)
        assert n_prim - n_v21 == 2


def test_primitive_vassiliev_formula():
    assert higher_vassiliev_primitive_min(12, 12) == 24
    assert higher_vassiliev_primitive_min(16, 16) == 32


# ---------------------------------------------------------------------------
# Mechanism 2 : RT color decomposition
# ---------------------------------------------------------------------------

def test_rt_color_decomposition_non_axiomatic():
    # Honest flag: crossings ≠ trivalent vertices in RT.
    d = color_decomposition_steps(12, 12)
    assert d["axiomatic"] is False


def test_rt_color_decomposition_positive():
    d = color_decomposition_steps(12, 12)
    assert d["delta_c_RT"] > 0


# ---------------------------------------------------------------------------
# Mechanism 3 : quadrupole Q^(2)
# ---------------------------------------------------------------------------

def test_quadrupole_vanishes_on_amphichirals():
    # Q_phys = 0 on all amphichirals → Q^(2) = 0 → no additional crossings.
    a, b = AMPHICHIRAL_DATA[0], AMPHICHIRAL_DATA[1]
    d = quadrupole_topological_conservation(a, b)
    assert d["delta_c_Q2"] == 0
    assert d["axiomatic"] is False


# ---------------------------------------------------------------------------
# Mechanism 4 : framing anomaly differential
# ---------------------------------------------------------------------------

def test_framing_anomaly_amphichiral_small():
    # |Δh| < 1/2 ⇒ absorbed by canonical framing choice ⇒ Δc=0.
    d = framing_anomaly_addition_crossings(6, 6)
    if abs(d["delta_h"]) < 0.5:
        assert d["delta_c_framing"] == 0 or not d["axiomatic"]


def test_framing_anomaly_returns_expected_keys():
    d = framing_anomaly_addition_crossings(12, 12)
    for key in ("delta_h", "delta_c_framing", "axiomatic", "rationale"):
        assert key in d


# ---------------------------------------------------------------------------
# Mechanism 5 : one-loop conformal weight
# ---------------------------------------------------------------------------

def test_one_loop_cw_is_small():
    # The α/96π suppression makes Δh_loop ≪ 1, so Δc_loop is bounded.
    d = one_loop_conformal_weight_correction(12, 12)
    assert 0 <= d["delta_c_loop"] <= 3
    assert d["axiomatic"] is False


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def test_evaluate_all_returns_five_mechanisms():
    a = AMPHICHIRAL_DATA[0]
    mechs = evaluate_all_mechanisms(a, a)
    assert len(mechs) == 5
    names = {m.name for m in mechs}
    assert names == {
        "primitive_vassiliev",
        "rt_color_decomposition",
        "quadrupole_Q2",
        "framing_anomaly_differential",
        "one_loop_conformal_weight",
    }


def test_only_primitive_vassiliev_is_axiomatic_on_amphi_pair():
    # On the lightest amphi self-link, only M1 should be axiomatic
    # (M2-M5 are explicitly flagged non-axiomatic on amphichirals).
    a = AMPHICHIRAL_DATA[0]
    mechs = evaluate_all_mechanisms(a, a)
    axiomatic = [m for m in mechs if m.axiomatic]
    assert len(axiomatic) == 1
    assert axiomatic[0].name == "primitive_vassiliev"
    assert axiomatic[0].delta_c_additional == 2


def test_delta_c_min_total_keys():
    a, b = AMPHICHIRAL_DATA[0], AMPHICHIRAL_DATA[1]
    d = delta_c_min_total(a, b)
    for k in (
        "v5p21_contribs", "v5p21_binding", "v5p21_total",
        "v5p22_mechanisms", "v5p22_axiomatic_sum", "v5p22_all_sum",
        "delta_c_min_v5p22_honest", "delta_c_min_v5p22_max",
        "binding_mechanism",
    ):
        assert k in d


def test_honest_lt_upper_bound():
    # Honest result is always at most the all-mechanisms upper bound.
    a, b = AMPHICHIRAL_DATA[0], AMPHICHIRAL_DATA[1]
    d = delta_c_min_total(a, b)
    assert d["delta_c_min_v5p22_honest"] <= d["delta_c_min_v5p22_max"]


def test_all_pairs_returns_21_entries():
    pairs = all_amphichiral_pairs_v5p22()
    assert len(pairs) == 21


# ---------------------------------------------------------------------------
# Final verdict (HONEST)
# ---------------------------------------------------------------------------

def test_v5p22_honest_min_is_24():
    # V5.21 binding pair (6_3+6_3 self, Vassiliev=22) gets only +2 from
    # primitive Vassiliev → 24.
    assert DELTA_C_MIN_V5P22_HONEST == 24


def test_v5p22_upper_min_is_at_least_30():
    # The untrusted upper bound clears 30 for binding pairs, but it
    # cannot be invoked physically — recorded for transparency only.
    assert DELTA_C_MIN_V5P22_UPPER >= 30


def test_v5p22_HONEST_FAILS_to_redeem_lines():
    # The honest verdict : V5.22 does NOT reach the Fermi line threshold.
    # F3 falsification remains open.
    assert DELTA_C_MIN_V5P22_HONEST < FERMI_LINE_SEARCH_THRESHOLD
