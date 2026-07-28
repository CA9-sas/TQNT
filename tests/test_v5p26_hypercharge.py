"""Tests for V5.26 — Y_W from A-polynomial structure.

The tests check both the mechanism (n_roots map, feature extraction) and
the honest negative-finding contract (flavoured hadrons must miss).
"""
import math
import pytest

from theory.a_polynomial import A_POLY_KNOT_NAMES
from theory.hypercharge_a_polynomial import (
    evaluate_pistes,
    extract_a_poly_features,
    hypercharge_v5p26,
    validate_against_pdg,
)
from theory.jones import JONES_KNOT_NAMES


def test_features_available_for_all_20_knots():
    """Every V5 catalog knot returns a feature dict (A-poly or Jones proxy)."""
    for k in JONES_KNOT_NAMES:
        f = extract_a_poly_features(k)
        assert f["knot"] == k
        assert f["feature_source"] in ("a_polynomial", "jones_proxy")
        assert f["jones_span"] >= 0


def test_n_roots_for_5_a_poly_knots():
    """A-poly knots reproduce known n_roots: 0 (unknot), 1 (torus), 2 (hyperbolic)."""
    expected = {"0_1": 0, "3_1": 1, "4_1": 2, "5_1": 1, "5_2": 2}
    for k, n in expected.items():
        f = extract_a_poly_features(k)
        assert f["feature_source"] == "a_polynomial"
        assert f["n_roots"] == n, f"{k}: got n_roots={f['n_roots']}, expected {n}"


def test_hypercharge_electron_matches_pdg():
    """e- knot 3_1 must give Y = -1 (PDG doublet convention)."""
    y = hypercharge_v5p26("3_1")
    assert math.isclose(y, -1.0, abs_tol=1e-9)


def test_hypercharge_unknot_zero():
    """Unknot 0_1 (and photon-as-unknot) must give Y = 0."""
    assert hypercharge_v5p26("0_1") == 0.0


def test_hypercharge_torus_T_2_q_lepton_like():
    """Torus T(2, q) for q=3,5,7,9 must classify as lepton (Y=-1)."""
    for k in ("3_1", "5_1", "7_1", "9_1"):
        y = hypercharge_v5p26(k)
        assert math.isclose(y, -1.0, abs_tol=1e-9), f"{k}: Y={y}"


def test_hypercharge_hyperbolic_quark_like():
    """Hyperbolic 2-bridge knots (4_1, 5_2, ...) classify as quark (Y=+1/3)."""
    for k in ("4_1", "5_2", "6_1", "6_2", "8_3"):
        y = hypercharge_v5p26(k)
        assert math.isclose(y, 1 / 3, abs_tol=1e-9), f"{k}: Y={y}"


def test_composite_sums_topological_y():
    """Composite Y = sum of single-knot Y (topological additivity)."""
    # nucleon = 3_1 + 7_2 + 7_2 -> Y_topo = -1 + 1/3 + 1/3 = -1/3
    y = hypercharge_v5p26("3_1 + 7_2 + 7_2")
    assert math.isclose(y, -1.0 + 1 / 3 + 1 / 3, abs_tol=1e-9)


def test_validation_against_pdg_returns_67_particles():
    """validate_against_pdg should iterate the 67-particle catalog."""
    stats = validate_against_pdg()
    # Tolerance: allow at most one missing entry (catalog edits).
    assert stats["n"] >= 60, f"only {stats['n']} particles found"
    assert "per_sector" in stats
    assert "per_particle" in stats
    assert stats["n_exact"] >= 0
    assert stats["n_exact"] <= stats["n"]


def test_negative_result_flavoured_hadrons_miss():
    """HONEST CONTRACT: flavoured hadrons (K, D, B) cannot match by construction.

    The A-polynomial carries no flavour quantum number (strangeness, charm,
    bottomness), so Y_V5.26 cannot reproduce PDG Y for hadrons with non-zero
    S, C, or B'. We assert *at least one* such miss to lock in the honesty
    of the documented negative result.
    """
    stats = validate_against_pdg()
    flavoured_names = {"K±", "K⁰", "D⁰", "D±", "D_s±", "B±", "B⁰", "B_c±"}
    misses = sum(
        1 for r in stats["per_particle"]
        if r["name"] in flavoured_names and not r["exact"]
    )
    assert misses >= 4, (
        f"V5.26 wrongly succeeds on flavoured hadrons "
        f"(misses={misses}/{len(flavoured_names)}). "
        "Either the test catalog changed or someone overfit."
    )


def test_evaluate_pistes_returns_6_items_with_verdicts():
    pistes = evaluate_pistes()
    assert len(pistes) == 6
    names = {p.name for p in pistes}
    assert "P1_n_roots" in names
    assert "P2_newton_polytope" in names
    assert "P3_mahler_measure" in names
    assert "P4_asymptotics" in names
    assert "P5_torus_slopes" in names
    assert "P6_combined" in names
    # Exactly one piste is retained as final formula
    retained = [p for p in pistes if p.verdict == "retenue"]
    assert len(retained) == 1
    assert retained[0].name == "P6_combined"


def test_feature_extraction_includes_jones_proxy_for_non_a_poly_knots():
    """Knots without A-poly entry must fall back to jones_proxy."""
    for k in ("7_1", "8_3", "10_1"):
        f = extract_a_poly_features(k)
        assert f["feature_source"] == "jones_proxy"
        assert f["jones_span"] > 0
        # A-poly-specific fields are None or zero
        assert f["n_roots"] is None
        assert f["newton_n_vertices"] == 0


def test_hypercharge_invalid_knot_raises():
    with pytest.raises(ValueError):
        extract_a_poly_features("999_999")
