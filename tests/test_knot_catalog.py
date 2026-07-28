# tests/test_knot_catalog.py
import numpy as np
from theory.knot_catalog import (
    knot_curve,
    distance_to_knot,
    KnotInfo,
    knot_info_table,
    crossing_number,
    arc_length,
    construction_wavelength,
)
from theory.jones import JONES_KNOT_NAMES


def test_unknot_is_circle():
    pts = knot_curve("0_1", n_points=100)
    assert pts.shape == (100, 3)
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    np.testing.assert_allclose(radii, 1.0, atol=0.01)
    np.testing.assert_allclose(pts[:, 2], 0.0, atol=0.01)


def test_trefoil_is_closed():
    pts = knot_curve("3_1", n_points=500)
    np.testing.assert_allclose(pts[0], pts[-1], atol=0.1)


def test_distance_to_knot_zero_on_curve():
    pts = knot_curve("3_1", n_points=200)
    test_point = pts[50]
    d = distance_to_knot(test_point, pts)
    assert d < 0.1


def test_distance_to_knot_positive_off_curve():
    far_point = np.array([100.0, 100.0, 100.0])
    pts = knot_curve("3_1", n_points=200)
    d = distance_to_knot(far_point, pts)
    assert d > 10.0


def test_knot_info_table_populated():
    table = knot_info_table()
    assert len(table) >= 15
    for info in table:
        assert info.name
        assert info.determinant > 0
        assert info.degree >= 0


def test_knot_info_sorted_by_retraction():
    table = knot_info_table()
    retractions = [k.retraction for k in table]
    assert retractions == sorted(retractions)


# --- Tests longueur d'onde minimale de construction ---

def test_crossing_number():
    assert crossing_number("3_1") == 3
    assert crossing_number("10_1") == 10
    assert crossing_number("0_1") == 0


def test_arc_length_positive():
    for name in JONES_KNOT_NAMES:
        L = arc_length(name)
        assert L > 0, f"arc_length({name}) should be > 0"


def test_arc_length_increases_torus():
    """Nœuds toriques T(2,p) : longueur croît avec p."""
    L3 = arc_length("3_1")
    L5 = arc_length("5_1")
    L7 = arc_length("7_1")
    L9 = arc_length("9_1")
    assert L3 < L5 < L7 < L9


def test_construction_wavelength_positive():
    for name in JONES_KNOT_NAMES:
        if crossing_number(name) > 0:
            lam = construction_wavelength(name)
            assert lam > 0, f"construction_wavelength({name}) should be > 0"


def test_unknot_wavelength_infinite():
    assert construction_wavelength("0_1") == float('inf')


def test_wavelength_decreases_torus():
    """λ décroît pour les toriques : L croît mais c croît plus vite."""
    lam3 = construction_wavelength("3_1")
    lam5 = construction_wavelength("5_1")
    lam7 = construction_wavelength("7_1")
    assert lam3 > lam5 > lam7
