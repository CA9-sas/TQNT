"""Tests de validation pour theory/knot_listing.py."""
import math

import pytest

from theory.jones import jones_polynomial, jones_span, jones_determinant
from theory.knot_listing import (
    torus_jones_2q,
    torus_alexander_det,
    torus_rho_j,
    enumerate_torus_knots,
    compute_knot_properties,
    generate_full_listing,
)


# --- torus_jones_2q ---

class TestTorusJones2q:
    """Verifie que torus_jones_2q reproduit les polynomes du catalogue."""

    @pytest.mark.parametrize("name,n", [
        ("3_1", 3),
        ("5_1", 5),
        ("7_1", 7),
        ("9_1", 9),
    ])
    def test_matches_catalog(self, name, n):
        """V_{T(2,n)} correspond aux entrees existantes du catalogue."""
        expected = jones_polynomial(name)
        computed = torus_jones_2q(n)
        assert set(computed.keys()) == set(expected.keys()), (
            f"Exposants differents pour T(2,{n}): "
            f"computed={sorted(computed.keys())}, expected={sorted(expected.keys())}"
        )
        for power in expected:
            assert computed[power] == pytest.approx(expected[power]), (
                f"Coefficient t^{power} pour T(2,{n}): "
                f"computed={computed[power]}, expected={expected[power]}"
            )

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11, 13, 15, 21, 39])
    def test_span_equals_n(self, n):
        """span(V_{T(2,n)}) = n."""
        poly = torus_jones_2q(n)
        assert jones_span(poly) == n

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 11, 13, 15, 21, 39])
    def test_determinant_equals_n(self, n):
        """|V_{T(2,n)}(-1)| = n."""
        poly = torus_jones_2q(n)
        assert jones_determinant(poly) == n

    def test_rejects_even(self):
        with pytest.raises(ValueError):
            torus_jones_2q(4)

    def test_rejects_small(self):
        with pytest.raises(ValueError):
            torus_jones_2q(1)


# --- torus_alexander_det ---

class TestTorusAlexanderDet:
    """Verifie le determinant analytique des noeuds toriques."""

    def test_t2_q(self):
        """T(2,q) : det = q."""
        for q in [3, 5, 7, 9, 11, 25, 39]:
            assert torus_alexander_det(2, q) == q

    def test_p_odd_q_even(self):
        """T(p,q) avec p impair, q pair : det = p."""
        assert torus_alexander_det(3, 4) == 3
        assert torus_alexander_det(5, 6) == 5
        assert torus_alexander_det(3, 8) == 3

    def test_both_odd(self):
        """T(p,q) avec p et q impairs : det = 1."""
        assert torus_alexander_det(3, 5) == 1
        assert torus_alexander_det(5, 7) == 1
        assert torus_alexander_det(3, 7) == 1


# --- enumerate_torus_knots ---

class TestEnumerateTorusKnots:
    def test_count_40(self):
        """41 noeuds toriques avec c <= 40."""
        knots = enumerate_torus_knots(40)
        assert len(knots) == 41

    def test_all_coprime(self):
        """Tous les (p,q) sont copremiers."""
        for k in enumerate_torus_knots(40):
            assert math.gcd(k["p"], k["q"]) == 1

    def test_crossing_bound(self):
        """Tous les croisements sont <= 40."""
        for k in enumerate_torus_knots(40):
            assert k["crossing_number"] <= 40

    def test_includes_known_torus(self):
        """Inclut les noeuds toriques connus."""
        knots = enumerate_torus_knots(40)
        names = {k["name"] for k in knots}
        assert "T(2,3)" in names   # trefoil
        assert "T(2,5)" in names   # 5_1
        assert "T(2,7)" in names   # 7_1
        assert "T(2,9)" in names   # 9_1
        assert "T(6,7)" in names   # le plus gros avec p=6


# --- compute_knot_properties ---

class TestComputeKnotProperties:
    def test_rho_j_monotone_t2(self):
        """rho_J croit avec le nombre de croisements pour T(2,q)."""
        rhos = []
        for q in [3, 5, 7, 9, 11, 13]:
            props = compute_knot_properties(f"T(2,{q})", 2, q, n_points=1000)
            rhos.append(props["rho_J"])
        for i in range(len(rhos) - 1):
            assert rhos[i] < rhos[i + 1], (
                f"rho_J non monotone: rho[{i}]={rhos[i]} >= rho[{i+1}]={rhos[i+1]}"
            )

    def test_spin_fermion_t2(self):
        """Tous les T(2,q) ont spin 1/2."""
        for q in [3, 5, 7, 9, 11, 13, 15, 21, 39]:
            props = compute_knot_properties(f"T(2,{q})", 2, q, n_points=500)
            assert props["spin"] == 0.5

    def test_spin_boson_t3(self):
        """Tous les T(3,q) ont spin 1."""
        for q in [4, 5, 7]:
            props = compute_knot_properties(f"T(3,{q})", 3, q, n_points=500)
            assert props["spin"] == 1.0

    def test_catalog_knot(self):
        """Un noeud du catalogue a toutes ses proprietes."""
        props = compute_knot_properties("3_1", 2, 3, n_points=1000)
        assert props["has_jones"] is True
        assert props["crossing_number"] == 3
        assert props["spin"] == 0.5
        assert props["rho_J"] > 0
        assert props["mass_mev"] > 0
        assert props["L"] > 0


# --- generate_full_listing ---

class TestGenerateFullListing:
    @pytest.fixture(scope="class")
    def listing(self):
        """Cache le listing (couteux) pour la classe."""
        return generate_full_listing(max_crossing=40, n_points=1000)

    def test_existing_knots_included(self, listing):
        """Les 20 noeuds du catalogue existant sont presents."""
        from theory.jones import JONES_KNOT_NAMES
        listing_names = {k["name"] for k in listing}
        for name in JONES_KNOT_NAMES:
            assert name in listing_names, f"{name} manquant du listing"

    def test_sorted_by_crossing(self, listing):
        """Le listing est trie par nombre de croisements."""
        crossings = [k["crossing_number"] for k in listing]
        assert crossings == sorted(crossings)

    def test_p16_flag_present(self, listing):
        """Tous les noeuds ont un flag 'excluded'."""
        for k in listing:
            assert "excluded" in k

    def test_total_count(self, listing):
        """Le listing contient environ 57 noeuds (20 catalogue + ~37 toriques)."""
        assert len(listing) >= 50
        assert len(listing) <= 70
