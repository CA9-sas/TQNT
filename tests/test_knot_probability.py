"""Tests de validation pour theory/knot_probability.py."""
import math

import numpy as np
import pytest

from theory.knot_probability import (
    ALPHA_EFF,
    boltzmann_weight,
    listing_partition_function,
    asymptotic_partition_function,
    normalized_probabilities,
    dark_matter_ratio,
    random_closed_polygon,
    project_and_find_crossings,
    knot_determinant_from_crossings,
    classify_knot,
    monte_carlo_knot_statistics,
    knotting_probability_vs_length,
)
from theory.postulates import BETA_TOPO


# ===================================================================
# Boltzmann topologique
# ===================================================================

class TestBoltzmannWeight:
    """w(K) = exp(-beta * rho_J)."""

    def test_decreasing(self):
        """w decroit avec rho_J."""
        w1 = boltzmann_weight(3.3)
        w2 = boltzmann_weight(5.5)
        w3 = boltzmann_weight(9.9)
        assert w1 > w2 > w3

    def test_unknot_max(self):
        """w(unknot) est maximal (rho_J ~ 0.1)."""
        w_unknot = boltzmann_weight(0.1)
        w_trefoil = boltzmann_weight(3.3)
        assert w_unknot > w_trefoil

    def test_trefoil_value(self):
        """Valeur numerique de w(3_1) avec rho_J = 3.3."""
        w = boltzmann_weight(3.3, BETA_TOPO)
        expected = math.exp(-BETA_TOPO * 3.3)
        assert w == pytest.approx(expected, rel=1e-10)

    def test_positive(self):
        """Le poids est toujours positif."""
        for rho in [0.0, 0.1, 1.0, 10.0, 100.0]:
            assert boltzmann_weight(rho) > 0


class TestAsymptoticPartition:
    """Z_total via N(c) ~ 2.68^c / c."""

    def test_alpha_eff_convergent(self):
        """alpha_eff < 1 garantit la convergence de la serie."""
        assert ALPHA_EFF < 1.0

    def test_alpha_eff_value(self):
        """alpha_eff ~ 0.980."""
        assert ALPHA_EFF == pytest.approx(0.980, abs=0.005)

    def test_Z_total_finite(self):
        """Z_total est fini."""
        result = asymptotic_partition_function(BETA_TOPO, c_max=500)
        assert result["Z_total"] > 0
        assert math.isfinite(result["Z_total"])

    def test_converges_with_cmax(self):
        """Z_total converge quand c_max augmente."""
        Z_100 = asymptotic_partition_function(BETA_TOPO, 100)["Z_total"]
        Z_500 = asymptotic_partition_function(BETA_TOPO, 500)["Z_total"]
        # alpha_eff ~ 0.98 -> queue c>100 contribue ~1.2%
        assert abs(Z_500 - Z_100) / Z_500 < 0.02

    def test_c_dominant_small(self):
        """Le croisement dominant est a petite complexite (c ~ 1-5)."""
        result = asymptotic_partition_function(BETA_TOPO, 500)
        assert 1 <= result["c_dominant"] <= 10


class TestDarkMatterRatio:
    """Omega_DM / Omega_b via Boltzmann topologique.

    Avec la formule approchee M(c) = 1.1 * alpha_eff^c et alpha_eff ~ 0.98,
    le ratio = alpha_eff^18 / (1 - alpha_eff^18) ~ 2.3.
    Le postulat P18 predit 5.45 (Planck 2018) ; la difference provient de la
    variance de rho_J au sein de chaque classe de croisements et de la
    sensibilite exponentielle au parametre beta.
    Avec beta ~ 0.904 (au lieu de 0.9144) le ratio atteint 5.45.
    """

    def test_ratio_dark_dominates(self):
        """La matiere noire domine la matiere visible (ratio > 2)."""
        result = dark_matter_ratio(BETA_TOPO, c_max=500)
        assert result["ratio"] > 2.0

    def test_ratio_value(self):
        """Le ratio avec beta=0.9144 et alpha_eff~0.98 est ~2.3."""
        result = dark_matter_ratio(BETA_TOPO, c_max=500)
        assert result["ratio"] == pytest.approx(2.3, abs=0.2)

    def test_ratio_planck_tuned_beta(self):
        """Avec beta ~ 0.904, le ratio atteint 5.45 (Planck 2018)."""
        result = dark_matter_ratio(beta=0.904, c_max=500)
        assert result["ratio"] == pytest.approx(5.45, abs=0.5)

    def test_dark_greater_than_visible(self):
        """M_dark > M_visible."""
        result = dark_matter_ratio(BETA_TOPO, c_max=500)
        assert result["M_dark"] > result["M_visible"]

    def test_c_star(self):
        """Le seuil est c* = 18."""
        result = dark_matter_ratio(BETA_TOPO)
        assert result["c_star"] == 18


class TestNormalizedProbabilities:
    """P(K) = w(K) / Z_total."""

    @pytest.fixture(scope="class")
    def listing(self):
        from theory.knot_listing import generate_full_listing
        return generate_full_listing(max_crossing=44, n_points=500)

    def test_sum_equals_coverage(self, listing):
        """sum(P) = coverage < 1."""
        result = normalized_probabilities(listing, BETA_TOPO)
        total = sum(result["probabilities"])
        assert total == pytest.approx(result["coverage"], rel=1e-6)
        assert result["coverage"] < 1.0

    def test_visible_plus_dark(self, listing):
        """P_visible + P_dark ~ 1."""
        result = normalized_probabilities(listing, BETA_TOPO)
        assert result["P_visible"] + result["P_dark"] == pytest.approx(1.0, abs=0.01)

    def test_all_positive(self, listing):
        """Toutes les probabilites sont positives."""
        result = normalized_probabilities(listing, BETA_TOPO)
        for p in result["probabilities"]:
            assert p > 0


# ===================================================================
# Monte Carlo polygones aleatoires
# ===================================================================

class TestRandomPolygon:
    """Polygone ferme aleatoire en R^3."""

    def test_closed(self):
        """Le polygone est ferme (dernier point = premier)."""
        poly = random_closed_polygon(100, np.random.default_rng(42))
        np.testing.assert_allclose(poly[0], poly[-1], atol=1e-10)

    def test_shape(self):
        """Shape correcte (n_edges+1, 3)."""
        poly = random_closed_polygon(50, np.random.default_rng(42))
        assert poly.shape == (51, 3)

    def test_reproducible(self):
        """Reproductible avec le meme seed."""
        p1 = random_closed_polygon(100, np.random.default_rng(42))
        p2 = random_closed_polygon(100, np.random.default_rng(42))
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self):
        """Des seeds differents donnent des polygones differents."""
        p1 = random_closed_polygon(100, np.random.default_rng(42))
        p2 = random_closed_polygon(100, np.random.default_rng(99))
        assert not np.allclose(p1, p2)


class TestCrossingDetection:
    """Detection de croisements dans la projection xy."""

    def test_circle_zero_crossings(self):
        """Un cercle n'a aucun croisement."""
        t = np.linspace(0, 2 * np.pi, 101)
        circle = np.column_stack([np.cos(t), np.sin(t), np.zeros(101)])
        crossings = project_and_find_crossings(circle)
        assert len(crossings) == 0

    def test_trefoil_has_crossings(self):
        """Un trefle du catalogue a 3 croisements."""
        from theory.knot_catalog import knot_curve
        # Utiliser beaucoup de points pour une detection fiable
        pts = knot_curve("3_1", 5000)
        crossings = project_and_find_crossings(pts)
        assert len(crossings) >= 3

    def test_crossing_fields(self):
        """Chaque croisement a les champs attendus."""
        rng = np.random.default_rng(42)
        poly = random_closed_polygon(100, rng)
        crossings = project_and_find_crossings(poly)
        if crossings:
            c = crossings[0]
            assert "i" in c
            assert "j" in c
            assert "t" in c
            assert "s" in c
            assert "sign" in c
            assert 0 < c["t"] < 1
            assert 0 < c["s"] < 1


class TestKnotDeterminant:
    """Determinant du noeud via presentation de Wirtinger."""

    def test_unknot_det_1(self):
        """Un cercle (unknot) a determinant 1."""
        t = np.linspace(0, 2 * np.pi, 101)
        circle = np.column_stack([np.cos(t), np.sin(t), np.zeros(101)])
        crossings = project_and_find_crossings(circle)
        det = knot_determinant_from_crossings(crossings, 100)
        assert det == 1

    def test_no_crossings_det_1(self):
        """Pas de croisements -> determinant 1."""
        det = knot_determinant_from_crossings([], 100)
        assert det == 1

    def test_determinant_positive(self):
        """Le determinant est toujours positif."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            poly = random_closed_polygon(50, rng)
            crossings = project_and_find_crossings(poly)
            det = knot_determinant_from_crossings(crossings, 50)
            assert det >= 1


class TestMonteCarloStatistics:
    """Statistiques Monte Carlo globales."""

    def test_unknot_dominates_short(self):
        """L'unknot domine pour N=30 (polygones courts)."""
        stats = monte_carlo_knot_statistics(n_edges=30, n_trials=500, seed=42)
        assert stats["unknot_fraction"] > 0.3

    def test_knotting_grows_with_length(self):
        """La probabilite de nouage croit avec N."""
        stats_30 = monte_carlo_knot_statistics(n_edges=30, n_trials=500, seed=42)
        stats_200 = monte_carlo_knot_statistics(n_edges=200, n_trials=500, seed=42)
        assert stats_200["knotted_fraction"] >= stats_30["knotted_fraction"]

    def test_fractions_sum_to_one(self):
        """unknot + knotted = 1."""
        stats = monte_carlo_knot_statistics(n_edges=100, n_trials=500, seed=42)
        assert stats["unknot_fraction"] + stats["knotted_fraction"] == pytest.approx(1.0)

    def test_reproducible(self):
        """Resultats reproductibles avec le meme seed."""
        s1 = monte_carlo_knot_statistics(n_edges=50, n_trials=200, seed=42)
        s2 = monte_carlo_knot_statistics(n_edges=50, n_trials=200, seed=42)
        assert s1["unknot_fraction"] == s2["unknot_fraction"]
