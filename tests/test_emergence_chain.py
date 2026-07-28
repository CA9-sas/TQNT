"""Tests pour la résolution partielle de la Faille 2 (Potentiel disconnect).

Teste la chaîne d'émergence CS → V_phenom et la corrélation ρ_J vs WKB.
"""
import pytest
import numpy as np
from theory.chern_simons import (
    emergence_chain,
    rho_vs_wkb_correlation,
    compare_potentials,
    _TORUS_KNOTS,
)


class TestEmergenceChain:
    """Test de la chaîne d'émergence CS → V_phenom."""

    @pytest.mark.parametrize("knot_name", ["3_1", "4_1", "5_1", "7_1"])
    def test_chain_has_4_steps(self, knot_name):
        result = emergence_chain(knot_name)
        assert len(result["chain_steps"]) == 4

    @pytest.mark.parametrize("knot_name", ["3_1", "5_1", "7_1", "9_1"])
    def test_torus_knots_have_wkb(self, knot_name):
        result = emergence_chain(knot_name)
        assert result["is_torus"] is True
        assert "wkb_coefficient" in result

    def test_non_torus_no_wkb(self):
        result = emergence_chain("4_1")
        assert result["is_torus"] is False
        assert "wkb_coefficient" not in result

    def test_rho_positive(self):
        for knot in ["3_1", "4_1", "5_1"]:
            result = emergence_chain(knot)
            assert result["rho_J"] > 0

    def test_limitations_documented(self):
        """Les limitations honnêtes sont documentées."""
        result = emergence_chain("3_1")
        assert len(result["limitations"]) >= 3
        # Check key limitation about 1/d² not being derived
        lim_text = " ".join(result["limitations"])
        assert "motivée" in lim_text or "motivee" in lim_text

    def test_conformal_argument_present(self):
        result = emergence_chain("3_1")
        assert "conformal_argument" in result
        assert "1/d²" in result["conformal_argument"] or "1/d" in result["conformal_argument"]

    def test_trefoil_wkb_coefficient(self):
        """T(2,3) : WKB coefficient = 2×3² = 18."""
        result = emergence_chain("3_1")
        assert result["wkb_coefficient"] == 18

    def test_formula_contains_rho(self):
        result = emergence_chain("3_1")
        rho = result["rho_J"]
        assert f"{rho:.4f}" in result["V_phenom_formula"]


class TestRhoVsWkbCorrelation:
    """Test de la corrélation ρ_J vs coefficient WKB."""

    def test_all_torus_knots_present(self):
        result = rho_vs_wkb_correlation()
        assert len(result["torus_knots"]) == len(_TORUS_KNOTS)

    def test_perfect_rank_correlation(self):
        """ρ_J et 2s² doivent avoir une corrélation de rang parfaite.

        Les deux croissent monotonement avec s, donc Spearman = 1.0.
        """
        result = rho_vs_wkb_correlation()
        assert result["spearman"] > 0.99

    def test_high_pearson_correlation(self):
        """Pearson doit être élevé (pas forcément 1, car relation non-linéaire)."""
        result = rho_vs_wkb_correlation()
        assert result["correlation"] > 0.95

    def test_wkb_coefficients_correct(self):
        """Vérifier les coefficients WKB = 2s²."""
        result = rho_vs_wkb_correlation()
        for entry in result["torus_knots"]:
            s = entry["s"]
            assert entry["wkb_coefficient"] == 2 * s**2

    def test_interpretation_present(self):
        result = rho_vs_wkb_correlation()
        assert "interpretation" in result
        assert len(result["interpretation"]) > 50


class TestComparePotentialsDocstring:
    """Vérifier que la docstring de compare_potentials est mise à jour."""

    def test_docstring_mentions_different_spaces(self):
        """La docstring doit mentionner que les potentiels vivent dans des espaces différents."""
        docstring = compare_potentials.__doc__
        assert "character variety" in docstring
        assert "R³" in docstring

    def test_docstring_mentions_limitations(self):
        """La docstring doit mentionner que la corrélation n'implique pas l'identité."""
        docstring = compare_potentials.__doc__
        assert "n'implique PAS" in docstring or "ne signifie PAS" in docstring

    def test_compare_potentials_still_works(self):
        """compare_potentials() doit toujours fonctionner après la mise à jour."""
        result = compare_potentials("3_1")
        assert "correlation" in result
        assert "V_cs" in result
        assert "V_retraction" in result
