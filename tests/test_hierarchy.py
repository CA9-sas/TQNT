"""Tests pour la résolution partielle de la Faille 1 (Circularité SM).

Teste les trois prédictions topologiques vérifiables :
  - mass_hierarchy_test() : ρ_total ordonne-t-il les masses ?
  - constraint_multiplicity() : combien de configs valides par particule ?
  - topological_selection_score() : la config SM est-elle parmi les plus simples ?
"""
import pytest
from theory.multi_knot import (
    mass_hierarchy_test,
    constraint_multiplicity,
    topological_selection_score,
    sm_simplicity_ranking,
    SM_ASSIGNMENTS,
)


class TestMassHierarchy:
    """Test de hiérarchie de masse via ρ_total."""

    def test_returns_all_particles(self):
        result = mass_hierarchy_test()
        assert result["n_particles"] == 13

    def test_rho_correlation_positive(self):
        """ρ_total doit corréler positivement avec la masse SM."""
        result = mass_hierarchy_test()
        assert result["rho_rank_vs_mass_rank"] > 0.0

    def test_mraw_ordering_perfect(self):
        """M_raw ordering vs m_SM doit être parfait (par construction via η)."""
        result = mass_hierarchy_test()
        assert result["mraw_rank_vs_mass_rank"] > 0.99
        assert result["mraw_inversions"] == 0

    def test_rho_inversions_bounded(self):
        """ρ_total seul a des inversions (k contribue aussi), mais pas trop."""
        result = mass_hierarchy_test()
        max_inv = result["max_possible_inversions"]
        # ρ_total alone should do better than random (78/2 = 39 inversions)
        assert result["rho_inversions"] < max_inv / 2

    def test_particle_data_complete(self):
        result = mass_hierarchy_test()
        for p in result["particles"]:
            assert "name" in p
            assert "rho_total" in p
            assert "M_raw" in p
            assert "m_sm" in p
            assert p["rho_total"] > 0
            assert p["M_raw"] > 0


class TestConstraintMultiplicity:
    """Test de multiplicité contrainte."""

    def test_returns_all_particles(self):
        result = constraint_multiplicity()
        assert len(result) == 13

    def test_each_particle_has_valid_configs(self):
        """Chaque particule SM doit avoir au moins 1 config valide (la sienne)."""
        result = constraint_multiplicity()
        for entry in result:
            assert entry["n_valid"] >= 1, (
                f"{entry['name']}: n_valid={entry['n_valid']} < 1"
            )

    def test_electron_few_configs(self):
        """L'électron (config simple 3_1, k=1) devrait avoir peu de configs."""
        result = constraint_multiplicity()
        electron = next(e for e in result if e["name"] == "e-")
        # Electron is very light, few configs can match
        assert electron["n_valid"] <= 20

    def test_fermion_boson_classification(self):
        """Les fermions et bosons sont correctement classifiés."""
        result = constraint_multiplicity()
        fermions = {"e-", "u", "d", "s", "mu-", "c", "tau-", "b", "t", "nu_tau"}
        bosons = {"W+", "Z0", "H"}
        for entry in result:
            if entry["name"] in fermions:
                assert entry["is_fermion"] is True
            elif entry["name"] in bosons:
                assert entry["is_fermion"] is False


class TestTopologicalSelectionScore:
    """Test du score de sélection topologique."""

    def test_score_is_positive(self):
        for p in SM_ASSIGNMENTS:
            score = topological_selection_score(p["knots"], p["k"])
            assert score > 0

    def test_simpler_config_lower_score(self):
        """3_1 k=1 doit avoir un score plus bas que 7_1 k=10."""
        s1 = topological_selection_score(["3_1"], 1)
        s2 = topological_selection_score(["7_1"], 10)
        assert s1 < s2

    def test_score_computation(self):
        """Score = sum of crossing numbers + k."""
        # 3_1 has crossing number 3, k=1 → score = 3 + 1 = 4
        assert topological_selection_score(["3_1"], 1) == 4
        # 3_1 + 5_1, k=2 → score = 3 + 5 + 2 = 10
        assert topological_selection_score(["3_1", "5_1"], 2) == 10


class TestSimplicityRanking:
    """Test du ranking de simplicité."""

    def test_returns_all_particles(self):
        result = sm_simplicity_ranking()
        assert len(result) == 13

    def test_sm_score_positive(self):
        result = sm_simplicity_ranking()
        for entry in result:
            assert entry["sm_score"] > 0

    def test_has_top3(self):
        """Chaque particule doit avoir des configs dans le top 3 (si n_valid >= 3)."""
        result = sm_simplicity_ranking()
        for entry in result:
            n_top = min(3, entry["n_valid"])
            assert len(entry["top3_configs"]) <= n_top
