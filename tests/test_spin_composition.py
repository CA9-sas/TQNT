"""Tests pour P7bis : composition de spin des entrelacs multi-noeuds."""
import pytest
from theory.jones import braid_index, is_half_integer_spin, topological_spin
from theory.multi_knot import (
    link_spin, link_spin_parity, SM_ASSIGNMENTS,
    FERMION_GENERATIONS, generation_count,
)


# --- Fermions SM (spin total demi-entier) ---
_FERMION_NAMES = {"e-", "u", "d", "nu_tau", "s", "mu-", "c", "tau-", "b", "t"}
# --- Bosons SM (spin total entier) ---
_BOSON_NAMES = {"W+", "Z0", "H"}


class TestIsHalfIntegerSpin:
    """Test is_half_integer_spin on individual knots."""

    def test_torus_knots_are_half_integer(self):
        for name in ["3_1", "5_1", "7_1", "9_1"]:
            assert is_half_integer_spin(name), f"{name} should be half-integer (b=2)"

    def test_b3_knots_are_integer(self):
        for name in ["4_1", "5_2", "6_1", "6_2", "7_2", "7_4", "8_1", "8_2", "10_1"]:
            assert not is_half_integer_spin(name), f"{name} should be integer (b=3)"

    def test_b4_knots_are_half_integer(self):
        for name in ["6_3", "7_3", "7_5", "7_6", "7_7", "8_3"]:
            assert is_half_integer_spin(name), f"{name} should be half-integer (b=4)"


class TestLinkSpinParity:
    """Test spin parity theorem (P7bis)."""

    def test_single_fermion(self):
        assert link_spin_parity(["3_1"]) == "half-integer"

    def test_two_fermions_give_boson(self):
        assert link_spin_parity(["3_1", "5_1"]) == "integer"

    def test_three_fermions_give_fermion(self):
        assert link_spin_parity(["3_1", "5_1", "7_1"]) == "half-integer"

    def test_single_boson(self):
        assert link_spin_parity(["4_1"]) == "integer"

    def test_mixed_odd_half(self):
        # 1 half-integer (3_1) + 1 integer (4_1) = 1 half → half-integer
        assert link_spin_parity(["3_1", "4_1"]) == "half-integer"


class TestSpinComposition:
    """Test link_spin (ground state spin)."""

    def test_single_torus_spin_half(self):
        assert link_spin(["3_1"]) == 0.5

    def test_two_torus_boson_spin_1(self):
        # 2 half-integer → even → boson with fermionics → spin 1
        assert link_spin(["3_1", "5_1"]) == 1.0

    def test_pure_boson_max_spin(self):
        # All integer spin knots → max of individual spins
        assert link_spin(["4_1"]) == 1.0  # b=3, s=1
        assert link_spin(["4_1", "5_2"]) == 1.0  # both s=1

    def test_fermion_composite_spin_half(self):
        # 3 half-integer → odd → fermion → s=1/2
        assert link_spin(["3_1", "5_1", "7_1"]) == 0.5


class TestSMConsistency:
    """Test that all SM assignments respect P7bis spin parity."""

    @pytest.mark.parametrize("particle", SM_ASSIGNMENTS,
                             ids=[p["name"] for p in SM_ASSIGNMENTS])
    def test_spin_parity_consistent(self, particle):
        name = particle["name"]
        parity = link_spin_parity(particle["knots"])
        if name in _FERMION_NAMES:
            assert parity == "half-integer", (
                f"{name} is a fermion but has {parity} spin parity"
            )
        else:
            assert parity == "integer", (
                f"{name} is a boson but has {parity} spin parity"
            )


class TestKnotSwaps:
    """Test that the 5 knot swaps conserve masses within ±5%."""

    def test_all_masses_within_tolerance(self):
        from theory.multi_knot import sm_mass_table
        table = sm_mass_table()
        for row in table:
            ratio = row["ratio"]
            assert abs(ratio - 1.0) < 0.05, (
                f"{row['name']}: ratio={ratio:.4f} exceeds ±5% tolerance"
            )


class TestGenerations:
    """Test P7ter : 3 generations fermioniques."""

    def test_generation_count(self):
        assert generation_count() == 3

    def test_fermion_generations_content(self):
        assert FERMION_GENERATIONS == [(3, "3_1"), (5, "5_1"), (7, "7_1")]


# ════════════════════════════════════════════════════════════════════
#  Tests P7quater — spectre complet de spin par couplage Clebsch-Gordan
# ════════════════════════════════════════════════════════════════════

from theory.multi_knot import link_spin_set, link_spin_level


class TestLinkSpinSet:
    """Spectre Clebsch-Gordan : J ∈ {P, P+1, …, S_max}."""

    def test_single_b2_knot(self):
        # 3_1 a b=2 → s=1/2, donc J_set = {1/2}
        assert link_spin_set(["3_1"]) == {0.5}

    def test_single_b3_knot(self):
        # 4_1 a b=3 → s=1, donc J_set = {0, 1}
        assert link_spin_set(["4_1"]) == {0.0, 1.0}

    def test_single_b4_knot(self):
        # 6_3 a b=4 → s=3/2, donc J_set = {1/2, 3/2}
        assert link_spin_set(["6_3"]) == {0.5, 1.5}

    def test_two_fermions_give_integer_set(self):
        # 3_1+3_1 : S_max = 1, J ∈ {0, 1}
        assert link_spin_set(["3_1", "3_1"]) == {0.0, 1.0}

    def test_proton_config(self):
        # 3_1 (s=1/2) + 7_2 (s=1) + 7_2 (s=1) : S_max = 5/2, J ∈ {1/2, 3/2, 5/2}
        assert link_spin_set(["3_1", "7_2", "7_2"]) == {0.5, 1.5, 2.5}

    def test_higgs_config(self):
        # 6_3+8_2+8_3 : s = 3/2+1+3/2 = 4, J ∈ {0, 1, 2, 3, 4}
        assert link_spin_set(["6_3", "8_2", "8_3"]) == {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_empty(self):
        assert link_spin_set([]) == {0.0}

    def test_parity_constraint(self):
        # Pour toute config, P = S_max mod 1 doit appartenir à J_set
        for cfg in [["3_1"], ["3_1", "4_1"], ["4_1", "4_1", "4_1"],
                    ["3_1", "7_2", "7_2"], ["6_3", "8_2", "8_3"]]:
            Js = link_spin_set(cfg)
            s_max = sum(topological_spin(k) for k in cfg)
            parity = s_max - int(s_max)
            assert parity in Js
            assert s_max in Js
            assert all((s_max - J) == int(s_max - J) for J in Js)


class TestLinkSpinLevel:
    """Étiquettes physiques de la chaîne CG."""

    def test_singulet_J0(self):
        assert link_spin_level(["4_1", "4_1"], 0.0) == "singulet"

    def test_fundamental_J_half(self):
        # Proton J=1/2 → état fondamental fermionique
        assert link_spin_level(["3_1", "7_2", "7_2"], 0.5) == "fondamental"

    def test_stretched_top(self):
        # Stretched : J = S_max
        proton = ["3_1", "7_2", "7_2"]
        s_max = sum(topological_spin(k) for k in proton)
        assert link_spin_level(proton, s_max) == "stretched"

    def test_orbital_excitation_L1(self):
        # 3_1+7_2+7_2 (S_max=5/2) : J=3/2 est L=1 (J=2.5 serait stretched)
        assert link_spin_level(["3_1", "7_2", "7_2"], 1.5) == "L=1"

    def test_invalid_J_raises(self):
        with pytest.raises(ValueError):
            link_spin_level(["3_1"], 1.5)  # J=3/2 hors spectre {1/2}


class TestBackwardCompatibility:
    """link_spin (P7bis) doit toujours retourner un J ∈ link_spin_set (P7quater)."""

    def test_fundamental_in_set(self):
        for cfg in [["3_1"], ["4_1"], ["3_1", "5_1"], ["4_1", "4_1"],
                    ["3_1", "7_2", "7_2"], ["6_1", "7_3", "7_7"]]:
            assert link_spin(cfg) in link_spin_set(cfg)
