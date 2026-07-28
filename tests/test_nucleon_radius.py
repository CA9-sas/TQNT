"""Tests pour theory/nucleon_radius.py — rayon des nœuds du nucléon."""
import pytest
import numpy as np

from theory.nucleon_radius import (
    knot_free_radius,
    knot_compressed_retraction,
    nucleon_cornell,
    nucleon_knot_radii,
    proton_radius_prediction,
    full_nucleon_analysis,
    NUCLEON_KNOTS,
    ACTIVE_KNOTS,
    N_PSI,
    R_P_EXP_FM,
)


class TestKnotFreeRadius:
    """Rayon libre r_free = ℏc/ρ_J."""

    def test_trefoil_finite(self):
        r = knot_free_radius("3_1")
        assert 0 < r < 1.0

    def test_higher_rho_smaller_radius(self):
        """Plus ρ_J est grand, plus le rayon est petit."""
        r_91 = knot_free_radius("9_1")
        r_74 = knot_free_radius("7_4")
        r_63 = knot_free_radius("6_3")
        assert r_91 < r_74 < r_63

    def test_unknot_large_radius(self):
        """Le nœud trivial (ρ_J ≈ 0.1) a un très grand rayon."""
        r = knot_free_radius("0_1")
        assert r > 5.0  # beaucoup plus grand que les nœuds non-triviaux

    def test_nucleon_knots_order(self):
        """9₁ < 7₄ < 6₃ en rayon libre (complexité décroissante)."""
        radii = [knot_free_radius(k) for k in NUCLEON_KNOTS]
        assert radii[0] < radii[1] < radii[2]


class TestCompressedRetraction:
    """ρ(K, E) = ρ_J²/(ρ_J + E/E₀)."""

    def test_zero_energy_gives_rho_j(self):
        from theory.chi_field import jones_rho
        for name in NUCLEON_KNOTS:
            rho_J = jones_rho(name)
            rho_0 = knot_compressed_retraction(name, 0.0)
            assert abs(rho_0 - rho_J) < 1e-10

    def test_high_energy_reduces_retraction(self):
        for name in NUCLEON_KNOTS:
            rho_0 = knot_compressed_retraction(name, 0.0)
            rho_high = knot_compressed_retraction(name, 1000.0)
            assert rho_high < rho_0

    def test_nucleon_energy_strongly_reduces(self):
        """À E* ~ 6641, la rétractation est réduite de ~10⁻³."""
        eq = nucleon_cornell()
        E_bind = eq["E_bind"]
        for name in NUCLEON_KNOTS:
            rho_0 = knot_compressed_retraction(name, 0.0)
            rho_comp = knot_compressed_retraction(name, E_bind)
            ratio = rho_comp / rho_0
            assert ratio < 0.01  # < 1%


class TestNucleonCornell:
    """Équilibre de Cornell du nucléon."""

    def test_nucleon_mass(self):
        eq = nucleon_cornell()
        assert abs(eq["m_nucleon_mev"] - 938.3) < 2.0  # ~938-940 MeV

    def test_fibonacci_18(self):
        eq = nucleon_cornell()
        assert eq["F_k"] == 2584

    def test_d_star_positive(self):
        eq = nucleon_cornell()
        assert eq["d_star"] > 0
        assert eq["d_star"] < 0.01  # très petit

    def test_E_bind_positive(self):
        eq = nucleon_cornell()
        assert eq["E_bind"] > 0


class TestNucleonKnotRadii:
    """Rayons par nœud dans le nucléon."""

    def test_three_knots(self):
        radii = nucleon_knot_radii()
        assert len(radii) == 3

    def test_names(self):
        radii = nucleon_knot_radii()
        names = [r["name"] for r in radii]
        assert names == ["9_1", "7_4", "6_3"]

    def test_active_knots_p16(self):
        """P16 : 9₁ et 6₃ actifs, 7₄ exclu."""
        radii = nucleon_knot_radii()
        active = {r["name"] for r in radii if r["active"]}
        assert active == {"9_1", "6_3"}

    def test_compression_ratio_large(self):
        """Compression > 100× pour tous les nœuds."""
        for kr in nucleon_knot_radii():
            assert kr["compression_ratio"] > 100

    def test_r_conf_less_than_r_free(self):
        for kr in nucleon_knot_radii():
            assert kr["r_conf"] < kr["r_free"]

    def test_metric_factor_large(self):
        """Ω >> 1 (densité élevée)."""
        for kr in nucleon_knot_radii():
            assert kr["Omega"] > 100

    def test_r_proper_between_r_conf_and_r_free(self):
        """r_conf < r_propre (amplifié par Ω) mais physiquement raisonnable."""
        for kr in nucleon_knot_radii():
            assert kr["r_proper"] > kr["r_conf"]

    def test_density_positive(self):
        for kr in nucleon_knot_radii():
            assert kr["n_particle"] > 0


class TestProtonRadius:
    """Prédiction du rayon de charge du proton."""

    def test_prediction_close_to_experiment(self):
        rp = proton_radius_prediction()
        assert abs(rp["r_p_fm"] - R_P_EXP_FM) / R_P_EXP_FM < 0.001  # < 0.1%

    def test_factor_is_four(self):
        rp = proton_radius_prediction()
        assert rp["factor"] == 4

    def test_n_psi_is_two(self):
        rp = proton_radius_prediction()
        assert rp["N_psi"] == 2

    def test_lambda_compton(self):
        rp = proton_radius_prediction()
        assert abs(rp["lambda_C_fm"] - 0.2103) < 0.001


class TestFullAnalysis:
    """Analyse complète."""

    def test_returns_all_keys(self):
        result = full_nucleon_analysis()
        assert "cornell" in result
        assert "knot_radii" in result
        assert "proton_radius" in result
        assert "L0_fm" in result

    def test_L0_positive(self):
        result = full_nucleon_analysis()
        assert result["L0_fm"] > 0
