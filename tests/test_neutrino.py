"""Tests pour theory/neutrino.py — Neutrinos comme reliquats ondulatoires (P18)"""
import math
import pytest

from theory.neutrino import (
    FLAVOR_NAMES, HBAR_C_MEV_FM,
    mass_from_radius, boomerang_radius,
    boomerang_velocity, boomerang_omega,
    oscillation_delta_m2, oscillation_length,
    oscillation_frequency,
)


class TestMassFromRadius:
    """Tests pour la masse par inertie rotationnelle : m = ℏ/(2Rc)."""

    def test_mass_positive(self):
        """m > 0 pour tout rayon > 0."""
        for R in [1.0, 10.0, 100.0]:
            assert mass_from_radius(R) > 0

    def test_mass_decreases_with_radius(self):
        """Plus le rayon est grand, plus la masse est faible."""
        m1 = mass_from_radius(1.0)
        m2 = mass_from_radius(10.0)
        m3 = mass_from_radius(100.0)
        assert m1 > m2 > m3

    def test_mass_radius_consistency(self):
        """mass_from_radius et boomerang_radius sont inverses l'une de l'autre."""
        for R in [5.0, 20.0, 50.0]:
            m = mass_from_radius(R)
            R_back = boomerang_radius(m)
            assert abs(R_back - R) < 1e-10

    def test_mass_formula_exact(self):
        """m = ℏc / (2R) exactement."""
        R = 10.0
        expected = HBAR_C_MEV_FM / (2.0 * R)
        assert abs(mass_from_radius(R) - expected) < 1e-12

    def test_invalid_radius_raises(self):
        """Rayon ≤ 0 lève ValueError."""
        with pytest.raises(ValueError):
            mass_from_radius(0.0)
        with pytest.raises(ValueError):
            mass_from_radius(-5.0)


class TestBoomerangVelocity:
    """Tests pour la vitesse boomerang des neutrinos."""

    def test_boomerang_velocity_is_c(self):
        """v/c = 1.0 pour différentes masses."""
        for m in [0.01, 1.0, 5.0, 50.0]:
            assert boomerang_velocity(m) == 1.0

    def test_boomerang_velocity_invalid_mass(self):
        """Masse ≤ 0 lève ValueError."""
        with pytest.raises(ValueError):
            boomerang_velocity(0.0)
        with pytest.raises(ValueError):
            boomerang_velocity(-1.0)

    def test_boomerang_radius_positive(self):
        """R > 0 pour différentes masses."""
        for m in [0.01, 1.0, 15.5]:
            assert boomerang_radius(m) > 0

    def test_boomerang_radius_decreases_with_mass(self):
        """R diminue avec la masse — plus lourd = plus compact."""
        R1 = boomerang_radius(1.0)
        R2 = boomerang_radius(5.0)
        R3 = boomerang_radius(15.5)
        assert R1 > R2 > R3

    def test_boomerang_omega_positive(self):
        """ω > 0 pour différentes masses."""
        for m in [0.01, 1.0, 15.5]:
            assert boomerang_omega(m) > 0

    def test_boomerang_omega_increases_with_mass(self):
        """ω augmente avec la masse — plus lourd = tourne plus vite."""
        w1 = boomerang_omega(1.0)
        w2 = boomerang_omega(5.0)
        w3 = boomerang_omega(15.5)
        assert w1 < w2 < w3


class TestOscillation:
    """Tests pour l'oscillation de saveur par courbure gravitationnelle."""

    def test_delta_m2_sign(self):
        """Δm²(m1, m2) < 0 si m1 < m2."""
        assert oscillation_delta_m2(1.0, 5.0) < 0

    def test_delta_m2_antisymmetric(self):
        """Δm²(m1, m2) = -Δm²(m2, m1)."""
        dm12 = oscillation_delta_m2(1.0, 5.0)
        dm21 = oscillation_delta_m2(5.0, 1.0)
        assert abs(dm12 + dm21) < 1e-10

    def test_delta_m2_zero_same_mass(self):
        """Δm² = 0 pour masses identiques."""
        assert oscillation_delta_m2(3.0, 3.0) == 0.0

    def test_oscillation_length_positive(self):
        """L_osc > 0 pour masses différentes."""
        L = oscillation_length(1.0, 5.0, 1000.0)
        assert L > 0 and L < float('inf')

    def test_oscillation_length_same_mass_infinite(self):
        """L_osc = ∞ pour même masse (pas d'oscillation)."""
        assert oscillation_length(3.0, 3.0, 1000.0) == float('inf')

    def test_oscillation_length_increases_with_energy(self):
        """L_osc ∝ E — neutrinos plus énergétiques oscillent plus lentement."""
        L_low = oscillation_length(1.0, 5.0, 100.0)
        L_high = oscillation_length(1.0, 5.0, 10000.0)
        assert L_high > L_low

    def test_oscillation_frequency_positive(self):
        """ω > 0 pour masses différentes."""
        omega = oscillation_frequency(1.0, 10.0, 1000.0)
        assert omega > 0

    def test_oscillation_frequency_decreases_with_energy(self):
        """ω ∝ 1/E — haute énergie → oscillation plus lente."""
        w_low = oscillation_frequency(1.0, 5.0, 100.0)
        w_high = oscillation_frequency(1.0, 5.0, 10000.0)
        assert w_low > w_high


class TestCollisionIntegration:
    """Tests d'intégration avec le système de collision."""

    def test_nu_e_in_catalog(self):
        """νe présent dans le catalogue de collision."""
        from theory.collision import build_catalog
        catalog = build_catalog()
        assert "nu_e" in catalog
        assert "nu_e_bar" in catalog

    def test_nu_mu_in_catalog(self):
        """νμ présent dans le catalogue de collision."""
        from theory.collision import build_catalog
        catalog = build_catalog()
        assert "nu_mu" in catalog
        assert "nu_mu_bar" in catalog

    def test_neutrino_charges_zero(self):
        """Tous les neutrinos ont Q=0, B=0."""
        from theory.collision import build_catalog
        catalog = build_catalog()
        for name in ["nu_e", "nu_e_bar", "nu_mu", "nu_mu_bar",
                     "nu_tau", "nu_tau_bar"]:
            assert catalog[name].charge == 0.0
            assert catalog[name].baryon_number == 0.0

    def test_neutrino_knots_empty(self):
        """Les neutrinos P18 n'ont pas de nœud propre (knots=[])."""
        from theory.collision import build_catalog
        catalog = build_catalog()
        assert catalog["nu_e"].knots == []
        assert catalog["nu_e_bar"].knots == []
        assert catalog["nu_mu"].knots == []
        assert catalog["nu_mu_bar"].knots == []

    def test_catalog_size(self):
        """Le catalogue contient ≥ 32 entrées."""
        from theory.collision import build_catalog
        catalog = build_catalog()
        assert len(catalog) >= 32
