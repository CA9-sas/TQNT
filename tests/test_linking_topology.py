"""Tests pour theory/linking_topology.py — Topologie des rencontres onde-onde"""
import numpy as np
import pytest

from theory.linking_topology import (
    gauss_linking_integral,
    circular_loop,
    encounter_geometry,
    classify_encounter,
    scan_encounter_space,
    linking_cross_section_effective,
)


class TestGaussLinkingIntegral:
    """Tests pour l'intégrale de Gauss du linking number."""

    def test_unlinked_concentric(self):
        """Deux cercles concentriques dans le même plan → ℓ = 0."""
        c1 = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]))
        c2 = circular_loop(2.0, np.array([0, 0, 0]), np.array([0, 0, 1]))
        assert gauss_linking_integral(c1, c2) == 0

    def test_unlinked_separated(self):
        """Deux cercles très éloignés → ℓ = 0."""
        c1 = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]))
        c2 = circular_loop(1.0, np.array([100, 0, 0]), np.array([0, 0, 1]))
        assert gauss_linking_integral(c1, c2) == 0

    def test_hopf_link_positive(self):
        """Hopf link classique → |ℓ| = 1.

        c1 dans le plan xy, c2 dans le plan xz décalée en x=1 :
        c2 traverse le disque de c1 une fois (à l'intérieur) → Hopf link.
        """
        c1 = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]))
        c2 = circular_loop(1.0, np.array([1, 0, 0]), np.array([0, 1, 0]))
        assert abs(gauss_linking_integral(c1, c2)) == 1

    def test_hopf_link_reversed(self):
        """Hopf link inversé (orientation opposée) → ℓ = −1."""
        c1 = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]))
        c2 = circular_loop(1.0, np.array([1, 0, 0]), np.array([0, 1, 0]))
        lk = gauss_linking_integral(c1, c2)
        # Reversed curve
        lk_rev = gauss_linking_integral(c1, c2[::-1])
        assert lk_rev == -lk

    def test_solomon_link(self):
        """Solomon link (2 tours) → |ℓ| = 2."""
        # Two circles linked with winding number 2:
        # Circle 1 in xy plane, Circle 2 winds twice through it
        n = 400
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Standard circle
        c1 = circular_loop(2.0, np.array([0, 0, 0]), np.array([0, 0, 1]), n)
        # Curve that links c1 twice: a (2,1) torus knot on a torus around z-axis
        R_major = 2.0
        R_minor = 0.8
        c2 = np.column_stack([
            (R_major + R_minor * np.cos(2 * t)) * np.cos(t),
            (R_major + R_minor * np.cos(2 * t)) * np.sin(t),
            R_minor * np.sin(2 * t),
        ])
        assert abs(gauss_linking_integral(c1, c2)) == 2

    def test_numerical_precision(self):
        """La valeur brute de l'intégrale est proche d'un entier (< 0.01)."""
        c1 = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]), 300)
        c2 = circular_loop(1.0, np.array([1, 0, 0]), np.array([0, 1, 0]), 300)
        # Compute raw (non-rounded) value
        dg1 = np.roll(c1, -1, axis=0) - c1
        dg2 = np.roll(c2, -1, axis=0) - c2
        mid1 = c1 + 0.5 * dg1
        mid2 = c2 + 0.5 * dg2
        r = mid1[:, np.newaxis, :] - mid2[np.newaxis, :, :]
        r_norm = np.linalg.norm(r, axis=2)
        r_norm_cubed = np.where(r_norm ** 3 < 1e-30, 1e-30, r_norm ** 3)
        cross = np.cross(dg1[:, np.newaxis, :], dg2[np.newaxis, :, :])
        integrand = np.sum(r * cross, axis=2) / r_norm_cubed
        lk_raw = np.sum(integrand) / (4.0 * np.pi)
        assert abs(lk_raw - round(lk_raw)) < 0.01


class TestCircularLoop:
    """Tests pour circular_loop."""

    def test_points_at_distance_R(self):
        """Tous les points sont à distance R du centre."""
        R = 2.5
        center = np.array([1.0, 2.0, 3.0])
        c = circular_loop(R, center, np.array([0, 0, 1]))
        distances = np.linalg.norm(c - center, axis=1)
        np.testing.assert_allclose(distances, R, atol=1e-12)

    def test_points_in_plane(self):
        """Tous les points sont dans le plan normal au vecteur donné."""
        normal = np.array([1.0, 1.0, 1.0])
        normal = normal / np.linalg.norm(normal)
        center = np.array([0, 0, 0])
        c = circular_loop(1.0, center, normal)
        # Dot product with normal should be ~0
        dots = (c - center) @ normal
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_correct_number_of_points(self):
        """Le nombre de points retournés correspond à n_points."""
        c = circular_loop(1.0, np.array([0, 0, 0]), np.array([0, 0, 1]), 100)
        assert c.shape == (100, 3)


class TestEncounterGeometry:
    """Tests pour encounter_geometry."""

    def test_encounter_hopf(self):
        """θ=π/2, b=R, φ=π/2 : c2 traverse le disque de c1 → Hopf link (|ℓ|=1).

        Avec cette géométrie, normal2=(1,0,0), center2=(0,1,0).
        c2 passe par (0,0,0) à z=0 (intérieur du disque de c1) → lié.
        """
        c1, c2 = encounter_geometry(1.0, 1.0, np.pi / 2, 1.0, np.pi / 2)
        lk = gauss_linking_integral(c1, c2)
        assert abs(lk) == 1

    def test_perpendicular_far_unlinked(self):
        """θ=π/2, b >> R → non lié (ℓ=0)."""
        c1, c2 = encounter_geometry(1.0, 1.0, np.pi / 2, 10.0, 0.0)
        lk = gauss_linking_integral(c1, c2)
        assert lk == 0

    def test_phi_symmetry(self):
        """φ → φ + π ne change pas |ℓ|."""
        R = 1.0
        theta = np.pi / 2
        b = 1.0
        c1a, c2a = encounter_geometry(R, R, theta, b, np.pi / 2)
        c1b, c2b = encounter_geometry(R, R, theta, b, np.pi / 2 + np.pi)
        lk_a = gauss_linking_integral(c1a, c2a)
        lk_b = gauss_linking_integral(c1b, c2b)
        assert abs(lk_a) == abs(lk_b)

    def test_curve_shapes(self):
        """Les courbes retournées ont la bonne forme."""
        c1, c2 = encounter_geometry(1.0, 2.0, 0.5, 0.3, 0.1)
        assert c1.shape == (200, 3)
        assert c2.shape == (200, 3)


class TestClassifyEncounter:
    """Tests pour classify_encounter."""

    def test_unlinked(self):
        """Grande séparation → topology 'unlinked'."""
        result = classify_encounter(1.0, 1.0, np.pi / 2, 10.0, 0.0)
        assert result["topology"] == "unlinked"
        assert result["linked"] is False
        assert result["linking_number"] == 0

    def test_hopf(self):
        """θ=π/2, b=R, φ=π/2 → topology 'hopf'."""
        result = classify_encounter(1.0, 1.0, np.pi / 2, 1.0, np.pi / 2)
        assert result["topology"] == "hopf"
        assert result["linked"] is True
        assert abs(result["linking_number"]) == 1


class TestScanEncounterSpace:
    """Tests pour scan_encounter_space."""

    def test_fraction_linked_range(self):
        """La fraction liée est dans [0, 1]."""
        result = scan_encounter_space(1.0, 1.0, n_theta=10, n_b=10, n_phi=4)
        assert 0 <= result["fraction_linked"] <= 1

    def test_large_b_unlinked(self):
        """Pour b → ∞ effectif, la fraction liée tend vers 0."""
        # Use very small R so that b_max = 3*R is still small
        # and most configs are unlinked
        result = scan_encounter_space(0.1, 0.1, n_theta=5, n_b=10, n_phi=4)
        # At least some are unlinked
        assert 0 in result["distribution"]

    def test_grid_shapes(self):
        """Les grilles ont les bonnes dimensions."""
        result = scan_encounter_space(1.0, 1.0, n_theta=5, n_b=7, n_phi=3)
        assert len(result["theta_grid"]) == 5
        assert len(result["b_grid"]) == 7
        assert len(result["phi_grid"]) == 3
        assert result["linking_map"].shape == (5, 7, 3)


class TestEffectiveCrossSection:
    """Tests pour linking_cross_section_effective."""

    def test_upper_bound(self):
        """σ_eff ≤ π(R₁+R₂)² (borne supérieure géométrique)."""
        R1, R2 = 1.0, 1.0
        sigma = linking_cross_section_effective(R1, R2, n_samples=500)
        sigma_max = np.pi * (R1 + R2) ** 2
        assert sigma <= sigma_max * 1.05  # 5% tolerance for MC noise

    def test_positive_for_equal_radii(self):
        """σ_eff > 0 pour R₁ ≈ R₂."""
        sigma = linking_cross_section_effective(1.0, 1.0, n_samples=500)
        assert sigma > 0
