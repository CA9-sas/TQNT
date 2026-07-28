"""Tests pour P20 : constante de structure fine emergente."""
import numpy as np
from theory.chern_simons import (
    fine_structure_constant, cs_level_from_alpha,
    wkb_potential_torus,
)
from theory.postulates import K_CS


class TestFineStructureConstant:
    """Test alpha = 1/(k+2) from Chern-Simons (standard normalization)."""

    def test_alpha_at_k135(self):
        alpha = fine_structure_constant(K_CS)
        alpha_exp = 1 / 137.036
        rel_error = abs(alpha - alpha_exp) / alpha_exp
        assert rel_error < 0.001, f"alpha={alpha:.6f}, error={rel_error:.4f}"

    def test_alpha_at_k135_value(self):
        alpha = fine_structure_constant(135)
        expected = 1.0 / 137
        assert abs(alpha - expected) < 1e-12

    def test_cs_level_from_alpha(self):
        k = cs_level_from_alpha(1 / 137.036)
        assert k == 135

    def test_cs_level_roundtrip(self):
        alpha = fine_structure_constant(135)
        k_back = cs_level_from_alpha(alpha)
        assert k_back == 135

    def test_k_cs_constant(self):
        assert K_CS == 135


class TestWKBPotential:
    """Test WKB potential for torus knots."""

    def test_trefoil_wkb_shape(self):
        x, V = wkb_potential_torus(3)  # T(2,3)
        # V = -2*s^2*x^2 = -18*x^2
        assert len(x) == 200
        assert len(V) == 200
        # At x=0, V=0
        mid = len(x) // 2
        assert abs(V[mid]) < 0.1
        # V is negative away from x=0
        assert V[0] < 0
        assert V[-1] < 0

    def test_wkb_formula(self):
        s = 5  # T(2,5)
        x, V = wkb_potential_torus(s, n_points=100)
        V_expected = -2 * s**2 * x**2
        np.testing.assert_allclose(V, V_expected, atol=1e-12)

    def test_wkb_parabolic(self):
        """WKB potential is a downward parabola (harmonic-like)."""
        x, V = wkb_potential_torus(3, n_points=50, x_range=(-1.0, 1.0))
        # Fit V = a*x^2 + b*x + c
        coeffs = np.polyfit(x, V, 2)
        # Leading coefficient should be -2*s^2 = -18
        assert abs(coeffs[0] - (-18.0)) < 1e-10
