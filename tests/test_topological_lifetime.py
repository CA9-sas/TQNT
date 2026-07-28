"""Tests pour la formulation algébrique de la durée de vie (Cas 4)."""
import numpy as np
import pytest
from fractions import Fraction


class TestStructuralStability:
    def test_s_positive_all_knots(self):
        """S(K) > 0 pour les 19 nœuds non-triviaux."""
        from theory.topological_lifetime import structural_stability
        from theory.jones import JONES_KNOT_NAMES

        for name in JONES_KNOT_NAMES:
            if name == "0_1":
                continue
            result = structural_stability(name)
            assert result["S_K"] > 0, f"S({name}) = {result['S_K']} <= 0"
            assert result["rho_J"] > 0
            assert result["U"] > 0

    def test_trefoil_more_stable_than_8_3(self):
        """Le trèfle (simple) est plus stable que 8_3 (complexe)."""
        from theory.topological_lifetime import structural_stability

        s_trefoil = structural_stability("3_1")
        s_8_3 = structural_stability("8_3")
        assert s_trefoil["S_K"] > s_8_3["S_K"]

    def test_s_uses_rho_squared_over_u(self):
        """Vérification explicite S(K) = ρ_J² / U sur le trèfle."""
        from theory.topological_lifetime import structural_stability

        result = structural_stability("3_1")
        expected = result["rho_J"] ** 2 / result["U"]
        assert result["S_K"] == pytest.approx(expected, rel=1e-10)

    def test_s_uses_reference_n50k(self):
        """Avec n_points=50000, utilise REFERENCE_N50K (sqrt_U²)."""
        from theory.topological_lifetime import structural_stability
        from theory.abraham import REFERENCE_N50K

        result = structural_stability("3_1", n_points=50000)
        sqrt_U_ref = REFERENCE_N50K["3_1"]["sqrt_U"]
        assert result["U"] == pytest.approx(sqrt_U_ref ** 2, rel=1e-10)

    def test_s_link_positive_all_particles(self):
        """S_link > 0 pour les 13 particules SM."""
        from theory.topological_lifetime import link_structural_stability
        from theory.multi_knot import SM_ASSIGNMENTS

        for p in SM_ASSIGNMENTS:
            result = link_structural_stability(p["knots"])
            assert result["S_link"] > 0, f"S_link({p['name']}) <= 0"
            assert result["n_components"] == len(p["knots"])


class TestAlgebraicExponents:
    def test_nu_weak_is_5pi_over_2(self):
        from theory.topological_lifetime import NU_WEAK_ALG
        assert NU_WEAK_ALG == pytest.approx(5 * np.pi / 2, abs=1e-10)

    def test_nu_strong_is_7_over_4(self):
        from theory.topological_lifetime import NU_STRONG_ALG
        assert NU_STRONG_ALG == Fraction(7, 4)
        assert float(NU_STRONG_ALG) == 1.75

    def test_lambda_strong_is_7_over_4(self):
        from theory.topological_lifetime import LAMBDA_STRONG_ALG
        assert LAMBDA_STRONG_ALG == Fraction(7, 4)
        assert float(LAMBDA_STRONG_ALG) == 1.75


class TestAlgebraicLifetime:
    def test_muon_calibration(self):
        """Le muon est le point de calibration faible → erreur < 1%."""
        from theory.topological_lifetime import algebraic_lifetime, TAU_MU
        from theory.multi_knot import SM_ASSIGNMENTS

        mu_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "mu-")
        tau = algebraic_lifetime(mu_cfg["knots"], mu_cfg["k"], "weak")
        assert tau == pytest.approx(TAU_MU, rel=0.01)

    def test_w_calibration(self):
        """Le W est le point de calibration fort → erreur < 1%."""
        from theory.topological_lifetime import algebraic_lifetime, TAU_W
        from theory.multi_knot import SM_ASSIGNMENTS

        w_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "W+")
        tau = algebraic_lifetime(w_cfg["knots"], w_cfg["k"], "strong")
        assert tau == pytest.approx(TAU_W, rel=0.01)

    def test_all_within_one_decade(self):
        """Toutes les 10 particules avec τ_exp : |log₁₀(τ_alg/τ_exp)| < 1.0."""
        from theory.topological_lifetime import sm_lifetime_table_algebraic

        table = sm_lifetime_table_algebraic()
        for row in table:
            if row["log_ratio_alg"] is not None:
                assert abs(row["log_ratio_alg"]) < 1.0, (
                    f"{row['name']}: |Δlog| = {abs(row['log_ratio_alg']):.3f} >= 1.0"
                )

    def test_rms_below_030(self):
        """RMS ≤ 0.30 décades (marge sur les 0.239 calculés)."""
        from theory.topological_lifetime import sm_lifetime_table_algebraic

        table = sm_lifetime_table_algebraic()
        log_ratios = [
            row["log_ratio_alg"]
            for row in table
            if row["log_ratio_alg"] is not None
        ]
        rms = np.sqrt(np.mean(np.array(log_ratios) ** 2))
        assert rms <= 0.30, f"RMS = {rms:.3f} > 0.30"

    def test_channel_auto_detection(self):
        """Le canal est correctement auto-détecté pour les SM_ASSIGNMENTS."""
        from theory.topological_lifetime import (
            algebraic_lifetime,
            STRONG_DECAY,
        )
        from theory.multi_knot import SM_ASSIGNMENTS

        for p in SM_ASSIGNMENTS:
            tau_auto = algebraic_lifetime(p["knots"], p["k"], "auto")
            expected_channel = "strong" if p["name"] in STRONG_DECAY else "weak"
            tau_explicit = algebraic_lifetime(
                p["knots"], p["k"], expected_channel
            )
            assert tau_auto == pytest.approx(tau_explicit, rel=1e-12), (
                f"{p['name']}: auto vs explicit mismatch"
            )


class TestComparison:
    def test_rms_no_major_regression(self):
        """RMS algébrique ≤ RMS P15 + 0.05 (pas de régression majeure)."""
        from theory.topological_lifetime import stability_summary

        summary = stability_summary()
        assert summary["rms_alg"] <= summary["rms_p15"] + 0.05, (
            f"rms_alg={summary['rms_alg']:.3f} > rms_p15={summary['rms_p15']:.3f} + 0.05"
        )

    def test_table_complete(self):
        """La table complète retourne 13 entrées."""
        from theory.topological_lifetime import sm_lifetime_table_algebraic

        table = sm_lifetime_table_algebraic()
        assert len(table) == 13
        names = {r["name"] for r in table}
        assert "e-" in names
        assert "t" in names
        assert "H" in names
