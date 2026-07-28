"""Tests de validation pour theory/abraham.py — Formule d'Abraham nodale."""
import pytest
import numpy as np

from theory.abraham import (
    frenet_geometry,
    geometric_integrals,
    sigma_phi,
    abraham_mass,
    all_knots_abraham,
    convergence_study,
    fary_milnor_ratio,
    tight_c_factor,
    fary_milnor_analysis,
    fary_milnor_summary,
    ABRAHAM_TABLE,
    ALPHA_CS,
    ETA,
    REFERENCE_N50K,
)
from theory.jones import JONES_KNOT_NAMES, bridge_number


# ============================================================
# TestFrenetGeometry
# ============================================================
class TestFrenetGeometry:
    """Tests de la géométrie différentielle de Frenet-Serret."""

    def test_trefoil_kappa_positive(self):
        """κ ≥ 0 partout sur le trèfle."""
        geo = frenet_geometry("3_1", 5000)
        assert np.all(geo["kappa"] >= -1e-10)

    def test_unknot_constant_kappa(self):
        """Cercle (unknot) : κ = 1/R constant (R = 1)."""
        geo = frenet_geometry("0_1", 5000)
        kappa = geo["kappa"]
        # Le cercle unitaire a κ = 1
        assert np.allclose(kappa, 1.0, atol=0.01), (
            f"κ moyen = {kappa.mean():.4f}, écart-type = {kappa.std():.4f}"
        )

    def test_unknot_zero_torsion(self):
        """Cercle (unknot) : τ = 0 (courbe plane)."""
        geo = frenet_geometry("0_1", 5000)
        tau = geo["tau"]
        assert np.allclose(tau, 0.0, atol=0.01), (
            f"τ moyen = {tau.mean():.6f}, max|τ| = {np.abs(tau).max():.6f}"
        )


# ============================================================
# TestGeometricIntegrals
# ============================================================
class TestGeometricIntegrals:
    """Tests des intégrales de ligne."""

    def test_trefoil_L_reference(self):
        """L(3_1) ≈ 28.83 (±1%)."""
        gi = geometric_integrals("3_1", 10000)
        assert gi["L"] == pytest.approx(28.83, rel=0.01), (
            f"L = {gi['L']:.2f}"
        )

    def test_trefoil_int_kappa_reference(self):
        """∮κds(3_1) ≈ 13.95 (±2%)."""
        gi = geometric_integrals("3_1", 10000)
        assert gi["int_kappa"] == pytest.approx(13.95, rel=0.02), (
            f"∮κds = {gi['int_kappa']:.2f}"
        )

    def test_all_knots_L_positive(self):
        """L > 0 pour tous les nœuds du catalogue."""
        for name in JONES_KNOT_NAMES:
            gi = geometric_integrals(name, 2000)
            assert gi["L"] > 0, f"L({name}) = {gi['L']}"

    def test_unknot_arc_length(self):
        """L(0_1) = 2πR = 2π pour le cercle unitaire."""
        gi = geometric_integrals("0_1", 5000)
        assert gi["L"] == pytest.approx(2 * np.pi, rel=0.005), (
            f"L = {gi['L']:.4f}, attendu = {2*np.pi:.4f}"
        )


# ============================================================
# TestSigmaPhi
# ============================================================
class TestSigmaPhi:
    """Tests de la tension EM ΣΦ."""

    def test_trefoil_sigma_phi(self):
        """ΣΦ(3_1) ≈ 15.9 (±2%)."""
        sp = sigma_phi("3_1", 10000)
        assert sp["sigma_phi"] == pytest.approx(15.9, rel=0.02), (
            f"ΣΦ = {sp['sigma_phi']:.2f}"
        )

    def test_trefoil_n_osc(self):
        """Nombre d'oscillations raisonnable pour le trèfle (≥3)."""
        sp = sigma_phi("3_1", 10000)
        assert sp["n_osc"] >= 3, f"n_osc = {sp['n_osc']}"

    def test_sigma_phi_positive(self):
        """ΣΦ > 0 pour tous les nœuds non-triviaux."""
        for name in ABRAHAM_TABLE:
            sp = sigma_phi(name, 2000)
            assert sp["sigma_phi"] > 0, f"ΣΦ({name}) = {sp['sigma_phi']}"


# ============================================================
# TestAbrahamMass
# ============================================================
class TestAbrahamMass:
    """Tests de la formule de masse d'Abraham."""

    def test_electron_mass(self):
        """m_e = (4/3)×α×ρ_J×ΣΦ ≈ 0.511 MeV (±1%)."""
        res = abraham_mass("3_1", (4, 3), "sigma_phi", 10000)
        assert res["mass_predicted"] == pytest.approx(0.511, rel=0.01), (
            f"m_pred = {res['mass_predicted']:.4f} MeV"
        )

    def test_all_knots_error_below_2pct(self):
        """Erreur < 2% pour tous les nœuds (tolérance élargie pour N=10000)."""
        results = all_knots_abraham(10000)
        for r in results:
            assert r["error_pct"] < 2.0, (
                f"{r['knot']}: erreur = {r['error_pct']:.2f}%"
            )

    def test_alpha_cs_value(self):
        """α = 1/(k+2) = 1/137 (normalisation standard CS)."""
        assert ALPHA_CS == pytest.approx(1.0 / 137, rel=1e-10)
        assert 1 / ALPHA_CS == pytest.approx(137.0, rel=1e-10)


# ============================================================
# TestConvergence
# ============================================================
class TestConvergence:
    """Tests de convergence numérique."""

    def test_L_converges(self):
        """L se stabilise pour N croissant."""
        study = convergence_study("3_1", [500, 2000, 10000])
        L_vals = study["values"]["L"]
        # La variation entre les 2 derniers N doit être < 0.1%
        rel_change = abs(L_vals[-1] - L_vals[-2]) / abs(L_vals[-1])
        assert rel_change < 0.001, (
            f"Variation L : {rel_change*100:.3f}%"
        )

    def test_relative_error_decreases(self):
        """L'erreur relative décroît quand N augmente."""
        study = convergence_study("3_1", [200, 1000, 5000, 10000])
        errors = study["errors"]["L"]
        # Chaque erreur doit être ≤ la précédente (monotonement décroissante)
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] + 0.01, (
                f"Erreur L non décroissante : {errors}"
            )


# ============================================================
# TestFaryMilnor
# ============================================================
class TestFaryMilnor:
    """Tests de l'analyse de Fáry-Milnor — nombres de ponts et plongements tendus."""

    # --- bridge_number ---

    def test_bridge_number_trefoil(self):
        """br(3_1) = 2."""
        assert bridge_number("3_1") == 2

    def test_bridge_number_unknot(self):
        """br(0_1) = 1."""
        assert bridge_number("0_1") == 1

    def test_bridge_number_torus_knots(self):
        """Tous les nœuds toriques T(2,q) ont br = 2."""
        for name in ["3_1", "5_1", "7_1", "9_1"]:
            assert bridge_number(name) == 2, f"br({name}) != 2"

    def test_bridge_number_unknown_raises(self):
        """ValueError pour un nœud inconnu."""
        with pytest.raises(ValueError):
            bridge_number("99_99")

    # --- tight_c_factor ---

    def test_tight_c_factor_br2(self):
        """C_K^tight(br=2) ≈ 1.691."""
        c = tight_c_factor(2)
        assert c == pytest.approx(1.691, rel=0.01), f"C_tight(2) = {c:.4f}"

    def test_tight_c_factor_br3_ratio(self):
        """C_K^tight(br=3) = (2/3) × C_K^tight(br=2)."""
        c2 = tight_c_factor(2)
        c3 = tight_c_factor(3)
        assert c3 == pytest.approx(c2 * 2 / 3, rel=1e-10)

    def test_tight_c_factor_invalid_raises(self):
        """ValueError si br < 1."""
        with pytest.raises(ValueError):
            tight_c_factor(0)

    # --- Fáry-Milnor bound (using REFERENCE_N50K for speed) ---

    def test_fm_bound_trefoil(self):
        """∮κds(3_1) = 13.95 ≥ 4π = 12.57 (Fáry-Milnor satisfait)."""
        int_kappa = REFERENCE_N50K["3_1"]["int_kappa"]
        fm_bound = 2 * np.pi * 2  # br = 2
        assert int_kappa >= fm_bound * 0.995, (
            f"∮κds = {int_kappa:.2f} < FM bound = {fm_bound:.2f}"
        )

    def test_fm_bound_all_19_knots(self):
        """Fáry-Milnor satisfait pour les 19 nœuds (données N=50K)."""
        for knot_name, ref in REFERENCE_N50K.items():
            br = bridge_number(knot_name)
            fm_bound = 2 * np.pi * br
            assert ref["int_kappa"] >= fm_bound * 0.995, (
                f"{knot_name}: ∮κds = {ref['int_kappa']:.2f} < FM = {fm_bound:.2f}"
            )

    # --- fary_milnor_analysis ---

    def test_analysis_returns_19(self):
        """L'analyse retourne 19 résultats."""
        results = fary_milnor_analysis(n_points=2000)
        assert len(results) == 19

    def test_analysis_sorted_by_fK(self):
        """Résultats triés par f_K croissant."""
        results = fary_milnor_analysis(n_points=2000)
        f_K_vals = [r["f_K"] for r in results]
        assert f_K_vals == sorted(f_K_vals)

    def test_analysis_fK_positive(self):
        """f_K > 0 pour tous les nœuds."""
        results = fary_milnor_analysis(n_points=2000)
        for r in results:
            assert r["f_K"] > 0, f"{r['knot']}: f_K = {r['f_K']}"

    def test_analysis_all_br2_same_c_tight(self):
        """Tous les nœuds (tous br=2) ont le même C_K^tight."""
        results = fary_milnor_analysis(n_points=2000)
        c_values = [r["C_K_tight"] for r in results]
        assert all(c == pytest.approx(c_values[0], rel=1e-10) for c in c_values)

    # --- eta / alpha ---

    def test_eta_over_alpha(self):
        """η/α ≈ 21.24."""
        ratio = ETA / ALPHA_CS
        assert ratio == pytest.approx(21.24, rel=0.01), f"η/α = {ratio:.2f}"
