"""Tests V5.8 — Dérivation des équations d'Einstein depuis TQNT-V5.

Vérifie :
  1. Cohérence κ_g ↔ G_Newton à <0,5 % de CODATA.
  2. Métrique conforme g_μν = Ω² η_μν.
  3. Symboles de Christoffel pour métrique conformément plate.
  4. Tenseur de Ricci/Einstein dans le cas conforme.
  5. Tenseur stress-énergie T_μν du χ-field.
  6. Limite Newton : ∇²Φ = 4πGρ.
  7. Limite Schwarzschild (jauge isotrope, faible champ).
  8. Lien dimensionnel κ_g = (8π m_p / c²) · G.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.einstein_emergence import (
    C_LIGHT,
    G_CODATA,
    HBAR,
    M_PROTON,
    EinsteinCheck,
    G_from_kappa_g_consistency,
    christoffel_symbols_conformal,
    conformal_factor_h,
    einstein_tensor_V5,
    energy_density_from_psi2,
    gravitational_potential_from_h,
    metric_tensor_V5,
    newton_consistency_pointmass,
    newton_limit_check,
    ricci_scalar_conformal,
    ricci_tensor_conformal,
    schwarzschild_limit_check,
    summary,
    T_munu_chi_field,
    verify_einstein_equation,
)
from theory.coupling_kg import G_newton_from_kappa_g, kappa_g_V5p5


# ---------------------------------------------------------------------------
# 1. G prédit ↔ CODATA.
# ---------------------------------------------------------------------------
def test_G_predicted_within_half_percent_codata():
    """G dérivé doit être à <0,5 % de CODATA (V5.5 étape 6)."""
    G = G_newton_from_kappa_g()
    rel = abs(G - G_CODATA) / G_CODATA
    assert rel < 5e-3, f"G écart {rel*100:.3f}% > 0,5%"


def test_G_predicted_matches_prompt_value():
    """G écart proche de +0,42 % selon le prompt V5.8."""
    G = G_newton_from_kappa_g()
    pct = (G - G_CODATA) / G_CODATA * 100
    assert 0.3 < pct < 0.5, f"G écart {pct:+.3f}% hors plage [0.3, 0.5]"


# ---------------------------------------------------------------------------
# 2. Métrique conforme.
# ---------------------------------------------------------------------------
def test_metric_flat_when_psi2_zero():
    """Ω² = 1 quand |Ψ|² = 0 → g_μν = η_μν."""
    g = metric_tensor_V5(0.0)
    eta = np.diag([-1, 1, 1, 1]).astype(float)
    np.testing.assert_allclose(g, eta, atol=1e-15)


def test_metric_signature_minkowski():
    """Signature (-,+,+,+) par défaut."""
    g = metric_tensor_V5(0.5)
    assert g[0, 0] < 0
    assert g[1, 1] > 0
    assert g[2, 2] > 0
    assert g[3, 3] > 0


def test_metric_conformal_proportional_to_eta():
    """g_μν = Ω² η_μν → g/Ω² = η_μν."""
    h = 0.3
    g = metric_tensor_V5(h)
    eta = np.diag([-1, 1, 1, 1]).astype(float)
    np.testing.assert_allclose(g / (1.0 + h), eta, atol=1e-12)


def test_conformal_factor_h_positive():
    """Ω² > 0 pour h ≥ 0 (matière densifie)."""
    assert conformal_factor_h(0.0) == 1.0
    assert conformal_factor_h(0.5) == 1.5
    assert conformal_factor_h(1e-39) == pytest.approx(1.0 + 1e-39, rel=1e-20)


# ---------------------------------------------------------------------------
# 3. Christoffel.
# ---------------------------------------------------------------------------
def test_christoffel_zero_for_flat():
    """Si ∂_μ ln Ω = 0 (espace plat) → Γ^λ_μν = 0."""
    Gamma = christoffel_symbols_conformal(np.zeros(4))
    np.testing.assert_allclose(Gamma, 0.0, atol=1e-15)


def test_christoffel_symmetry_lower_indices():
    """Γ^λ_μν = Γ^λ_νμ (sans torsion)."""
    grad = np.array([0.1, 0.2, -0.3, 0.05])
    Gamma = christoffel_symbols_conformal(grad)
    for lam in range(4):
        np.testing.assert_allclose(Gamma[lam], Gamma[lam].T, atol=1e-14)


def test_christoffel_trace_conformal_dimension():
    """Trace Γ^μ_μν = n · ∂_ν ln Ω + (n-2)/... — en n=4 : 4·∂_ν ln Ω."""
    grad = np.array([0.1, 0.2, -0.3, 0.05])
    Gamma = christoffel_symbols_conformal(grad)
    # Trace : Γ^μ_μν
    for nu in range(4):
        tr = sum(Gamma[mu, mu, nu] for mu in range(4))
        # En conforme n=4 : Γ^μ_μν = n · ∂_ν ln Ω = 4 · grad[nu]
        assert tr == pytest.approx(4.0 * grad[nu], rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Ricci et Einstein.
# ---------------------------------------------------------------------------
def test_ricci_zero_in_flat_space():
    """Ω² = 1, ∂Ω² = 0 → R_μν = 0."""
    R = ricci_tensor_conformal(1.0, np.zeros(4), np.zeros((4, 4)))
    np.testing.assert_allclose(R, 0.0, atol=1e-15)


def test_ricci_symmetric():
    """R_μν = R_νμ."""
    grad = np.array([0.0, 0.01, 0.02, 0.0])
    hess = np.diag([0.0, 0.001, 0.001, 0.001])
    R = ricci_tensor_conformal(1.05, grad, hess)
    np.testing.assert_allclose(R, R.T, atol=1e-14)


def test_einstein_tensor_zero_in_flat():
    """G_μν = 0 dans Minkowski plat."""
    G = einstein_tensor_V5(1.0, np.zeros(4), np.zeros((4, 4)))
    np.testing.assert_allclose(G, 0.0, atol=1e-15)


def test_ricci_scalar_real():
    """R doit être scalaire réel."""
    grad = np.array([0.0, 0.01, 0.0, 0.0])
    hess = np.diag([0.0, 0.005, 0.005, 0.005])
    R = ricci_scalar_conformal(1.02, grad, hess)
    assert isinstance(R, float)
    assert math.isfinite(R)


# ---------------------------------------------------------------------------
# 5. Tenseur stress-énergie.
# ---------------------------------------------------------------------------
def test_T_munu_static_psi_diagonal():
    """Pour Ψ statique sans gradient : T_μν diagonal."""
    T = T_munu_chi_field(psi=1.0, grad_psi=np.zeros(4))
    # T_00 non nul (énergie au repos), T_ii = -L · g_ii (pression)
    assert T[0, 0] != 0
    for mu in range(4):
        for nu in range(4):
            if mu != nu:
                assert abs(T[mu, nu]) < 1e-30


def test_energy_density_formula():
    """ρc² = m c² |Ψ|²."""
    rho_c2 = energy_density_from_psi2(2.0)
    expected = M_PROTON * C_LIGHT ** 2 * 2.0
    assert rho_c2 == pytest.approx(expected, rel=1e-12)


def test_T_munu_zero_for_zero_psi():
    """Ψ = 0, ∂Ψ = 0 → T_μν = 0."""
    T = T_munu_chi_field(psi=0.0, grad_psi=np.zeros(4))
    np.testing.assert_allclose(T, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# 6. Limite Newton : ∇²Φ = 4πGρ.
# ---------------------------------------------------------------------------
def test_newton_limit_check_zero_density():
    """ρ = 0 → 4πGρ = 0."""
    out = newton_limit_check(rho_kg_m3=0.0, lap_phi=0.0)
    assert out["four_pi_G_rho"] == 0.0


def test_newton_limit_check_consistency():
    """Si ∇²Φ = 4πGρ alors rapport = 1."""
    G = G_newton_from_kappa_g()
    rho = 1.0
    lap = 4.0 * math.pi * G * rho
    out = newton_limit_check(rho_kg_m3=rho, lap_phi=lap)
    assert out["ratio"] == pytest.approx(1.0, rel=1e-12)


def test_gravitational_potential_from_h():
    """Φ = -(c²/2) h."""
    h = 1e-9
    phi = gravitational_potential_from_h(h)
    assert phi == pytest.approx(-0.5 * C_LIGHT ** 2 * h, rel=1e-12)


def test_newton_pointmass_earth_surface():
    """Φ(R_earth) = -GM_e/R_e ≈ -6.25×10⁷ m²/s²."""
    M_e = 5.972e24
    R_e = 6.371e6
    out = newton_consistency_pointmass(M_e, R_e)
    assert -6.5e7 < out["phi_codata"] < -6.0e7
    # V5 prédit avec G légèrement différent → écart 0,42 %
    assert abs(out["relative_error"]) < 5e-3


def test_verify_einstein_equation_self_consistent_uniform_sphere():
    """Boule uniforme : ∇²h = -8πGρ/c² → G_00 = (8πG/c⁴) T_00."""
    G = G_newton_from_kappa_g()
    M = 1.989e30
    R = 6.957e8
    rho = 3 * M / (4 * math.pi * R ** 3)

    def h_prof(r):
        return G * M / (R ** 3 * C_LIGHT ** 2) * (3 * R ** 2 - r ** 2)
    def lap_h(r):
        A = G * M / (R ** 3 * C_LIGHT ** 2)
        return -6.0 * A
    def rho_prof(r):
        return rho

    chk = verify_einstein_equation(h_prof, lap_h, rho_prof, r=R / 2)
    # G_00 et rhs_00 doivent être égaux (auto-cohérence Poisson)
    assert chk.ratio_00 == pytest.approx(1.0, rel=1e-10)
    assert chk.poisson_ratio == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 7. Limite Schwarzschild.
# ---------------------------------------------------------------------------
def test_schwarzschild_check_sun():
    """Au rayon du Soleil, GM/(rc²) ≈ 2,1×10⁻⁶ : régime linéaire."""
    out = schwarzschild_limit_check(M=1.989e30, r=6.957e8)
    assert out["regime_valid"]
    # Linéarisation : Ω²_Schw ≈ 1 + 2GM/(rc²)
    assert abs(out["linearization_error"]) < 1e-4


def test_schwarzschild_check_earth():
    """Au rayon de la Terre, GM/(rc²) ≈ 7×10⁻¹⁰ : ultra-linéaire."""
    out = schwarzschild_limit_check(M=5.972e24, r=6.371e6)
    assert out["regime_valid"]
    # Linéarisation : erreur ~ 1,5·ε avec ε ≈ 3,5×10⁻¹⁰, mais sensibilité
    # numérique au calcul de (1+ε)⁴-1 limite à ~10⁻⁶ pour ε si petit.
    assert abs(out["linearization_error"]) < 1e-5


def test_schwarzschild_radius_formula():
    """r_s = 2GM/c² (formule Schwarzschild)."""
    M = 1.989e30
    out = schwarzschild_limit_check(M=M, r=6.957e8)
    G = G_newton_from_kappa_g()
    expected_rs = 2.0 * G * M / C_LIGHT ** 2
    assert out["schwarzschild_radius"] == pytest.approx(expected_rs, rel=1e-12)


def test_schwarzschild_h_required_positive():
    """h doit être positif (matière attire → Ω² > 1)."""
    out = schwarzschild_limit_check(M=1.989e30, r=6.957e8)
    assert out["h_required_for_schw"] > 0


# ---------------------------------------------------------------------------
# 8. Lien κ_g ↔ G : κ_g = (8π m_p / c²) · G.
# ---------------------------------------------------------------------------
def test_kappa_g_from_G_formula_exact():
    """κ_g = (8π m_p / c²) · G — relation exacte V5.5 étape 6.

    Précision limitée par les arrondis CODATA sur (ℏ, c, m_p) ; ~10⁻⁸.
    """
    out = G_from_kappa_g_consistency()
    assert abs(out["relative_error"]) < 1e-7, (
        f"Lien κ_g ↔ G inexact : {out['relative_error']*100:.4f}%"
    )


def test_kappa_g_dimensions_consistent():
    """[κ_g] = m. Vérification dimensionnelle via la relation."""
    G = G_newton_from_kappa_g()  # m³/(kg s²)
    kg_dim = 8.0 * math.pi * M_PROTON * G / (C_LIGHT ** 2)  # m
    kg = kappa_g_V5p5("m")
    assert kg == pytest.approx(kg_dim, rel=1e-7)


# ---------------------------------------------------------------------------
# 9. Synthèse globale.
# ---------------------------------------------------------------------------
def test_summary_returns_all_keys():
    """summary() doit renvoyer toutes les clés clés."""
    s = summary()
    expected = {
        "kappa_g_m", "G_predicted", "G_codata", "G_error_pct",
        "einstein_check_G00", "einstein_check_rhs00", "einstein_check_ratio",
        "poisson_lhs", "poisson_rhs", "poisson_ratio",
        "schw_h_required", "schw_h_linear", "schw_linearization_error",
        "G_to_kappa_g_formula", "G_to_kappa_g_ratio", "G_to_kappa_g_rel_error",
    }
    assert expected.issubset(s.keys())


def test_summary_einstein_self_consistent():
    """summary() rapport Poisson = 1.0 sur profil uniforme."""
    s = summary()
    assert s["poisson_ratio"] == pytest.approx(1.0, rel=1e-10)
    assert s["einstein_check_ratio"] == pytest.approx(1.0, rel=1e-10)


def test_summary_schwarzschild_low_field():
    """Soleil : linéarisation < 0,001 %."""
    s = summary()
    assert abs(s["schw_linearization_error"]) < 1e-5


def test_summary_G_error_below_half_percent():
    """G error < 0,5 % CODATA."""
    s = summary()
    assert abs(s["G_error_pct"]) < 0.5


# ---------------------------------------------------------------------------
# 10. Test optionnel : dérivation sympy (lente).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_symbolic_ricci_isotropic():
    """Le calcul sympy doit produire un Ricci symétrique pour Ω²=1+h(r)."""
    try:
        import sympy as sp
    except ImportError:
        pytest.skip("sympy non installé")
    from theory.einstein_emergence import symbolic_ricci_conformal_isotropic
    res = symbolic_ricci_conformal_isotropic()
    R = res["ricci"]
    # Symétrie R_μν = R_νμ
    for mu in range(4):
        for nu in range(mu + 1, 4):
            diff = sp.simplify(R[mu, nu] - R[nu, mu])
            assert diff == 0, f"R[{mu},{nu}] ≠ R[{nu},{mu}] : {diff}"
