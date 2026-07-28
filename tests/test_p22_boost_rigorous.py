"""Tests for theory.p22_boost_rigorous (V5.18)."""
from __future__ import annotations

import math

import pytest

from theory.p22_boost_rigorous import (
    KAPPA_G_M, LAMBDABAR_C_E_M, LAMBDABAR_C_P_M, ETA_P22_DEFAULT,
    hubble, rho_crit, r_vir, n_baryon_in_halo,
    h_halo, h_cosmo, delta_h_halo_to_cosmo,
    omega_squared_halo, omega_squared_cosmo,
    grad_omega_squared_halo_boundary,
    xi_at_halo_boundary, boost_P22_rigorous,
    xi_required_for_target_boost, gap_to_jwst,
    diagnose,
)


def test_kappa_g_is_3p1e_minus_53_m():
    """κ_g dérivé V5 §coupling_kg ≈ 3,12 × 10⁻⁵³ m."""
    assert 3.0e-53 < KAPPA_G_M < 3.3e-53


def test_lambda_compton_electron():
    """λbar_C(électron) = ℏ/(m_e c) ≈ 3,86 × 10⁻¹³ m."""
    assert 3.8e-13 < LAMBDABAR_C_E_M < 3.9e-13


def test_lambda_compton_proton_vastly_smaller():
    """Compton du proton ~ 2 × 10⁻¹⁶ m, ~ 1836× plus petit que l'électron."""
    ratio = LAMBDABAR_C_E_M / LAMBDABAR_C_P_M
    assert 1830 < ratio < 1845


def test_hubble_today():
    """H(z=0) = H0 = 67,4 km/s/Mpc en s⁻¹."""
    H0 = 67.4 * 1000.0 / 3.0857e22
    assert abs(hubble(0.0) - H0) / H0 < 1e-3


def test_hubble_grows_with_z():
    """H(z=10) > H(z=0) en ΛCDM (matière domine à haut z)."""
    assert hubble(10.0) > 10 * hubble(0.0)


def test_rho_crit_today():
    """ρ_crit(0) ≈ 9 × 10⁻²⁷ kg/m³ pour H0 = 67,4."""
    rc = rho_crit(0.0)
    assert 8e-27 < rc < 1e-26


def test_r_vir_decreases_with_z():
    """À M fixé, r_vir ∝ ρ_crit(z)⁻¹/³ → diminue à haut z."""
    M = 1e10
    assert r_vir(M, 10.0) < r_vir(M, 5.0) < r_vir(M, 0.5)


def test_r_vir_grows_as_M13():
    """r_vir ∝ M_h^(1/3) à z fixé."""
    z = 10.0
    r1 = r_vir(1e9, z)
    r2 = r_vir(1e12, z)
    assert abs(math.log10(r2 / r1) / math.log10(1e3) - 1.0 / 3.0) < 1e-3


def test_h_halo_is_extremely_small():
    """h = κ_g · |Ψ|² ≪ 1 dans tout halo cosmologique."""
    for z in (5.0, 10.0, 15.0):
        for M in (1e8, 1e12):
            assert h_halo(M, z) < 1e-40


def test_h_halo_exceeds_h_cosmo():
    """À l'intérieur d'un halo, |Ψ|² > densité moyenne, donc h_halo > h_cosmo.

    Le rapport vaut Δ_vir / Ω_m(z) (puisque h_halo ~ Δ_vir·ρ_crit(z) et
    h_cosmo ~ Ω_m·ρ_crit(z), avec ratio = 200/Ω_m ≈ 200..635 selon z).
    """
    z = 10.0
    M = 1e10
    assert h_halo(M, z) > h_cosmo(z)
    # h_halo / h_cosmo = Δ_vir / Ω_m
    ratio = h_halo(M, z) / h_cosmo(z)
    assert 200 < ratio < 1000  # consistent avec Ω_m de 0.2 à 1


def test_delta_h_positive():
    """Δh = h_halo - h_cosmo > 0 (le halo est surdense)."""
    assert delta_h_halo_to_cosmo(1e10, 10.0) > 0


def test_grad_omega_squared_positive():
    """|∇Ω²| > 0 au bord du halo (gradient bien défini)."""
    assert grad_omega_squared_halo_boundary(1e10, 10.0) > 0


def test_xi_far_below_unity():
    """ξ ≪ 1 dans tout halo cosmologique (V5.18 verdict)."""
    for z in (5.0, 10.0, 12.0, 15.0):
        for M in (1e8, 1e10, 1e12):
            assert xi_at_halo_boundary(M, z) < 1e-70


def test_boost_indistinct_from_unity():
    """boost = exp(η ξ) ≈ 1 à toute précision raisonnable."""
    for z in (8.0, 10.0, 12.0):
        for M in (1e9, 1e10, 1e11):
            b = boost_P22_rigorous(M, z)
            assert abs(b - 1.0) < 1e-10


def test_boost_strictly_above_one():
    """exp(positive xi) > 1 même quand le résultat numérique arrondit à 1."""
    for z in (10.0, 12.0):
        xi = xi_at_halo_boundary(1e10, z)
        assert xi > 0


def test_xi_required_for_jwst():
    """ξ requis pour boost = 10³ vaut ln(1000) ≈ 6,9."""
    xi_req = xi_required_for_target_boost(1e3, eta=1.0)
    assert abs(xi_req - math.log(1e3)) < 1e-10


def test_gap_log10_huge():
    """L'écart entre ξ rigoureux et ξ requis dépasse 70 décades."""
    g = gap_to_jwst(1e10, 10.0)
    assert g["log10_gap"] > 70  # gap astronomique


def test_diagnose_returns_dataclass():
    """diagnose() fournit toutes les quantités intermédiaires."""
    d = diagnose(1e10, 10.0)
    assert d.z == 10.0
    assert d.M_h_Msun == 1e10
    assert d.r_vir_m > 0
    assert d.n_baryon_m3 > 0
    assert d.h_halo > 0
    assert d.xi > 0
    assert d.boost > 0


def test_eta_zero_gives_boost_one():
    """η = 0 ⇒ boost = exp(0) = 1 exactement."""
    assert boost_P22_rigorous(1e10, 10.0, eta=0.0) == 1.0


def test_eta_unphysically_huge_could_amplify():
    """Sanity-check : seule une η ~ 10⁸⁰ amplifierait à 10³.

    Cela montre qu'il n'y a PAS de calibration physique réaliste de η
    qui ferme l'écart sans casser l'usage de η ~ 1 dans toutes les
    autres prédictions V5 (résidu DE, muon, etc.).
    """
    xi = xi_at_halo_boundary(1e10, 10.0)
    eta_needed = math.log(1e3) / xi
    assert eta_needed > 1e75  # totalement non-physique
