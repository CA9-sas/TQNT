"""Tests V5.10a — Théorie Kerr-V5 depuis le champ χ (scoping).

Vérifie :
  1. Limite a → 0 : Schwarzschild V5.8 retrouvé.
  2. Asymptote r → ∞ : Minkowski plat.
  3. g_tφ(V5) → g_tφ(Boyer-Lindquist) à grand r.
  4. Cohérence κ_a = -κ_g · a (identification asymptotique).
  5. Ansatz Ψ_rot : symétrie axiale (sin²θ), parité polaire.
  6. Spin topologique : formule S = ℏ(b-1)/2 si impair, ℏb/2 si pair.
  7. Conversion spin topologique → J macroscopique.
  8. Fréquence de frame dragging asymptotique ω ∝ 1/r³.
  9. Schwarzschild radius cohérent avec G_V5.
 10. Summary : tous les drapeaux verts sur trou noir solaire à 0.5 r_g.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.einstein_emergence import C_LIGHT, HBAR, M_PROTON
from theory.kerr_chi_field_theory import (
    G_V5,
    KAPPA_G_M,
    Psi_rot_ansatz,
    Psi_static_squared,
    asymptotic_consistency_check,
    asymptotic_minkowski_limit,
    f_align_from_kerr_a,
    frame_dragging_omega,
    g_tphi_asymptotic,
    g_tphi_boyer_lindquist,
    kappa_a_from_V5,
    kerr_parameter,
    schwarzschild_limit_a_to_zero,
    schwarzschild_radius,
    spin_topological,
    spin_topological_to_J,
    summary,
)


M_SUN = 1.989e30


# ---------------------------------------------------------------------------
# 1. Limite a → 0 ⟹ Schwarzschild V5.8.
# ---------------------------------------------------------------------------
def test_schwarzschild_limit_a_zero():
    """À a = 0 la métrique V5.10a se réduit exactement à V5.8."""
    rg = schwarzschild_radius(M_SUN)
    r = 100.0 * rg
    chk = schwarzschild_limit_a_to_zero(r, M_SUN)
    assert chk["psi2_match"], "Ψ_rot doit égaler Ψ_static à a=0"
    assert chk["g_tphi_zero"], "g_tφ doit être 0 à a=0"
    assert chk["omega_zero"], "ω(frame dragging) doit être 0 à a=0"
    assert chk["kappa_zero"], "κ_a doit être 0 à a=0"
    assert chk["schwarzschild_recovered"]


def test_psi_rot_zero_a_equals_static_at_arbitrary_theta():
    """Pour a = 0, Ψ_rot ne dépend pas de θ et égale Ψ_static."""
    rg = schwarzschild_radius(M_SUN)
    r = 10.0 * rg
    for theta in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi):
        psi_static = Psi_static_squared(r, M_SUN)
        psi_rot = Psi_rot_ansatz(r, theta, M_SUN, a_m=0.0)
        assert math.isclose(psi_rot, psi_static, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. Asymptote r → ∞ ⟹ Minkowski.
# ---------------------------------------------------------------------------
def test_asymptotic_minkowski():
    """À r ≫ r_g la métrique tend vers Minkowski plat (h, g_tφ, ω → 0)."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    chk = asymptotic_minkowski_limit(r_factor=1.0e10, M_kg=M_SUN, a_m=a)
    assert chk["minkowski_recovered"], (
        f"Métrique pas Minkowski : h={chk['h']:.3e}, "
        f"g_tφ_dimless={chk['g_tphi_dimless']:.3e}, "
        f"ω_dimless={chk['omega_dimless']:.3e}"
    )
    assert chk["omega2"] > 1.0
    assert chk["omega2"] - 1.0 < 1e-6


def test_psi_static_decays_as_1_over_r():
    """|Ψ_static|² ∝ 1/r (cohérence avec h = 2GM/(c²r))."""
    psi2_r1 = Psi_static_squared(1.0e8, M_SUN)
    psi2_r2 = Psi_static_squared(2.0e8, M_SUN)
    # ratio doit valoir 2 (1/r décroît d'un facteur 2 quand r double).
    assert math.isclose(psi2_r1 / psi2_r2, 2.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. g_tφ(V5) → g_tφ(BL) à grand r.
# ---------------------------------------------------------------------------
def test_g_tphi_matches_BL_asymptotically():
    """À r ≫ r_g et r ≫ a, g_tφ_V5 = g_tφ_BL à 0.01 % près."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    r = 1000.0 * rg
    chk = asymptotic_consistency_check(r, math.pi / 2, M_SUN, a)
    assert chk.asymptotic_regime
    assert math.isclose(chk.ratio, 1.0, rel_tol=1e-4), (
        f"Rapport g_tφ_V5/g_tφ_BL = {chk.ratio} (devrait être ~1)"
    )


def test_g_tphi_polar_axis_vanishes():
    """Sur l'axe θ = 0 (ou π), g_tφ = 0 par sin²θ (à eps machine)."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    r = 100.0 * rg
    # θ = 0 exactement : sin = 0.
    assert g_tphi_asymptotic(r, 0.0, M_SUN, a) == 0.0
    # θ = π : sin a un eps machine ; on tolère un g_tφ ≪ équateur.
    g_pi = abs(g_tphi_asymptotic(r, math.pi, M_SUN, a))
    g_eq = abs(g_tphi_asymptotic(r, math.pi / 2, M_SUN, a))
    assert g_pi < 1e-20 * max(g_eq, 1.0)


def test_g_tphi_equatorial_maximal():
    """À l'équateur θ = π/2, |g_tφ| est maximal."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    r = 100.0 * rg
    g_eq = abs(g_tphi_asymptotic(r, math.pi / 2, M_SUN, a))
    g_45 = abs(g_tphi_asymptotic(r, math.pi / 4, M_SUN, a))
    assert g_eq > g_45 > 0.0


# ---------------------------------------------------------------------------
# 4. Cohérence κ_a = -κ_g · a.
# ---------------------------------------------------------------------------
def test_kappa_a_identification():
    """κ_a = -κ_g · a (identification depuis Boyer-Lindquist)."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.3 * rg
    kappa_a = kappa_a_from_V5(a)
    expected = -KAPPA_G_M * a
    assert math.isclose(kappa_a, expected, rel_tol=1e-15)


def test_kappa_a_vanishes_at_zero_spin():
    """κ_a(a=0) = 0 (pas de couplage axial sans rotation)."""
    assert kappa_a_from_V5(0.0) == 0.0


# ---------------------------------------------------------------------------
# 5. Ansatz Ψ_rot : symétrie axiale, parité polaire.
# ---------------------------------------------------------------------------
def test_psi_rot_invariant_phi():
    """Ψ_rot ne dépend pas de φ (stationnarité + axisymétrie)."""
    # On ne passe pas de φ : c'est une propriété structurelle de
    # l'ansatz ; ce test contrôle la signature (r, θ, M, a).
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    val = Psi_rot_ansatz(10.0 * rg, math.pi / 3, M_SUN, a)
    assert val > 0.0


def test_psi_rot_polar_axis_no_modulation():
    """Sur l'axe θ = 0 (ou π) : Ψ_rot = Ψ_static (pas de modulation)."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    r = 10.0 * rg
    psi_static = Psi_static_squared(r, M_SUN)
    psi_rot_0 = Psi_rot_ansatz(r, 0.0, M_SUN, a)
    psi_rot_pi = Psi_rot_ansatz(r, math.pi, M_SUN, a)
    assert math.isclose(psi_rot_0, psi_static, rel_tol=1e-12)
    assert math.isclose(psi_rot_pi, psi_static, rel_tol=1e-12)


def test_psi_rot_equatorial_max_modulation():
    """À l'équateur la modulation est maximale (sin²θ = 1)."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    r = 10.0 * rg
    psi_static = Psi_static_squared(r, M_SUN)
    psi_eq = Psi_rot_ansatz(r, math.pi / 2, M_SUN, a)
    psi_45 = Psi_rot_ansatz(r, math.pi / 4, M_SUN, a)
    assert psi_eq > psi_45 > psi_static


# ---------------------------------------------------------------------------
# 6. Spin topologique.
# ---------------------------------------------------------------------------
def test_spin_topological_fermion():
    """b impair ⟹ S = ℏ(b-1)/2 (fermion : 0, 1/2, 3/2, ...)."""
    # b = 1 ⟹ S = 0  (singlet)
    # b = 3 ⟹ S = ℏ
    # b = 5 ⟹ S = 2ℏ
    assert math.isclose(spin_topological(1), 0.0)
    assert math.isclose(spin_topological(3), HBAR)
    assert math.isclose(spin_topological(5), 2.0 * HBAR)


def test_spin_topological_boson():
    """b pair ⟹ S = ℏb/2 (boson : 1, 2, 3, ...)."""
    assert math.isclose(spin_topological(2), HBAR)
    assert math.isclose(spin_topological(4), 2.0 * HBAR)


def test_spin_topological_negative_raises():
    """b < 1 n'a pas de sens topologique."""
    with pytest.raises(ValueError):
        spin_topological(0)


# ---------------------------------------------------------------------------
# 7. Conversion spin → J macroscopique.
# ---------------------------------------------------------------------------
def test_J_macro_proportional_to_M_and_f_align():
    """J_macro ∝ M et ∝ f_align (linéaire dans les deux)."""
    J1 = spin_topological_to_J(b_eff=1.0, M_kg=M_SUN, f_align=0.5)
    J2 = spin_topological_to_J(b_eff=1.0, M_kg=2.0 * M_SUN, f_align=0.5)
    J3 = spin_topological_to_J(b_eff=1.0, M_kg=M_SUN, f_align=1.0)
    assert math.isclose(J2 / J1, 2.0, rel_tol=1e-12)
    assert math.isclose(J3 / J1, 2.0, rel_tol=1e-12)


def test_J_macro_zero_at_no_alignment():
    """f_align = 0 ⟹ J = 0."""
    J = spin_topological_to_J(b_eff=1.0, M_kg=M_SUN, f_align=0.0)
    assert J == 0.0


def test_f_align_out_of_range_raises():
    """f_align hors [0,1] doit lever une erreur."""
    with pytest.raises(ValueError):
        spin_topological_to_J(b_eff=1.0, M_kg=M_SUN, f_align=-0.1)
    with pytest.raises(ValueError):
        spin_topological_to_J(b_eff=1.0, M_kg=M_SUN, f_align=1.5)


# ---------------------------------------------------------------------------
# 8. Frame dragging ω ∝ 1/r³.
# ---------------------------------------------------------------------------
def test_frame_dragging_decays_as_inverse_r_cubed():
    """ω(r) ∝ 1/r³ à grand r."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.5 * rg
    om_r1 = frame_dragging_omega(1000.0 * rg, math.pi / 2, M_SUN, a)
    om_r2 = frame_dragging_omega(2000.0 * rg, math.pi / 2, M_SUN, a)
    # r double ⟹ ω divisé par 8.
    assert math.isclose(om_r1 / om_r2, 8.0, rel_tol=1e-10)


def test_frame_dragging_zero_at_no_spin():
    """ω = 0 si a = 0."""
    rg = schwarzschild_radius(M_SUN)
    om = frame_dragging_omega(100.0 * rg, math.pi / 2, M_SUN, a_m=0.0)
    assert om == 0.0


# ---------------------------------------------------------------------------
# 9. Schwarzschild radius cohérent avec G_V5.
# ---------------------------------------------------------------------------
def test_schwarzschild_radius_uses_G_V5():
    """r_g = G_V5 · M / c² (et G_V5 est CODATA + 0.42 %)."""
    M = 1.0e30
    rg_pred = schwarzschild_radius(M)
    rg_expected = G_V5 * M / (C_LIGHT ** 2)
    assert math.isclose(rg_pred, rg_expected, rel_tol=1e-15)
    # Sanity : G_V5 ≈ CODATA × (1 + 0.0042)
    G_codata = 6.67430e-11
    assert abs(G_V5 / G_codata - 1.0) < 0.01


def test_kerr_parameter_inverse():
    """kerr_parameter(J=Mac, M) = a."""
    a_in = 1.5e3
    J = M_SUN * a_in * C_LIGHT
    a_out = kerr_parameter(J, M_SUN)
    assert math.isclose(a_out, a_in, rel_tol=1e-14)


# ---------------------------------------------------------------------------
# 10. Summary global : trou noir solaire à 0.5 r_g.
# ---------------------------------------------------------------------------
def test_summary_all_checks_green():
    """Le summary V5.10a sur un trou noir solaire passe toutes les
    cohérences."""
    s = summary()
    assert s["asymptotic_regime_ok"]
    assert s["schwarzschild_limit_ok"]
    assert s["minkowski_limit_ok"]
    # Rapport asymptotique à 0.1 %.
    assert abs(s["asymptotic_ratio_V5_over_BL"] - 1.0) < 1e-3
    # κ_a non-nul (a ≠ 0).
    assert s["kappa_a_si"] != 0.0
    # G_V5 ≈ CODATA à 1 %.
    assert abs(s["G_V5"] - 6.67430e-11) / 6.67430e-11 < 0.01


def test_f_align_from_kerr_a_sanity():
    """f_align_from_kerr_a est défini et fini pour un trou noir solaire."""
    rg = schwarzschild_radius(M_SUN)
    f = f_align_from_kerr_a(0.5 * rg, M_SUN)
    assert math.isfinite(f)
    assert f >= 0.0


# ---------------------------------------------------------------------------
# 11. Cross-check : BL avec G=G_V5 cohérent.
# ---------------------------------------------------------------------------
def test_BL_at_horizon_finite():
    """g_tφ(BL) est fini hors horizon."""
    rg = schwarzschild_radius(M_SUN)
    a = 0.3 * rg
    # r = 3 r_g, θ = π/2 : bien hors horizon (r_+ ≈ r_g pour Kerr).
    g = g_tphi_boyer_lindquist(3.0 * rg, math.pi / 2, M_SUN, a)
    assert math.isfinite(g)
    assert g < 0.0      # signe attendu (rotation dans le sens +φ)
