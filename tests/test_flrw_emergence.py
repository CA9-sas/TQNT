"""Tests V5.11 — Cosmologie FLRW émergente depuis le χ-field."""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.flrw_emergence import (
    ALPHA_CONDENSATION,
    C_LIGHT,
    DM_OVER_RS_Z05_DESI,
    G_CODATA,
    H0_PLANCK_KM_S_MPC,
    H0_PLANCK_SI,
    H0_SH0ES_KM_S_MPC,
    KM_S_MPC_TO_SI,
    MPC_M,
    OMEGA_B_OVER_M_PLANCK,
    OMEGA_DM_OVER_B_TARGET,
    OMEGA_L_PLANCK,
    OMEGA_M_PLANCK,
    R_DRAG_MPC,
    CosmoState,
    comoving_distance_flat,
    conservation_eq,
    cosmic_evolution,
    cosmo_state_V5,
    critical_density,
    friedmann_1,
    friedmann_2,
    hubble_rate,
    hubble_tension_P22,
    omega_dm_over_b_from_alpha,
    scale_factor_from_psi2,
    summary,
    validate_desi_2024,
    validate_planck_2018,
)


# ---------------------------------------------------------------------------
# 1. Constantes et unités.
# ---------------------------------------------------------------------------
def test_h0_planck_si_units():
    expected = H0_PLANCK_KM_S_MPC * 1e3 / MPC_M
    assert abs(H0_PLANCK_SI - expected) / expected < 1e-12


def test_km_s_mpc_to_si():
    assert abs(KM_S_MPC_TO_SI - 1e3 / MPC_M) < 1e-30


def test_critical_density_value():
    # ρ_crit attendu ≈ 8.5e-27 kg/m³ pour H0 = 67.4
    rho_c = critical_density(H0_PLANCK_SI)
    assert 7.5e-27 < rho_c < 9.5e-27


# ---------------------------------------------------------------------------
# 2. Facteur d'échelle et Ω².
# ---------------------------------------------------------------------------
def test_scale_factor_vacuum():
    """a(0) = 1 quand |Ψ|² = 0 (vide)."""
    assert scale_factor_from_psi2(0.0) == pytest.approx(1.0)


def test_scale_factor_growth():
    """a augmente quand |Ψ|² augmente."""
    a1 = scale_factor_from_psi2(0.1)
    a2 = scale_factor_from_psi2(0.5)
    assert a2 > a1 > 1.0


# ---------------------------------------------------------------------------
# 3. Friedmann I.
# ---------------------------------------------------------------------------
def test_friedmann_1_recovers_H0():
    """H = sqrt((8πG/3)ρ_tot) = H₀ pour ρ_tot = ρ_crit."""
    state = cosmo_state_V5()
    rho_tot = state.rho_m + state.rho_lambda
    H_pred = friedmann_1(rho_tot, a=1.0, k=0, G=state.G)
    assert abs(H_pred - state.H0_si) / state.H0_si < 1e-12


def test_friedmann_1_negative_curvature():
    """k = -1 augmente H² (univers ouvert)."""
    state = cosmo_state_V5()
    rho_tot = state.rho_m + state.rho_lambda
    H_flat = friedmann_1(rho_tot, a=1.0, k=0, G=state.G)
    H_open = friedmann_1(rho_tot, a=1.0, k=-1, G=state.G)
    assert H_open > H_flat


# ---------------------------------------------------------------------------
# 4. Friedmann II.
# ---------------------------------------------------------------------------
def test_friedmann_2_matter_decelerates():
    """w = 0 (poussière) : ä/a < 0 (univers décélère)."""
    a_ddot = friedmann_2(rho=1e-26, p=0.0)
    assert a_ddot < 0


def test_friedmann_2_lambda_accelerates():
    """w = -1 (Λ) : ä/a > 0 (accélération)."""
    rho = 1e-26
    p = -rho * C_LIGHT ** 2     # w = -1
    a_ddot = friedmann_2(rho=rho, p=p)
    assert a_ddot > 0


def test_friedmann_2_today_accelerating():
    """Univers actuel (Planck) : Λ domine → ä/a > 0."""
    s = summary()
    assert s["a_ddot_over_a_today_s2"] > 0
    assert s["accelerating_universe"] is True


# ---------------------------------------------------------------------------
# 5. Conservation.
# ---------------------------------------------------------------------------
def test_conservation_dust():
    """w = 0, ρ̇ = -3 H ρ."""
    rho = 1e-26
    H = 2.2e-18
    rho_dot = conservation_eq(rho, p=0.0, H=H)
    assert abs(rho_dot - (-3 * H * rho)) / abs(rho_dot) < 1e-12


def test_conservation_lambda():
    """Λ (w=-1) : ρ̇ = 0."""
    rho = 1e-26
    p = -rho * C_LIGHT ** 2
    rho_dot = conservation_eq(rho, p, H=2.2e-18)
    assert abs(rho_dot) < 1e-40


# ---------------------------------------------------------------------------
# 6. État cosmologique.
# ---------------------------------------------------------------------------
def test_cosmo_state_default():
    state = cosmo_state_V5()
    assert abs(state.Omega_m - OMEGA_M_PLANCK) < 1e-12
    assert abs(state.Omega_lambda - OMEGA_L_PLANCK) < 1e-12
    assert abs(state.H_km_s_Mpc - H0_PLANCK_KM_S_MPC) < 1e-9


def test_cosmo_state_densities_sum():
    """ρ_b + ρ_DM = ρ_m."""
    state = cosmo_state_V5()
    assert abs(state.rho_b + state.rho_dm - state.rho_m) / state.rho_m < 1e-12


# ---------------------------------------------------------------------------
# 7. Validation Planck.
# ---------------------------------------------------------------------------
def test_planck_validation_H0():
    v = validate_planck_2018()
    assert abs(v.H0_relative_error) < 1e-9


def test_planck_validation_omega_b_over_m():
    v = validate_planck_2018()
    assert abs(v.Omega_b_over_m_V5 - v.Omega_b_over_m_target) < 1e-9


def test_omega_dm_over_b_from_alpha():
    """α^9.29 ≈ 5.40 (Planck Ω_DM/Ω_b)."""
    dm_b = omega_dm_over_b_from_alpha(ALPHA_CONDENSATION, 9.29)
    assert abs(dm_b - OMEGA_DM_OVER_B_TARGET) / OMEGA_DM_OVER_B_TARGET < 1e-2


# ---------------------------------------------------------------------------
# 8. Validation DESI.
# ---------------------------------------------------------------------------
def test_desi_validation_within_5pct():
    """D_M(0.5)/r_s V5 devrait être dans ±5 % de la mesure DESI."""
    d = validate_desi_2024()
    assert abs(d.DM_over_rs_z05_relative_error) < 0.05
    assert d.consistent_with_lambda


def test_comoving_distance_monotone():
    """D_M(z) croissant en z."""
    state = cosmo_state_V5()
    d1 = comoving_distance_flat(0.3, state)
    d2 = comoving_distance_flat(0.6, state)
    d3 = comoving_distance_flat(1.0, state)
    assert d1 < d2 < d3


def test_comoving_distance_z0_zero():
    state = cosmo_state_V5()
    assert comoving_distance_flat(0.0, state) == 0.0


# ---------------------------------------------------------------------------
# 9. Tension de Hubble (proposition P22).
# ---------------------------------------------------------------------------
def test_hubble_tension_observed():
    t = hubble_tension_P22()
    expected = (H0_SH0ES_KM_S_MPC - H0_PLANCK_KM_S_MPC) / H0_PLANCK_KM_S_MPC * 100
    assert abs(t.delta_H0_observed_pct - expected) < 1e-9


def test_hubble_tension_P22_proposition_close():
    """Avec η=1, ξ=0.08, P22 résout > 90 % de la tension."""
    t = hubble_tension_P22(eta=1.0, xi_local=0.080)
    assert t.fraction_resolved > 0.90


# ---------------------------------------------------------------------------
# 10. Cosmic evolution.
# ---------------------------------------------------------------------------
def test_cosmic_evolution_increases_a():
    """a(t) augmente quand t augmente (univers en expansion)."""
    state = cosmo_state_V5()
    t_arr = np.linspace(0, 1e15, 50)        # ~30 Myr
    res = cosmic_evolution(t_arr, state.rho_m, state.rho_lambda, a0=1.0,
                            G=state.G)
    assert res["a"][-1] > res["a"][0]


def test_cosmic_evolution_omega_squared_matches_a2():
    state = cosmo_state_V5()
    t_arr = np.linspace(0, 1e14, 20)
    res = cosmic_evolution(t_arr, state.rho_m, state.rho_lambda)
    np.testing.assert_allclose(res["omega_squared"], res["a"] ** 2, rtol=1e-12)


# ---------------------------------------------------------------------------
# 11. Synthèse.
# ---------------------------------------------------------------------------
def test_summary_keys_present():
    s = summary()
    for k in ("G_V5", "H0_input_km_s_Mpc", "H_from_Friedmann1_km_s_Mpc",
              "rho_crit", "rho_m", "rho_lambda", "DM_over_rs_z05_V5",
              "DESI_error_pct", "Hubble_tension_obs_pct",
              "Hubble_tension_P22_pct"):
        assert k in s


def test_summary_friedmann_consistency():
    """H reconstruit depuis Friedmann 1 = H₀ input à 1e-9 près."""
    s = summary()
    assert abs(s["Friedmann1_consistency"] - 1.0) < 1e-9


def test_summary_G_within_05_pct():
    """G_V5 doit être dans 0.5 % de CODATA."""
    s = summary()
    assert abs(s["G_error_pct"]) < 0.5
