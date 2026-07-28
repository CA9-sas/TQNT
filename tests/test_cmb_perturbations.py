"""Tests V5.12 — Perturbations CMB et spectre C_ℓ depuis χ-field."""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.cmb_perturbations import (
    A_S_PLANCK,
    C_L_PEAK1_PLANCK_MUK2,
    L_DAMPING_SILK,
    L_PEAK1_PLANCK,
    L_PEAK2_PLANCK,
    L_PEAK3_PLANCK,
    N_S_PLANCK,
    T_CMB_K,
    Z_DRAG,
    Z_LSS,
    CMBValidation,
    C_l_spectrum_V5,
    Phi_dot_LCDM,
    Phi_from_delta_psi,
    acoustic_peak_scale,
    angular_diameter_distance_LSS,
    integrated_sachs_wolfe,
    perturbed_metric,
    primordial_power_spectrum,
    sachs_wolfe,
    sound_horizon_at_drag,
    summary,
    validate_planck_peaks,
)
from theory.flrw_emergence import cosmo_state_V5


# ---------------------------------------------------------------------------
# 1. Constantes CMB.
# ---------------------------------------------------------------------------
def test_constants_planck_2018():
    assert L_PEAK1_PLANCK == pytest.approx(220.0)
    assert L_PEAK2_PLANCK == pytest.approx(540.0)
    assert L_PEAK3_PLANCK == pytest.approx(815.0)
    assert Z_LSS > Z_DRAG > 1000
    assert 2.7 < T_CMB_K < 2.75
    assert 1e-9 < A_S_PLANCK < 3e-9
    assert 0.9 < N_S_PLANCK < 1.0


# ---------------------------------------------------------------------------
# 2. Métrique perturbée.
# ---------------------------------------------------------------------------
def test_perturbed_metric_scalar_isotropic():
    m = perturbed_metric(Phi=1e-5, Psi=1e-5, a=1.0)
    assert m["isotropic"] is True
    # g_00 = -(1 + 2Φ)
    assert m["g00"] == pytest.approx(-(1.0 + 2e-5))
    # g_ii = a²(1-2Ψ)
    assert m["gii"] == pytest.approx(1.0 - 2e-5)


def test_perturbed_metric_array_no_anisotropic_stress():
    Phi = np.array([1e-6, 2e-6, 3e-6])
    Psi = Phi.copy()
    m = perturbed_metric(Phi, Psi, a=2.0)
    assert np.allclose(m["anisotropic_stress"], 0.0)
    assert m["a"] == 2.0


def test_perturbed_metric_a_squared_factor():
    a = 0.5
    m = perturbed_metric(Phi=0.0, Psi=0.0, a=a)
    # Sans perturbation, g_ii = a²
    assert float(m["gii"]) == pytest.approx(a ** 2)


# ---------------------------------------------------------------------------
# 3. Φ depuis δΨ.
# ---------------------------------------------------------------------------
def test_phi_from_delta_psi_linearity():
    Phi1 = Phi_from_delta_psi(1.0e-5, psi_0=1.0)
    Phi2 = Phi_from_delta_psi(2.0e-5, psi_0=1.0)
    assert Phi2 / Phi1 == pytest.approx(2.0)


def test_phi_from_delta_psi_kappa_g_factor():
    # Avec κ_g_test = 1, ψ_0 = 1, δψ = 1e-5 → Φ = 1e-5
    Phi = Phi_from_delta_psi(1.0e-5, psi_0=1.0, kappa_g=1.0)
    assert float(Phi) == pytest.approx(1.0e-5)


def test_phi_from_delta_psi_zero_perturbation():
    Phi = Phi_from_delta_psi(0.0, psi_0=1.0)
    assert float(Phi) == 0.0


# ---------------------------------------------------------------------------
# 4. Sachs-Wolfe.
# ---------------------------------------------------------------------------
def test_sachs_wolfe_factor_one_third():
    dT_over_T = sachs_wolfe(3.0e-5)
    assert float(dT_over_T) == pytest.approx(1.0e-5)


def test_sachs_wolfe_array():
    Phi = np.array([3e-5, 6e-5, 9e-5])
    result = sachs_wolfe(Phi)
    assert np.allclose(result, np.array([1e-5, 2e-5, 3e-5]))


def test_sachs_wolfe_zero_phi_zero_temp():
    assert float(sachs_wolfe(0.0)) == 0.0


# ---------------------------------------------------------------------------
# 5. ISW.
# ---------------------------------------------------------------------------
def test_integrated_sachs_wolfe_zero_derivative():
    eta = np.linspace(0.0, 1.0, 100)
    Phi_dot = np.zeros_like(eta)
    assert integrated_sachs_wolfe(Phi_dot, eta) == 0.0


def test_integrated_sachs_wolfe_sign():
    """ISW positif si Φ̇ < 0 (Φ décroît) : -2 ∫ Φ̇ > 0."""
    eta = np.linspace(0.0, 1.0, 100)
    Phi_dot = -1e-5 * np.ones_like(eta)
    isw = integrated_sachs_wolfe(Phi_dot, eta)
    assert isw > 0.0


def test_phi_dot_lcdm_shape():
    a = np.linspace(0.1, 1.0, 200)
    state = cosmo_state_V5()
    Phi_dot = Phi_dot_LCDM(a, state)
    assert Phi_dot.shape == a.shape


def test_phi_dot_lcdm_late_decay():
    """Φ̇ < 0 quand Ω_Λ devient dominant (a proche de 1)."""
    a = np.linspace(0.1, 1.0, 500)
    state = cosmo_state_V5()
    Phi_dot = Phi_dot_LCDM(a, state)
    # À a ~ 1, Φ décroît → dΦ/dη < 0 typiquement
    assert Phi_dot[-1] <= Phi_dot[0]


# ---------------------------------------------------------------------------
# 6. Horizon sonore et distance angulaire.
# ---------------------------------------------------------------------------
def test_sound_horizon_drag_consistent_planck_desi():
    r_s = sound_horizon_at_drag()
    # Cohérent avec la valeur Planck/DESI 147.05 Mpc à 1 %.
    assert abs(r_s - 147.05) / 147.05 < 0.01


def test_angular_diameter_distance_lss_positive():
    D_A = angular_diameter_distance_LSS()
    # D_A(LSS) ≈ 12.8 Mpc dans Planck 2018
    assert 12.0 < D_A < 14.0


# ---------------------------------------------------------------------------
# 7. Pic acoustique principal.
# ---------------------------------------------------------------------------
def test_acoustic_peak_scale_l_A_around_300():
    peaks = acoustic_peak_scale()
    # ℓ_A ≈ π D_M / r_s ≈ 300 (Planck)
    assert 290 < peaks["l_A"] < 310


def test_acoustic_peak_scale_l_1_close_220():
    peaks = acoustic_peak_scale()
    assert abs(peaks["l_1"] - 220.0) / 220.0 < 0.05


def test_acoustic_peak_scale_l_2_close_540():
    peaks = acoustic_peak_scale()
    assert abs(peaks["l_2"] - 540.0) / 540.0 < 0.05


def test_acoustic_peak_scale_l_3_close_815():
    peaks = acoustic_peak_scale()
    assert abs(peaks["l_3"] - 815.0) / 815.0 < 0.05


def test_acoustic_peak_scale_returns_all_keys():
    peaks = acoustic_peak_scale()
    for k in ("l_A", "l_1", "l_2", "l_3", "r_s_Mpc", "D_M_Mpc", "D_A_Mpc",
              "phi_shifts"):
        assert k in peaks


# ---------------------------------------------------------------------------
# 8. Spectre primordial.
# ---------------------------------------------------------------------------
def test_primordial_power_pivot_normalization():
    Pk = primordial_power_spectrum(0.05)
    assert float(Pk) == pytest.approx(A_S_PLANCK)


def test_primordial_power_scale_invariance_limit():
    """Si n_s = 1 (Harrison-Zeldovich), P(k) constant."""
    k = np.array([0.001, 0.05, 1.0])
    Pk = primordial_power_spectrum(k, n_s=1.0)
    assert np.allclose(Pk, A_S_PLANCK)


# ---------------------------------------------------------------------------
# 9. Spectre C_ℓ TT V5.
# ---------------------------------------------------------------------------
def test_c_l_spectrum_shape():
    l_arr = np.arange(2, 2501)
    spec = C_l_spectrum_V5(l_arr)
    assert spec["D_l_muK2"].shape == l_arr.shape
    assert np.all(spec["D_l_muK2"] >= 0.0)


def test_c_l_spectrum_first_peak_amplitude():
    l_arr = np.arange(2, 2501)
    spec = C_l_spectrum_V5(l_arr)
    i_peak1 = int(np.argmin(np.abs(l_arr - spec["l_peaks_V5"][0])))
    Dl = float(spec["D_l_muK2"][i_peak1])
    # Premier pic dans 10 % de 5755 µK² Planck
    assert abs(Dl - C_L_PEAK1_PLANCK_MUK2) / C_L_PEAK1_PLANCK_MUK2 < 0.10


def test_c_l_spectrum_silk_damping():
    """Le spectre doit être fortement amorti à ℓ → 2500 (Silk damping)."""
    l_arr = np.array([2200, 2400, 2500])
    spec = C_l_spectrum_V5(l_arr)
    # Damping doit fortement supprimer à ℓ ~ 2500 vs ℓ ~ ℓ_peak1
    l_peak = np.array([220])
    spec_peak = C_l_spectrum_V5(l_peak)
    Dl_tail = float(spec["D_l_muK2"][-1])
    Dl_peak = float(spec_peak["D_l_muK2"][0])
    assert Dl_tail < 0.05 * Dl_peak


# ---------------------------------------------------------------------------
# 10. Validation Planck — pics acoustiques.
# ---------------------------------------------------------------------------
def test_validate_planck_peaks_returns_validation():
    val = validate_planck_peaks()
    assert isinstance(val, CMBValidation)


def test_validate_planck_peaks_l_1_within_1pct():
    val = validate_planck_peaks()
    assert abs(val.l_1_error_pct) < 1.0


def test_validate_planck_peaks_l_2_within_5pct():
    val = validate_planck_peaks()
    assert abs(val.l_2_error_pct) < 5.0


def test_validate_planck_peaks_l_3_within_5pct():
    val = validate_planck_peaks()
    assert abs(val.l_3_error_pct) < 5.0


def test_validate_planck_peaks_ratios_close():
    val = validate_planck_peaks()
    # ℓ_2/ℓ_1 V5 doit approcher 2.455 Planck à 5 %
    assert abs(val.ratio_l2_l1_V5 - val.ratio_l2_l1_planck) < 0.15
    # ℓ_3/ℓ_1 V5 doit approcher 3.705 Planck à 5 %
    assert abs(val.ratio_l3_l1_V5 - val.ratio_l3_l1_planck) < 0.20


# ---------------------------------------------------------------------------
# 11. Summary intégral.
# ---------------------------------------------------------------------------
def test_summary_runs_no_exception():
    s = summary()
    assert isinstance(s, dict)


def test_summary_contains_all_peaks():
    s = summary()
    for k in ("l_1_V5", "l_2_V5", "l_3_V5", "l_A_V5",
              "Dl_peak1_V5_muK2", "Dl_peak1_planck_muK2",
              "r_s_Mpc", "D_A_LSS_Mpc"):
        assert k in s


def test_summary_first_peak_amplitude_reasonable():
    s = summary()
    Dl = s["Dl_peak1_V5_muK2"]
    # Premier pic V5 doit reproduire Planck (~5755 µK²) à 10 %
    assert abs(Dl - C_L_PEAK1_PLANCK_MUK2) / C_L_PEAK1_PLANCK_MUK2 < 0.10
