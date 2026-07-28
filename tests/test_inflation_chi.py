"""Tests V5.13 — inflation topologique χ-field."""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.inflation_chi_field import (
    A_S_PLANCK,
    ALPHA_COND_V5,
    M_PL_GEV,
    N_EFOLDS_PIVOT,
    N_S_PLANCK,
    N_S_PLANCK_SIGMA,
    Q_INFLATION_V5,
    R_PLANCK_BICEP,
    A_s_V5_dimensionless,
    N_efolds,
    V0_calibrated_from_As,
    V_potential,
    V_potential_phi,
    Vprime_phi,
    Vsecond_phi,
    find_psi_pivot,
    n_s_V5,
    phi_end_inflation,
    predict_inflation_V5,
    predict_starobinsky,
    r_V5,
    slow_roll_epsilon,
    slow_roll_eta,
    summary,
)


# ---------------------------------------------------------------------------
# Tests sur les constantes / paramètres fondamentaux.
# ---------------------------------------------------------------------------
def test_alpha_cond_value():
    """α_cond V5 vaut 1,1991 (pénalité par croisement P18 / §13.6.4)."""
    assert ALPHA_COND_V5 == pytest.approx(1.1991, abs=1e-4)


def test_q_inflation_value():
    """q = 2 α_cond ≈ 2.40 (dérivé pas ajusté)."""
    assert Q_INFLATION_V5 == pytest.approx(2.3982, abs=1e-3)


def test_M_pl_GeV():
    """Échelle de Planck réduite ≈ 2.435e18 GeV."""
    assert M_PL_GEV == pytest.approx(2.435e18, rel=5e-3)


# ---------------------------------------------------------------------------
# Tests sur le potentiel topologique.
# ---------------------------------------------------------------------------
def test_V_potential_psi_squared_scaling():
    """V(|Ψ|²) ∝ (|Ψ|²)^(-α_cond)."""
    V1 = V_potential(1.0, psi2_0=1.0, V0=1.0)
    V2 = V_potential(2.0, psi2_0=1.0, V0=1.0)
    assert V2 / V1 == pytest.approx(2.0 ** (-ALPHA_COND_V5), rel=1e-12)


def test_V_potential_phi_consistency():
    """V_potential_phi(φ) = V_potential(φ²/2) à constante près."""
    phi = 3.0
    V_phi = V_potential_phi(phi, phi_0=1.0, V0=1.0)
    # V(φ) = φ^{-q}, q = 2α
    assert V_phi == pytest.approx(phi ** (-Q_INFLATION_V5), rel=1e-12)


def test_Vprime_numerical_vs_analytical():
    """V' analytique = numérique."""
    phi = 5.0
    h = 1e-6
    V_p = V_potential_phi(phi + h)
    V_m = V_potential_phi(phi - h)
    Vprime_num = (V_p - V_m) / (2.0 * h)
    Vprime_an = Vprime_phi(phi)
    assert Vprime_an == pytest.approx(Vprime_num, rel=1e-6)


def test_Vsecond_numerical_vs_analytical():
    """V'' analytique = numérique."""
    phi = 5.0
    h = 1e-4
    V_p = V_potential_phi(phi + h)
    V_0 = V_potential_phi(phi)
    V_m = V_potential_phi(phi - h)
    Vsec_num = (V_p - 2.0 * V_0 + V_m) / (h ** 2)
    Vsec_an = Vsecond_phi(phi)
    assert Vsec_an == pytest.approx(Vsec_num, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests slow-roll.
# ---------------------------------------------------------------------------
def test_epsilon_at_phi_end_equals_one():
    """ε(φ_end) = 1 (définition de φ_end)."""
    phi_end = phi_end_inflation()
    eps = slow_roll_epsilon(phi_end)
    assert eps == pytest.approx(1.0, rel=1e-12)


def test_eta_positive_for_inverse_power_law():
    """V ∝ φ^{-q} ⇒ V'' > 0 ⇒ η > 0 (concave inversé)."""
    et = slow_roll_eta(10.0)
    assert et > 0.0


def test_phi_end_formula():
    """φ_end = q/√2 (q = 2 α_cond)."""
    phi_end = phi_end_inflation()
    assert phi_end == pytest.approx(Q_INFLATION_V5 / math.sqrt(2.0), rel=1e-12)


# ---------------------------------------------------------------------------
# Tests e-folds.
# ---------------------------------------------------------------------------
def test_N_efolds_zero_at_phi_end():
    """N_e = 0 quand φ_initial = φ_end."""
    phi_end = phi_end_inflation()
    N = N_efolds(phi_end)
    assert N == pytest.approx(0.0, abs=1e-12)


def test_find_psi_pivot_consistency_with_N_efolds():
    """find_psi_pivot(N_e) ↔ N_efolds inverse."""
    phi_star = find_psi_pivot(N_e_target=60.0)
    N_back = N_efolds(phi_star)
    assert N_back == pytest.approx(60.0, rel=1e-10)


def test_phi_star_grows_with_N():
    """φ_★(N=60) > φ_★(N=50) (plus d'e-folds = champ plus grand)."""
    phi60 = find_psi_pivot(N_e_target=60.0)
    phi50 = find_psi_pivot(N_e_target=50.0)
    assert phi60 > phi50


# ---------------------------------------------------------------------------
# Tests sur n_s, r prédits.
# ---------------------------------------------------------------------------
def test_n_s_V5_close_to_one():
    """n_s V5 pour V ∝ φ^{-q} est proche de 1 (slow-roll ε petit)."""
    n_s = n_s_V5(N_e=60.0)
    assert 0.99 < n_s < 1.0


def test_r_V5_too_large_honest():
    """V5 pure power-law a r > 0.06 BICEP : tension à documenter honnêtement."""
    r = r_V5(N_e=60.0)
    assert r > R_PLANCK_BICEP, "Tension r vs BICEP — voir docs/V5p13"


def test_consistency_n_s_r_inflation_consistency():
    """Relation de cohérence : r = 16 ε et 1 - n_s ≈ 2(q-2)/(q+4N) à l'OL.

    Pour N grand : 1 - n_s ≈ 2/N_e × (1 - 2/q), r ≈ 4q/N.
    Vérifie que n_s et r sont cohérents avec le même φ_★.
    """
    N = 60.0
    n_s = n_s_V5(N_e=N)
    r = r_V5(N_e=N)
    # eps = r/16
    eps = r / 16.0
    # eta = q(q+1)/φ²
    q = Q_INFLATION_V5
    phi_star_sq = q ** 2 / 2.0 + 2.0 * q * N
    eta = q * (q + 1.0) / phi_star_sq
    n_s_recompute = 1.0 - 6.0 * eps + 2.0 * eta
    assert n_s == pytest.approx(n_s_recompute, rel=1e-12)


# ---------------------------------------------------------------------------
# Tests sur la calibration V_0.
# ---------------------------------------------------------------------------
def test_V0_calibration_reproduces_A_s_Planck():
    """V_0 calibré sur A_s ⇒ A_s prédit = A_s_Planck."""
    V0 = V0_calibrated_from_As(N_e=60.0, A_s_target=A_S_PLANCK)
    A_s_pred = A_s_V5_dimensionless(N_e=60.0, V0_over_Mpl4=V0)
    assert A_s_pred == pytest.approx(A_S_PLANCK, rel=1e-10)


def test_V0_calibrated_positive():
    """V_0 calibré > 0."""
    V0 = V0_calibrated_from_As(N_e=60.0)
    assert V0 > 0.0


# ---------------------------------------------------------------------------
# Test synthèse / API publique.
# ---------------------------------------------------------------------------
def test_predict_inflation_V5_keys():
    """API summary() expose les bonnes clés."""
    s = summary(N_e=60.0)
    for k in ("alpha_cond_V5", "q_V_phi", "N_e", "phi_star_Mpl",
              "n_s_V5", "n_s_Planck", "r_V5", "r_Planck_upper",
              "A_s_target", "V0_over_Mpl4_calibrated"):
        assert k in s


def test_predict_inflation_V5_dataclass():
    """predict_inflation_V5() renvoie un objet avec les bons champs."""
    pred = predict_inflation_V5(N_e=60.0)
    assert pred.alpha_cond == ALPHA_COND_V5
    assert pred.q == Q_INFLATION_V5
    assert pred.N_e == 60.0
    assert pred.phi_star > pred.phi_end
    assert 0.0 < pred.epsilon < 1.0
    # tension avec BICEP : on documente, pas on échoue
    assert isinstance(pred.r_passes_upper, bool)


def test_predict_starobinsky_compatible_planck():
    """Référence : Starobinsky R² est compatible Planck (sanity)."""
    s = predict_starobinsky(N_e=60.0)
    assert s["compatible_planck_n_s"] is True
    assert s["compatible_BICEP_r"] is True
    assert s["n_s"] == pytest.approx(0.9667, abs=1e-3)
    assert s["r"] == pytest.approx(0.00333, abs=1e-3)
