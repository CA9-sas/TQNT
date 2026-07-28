"""Tests V5 — SGWB NANOGrav 15-yr (Agazie+ 2023).

Verifie :
  1. Conversion alpha <-> gamma (convention NANOGrav).
  2. Lois de puissance h_c(f) : exposants SMBHB, strings, V5.
  3. Conversion h_c -> Omega_GW (densite d'energie).
  4. Modeles : SMBHB, strings, V5 ponctuel, V5 membrane.
  5. Mixture h_c^2 incoherente (additivite).
  6. Pente effective alpha_eff(f) d'une mixture.
  7. Donnees NANOGrav 15-yr : 5 bins, monotonie decroissante.
  8. Fit SMBHB pur (analytique en log-amplitude).
  9. Fit V5 combine (chi^2 sur grille A_string, A_mem).
 10. Bayes factor BIC : signe et coherence des verdicts.
 11. Pente theorique SMBHB seule = 13/3.
 12. summary() : cles attendues.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.nanograv_sgwb import (
    ALPHA_SMBHB,
    ALPHA_STRING,
    ALPHA_V5_MEMBRANE,
    ALPHA_V5_POINT,
    F_YR_HZ,
    GAMMA_SMBHB,
    GAMMA_STRING,
    GAMMA_V5_MEMBRANE,
    GAMMA_V5_POINT,
    NANOGRAV_GAMMA_HD,
    alpha_to_gamma,
    bayes_factor_V5_vs_SMBHB,
    chi2_model,
    effective_alpha,
    fit_SMBHB_pure,
    fit_V5_combined,
    gamma_to_alpha,
    h_c_SMBHB,
    h_c_V5_combined,
    h_c_V5_membrane,
    h_c_V5_point,
    h_c_powerlaw,
    h_c_strings,
    nanograv_15yr_bins,
    omega_gw_from_hc,
    run_full_test,
    summary,
)


# ---------------------------------------------------------------------------
# 1. Convention NANOGrav.
# ---------------------------------------------------------------------------
def test_alpha_gamma_inverse():
    """alpha -> gamma -> alpha (involution)."""
    for a in (-2/3, -7/6, -1.0, 0.5, 0.0):
        assert math.isclose(gamma_to_alpha(alpha_to_gamma(a)), a, abs_tol=1e-12)


def test_gamma_SMBHB_is_13_over_3():
    """SMBHB Phinney 2001 : gamma = 13/3."""
    assert math.isclose(GAMMA_SMBHB, 13.0/3.0, abs_tol=1e-12)
    assert math.isclose(ALPHA_SMBHB, -2.0/3.0, abs_tol=1e-12)


def test_gamma_string_is_16_over_3():
    """Cordes cusps : gamma = 16/3."""
    assert math.isclose(GAMMA_STRING, 16.0/3.0, abs_tol=1e-12)


def test_gamma_V5_point_membrane():
    """V5 ponctuel = 5 ; V5 membrane = 7/3."""
    assert math.isclose(GAMMA_V5_POINT, 5.0, abs_tol=1e-12)
    assert math.isclose(GAMMA_V5_MEMBRANE, 7.0/3.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. Lois de puissance.
# ---------------------------------------------------------------------------
def test_h_c_powerlaw_scaling():
    """h_c(2 f_ref) / h_c(f_ref) = 2^alpha."""
    A, alpha = 1.0e-15, -2.0/3.0
    h1 = h_c_powerlaw(F_YR_HZ, A, alpha)
    h2 = h_c_powerlaw(2.0 * F_YR_HZ, A, alpha)
    assert math.isclose(h2 / h1, 2.0 ** alpha, rel_tol=1e-10)


def test_h_c_powerlaw_pivot():
    """h_c(f_ref) = A par construction."""
    assert math.isclose(h_c_powerlaw(F_YR_HZ, 1.5e-15, -0.5),
                         1.5e-15, rel_tol=1e-12)


def test_h_c_powerlaw_negative_A_rejected():
    with pytest.raises(ValueError):
        h_c_powerlaw(F_YR_HZ, -1.0, -2.0/3.0)


# ---------------------------------------------------------------------------
# 3. Conversion h_c -> Omega_GW.
# ---------------------------------------------------------------------------
def test_omega_gw_quadratic_in_hc():
    """Omega_GW ∝ h_c^2 a f fixe."""
    f = 1e-9
    omega1 = omega_gw_from_hc(f, 1.0e-15)
    omega2 = omega_gw_from_hc(f, 2.0e-15)
    assert math.isclose(omega2 / omega1, 4.0, rel_tol=1e-10)


def test_omega_gw_1nHz_order_of_magnitude():
    """A f=1 nHz, A_yr=2.4e-15 doit donner Omega_GW ~ 1e-9 (NANOGrav order)."""
    f = 1e-9
    h = h_c_SMBHB(f, A_yr=2.4e-15)
    omega = float(omega_gw_from_hc(f, h))
    # Ordre attendu ~1e-9 a 1 dex pres
    assert 1e-11 < omega < 1e-7


# ---------------------------------------------------------------------------
# 4. Modeles individuels.
# ---------------------------------------------------------------------------
def test_h_c_SMBHB_slope():
    """SMBHB : pente -2/3 en log-log."""
    f1, f2 = 1e-9, 1e-7
    h1 = float(h_c_SMBHB(f1))
    h2 = float(h_c_SMBHB(f2))
    slope = math.log(h2 / h1) / math.log(f2 / f1)
    assert math.isclose(slope, ALPHA_SMBHB, abs_tol=1e-10)


def test_h_c_V5_point_slope():
    """V5 ponctuel : pente -1."""
    f1, f2 = 1e-9, 1e-7
    h1 = float(h_c_V5_point(f1, A_V5=1e-15))
    h2 = float(h_c_V5_point(f2, A_V5=1e-15))
    slope = math.log(h2 / h1) / math.log(f2 / f1)
    assert math.isclose(slope, ALPHA_V5_POINT, abs_tol=1e-10)


def test_h_c_V5_membrane_slope():
    """V5 membrane : pente +1/3 (spectre bleu)."""
    f1, f2 = 1e-9, 1e-7
    h1 = float(h_c_V5_membrane(f1, A_mem=1e-15))
    h2 = float(h_c_V5_membrane(f2, A_mem=1e-15))
    slope = math.log(h2 / h1) / math.log(f2 / f1)
    assert math.isclose(slope, ALPHA_V5_MEMBRANE, abs_tol=1e-10)


def test_h_c_strings_scaling_with_Gmu():
    """A_string proportionnel a G mu."""
    f = 1e-8
    h1 = float(h_c_strings(f, G_mu=1e-11))
    h2 = float(h_c_strings(f, G_mu=2e-11))
    assert math.isclose(h2 / h1, 2.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 5. Mixture incoherente.
# ---------------------------------------------------------------------------
def test_mixture_additive_in_h2():
    """h_c^2 mixture = sum des h_c^2 individuels."""
    f = np.array([2e-9, 1e-8, 5e-8])
    A_S, A_str, A_mem = 1e-15, 5e-16, 3e-16
    h_mix = h_c_V5_combined(f, A_S, A_str, A_mem)
    h_S = h_c_SMBHB(f, A_S)
    h_str = h_c_powerlaw(f, A_str, ALPHA_STRING)
    h_mem = h_c_powerlaw(f, A_mem, ALPHA_V5_MEMBRANE)
    expected = np.sqrt(h_S ** 2 + h_str ** 2 + h_mem ** 2)
    np.testing.assert_allclose(h_mix, expected, rtol=1e-12)


def test_mixture_reduces_to_SMBHB_when_others_zero():
    """A_string = A_mem = 0 -> mixture = SMBHB pur."""
    f = np.array([1e-9, 1e-8, 1e-7])
    np.testing.assert_allclose(
        h_c_V5_combined(f, 2e-15, 0.0, 0.0),
        h_c_SMBHB(f, 2e-15), rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. Pente effective.
# ---------------------------------------------------------------------------
def test_effective_alpha_pure_SMBHB():
    """SMBHB pur : alpha_eff = -2/3 partout."""
    f = np.logspace(-9, -7, 50)
    alpha = effective_alpha(f, A_SMBHB=1e-15, A_string=0.0, A_mem=0.0)
    # Bords moins precis (gradient one-sided) -> tester l'interieur
    np.testing.assert_allclose(alpha[5:-5], ALPHA_SMBHB, atol=1e-6)


def test_effective_alpha_membrane_pulls_up():
    """Une grosse contribution membrane (pente positive) tire alpha_eff
    vers le haut (donc gamma_eff vers le bas)."""
    f = np.logspace(-9, -7, 50)
    a_S = effective_alpha(f, 1e-15, 0.0, 0.0)[25]
    a_S_plus_mem = effective_alpha(f, 1e-15, 0.0, 1e-15)[25]
    # mem domine en haute f donc alpha_eff > alpha_SMBHB
    assert a_S_plus_mem > a_S


# ---------------------------------------------------------------------------
# 7. Donnees NANOGrav.
# ---------------------------------------------------------------------------
def test_nanograv_bins_shape_and_monotone():
    """5 bins, freq monotone, h_c decroissant."""
    f, h, sig = nanograv_15yr_bins()
    assert len(f) == 5
    assert np.all(np.diff(f) > 0)
    assert np.all(np.diff(h) < 0)         # spectre rouge
    assert np.all(sig > 0)


def test_nanograv_bins_range():
    """Frequences dans [1 nHz, 100 nHz]."""
    f, _, _ = nanograv_15yr_bins()
    assert f.min() >= 1e-9
    assert f.max() <= 1e-7


# ---------------------------------------------------------------------------
# 8. Fits.
# ---------------------------------------------------------------------------
def test_fit_SMBHB_returns_reasonable_amplitude():
    """Le fit SMBHB pur trouve A_yr ~ 1-3 x 1e-15 (NANOGrav 15yr)."""
    f, h, sig = nanograv_15yr_bins()
    fit = fit_SMBHB_pure(f, h, sig)
    assert 1.0e-15 < fit["A_yr"] < 5.0e-15
    assert fit["chi2"] >= 0
    assert fit["dof"] == 4


def test_fit_SMBHB_optimal_minimizes_chi2():
    """Decaler A_yr de la valeur fittee augmente chi^2."""
    f, h, sig = nanograv_15yr_bins()
    fit = fit_SMBHB_pure(f, h, sig)
    chi2_opt = fit["chi2"]
    A = fit["A_yr"]
    for factor in (0.5, 2.0):
        h_test = h_c_SMBHB(f, A_yr=A * factor)
        chi2_test = chi2_model(h_test, h, sig)
        assert chi2_test > chi2_opt


def test_fit_V5_combined_does_not_exceed_SMBHB_chi2():
    """V5 (plus de parametres) doit avoir chi^2 <= chi^2 SMBHB pur."""
    f, h, sig = nanograv_15yr_bins()
    fS = fit_SMBHB_pure(f, h, sig)
    fV = fit_V5_combined(f, h, sig, include_string=True, include_membrane=True)
    assert fV["chi2"] <= fS["chi2"] + 1e-6


# ---------------------------------------------------------------------------
# 9. Bayes factor BIC.
# ---------------------------------------------------------------------------
def test_bayes_factor_BIC_sign():
    """Si chi^2 V5 est seulement marginalement meilleur, BIC penalise V5."""
    f, h, sig = nanograv_15yr_bins()
    fS = fit_SMBHB_pure(f, h, sig)
    fV = fit_V5_combined(f, h, sig, include_string=True, include_membrane=True)
    b = bayes_factor_V5_vs_SMBHB(fS, fV)
    assert "ln_Bayes_V5_over_SMBHB" in b
    assert b["verdict"] in {"decisive_for_V5", "strong_for_V5",
                              "substantial_for_V5", "indistinct",
                              "substantial_for_SMBHB", "decisive_for_SMBHB"}


def test_bayes_factor_when_chi2_equal():
    """Si chi^2 identiques, V5 (plus de parametres) est penalise."""
    fS = {"chi2": 1.0, "n_params": 1, "n_data": 5}
    fV = {"chi2": 1.0, "n_params": 3, "n_data": 5}
    b = bayes_factor_V5_vs_SMBHB(fS, fV)
    assert b["ln_Bayes_V5_over_SMBHB"] < 0


# ---------------------------------------------------------------------------
# 10. summary().
# ---------------------------------------------------------------------------
def test_summary_keys():
    s = summary()
    for k in ("gamma_SMBHB_theory", "gamma_V5_point_theory",
              "gamma_V5_membrane_theory", "gamma_obs_NANOGrav",
              "chi2_SMBHB_pure", "chi2_V5_full", "verdict"):
        assert k in s
    assert math.isclose(s["gamma_SMBHB_theory"], 13.0/3.0)
    assert math.isclose(s["gamma_obs_NANOGrav"], NANOGRAV_GAMMA_HD)


def test_run_full_test_consistent():
    """run_full_test() donne des fits coherents avec les helpers individuels."""
    r = run_full_test()
    f, h, sig = nanograv_15yr_bins()
    fS = fit_SMBHB_pure(f, h, sig)
    assert math.isclose(r.fit_SMBHB["chi2"], fS["chi2"], rel_tol=1e-10)
    assert math.isclose(r.fit_SMBHB["A_yr"], fS["A_yr"], rel_tol=1e-10)
