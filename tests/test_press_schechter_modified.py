"""Tests du Press-Schechter modifié V5 (V5.5 étape 7)."""
import math
import pytest

from theory.press_schechter_modified import (
    variance_M, press_schechter_HMF, V5_HMF_correction, V5_HMF,
    concentration_WB, r_virial_kpc, r_half_baryon_kpc,
    psi_squared_halo, beta_psi, h_at_M, delta_CGQ_predicted, summary,
)


def test_variance_at_M_8():
    """σ(M_8) ≈ σ_8 ≈ 0.81 (calibration Planck)."""
    M_8 = 5.97e14
    sigma = math.sqrt(variance_M(M_8))
    assert sigma == pytest.approx(0.811, rel=0.01)


def test_press_schechter_HMF_positive():
    """HMF doit être positive et finie."""
    for log_M in [9, 10, 11, 12, 13, 14, 15]:
        n = press_schechter_HMF(10**log_M)
        assert n > 0
        assert math.isfinite(n)


def test_V5_correction_suppression_low_mass():
    """V5 P22 supprime la HMF à petite masse (~5%)."""
    corr_low = V5_HMF_correction(1e8)
    corr_med = V5_HMF_correction(1e12)
    assert corr_low < corr_med
    assert corr_low > 0.93  # suppression ~5%
    assert corr_med == pytest.approx(1.0, abs=0.05)


def test_concentration_WB_decreases_with_M():
    """WB : c(M) décroît avec M (signe correct)."""
    c1 = concentration_WB(1e9)
    c2 = concentration_WB(1e15)
    assert c1 > c2


def test_r_virial_scales_M_third():
    """r_vir ∝ M^(1/3)."""
    r1 = r_virial_kpc(1e10)
    r2 = r_virial_kpc(1e13)
    ratio = r2 / r1
    expected = 10 ** (3 * 1/3)  # log(10^3)/3
    assert ratio == pytest.approx(expected, rel=0.01)


def test_psi_squared_decreasing():
    """|Ψ_halo|² décroît avec M_h (densité centrale plus faible pour gros halos)."""
    psi1 = psi_squared_halo(1e10)
    psi2 = psi_squared_halo(1e14)
    assert psi1 > psi2


def test_beta_negative():
    """β = d log|Ψ|²/d log M_h doit être négatif."""
    for M in [1e10, 1e12, 1e14]:
        b = beta_psi(M)
        assert b < 0, f"β({M:.0e}) = {b} (devrait être < 0)"


def test_summary_keys():
    s = summary()
    expected = {"M_h_Msun", "c_WB", "r_vir_kpc", "r_half_baryon_kpc",
                "psi_squared_M_sun_kpc3", "h_dim", "beta_psi",
                "delta_CGQ_predicted", "delta_CGQ_target_V5p5_step5"}
    assert expected.issubset(s.keys())


def test_beta_scoping_NFW_insufficient():
    """SCOPING : β NFW pur est ~10-20× plus faible que β requis (-6.29).

    Cette assertion documente que la fermeture V5.5 étape 7 nécessite
    un mécanisme amplificateur (Einasto profile + P22 boundary enhancement).
    """
    beta_NFW = beta_psi(1e12)
    # NFW seul donne β ~ -0.1 à -1
    assert -2 < beta_NFW < 0, f"β_NFW = {beta_NFW}"
    # Cible V5.5 étape 5 : -6.29
    # Le facteur d'amplification requis est ~5-20×
    assert abs(beta_NFW / -6.29) < 0.5  # confirme que NFW pur est insuffisant
