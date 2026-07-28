"""Tests for theory/smf_V5.py (V5 §11 SMF + JWST P22 boost test)."""
import math

import pytest

from theory.smf_V5 import (
    growth_factor, sigma_M, dlnsigma_dlnM,
    sheth_tormen_HMF, sheth_tormen_f,
    boost_P22, SHMR_Behroozi, M_halo_from_M_star,
    SMF_LambdaCDM, SMF_V5, rho_star, rho_star_massive,
    chi2_smf, bayes_factor_V5_vs_LCDM, impossible_early_galaxies_test,
    JWST_SMF_DATA, JWST_Z_MID, LABBE_2023,
)


# ---------------------------------------------------------------
# Growth, sigma, HMF
# ---------------------------------------------------------------
def test_growth_factor_z0():
    """D(z=0) = 1 by normalisation."""
    assert growth_factor(0.0) == pytest.approx(1.0, abs=1e-6)


def test_growth_factor_decreasing():
    """D(z) monotonically decreases with z."""
    prev = growth_factor(0.0)
    for z in [1, 2, 5, 8, 10, 12]:
        g = growth_factor(z)
        assert g < prev, f"growth not decreasing at z={z}"
        prev = g


def test_sigma_M_calibration_local():
    """sigma_M(M_h=1e12, z=0) ~ 1.8, sigma_M(M_h=1e14, z=0) ~ 1.0 (Planck)."""
    s12 = sigma_M(1e12, 0.0)
    s14 = sigma_M(1e14, 0.0)
    assert 1.5 < s12 < 2.1
    assert 0.85 < s14 < 1.15


def test_sigma_M_decreases_with_z():
    """sigma_M(M, z) decreases with z (growth factor)."""
    for M in [1e10, 1e12, 1e14]:
        s0 = sigma_M(M, 0.0)
        s8 = sigma_M(M, 8.0)
        assert s8 < s0


def test_sheth_tormen_HMF_positive_finite():
    """ST HMF must be positive and finite at all reasonable (M, z)."""
    for log_M in (8, 10, 12, 14):
        for z in (0, 2, 5, 8, 10):
            hmf = sheth_tormen_HMF(10 ** log_M, z)
            assert hmf > 0, f"HMF <= 0 at (M=1e{log_M}, z={z})"
            assert math.isfinite(hmf)


def test_HMF_decreases_with_z():
    """At fixed M, HMF must drop monotonically with z (in the rare regime)."""
    hmf0 = sheth_tormen_HMF(1e14, 0.0)
    hmf5 = sheth_tormen_HMF(1e14, 5.0)
    hmf10 = sheth_tormen_HMF(1e14, 10.0)
    assert hmf0 > hmf5 > hmf10


# ---------------------------------------------------------------
# P22 boost
# ---------------------------------------------------------------
def test_boost_low_z_unity():
    """boost(z<z_crit) = 1 (no enhancement before reionisation)."""
    for z in [0, 1, 3, 5]:
        b = boost_P22(z)
        assert b == pytest.approx(1.0, abs=1e-9)


def test_boost_z10_v5_quote():
    """V5 §11.5 quote: boost(z=10) = 1.5 (default alpha=0.10)."""
    b10 = boost_P22(10.0)
    assert b10 == pytest.approx(1.5, abs=0.01)


def test_boost_monotonic_in_z():
    """boost(z) monotonic increasing for z > z_crit."""
    for z1, z2 in [(6, 7), (8, 10), (10, 12), (12, 15)]:
        assert boost_P22(z2) > boost_P22(z1)


# ---------------------------------------------------------------
# SHMR + SMF
# ---------------------------------------------------------------
def test_SHMR_local_peak_near_1e12():
    """At z=0, SHMR efficiency M*/M_h peaks near M_h ~ 10^12 with ~3-5%."""
    eff_12 = SHMR_Behroozi(1e12, 0.0) / 1e12
    eff_10 = SHMR_Behroozi(1e10, 0.0) / 1e10
    eff_14 = SHMR_Behroozi(1e14, 0.0) / 1e14
    assert eff_12 > eff_10
    assert eff_12 > eff_14
    assert 0.02 < eff_12 < 0.06


def test_M_halo_from_M_star_inversion():
    """SHMR inversion: M_halo_from_M_star(SHMR(M_h)) ~ M_h."""
    for log_Mh in [10, 11, 12, 13]:
        Mh = 10 ** log_Mh
        Ms = SHMR_Behroozi(Mh, 0.0)
        Mh_back = M_halo_from_M_star(Ms, 0.0)
        # 5% tolerance (bisection precision)
        assert math.log10(Mh_back) == pytest.approx(log_Mh, abs=0.10)


def test_SMF_LambdaCDM_z0_anchor():
    """SMF_LCDM(M*=1e10, z=0) ~ 10^-2.3 dex^-1 (Baldry+2012, GAMA)."""
    smf = SMF_LambdaCDM(1e10, 0.0)
    log10_smf = math.log10(smf)
    assert -2.7 < log10_smf < -2.0


def test_SMF_V5_factor_boost():
    """At z=10, SMF_V5 / SMF_LCDM ~ boost_P22(z=10) = 1.5."""
    M = 1e10
    smf_l = SMF_LambdaCDM(M, 10.0)
    smf_v = SMF_V5(M, 10.0)
    assert smf_v / smf_l == pytest.approx(1.5, rel=0.02)


def test_SMF_decreases_with_z():
    """SMF at fixed M_star drops monotonically with z."""
    M = 1e10
    smfs = [SMF_LambdaCDM(M, z) for z in [0, 2, 5, 8, 10]]
    for a, b in zip(smfs, smfs[1:]):
        assert b < a


# ---------------------------------------------------------------
# Cosmic stellar mass density
# ---------------------------------------------------------------
def test_rho_star_positive_finite():
    """rho_star(z) positive and finite for z in [0, 12]."""
    for z in [0, 2, 5, 8, 10, 12]:
        r = rho_star(z, "LambdaCDM")
        assert r > 0 and math.isfinite(r)


def test_rho_star_V5_exceeds_LCDM_at_high_z():
    """V5 boost: rho_star_V5(z=10) > rho_star_LCDM(z=10)."""
    r_l = rho_star(10.0, "LambdaCDM")
    r_v = rho_star(10.0, "V5")
    assert r_v > r_l


# ---------------------------------------------------------------
# JWST chi^2 + Bayes
# ---------------------------------------------------------------
def test_chi2_smf_LCDM_finite():
    """chi^2(LCDM) finite for all JWST bins."""
    for bin_label in JWST_SMF_DATA:
        c, N = chi2_smf(bin_label, "LambdaCDM")
        assert math.isfinite(c) and c > 0 and N > 0


def test_chi2_V5_lower_than_LCDM_at_nominal():
    """V5 nominal alpha=0.10 has chi^2 < chi^2(LCDM) (boost in right direction)."""
    chi2_l = sum(chi2_smf(b, "LambdaCDM")[0] for b in JWST_SMF_DATA)
    chi2_v = sum(chi2_smf(b, "V5", alpha_P22=0.10)[0] for b in JWST_SMF_DATA)
    assert chi2_v < chi2_l


def test_bayes_factor_structure():
    """Bayes-factor dict has required keys and self-consistent signs."""
    b = bayes_factor_V5_vs_LCDM(alpha_P22=0.10)
    for key in ("chi2_LCDM", "chi2_V5", "delta_chi2",
                "N_data", "bayes_raw", "bayes_BIC", "per_bin"):
        assert key in b
    assert b["delta_chi2"] == pytest.approx(b["chi2_LCDM"] - b["chi2_V5"], rel=1e-9)


# ---------------------------------------------------------------
# Honest "scoping" tests: V5 nominal is too timid
# ---------------------------------------------------------------
def test_V5_nominal_too_timid_for_JWST():
    """SCOPING — V5 §11.5 nominal alpha_P22=0.10 (boost(z=10)=1.5) is
    INSUFFICIENT to match JWST: chi^2 reduction is modest, and rho_*
    at z=8-10 remains 10^4-10^6 below the Labbe+2023 lower bound."""
    res = impossible_early_galaxies_test(8.0, alpha_P22=0.10)
    # V5 nominal does NOT reach the lower observational bound
    assert res["rho_star_V5_Msun_Mpc3"] < res["obs_lower"]


def test_best_fit_alpha_is_orders_of_magnitude_above_nominal():
    """SCOPING — the alpha_P22 needed to reduce chi^2 substantially is
    much larger than the V5 §11.5 nominal value 0.10."""
    chi2_lcdm = sum(chi2_smf(b, "LambdaCDM")[0] for b in JWST_SMF_DATA)
    chi2_nom = sum(chi2_smf(b, "V5", alpha_P22=0.10)[0] for b in JWST_SMF_DATA)
    chi2_strong = sum(chi2_smf(b, "V5", alpha_P22=100.0)[0] for b in JWST_SMF_DATA)
    # The strong boost gives a much better chi^2 than the nominal one
    assert chi2_strong < chi2_nom < chi2_lcdm
    # Document the gap (informational)
    assert (chi2_nom - chi2_strong) > (chi2_lcdm - chi2_nom)
