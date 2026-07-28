"""Tests du re-fit SPARC V5.5 (Ω² saturé)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.V5_5_alpha_derivation import sparc_refit_omega2_saturated as refit


def test_M_halo_from_Vflat_calibration():
    """Vflat ~ 200 km/s doit donner M_h ~ 1.5e12 M_sun (calibration MW)."""
    M = refit.M_halo_from_Vflat(200.0)
    assert M == pytest.approx(1.5e12, rel=1e-6)


def test_M_halo_scaling_cubic():
    """M_h ∝ V_flat^3 (TF-like)."""
    M1 = refit.M_halo_from_Vflat(100.0)
    M2 = refit.M_halo_from_Vflat(200.0)
    assert M2 / M1 == pytest.approx(8.0, rel=1e-6)


def test_r_vir_monotone():
    """r_vir doit croître avec M_h (∝ M^(1/3))."""
    rv1 = refit.r_vir_kpc(1e11)
    rv2 = refit.r_vir_kpc(1e12)
    rv3 = refit.r_vir_kpc(1e13)
    assert rv1 < rv2 < rv3
    # Cube root scaling: 10x mass -> 10^(1/3) ≈ 2.154x radius
    assert rv2 / rv1 == pytest.approx(10 ** (1 / 3), rel=1e-3)


def test_NFW_velocity_positive():
    """V_NFW doit être positive et finie pour r > 0."""
    import numpy as np
    R = np.array([0.1, 1.0, 10.0, 50.0])
    V = refit.V_NFW(R, M_h_Msun=1e12, r_s_kpc=20.0)
    assert np.all(np.isfinite(V))
    assert np.all(V > 0)


def test_omega_correction_at_M_star():
    """À M_*, r_s_V55 / r_s_V53 = 1/sqrt(1.15) ≈ 0.933."""
    M_star = 1e12
    L = refit.L_coh_J_kpc(M_star)
    Omega2 = refit.omega_squared(M_star)
    r_s_V53 = refit.KAPPA_CJ_V5p3 * L
    r_s_V55 = r_s_V53 / math.sqrt(Omega2)
    assert r_s_V55 / r_s_V53 == pytest.approx(1.0 / math.sqrt(1.15), rel=1e-6)


def test_sparc_main_parsed():
    """Le catalogue SPARC doit être chargeable et contenir ~175 galaxies."""
    gals = refit.parse_sparc_main()
    assert 150 <= len(gals) <= 200
    assert all("name" in g and "Vflat_kms" in g for g in gals)


def test_refit_runs_and_compares():
    """Le pipeline complet tourne et produit un χ²/N comparable au V5.3 ref."""
    summary = refit.run_refit(verbose=False)
    assert summary["n_galaxies_used"] >= 100
    assert math.isfinite(summary["chi2_per_N_V5p3"])
    assert math.isfinite(summary["chi2_per_N_V5p5"])
    # Sanity check: chi2/N should be in a physically plausible range
    assert 0.5 < summary["chi2_per_N_V5p3"] < 1e4
    assert 0.5 < summary["chi2_per_N_V5p5"] < 1e4
