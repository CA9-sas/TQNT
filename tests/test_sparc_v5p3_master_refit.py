"""Tests pour la re-validation V5.6 du pipeline V5.3 master sur SPARC.

But : verrouiller que le re-fit standardise reproduit bien
    slope (d log c / d log M_h) = +0.575 +/- 0.038
revendique par la V5.3 master (commit 6d06ac6).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.V5_6_re_validation import sparc_refit_V5p3_master_standardized as mod


# ---------------------------------------------------------------------------
# Disponibilite des donnees
# ---------------------------------------------------------------------------

DATA_AVAILABLE = mod.SPARC_MAIN.exists() and mod.ROTMOD_DIR.exists()


pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="SPARC data not present at /mnt/data/data_recherche/sparc"
)


# ---------------------------------------------------------------------------
# Tests unitaires : fonctions analytiques
# ---------------------------------------------------------------------------


def test_M_halo_from_Vflat_milky_way():
    """Vflat = 200 km/s -> Mh = 1e12 Msun (calibration V5.3 master)."""
    M = mod.M_halo_from_Vflat(200.0)
    assert math.isclose(M, 1.0e12, rel_tol=1e-9)


def test_M_halo_from_Vflat_cubic_scaling():
    """Mh propto Vflat^3."""
    assert math.isclose(
        mod.M_halo_from_Vflat(400.0) / mod.M_halo_from_Vflat(200.0),
        8.0,
        rel_tol=1e-9,
    )


def test_r_vir_kpc_milky_way_scale():
    """Pour Mh = 1e12 Msun, r_vir doit etre ~200 kpc."""
    r = mod.r_vir_kpc(1.0e12)
    # Avec rho_crit = 140 Msun/kpc^3 et Delta=200 :
    #   r_vir = (3 * 1e12 / (4 pi * 200 * 140))^(1/3) ~ 207 kpc
    assert 150.0 < r < 260.0


def test_L_coh_J_finite_positive():
    rho = 1.0e6  # Msun/kpc^3 ~ disc density
    L = mod.L_coh_J_kpc(rho)
    assert L > 0 and math.isfinite(L)


def test_fit_log_log_slope_recovers_known_line():
    rng = np.random.default_rng(0)
    log_M = rng.uniform(-2, 2, size=100)
    true_slope = 0.42
    log_c = true_slope * log_M + 0.1
    slope, intercept = mod.fit_log_log_slope(log_M, log_c)
    assert math.isclose(slope, true_slope, abs_tol=1e-6)
    assert math.isclose(intercept, 0.1, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests integration : pipeline complet
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    return mod.run_pipeline(verbose=False)


def test_catalog_size(report):
    """SPARC Lelli+2016 : 175 galaxies au catalogue ; ~135 avec Vflat mesure."""
    assert report["n_galaxies_catalog"] >= 130
    assert report["n_galaxies_used"] >= 100


def test_slope_matches_v5p3_master_claim(report):
    """Le pipeline standardise reproduit slope = +0.575 +/- 0.038 a < 0.5 sigma.

    C'est LA verification cle du re-fit : V5.3 master annonce +0.575 +/- 0.038
    sur les 135 galaxies SPARC avec Vflat. La re-implementation doit donner
    le meme resultat (un ecart < 0.5 sigma est tolere : bootstrap, choix
    rho_b numeriques)."""
    slope = report["slope_measured"]
    err = report["slope_bootstrap_std"]
    diff = abs(slope - mod.V5p3_CLAIMED_SLOPE)
    quad_err = math.sqrt(err ** 2 + mod.V5p3_CLAIMED_SLOPE_ERR ** 2)
    assert diff < 0.5 * quad_err, (
        f"Slope re-mesure {slope:+.4f} incompatible avec claim V5.3 master "
        f"{mod.V5p3_CLAIMED_SLOPE:+.4f} +/- {mod.V5p3_CLAIMED_SLOPE_ERR:.4f}: "
        f"ecart {diff:.4f} (~{diff/quad_err:.1f} sigma combine)"
    )


def test_slope_incompatible_with_WB(report):
    """La pente mesuree est tres differente de WB (-0.10) -> tension reelle.

    Si la tension 18 sigma vs WB n'etait qu'un artefact, on devrait pouvoir
    invalider le test ici. On EXIGE au contraire une tension > 10 sigma :
    c'est la realite du diagnostic."""
    slope = report["slope_measured"]
    err = report["slope_bootstrap_std"]
    tension = abs(slope - mod.WB_SLOPE) / err
    assert tension > 10.0, (
        f"Tension vs WB attendue > 10 sigma, mesuree {tension:.1f} sigma "
        f"(slope = {slope:+.4f} +/- {err:.4f}, WB = {mod.WB_SLOPE:+.3f})"
    )


def test_slope_sign_positive(report):
    """Sign de la pente CGQ-Jeans : doit etre POSITIVE (inverse de WB)."""
    assert report["slope_measured"] > 0.3


def test_bootstrap_consistent_with_lstsq(report):
    """Bootstrap mean ne doit pas devier de >2 sigma_boot du fit lstsq."""
    slope = report["slope_measured"]
    boot = report["slope_bootstrap_mean"]
    err = report["slope_bootstrap_std"]
    assert abs(boot - slope) < 2.0 * err


def test_output_figure_exists():
    """Le figure de re-validation doit etre genere par le script principal."""
    out_png = mod.OUTPUT_DIR / "v5_6_v5p3_master_refit.png"
    # Si pas encore genere par l'execution manuelle, on le genere ici.
    if not out_png.exists():
        rpt = mod.run_pipeline(verbose=False)
        mod.plot_results(rpt, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 10_000  # PNG non vide
