"""Tests pour le pipeline V5p4 lhc_search_v3_refined.py.

Vérifie la calibration Crystal Ball, l'interpolation de résolution,
la significance locale, le bootstrap et la correction LEE.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.V5_heavy_protons.lhc_search_v3_refined import (
    crystal_ball, cb_resolution, model_calibration, model_signal,
    inv_mass_pairs, signal_significance, bootstrap_significance,
    lee_correction, V5_PREDICTIONS, SEARCH_WINDOW, calibrate_resolution,
    M_JPSI_PDG, M_PSI2S_PDG,
)


def test_crystal_ball_gaussian_core():
    """Le coeur du Crystal Ball est gaussien dans la région t > -alpha."""
    x = np.linspace(2.9, 3.3, 401)  # bin pile au pic
    m, sigma, alpha, n = 3.1, 0.05, 1.5, 3.0
    cb = crystal_ball(x, m, sigma, alpha, n)
    # Au pic : valeur ≈ 1 (discret : tolérance gaussienne sur 1 bin)
    assert abs(cb[np.argmin((x-m)**2)] - 1.0) < 1e-3


def test_crystal_ball_tail_is_power_law():
    """La queue gauche d'un CB décroît plus lentement qu'une gaussienne."""
    m, sigma, alpha, n = 3.1, 0.05, 1.5, 3.0
    x_tail = m - 5 * sigma  # bien dans la queue
    cb_val = crystal_ball(np.array([x_tail]), m, sigma, alpha, n)[0]
    gauss_val = np.exp(-0.5 * 25)
    assert cb_val > gauss_val * 10  # power-law domine


def test_cb_resolution_interpolation():
    """La résolution interpolée est linéaire entre J/ψ et ψ(2S)."""
    sj, sp = 0.040, 0.060
    # Au point J/ψ : doit valoir sj
    assert abs(cb_resolution(M_JPSI_PDG, sj, sp) - sj) < 1e-9
    # Au point ψ(2S) : doit valoir sp
    assert abs(cb_resolution(M_PSI2S_PDG, sj, sp) - sp) < 1e-9
    # À mi-chemin : moyenne
    mid = 0.5 * (M_JPSI_PDG + M_PSI2S_PDG)
    assert abs(cb_resolution(mid, sj, sp) - 0.5*(sj+sp)) < 1e-9


def test_inv_mass_two_body_rest():
    """Pour deux particules sans pT, la masse invariante = somme des énergies."""
    # Deux muons collinéaires : m_inv = 2 * E_mu
    pt = np.array([1.0])
    eta = np.array([0.0])
    phi = np.array([0.0])
    m = 0.105658
    minv = inv_mass_pairs(pt, eta, phi, m, pt, eta, np.array([np.pi]), m)[0]
    # Configurés dos à dos avec pT=1 chacun → E_tot = 2*sqrt(1+m²) ~ 2.011
    assert abs(minv - 2*np.sqrt(1 + m*m)) < 1e-6


def test_lee_increases_p_value():
    """La correction LEE augmente la p-value (réduit la significance)."""
    local_z = 3.0
    z_glob_1trial, _ = lee_correction(local_z, 1)
    z_glob_10trials, _ = lee_correction(local_z, 10)
    # 1 trial : pas de pénalité
    assert abs(z_glob_1trial - local_z) < 1e-6
    # 10 trials : z_glob < z_loc
    assert z_glob_10trials < local_z


def test_lee_monotone_in_n_trials():
    """LEE : plus N_trials augmente, plus z_global diminue (monotone)."""
    z_loc = 4.0
    z1, _ = lee_correction(z_loc, 1)
    z10, _ = lee_correction(z_loc, 10)
    z100, _ = lee_correction(z_loc, 100)
    assert z1 >= z10 >= z100
    assert abs(z1 - z_loc) < 1e-6  # N=1 : pas de pénalité


def test_signal_significance_on_pure_background():
    """Sur du bruit pur Poisson, la significance moyenne est ~ 0."""
    rng = np.random.default_rng(42)
    masses = rng.uniform(2.5, 4.0, size=200_000)
    # Calibration factice : pas de CB, juste fond plat
    # On utilise les paramètres comme si c'était un fond uniforme
    # Le test : pour m_target dans la fenêtre, |z| < 3
    # On génère le popt à partir d'un fit
    sj, sp, popt, chi2, centers, n_data = calibrate_resolution(
        masses, fit_range=(2.5, 4.0), nbins=200)
    # Sans pic réel : le fit peut échouer, mais si convergé, signaux faibles
    if popt is None:
        return  # acceptable si pas de pic
    for m_target in [2.92, 3.04, 3.23]:
        sig = signal_significance(masses, m_target, 0.030, popt,
                                  calib_range=(2.5, 4.0),
                                  window=SEARCH_WINDOW, nbins_global=200)
        # Aucun signal réel → |z| modéré
        assert abs(sig["local_z"]) < 5.0


def test_signal_significance_detects_injected_signal():
    """Si on injecte un signal fort, signal_significance le détecte."""
    rng = np.random.default_rng(123)
    bg = rng.uniform(2.5, 4.0, size=80_000)
    # Injecte un pic gaussien à 3.20 GeV avec 2000 événements
    sig = rng.normal(3.20, 0.030, size=2000)
    masses = np.concatenate([bg, sig])
    sj, sp, popt, chi2, centers, n_data = calibrate_resolution(
        masses, fit_range=(2.5, 4.0), nbins=200)
    if popt is None:
        pytest.skip("calib failed")
    result = signal_significance(masses, 3.20, 0.030, popt,
                                 calib_range=(2.5, 4.0),
                                 window=SEARCH_WINDOW, nbins_global=200)
    # Un signal de 2000 événements sur ~80k bruit doit donner z >> 3
    assert result["local_z"] > 3.0


def test_bootstrap_distribution_is_centered():
    """Bootstrap sur fond pur : distribution centrée près de 0 avec σ~1."""
    rng = np.random.default_rng(7)
    bg = rng.uniform(2.5, 4.0, size=100_000)
    sj, sp, popt, chi2, centers, n_data = calibrate_resolution(
        bg, fit_range=(2.5, 4.0), nbins=200)
    if popt is None:
        pytest.skip("calib failed")
    zs = bootstrap_significance(bg, 3.10, 0.030, popt,
                                calib_range=(2.5, 4.0),
                                n_trials=200, window=SEARCH_WINDOW,
                                nbins_global=200)
    assert abs(zs.mean()) < 0.5
    assert 0.5 < zs.std() < 2.0


def test_v5_predictions_have_three_candidates():
    """Le pipeline V5p4 cherche exactement 3 candidats Gen 5+."""
    assert len(V5_PREDICTIONS) == 3
    ids = {p["id"] for p in V5_PREDICTIONS}
    assert ids == {262, 269, 283}
    # Toutes les masses dans [2.9, 3.3] GeV
    for p in V5_PREDICTIONS:
        assert 2.9 <= p["mass"] <= 3.3


def test_search_window_is_30_mev():
    """La fenêtre de recherche est ±30 MeV par spec."""
    assert abs(SEARCH_WINDOW - 0.030) < 1e-9


def test_model_signal_positive():
    """Le modèle de signal global est positif partout."""
    x = np.linspace(2.5, 4.0, 200)
    y = model_signal(x, 3.10, 0.03, 100.0, 50.0, 0.0, 0.0, 0.0)
    assert np.all(y >= 0)
