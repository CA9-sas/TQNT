"""Tests for V5.20 IceCube neutrino sub-PeV test.

scripts/V5_IceCube/v5p20_neutrino_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_IceCube"))

import v5p20_neutrino_test as nu_mod
from v5p20_neutrino_test import (
    E_CASCADE_GEV,
    F_NU_BRANCHING,
    N_NU_PER_EVENT,
    GAMMA_CASCADE,
    E_CUTOFF_GEV,
    E_MIN_GEV,
    ICECUBE_PHI0,
    ICECUBE_E0_GEV,
    ICECUBE_GAMMA,
    ICECUBE_E_MIN_GEV,
    ICECUBE_E_MAX_GEV,
    icecube_diffuse_flux,
    atmospheric_neutrino_flux,
    v5p20_neutrino_spectrum_per_event,
    v5p20_event_density,
    v5p20_diffuse_flux,
    compare_v5_vs_icecube,
    verdict,
    plot_v5_vs_icecube,
)


# ----------------------------------------------------------------------------
# 1. Constantes physiques V5.20
# ----------------------------------------------------------------------------

def test_cascade_energy_around_1p5_GeV():
    """L'energie de cascade par evenement est ~1,5 GeV (pearl 6_3+6_3)."""
    assert 1.0 <= E_CASCADE_GEV <= 3.0


def test_nu_branching_fraction_in_physical_range():
    """Branching ratio neutrinique : 10-25 % (cascade hadronic GeV)."""
    assert 0.05 <= F_NU_BRANCHING <= 0.30


def test_cascade_spectral_index_soft():
    """L'indice spectral cascade est >= 2 (soft)."""
    assert GAMMA_CASCADE >= 2.0
    assert GAMMA_CASCADE <= 3.5


def test_E_cutoff_below_TeV():
    """Le cutoff cascade est largement sous la bande IceCube."""
    assert E_CUTOFF_GEV < 10.0
    assert E_CUTOFF_GEV < ICECUBE_E_MIN_GEV / 1000.0


# ----------------------------------------------------------------------------
# 2. IceCube reference
# ----------------------------------------------------------------------------

def test_icecube_flux_at_100TeV_matches_normalisation():
    """phi_IC(100 TeV) = phi_0 (par definition)."""
    phi = float(icecube_diffuse_flux(np.array([ICECUBE_E0_GEV]))[0])
    assert phi == pytest.approx(ICECUBE_PHI0, rel=1e-6)


def test_icecube_flux_falls_as_power_law():
    """phi_IC scales as E^{-gamma}."""
    E1 = 1.0e5
    E2 = 1.0e6
    phi1 = float(icecube_diffuse_flux(np.array([E1]))[0])
    phi2 = float(icecube_diffuse_flux(np.array([E2]))[0])
    expected_ratio = (E2 / E1) ** (-ICECUBE_GAMMA)
    assert phi2 / phi1 == pytest.approx(expected_ratio, rel=1e-6)


def test_atmospheric_dominates_below_100GeV():
    """A 10 GeV, le flux atmospherique est largement plus grand que celui de
    l'astrophysique IceCube extrapole vers le bas."""
    E = np.array([10.0])
    phi_atm = float(atmospheric_neutrino_flux(E)[0])
    phi_ic = float(icecube_diffuse_flux(E)[0])
    assert phi_atm > phi_ic


# ----------------------------------------------------------------------------
# 3. Spectre V5.20 par evenement
# ----------------------------------------------------------------------------

def test_spectrum_normalised_to_n_nu():
    """Integrale du spectre dN/dE entre E_min et E_cut = N_NU_PER_EVENT."""
    E_grid = np.linspace(E_MIN_GEV, E_CUTOFF_GEV, 5000)
    dN_dE = v5p20_neutrino_spectrum_per_event(E_grid)
    integral = np.trapezoid(dN_dE, E_grid) \
        if hasattr(np, "trapezoid") else np.trapz(dN_dE, E_grid)
    assert integral == pytest.approx(N_NU_PER_EVENT, rel=1e-2)


def test_spectrum_zero_below_E_min():
    """dN/dE = 0 sous E_min."""
    E = np.array([E_MIN_GEV / 2.0])
    dN_dE = v5p20_neutrino_spectrum_per_event(E)
    assert float(dN_dE[0]) == 0.0


def test_spectrum_exp_cutoff_above_E_cut():
    """Au dessus de E_cut, le spectre est exponentiellement supprime."""
    E_cut = E_CUTOFF_GEV
    E_above = np.array([E_cut * 5.0])
    dN_above = float(v5p20_neutrino_spectrum_per_event(E_above)[0])
    E_at = np.array([E_cut * 1.001])
    dN_at = float(v5p20_neutrino_spectrum_per_event(E_at)[0])
    # Suppression d'au moins 4 ordres a 5 * E_cut (exp(-4) ~ 0.018)
    assert dN_above < dN_at * 1e-1


# ----------------------------------------------------------------------------
# 4. Densite d'evenements
# ----------------------------------------------------------------------------

def test_event_density_positive_and_finite():
    """n_event_dot > 0 et finie."""
    n_dot = v5p20_event_density()
    assert np.isfinite(n_dot)
    assert n_dot > 0


def test_event_density_in_expected_range():
    """Pour delta_c=10, on attend n_dot ~ 1e-7 a 1e-3 m^-3 s^-1 (10^90 a
    10^94 evenements dans le volume de Hubble sur t_Hubble)."""
    n_dot = v5p20_event_density()
    assert 1e-10 < n_dot < 1e-2


# ----------------------------------------------------------------------------
# 5. Flux diffus V5.20
# ----------------------------------------------------------------------------

def test_v5_diffuse_flux_zero_at_high_energy():
    """A 1 TeV (au-dessus du cutoff), phi_V5 est negligeable."""
    phi = float(v5p20_diffuse_flux(np.array([1.0e3]))[0])
    assert phi < 1e-20   # plusieurs ordres sous IceCube ~ 7e-13


def test_v5_diffuse_flux_strictly_negligible_in_IceCube_band():
    """Dans la bande IceCube [16 TeV, 2.6 PeV], V5.20 est ~ 0."""
    E_band = np.logspace(
        math.log10(ICECUBE_E_MIN_GEV), math.log10(ICECUBE_E_MAX_GEV), 20
    )
    phi_v5 = v5p20_diffuse_flux(E_band)
    phi_ic = icecube_diffuse_flux(E_band)
    # phi_V5 < 1e-9 * phi_IceCube partout dans la bande
    ratio_max = float(np.max(phi_v5 / phi_ic))
    assert ratio_max < 1e-9


def test_v5_diffuse_flux_positive_below_cutoff():
    """Sous le cutoff, phi_V5 > 0."""
    phi = float(v5p20_diffuse_flux(np.array([0.5]))[0])
    assert phi > 0


# ----------------------------------------------------------------------------
# 6. Comparaison et verdict
# ----------------------------------------------------------------------------

def test_compare_returns_expected_keys():
    """compare_v5_vs_icecube renvoie tous les champs attendus."""
    E = np.logspace(0, 6, 10)
    cmp = compare_v5_vs_icecube(E)
    for key in ["E_GeV", "phi_V5", "phi_IceCube", "phi_atm", "ratio"]:
        assert key in cmp
    assert cmp["phi_V5"].shape == E.shape


def test_verdict_status_is_sub_dominant():
    """Le verdict V5.20 vs IceCube est sub-dominant (cutoff << TeV)."""
    E = np.logspace(-2, 8, 200)
    v = verdict(E)
    assert "SUB-DOMINANT" in v["status"]


def test_verdict_ratio_1TeV_below_unity():
    """A 1 TeV, ratio V5/IceCube << 1."""
    E = np.logspace(-2, 8, 200)
    v = verdict(E)
    assert v["ratio_1TeV"] < 1.0


def test_verdict_reports_event_density():
    """Le verdict expose n_event_density."""
    E = np.logspace(-2, 8, 50)
    v = verdict(E)
    assert "n_event_density_m3_s" in v
    assert v["n_event_density_m3_s"] > 0


# ----------------------------------------------------------------------------
# 7. Plot integration
# ----------------------------------------------------------------------------

def test_plot_creates_output_file(tmp_path):
    """plot_v5_vs_icecube ecrit un fichier PNG."""
    out = tmp_path / "v5p20_icecube_test.png"
    plot_v5_vs_icecube(out)
    assert out.exists()
    assert out.stat().st_size > 1000   # PNG non trivial
