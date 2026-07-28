"""Tests V5.9 — Ondes gravitationnelles depuis χ-field (régime non-conforme).

Vérifie :
  1. Projecteur TT : transverse et sans trace.
  2. Moment quadrupolaire (forme générale et binaire).
  3. Chirp mass (formule MTW).
  4. Strain GW pour binaire (formule quadrupolaire, ordre de grandeur).
  5. Énergie rayonnée Peters 1964 (signe et scaling).
  6. Backreaction da/dt (scaling 1/a³).
  7. Temps de coalescence (formule Peters 1964).
  8. Évolution inspiral (chirp f_GW(t) ∝ τ⁻³/⁸).
  9. Validation GW150914 (chirp mass 28 M_sun, durée ~200 ms).
  10. SGWB NANOGrav (Ω_GW à 1 nHz).
  11. Cohérence V5 ↔ G_CODATA dans le strain.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.gravitational_waves import (
    C_LIGHT,
    G_CODATA,
    G_SI,
    M_SUN,
    MPC,
    chirp_frequency,
    chirp_mass,
    coalescence_time,
    GW_energy_loss_rate,
    GW_strain_binary,
    inspiral_evolution,
    linearized_TT_perturbation,
    orbital_separation_evolution_rate,
    project_TT,
    quadrupole_moment,
    quadrupole_binary_circular,
    sgwb_omega_gw_smbhb,
    summary,
    time_to_merger_from_frequency,
    TT_projector,
    T_ij_TT_binary,
    validate_GW150914,
)


# ---------------------------------------------------------------------------
# 1. Projecteur TT.
# ---------------------------------------------------------------------------
def test_TT_projector_traceless():
    """Trace sur les deux derniers indices : Λ_{ij,kk} = 0."""
    Lam = TT_projector(np.array([0.0, 0.0, 1.0]))
    trace = np.einsum("ijkk->ij", Lam)
    np.testing.assert_allclose(trace, 0.0, atol=1e-14)


def test_TT_projector_transverse_to_k():
    """k^i Λ_{ij,kl} = 0."""
    k = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    Lam = TT_projector(k)
    contraction = np.einsum("i,ijkl->jkl", k, Lam)
    np.testing.assert_allclose(contraction, 0.0, atol=1e-14)


def test_project_TT_traceless():
    """h^TT_ii = 0 (sans trace)."""
    h = np.array([[1.0, 0.5, 0.3],
                  [0.5, -0.5, 0.2],
                  [0.3, 0.2, 0.7]])
    h_TT = project_TT(h, np.array([0.0, 0.0, 1.0]))
    assert abs(np.trace(h_TT)) < 1e-14


def test_project_TT_transverse():
    """k^i h^TT_ij = 0."""
    h = np.array([[2.0, 0.3, 0.1],
                  [0.3, -1.0, 0.4],
                  [0.1, 0.4, 0.5]])
    k = np.array([0.0, 0.0, 1.0])
    h_TT = project_TT(h, k)
    np.testing.assert_allclose(k @ h_TT, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Moment quadrupolaire.
# ---------------------------------------------------------------------------
def test_quadrupole_traceless():
    """I_ij sans trace."""
    m = np.array([1.0, 2.0, 3.0])
    pos = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.5, 0.0],
                    [0.5, -0.5, 0.7]])
    I = quadrupole_moment(m, pos)
    assert abs(np.trace(I)) < 1e-12


def test_quadrupole_binary_xy_signature():
    """Quadrupôle d'un binaire au temps t=0 : non-nul dans xx, yy."""
    m1, m2 = 36 * M_SUN, 29 * M_SUN
    a = 1.0e7  # 10 000 km
    I = quadrupole_binary_circular(m1, m2, a, t=0.0)
    # À t=0 : masses sur axe x → I_xx > 0, I_yy < 0 (trace nulle)
    assert I[0, 0] > 0
    assert I[1, 1] < 0
    # Trace ≃ 0 (relative au plus grand élément), précision float ~ 1e-15.
    assert abs(np.trace(I)) / abs(I[0, 0]) < 1e-12


# ---------------------------------------------------------------------------
# 3. Chirp mass.
# ---------------------------------------------------------------------------
def test_chirp_mass_GW150914():
    """Chirp mass GW150914 : (36 × 29)^{3/5} / (65)^{1/5} ≈ 28,1 M_sun."""
    M_c = chirp_mass(36.0 * M_SUN, 29.0 * M_SUN) / M_SUN
    assert M_c == pytest.approx(28.1, rel=0.01)


def test_chirp_mass_equal_masses():
    """m1 = m2 = m → M_c = (m²)^{3/5}/(2m)^{1/5} = m / 2^{1/5}."""
    m = 30.0 * M_SUN
    M_c = chirp_mass(m, m)
    expected = m / 2.0 ** (1.0 / 5.0)
    assert M_c == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. Strain GW pour binaire.
# ---------------------------------------------------------------------------
def test_GW_strain_GW150914_order_of_magnitude():
    """h_+ pour GW150914 à f_GW = 150 Hz, d = 410 Mpc : ~10⁻²¹.

    Acceptance : ordre de grandeur (10⁻²¹) à un facteur 5.
    """
    h_p, _ = GW_strain_binary(
        m1=36 * M_SUN, m2=29 * M_SUN,
        distance=410 * MPC, f_gw=150.0, iota=0.0
    )
    assert 1e-22 < abs(h_p) < 5e-21


def test_GW_strain_distance_scaling():
    """h ∝ 1/d : doubler d divise h par 2."""
    args = dict(m1=30 * M_SUN, m2=30 * M_SUN, f_gw=100.0)
    h1, _ = GW_strain_binary(distance=1 * MPC, **args)
    h2, _ = GW_strain_binary(distance=2 * MPC, **args)
    assert h1 / h2 == pytest.approx(2.0, rel=1e-10)


def test_GW_strain_frequency_scaling():
    """h ∝ f^{2/3} pour M_c constant."""
    args = dict(m1=30 * M_SUN, m2=30 * M_SUN, distance=100 * MPC)
    h_low, _ = GW_strain_binary(f_gw=50.0, **args)
    h_hi, _ = GW_strain_binary(f_gw=200.0, **args)
    ratio = h_hi / h_low
    expected = (200.0 / 50.0) ** (2.0 / 3.0)
    assert ratio == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 5. Énergie rayonnée (Peters 1964).
# ---------------------------------------------------------------------------
def test_GW_energy_loss_negative():
    """dE/dt < 0 (la binaire perd de l'énergie)."""
    dE_dt = GW_energy_loss_rate(36 * M_SUN, 29 * M_SUN, a=1e9)
    assert dE_dt < 0


def test_GW_energy_loss_a_scaling():
    """dE/dt ∝ 1/a⁵."""
    m1, m2 = 30 * M_SUN, 30 * M_SUN
    dE1 = GW_energy_loss_rate(m1, m2, a=1e9)
    dE2 = GW_energy_loss_rate(m1, m2, a=2e9)
    ratio = dE1 / dE2
    # |dE|(a) ∝ 1/a^5 → ratio(a=a0)/(a=2a0) = 2^5 = 32
    assert ratio == pytest.approx(32.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 6. Backreaction da/dt.
# ---------------------------------------------------------------------------
def test_da_dt_negative():
    """da/dt < 0 : séparation décroit."""
    da_dt = orbital_separation_evolution_rate(30 * M_SUN, 30 * M_SUN, a=1e9)
    assert da_dt < 0


def test_da_dt_a_scaling():
    """da/dt ∝ 1/a³."""
    m1, m2 = 30 * M_SUN, 30 * M_SUN
    d1 = orbital_separation_evolution_rate(m1, m2, a=1e9)
    d2 = orbital_separation_evolution_rate(m1, m2, a=2e9)
    assert d1 / d2 == pytest.approx(8.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 7. Temps de coalescence.
# ---------------------------------------------------------------------------
def test_coalescence_time_positive():
    """t_coal > 0."""
    t = coalescence_time(36 * M_SUN, 29 * M_SUN, a0=1e7)
    assert t > 0


def test_coalescence_time_a_scaling():
    """t_coal ∝ a⁴."""
    m1, m2 = 30 * M_SUN, 30 * M_SUN
    t1 = coalescence_time(m1, m2, 1e7)
    t2 = coalescence_time(m1, m2, 2e7)
    assert t2 / t1 == pytest.approx(16.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 8. Chirp f_GW(τ).
# ---------------------------------------------------------------------------
def test_chirp_frequency_scaling():
    """f_GW ∝ τ^{-3/8}."""
    m1, m2 = 36 * M_SUN, 29 * M_SUN
    f1 = chirp_frequency(m1, m2, 1.0)
    f2 = chirp_frequency(m1, m2, 0.1)
    ratio = f2 / f1
    expected = (1.0 / 0.1) ** (3.0 / 8.0)
    assert ratio == pytest.approx(expected, rel=1e-10)


def test_chirp_frequency_inverse():
    """time_to_merger_from_frequency ∘ chirp_frequency = identité."""
    m1, m2 = 36 * M_SUN, 29 * M_SUN
    tau = 0.5
    f = chirp_frequency(m1, m2, tau)
    tau_rec = time_to_merger_from_frequency(m1, m2, f)
    assert tau_rec == pytest.approx(tau, rel=1e-10)


# ---------------------------------------------------------------------------
# 9. Inspiral evolution.
# ---------------------------------------------------------------------------
def test_inspiral_evolution_chirp():
    """f_GW(t) doit augmenter au cours du temps (chirp ascendant)."""
    m1, m2 = 36 * M_SUN, 29 * M_SUN
    # On part d'une grande séparation pour avoir un chirp lent
    a0 = 1e8
    res = inspiral_evolution(m1, m2, a0, t_max=coalescence_time(m1, m2, a0) * 0.5)
    f = res["f_gw"]
    # f doit être croissante
    assert all(f[i + 1] >= f[i] for i in range(0, len(f) - 1, 100))


def test_inspiral_a_decreasing():
    """a(t) doit décroître."""
    m1, m2 = 36 * M_SUN, 29 * M_SUN
    a0 = 1e8
    res = inspiral_evolution(m1, m2, a0, t_max=coalescence_time(m1, m2, a0) * 0.5)
    a = res["a"]
    assert a[0] > a[-1]


# ---------------------------------------------------------------------------
# 10. Validation GW150914.
# ---------------------------------------------------------------------------
def test_validate_GW150914_chirp_mass():
    """Chirp mass GW150914 = 28,1 M_sun (LIGO PRL 116, 061102, 2016)."""
    res = validate_GW150914()
    assert res.chirp_mass_msun == pytest.approx(28.1, rel=0.02)


def test_validate_GW150914_strain_order():
    """h_peak prédit dans la fenêtre 10⁻²² – 10⁻²⁰."""
    res = validate_GW150914()
    assert 1e-22 < res.h_peak_predicted < 1e-20


def test_validate_GW150914_chirp_duration():
    """Le chirp depuis 35 Hz dure ~200 ms (LIGO observation : ~200 ms)."""
    res = validate_GW150914()
    # 35 Hz → fusion : ~200 ms pour GW150914
    assert 0.1 < res.t_chirp_s < 0.3


# ---------------------------------------------------------------------------
# 11. SGWB NANOGrav.
# ---------------------------------------------------------------------------
def test_sgwb_omega_gw_order_of_magnitude():
    """Ω_GW(1 nHz) ~ 10⁻⁹ – 10⁻⁷ (NANOGrav 15yr range)."""
    omega = sgwb_omega_gw_smbhb(1e-9)
    # Avec A=2.4e-15 et H0=67.4, on est attendu ~1e-9 à 1 nHz
    assert 1e-11 < omega < 1e-6


# ---------------------------------------------------------------------------
# 12. Linearized TT perturbation.
# ---------------------------------------------------------------------------
def test_linearized_TT_inverse_distance():
    """h^TT ∝ 1/distance."""
    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, 0.0]])
    h1 = linearized_TT_perturbation(T, omega=1.0, distance=1.0)
    h2 = linearized_TT_perturbation(T, omega=1.0, distance=2.0)
    assert np.allclose(h1 / 2.0, h2)


def test_linearized_TT_inverse_omega2():
    """h^TT ∝ 1/ω²."""
    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, 0.0]])
    h_lo = linearized_TT_perturbation(T, omega=1.0, distance=1.0)
    h_hi = linearized_TT_perturbation(T, omega=10.0, distance=1.0)
    assert np.allclose(h_lo / 100.0, h_hi)


def test_linearized_TT_invalid_args():
    """Erreur si ω ≤ 0 ou distance ≤ 0."""
    T = np.zeros((3, 3))
    with pytest.raises(ValueError):
        linearized_TT_perturbation(T, omega=0.0, distance=1.0)
    with pytest.raises(ValueError):
        linearized_TT_perturbation(T, omega=1.0, distance=-1.0)


# ---------------------------------------------------------------------------
# 13. T_ij^TT binary.
# ---------------------------------------------------------------------------
def test_T_ij_TT_binary_traceless():
    """d²I^TT/dt² a une trace nulle (relativement aux composantes max)."""
    Q2 = T_ij_TT_binary(36 * M_SUN, 29 * M_SUN, a=1e7, t=0.0)
    norm = np.max(np.abs(Q2))
    assert abs(np.trace(Q2)) / norm < 1e-12


# ---------------------------------------------------------------------------
# 14. Summary.
# ---------------------------------------------------------------------------
def test_summary_has_expected_keys():
    """summary() retourne toutes les clés clés."""
    s = summary()
    expected = {
        "G_V5_used", "G_codata", "G_error_pct",
        "gw150914_chirp_mass_msun", "gw150914_h_peak_predicted",
        "gw150914_h_peak_observed", "gw150914_relative_error",
        "gw150914_t_chirp_s", "TT_projector_trace_max",
        "sgwb_omega_gw_1nHz",
    }
    assert expected.issubset(s.keys())


def test_summary_G_within_half_percent():
    """G utilisé dans V5.9 : <0,5 % de CODATA (héritage V5.8)."""
    s = summary()
    assert abs(s["G_error_pct"]) < 0.5
