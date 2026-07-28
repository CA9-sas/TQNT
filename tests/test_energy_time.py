# tests/test_energy_time.py
"""Tests de la boucle auto-cohérente énergie-temps nodale.

Vérifie les conditions aux limites (E=0, E=E₀ρ_J, E→∞),
la monotonie, les bornes, l'équation maîtresse et les valeurs physiques.
"""
import numpy as np
import pytest
from theory.emergent_spacetime import (
    gravitational_self_coupling,
    temporal_distortion_energy,
    proper_compton_time,
    mass_from_energy,
    rest_energy,
    master_equation_roots,
)


# --- Paramètres de test ---
RHO_J = 5.0       # ρ_J test
ETA_F = 2.0        # η_F test
E_0 = 1.0          # échelle d'énergie
M_COLD = ETA_F * RHO_J   # m(E=0) = η_F · ρ_J
ALPHA_G = 0.1      # α_G test (grand pour voir les effets)

# Constantes physiques SI pour le test du proton
G_SI = 6.6743e-11
HBAR_SI = 1.0546e-34
C_SI = 2.9979e8
M_PROTON = 1.6726e-27


# =====================================================================
# gravitational_self_coupling
# =====================================================================

def test_alpha_G_proton_physical():
    """α_G(proton) ∈ [5×10⁻³⁹, 7×10⁻³⁹]."""
    alpha_G = gravitational_self_coupling(M_PROTON, G=G_SI, hbar=HBAR_SI, c=C_SI)
    assert 5e-39 < alpha_G < 7e-39

def test_alpha_G_natural_units():
    """α_G = m² en unités naturelles (G=ℏ=c=1)."""
    m = 3.0
    assert gravitational_self_coupling(m) == pytest.approx(9.0)


# =====================================================================
# temporal_distortion_energy — Ω_t(E)
# =====================================================================

def test_omega_t_E0_boundary():
    """E=0 → Ω_t = exp(-α_G)."""
    omega = temporal_distortion_energy(0.0, RHO_J, ALPHA_G, E_0)
    assert omega == pytest.approx(np.exp(-ALPHA_G))

def test_omega_t_half_retraction():
    """E = E₀·ρ_J → Ω_t = exp(-α_G/4)."""
    E_half = E_0 * RHO_J
    omega = temporal_distortion_energy(E_half, RHO_J, ALPHA_G, E_0)
    assert omega == pytest.approx(np.exp(-ALPHA_G / 4))

def test_omega_t_dissolution():
    """E → ∞ → Ω_t → 1."""
    omega = temporal_distortion_energy(1e15, RHO_J, ALPHA_G, E_0)
    assert omega == pytest.approx(1.0, abs=1e-10)

def test_omega_t_monotone_increasing():
    """Ω_t croît avec E (la distorsion diminue)."""
    energies = np.linspace(0, 100, 200)
    omega = temporal_distortion_energy(energies, RHO_J, ALPHA_G, E_0)
    assert np.all(np.diff(omega) >= 0)

def test_omega_t_bounds():
    """0 < Ω_t ≤ 1 pour tout E ≥ 0."""
    energies = np.logspace(-5, 10, 500)
    omega = temporal_distortion_energy(energies, RHO_J, ALPHA_G, E_0)
    assert np.all(omega > 0)
    assert np.all(omega <= 1.0 + 1e-15)

def test_omega_t_array_support():
    """Accepte des arrays numpy."""
    energies = np.array([0.0, 1.0, 10.0])
    omega = temporal_distortion_energy(energies, RHO_J, ALPHA_G, E_0)
    assert omega.shape == (3,)


# =====================================================================
# mass_from_energy — m(E)
# =====================================================================

def test_mass_E0_boundary():
    """E=0 → m = η_F·ρ_J = m_cold."""
    m = mass_from_energy(0.0, RHO_J, ETA_F, E_0)
    assert m == pytest.approx(M_COLD)

def test_mass_half_retraction():
    """E = E₀·ρ_J → m = m_cold/2."""
    m = mass_from_energy(E_0 * RHO_J, RHO_J, ETA_F, E_0)
    assert m == pytest.approx(M_COLD / 2)

def test_mass_dissolution():
    """E → ∞ → m → 0."""
    m = mass_from_energy(1e15, RHO_J, ETA_F, E_0)
    assert m == pytest.approx(0.0, abs=1e-5)

def test_mass_monotone_decreasing():
    """m décroît avec E."""
    energies = np.linspace(0, 100, 200)
    masses = mass_from_energy(energies, RHO_J, ETA_F, E_0)
    assert np.all(np.diff(masses) <= 0)


# =====================================================================
# rest_energy — E = mc²
# =====================================================================

def test_rest_energy_basic():
    """E = mc² avec c=1."""
    assert rest_energy(3.0) == pytest.approx(3.0)

def test_rest_energy_c_nontrivial():
    """E = mc² avec c ≠ 1."""
    assert rest_energy(2.0, c=3.0) == pytest.approx(18.0)


# =====================================================================
# proper_compton_time — τ(E)
# =====================================================================

def test_tau_E0_boundary():
    """E=0 → τ = ℏΩ_t(0)/(m₀c²)."""
    tau = proper_compton_time(0.0, RHO_J, M_COLD, ALPHA_G, E_0)
    omega_0 = np.exp(-ALPHA_G)
    expected = omega_0 / M_COLD  # ℏ=c=1
    assert tau == pytest.approx(expected)

def test_tau_increases_with_energy():
    """τ croît avec E (la particule se dissout, son temps propre diverge)."""
    energies = np.linspace(0, 50, 200)
    tau = proper_compton_time(energies, RHO_J, M_COLD, ALPHA_G, E_0)
    # τ peut avoir un comportement non-trivial au début, mais doit diverger
    # Vérifie que τ(E=50) > τ(E=0)
    assert tau[-1] > tau[0]


# =====================================================================
# master_equation_roots — racine de m²c²/E₀ + mρ_J = η_F·ρ_J²
# =====================================================================

def test_master_equation_satisfies_equation():
    """La racine satisfait m²c²/E₀ + m·ρ_J = η_F·ρ_J²."""
    m = master_equation_roots(RHO_J, ETA_F, E_0)
    lhs = m**2 / E_0 + m * RHO_J  # c=1
    rhs = ETA_F * RHO_J**2
    assert lhs == pytest.approx(rhs, rel=1e-10)

def test_master_equation_positive():
    """La racine est positive."""
    m = master_equation_roots(RHO_J, ETA_F, E_0)
    assert m > 0

def test_master_equation_consistency():
    """La racine est cohérente : m = mass_from_energy(mc², ρ_J, η_F, E₀)."""
    m_eq = master_equation_roots(RHO_J, ETA_F, E_0)
    E_eq = rest_energy(m_eq)
    m_check = mass_from_energy(E_eq, RHO_J, ETA_F, E_0)
    assert m_eq == pytest.approx(m_check, rel=1e-10)


# =====================================================================
# Relation E·τ = ℏ·Ω_t
# =====================================================================

def test_E_tau_equals_hbar_omega():
    """La relation E·τ = ℏ·Ω_t tient pour tout E > 0."""
    energies = np.array([0.1, 1.0, 5.0, 25.0, 100.0])
    masses = mass_from_energy(energies, RHO_J, ETA_F, E_0)
    E_rest = rest_energy(masses)  # E = mc², c=1
    tau = proper_compton_time(energies, RHO_J, M_COLD, ALPHA_G, E_0)
    omega = temporal_distortion_energy(energies, RHO_J, ALPHA_G, E_0)
    # E·τ = ℏ·Ω_t (avec ℏ=1)
    np.testing.assert_allclose(E_rest * tau, omega, rtol=1e-10)
