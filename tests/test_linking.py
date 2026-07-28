"""Tests pour theory/linking.py — Entrelacs neutrino-photon (Hopf link minimal)"""
import math
import pytest

from theory.neutrino import HBAR_C_MEV_FM, boomerang_radius
from theory.linking import (
    linking_cross_section,
    spectral_matching_energy,
    planck_spectral_density,
    photon_density_at_wavelength,
    interaction_rate_density,
    mean_free_path,
    total_events_hubble,
    critical_temperature,
    critical_redshift,
    linking_rate_per_neutrino,
    linking_onset_mass,
    linking_mean_free_path_neutrino,
    hopf_link_properties,
    T_CMB, N_GAMMA_TOTAL, K_B_EV_PER_K, HBAR_C_EV_CM, T_HUBBLE_S,
)


class TestLinkingCrossSection:
    """Tests pour la section efficace géométrique σ = π R²."""

    def test_sigma_positive(self):
        """σ > 0 pour tout R > 0."""
        for R in [1.0, 100.0, 1e8]:
            assert linking_cross_section(R) > 0

    def test_sigma_proportional_R_squared(self):
        """σ ∝ R² : doubler R → quadrupler σ."""
        R = 1000.0
        sigma1 = linking_cross_section(R)
        sigma2 = linking_cross_section(2 * R)
        assert abs(sigma2 / sigma1 - 4.0) < 1e-10

    def test_sigma_173nm(self):
        """σ(R=173 nm) ≈ 9.4 × 10⁻¹⁰ cm² (valeur du plan)."""
        R_fm = 173e-9 * 1e15  # 173 nm → fm = 1.73e8 fm
        sigma_fm2 = linking_cross_section(R_fm)
        sigma_cm2 = sigma_fm2 * 1e-26  # fm² → cm²
        # π(1.73e8)² fm² = π · 2.993e16 fm² ≈ 9.4e16 fm² = 9.4e-10 cm²
        assert abs(sigma_cm2 - 9.4e-10) / 9.4e-10 < 0.01

    def test_sigma_formula_exact(self):
        """σ = π R² exactement."""
        R = 42.0
        assert abs(linking_cross_section(R) - math.pi * R ** 2) < 1e-12


class TestSpectralCondition:
    """Tests pour la condition spectrale E_match = ℏc/R."""

    def test_E_match_positive(self):
        """E_match > 0 pour tout R > 0."""
        assert spectral_matching_energy(1.0) > 0

    def test_E_match_173nm_in_eV(self):
        """E_match(R=173 nm) ≈ 1.14 eV (infrarouge proche)."""
        R_fm = 1.73e8  # 173 nm en fm
        E_MeV = spectral_matching_energy(R_fm)
        E_eV = E_MeV * 1e6
        assert abs(E_eV - 1.14) / 1.14 < 0.01

    def test_E_match_decreases_with_R(self):
        """E ∝ 1/R : plus grand rayon → plus petite énergie."""
        E1 = spectral_matching_energy(100.0)
        E2 = spectral_matching_energy(1000.0)
        assert E1 > E2


class TestPlanckDistribution:
    """Tests pour la distribution spectrale de Planck."""

    def test_planck_positive_at_finite_T(self):
        """dn/dE > 0 pour E > 0 et T > 0."""
        assert planck_spectral_density(0.1, 5000.0) > 0

    def test_planck_zero_at_zero_T(self):
        """dn/dE = 0 pour T = 0."""
        assert planck_spectral_density(0.1, 0.0) == 0.0

    def test_planck_integral_CMB(self):
        """Intégrale numérique ≈ 411 cm⁻³ pour T_CMB = 2.7255 K.

        n_total = ∫₀^∞ (dn/dE) dE ≈ 2·ζ(3)/π² · (k_B T / ℏc)³ · 8π = 411 cm⁻³
        """
        # Intégration par trapèzes sur [0.0001, 0.01] eV (couvre 99%+ du CMB)
        N = 10000
        E_min, E_max = 1e-5, 0.01  # eV
        dE = (E_max - E_min) / N
        total = 0.0
        for i in range(N):
            E = E_min + (i + 0.5) * dE
            total += planck_spectral_density(E, T_CMB) * dE
        # Tolérance de 2% car on tronque les queues
        assert abs(total - N_GAMMA_TOTAL) / N_GAMMA_TOTAL < 0.02

    def test_planck_exponentially_suppressed_at_1eV_CMB(self):
        """dn/dE(1 eV, T_CMB) ~ exp(−4250) ≈ 0."""
        val = planck_spectral_density(1.0, T_CMB)
        assert val == 0.0  # x = 1/(8.617e-5 × 2.73) ≈ 4250 >> 700

    def test_planck_peak_shifts_with_temperature(self):
        """Le pic se déplace vers les hautes énergies avec T."""
        # À T=5000 K, pic ≈ 2.82 k_B T ≈ 1.21 eV
        # À T=3 K, pic ≈ 0.73 meV
        d_hot = planck_spectral_density(1.0, 5000.0)
        d_cold = planck_spectral_density(1.0, 300.0)
        assert d_hot > d_cold


class TestInteractionRate:
    """Tests pour le taux d'interaction volumique."""

    def test_rate_positive(self):
        """Γ/V > 0 pour n_γ, n_ν, σ > 0."""
        rate = interaction_rate_density(100.0, 56.0, 1e16)
        assert rate > 0

    def test_rate_proportional_to_density(self):
        """Γ/V ∝ n_γ : doubler n_γ → doubler Γ/V."""
        r1 = interaction_rate_density(100.0, 56.0, 1e16)
        r2 = interaction_rate_density(200.0, 56.0, 1e16)
        assert abs(r2 / r1 - 2.0) < 1e-10

    def test_rate_increases_with_temperature(self):
        """Taux croît avec T car n_γ(E_match) croît."""
        R_fm = 1.73e8
        sigma = linking_cross_section(R_fm)
        E_eV = spectral_matching_energy(R_fm) * 1e6

        n1 = planck_spectral_density(E_eV, 3000.0) * 1.0
        n2 = planck_spectral_density(E_eV, 6000.0) * 1.0
        r1 = interaction_rate_density(n1, 56.0, sigma)
        r2 = interaction_rate_density(n2, 56.0, sigma)
        assert r2 > r1

    def test_rate_zero_at_CMB(self):
        """Taux = 0 pour le CMB actuel (exp(−4250) suppression)."""
        R_fm = 1.73e8
        sigma = linking_cross_section(R_fm)
        E_eV = spectral_matching_energy(R_fm) * 1e6
        n_gamma = planck_spectral_density(E_eV, T_CMB)
        rate = interaction_rate_density(n_gamma, N_GAMMA_TOTAL, sigma)
        assert rate == 0.0


class TestMeanFreePath:
    """Tests pour le libre parcours moyen."""

    def test_mfp_positive(self):
        """l > 0 pour n > 0 et σ > 0."""
        assert mean_free_path(100.0, 1e16) > 0

    def test_mfp_infinite_for_zero_density(self):
        """l = ∞ si n = 0."""
        assert mean_free_path(0.0, 1e16) == float('inf')

    def test_mfp_decreases_with_density(self):
        """l ∝ 1/n : plus de cibles → plus court parcours."""
        l1 = mean_free_path(100.0, 1e16)
        l2 = mean_free_path(1000.0, 1e16)
        assert l1 > l2


class TestHopfLinkProperties:
    """Tests pour les propriétés du Hopf link."""

    def test_linking_number(self):
        """Linking number = 1."""
        props = hopf_link_properties(1.73e8)
        assert props["linking_number"] == 1

    def test_components(self):
        """Deux composantes 0₁."""
        props = hopf_link_properties(1.73e8)
        assert props["components"] == ["0_1", "0_1"]

    def test_mass_positive(self):
        """Masse du Hopf link > 0."""
        props = hopf_link_properties(1.73e8)
        assert props["mass_MeV"] > 0

    def test_mass_order_of_magnitude(self):
        """Masse ~ 0.03 MeV (ordre de grandeur)."""
        props = hopf_link_properties(1.73e8)
        assert 0.01 < props["mass_MeV"] < 0.1

    def test_stable(self):
        """Le Hopf link est topologiquement stable."""
        props = hopf_link_properties(1.73e8)
        assert props["stable"] is True

    def test_E_match_consistent(self):
        """E_match dans les propriétés est cohérent avec spectral_matching_energy."""
        R = 1.73e8
        props = hopf_link_properties(R)
        assert abs(props["E_match_MeV"] - spectral_matching_energy(R)) < 1e-12

    def test_sigma_consistent(self):
        """σ dans les propriétés est cohérent avec linking_cross_section."""
        R = 1.73e8
        props = hopf_link_properties(R)
        assert abs(props["sigma_fm2"] - linking_cross_section(R)) < 1e-6


class TestCriticalTemperature:
    """Tests pour la température critique et le redshift."""

    def test_T_crit_order_of_magnitude(self):
        """T_crit ≈ 4700 K pour R = 173 nm."""
        R_fm = 1.73e8
        T = critical_temperature(R_fm)
        assert abs(T - 4700) / 4700 < 0.05

    def test_redshift_order_of_magnitude(self):
        """z_crit ≈ 1720 pour T_crit ≈ 4700 K."""
        R_fm = 1.73e8
        T = critical_temperature(R_fm)
        z = critical_redshift(T)
        assert abs(z - 1720) / 1720 < 0.05

    def test_T_crit_decreases_with_R(self):
        """T_crit ∝ 1/R : plus grand rayon → T critique plus basse."""
        T1 = critical_temperature(1e7)
        T2 = critical_temperature(1e8)
        assert T1 > T2


class TestLinkingOnsetMass:
    """Tests pour le seuil de masse d'entrelac dans le CMB."""

    def test_onset_mass_order_of_magnitude(self):
        """m_onset ≈ 8–9 meV dans le CMB actuel."""
        m = linking_onset_mass(T_CMB)
        assert 5e-3 < m < 15e-3  # 5–15 meV

    def test_onset_mass_decreases_with_temperature(self):
        """Plus T est élevé → seuil plus haut (entrelac plus facile)."""
        m_cmb = linking_onset_mass(T_CMB)
        m_hot = linking_onset_mass(100.0)
        assert m_hot > m_cmb

    def test_rate_above_onset_exceeds_hubble(self):
        """Pour m < m_onset, Γ_ν > 1/t_H."""
        m_onset = linking_onset_mass(T_CMB)
        rate = linking_rate_per_neutrino(m_onset * 0.5, T_CMB)
        assert rate > 1.0 / T_HUBBLE_S

    def test_rate_below_onset_under_hubble(self):
        """Pour m > m_onset, Γ_ν < 1/t_H."""
        m_onset = linking_onset_mass(T_CMB)
        rate = linking_rate_per_neutrino(m_onset * 2.0, T_CMB)
        assert rate < 1.0 / T_HUBBLE_S

    def test_onset_near_solar_delta_m2(self):
        """Coïncidence : m_onset ≈ √(Δm²_sol) ≈ 8.7 meV."""
        m_onset = linking_onset_mass(T_CMB)
        dm2_sol = 7.53e-5  # eV²
        m2_min = dm2_sol ** 0.5  # ≈ 8.7 meV
        # Coïncidence à 20% près
        assert abs(m_onset - m2_min) / m2_min < 0.20

    def test_mfp_finite_below_onset(self):
        """Le libre parcours moyen est fini pour m < m_onset."""
        l = linking_mean_free_path_neutrino(1e-3, T_CMB)
        assert l < float('inf')
        assert l > 0

    def test_mfp_infinite_for_beta_neutrino(self):
        """Le libre parcours moyen est infini pour m = 0.57 eV (β decay)."""
        l = linking_mean_free_path_neutrino(0.57, T_CMB)
        assert l == float('inf')
