"""V5.10b — Tests QNM Kerr tabulés (Berti-Cardoso-Starinets 2009).

Vérifie :
  1. Modes (2,2,0), (2,1,0), (3,3,0) disponibles.
  2. Schwarzschild (a*=0) : M ω_R(2,2,0) ≈ 0.3737, M ω_I ≈ 0.0890.
  3. Interpolation a* dans [0, 0.99].
  4. Erreurs hors grille.
  5. f ∝ 1/M et τ ∝ M (échelle gravitationnelle).
  6. Q sans dimension constant.
  7. GW150914 ringdown : f_(2,2,0) entre 230 et 290 Hz pour
     M_f=62 M_sun, a*=0,67 (cohérent LIGO ~250 Hz).
  8. τ_(2,2,0) entre 3 et 5 ms (cohérent LIGO ~4 ms).
  9. ω_R croît monotone avec a* (frame dragging).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.ringdown_QNM_tabulated import (
    GM_SUN_OVER_C3,
    available_modes,
    damping_time_ms,
    kerr_QNM_mode,
    quality_factor,
    ringdown_frequency_Hz,
)


def test_available_modes():
    modes = available_modes()
    assert (2, 2, 0) in modes
    assert (2, 1, 0) in modes
    assert (3, 3, 0) in modes


def test_schwarzschild_220_value():
    """Schwarzschild (a*=0), mode (2,2,0) : valeurs BCS09 standard.

    BCS09 donne M ω = 0.37367 - 0.08896 i.
    """
    omR, omI = kerr_QNM_mode(2, 2, 0, 0.0)
    assert math.isclose(omR, 0.37367, abs_tol=1e-4)
    assert math.isclose(omI, 0.08896, abs_tol=1e-4)


def test_schwarzschild_330_value():
    """Schwarzschild (2,2,0) et (3,3,0) ont des fréquences distinctes."""
    om22, _ = kerr_QNM_mode(2, 2, 0, 0.0)
    om33, _ = kerr_QNM_mode(3, 3, 0, 0.0)
    assert om33 > om22  # mode l=3 plus rapide


def test_omega_R_increases_with_spin():
    """ω_R(2,2,0) augmente strictement avec a* (frame dragging)."""
    spins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    omRs = [kerr_QNM_mode(2, 2, 0, a)[0] for a in spins]
    for i in range(len(omRs) - 1):
        assert omRs[i + 1] > omRs[i], (
            f"non-monotone at a*={spins[i + 1]}"
        )


def test_omega_I_decreases_with_spin_220():
    """ω_I(2,2,0) décroît avec a* (modes plus longs vie pour Kerr quasi-extrême)."""
    om0 = kerr_QNM_mode(2, 2, 0, 0.0)[1]
    om99 = kerr_QNM_mode(2, 2, 0, 0.99)[1]
    assert om99 < om0


def test_interpolation_at_grid_point():
    """À un point de la grille, interpolation linéaire = valeur tabulée."""
    omR, _ = kerr_QNM_mode(2, 2, 0, 0.50)
    # BCS09 : a*=0.50, M ω_R(2,2,0) = 0.46175.
    assert math.isclose(omR, 0.46175, abs_tol=1e-4)


def test_interpolation_between_grid_points():
    """L'interpolation à mi-chemin donne la moyenne pour grille linéaire."""
    omR_lo, _ = kerr_QNM_mode(2, 2, 0, 0.20)
    omR_hi, _ = kerr_QNM_mode(2, 2, 0, 0.30)
    omR_mid, _ = kerr_QNM_mode(2, 2, 0, 0.25)
    assert math.isclose(omR_mid, 0.5 * (omR_lo + omR_hi), abs_tol=1e-9)


def test_unknown_mode_raises():
    with pytest.raises(KeyError):
        kerr_QNM_mode(4, 4, 0, 0.5)


def test_spin_out_of_range_raises():
    with pytest.raises(ValueError):
        kerr_QNM_mode(2, 2, 0, -0.1)
    with pytest.raises(ValueError):
        kerr_QNM_mode(2, 2, 0, 1.5)


def test_invalid_mass_raises():
    with pytest.raises(ValueError):
        ringdown_frequency_Hz(-1.0, 2, 2, 0, 0.5)
    with pytest.raises(ValueError):
        damping_time_ms(0.0, 2, 2, 0, 0.5)


def test_frequency_scales_inverse_mass():
    """f ∝ 1/M_BH (échelle gravitationnelle)."""
    f1 = ringdown_frequency_Hz(10.0, 2, 2, 0, 0.5)
    f2 = ringdown_frequency_Hz(20.0, 2, 2, 0, 0.5)
    assert math.isclose(f1 / f2, 2.0, rel_tol=1e-9)


def test_damping_time_scales_linear_mass():
    """τ ∝ M_BH."""
    tau1 = damping_time_ms(10.0, 2, 2, 0, 0.5)
    tau2 = damping_time_ms(20.0, 2, 2, 0, 0.5)
    assert math.isclose(tau2 / tau1, 2.0, rel_tol=1e-9)


def test_quality_factor_dimensionless():
    """Q ne dépend pas de M_BH (sans dimension)."""
    q1 = quality_factor(2, 2, 0, 0.6)
    q2 = quality_factor(2, 2, 0, 0.6)
    assert q1 == q2
    # Vérif valeur : Q(2,2,0, a*=0) = 0.37367/(2*0.08896) ≈ 2.10
    q0 = quality_factor(2, 2, 0, 0.0)
    assert math.isclose(q0, 0.37367 / (2.0 * 0.08896), rel_tol=1e-4)


def test_GW150914_frequency_consistent_with_LIGO():
    """GW150914 ringdown : f_(2,2,0) ≈ 250 Hz (Abbott+ 2016).

    Pour M_f=62 M_sun, a*=0.67, BCS09 donne ~267 Hz.
    LIGO rapporte ~250 Hz comme ordre de grandeur. Tolérance 15 %.
    """
    f = ringdown_frequency_Hz(62.0, 2, 2, 0, 0.67)
    assert 230.0 < f < 290.0


def test_GW150914_damping_time_consistent_with_LIGO():
    """GW150914 ringdown : τ_(2,2,0) ≈ 4 ms (Abbott+ 2016).

    Pour M_f=62 M_sun, a*=0.67, BCS09 donne ~3.6 ms. Tolérance 1 ms.
    """
    tau = damping_time_ms(62.0, 2, 2, 0, 0.67)
    assert 3.0 < tau < 5.0


def test_gm_sun_over_c3_value():
    """GM_sun/c³ ≈ 4,9255 µs (constante astrophysique standard)."""
    assert math.isclose(GM_SUN_OVER_C3 * 1e6, 4.9255, abs_tol=0.01)
