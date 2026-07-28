"""V5.10b — Tests PN-1 précession périhélie (formule fermée GR).

Vérifie :
  1. Formule fermée Δφ = 6πGM/(c²a(1-e²)) par révolution.
  2. Validation Mercure 42,98″/siècle à mieux que 1 %.
  3. Validation des 6 planètes du catalogue à mieux que 1 %.
  4. Scaling 1/a et 1/(1-e²) cohérent.
  5. Erreurs sur entrées invalides.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.post_newtonian_PN1 import (
    ARCSEC_PER_RAD,
    C_LIGHT,
    CENTURY,
    G_NEWTON,
    M_SUN,
    PLANETS,
    perihelion_precession_arcsec_per_century,
    perihelion_precession_per_revolution,
    validate_all_planets,
    validate_mercury,
)


def test_formula_per_revolution_value():
    """Δφ_rev pour Mercure ≈ 5.02e-7 rad (cf. MTW)."""
    p = PLANETS["Mercury"]
    delta = perihelion_precession_per_revolution(M_SUN, p.a, p.e)
    expected = 6.0 * math.pi * G_NEWTON * M_SUN / (C_LIGHT ** 2 * p.a * (1.0 - p.e ** 2))
    assert math.isclose(delta, expected, rel_tol=1e-12)


def test_mercury_reproduces_42_98_arcsec():
    """Mercure : 42,98″/siècle reproduit à mieux que 0,1 %."""
    r = validate_mercury()
    assert r["relative_error"] < 1e-3
    assert math.isclose(r["predicted_arcsec_per_century"], 42.98, abs_tol=0.05)


def test_all_planets_within_1_percent():
    """Les 6 planètes reproduites à mieux que 1 % de Will (2014)."""
    results = validate_all_planets()
    assert len(results) == 6
    for r in results:
        # Saturne a 0,59 %, les autres < 0,1 %. Tolérance 1 %.
        assert r["relative_error"] < 0.01, (
            f"{r['name']}: err = {r['relative_error']:.4f}"
        )


def test_scaling_inverse_semi_major_axis():
    """Δφ ∝ 1/a pour e fixé."""
    a1 = 1.0e10
    a2 = 2.0e10
    d1 = perihelion_precession_per_revolution(M_SUN, a1, 0.0)
    d2 = perihelion_precession_per_revolution(M_SUN, a2, 0.0)
    assert math.isclose(d1 / d2, 2.0, rel_tol=1e-12)


def test_scaling_eccentricity():
    """Δφ ∝ 1/(1 - e²) pour a fixé."""
    a = 1.0e11
    d0 = perihelion_precession_per_revolution(M_SUN, a, 0.0)
    d_05 = perihelion_precession_per_revolution(M_SUN, a, 0.5)
    ratio = d_05 / d0
    expected = 1.0 / (1.0 - 0.25)
    assert math.isclose(ratio, expected, rel_tol=1e-12)


def test_arcsec_century_conversion():
    """Conversion radians/rév → arcsec/siècle cohérente."""
    p = PLANETS["Mercury"]
    delta_rad = perihelion_precession_per_revolution(M_SUN, p.a, p.e)
    revs = CENTURY / p.T
    expected = delta_rad * revs * ARCSEC_PER_RAD
    actual = perihelion_precession_arcsec_per_century(M_SUN, p.a, p.e, p.T)
    assert math.isclose(actual, expected, rel_tol=1e-12)


def test_invalid_a_raises():
    with pytest.raises(ValueError):
        perihelion_precession_per_revolution(M_SUN, -1.0, 0.1)
    with pytest.raises(ValueError):
        perihelion_precession_per_revolution(M_SUN, 0.0, 0.1)


def test_invalid_e_raises():
    with pytest.raises(ValueError):
        perihelion_precession_per_revolution(M_SUN, 1e10, -0.1)
    with pytest.raises(ValueError):
        perihelion_precession_per_revolution(M_SUN, 1e10, 1.0)


def test_invalid_T_raises():
    with pytest.raises(ValueError):
        perihelion_precession_arcsec_per_century(M_SUN, 1e10, 0.1, -1.0)


def test_jupiter_subarcsec():
    """Jupiter doit donner ~0,062″/siècle (loin du Soleil)."""
    p = PLANETS["Jupiter"]
    val = perihelion_precession_arcsec_per_century(M_SUN, p.a, p.e, p.T)
    assert math.isclose(val, 0.0623, abs_tol=0.005)


def test_solar_mass_scaling():
    """Δφ ∝ M (masse centrale)."""
    a = 1.0e11
    e = 0.1
    d1 = perihelion_precession_per_revolution(M_SUN, a, e)
    d2 = perihelion_precession_per_revolution(2.0 * M_SUN, a, e)
    assert math.isclose(d2 / d1, 2.0, rel_tol=1e-12)
