"""
Tests pytest pour theory/framing_anomaly_V5.py (V5.6 etape 9).

Verifie :
  - le scaling epsilon = -(n+3)/4 = -3/8 pour n_eff = -1.5
  - la calibration n_0 ~ 1.6 (cinetique condensation T_c)
  - delta_theorique ~ -0.6 (proche du best fit empirique a68d8a3)
  - kappa_CJ(M_h) module a le bon comportement aux limites
  - contribution de framing sur la pente est negative et |.| ~ 0.4
  - l'API plus_1overc combine correctement les deux mecanismes
  - delta scale lineairement avec n_0
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.framing_anomaly_V5 import (
    K_LEVEL,
    N_EFF_GALACTIC,
    M_REF_MSUN,
    epsilon_scaling,
    n_0_from_condensation,
    n_twist,
    delta_framing,
    kappa_CJ_with_framing,
    kappa_CJ_framing_plus_1overc,
    framing_slope_contribution,
    summary,
)


def test_k_level_V5():
    """V5 utilise k = 135 (k+2 = 137 = 1/alpha_EM)."""
    assert K_LEVEL == 135


def test_epsilon_scaling_value():
    """Pour n_eff = -1.5 (Planck galactic), epsilon = -3/8 = -0.375."""
    eps = epsilon_scaling(N_EFF_GALACTIC)
    assert eps == pytest.approx(-0.375, abs=1e-9)


def test_epsilon_scaling_linearity_in_n():
    """epsilon est lineaire en n_eff."""
    e1 = epsilon_scaling(-2.0)
    e2 = epsilon_scaling(-1.0)
    # Difference doit etre = -(1.0 - 2.0)/4 = +0.25 / ... attention au signe
    diff = e1 - e2
    # epsilon(n) = -(n+3)/4 -> e1 = -(-2+3)/4 = -0.25; e2 = -(-1+3)/4 = -0.5
    # diff = -0.25 - (-0.5) = 0.25
    assert diff == pytest.approx(0.25, abs=1e-9)


def test_n_0_calibration_order_of_magnitude():
    """n_0 calibre via T_c = 67 MeV doit etre O(1) - O(10)."""
    n_0 = n_0_from_condensation()
    assert 0.5 < n_0 < 5.0
    # Plus precisement : on attend ~1.6 par construction
    assert 1.4 < n_0 < 1.8


def test_delta_theorique_negative():
    """delta_theorique doit etre negatif (kappa_CJ decroit avec M_h)."""
    delta = delta_framing()
    assert delta < 0


def test_delta_theorique_close_to_empirical():
    """delta_theorique doit etre proche du best fit empirique -0.6."""
    delta = delta_framing()
    delta_empirical = -0.60
    # On exige une coincidence a 10 % pres
    assert abs(delta - delta_empirical) < 0.10
    # Et a 5 % pres c'est ideal
    assert abs(delta - delta_empirical) / abs(delta_empirical) < 0.05


def test_delta_scales_linearly_with_n0():
    """delta = n_0 * epsilon : si on double n_0, delta double."""
    d1 = delta_framing(n_0=1.0)
    d2 = delta_framing(n_0=2.0)
    assert d2 == pytest.approx(2 * d1, rel=1e-9)


def test_n_twist_at_pivot():
    """n_twist(M_ref) = n_0 par construction."""
    n_0 = n_0_from_condensation()
    val = n_twist(M_REF_MSUN)
    assert float(val) == pytest.approx(n_0, rel=1e-9)


def test_n_twist_decreases_with_M_h():
    """n_twist decroit avec M_h (epsilon < 0)."""
    M_low = 1e10
    M_high = 1e13
    val_low = float(n_twist(M_low))
    val_high = float(n_twist(M_high))
    assert val_low > val_high


def test_kappa_CJ_at_pivot():
    """A M_h = M_ref, f_eff = f_0, kappa = kappa_0 * |cos(pi h_j f_0)|."""
    from theory.su2_modular import conformal_weight
    kappa = float(kappa_CJ_with_framing(M_REF_MSUN, j=5.0, f_0=1.0,
                                        kappa_0=62.7))
    h_j = conformal_weight(5.0, 135)
    expected = 62.7 * abs(np.cos(np.pi * h_j * 1.0))
    assert kappa == pytest.approx(expected, rel=1e-9)


def test_kappa_CJ_bounded():
    """kappa_CJ(M_h) doit rester dans [0, kappa_0]."""
    M_grid = np.logspace(8, 14, 50)
    k = kappa_CJ_with_framing(M_grid, j=5.0, f_0=1.0, kappa_0=62.7)
    assert np.all(k >= 0)
    assert np.all(k <= 62.7)


def test_framing_slope_contribution_negative():
    """La contribution de framing sur d log c / d log M_h doit etre negative."""
    s = framing_slope_contribution()
    assert s < 0
    # En magnitude : 0.3 - 0.5 (proche de -0.41 empirique a68d8a3)
    assert 0.3 < abs(s) < 0.5


def test_kappa_combined_AC_at_pivot():
    """A M_h = M_ref, le facteur (C) = 1 (log = 0)."""
    from theory.su2_modular import conformal_weight
    kappa = float(kappa_CJ_framing_plus_1overc(M_REF_MSUN, j=5.0, f_0=1.0,
                                               alpha_c=10.0, kappa_0=62.7))
    h_j = conformal_weight(5.0, 135)
    expected = 62.7 * abs(np.cos(np.pi * h_j * 1.0))  # facteur C = 1+0 = 1
    assert kappa == pytest.approx(expected, rel=1e-9)


def test_summary_returns_dict():
    """summary() renvoie un dict avec les cles attendues."""
    res = summary(verbose=False)
    expected_keys = {"k_level", "n_eff_galactic", "epsilon_scaling",
                     "n_0_calibre", "delta_theorique",
                     "delta_empirique_a68d8a3", "h_j_fund",
                     "h_j_best_j5", "slope_contribution"}
    assert expected_keys.issubset(set(res.keys()))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
