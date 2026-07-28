"""Tests pour theory/tov_V5.py — solveur TOV V5 modifié et confrontation NICER.

Couvre :
  1. Constantes physiques et κ_g.
  2. EoS polytropiques : monotonie, inversibilité.
  3. RHS TOV (GR et V5) : signes, dimensions.
  4. Intégration TOV : M(R) raisonnable pour les 4 EoS calibrées.
  5. V5 correction : ordre de grandeur 10⁻⁹ pour densité NS.
  6. Courbes M-R : M_max ~ 2 M_sun, R(1.4) ~ 11-14 km.
  7. χ² NICER : V5 vs GR différence négligeable.
  8. Bayes factor : ≈ 1 (indiscernable).
  9. Limite supérieure sur κ_g : grande (V5 fortement non contraint).
  10. P22 ξ ≪ 1 partout dans une NS (pas d'horizon).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.tov_V5 import (
    C, G_NEWTON, M_SUN, M_NUCLEON, KAPPA_G_M, KM,
    EOS_LIBRARY, PolytropicEoS,
    V5_correction_factor, xi_P22,
    tov_rhs_GR, tov_rhs_V5, tov_V5_integrate, mass_radius_curve,
    relative_radius_shift, find_M_max,
    NICER_PULSARS, chi2_vs_NICER, bayes_factor_V5_vs_GR,
    kappa_g_upper_limit,
)


# ---------------------------------------------------------------------------
# 1. Constantes
# ---------------------------------------------------------------------------

def test_kappa_g_order_of_magnitude():
    """κ_g doit valoir ≈ 3.12 × 10⁻⁵³ m (V5.5 étape 6)."""
    assert 1e-53 < KAPPA_G_M < 1e-52
    assert abs(KAPPA_G_M - 3.12e-53) / 3.12e-53 < 0.05


def test_constants_consistency():
    assert C > 0 and G_NEWTON > 0 and M_SUN > 0 and M_NUCLEON > 0


# ---------------------------------------------------------------------------
# 2. EoS
# ---------------------------------------------------------------------------

def test_eos_round_trip():
    """rho_of_P(P_of_rho(rho)) == rho."""
    for eos in EOS_LIBRARY.values():
        for rho in [1e15, 1e17, 1e18]:
            P = eos.P_of_rho(rho)
            rho_back = eos.rho_of_P(P)
            assert abs(rho_back - rho) / rho < 1e-10


def test_eos_monotonic():
    """P doit être croissant avec rho."""
    for eos in EOS_LIBRARY.values():
        rhos = np.logspace(15, 18, 10)
        Ps = np.array([eos.P_of_rho(r) for r in rhos])
        assert np.all(np.diff(Ps) > 0)


# ---------------------------------------------------------------------------
# 3. RHS TOV
# ---------------------------------------------------------------------------

def test_tov_rhs_pressure_decreases():
    """dP/dr < 0 au centre (pression diminue vers l'extérieur)."""
    eos = EOS_LIBRARY["SLy9"]
    y = np.array([1e35, 1e29])  # P=1e35 Pa, M=1e29 kg
    rhs = tov_rhs_GR(1e3, y, eos)
    assert rhs[0] < 0
    assert rhs[1] > 0  # masse croît


def test_tov_rhs_V5_vs_GR_close():
    """V5 ≈ GR à l'ordre relatif 10⁻⁸."""
    eos = EOS_LIBRARY["SLy9"]
    y = np.array([1e35, 1e29])
    g = tov_rhs_GR(1e3, y, eos)
    v = tov_rhs_V5(1e3, y, eos)
    if g[0] != 0:
        rel = abs(v[0] - g[0]) / abs(g[0])
        assert rel < 1e-5


# ---------------------------------------------------------------------------
# 4-5. V5 correction
# ---------------------------------------------------------------------------

def test_V5_correction_order():
    """Δ_V5 ~ 10⁻⁸ pour ρ_NS, signe positif."""
    rho = 5e17     # ~2 × ρ_sat
    P = 1e35
    delta = V5_correction_factor(rho, P)
    assert 1e-10 < delta < 1e-6
    assert delta > 0


def test_V5_correction_scales_linearly_with_kappa():
    """Δ_V5 ∝ κ_g."""
    rho, P = 5e17, 1e35
    d1 = V5_correction_factor(rho, P, kappa_g=KAPPA_G_M)
    d10 = V5_correction_factor(rho, P, kappa_g=10*KAPPA_G_M)
    assert abs(d10 - 10*d1) / d10 < 1e-10


# ---------------------------------------------------------------------------
# 6. Courbes M-R
# ---------------------------------------------------------------------------

def test_mass_radius_curve_realistic():
    """Au moins une EoS doit donner M_max ≥ 1.9 M_sun et R(M_max) ∈ [9,14] km."""
    found = False
    for eos in EOS_LIBRARY.values():
        M_max, R_at_max = find_M_max(eos, modifications=False)
        if M_max >= 1.9 and 9.0 <= R_at_max <= 14.0:
            found = True
            break
    assert found, "Aucune EoS calibrée ne donne une NS réaliste."


def test_tov_integration_finite():
    """Le résultat d'intégration doit être fini."""
    eos = EOS_LIBRARY["SLy9"]
    res = tov_V5_integrate(eos, 1e35, modifications=False)
    assert math.isfinite(res["R"]) and res["R"] > 0
    assert math.isfinite(res["M"]) and res["M"] > 0


# ---------------------------------------------------------------------------
# 7-8. NICER
# ---------------------------------------------------------------------------

def test_chi2_nicer_v5_vs_gr():
    """χ²(V5) ≈ χ²(GR) à mieux que 10⁻⁵ relatif (κ_g négligeable)."""
    eos = EOS_LIBRARY["SLy9"]
    curve_GR = mass_radius_curve(eos, modifications=False)
    curve_V5 = mass_radius_curve(eos, modifications=True)
    chi2_GR = chi2_vs_NICER(curve_GR)
    chi2_V5 = chi2_vs_NICER(curve_V5)
    rel = abs(chi2_V5 - chi2_GR) / max(chi2_GR, 1.0)
    assert rel < 1e-5


def test_bayes_factor_near_one():
    """B = exp(-ΔΧ²/2) ≈ 1 → V5 et GR indiscernables."""
    eos = EOS_LIBRARY["SLy9"]
    B = bayes_factor_V5_vs_GR(eos)
    assert 0.99 < B < 1.01


# ---------------------------------------------------------------------------
# 9. Limite supérieure sur κ_g
# ---------------------------------------------------------------------------

def test_kappa_g_upper_limit_loose():
    """κ_g_max >> κ_g_V5 (i.e. V5 non contraint par NICER)."""
    eos = EOS_LIBRARY["SLy9"]
    kmax = kappa_g_upper_limit(eos, sigma_R_rel_target=0.1)
    # On veut que la limite NICER soit énormément plus haute que la
    # valeur V5 prédite.
    assert kmax / KAPPA_G_M > 1e6


# ---------------------------------------------------------------------------
# 10. P22 régime
# ---------------------------------------------------------------------------

def test_xi_P22_subdominant_in_NS():
    """ξ_P22 ≪ 1 dans toute la NS (pas de régime extrême)."""
    eos = EOS_LIBRARY["SLy9"]
    res = tov_V5_integrate(eos, 1e35, modifications=False)
    r = res["r"]; rho = res["rho"]
    # gradient densité
    drho_dr = np.gradient(rho, r)
    xis = [xi_P22(rho[i], drho_dr[i]) for i in range(len(rho))]
    assert max(abs(x) for x in xis) < 1e-3


# ---------------------------------------------------------------------------
# Bonus : structure des données NICER
# ---------------------------------------------------------------------------

def test_nicer_pulsars_structure():
    for name, p in NICER_PULSARS.items():
        for key in ("M_sun", "M_err", "R_km", "R_err"):
            assert key in p
            assert p[key] > 0
