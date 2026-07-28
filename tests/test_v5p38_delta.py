"""Tests V5.38 — derivation rigoureuse delta = -4 ln(d_{1/2}(k=135))."""
from __future__ import annotations

import math

import pytest

from theory.su2_modular import V5_LEVEL, quantum_dim
from theory.v5p38_delta_derivation import (
    DELTA_CALIBRATED_V5P35D,
    DELTA_D1_AMPHI,
    apply_delta_exact,
    compare_calibrated_vs_exact,
    decompose_4lnd,
    delta_first_principles,
    enumerate_interpretations,
    omega_with_delta_exact,
    retained_interpretation,
    summary_report,
)


# ---------------------------------------------------------------------------
# 1. Valeur exacte first-principles : k=135 reproduit l'expectation
# ---------------------------------------------------------------------------
def test_delta_first_principles_value():
    """delta_exact(k=135) doit egaler -4 ln(2 cos(pi/137)) = -2.771537..."""
    val = delta_first_principles()
    expected = -4.0 * math.log(2.0 * math.cos(math.pi / 137.0))
    assert val == pytest.approx(expected, abs=1e-12)
    assert val == pytest.approx(-2.7715369380, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. delta_exact vs DELTA_D1_AMPHI (-ln 16, limite k -> oo)
# ---------------------------------------------------------------------------
def test_d1_amphi_is_minus_ln_16():
    """D1 = -ln(16) = -4 ln(2) = -2.7726..."""
    assert DELTA_D1_AMPHI == pytest.approx(-2.7725887222, abs=1e-9)
    assert DELTA_D1_AMPHI == pytest.approx(-4.0 * math.log(2.0), abs=1e-12)


def test_delta_exact_converges_to_d1_when_k_large():
    """A k tres grand, -4 ln(d_{1/2}(k)) -> -ln(16)."""
    for k in (1000, 10_000, 100_000):
        val = delta_first_principles(k=k)
        # converge avec correction 2(pi/(k+2))^2
        assert val == pytest.approx(
            DELTA_D1_AMPHI + 2.0 * (math.pi / (k + 2)) ** 2,
            abs=1e-10,
        )


# ---------------------------------------------------------------------------
# 3. Decomposition de Taylor : sommation reproduit exactement la valeur
# ---------------------------------------------------------------------------
def test_taylor_decomposition_sums_to_full():
    """D1 + 2nd + 4th + residual = full (par construction)."""
    dec = decompose_4lnd()
    s = (
        dec.leading_minus_4_ln2
        + dec.second_order
        + dec.fourth_order
        + dec.residual
    )
    assert s == pytest.approx(dec.full, abs=1e-15)


def test_taylor_signs_and_magnitudes():
    """Ordre 2 et ordre 4 sont positifs (-4 ln cos x = +2x^2 + (4/12)x^4 + ...).
    Le residu d'ordre 6 est minuscule a k=135."""
    dec = decompose_4lnd()
    assert dec.leading_minus_4_ln2 < 0
    assert dec.second_order > 0
    assert dec.fourth_order > 0
    assert abs(dec.residual) < 1e-9


# ---------------------------------------------------------------------------
# 4. Aucun parametre libre : k=135 est deja fixe par alpha_EM
# ---------------------------------------------------------------------------
def test_k_is_alpha_em_fixed():
    """k=135 est le niveau CS qui realise alpha_EM = 1/(k+2) = 1/137."""
    assert V5_LEVEL == 135
    assert 1.0 / (V5_LEVEL + 2) == pytest.approx(1.0 / 137.0, abs=0.0)


def test_d_half_value():
    """d_{1/2}(k=135) = 2 cos(pi/137) ≈ 1.99947."""
    val = quantum_dim(0.5, V5_LEVEL)
    assert val == pytest.approx(2.0 * math.cos(math.pi / 137.0), abs=1e-12)
    assert val == pytest.approx(1.9994741770, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. delta_exact vs delta_calibrated : ecart faible (< 0.2 %)
# ---------------------------------------------------------------------------
def test_delta_exact_close_to_calibrated():
    """delta_exact - delta_calib < 0.2 % (conjecture STRONG V5.36)."""
    delta_exact = delta_first_principles()
    rel = abs(delta_exact - DELTA_CALIBRATED_V5P35D) / abs(DELTA_CALIBRATED_V5P35D)
    assert rel < 2.0e-3  # 0.2 % de marge


# ---------------------------------------------------------------------------
# 6. Sigma|log| avec delta_exact reste tres petit (< 0.05 dex)
# ---------------------------------------------------------------------------
def test_sigma_log_with_delta_exact():
    """Sigma|log| (delta_exact) sur {K0,D0,B0,Bs0} doit etre < 0.05 dex.

    Calibre donne ~ 0.001 dex ; le penality d'utiliser la valeur exacte
    (sans calibration) doit rester sous 0.05 dex pour preserver la
    fermeture V5.35d.
    """
    cmp = compare_calibrated_vs_exact()
    assert cmp["sum_abs_log_exact"] < 0.05
    # Sigma|log| (exact) doit etre au plus 100 fois le calibrated (en
    # pratique ~5x dans le baseline actuel) ; on garde une marge large.
    assert cmp["sum_abs_log_exact"] / max(cmp["sum_abs_log_calib"], 1e-6) < 100


def test_per_meson_log_ratios_bounded():
    """Aucun meson individuel ne doit derive > 0.02 dex avec delta_exact."""
    cmp = compare_calibrated_vs_exact()
    for name, p in cmp["per_meson"].items():
        assert abs(p["log_r_exact"]) < 2.0e-2, (
            f"{name}: |log_r| = {abs(p['log_r_exact']):.4e} trop grand"
        )


# ---------------------------------------------------------------------------
# 7. apply_delta_exact preserve la signature V5.35d
# ---------------------------------------------------------------------------
def test_apply_delta_exact_returns_expected_keys():
    """apply_delta_exact retourne un dict avec les memes cles que V5.35d."""
    res = apply_delta_exact(["6_1", "6_1", "6_1"], 497.6)  # K^0
    assert "delta_m_MeV" in res
    assert "R_chi" in res
    assert "factor_3body" in res
    assert "delta" in res
    assert res["delta"] == pytest.approx(delta_first_principles(), abs=1e-12)


def test_omega_with_delta_exact_positive():
    """omega_chi(L) doit etre strictement positif pour K^0 (entrelacs reel)."""
    res = omega_with_delta_exact(["6_1", "6_1", "6_1"])
    assert res["omega_chi"] > 0
    assert res["delta"] == pytest.approx(delta_first_principles(), abs=1e-12)


# ---------------------------------------------------------------------------
# 8. Interpretations : exactement une retenue ('directions_4')
# ---------------------------------------------------------------------------
def test_exactly_one_interpretation_retained():
    """Une seule des 7 lectures doit etre retained=True."""
    n_ret = sum(1 for itp in enumerate_interpretations() if itp.is_retained)
    assert n_ret == 1
    rt = retained_interpretation()
    assert rt.label == "directions_4"


def test_interpretations_have_descriptions():
    """Toutes les interpretations sont documentees (description non vide)."""
    for itp in enumerate_interpretations():
        assert isinstance(itp.label, str) and len(itp.label) > 0
        assert isinstance(itp.description, str) and len(itp.description) > 20


# ---------------------------------------------------------------------------
# 9. summary_report contient les valeurs cles
# ---------------------------------------------------------------------------
def test_summary_report_has_key_strings():
    """Le rapport doit mentionner delta, k=135, et les 4 mesons."""
    rep = summary_report()
    assert "delta" in rep
    assert "k=135" in rep
    for name in ("K0", "D0", "B0", "Bs0"):
        assert name in rep


# ---------------------------------------------------------------------------
# 10. Lien D1 / D2 : D2 = D1 + 2 (pi/137)^2 + O(...) — coherence numerique
# ---------------------------------------------------------------------------
def test_d2_minus_d1_is_2pi_over_137_squared():
    """D2 - D1 = 2 (pi/137)^2 + O((pi/137)^4) au 4eme ordre pres."""
    d2 = delta_first_principles()
    diff = d2 - DELTA_D1_AMPHI
    expected_2nd = 2.0 * (math.pi / 137.0) ** 2
    # diff est legerement plus grand que 2 (pi/137)^2 a cause du 4eme ordre
    assert diff == pytest.approx(expected_2nd, abs=1e-6)
    # mais < 0.5 % de l'ordre 2
    assert abs(diff - expected_2nd) / expected_2nd < 5e-4
