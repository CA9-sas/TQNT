"""Tests V5.31 — Spectre vibratoire chiral asymétrique."""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import is_amphicheiral, jones_polynomial, jones_power_center
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import (
    ALPHA_VIB_DEFAULT,
    delta_m_v5p30c,
)
from theory.v5p31_chiral_vibrational import (
    AMPHI_PENALTY,
    BETA_CHIRAL_DEFAULT,
    calibrate_beta_ratio,
    chiral_factor,
    chirality_signature,
    delta_m_v5p31,
    p_chiral,
    v5p31_table,
)


# ---------------------------------------------------------------------------
# 1. Signatures de chiralité
# ---------------------------------------------------------------------------
def test_chirality_signature_amphi_is_negative_unit():
    """Les amphichiraux 4_1, 6_3, 8_3 doivent avoir σ = -AMPHI_PENALTY."""
    for k in ["4_1", "6_3", "8_3"]:
        sig = chirality_signature(k)
        assert sig["is_amphi"] is True
        assert sig["sigma"] == pytest.approx(-AMPHI_PENALTY)


def test_chirality_signature_chiral_is_abs_power_center():
    """Les chiraux : σ = |jones_power_center(V_K)|, σ > 0."""
    for k in ["3_1", "5_1", "6_1", "6_2", "7_2", "7_6", "7_7"]:
        sig = chirality_signature(k)
        assert sig["is_amphi"] is False
        expected = abs(jones_power_center(jones_polynomial(k)))
        assert sig["sigma"] == pytest.approx(expected)
        assert sig["sigma"] > 0  # tous chiraux ont power_center != 0


def test_chirality_signature_meson_components_all_valid():
    """Toutes les composantes des 4 mésons doivent retourner σ valide."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        for k in cfg["knots"]:
            sig = chirality_signature(k)
            assert "sigma" in sig
            assert np.isfinite(sig["sigma"])


# ---------------------------------------------------------------------------
# 2. Facteur chiral par composante
# ---------------------------------------------------------------------------
def test_chiral_factor_amphi_suppresses():
    """β > 0 : amphi → exp(-β) < 1 (suppression)."""
    beta = 0.5
    f = chiral_factor("6_3", beta)
    assert f["is_amphi"] is True
    assert f["factor"] == pytest.approx(np.exp(-beta))
    assert f["factor"] < 1.0


def test_chiral_factor_chiral_amplifies():
    """β > 0 : chiral → exp(+β|σ|) > 1 (amplification)."""
    beta = 0.5
    f = chiral_factor("3_1", beta)
    assert f["is_amphi"] is False
    pc = jones_power_center(jones_polynomial("3_1"))
    assert f["factor"] == pytest.approx(np.exp(beta * abs(pc)))
    assert f["factor"] > 1.0


def test_chiral_factor_beta_zero_is_identity():
    """β = 0 ⇒ facteur = 1 pour toutes les composantes."""
    for k in ["3_1", "6_1", "6_3", "7_6", "8_3"]:
        f = chiral_factor(k, 0.0)
        assert f["factor"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Préfacteur P_chiral global
# ---------------------------------------------------------------------------
def test_p_chiral_K0_no_amphi():
    """K⁰ = 3×6_1 : n_amphi = 0 (6_1 est chiral)."""
    pc = p_chiral(["6_1", "6_1", "6_1"], 0.5)
    assert pc["n_amphi"] == 0
    assert pc["n_chiral"] == 3


def test_p_chiral_Bs0_one_amphi():
    """B_s⁰ = 6_3 + 7_6 : 1 amphi (6_3) + 1 chiral (7_6)."""
    pc = p_chiral(["6_3", "7_6"], 0.5)
    assert pc["n_amphi"] == 1
    assert pc["n_chiral"] == 1


def test_p_chiral_B0_no_amphi():
    """B⁰ = 3_1 + 6_2 + 7_7 : zero amphi (asymétrie clé du diagnostic)."""
    pc = p_chiral(["3_1", "6_2", "7_7"], 0.5)
    assert pc["n_amphi"] == 0
    assert pc["n_chiral"] == 3


def test_p_chiral_product_form():
    """P_chiral = Π exp(β σ_i) = exp(β · Σ σ_i)."""
    knots = ["3_1", "6_3", "7_6"]
    beta = 0.4
    pc = p_chiral(knots, beta)
    expected = np.exp(beta * pc["sigma_sum"])
    assert pc["p_chiral"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Δm V5.31
# ---------------------------------------------------------------------------
def test_delta_m_v5p31_K0_calibrated_to_PDG():
    """V5.31(K⁰) = PDG(K⁰) par construction (R_chiral(K⁰)=1)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p31(
        cfg["knots"], cfg["mass_pdg"],
        BETA_CHIRAL_DEFAULT, ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
    )
    assert res["delta_m_MeV"] == pytest.approx(DELTA_M_PDG["K0"], rel=1e-9)
    assert res["R_chiral"] == pytest.approx(1.0, rel=1e-9)


def test_delta_m_v5p31_beta_zero_equals_v5p30c():
    """β = 0 ⇒ R_chiral = 1 partout ⇒ V5.31 ≡ V5.30c."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        res_v531 = delta_m_v5p31(
            cfg["knots"], cfg["mass_pdg"],
            0.0, ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
        )
        res_v530c = delta_m_v5p30c(
            cfg["knots"], cfg["mass_pdg"],
            ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
        )
        assert res_v531["delta_m_MeV"] == pytest.approx(
            res_v530c["delta_m_MeV"], rel=1e-9
        )


def test_delta_m_v5p31_positive_and_finite():
    """Δm V5.31 > 0 et finis sur tous les mésons à β par défaut."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        res = delta_m_v5p31(
            cfg["knots"], cfg["mass_pdg"],
            BETA_CHIRAL_DEFAULT, ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
        )
        assert res["delta_m_MeV"] > 0
        assert np.isfinite(res["delta_m_MeV"])


# ---------------------------------------------------------------------------
# 5. Calibration sur ratio Bs/B
# ---------------------------------------------------------------------------
def test_calibrate_beta_ratio_closes_target():
    """β* calibré ferme effectivement le ratio B_s/B à |log| < 0.05 dex."""
    cal = calibrate_beta_ratio(
        target_ratio=35.0, beta_range=(-3.0, 3.0), n_grid=6001
    )
    assert "beta" in cal
    assert abs(cal["log_diff_dex"]) < 0.05
    assert 30.0 <= cal["ratio_v531"] <= 40.0


def test_calibrate_beta_ratio_K0_unchanged():
    """Après calibration de β, K⁰ doit rester à log_ratio = 0 (R_chiral(K⁰)=1)."""
    cal = calibrate_beta_ratio(
        target_ratio=35.0, beta_range=(-3.0, 3.0), n_grid=1001
    )
    assert abs(cal["per_meson"]["K0"]["log_ratio"]) < 1e-6


# ---------------------------------------------------------------------------
# 6. Table V5.31
# ---------------------------------------------------------------------------
def test_v5p31_table_has_all_4_mesons():
    rows = v5p31_table(beta=BETA_CHIRAL_DEFAULT)
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}


def test_v5p31_table_includes_required_fields():
    rows = v5p31_table(beta=BETA_CHIRAL_DEFAULT)
    required = {
        "name", "knots", "delta_m_v5p31_MeV", "delta_m_v5p30c_MeV",
        "delta_m_pdg_MeV", "log_ratio_v5p31", "log_ratio_v5p30c",
        "R_chiral", "P_chiral", "sigma_sum", "n_amphi", "n_chiral",
        "beta", "alpha", "kappa_K",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"Champs manquants dans {r['name']}: {missing}"


def test_v5p31_ratio_closes_at_default_beta():
    """À β = BETA_CHIRAL_DEFAULT, le ratio Bs/B doit être ≈ 35."""
    rows = v5p31_table(beta=BETA_CHIRAL_DEFAULT)
    by_name = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows}
    ratio = by_name["Bs0"] / by_name["B0"]
    assert 30.0 <= ratio <= 40.0, f"Ratio Bs/B = {ratio:.2f} hors [30,40]"
