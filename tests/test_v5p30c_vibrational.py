"""Tests V5.30c — Modes vibratoires comme préfacteur dynamique."""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import braid_index
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import (
    KAPPA_K_V5p30_DEFAULT,
    delta_m_v5p30,
)
from theory.v5p30c_vibrational import (
    ALPHA_VIB_DEFAULT,
    calibrate_alpha_global,
    coupling_factor,
    delta_m_v5p30c,
    p_vibrational,
    v5p30c_table,
    vibrational_frequency,
)


# ---------------------------------------------------------------------------
# 1. Fréquences vibratoires
# ---------------------------------------------------------------------------
def test_vibrational_frequency_signature():
    res = vibrational_frequency("3_1")
    assert set(res.keys()) >= {"knot", "b", "c", "omega0"}
    assert res["knot"] == "3_1"
    assert res["b"] == braid_index("3_1")
    assert res["c"] == crossing_number("3_1")
    assert res["omega0"] == pytest.approx(
        braid_index("3_1") / np.sqrt(crossing_number("3_1"))
    )


def test_vibrational_frequency_positive_for_all_assignments():
    """Toutes les composantes des 4 mésons doivent avoir ω_0 > 0."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        for k in cfg["knots"]:
            v = vibrational_frequency(k)
            assert v["omega0"] > 0
            assert np.isfinite(v["omega0"])


# ---------------------------------------------------------------------------
# 2. Couplage inter-composantes
# ---------------------------------------------------------------------------
def test_coupling_factor_K0():
    """K⁰ = 3 composantes 6_1 identiques."""
    cpl = coupling_factor(["6_1", "6_1", "6_1"])
    assert cpl["n_comp"] == 3
    # 3 composantes identiques ⇒ ⟨ω_0⟩ = ω_0(6_1)
    assert cpl["omega0_geom"] == pytest.approx(
        braid_index("6_1") / np.sqrt(crossing_number("6_1"))
    )
    assert cpl["omega_link"] == pytest.approx(3 * cpl["omega0_geom"])


def test_coupling_factor_Bs0_has_2_components():
    """B_s⁰ = 6_3 + 7_6 ⇒ n_comp = 2 (clé du diagnostic)."""
    cpl = coupling_factor(["6_3", "7_6"])
    assert cpl["n_comp"] == 2
    assert cpl["omega_link"] > 0


def test_coupling_factor_B0_has_3_components():
    """B⁰ a 3 composantes contre 2 pour B_s⁰."""
    cpl_B = coupling_factor(["3_1", "6_2", "7_7"])
    cpl_Bs = coupling_factor(["6_3", "7_6"])
    assert cpl_B["n_comp"] == 3
    assert cpl_Bs["n_comp"] == 2
    # facteur n_comp dans Ω_link explicite dans la formule
    ratio_n = cpl_B["n_comp"] / cpl_Bs["n_comp"]
    assert ratio_n == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 3. Préfacteur P_vib
# ---------------------------------------------------------------------------
def test_p_vibrational_alpha_zero_is_identity():
    """α = 0 ⇒ P_vib = 1 partout."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        pv = p_vibrational(cfg["knots"], 0.0)
        assert pv["p_vib"] == pytest.approx(1.0)


def test_p_vibrational_monotonic_in_alpha():
    """Pour Ω_link > 1, P_vib croît avec α ; pour Ω_link < 1, décroît."""
    knots_K = MESON_OSCILLATION_ASSIGNMENTS["K0"]["knots"]
    pv_neg = p_vibrational(knots_K, -1.0)["p_vib"]
    pv_zero = p_vibrational(knots_K, 0.0)["p_vib"]
    pv_pos = p_vibrational(knots_K, 1.0)["p_vib"]
    # Ω_link(K⁰) = 3 · 3 / √6 ≈ 3.67 > 1, donc P_vib croît avec α
    omega_K0 = coupling_factor(knots_K)["omega_link"]
    assert omega_K0 > 1
    assert pv_pos > pv_zero > pv_neg


def test_p_vibrational_alpha_one_matches_omega():
    """α = 1 ⇒ P_vib = Ω_link."""
    knots = ["3_1", "5_1", "7_2"]
    pv = p_vibrational(knots, 1.0)
    cpl = coupling_factor(knots)
    assert pv["p_vib"] == pytest.approx(cpl["omega_link"])


# ---------------------------------------------------------------------------
# 4. Δm V5.30c
# ---------------------------------------------------------------------------
def test_delta_m_v5p30c_K0_calibrated_to_PDG():
    """V5.30c(K⁰) = PDG(K⁰) par construction (R_vib(K⁰)=1)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p30c(
        cfg["knots"], cfg["mass_pdg"], ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT
    )
    assert res["delta_m_MeV"] == pytest.approx(DELTA_M_PDG["K0"], rel=1e-9)
    assert res["R_vib"] == pytest.approx(1.0, rel=1e-9)


def test_delta_m_v5p30c_alpha_zero_equals_v5p30():
    """α = 0 ⇒ R_vib = 1 partout ⇒ V5.30c ≡ V5.30."""
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res_v530c = delta_m_v5p30c(
            cfg["knots"], cfg["mass_pdg"], 0.0, KAPPA_K_V5p30_DEFAULT
        )
        res_v530 = delta_m_v5p30(
            cfg["knots"], cfg["mass_pdg"], KAPPA_K_V5p30_DEFAULT
        )
        assert res_v530c["delta_m_MeV"] == pytest.approx(
            res_v530["delta_m_MeV"], rel=1e-9
        )


def test_delta_m_v5p30c_positive_and_finite():
    """Toutes les Δm V5.30c sont positives, finies, pour α calibré."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        res = delta_m_v5p30c(
            cfg["knots"], cfg["mass_pdg"],
            ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
        )
        assert res["delta_m_MeV"] > 0
        assert np.isfinite(res["delta_m_MeV"])


def test_delta_m_v5p30c_keeps_hierarchy_K_lt_Bs():
    """V5.30c doit préserver K⁰ << B_s⁰ (le plus discriminant)."""
    cfg_K = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    cfg_Bs = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    dm_K = delta_m_v5p30c(
        cfg_K["knots"], cfg_K["mass_pdg"],
        ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
    )["delta_m_MeV"]
    dm_Bs = delta_m_v5p30c(
        cfg_Bs["knots"], cfg_Bs["mass_pdg"],
        ALPHA_VIB_DEFAULT, KAPPA_K_V5p30_DEFAULT,
    )["delta_m_MeV"]
    assert dm_K < dm_Bs


# ---------------------------------------------------------------------------
# 5. Calibration
# ---------------------------------------------------------------------------
def test_calibrate_alpha_global_returns_valid_alpha():
    cal = calibrate_alpha_global(
        alpha_range=(-5.0, 15.0),
        n_grid=401,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    assert "alpha" in cal
    assert "chi2_log" in cal
    assert "per_meson" in cal
    assert np.isfinite(cal["alpha"])
    assert np.isfinite(cal["chi2_log"])
    # K⁰ doit avoir log_ratio ≈ 0 (calibrated)
    assert abs(cal["per_meson"]["K0"]["log_ratio"]) < 1e-6


def test_calibrate_alpha_global_improves_over_v5p30():
    """α optimal doit donner Σ|log| < Σ|log| V5.30."""
    from theory.v5p30_combined_oscillation import v5p30_table

    rows_v530 = v5p30_table(KAPPA_K_V5p30_DEFAULT)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows_v530)
    cal = calibrate_alpha_global(
        alpha_range=(-5.0, 15.0),
        n_grid=2001,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    assert cal["sum_abs_log"] < s_530, (
        f"V5.30c (Σ|log|={cal['sum_abs_log']:.3f}) doit améliorer "
        f"V5.30 (Σ|log|={s_530:.3f})"
    )


# ---------------------------------------------------------------------------
# 6. Table de comparaison
# ---------------------------------------------------------------------------
def test_v5p30c_table_has_all_4_mesons():
    rows = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    names = [r["name"] for r in rows]
    assert set(names) == {"K0", "D0", "B0", "Bs0"}


def test_v5p30c_table_includes_required_fields():
    rows = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    required = {
        "name", "knots", "delta_m_v5p30c_MeV", "delta_m_v5p30_MeV",
        "delta_m_pdg_MeV", "log_ratio_v5p30c", "log_ratio_v5p30",
        "R_vib", "P_vib", "omega_link", "n_comp", "alpha", "kappa_K",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"Champs manquants dans {r['name']}: {missing}"
