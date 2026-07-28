"""Tests V5.33 — Oscillation collective par interaction EM mutuelle."""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import braid_index, is_amphicheiral, jones_polynomial
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30c_vibrational import delta_m_v5p30c
from theory.v5p33_mutual_EM_oscillation import (
    ALPHA_MUTUAL_DEFAULT,
    _PREFACTOR_V5p33_MEV,
    _omega_K0,
    calibrate_alpha_global,
    collective_omega,
    coupling_matrix,
    delta_m_v5p33,
    effective_linking,
    magnetic_moment,
    mutual_torque,
    self_frequency,
    v5p33_table,
)


# ---------------------------------------------------------------------------
# 1. Moments magnétiques
# ---------------------------------------------------------------------------
def test_magnetic_moment_is_braid_over_crossing():
    """μ(K) = b(K) / c(K), entier positif divisé par entier positif."""
    for k in ["3_1", "5_1", "6_1", "6_2", "6_3", "7_2", "7_6", "7_7"]:
        m = magnetic_moment(k)
        assert m["b"] == braid_index(k)
        assert m["c"] == crossing_number(k)
        assert m["mu"] == pytest.approx(m["b"] / m["c"])
        assert m["mu"] > 0


def test_magnetic_moment_6_3_amphi_high():
    """6_3 amphichiral a μ = 4/6 = 0.667 (élevé : b=4, c=6)."""
    m = magnetic_moment("6_3")
    assert m["mu"] == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# 2. Fréquence propre
# ---------------------------------------------------------------------------
def test_self_frequency_is_b_over_sqrt_c():
    """ω_self(K) = b(K) / √c(K)."""
    for k in ["3_1", "6_1", "6_3", "7_2"]:
        s = self_frequency(k)
        assert s["omega_self"] == pytest.approx(
            braid_index(k) / np.sqrt(crossing_number(k))
        )
        assert s["omega_self_sq"] == pytest.approx(s["omega_self"] ** 2)


def test_self_frequency_amphichiral_highest():
    """6_3 (b=4, c=6) a ω_self = 4/√6 ≈ 1.633, le plus élevé du catalogue."""
    s_63 = self_frequency("6_3")
    s_31 = self_frequency("3_1")
    s_61 = self_frequency("6_1")
    assert s_63["omega_self"] > s_31["omega_self"]
    assert s_63["omega_self"] > s_61["omega_self"]


# ---------------------------------------------------------------------------
# 3. Linking effectif
# ---------------------------------------------------------------------------
def test_effective_linking_chiral_pair_positive():
    """Deux chiraux : χ_ij = +1, Lk_eff = +√(c_i c_j) > 0."""
    lk = effective_linking("3_1", "5_1")
    assert lk["is_amphi_i"] is False
    assert lk["is_amphi_j"] is False
    assert lk["chi_ij"] == 1.0
    assert lk["lk_eff"] == pytest.approx(np.sqrt(3 * 5))


def test_effective_linking_amphichiral_pair_negative():
    """Au moins un amphichiral : χ_ij = -1, Lk_eff < 0."""
    lk = effective_linking("6_3", "7_6")
    assert lk["is_amphi_i"] is True  # 6_3 amphichiral
    assert lk["chi_ij"] == -1.0
    assert lk["lk_eff"] < 0
    assert abs(lk["lk_eff"]) == pytest.approx(np.sqrt(6 * 7))


def test_effective_linking_symmetric_in_args():
    """Lk_eff(i, j) = Lk_eff(j, i)."""
    a = effective_linking("3_1", "7_2")
    b = effective_linking("7_2", "3_1")
    assert a["lk_eff"] == pytest.approx(b["lk_eff"])


# ---------------------------------------------------------------------------
# 4. Couple mutuel
# ---------------------------------------------------------------------------
def test_mutual_torque_scales_with_alpha():
    """τ_ij est linéaire en α."""
    t1 = mutual_torque("3_1", "5_1", alpha=1.0)["tau"]
    t2 = mutual_torque("3_1", "5_1", alpha=2.0)["tau"]
    assert t2 == pytest.approx(2.0 * t1)


def test_mutual_torque_zero_when_alpha_zero():
    """α = 0 ⇒ τ_ij = 0 pour toute paire."""
    t = mutual_torque("3_1", "7_2", alpha=0.0)["tau"]
    assert t == 0.0


def test_mutual_torque_amphi_flips_sign():
    """La paire amphichirale (Bs : 6_3+7_6) a Lk_eff < 0 ⇒ τ < 0 quand α > 0."""
    t_chiral = mutual_torque("3_1", "5_1", alpha=1.0)["tau"]
    t_amphi = mutual_torque("6_3", "7_6", alpha=1.0)["tau"]
    assert t_chiral > 0
    assert t_amphi < 0


# ---------------------------------------------------------------------------
# 5. Matrice de couplage
# ---------------------------------------------------------------------------
def test_coupling_matrix_symmetric():
    """K = K^T pour tout link."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        M = coupling_matrix(cfg["knots"], alpha=1.5)["matrix"]
        assert np.allclose(M, M.T, rtol=1e-12)


def test_coupling_matrix_diagonal_is_omega_self_sq():
    """K[i,i] = ω_self(K_i)²."""
    knots = ["3_1", "5_1", "7_2"]  # D⁰
    M = coupling_matrix(knots, alpha=0.5)["matrix"]
    for i, k in enumerate(knots):
        assert M[i, i] == pytest.approx(self_frequency(k)["omega_self_sq"])


def test_coupling_matrix_alpha_zero_is_diagonal():
    """α=0 ⇒ K diagonale (pas de couplage mutuel)."""
    knots = ["3_1", "5_1", "7_2"]
    M = coupling_matrix(knots, alpha=0.0)["matrix"]
    off_diag = M - np.diag(np.diag(M))
    assert np.allclose(off_diag, 0)


# ---------------------------------------------------------------------------
# 6. Mode collectif
# ---------------------------------------------------------------------------
def test_collective_omega_finite_all_systems():
    """ω_collective fini > 0 pour les 4 mésons."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        co = collective_omega(cfg["knots"], alpha=ALPHA_MUTUAL_DEFAULT)
        assert co["omega_collective"] > 0
        assert np.isfinite(co["omega_collective"])


def test_collective_omega_equals_max_eigenvalue_sqrt():
    """ω_collective = √λ_max."""
    knots = ["6_1", "6_1", "6_1"]
    co = collective_omega(knots, alpha=1.0)
    lmax = co["lambda_max"]
    assert co["omega_collective"] == pytest.approx(np.sqrt(abs(lmax)))


# ---------------------------------------------------------------------------
# 7. Δm V5.33
# ---------------------------------------------------------------------------
def test_delta_m_v5p33_K0_equals_pdg_by_construction():
    """Δm V5.33(K⁰) = Δm_PDG(K⁰) par construction (R_EM(K⁰)=1)."""
    _PREFACTOR_V5p33_MEV.clear()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p33(
        cfg["knots"], cfg["mass_pdg"], alpha=ALPHA_MUTUAL_DEFAULT
    )
    assert res["R_EM"] == pytest.approx(1.0)
    # Et par construction Δm V5.30c(K⁰) = Δm_PDG(K⁰)
    assert res["delta_m_MeV"] == pytest.approx(
        DELTA_M_PDG["K0"], rel=1e-10
    )


def test_delta_m_v5p33_reduces_to_v530c_when_R_EM_is_one():
    """Si R_EM = 1 (cas K⁰) Δm V5.33 = Δm V5.30c."""
    _PREFACTOR_V5p33_MEV.clear()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p33(
        cfg["knots"], cfg["mass_pdg"], alpha=ALPHA_MUTUAL_DEFAULT
    )
    assert res["delta_m_MeV"] == pytest.approx(
        res["delta_m_v5p30c_MeV"], rel=1e-10
    )


def test_delta_m_v5p33_finite_all_systems():
    """Δm V5.33 fini et > 0 pour les 4 mésons."""
    _PREFACTOR_V5p33_MEV.clear()
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        res = delta_m_v5p33(
            cfg["knots"], cfg["mass_pdg"], alpha=ALPHA_MUTUAL_DEFAULT
        )
        assert res["delta_m_MeV"] > 0
        assert np.isfinite(res["delta_m_MeV"])


# ---------------------------------------------------------------------------
# 8. Calibration globale
# ---------------------------------------------------------------------------
def test_calibrate_alpha_returns_valid_alpha():
    """La calibration retourne un α fini."""
    _PREFACTOR_V5p33_MEV.clear()
    cal = calibrate_alpha_global(
        alpha_range=(-50.0, 50.0), n_grid=501
    )
    assert np.isfinite(cal["alpha"])
    assert -50.0 <= cal["alpha"] <= 50.0
    assert cal["sum_abs_log"] >= 0


def test_calibrate_alpha_K0_contributes_zero():
    """K⁰ contribue zéro au χ² (R_EM(K⁰) = 1 par construction)."""
    _PREFACTOR_V5p33_MEV.clear()
    cal = calibrate_alpha_global(
        alpha_range=(-10.0, 10.0), n_grid=201
    )
    assert cal["per_meson"]["K0"]["log_ratio"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 9. Table V5.33
# ---------------------------------------------------------------------------
def test_v5p33_table_has_four_mesons():
    """La table contient exactement K0, D0, B0, Bs0."""
    rows = v5p33_table(alpha=ALPHA_MUTUAL_DEFAULT)
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}


def test_v5p33_table_log_ratios_finite():
    """Tous les log_ratio sont finis (Δm > 0)."""
    rows = v5p33_table(alpha=ALPHA_MUTUAL_DEFAULT)
    for r in rows:
        assert r["log_ratio_v5p33"] is not None
        assert np.isfinite(r["log_ratio_v5p33"])
        assert r["log_ratio_v5p30c"] is not None


def test_v5p33_table_K0_row_matches_pdg():
    """La ligne K⁰ de la table reproduit exactement Δm_PDG(K⁰)."""
    rows = v5p33_table(alpha=ALPHA_MUTUAL_DEFAULT)
    K0_row = next(r for r in rows if r["name"] == "K0")
    assert K0_row["delta_m_v5p33_MeV"] == pytest.approx(
        DELTA_M_PDG["K0"], rel=1e-10
    )
    assert K0_row["log_ratio_v5p33"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 10. Cohérence avec V5.30c
# ---------------------------------------------------------------------------
def test_v5p33_recovers_v530c_when_R_EM_is_one():
    """Quand R_EM = 1 (cas K⁰), Δm V5.33 = Δm V5.30c."""
    _PREFACTOR_V5p33_MEV.clear()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    dm_530c = delta_m_v5p30c(cfg["knots"], cfg["mass_pdg"])["delta_m_MeV"]
    dm_533 = delta_m_v5p33(
        cfg["knots"], cfg["mass_pdg"], alpha=ALPHA_MUTUAL_DEFAULT
    )["delta_m_MeV"]
    assert dm_533 == pytest.approx(dm_530c, rel=1e-10)
