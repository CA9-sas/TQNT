"""Tests V5.29c — Kashaev / volume conjecture comme préfacteur knot-spécifique.

Cadre : extension de V5.28 (commit 962c96f) par un préfacteur volume
hyperbolique motivé par la volume conjecture (Kashaev 1995, Murakami-
Murakami 2001).

Ces tests vérifient :
1.  Volumes hyperboliques tabulés (torus knots = 0, autres > 0)
2.  Pas d'erreur silencieuse sur knots manquants
3.  Préfacteur P_Kashaev(L) = exp(κ_K · Σ vol)
4.  Calibration K⁰ préservée (par construction)
5.  Réduction du gap B⁰ → B_s⁰ (V5.29c amplifie B_s par rapport à V5.28)
6.  Honnêteté : la hiérarchie complète n'est pas reproduite (D⁰ tombe sous K⁰)
7.  Volume conjecture qualitative : |J(K; exp(2πi/N))| reste fini pour N ≥ 2
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.kashaev_invariants import (
    hyperbolic_volume,
    is_hyperbolic,
    kashaev_phase,
    total_hyperbolic_volume,
    prefactor_volume,
    hyperbolic_volume_table,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    KAPPA_K_DEFAULT,
    delta_m_chiral_oscillation,
    delta_m_chiral_oscillation_v29c,
    chiral_oscillation_table_v29c,
    calibrate_kappa_K_to_hierarchy,
)


# --------------------------------------------------------------------------
# 1. Volumes hyperboliques
# --------------------------------------------------------------------------
def test_torus_knots_vol_zero():
    """Les torus knots T(2, q) du catalogue ont vol = 0 (non-hyperboliques)."""
    for name in ("0_1", "3_1", "5_1", "7_1", "9_1"):
        assert hyperbolic_volume(name) == 0.0, (
            f"{name} doit avoir vol = 0 (torus / unknot)"
        )
        assert not is_hyperbolic(name)


def test_hyperbolic_knots_vol_positive():
    """Les nœuds hyperboliques du catalogue ont vol > 0."""
    for name in ("4_1", "5_2", "6_1", "6_2", "6_3", "7_2", "8_3"):
        assert hyperbolic_volume(name) > 0.0
        assert is_hyperbolic(name)


def test_figure_eight_volume_canonical():
    """4_1 (figure-eight) : vol ≈ 2 × Gieseking ≈ 2.0299."""
    vol = hyperbolic_volume("4_1")
    assert vol == pytest.approx(2.0299, abs=1e-3)


def test_unknown_knot_raises():
    """Demander vol(K) pour un nœud absent du catalogue → KeyError."""
    with pytest.raises(KeyError):
        hyperbolic_volume("99_42")


# --------------------------------------------------------------------------
# 2. Volume total et préfacteur
# --------------------------------------------------------------------------
def test_total_volume_additive():
    """vol_tot(L) = Σ vol(K_i) — sommation simple."""
    vol = total_hyperbolic_volume(["3_1", "6_2", "7_7"])
    expected = 0.0 + hyperbolic_volume("6_2") + hyperbolic_volume("7_7")
    assert vol == pytest.approx(expected, abs=1e-10)


def test_prefactor_volume_unity_at_zero_kappa():
    """κ_K = 0 → P_Kashaev = 1 (V5.28 retrouvé)."""
    p = prefactor_volume(["6_1", "6_1", "6_1"], 0.0)
    assert p == pytest.approx(1.0, abs=1e-12)


def test_prefactor_volume_exponential_in_vol():
    """P_Kashaev = exp(κ_K · vol_tot)."""
    knots = ["3_1", "6_2", "7_7"]
    kappa = 0.5
    p = prefactor_volume(knots, kappa)
    vol = total_hyperbolic_volume(knots)
    assert p == pytest.approx(np.exp(kappa * vol), rel=1e-12)


# --------------------------------------------------------------------------
# 3. Kashaev phase (sanity check)
# --------------------------------------------------------------------------
def test_kashaev_phase_finite():
    """|J(K; exp(2πi/N))| reste fini pour N ≥ 2 sur tous les knots."""
    for name in ("3_1", "4_1", "6_3", "7_2", "8_3"):
        for N in (2, 5, 10, 20):
            val = kashaev_phase(name, N)
            assert np.isfinite(abs(val))


def test_kashaev_phase_unknot_trivial():
    """⟨unknot⟩_N = J(0_1; q) = 1 ∀ q."""
    for N in (2, 5, 10):
        val = kashaev_phase("0_1", N)
        assert abs(val - 1.0) < 1e-12


# --------------------------------------------------------------------------
# 4. Δm V5.29c : calibration K⁰ et structure
# --------------------------------------------------------------------------
def test_K0_calibration_v29c():
    """K⁰ V5.29c = K⁰ PDG par construction (le préfacteur global absorbe
    ω_0·ℏ·P_Kashaev(K⁰))."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for kappa in (0.0, 0.5, 1.0, 1.5):
        res = delta_m_chiral_oscillation_v29c(
            cfg["knots"], cfg["mass_pdg"], kappa
        )
        rel = abs(res["delta_m_MeV"] - DELTA_M_PDG["K0"]) / DELTA_M_PDG["K0"]
        assert rel < 1e-9, f"K0 calibration drift à κ={kappa}: {rel}"


def test_v29c_reduces_to_v528_when_kappa_zero():
    """κ_K = 0 → V5.29c ≡ V5.28 (le préfacteur de volume devient 1, et la
    renormalisation sur K⁰ est commune)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["B0"]
    r29 = delta_m_chiral_oscillation_v29c(cfg["knots"], cfg["mass_pdg"], 0.0)
    r528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
    rel = abs(r29["delta_m_MeV"] - r528["delta_m_MeV"]) / r528["delta_m_MeV"]
    assert rel < 1e-9, f"V5.29c(κ=0) doit être égal à V5.28 ; rel = {rel}"


def test_v29c_table_structure():
    """La table V5.29c contient les 4 mésons et tous les champs requis."""
    rows = chiral_oscillation_table_v29c(KAPPA_K_DEFAULT)
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}
    required = {
        "delta_m_v29c_MeV", "delta_m_v528_MeV", "delta_m_pdg_MeV",
        "log_ratio_v29c", "log_ratio_v528", "vol_tot", "P_Kashaev",
    }
    for r in rows:
        assert required.issubset(set(r.keys()))


# --------------------------------------------------------------------------
# 5. Honnêteté : V5.29c améliore B_s mais ne reproduit pas tout
# --------------------------------------------------------------------------
def test_Bs_gap_closed_relative_to_v528():
    """V5.29c doit amplifier B_s⁰ (vol_tot grand → P_Kashaev grand) et
    réduire le gap par rapport à V5.28."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    r29 = delta_m_chiral_oscillation_v29c(
        cfg["knots"], cfg["mass_pdg"], KAPPA_K_DEFAULT
    )
    r528 = delta_m_chiral_oscillation(cfg["knots"], cfg["mass_pdg"])
    log_29c = abs(np.log10(r29["delta_m_MeV"] / DELTA_M_PDG["Bs0"]))
    log_528 = abs(np.log10(r528["delta_m_MeV"] / DELTA_M_PDG["Bs0"]))
    assert log_29c < log_528, (
        f"V5.29c devrait améliorer B_s ; |log_29c|={log_29c:.3f} "
        f"vs |log_528|={log_528:.3f}"
    )


def test_Bs_over_B_ratio_increases():
    """V5.29c augmente le ratio B_s⁰/B⁰ (vol(B_s) > vol(B), donc le
    préfacteur amplifie davantage B_s)."""
    rows = chiral_oscillation_table_v29c(KAPPA_K_DEFAULT)
    by_name = {r["name"]: r["delta_m_v29c_MeV"] for r in rows}
    by_name_528 = {r["name"]: r["delta_m_v528_MeV"] for r in rows}
    ratio_29c = by_name["Bs0"] / by_name["B0"]
    ratio_528 = by_name_528["Bs0"] / by_name_528["B0"]
    assert ratio_29c > ratio_528, (
        f"V5.29c doit augmenter Bs/B (vol_tot(Bs)>vol_tot(B)) ; "
        f"V5.29c={ratio_29c:.3f}, V5.28={ratio_528:.3f}"
    )


def test_hierarchy_K_B_Bs_correct_but_D_problematic():
    """Honnêteté : V5.29c reproduit l'ordre K < B < B_s mais pas D⁰ > K⁰.
    Le préfacteur supprime D⁰ trop fortement (vol_tot(D) = 3.33 alors que
    vol_tot(K) = 9.49, donc P_Kashaev(D)/P_Kashaev(K) = exp(-κ × 6.16))."""
    rows = chiral_oscillation_table_v29c(KAPPA_K_DEFAULT)
    by_name = {r["name"]: r["delta_m_v29c_MeV"] for r in rows}
    assert by_name["K0"] < by_name["B0"] < by_name["Bs0"], (
        "Le sous-ordre K < B < B_s doit être correct"
    )
    # D⁰ tombe en dessous de K⁰ — limite honnête du modèle
    assert by_name["D0"] < by_name["K0"], (
        "Honnêteté : V5.29c supprime D⁰ trop ; à documenter"
    )


# --------------------------------------------------------------------------
# 6. Calibration empirique
# --------------------------------------------------------------------------
def test_calibration_kappa_finds_minimum():
    """Le scan empirique sur D, B, B_s trouve un κ_K fini dans la fenêtre."""
    res = calibrate_kappa_K_to_hierarchy(kappa_range=(-1.0, 2.0), n_grid=201)
    assert -1.0 < res["kappa_K"] < 2.0
    assert np.isfinite(res["chi2_log"])
    # Le minimum doit correspondre à un κ_K positif (préfacteur amplifie
    # vol_tot grand → B, B_s).
    assert res["kappa_K"] > 0.0, (
        "κ_K optimal devrait être positif (amplifier B_s)"
    )


def test_default_kappa_K_consistent_with_calibration():
    """KAPPA_K_DEFAULT doit être proche du résultat du scan empirique."""
    res = calibrate_kappa_K_to_hierarchy(kappa_range=(-1.0, 2.0), n_grid=201)
    assert abs(KAPPA_K_DEFAULT - res["kappa_K"]) < 0.05, (
        f"KAPPA_K_DEFAULT={KAPPA_K_DEFAULT} doit être proche de "
        f"l'optimum calibré {res['kappa_K']:.3f}"
    )
