"""Tests V5p27 (nouveau) — Oscillations K-K̄, B-B̄, D-D̄ via P15.

CLAUDE.md règle 0 : la saveur SM est un artefact, pas un concept TQNT.
Les phénomènes d'oscillation (K, D, B, Bs) sont des reconnexions
topologiques P15.

La matrice CKM (mélange de saveurs) et la phase δ_CP (qui confondait
chiralité Dirac et opération miroir topologique) ne sont PAS dérivées
ici : ce sont des artefacts du SM (dérivés d'artefacts).  En TQNT, la
chiralité = opération miroir V_K*(t) = V_K(t⁻¹) sur les invariants
topologiques, sans rapport avec la handedness L/R des spineurs.
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.p15_topological_reconnection import (
    chirality_sigma,
    is_link_amphichiral,
    topological_distance_to_mirror,
    reconnection_rate,
    delta_m_oscillation,
    oscillation_comparison_table,
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    HBAR_MEV_S,
    NU_OSC,
    LAMBDA_OSC,
)


# --- 1. Chiralité = opération miroir topologique (PAS handedness Dirac) ---
def test_chirality_amphichiral_zero():
    """Les nœuds amphichiraux (4_1, 6_3, 8_3) ont σ = 0 (V_K = V_K*)."""
    for n in ["4_1", "6_3", "8_3"]:
        assert chirality_sigma(n) == 0


def test_chirality_chiral_nonzero():
    """Les nœuds chiraux (3_1, 5_1, 7_1) ont σ ≠ 0 (V_K ≠ V_K*)."""
    for n in ["3_1", "5_1", "7_1"]:
        assert chirality_sigma(n) != 0


def test_tqnt_chirality_is_mirror_operation():
    """TQNT chirality = opération miroir V_K*(t) = V_K(t⁻¹), PAS la
    handedness Dirac L/R des spineurs de Weyl.

    Un nœud amphichiral satisfait V_K(t) = V_K(t⁻¹) à un facteur
    monomial près : σ(K) = 0.  Test exact via mirror_jones du module
    theory.jones.
    """
    from theory.jones import jones_polynomial, mirror_jones, is_amphicheiral
    # 4_1 : amphichiral connu (figure-eight knot)
    p = jones_polynomial("4_1")
    pm = mirror_jones(p)
    assert is_amphicheiral(p), "4_1 doit être amphichiral"
    # Sous l'opération miroir, le polynôme de Jones de 4_1 est invariant
    assert p == pm, "V_{4_1}(t) doit être égal à V_{4_1}(t⁻¹)"
    # 3_1 : chiral (trèfle) — V_K ≠ V_K*
    p3 = jones_polynomial("3_1")
    pm3 = mirror_jones(p3)
    assert p3 != pm3, "V_{3_1}(t) doit différer de V_{3_1}(t⁻¹)"
    assert not is_amphicheiral(p3)


def test_no_ckm_exposed_in_module():
    """Aucun symbole CKM/δ_CP/chiralité-Dirac exposé par le module P15.

    CKM mélange des saveurs (artefact SM) ; δ_CP confondait chiralité
    Dirac et opération miroir.  En vertu de la règle 0 CLAUDE.md,
    aucun de ces concepts n'est dérivé.
    """
    import theory.p15_topological_reconnection as p15
    names = dir(p15)
    forbidden_substrings = [
        "ckm",          # CKM matrix mixing flavor eigenstates
        "cp_phase",     # δ_CP confounded Dirac chirality with mirror op
        "delta_cp",
        "CKM",
        "delta_CP",
        "CP_phase",
    ]
    for sym in names:
        for bad in forbidden_substrings:
            assert bad not in sym, (
                f"Symbole interdit '{sym}' (contient '{bad}') : "
                "CKM/δ_CP retirés (artefacts SM)"
            )


def test_link_amphichiral_K0_not():
    """K⁰ = 6_1 + 6_1 + 6_1 : 6_1 est chiral, donc le link n'est pas amphichiral."""
    assert is_link_amphichiral(["6_1", "6_1", "6_1"]) is False


def test_link_amphichiral_pure():
    """Un link 4_1 + 6_3 (deux amphichiraux) est amphichiral."""
    assert is_link_amphichiral(["4_1", "6_3"]) is True


# --- 2. Distance topologique au miroir ---
def test_distance_mirror_positive():
    """d(L, L*) > 0 pour tous les mésons (oscillation possible)."""
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        d = topological_distance_to_mirror(cfg["knots"])
        assert d > 0, f"{name} a d=0 (pas d'oscillation possible)"


def test_distance_mirror_K0_value():
    """d(K0=6_1+6_1+6_1, L*) = 3×6 = 18 (somme des croisements)."""
    assert topological_distance_to_mirror(["6_1", "6_1", "6_1"]) == 18


# --- 3. Calibration K0 ---
def test_K0_delta_m_calibrated():
    """K⁰ : Δm V5 = Δm PDG (par construction de la calibration)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_oscillation(cfg["knots"], cfg["k"])
    rel = abs(res["delta_m_MeV"] - DELTA_M_PDG["K0"]) / DELTA_M_PDG["K0"]
    assert rel < 1e-6, f"K0 calibration drift : {rel}"


# --- 4. Δm pour les mésons : prédictions V5 ---
def test_D0_delta_m_within_decade():
    """D⁰ : Δm V5 dans 1 décade de PDG (test honnête : facteur ~0.5)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["D0"]
    res = delta_m_oscillation(cfg["knots"], cfg["k"])
    log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["D0"])
    assert abs(log_ratio) < 1.0, f"D0 hors décade : log10={log_ratio:.2f}"


def test_B0_delta_m_underpredicted():
    """B⁰ : V5 sous-prédit (limite honnête du modèle, log ≈ -3.3)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["B0"]
    res = delta_m_oscillation(cfg["knots"], cfg["k"])
    # Honnête : V5 sous-prédit B0 de ~3 décades
    assert res["delta_m_MeV"] < DELTA_M_PDG["B0"]
    log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["B0"])
    assert log_ratio < -2.0, "B0 devrait être sous-prédit"


def test_Bs0_delta_m_underpredicted():
    """Bs⁰ : V5 sous-prédit (~2 décades), honnête."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    res = delta_m_oscillation(cfg["knots"], cfg["k"])
    assert res["delta_m_MeV"] < DELTA_M_PDG["Bs0"]


# --- 5. Taux de reconnexion / Δm relation ---
def test_delta_m_equals_hbar_rate():
    """Δm = ℏ × Γ_reconnex (cohérence dimensionnelle)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    r = reconnection_rate(cfg["knots"], cfg["k"])
    dm = delta_m_oscillation(cfg["knots"], cfg["k"])
    assert abs(dm["delta_m_MeV"] - HBAR_MEV_S * r["rate"]) / dm["delta_m_MeV"] < 1e-9


# --- 6. Table comparaison ---
def test_oscillation_table_complete():
    """La table contient les 4 mésons (K0, D0, B0, Bs0)."""
    rows = oscillation_comparison_table()
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}


def test_oscillation_table_log_ratios_finite():
    """Tous les log_ratios sont finis (V5 prédit Δm > 0)."""
    rows = oscillation_comparison_table()
    for r in rows:
        assert r["log_ratio"] is not None
        assert np.isfinite(r["log_ratio"])


# --- 7. Constantes ---
def test_oscillation_exponents():
    """Exposants d'oscillation distincts de la durée de vie (ν=1, λ=1
    vs ν=5π/2, λ=5)."""
    assert NU_OSC == 1.0
    assert LAMBDA_OSC == 1.0
