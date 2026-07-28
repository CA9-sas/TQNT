"""Tests pour V5.35e — extension oscillations a K_S/K_L, quarkonium, XYZ."""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.v5p35e_extended_systems import (
    EXTENDED_TOPOLOGY,
    DELTA_M_EXTENDED_PDG_MEV,
    V5p35D_DELTA,
    V5p35D_FORM,
    V5p35D_BETA,
    V5p35D_FORM_B,
    assign_topology,
    omega_chi_v5p35d,
    delta_m_v5p35d_generalized,
    v5p35e_extended_table,
    verdict_v5p35e,
)


# ---------------------------------------------------------------------------
# 1. Parametres calibres
# ---------------------------------------------------------------------------
def test_v5p35d_parameters_are_calibrated():
    """δ, β, formes sont les valeurs V5.35d calibrees (commit fee4d23)."""
    assert V5p35D_FORM == "seq_asym"
    assert V5p35D_FORM_B == "rel"
    assert abs(V5p35D_DELTA - (-2.78)) < 0.05
    assert abs(V5p35D_BETA - (-1.30)) < 0.05


# ---------------------------------------------------------------------------
# 2. Assignations topologiques V5 (catalogue xlsx)
# ---------------------------------------------------------------------------
def test_K_S_and_K_L_share_K0_topology():
    """K_S, K_L sont des etats propres CP du meme nœuds 6_1^3 que K0."""
    cfg_S = assign_topology("K_S")
    cfg_L = assign_topology("K_L")
    assert cfg_S["knots"] == ["6_1", "6_1", "6_1"]
    assert cfg_L["knots"] == ["6_1", "6_1", "6_1"]
    # Memes nœuds pour K_S et K_L
    assert sorted(cfg_S["knots"]) == sorted(cfg_L["knots"])


def test_J_psi_and_psi2S_have_distinct_topologies():
    """J/ψ et ψ(2S) sont des topologies distinctes en V5 (pas un splitting)."""
    cfg_j = assign_topology("J/psi")
    cfg_p = assign_topology("psi(2S)")
    assert cfg_j["knots"] == ["3_1", "4_1", "6_3"]
    assert cfg_p["knots"] == ["6_1", "8_2"]
    assert sorted(cfg_j["knots"]) != sorted(cfg_p["knots"])


def test_X_3872_and_Zc_3900_share_topology():
    """X(3872) et Z_c(3900) partagent 3_1+7_2+7_7 (catalogue V5 onglet 3)."""
    cfg_x = assign_topology("X(3872)")
    cfg_z = assign_topology("Z_c(3900)")
    assert cfg_x["knots"] == ["3_1", "7_2", "7_7"]
    assert cfg_z["knots"] == ["3_1", "7_2", "7_7"]


# ---------------------------------------------------------------------------
# 3. Δm V5.35d generalisee : mode chiral_oscillation pour K_S
# ---------------------------------------------------------------------------
def test_K_S_chiral_oscillation_reproduces_K0_PDG():
    """Δm V5.35d sur K_S = Δm calibree sur K0 = 3.484e-12 MeV (PDG exact)."""
    res = delta_m_v5p35d_generalized("K_S", None)
    assert res["mode"] == "chiral_oscillation"
    assert res["applicable"] is True
    # K_S est etat propre CP du systeme K0 : ω_χ et Δm sont identiques
    # au K0 deja calibre par V5.35d (Σ|log| ≈ 0.005 sur K0/D0/B0/Bs0).
    dm = res["delta_m_v5p35d_MeV"]
    assert dm > 0
    # PDG : 3.484e-12 MeV.  Tolerance large car calibration K0 globale.
    log_r = math.log10(dm / 3.484e-12)
    assert abs(log_r) < 0.05


# ---------------------------------------------------------------------------
# 4. Δm V5.35d generalisee : mode same_topology
# ---------------------------------------------------------------------------
def test_K_S_vs_K_L_same_topology_no_splitting():
    """K_S vs K_L : memes nœuds → V5.35d retourne Δm = 0 (correctement)."""
    res = delta_m_v5p35d_generalized("K_S", "K_L")
    assert res["mode"] == "same_topology"
    assert res["applicable"] is False
    assert res["delta_m_v5p35d_MeV"] == 0.0
    assert res["R_chi"] == 1.0


def test_X_3872_vs_Zc_3900_same_topology_no_splitting():
    """X(3872) et Z_c(3900) partagent la topologie → Δm V5.35d = 0."""
    res = delta_m_v5p35d_generalized("X(3872)", "Z_c(3900)")
    assert res["mode"] == "same_topology"
    assert res["applicable"] is False
    assert res["delta_m_v5p35d_MeV"] == 0.0


# ---------------------------------------------------------------------------
# 5. Δm V5.35d generalisee : mode radial_splitting (J/ψ-ψ(2S), Y(1S)-Y(2S))
# ---------------------------------------------------------------------------
def test_J_psi_vs_psi2S_radial_splitting_marked_not_applicable():
    """J/ψ → ψ(2S) : topologies distinctes, mode radial_splitting."""
    res = delta_m_v5p35d_generalized("J/psi", "psi(2S)")
    assert res["mode"] == "radial_splitting"
    assert res["applicable"] is False  # V5.35d n'est PAS un splitting radial
    # log_ratio sera tres negatif (>3 dex) → V5.35d sous-predit massivement
    log_r = math.log10(
        abs(res["delta_m_v5p35d_MeV"]) / DELTA_M_EXTENDED_PDG_MEV["psi(2S)-J/psi"]
    )
    assert log_r < -3.0


def test_Upsilon_radial_splitting_marked_not_applicable():
    """Y(1S) → Y(2S) : topologies distinctes, mode radial_splitting."""
    res = delta_m_v5p35d_generalized("Upsilon(1S)", "Upsilon(2S)")
    assert res["mode"] == "radial_splitting"
    assert res["applicable"] is False
    # ω_χ Y(2S) - ω_χ Y(1S) doit etre largement <<<< PDG (560 MeV)
    assert abs(res["delta_m_v5p35d_MeV"]) < 10.0


# ---------------------------------------------------------------------------
# 6. Verdict d'universalite
# ---------------------------------------------------------------------------
def test_verdict_universality_is_SPECIALIZED():
    """Sur les 5 systemes etendus, V5.35d s'applique seulement a 1 (K_S)."""
    rows = v5p35e_extended_table()
    v = verdict_v5p35e(rows)
    assert v["universality"] == "SPECIALIZED"
    assert v["applicable_count"] == 1
    assert v["total_count"] == 5
    # Sur le seul cas applicable (K_S), log_ratio doit etre ≈ 0.
    assert v["mean_log_applicable"] < 0.05
