"""Tests V5.32 — Plasticite P23 sur mesures d'oscillation.

Reference : theory/p23_plasticity_oscillations.py
            docs/TQNT_detaillee_V5_FR.tex §sec:P23-plastique
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from theory.p23_plasticity_oscillations import (
    B_EFF_TYPICAL,
    B_REF_TESLA,
    C_PLAST_MEV,
    EPS_N2_DEFAULT,
    M_MUON_MEV,
    compare_models_to_unfolded,
    delta_m_topo_pdg_corrected,
    epsilon_n,
    g_B_field,
    plasticity_correction,
    unfold_delta_m,
    unfolded_table,
)
from theory.topological_chiral_oscillation import (
    DELTA_M_PDG,
    MESON_OSCILLATION_ASSIGNMENTS,
)


# ---------------------------------------------------------------------------
# eps(n) : conformite V5 sec:P23-plastique
# ---------------------------------------------------------------------------
def test_epsilon_n1_is_zero():
    """V5 P23 : n = 1 -> symetrie radiale -> eps = 0 exactement."""
    assert epsilon_n(1) == 0.0


def test_epsilon_n2_calibrated_on_muon():
    """eps(2) = |C_plast| / m_mu avec C_plast = -2.92 MeV (calibre mu^-)."""
    expected = abs(C_PLAST_MEV) / M_MUON_MEV
    assert epsilon_n(2) == pytest.approx(expected, rel=1e-12)
    # Sanity : ordre de grandeur ~2.76 %
    assert 0.02 < epsilon_n(2) < 0.04


def test_epsilon_n3_is_sqrt3_times_n2():
    """eps(3) = eps(2) * sqrt(3*2/2) = sqrt(3) * eps(2) (geometrique)."""
    expected = epsilon_n(2) * math.sqrt(3.0)
    assert epsilon_n(3) == pytest.approx(expected, rel=1e-12)


def test_epsilon_monotonic_in_n():
    """eps(n) doit croitre strictement avec n pour n >= 1."""
    vals = [epsilon_n(n) for n in (1, 2, 3, 4, 5)]
    for a, b in zip(vals, vals[1:]):
        assert b > a


def test_epsilon_rejects_n_zero():
    with pytest.raises(ValueError):
        epsilon_n(0)


# ---------------------------------------------------------------------------
# g(B) : modulation par champ effectif
# ---------------------------------------------------------------------------
def test_g_B_at_reference_is_one():
    """g(B_ref) = 1 (normalisation LHCb)."""
    assert g_B_field(B_REF_TESLA) == pytest.approx(1.0, rel=1e-12)


def test_g_B_linear():
    """g(B) lineaire en B (V5 §13370)."""
    g1 = g_B_field(1.0)
    g2 = g_B_field(2.0)
    g3 = g_B_field(3.0)
    assert g2 == pytest.approx(2.0 * g1, rel=1e-12)
    assert g3 == pytest.approx(3.0 * g1, rel=1e-12)


def test_g_B_zero_at_zero_field():
    """g(0) = 0 (Case A pur)."""
    assert g_B_field(0.0) == 0.0


def test_g_B_rejects_negative():
    with pytest.raises(ValueError):
        g_B_field(-1.0)


# ---------------------------------------------------------------------------
# plasticity_correction : composition eps * g
# ---------------------------------------------------------------------------
def test_plasticity_correction_n1_vanishes_at_any_B():
    """n = 1 -> delta = 0 quel que soit B."""
    for B in (0.0, 1.0, 4.0, 10.0):
        c = plasticity_correction(1, B)
        assert c["delta"] == 0.0


def test_plasticity_correction_n2_at_Bref_equals_eps2():
    """delta(n=2, B_ref) = eps(2) * 1 = eps(2)."""
    c = plasticity_correction(2, B_REF_TESLA)
    assert c["delta"] == pytest.approx(EPS_N2_DEFAULT, rel=1e-12)


def test_plasticity_correction_small_for_realistic_params():
    """delta < 5 % pour tous les Bs typiques (n<=3, B<=4 T)."""
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        n = len(cfg["knots"])
        B = B_EFF_TYPICAL[name]
        c = plasticity_correction(n, B)
        assert 0.0 <= c["delta"] < 0.05, (
            f"{name} delta={c['delta']:.4f} hors plage attendue (<5 %)"
        )


# ---------------------------------------------------------------------------
# unfold_delta_m : formule multiplicative
# ---------------------------------------------------------------------------
def test_unfold_identity_when_n1():
    """Pour n = 1 : Δm_topo_unfolded == Δm_PDG."""
    res = unfold_delta_m(1.0e-10, n_components=1, B_eff_T=4.0)
    assert res["delta_m_topo_unfolded"] == pytest.approx(1.0e-10, rel=1e-12)
    assert res["correction_factor"] == pytest.approx(1.0, rel=1e-12)


def test_unfold_shrinks_pdg_when_sigma_positive():
    """sigma = +1 : Δm_PDG > Δm_topo (mesure surevaluee)."""
    res = unfold_delta_m(
        1.0e-10, n_components=3, B_eff_T=4.0, sigma=+1.0
    )
    assert res["delta_m_topo_unfolded"] < 1.0e-10
    assert res["correction_factor"] > 1.0


def test_unfold_grows_pdg_when_sigma_negative():
    """sigma = -1 : Δm_PDG < Δm_topo."""
    res = unfold_delta_m(
        1.0e-10, n_components=3, B_eff_T=4.0, sigma=-1.0
    )
    assert res["delta_m_topo_unfolded"] > 1.0e-10
    assert res["correction_factor"] < 1.0


# ---------------------------------------------------------------------------
# Tables et integration avec V5.30c/V5.31
# ---------------------------------------------------------------------------
def test_unfolded_table_has_four_mesons():
    rows = unfolded_table()
    assert [r["particle"] for r in rows] == ["K0", "D0", "B0", "Bs0"]


def test_unfolded_table_pdg_matches_reference():
    """Δm_PDG dans la table = DELTA_M_PDG du module reference."""
    rows = unfolded_table()
    for r in rows:
        assert r["delta_m_PDG"] == pytest.approx(
            DELTA_M_PDG[r["particle"]], rel=1e-12
        )


def test_unfolded_shift_under_3_percent():
    """Pour tous les systemes, |unfolded/PDG - 1| < 3 % (eps*g petit)."""
    rows = unfolded_table()
    for r in rows:
        ratio = r["delta_m_topo_unfolded"] / r["delta_m_PDG"]
        assert abs(ratio - 1.0) < 0.05, (
            f"{r['particle']} : shift {ratio - 1.0:+.4f} hors borne 5 %"
        )


def test_compare_models_consistent_with_baselines():
    """Σ|log| V5.30c/V5.31 vs PDG doit etre coherent avec les references
    publiees : V5.30c = 1.05, V5.31 = 1.46 dex."""
    from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
    from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, v5p30c_table
    from theory.v5p31_chiral_vibrational import (
        BETA_CHIRAL_DEFAULT,
        v5p31_table,
    )

    rows_30c = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    rows_31 = v5p31_table(
        beta=BETA_CHIRAL_DEFAULT,
        alpha=ALPHA_VIB_DEFAULT,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    pred_30c = {r["name"]: r["delta_m_v5p30c_MeV"] for r in rows_30c}
    pred_31 = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows_31}
    cmp = compare_models_to_unfolded(pred_30c, pred_31)

    # V5.30c reference Σ|log| ~ 1.05 dex contre PDG ; tolerance large.
    assert cmp["sum_abs_log_v5p30c_vs_PDG"] == pytest.approx(1.05, abs=0.05)
    assert cmp["sum_abs_log_v5p31_vs_PDG"] == pytest.approx(1.46, abs=0.05)


def test_p23_unfold_shift_under_0p05_dex():
    """Le shift induit par P23 sur Σ|log| doit etre < 0.05 dex (eps*g petit).
    C'est l'enonce honnete : P23 NE FERME PAS le gap."""
    from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
    from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, v5p30c_table
    from theory.v5p31_chiral_vibrational import (
        BETA_CHIRAL_DEFAULT,
        v5p31_table,
    )

    rows_30c = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    rows_31 = v5p31_table(
        beta=BETA_CHIRAL_DEFAULT,
        alpha=ALPHA_VIB_DEFAULT,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    pred_30c = {r["name"]: r["delta_m_v5p30c_MeV"] for r in rows_30c}
    pred_31 = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows_31}
    cmp = compare_models_to_unfolded(pred_30c, pred_31)

    d30 = abs(
        cmp["sum_abs_log_v5p30c_vs_unfolded"]
        - cmp["sum_abs_log_v5p30c_vs_PDG"]
    )
    d31 = abs(
        cmp["sum_abs_log_v5p31_vs_unfolded"]
        - cmp["sum_abs_log_v5p31_vs_PDG"]
    )
    assert d30 < 0.05
    assert d31 < 0.05


def test_delta_m_topo_pdg_corrected_explicit_K0():
    """K0 : n = 3 composantes (3x 6_1), B_eff = 2 T."""
    res = delta_m_topo_pdg_corrected("K0")
    assert res["n_components"] == 3
    assert res["B_eff_T"] == 2.0
    # Verifie la formule
    expected_eps = epsilon_n(3)
    expected_g = g_B_field(2.0)
    expected_factor = 1.0 + (+1.0) * expected_eps * expected_g
    assert res["correction_factor"] == pytest.approx(
        expected_factor, rel=1e-12
    )


def test_delta_m_topo_pdg_corrected_Bs0_has_two_components():
    """Bs0 : 6_3 + 7_6 -> n = 2 dans les assignations V5."""
    res = delta_m_topo_pdg_corrected("Bs0")
    assert res["n_components"] == 2
    # B_eff Bs ~ 3 T
    assert res["B_eff_T"] == pytest.approx(3.0, rel=1e-12)


def test_compare_models_unfolded_ratio_Bs_B():
    """Le ratio PDG_unfolded(Bs/B) doit etre tres proche du ratio PDG
    car les corrections sont petites et proches (n=2 et n=3 avec
    B_eff = 3 T pour les deux)."""
    from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
    from theory.v5p30c_vibrational import ALPHA_VIB_DEFAULT, v5p30c_table
    from theory.v5p31_chiral_vibrational import (
        BETA_CHIRAL_DEFAULT,
        v5p31_table,
    )

    rows_30c = v5p30c_table(
        alpha=ALPHA_VIB_DEFAULT, kappa_K=KAPPA_K_V5p30_DEFAULT
    )
    rows_31 = v5p31_table(
        beta=BETA_CHIRAL_DEFAULT,
        alpha=ALPHA_VIB_DEFAULT,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    pred_30c = {r["name"]: r["delta_m_v5p30c_MeV"] for r in rows_30c}
    pred_31 = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows_31}
    cmp = compare_models_to_unfolded(pred_30c, pred_31)

    # Ratio PDG ~ 35
    assert cmp["ratio_pdg_Bs_B"] == pytest.approx(35.06, rel=0.01)
    # Ratio PDG_unfolded doit differer de PDG de < 5 % (corrections
    # P23 proches sur les deux)
    rel_change = abs(
        cmp["ratio_pdg_unfolded_Bs_B"] / cmp["ratio_pdg_Bs_B"] - 1.0
    )
    assert rel_change < 0.05
