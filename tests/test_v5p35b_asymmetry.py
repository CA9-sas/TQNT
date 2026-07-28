"""Tests V5.35b — asymétrie de complexité dans le coupling inertiel."""
from __future__ import annotations

import numpy as np
import pytest

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p34_chiral_inertia import chiral_inertia
from theory.v5p35b_complexity_asymmetry import (
    FORMS,
    asymmetry_factor,
    calibrate_beta_global,
    chiral_inertia_v5p35b,
    compare_forms,
    delta_m_v5p35b,
    omega_chirality_v5p35b,
    v5p35b_table,
)


# ---------------------------------------------------------------------------
# 1. asymmetry_factor — invariance c_i = c_j
# ---------------------------------------------------------------------------
def test_asymmetry_equal_complexity_returns_one():
    """A_ij = 1 quand c_i = c_j (pour les deux formes)."""
    for form in FORMS:
        for beta in (-2.0, -0.5, 0.0, 0.5, 2.0):
            A = asymmetry_factor(6, 6, beta, form=form)
            assert A == pytest.approx(1.0), (
                f"form={form}, β={beta}: A({6},{6})={A} ≠ 1"
            )


# ---------------------------------------------------------------------------
# 2. asymmetry_factor — beta=0 redonne A=1 quelles que soient c_i, c_j
# ---------------------------------------------------------------------------
def test_asymmetry_beta_zero_is_unity():
    """β = 0 ⇒ A_ij = 1 partout (forme rel et ratio)."""
    for form in FORMS:
        for (ci, cj) in [(3, 7), (5, 11), (4, 4), (6, 9)]:
            A = asymmetry_factor(ci, cj, 0.0, form=form)
            assert A == pytest.approx(1.0), (
                f"form={form}, β=0, (c_i, c_j)=({ci},{cj}): A={A} ≠ 1"
            )


# ---------------------------------------------------------------------------
# 3. asymmetry_factor — symétrie c_i ↔ c_j
# ---------------------------------------------------------------------------
def test_asymmetry_symmetric_in_arguments():
    """A_ij(c_i, c_j) = A_ij(c_j, c_i)."""
    for form in FORMS:
        for beta in (-1.5, -0.3, 0.5, 1.7):
            A1 = asymmetry_factor(3, 7, beta, form=form)
            A2 = asymmetry_factor(7, 3, beta, form=form)
            assert A1 == pytest.approx(A2)


# ---------------------------------------------------------------------------
# 4. chiral_inertia_v5p35b — K⁰ identique à V5.34 (toutes c_i égales)
# ---------------------------------------------------------------------------
def test_K0_inertia_invariant_under_beta():
    """K⁰ = 3 × 6_1 a c uniformes ⇒ I_χ V5.35b(K⁰) indépendant de β."""
    K0_knots = MESON_OSCILLATION_ASSIGNMENTS["K0"]["knots"]
    I_v34 = chiral_inertia(K0_knots, "F3_coupling", 1.0)["I_chi"]
    for form in FORMS:
        for beta in (-2.0, -0.5, 0.0, 0.5, 2.0):
            I_v35b = chiral_inertia_v5p35b(K0_knots, beta, form)["I_chi"]
            assert I_v35b == pytest.approx(I_v34, rel=1e-10), (
                f"form={form}, β={beta}: I(K⁰) varie ({I_v35b} vs {I_v34})"
            )


# ---------------------------------------------------------------------------
# 5. chiral_inertia_v5p35b — β=0 redonne V5.34 F3 pour tous les mésons
# ---------------------------------------------------------------------------
def test_beta_zero_matches_v5p34_F3():
    """β = 0 ⇒ I_χ V5.35b ≡ I_χ V5.34(F3_coupling) pour les 4 mésons."""
    for name in ("K0", "D0", "B0", "Bs0"):
        knots = MESON_OSCILLATION_ASSIGNMENTS[name]["knots"]
        I_v34 = chiral_inertia(knots, "F3_coupling", 1.0)["I_chi"]
        for form in FORMS:
            I_v35b = chiral_inertia_v5p35b(knots, 0.0, form)["I_chi"]
            assert I_v35b == pytest.approx(I_v34, rel=1e-10), (
                f"{name}, form={form}, β=0: {I_v35b} vs {I_v34}"
            )


# ---------------------------------------------------------------------------
# 6. delta_m_v5p35b — K⁰ exactement = Δm PDG par construction
# ---------------------------------------------------------------------------
def test_delta_m_K0_calibrated():
    """Δm V5.35b(K⁰) = Δm PDG(K⁰) (à <1% près, calibration V5.30c × R_χ=1)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for form in FORMS:
        for beta in (-1.5, -0.3, 0.0, 0.7, 1.5):
            res = delta_m_v5p35b(cfg["knots"], cfg["mass_pdg"], beta, form)
            ratio = res["delta_m_MeV"] / DELTA_M_PDG["K0"]
            # K⁰ doit rester calibré à mieux que 1% (V5.30c calibré sur K⁰)
            assert abs(np.log10(ratio)) < 0.02, (
                f"K⁰ décalibré : form={form}, β={beta}, log_ratio="
                f"{np.log10(ratio):+.4f}"
            )


# ---------------------------------------------------------------------------
# 7. omega_chirality_v5p35b — ω_χ > 0 pour tous les systèmes, β raisonnable
# ---------------------------------------------------------------------------
def test_omega_chi_positive_finite():
    """ω_χ V5.35b > 0 et fini pour {K, D, B, B_s}, β ∈ [-2, 2]."""
    for name in ("K0", "D0", "B0", "Bs0"):
        knots = MESON_OSCILLATION_ASSIGNMENTS[name]["knots"]
        for form in FORMS:
            for beta in (-1.5, -0.5, 0.0, 0.5, 1.5):
                oc = omega_chirality_v5p35b(knots, beta, form)
                assert oc["omega_chi"] > 0
                assert np.isfinite(oc["omega_chi"])
                assert oc["I_chi"] > 0
                assert oc["F_torque"] > 0


# ---------------------------------------------------------------------------
# 8. calibrate_beta_global — convergence, β fini, sum_abs_log fini
# ---------------------------------------------------------------------------
def test_calibration_converges_both_forms():
    """La calibration converge et renvoie β fini + Σ|log| fini."""
    for form in FORMS:
        out = calibrate_beta_global(form=form, n_grid=1201)
        assert np.isfinite(out["beta"])
        assert np.isfinite(out["sum_abs_log"])
        assert out["sum_abs_log"] > 0
        # K⁰ doit être à zéro à <0.02 dex
        assert abs(out["per_meson"]["K0"]["log_ratio"]) < 0.02


# ---------------------------------------------------------------------------
# 9. calibration — Σ|log| V5.35b ≤ Σ|log| V5.34 (référence 0.820)
# ---------------------------------------------------------------------------
def test_v5p35b_does_not_degrade_v5p34():
    """La calibration V5.35b doit faire au moins aussi bien que V5.34 (β=0
    est dans l'espace de recherche → Σ|log| ≤ 0.820 dex)."""
    sigma_34 = 0.820
    tol = 0.01  # marge sur la grille discrète
    for form in FORMS:
        out = calibrate_beta_global(form=form, n_grid=1201)
        assert out["sum_abs_log"] <= sigma_34 + tol, (
            f"form={form}: V5.35b Σ|log|={out['sum_abs_log']:.3f} > V5.34"
            f" ({sigma_34}) — la calibration n'a pas pu battre β=0"
        )


# ---------------------------------------------------------------------------
# 10. compare_forms — Σ|log| amélioré significativement (V5.35b < V5.34)
#     et table v5p35b_table cohérente avec calibrate_beta_global
# ---------------------------------------------------------------------------
def test_compare_forms_and_table_consistency():
    """V5.35b améliore Σ|log| par rapport à V5.34 (B⁰ partiellement corrigé)
    et la table v5p35b_table est cohérente avec la calibration."""
    results = compare_forms()
    assert "best_form" in results
    best = results["best_form"]
    assert best in FORMS
    block = results[best]
    sigma_34 = 0.820
    # Amélioration nette attendue (au moins 0.1 dex de mieux que V5.34)
    assert block["sum_abs_log"] < sigma_34 - 0.1, (
        f"V5.35b ({best}) Σ|log|={block['sum_abs_log']:.3f} n'améliore pas"
        f" V5.34 ({sigma_34}) de plus de 0.1 dex"
    )
    # Cohérence avec v5p35b_table
    rows = v5p35b_table(block["beta"], form=best)
    by_name = {r["name"]: r for r in rows}
    for name in ("K0", "D0", "B0", "Bs0"):
        dm_tab = by_name[name]["delta_m_v5p35b_MeV"]
        dm_cal = block["per_meson"][name]["delta_m_MeV"]
        assert dm_tab == pytest.approx(dm_cal, rel=1e-10), (
            f"{name}: incohérence table vs calibration "
            f"({dm_tab} vs {dm_cal})"
        )
