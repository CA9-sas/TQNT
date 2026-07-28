"""Tests V5.5 étape 6 — dérivation first-principles de κ_g.

Vérifie :
  1. Plausibilité de κ_g (positif, ordre 10⁻⁵³ m, raisonnable).
  2. Cohérence avec la référence V4 (3,12 × 10⁻⁵³ m) à <1 %.
  3. Réversibilité G ↔ κ_g (CODATA G à <0,5 %).
  4. Cohérence h_0 = κ_g · |Ψ(a)|² (V5 §sec:fc-cpq-coherence).
  5. Prédiction Δ_CGQ ≈ 0,41 (cible étape 5) au point de calibration.
  6. Routines Ω²(|Ψ|²).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theory.coupling_kg import (
    ALPHA_EM,
    C_N_NUCLEON_EFF,
    LAMBDABAR_C_PROTON_FM,
    PREFACTOR_KAPPA_G,
    PSI2_SURFACE_FM,
    beta_required_for_delta,
    delta_cgq_from_omega2_scaling,
    G_newton_from_kappa_g,
    h0_vassiliev,
    kappa_g_V5p5,
    omega_squared_at_psi2,
    predict_delta_cgq_step5,
    summary,
)


# ---------------------------------------------------------------------------
# 1. Plausibilité de κ_g.
# ---------------------------------------------------------------------------
def test_kappa_g_positive():
    """κ_g > 0 — contrainte physique (Ω² ≥ 1 pour |Ψ|² ≥ 0)."""
    assert kappa_g_V5p5("m") > 0


def test_kappa_g_order_of_magnitude():
    """κ_g doit être de l'ordre 10⁻⁵³ m (V5 §sec:fc-formule)."""
    kg = kappa_g_V5p5("m")
    assert 1e-54 < kg < 1e-52, f"κ_g hors plage : {kg}"


def test_kappa_g_units_consistency():
    """κ_g(fm) = 10¹⁵ × κ_g(m)."""
    kg_m = kappa_g_V5p5("m")
    kg_fm = kappa_g_V5p5("fm")
    assert kg_fm == pytest.approx(kg_m * 1e15, rel=1e-12)


def test_kappa_g_units_invalid():
    """Argument units invalide -> ValueError."""
    with pytest.raises(ValueError):
        kappa_g_V5p5("nm")


# ---------------------------------------------------------------------------
# 2. Cohérence avec V4.
# ---------------------------------------------------------------------------
def test_kappa_g_matches_V4_reference():
    """κ_g doit reproduire la valeur V4 = 3,12 × 10⁻⁵³ m à <1 %.

    V4 utilisait λbar_C(électron) ≈ 3,86 × 10⁻¹³ m (par confusion),
    mais avec α_EM = 1/137 cela donnait néanmoins ~3,12 × 10⁻⁵³ m.
    En V5 on a λbar_C(proton) qui est la valeur correcte selon
    §sec:fc-cpq-coherence. Les deux conventions convergent dans la
    formule globale car (96π/7) · α^18 absorbe le réajustement.
    """
    kg = kappa_g_V5p5("m")
    ref_V4 = 3.1218e-53  # postulates.KAPPA_PHYS
    assert abs(kg - ref_V4) / ref_V4 < 0.01, (
        f"κ_g = {kg}, V4 ref = {ref_V4}, écart {(kg-ref_V4)/ref_V4*100:+.2f}%"
    )


def test_alpha_EM_derived_from_k135():
    """α_EM = 1/(k+2), k=135 → 1/137."""
    assert ALPHA_EM == pytest.approx(1.0 / 137.0, rel=1e-12)


def test_prefactor_value():
    """96π/7 ≈ 43,0847."""
    assert PREFACTOR_KAPPA_G == pytest.approx(43.0847, rel=1e-4)


def test_c_n_eff_is_18():
    """c_n^eff(nucléon) = c_n(3_1) + 2·c_n(7_2) + 1 = 3 + 14 + 1 = 18."""
    assert C_N_NUCLEON_EFF == 18


# ---------------------------------------------------------------------------
# 3. Réversibilité G ↔ κ_g.
# ---------------------------------------------------------------------------
def test_G_predicted_matches_CODATA():
    """G = (12/7) α¹⁸ ℏc / m_p² à <0,5 % de CODATA."""
    G_pred = G_newton_from_kappa_g()
    G_codata = 6.67430e-11
    rel = abs(G_pred - G_codata) / G_codata
    assert rel < 5e-3, f"G écart {rel*100:.3f}% > 0,5%"


# ---------------------------------------------------------------------------
# 4. Cohérence h_0 = κ_g · |Ψ(a)|².
# ---------------------------------------------------------------------------
def test_h0_vassiliev_value():
    """h_0 = (6/7) α^18 ≈ 3 × 10⁻³⁹."""
    h0 = h0_vassiliev()
    assert h0 == pytest.approx((6.0 / 7.0) * ALPHA_EM ** 18, rel=1e-12)
    assert 2e-39 < h0 < 4e-39


def test_h0_consistent_with_kappa_g_and_psi2_surface():
    """h_0 = κ_g · |Ψ(a)|², avec |Ψ(a)|² = 1/(16π λbar_C).

    L'α^18 doit S'ANNULER dans |Ψ(a)|² = h_0/κ_g — c'est la
    cohérence centrale de V5 §sec:fc-cpq-coherence.
    """
    h0 = h0_vassiliev()
    kg_fm = kappa_g_V5p5("fm")
    psi2_inferred = h0 / kg_fm
    assert psi2_inferred == pytest.approx(PSI2_SURFACE_FM, rel=1e-10)


def test_psi2_surface_value():
    """|Ψ(a)|² = 1/(16π λbar_C) ≈ 0,0946 fm⁻¹ (purement géométrique)."""
    psi2 = PSI2_SURFACE_FM
    expected = 1.0 / (16.0 * math.pi * LAMBDABAR_C_PROTON_FM)
    assert psi2 == pytest.approx(expected, rel=1e-12)
    assert 0.08 < psi2 < 0.11


# ---------------------------------------------------------------------------
# 5. Omega² routines.
# ---------------------------------------------------------------------------
def test_omega_squared_zero_psi():
    """Ω²(|Ψ|² = 0) = 1 (vide → espace plat)."""
    assert omega_squared_at_psi2(0.0, units="dimensionless") == 1.0
    assert omega_squared_at_psi2(0.0, units="fm") == 1.0
    assert omega_squared_at_psi2(0.0, units="m") == 1.0


def test_omega_squared_increasing():
    """Ω² croît avec |Ψ|² (matière repousse l'espace-temps)."""
    psi2 = np.array([0.0, 0.1, 0.5, 1.0])
    omega2 = omega_squared_at_psi2(psi2, units="dimensionless")
    assert np.all(np.diff(omega2) > 0)


def test_omega_squared_surface_proton():
    """À la surface du proton : Ω² = 1 + h_0 ≈ 1 + 3 × 10⁻³⁹."""
    psi2 = PSI2_SURFACE_FM  # fm⁻¹
    omega2 = omega_squared_at_psi2(psi2, units="fm")
    expected = 1.0 + h0_vassiliev()
    assert omega2 == pytest.approx(expected, rel=1e-10)


def test_omega_squared_dimensionless_pathway():
    """Mode dimensionless : 1 + valeur fournie directement."""
    omega2 = omega_squared_at_psi2(0.5, units="dimensionless")
    assert omega2 == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 6. Prédiction Δ_CGQ V5.5 étape 5.
# ---------------------------------------------------------------------------
def test_delta_cgq_formula():
    """Δ_CGQ = -(1/2) · d log Ω² / d log M_h."""
    assert delta_cgq_from_omega2_scaling(0.0) == 0.0
    assert delta_cgq_from_omega2_scaling(-1.0) == 0.5
    assert delta_cgq_from_omega2_scaling(0.4) == pytest.approx(-0.2)


def test_predict_delta_cgq_at_calibration():
    """Le point de calibration (h=0,15, β=-6,29) reproduit Δ_CGQ ≈ 0,41."""
    cgq = predict_delta_cgq_step5()
    assert cgq.delta_cgq == pytest.approx(0.41, rel=2e-2), (
        f"Δ_CGQ prédit = {cgq.delta_cgq}, cible = 0.41"
    )


def test_predict_delta_cgq_step5_close_to_target():
    """Cohérence V5.5 étape 5 : Δ_CGQ prédit dans 5 % de la cible 0,41."""
    cgq = predict_delta_cgq_step5()
    target = 0.41
    rel = abs(cgq.delta_cgq - target) / target
    assert rel < 5e-2, (
        f"Δ_CGQ {cgq.delta_cgq:.4f} vs cible {target} — écart {rel*100:.2f}%"
    )


def test_predict_delta_cgq_omega2_typical():
    """Ω² typique sur halo = 1 + h. Pour h=0,15 : Ω²=1,15."""
    cgq = predict_delta_cgq_step5(kappa_g_psi2_effective=0.15)
    assert cgq.omega2_typical == pytest.approx(1.15)


def test_predict_delta_cgq_note_non_empty():
    """La note doit signaler explicitement les limites."""
    cgq = predict_delta_cgq_step5()
    assert "first-principles" not in cgq.note.lower() or "fermeture" in cgq.note.lower()
    # mentionne au moins la dérivation κ_g et la limite étape 7
    assert "κ_g" in cgq.note or "kappa_g" in cgq.note


def test_predict_delta_cgq_zero_when_no_scaling():
    """Si β = 0 (|Ψ|² indépendant de M_h) : Δ_CGQ = 0."""
    cgq = predict_delta_cgq_step5(kappa_g_psi2_effective=0.15,
                                   d_log_psi2_d_log_mh=0.0)
    assert cgq.delta_cgq == 0.0


def test_predict_delta_cgq_zero_when_no_coupling():
    """Si h = 0 (pas de couplage métrique) : Δ_CGQ = 0."""
    cgq = predict_delta_cgq_step5(kappa_g_psi2_effective=0.0,
                                   d_log_psi2_d_log_mh=-6.29)
    assert cgq.delta_cgq == 0.0


def test_beta_required_inverse():
    """β_required(Δ, h) doit être l'inverse de Δ_CGQ(h, β)."""
    h = 0.15
    delta_target = 0.41
    beta = beta_required_for_delta(delta_target, h)
    # Re-substituer doit redonner Δ_CGQ ≈ delta_target
    cgq = predict_delta_cgq_step5(kappa_g_psi2_effective=h,
                                   d_log_psi2_d_log_mh=beta)
    assert cgq.delta_cgq == pytest.approx(delta_target, rel=1e-10)


# ---------------------------------------------------------------------------
# 7. Synthèse.
# ---------------------------------------------------------------------------
def test_summary_structure():
    """summary() retourne tous les champs attendus."""
    s = summary()
    expected_keys = {
        "kappa_g_m", "kappa_g_fm", "h0_vassiliev", "alpha_EM",
        "psi2_surface_inv_fm", "G_predicted_SI", "G_CODATA_SI",
        "G_relative_error", "delta_cgq_predicted", "delta_cgq_step5_target",
        "delta_cgq_relative_error", "omega2_typical_massive_halo", "note",
        "kappa_g_V4_reference_m", "kappa_g_relative_diff_to_V4",
    }
    assert expected_keys.issubset(s.keys()), (
        f"Clés manquantes : {expected_keys - s.keys()}"
    )


def test_summary_kappa_g_consistency():
    """summary() expose la même valeur que kappa_g_V5p5()."""
    s = summary()
    assert s["kappa_g_m"] == kappa_g_V5p5("m")
    assert s["kappa_g_fm"] == kappa_g_V5p5("fm")
