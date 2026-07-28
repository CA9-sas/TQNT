"""Tests for V5.36a — c_J(3_1) first-principles derivation from SU(2)_135.

Vérifie :
1. La cible TARGET_C_J_3_1 = e^{1,30} ≈ 3,669.
2. La définition canonique du code (M1) donne 3,30 exactement.
3. Les 10 méthodes retournent des valeurs cohérentes.
4. Verdict : WEAK (aucune méthode FP n'atteint <1 %).
5. Honnêteté : best FP est M9 (span·d_{1/2}/√3) à ~5,6 % d'écart.
6. Le tautologique M8 atteint 0 % par construction.
7. cJ_general('3_1') == c_j('3_1') == 3,3.
8. RT à q² donne ≈ 1 (q² ≈ 1 à k=135).
9. β = -ln(c_J) est cohérent pour chaque candidat.
10. Le summary_report contient les mots-clés requis.
"""
import math

import pytest

from theory.mass_formula import c_j
from theory.v5p36a_cJ_first_principles import (
    BETA_CALIBRATED,
    TARGET_C_J_3_1,
    CJCandidate,
    all_candidates,
    best_first_principles,
    cJ_3_1_code_definition,
    cJ_3_1_from_jones_at_q2,
    cJ_3_1_from_wilson_loop,
    cJ_3_1_kashaev_torus,
    cJ_3_1_mahler_measure,
    cJ_3_1_span_plus_two_thirds,
    cJ_3_1_sqrt_sum_sq,
    cJ_3_1_sum_abs_coeffs,
    cJ_3_1_target_tautological,
    cJ_general,
    rank_candidates,
    summary_report,
    verdict,
)


def test_target_value_is_exp_minus_beta_calibrated():
    """La cible c_J = e^{1,30} ≈ 3,669 (β_calibré = -1,30)."""
    assert math.isclose(TARGET_C_J_3_1, math.exp(-BETA_CALIBRATED), rel_tol=1e-12)
    assert math.isclose(TARGET_C_J_3_1, 3.669296668, rel_tol=1e-6)


def test_code_definition_M1_equals_3_3():
    """M1 (code canonique) doit retourner exactement 3,30."""
    val = cJ_3_1_code_definition()
    assert math.isclose(val, 3.3, abs_tol=1e-10)
    # Et c'est cohérent avec c_j('3_1')
    assert math.isclose(val, c_j("3_1"), abs_tol=1e-10)
    # Et avec cJ_general
    assert math.isclose(cJ_general("3_1"), 3.3, abs_tol=1e-10)


def test_M2_wilson_loop_q_squared_near_one():
    """À k=135, q² ≈ 1 donc V_K(q²) ≈ 1 + O(1/k)."""
    val = cJ_3_1_from_jones_at_q2()
    # 1 < val < 1,05 (vu 1,0247 numérique)
    assert 1.0 < val < 1.05


def test_M3_wilson_loop_normalized_below_one():
    """|V_K(q)|/d_{1/2} ≈ 0,503 pour 3_1 à k=135."""
    val = cJ_3_1_from_wilson_loop()
    assert 0.45 < val < 0.55


def test_M4_mahler_measure_in_range():
    """Mahler measure ≈ 1,466 (racine réelle dominante de t³-t²-1)."""
    val = cJ_3_1_mahler_measure()
    assert math.isclose(val, 1.4655712, rel_tol=1e-4)


def test_M5_L1_norm_equals_3():
    """Σ|a_n| = |1| + |1| + |−1| = 3 pour V_{3_1}(t) = t + t³ - t⁴."""
    assert math.isclose(cJ_3_1_sum_abs_coeffs(), 3.0, abs_tol=1e-10)


def test_M6_L2_norm_equals_sqrt3():
    """√(1²+1²+1²) = √3."""
    assert math.isclose(cJ_3_1_sqrt_sum_sq(), math.sqrt(3.0), abs_tol=1e-10)


def test_M7_span_plus_two_thirds_close_to_target():
    """span + 2/3 = 3,667 (0,07 % à la cible)."""
    val = cJ_3_1_span_plus_two_thirds()
    assert math.isclose(val, 3.0 + 2.0 / 3.0, abs_tol=1e-10)
    # Très proche de la cible mais NON first-principles
    err = abs(val - TARGET_C_J_3_1) / TARGET_C_J_3_1
    assert err < 0.001


def test_M8_tautological_matches_target_exactly():
    """exp(-β_calibré) = cible par construction (sanity check)."""
    val = cJ_3_1_target_tautological()
    assert math.isclose(val, TARGET_C_J_3_1, rel_tol=1e-12)


def test_M10_kashaev_torus_equals_1():
    """Torus knot, vol=0 → Kashaev = 1."""
    assert math.isclose(cJ_3_1_kashaev_torus(), 1.0, abs_tol=1e-12)


def test_all_candidates_count():
    """Exactement 10 méthodes."""
    cands = all_candidates()
    assert len(cands) == 10
    names = {c.name for c in cands}
    expected = {f"M{i}" for i in range(1, 11)}
    assert names == expected


def test_beta_consistency_for_each_candidate():
    """β = -ln(c_J) cohérent pour chaque candidat positif."""
    for c in all_candidates():
        if c.value > 0:
            assert math.isclose(c.beta, -math.log(c.value), rel_tol=1e-12)


def test_verdict_is_weak_no_FP_under_1_percent():
    """Aucune méthode first-principles n'atteint <1 % → verdict WEAK ou ACCEPTABLE."""
    v = verdict()
    # On accepte WEAK ou ACCEPTABLE car M9 est à ~5,6 % d'écart
    assert v in ("WEAK", "ACCEPTABLE")
    # M8 (tautologique) ne compte PAS comme FP
    best = best_first_principles()
    assert best is not None
    assert best.is_first_principles is True
    # M9 doit être le meilleur FP (span·d_{1/2}/√3)
    assert best.name == "M9"


def test_M7_M8_not_first_principles():
    """M7 (2/3) et M8 (tautologique) doivent être marqués comme heuristique."""
    cands = {c.name: c for c in all_candidates()}
    assert cands["M7"].is_first_principles is False
    assert cands["M8"].is_first_principles is False


def test_M1_marked_first_principles():
    """M1 (code TQNT) est marqué FP car uniquement span+γ·det (γ=0,10 calibré γ)."""
    cands = {c.name: c for c in all_candidates()}
    assert cands["M1"].is_first_principles is True


def test_rank_candidates_M8_first():
    """Le tri par err_cJ doit placer M8 (tautologique 0 %) en tête."""
    ranked = rank_candidates()
    assert ranked[0].name == "M8"
    # M7 (0,07 %) en deuxième
    assert ranked[1].name == "M7"


def test_summary_report_contains_keywords():
    """Le rapport mentionne les éléments clés du verdict honnête."""
    rep = summary_report()
    assert "V5.36a" in rep
    assert "SU(2)_135" in rep
    assert "3,30" in rep or "3.30" in rep or "3.3000" in rep
    assert "honn" in rep.lower() or "calibr" in rep.lower()
    assert "Verdict" in rep


def test_code_definition_beta_error_is_about_8_percent():
    """β résultant du code (3,30) donne -1,194 → écart 8,2 % à -1,30."""
    val = cJ_3_1_code_definition()
    beta = -math.log(val)
    err = abs(beta - BETA_CALIBRATED) / abs(BETA_CALIBRATED)
    assert 0.07 < err < 0.10  # ~ 8,2 %


def test_no_FP_method_under_1_percent_error():
    """Critère honnêteté : AUCUNE méthode FP < 1 % d'écart sur c_J."""
    for c in all_candidates():
        if c.is_first_principles:
            assert c.relative_error_cJ >= 0.01, (
                f"{c.name} ({c.formula}) atteint {c.relative_error_cJ*100:.2f} % "
                "— hypothèse honnêteté V5.36a violée."
            )
