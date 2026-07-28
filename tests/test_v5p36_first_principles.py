"""Tests V5.36 — dérivation first-principles de β et δ.

Vérifie :
1. L'API expose les 10 candidats β et 10 candidats δ.
2. Chaque candidat porte un id, formule, valeur, target, interprétation, flag FP.
3. Les valeurs calibrées de référence sont β = -1,3000 et δ = -2,7667.
4. Le meilleur candidat β first-principles a écart < 1 %.
5. Le meilleur candidat δ first-principles a écart < 1 %.
6. La validation produit un verdict ∈ {STRONG, ACCEPTABLE, WEAK, NONE}.
7. Le tri rank_candidates est strictement croissant en erreur.
8. La fonction summary_report retourne une chaîne non vide.
9. Les valeurs des candidats utilisent k=135 pour les formules SU(2) (cohérence avec V5).
10. La cohérence entre BETA_CALIBRATED / DELTA_CALIBRATED et V5.35b / V5.35d.
"""
from __future__ import annotations

import pytest

from theory.v5p36_first_principles_beta_delta import (
    BETA_CALIBRATED,
    DELTA_CALIBRATED,
    Candidate,
    best_first_principles,
    candidate_beta_formulas,
    candidate_delta_formulas,
    rank_candidates,
    summary_report,
    validate_against_calibrated_values,
)


# ---------------------------------------------------------------------------
# 1. Bonne taille des listes
# ---------------------------------------------------------------------------
def test_ten_beta_candidates():
    cands = candidate_beta_formulas()
    assert len(cands) == 10, "10 candidats β attendus"


def test_ten_delta_candidates():
    cands = candidate_delta_formulas()
    assert len(cands) == 10, "10 candidats δ attendus"


# ---------------------------------------------------------------------------
# 2. Structure de chaque Candidate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidates_fn", [candidate_beta_formulas, candidate_delta_formulas]
)
def test_candidate_structure(candidates_fn):
    for c in candidates_fn():
        assert isinstance(c, Candidate)
        assert isinstance(c.name, str) and c.name
        assert isinstance(c.formula, str) and c.formula
        assert isinstance(c.interpretation, str) and c.interpretation
        assert isinstance(c.value, float)
        assert isinstance(c.target, float)
        assert isinstance(c.is_first_principles, bool)
        assert c.target in (BETA_CALIBRATED, DELTA_CALIBRATED)


# ---------------------------------------------------------------------------
# 3. Valeurs calibrées de référence
# ---------------------------------------------------------------------------
def test_calibrated_constants():
    # V5.35b : β optimal = -1,3000 sur grid (-3, +3)
    assert BETA_CALIBRATED == pytest.approx(-1.30, abs=1e-6)
    # V5.35d : δ optimal = -2,7667 sur grid (-10, +10)
    assert DELTA_CALIBRATED == pytest.approx(-2.7667, abs=1e-3)


# ---------------------------------------------------------------------------
# 4-5. Le meilleur first-principles a écart < 1 % pour chaque paramètre
# ---------------------------------------------------------------------------
def test_best_fp_beta_acceptable():
    cands = candidate_beta_formulas()
    best = best_first_principles(cands)
    assert best is not None, "au moins un candidat β first-principles attendu"
    assert best.relative_error < 0.01, (
        f"meilleur β FP doit avoir <1 % d'écart, "
        f"trouvé {best.name} err {best.relative_error*100:.2f} %"
    )


def test_best_fp_delta_acceptable():
    cands = candidate_delta_formulas()
    best = best_first_principles(cands)
    assert best is not None, "au moins un candidat δ first-principles attendu"
    assert best.relative_error < 0.01, (
        f"meilleur δ FP doit avoir <1 % d'écart, "
        f"trouvé {best.name} err {best.relative_error*100:.2f} %"
    )


# ---------------------------------------------------------------------------
# 6. Verdict ∈ {STRONG, ACCEPTABLE, WEAK, NONE}
# ---------------------------------------------------------------------------
def test_verdict_in_known_set():
    res = validate_against_calibrated_values()
    for tag in ("beta", "delta"):
        assert res[tag]["verdict"] in {"STRONG", "ACCEPTABLE", "WEAK", "NONE"}
        # Les deux paramètres ont au moins un FP, donc pas NONE
        assert res[tag]["verdict"] != "NONE"


# ---------------------------------------------------------------------------
# 7. rank_candidates trie strictement croissant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidates_fn", [candidate_beta_formulas, candidate_delta_formulas]
)
def test_ranking_monotonic(candidates_fn):
    ranked = rank_candidates(candidates_fn())
    for a, b in zip(ranked, ranked[1:]):
        assert a.relative_error <= b.relative_error


# ---------------------------------------------------------------------------
# 8. summary_report retourne une chaîne non vide
# ---------------------------------------------------------------------------
def test_summary_report_non_empty():
    s = summary_report()
    assert isinstance(s, str)
    assert "V5.36" in s
    assert "BETA" in s and "DELTA" in s


# ---------------------------------------------------------------------------
# 9. SU(2)_135 utilisé : la valeur du candidat D2 dépend de k=135
# ---------------------------------------------------------------------------
def test_d2_uses_su2_135():
    # D2 = -4 ln(d_{1/2}(k=135)).  À k=135, d_{1/2} = 2 cos(π/137) ≈ 1,9995.
    # Donc D2 ≈ -4 × ln(1,9995) ≈ -2,7715.
    cands = candidate_delta_formulas()
    by_name = {c.name: c for c in cands}
    assert "D2" in by_name
    d2 = by_name["D2"]
    assert d2.value == pytest.approx(-2.7715, abs=5e-3)
    assert d2.is_first_principles


# ---------------------------------------------------------------------------
# 10. Cohérence avec V5.35b / V5.35d : β cible et δ cible vivent dans le bassin
# ---------------------------------------------------------------------------
def test_targets_consistent_with_calibration_grids():
    # V5.35b a calibré β sur grid (-3, 3) avec δ ∈ [-3, 3] : il est à l'intérieur
    assert -3 < BETA_CALIBRATED < 3
    # V5.35d a calibré δ sur grid (-10, 10) : il est à l'intérieur
    assert -10 < DELTA_CALIBRATED < 10
    # Les meilleurs FP doivent être très proches : la dérivation est crédible
    for tag, fn in (
        ("beta", candidate_beta_formulas),
        ("delta", candidate_delta_formulas),
    ):
        best = best_first_principles(fn())
        assert best is not None
        assert abs(best.value - best.target) < 0.05, (
            f"[{tag}] meilleur FP {best.name} trop éloigné de la cible"
        )
