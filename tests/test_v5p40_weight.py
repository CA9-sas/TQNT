"""Tests V5.40 - coefficients explicites weight system v_4 pour 3-link."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from theory.su2_modular import V5_LEVEL, quantum_dim
from theory.v5p38_delta_derivation import delta_first_principles
from theory.v5p40_weight_system_v4 import (
    CASIMIR_SPIN_HALF,
    DIM_FUNDAMENTAL,
    ChordDiagram,
    WeightSystemResult,
    chord_diagrams_4_chord_3_circle,
    delta_from_weight_system,
    factor_4_status,
    four_T_relations,
    primitive_dimension,
    summary_report,
    verify_casimir_consistency,
    weight_v4_su2,
)


# ---------------------------------------------------------------------------
# 1. Constantes Casimir et representation fondamentale
# ---------------------------------------------------------------------------
def test_casimir_spin_half_exact():
    """C_2(spin-1/2) = 3/4 exactement (rationnel)."""
    assert CASIMIR_SPIN_HALF == Fraction(3, 4)
    assert float(CASIMIR_SPIN_HALF) == 0.75


def test_dim_fundamental_equals_2():
    """dim V_{1/2} = 2 (representation fondamentale SU(2))."""
    assert DIM_FUNDAMENTAL == 2


# ---------------------------------------------------------------------------
# 2. Enumeration des chord diagrams
# ---------------------------------------------------------------------------
def test_chord_diagrams_enumeration_count():
    """Au moins 5 classes principales (Kontsevich, Boucles, Triangle, Split, Crossing)."""
    diagrams = chord_diagrams_4_chord_3_circle()
    assert len(diagrams) == 5
    for D in diagrams:
        assert isinstance(D, ChordDiagram)
        assert D.n_circles == 3
        assert D.n_chords == 4


def test_chord_diagrams_have_8_endpoints():
    """Chaque diagramme d'ordre 4 a exactement 8 points (4 cordes * 2)."""
    for D in chord_diagrams_4_chord_3_circle():
        counts = D.endpoints_per_circle
        assert sum(counts) == 8


def test_chord_diagrams_inter_intra_classification():
    """Les 5 classes ont les bonnes distributions (n_intra, n_inter)."""
    diagrams = chord_diagrams_4_chord_3_circle()
    expected = [
        (0, 4),  # D_K
        (2, 2),  # D_B
        (1, 3),  # D_T
        (4, 0),  # D_S
        (0, 4),  # D_X
    ]
    for D, (ni, ne) in zip(diagrams, expected):
        assert D.intra_chord_count == ni
        assert D.inter_chord_count == ne


# ---------------------------------------------------------------------------
# 3. 4T relations
# ---------------------------------------------------------------------------
def test_four_T_relations_returned():
    """Au moins une 4T relation generee."""
    relations = four_T_relations()
    assert len(relations) >= 1
    for R in relations:
        assert len(R.diagrams) == 4
        assert R.signs == (+1, -1, +1, -1)


# ---------------------------------------------------------------------------
# 4. Weight system : valeurs exactes
# ---------------------------------------------------------------------------
def test_weight_leading_kontsevich_diagram():
    """D_K (4 inter cordes) : w = (1/2)^4 * 2 = 1/8."""
    diagrams = chord_diagrams_4_chord_3_circle()
    D_K = diagrams[0]
    w = weight_v4_su2(D_K)
    assert w == Fraction(1, 8)


def test_weight_split_diagram():
    """D_S (4 intra cordes, 3 cercles) : w = (3/4)^4 * 2^3 = 81/256 * 8 = 648/256 = 81/32."""
    diagrams = chord_diagrams_4_chord_3_circle()
    D_S = diagrams[3]
    w = weight_v4_su2(D_S)
    assert w == Fraction(81, 32)


def test_weight_all_rational():
    """Tous les poids w_sl2 sont des fractions exactes (pas de flottants)."""
    for D in chord_diagrams_4_chord_3_circle():
        w = weight_v4_su2(D)
        assert isinstance(w, Fraction)


# ---------------------------------------------------------------------------
# 5. Dimension primitive P_4 et facteur 4
# ---------------------------------------------------------------------------
def test_primitive_dimension_equals_4():
    """dim P_4 pour 3-link = 4 (Bar-Natan 1995 Theorem 6 + couplage triple)."""
    assert primitive_dimension() == 4


# ---------------------------------------------------------------------------
# 6. delta from weight system : verification V5.38
# ---------------------------------------------------------------------------
def test_delta_from_weight_system_classical_limit():
    """Limite classique k -> inf : delta = -4 ln(2) = -ln(16) = -2.7725887..."""
    res = delta_from_weight_system()
    assert res.delta_classical == pytest.approx(-4.0 * math.log(2.0), abs=1e-12)
    assert res.delta_classical == pytest.approx(-math.log(16.0), abs=1e-12)


def test_delta_from_weight_system_quantum_matches_v5p38():
    """k = 135 : delta = -4 ln(d_{1/2}(135)) = -2.7715369380 (= V5.38)."""
    res = delta_from_weight_system()
    assert res.delta_quantum == pytest.approx(
        delta_first_principles(), abs=1e-12
    )
    assert res.is_consistent


def test_delta_factor_4_explicit():
    """Le 4 vient de dim P_4 = 4 (calcul explicite, pas conjecture)."""
    res = delta_from_weight_system()
    assert res.primitive_dim == 4
    assert res.factor_4_derived


# ---------------------------------------------------------------------------
# 7. Casimir consistency
# ---------------------------------------------------------------------------
def test_casimir_consistency():
    """C_2 = 3/4 + d_{1/2} -> 2 produit delta classique = -ln(16)."""
    chk = verify_casimir_consistency()
    assert chk["casimir_C2_spinhalf"] == 0.75
    assert chk["dim_V_fundamental"] == 2
    assert chk["d_half_classical"] == 2.0
    assert chk["consistent_with_v5p38"]
    # Recouvre V5.36 D1 dans la limite classique
    assert chk["delta_classical_minus4ln2"] == pytest.approx(
        -math.log(16.0), abs=1e-12
    )


# ---------------------------------------------------------------------------
# 8. Statut honnete factor 4
# ---------------------------------------------------------------------------
def test_factor_4_status_verdict():
    """Verdict V5.40 : factor 4 DERIVE RIGOUREUSEMENT (dim P_4 = 4)."""
    status = factor_4_status()
    assert status["verdict"] == "DERIVE RIGOUREUSEMENT"
    assert status["primitive_dimension_P4"] == 4
    assert status["factor_4_derived_as_dim_P4"]
    assert status["v5p38_consistency"]


def test_factor_4_open_questions_disclosed():
    """V5.40 reste honnete : les questions ouvertes sont enumerees."""
    status = factor_4_status()
    assert len(status["open_questions"]) >= 1
    assert any("xhaustive" in q for q in status["open_questions"])


# ---------------------------------------------------------------------------
# 9. Rapport synthetique
# ---------------------------------------------------------------------------
def test_summary_report_keywords():
    rep = summary_report()
    assert "V5.40" in rep
    assert "Casimir" in rep
    assert "Kontsevich" in rep
    assert "primitive" in rep.lower()
    assert "DERIVE RIGOUREUSEMENT" in rep or "PARTIAL" in rep


def test_summary_includes_5_diagrams():
    rep = summary_report()
    for label in ["D_K", "D_B", "D_T", "D_S", "D_X"]:
        assert label in rep
