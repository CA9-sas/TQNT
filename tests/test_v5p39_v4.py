"""Tests V5.39 - derivation rigoureuse facteur 4 via Vassiliev v_4."""
from __future__ import annotations

import math

import pytest

from theory.su2_modular import V5_LEVEL, quantum_dim
from theory.v5p38_delta_derivation import delta_first_principles
from theory.v5p39_vassiliev_v4 import (
    Factor4Derivation,
    VassilievV4Result,
    c2_knot,
    delta_from_v4,
    factor_4_derivation,
    retained_derivations,
    summary_report,
    v4_meson_table,
    vassiliev_v4,
)


# ---------------------------------------------------------------------------
# 1. c_2 table : valeurs cles correspondent a KnotInfo
# ---------------------------------------------------------------------------
def test_c2_known_knots():
    """c_2 (Casson/v_2) pour les noeuds standards V5."""
    assert c2_knot("3_1") == -1
    assert c2_knot("4_1") == +1   # figure-8 amphichirale
    assert c2_knot("6_1") == +2   # amphichirale
    assert c2_knot("7_2") == -3
    assert c2_knot("6_2") == +1


def test_c2_missing_raises():
    """Si un noeud n'est pas tabule, leve KeyError (pas d'extrapolation)."""
    with pytest.raises(KeyError):
        c2_knot("12_999_does_not_exist")


# ---------------------------------------------------------------------------
# 2. v_4 : structure du resultat et ordre primitif
# ---------------------------------------------------------------------------
def test_vassiliev_v4_returns_struct():
    """v_4 retourne un VassilievV4Result avec ordre primitif = 4."""
    res = vassiliev_v4(["6_1", "6_1", "6_1"])
    assert isinstance(res, VassilievV4Result)
    assert res.primitive_order == 4


def test_v4_homogeneous_61_triplet():
    """v_4(6_1+6_1+6_1) = 3 * c_2(6_1)^2 = 3*4 = 12 (mu_123=0)."""
    res = vassiliev_v4(["6_1", "6_1", "6_1"])
    expected = 3 * (c2_knot("6_1") ** 2)
    assert res.v4_total == pytest.approx(expected, abs=1e-12)
    assert res.is_nontrivial


def test_v4_heterogeneous_d0_triplet():
    """v_4(3_1+5_1+7_2) = c2(3_1)*c2(5_1) + c2(3_1)*c2(7_2) + c2(5_1)*c2(7_2)
       = (-1)*(-3) + (-1)*(-3) + (-3)*(-3) = 3 + 3 + 9 = 15."""
    res = vassiliev_v4(["3_1", "5_1", "7_2"])
    expected = (-1) * (-3) + (-1) * (-3) + (-3) * (-3)
    assert res.v4_total == pytest.approx(expected, abs=1e-12)
    assert res.v4_total == pytest.approx(15.0, abs=1e-12)


def test_v4_requires_3_components():
    """v_4 n'est defini que pour un 3-link strict."""
    with pytest.raises(ValueError):
        vassiliev_v4(["6_3", "7_6"])  # 2 composantes seulement (B_s^0)
    with pytest.raises(ValueError):
        vassiliev_v4(["3_1"])  # 1 composante


def test_v4_milnor_triple_contribution():
    """Si mu_123 != 0, v_4 acquiert une contribution quadratique."""
    res0 = vassiliev_v4(["3_1", "3_1", "3_1"], mu123=0.0)
    res1 = vassiliev_v4(["3_1", "3_1", "3_1"], mu123=2.0)
    # delta = mu^2 = 4
    assert res1.v4_total - res0.v4_total == pytest.approx(4.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. delta_from_v4 : prediction coherente avec V5.38
# ---------------------------------------------------------------------------
def test_delta_from_v4_consistent_with_v5p38():
    """delta_predicted via v_4 doit etre EXACTEMENT delta V5.38."""
    res = delta_from_v4(["6_1", "6_1", "6_1"])
    assert res["delta_predicted"] == pytest.approx(
        delta_first_principles(), abs=1e-12
    )
    assert res["is_consistent"]


def test_delta_from_v4_independent_of_link():
    """delta_predicted ne depend PAS du 3-link choisi : c'est une constante
    universelle -4 ln(d_{1/2}(k=135))."""
    r1 = delta_from_v4(["6_1", "6_1", "6_1"])
    r2 = delta_from_v4(["3_1", "5_1", "7_2"])
    r3 = delta_from_v4(["3_1", "6_2", "7_7"])
    assert r1["delta_predicted"] == pytest.approx(r2["delta_predicted"], abs=1e-14)
    assert r2["delta_predicted"] == pytest.approx(r3["delta_predicted"], abs=1e-14)
    assert r1["delta_predicted"] == pytest.approx(
        -4.0 * math.log(quantum_dim(0.5, V5_LEVEL)), abs=1e-12
    )


# ---------------------------------------------------------------------------
# 4. factor_4_derivation : 5 voies, 2 retenues
# ---------------------------------------------------------------------------
def test_five_derivation_paths():
    """Exactement 5 voies enumerees."""
    derivs = factor_4_derivation()
    assert len(derivs) == 5
    labels = {d.label for d in derivs}
    assert labels == {
        "P1_vassiliev_v4",
        "P2_cobordism_3link",
        "P3_modular_4vertex",
        "P4_wilson_quartic",
        "P5_barnatan_gauss_diagrams",
    }


def test_two_paths_retained():
    """Deux voies marquees RETENU (P1, P4) — convergence rigoureuse."""
    retained = retained_derivations()
    assert len(retained) == 2
    labels = {d.label for d in retained}
    assert labels == {"P1_vassiliev_v4", "P4_wilson_quartic"}


def test_retained_paths_are_rigorous():
    """Les voies retenues ont rigor = 'RIGOROUS'."""
    for d in retained_derivations():
        assert d.rigor == "RIGOROUS"


# ---------------------------------------------------------------------------
# 5. Table mesons : K^0, D^0, B^0 ont v_4 ; B_s^0 ne l'a pas (2 composantes)
# ---------------------------------------------------------------------------
def test_meson_table_v4_for_3links():
    """K0, D0, B0 ont un v_4 calcule ; B_s^0 a None (2 composantes)."""
    table = v4_meson_table()
    by_meson = {row["meson"]: row for row in table}
    assert by_meson["K0"]["v4_value"] is not None
    assert by_meson["D0"]["v4_value"] is not None
    assert by_meson["B0"]["v4_value"] is not None
    assert by_meson["Bs0"]["v4_value"] is None  # 2 composantes


def test_meson_table_delta_universal():
    """Tous les mesons predisent le meme delta = -4 ln(d_{1/2})."""
    table = v4_meson_table()
    delta_ref = delta_first_principles()
    for row in table:
        assert row["delta_predicted"] == pytest.approx(delta_ref, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. summary_report : contient les mots cles attendus
# ---------------------------------------------------------------------------
def test_summary_report_keywords():
    rep = summary_report()
    assert "V5.39" in rep
    assert "Vassiliev" in rep
    assert "facteur 4" in rep or "factor 4" in rep
    assert "RIGOREUX" in rep or "RIGOROUS" in rep
    for meson in ("K0", "D0", "B0", "Bs0"):
        assert meson in rep
