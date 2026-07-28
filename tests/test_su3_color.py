"""Tests V5.23 — Charge de couleur SU(3) depuis topologie TQNT."""
import pytest

from theory.su3_color_topology import (
    ANTI_COLORS,
    COLORS,
    F_3,
    LAMBDA_QCD_PDG_MEV,
    M_PROTON_MEV,
    assign_color_nucleon,
    color_charge_anti,
    color_count_consistency_check,
    color_from_knot_position,
    is_color_singlet,
    lambda_qcd_estimate,
    lambda_qcd_v5p23,
    required_color_count_from_topology,
)


# ── 1. Comptage des couleurs ──────────────────────────────────────

def test_three_colors_required():
    """V5.23 doit dériver exactement 3 couleurs (ni 2, ni 4)."""
    assert required_color_count_from_topology() == 3


def test_color_labels_distinct():
    """Les 3 étiquettes r/g/b doivent être distinctes."""
    assert len(set(COLORS)) == 3
    assert len(set(ANTI_COLORS)) == 3
    # disjointes
    assert not (set(COLORS) & set(ANTI_COLORS))


# ── 2. Orbite Z_3 ─────────────────────────────────────────────────

def test_z3_orbit_baryon():
    """Les 3 positions d'un baryon doivent donner 3 couleurs distinctes."""
    link = ("3_1", "7_2", "7_2")
    assigned = {color_from_knot_position(link, i) for i in range(3)}
    assert assigned == set(COLORS)


def test_z3_anti_orbit():
    """L'orbite Z_3⁻¹ doit produire les anti-couleurs."""
    link = ("3_1", "7_2", "7_2")
    anti = {color_from_knot_position(link, i, anti=True) for i in range(3)}
    assert anti == set(ANTI_COLORS)


# ── 3. Involution miroir (anti-couleur) ───────────────────────────

def test_anti_involution():
    """color_charge_anti doit être une involution : anti(anti(c)) = c."""
    for c in COLORS + ANTI_COLORS:
        assert color_charge_anti(color_charge_anti(c)) == c


def test_anti_matches_pairs():
    """r ↔ rbar, g ↔ gbar, b ↔ bbar."""
    assert color_charge_anti("r") == "rbar"
    assert color_charge_anti("g") == "gbar"
    assert color_charge_anti("b") == "bbar"


# ── 4. Singulets de couleur ───────────────────────────────────────

def test_nucleon_is_color_singlet():
    """Le nucléon (3₁+7₂+7₂) doit être un singulet de couleur."""
    assign = assign_color_nucleon()
    assert assign.is_singlet
    assert assign.components == ("3_1", "7_2", "7_2")
    assert assign.colors == ("r", "g", "b")


def test_meson_is_singlet():
    """Un méson (2 composantes) doit être un singulet (qq̄)."""
    assert is_color_singlet(("3_1", "3_1"))


def test_tetraquark_not_singlet_by_default():
    """4 composantes ne forment pas un singulet trivial Z_3."""
    assert not is_color_singlet(("3_1", "3_1", "3_1", "3_1"))


def test_invalid_link_size_raises():
    """1 composante (quark libre) doit lever une erreur."""
    with pytest.raises(ValueError):
        color_from_knot_position(("3_1",), 0)


# ── 5. Λ_QCD V5.23 ────────────────────────────────────────────────

def test_lambda_qcd_close_to_pdg():
    """V5.23 doit prédire Λ_QCD à mieux de 5 % de PDG (220 ± 10 MeV)."""
    lam = lambda_qcd_v5p23()
    delta = abs(lam - LAMBDA_QCD_PDG_MEV) / LAMBDA_QCD_PDG_MEV
    assert delta < 0.05, f"Λ_QCD = {lam:.2f} MeV (écart {delta*100:.2f} %)"


def test_lambda_qcd_formula():
    """Λ_QCD = m_p × (1 − f(3)) explicitement."""
    lam = lambda_qcd_estimate(m_proton_mev=M_PROTON_MEV, f3=F_3)
    expected = M_PROTON_MEV * (1.0 - F_3)
    assert abs(lam - expected) < 1e-9


# ── 6. Cohérence diagnostique ─────────────────────────────────────

def test_consistency_check_dict():
    """Le diagnostique de cohérence doit retourner toutes les clés."""
    diag = color_count_consistency_check()
    assert diag["n_colors_required"] == 3
    assert diag["z3_orbit_distinct"] is True
    assert diag["z2_subgroup_trivial"] is True
    assert diag["z4_extension_breaks"] is True
    assert diag["n_components_baryon"] == 3
    assert diag["n_components_meson"] == 2
