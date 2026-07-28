"""Tests V5.24 — Dérivation des 3 générations fermioniques."""
from __future__ import annotations

import pytest

from theory.generations_topology import (
    K_CS,
    M_E_PDG_MEV,
    M_MU_PDG_MEV,
    M_TAU_PDG_MEV,
    PISTES_V5p24,
    Q_MAX_PLASTICITY,
    TORUS_FAMILY_KNOTS,
    TORUS_FAMILY_Q,
    admissible_torus_orders,
    eigenstate_to_generation_map,
    evaluate_pistes,
    generation_knot,
    mass_ratios_pdg,
    mass_ratios_predicted,
    mass_v4_bare,
    n_generations_predicted,
    v5p24_summary,
)


# ── 1. N_gen = 3 dérivé ─────────────────────────────────────────────

def test_n_generations_predicted_is_3():
    """V5.24 : exactement 3 générations dérivées."""
    assert n_generations_predicted() == 3


def test_admissible_torus_orders_is_3_5_7():
    """V5.24 : q ∈ {3, 5, 7} sous contrainte plasticité P11."""
    assert admissible_torus_orders() == [3, 5, 7]


def test_q_max_plasticity_is_7():
    """V5.24 : q_max = 7 (V5 §16 plasticité)."""
    assert Q_MAX_PLASTICITY == 7


# ── 2. Mapping nœud ↔ génération ───────────────────────────────────

def test_generation_knot_mapping():
    """g=1 ↦ 3_1 ; g=2 ↦ 5_1 ; g=3 ↦ 7_1."""
    assert generation_knot(1) == "3_1"
    assert generation_knot(2) == "5_1"
    assert generation_knot(3) == "7_1"


def test_generation_knot_out_of_range_raises():
    """V5.24 : pas de 4e (ou 0e) génération."""
    with pytest.raises(ValueError):
        generation_knot(4)
    with pytest.raises(ValueError):
        generation_knot(0)


def test_eigenstate_to_generation_map():
    """Inverse complet sur les 3 torus knots."""
    m = eigenstate_to_generation_map()
    assert m == {"3_1": 1, "5_1": 2, "7_1": 3}


# ── 3. Pistes évaluées ──────────────────────────────────────────────

def test_evaluate_pistes_returns_6():
    """V5.24 évalue 6 pistes."""
    pistes = evaluate_pistes()
    assert len(pistes) == 6


def test_p5_is_retenu():
    """Mécanisme retenu = P5 (spectre torique + P11)."""
    p5 = [p for p in PISTES_V5p24 if p.label == "P5"]
    assert len(p5) == 1
    assert p5[0].verdict == "RETENU"


# ── 4. Masses V4 (gen 1 calibrée sur électron) ─────────────────────

def test_mass_g1_matches_electron_within_1pct():
    """m_V4(3_1) doit reproduire m_e à mieux que 1 % (calibration α₀)."""
    m1 = mass_v4_bare(1)
    rel_err = abs(m1 - M_E_PDG_MEV) / M_E_PDG_MEV
    assert rel_err < 0.01, f"m_g1 = {m1} MeV, écart = {rel_err:.4%}"


def test_mass_ordering_g1_lt_g2_lt_g3():
    """Hiérarchie respectée (au moins en ordre)."""
    m1 = mass_v4_bare(1)
    m2 = mass_v4_bare(2)
    m3 = mass_v4_bare(3)
    assert m1 < m2 < m3


# ── 5. Cohérence interne ────────────────────────────────────────────

def test_summary_consistency():
    """v5p24_summary expose toutes les clés requises."""
    s = v5p24_summary()
    for key in ("n_generations_predicted", "torus_orders_admissible",
                "torus_knots", "q_max_plasticity", "K_CS",
                "alpha_EM_inv", "verdict",
                "m_g1_MeV", "m_g2_MeV", "m_g3_MeV"):
        assert key in s, f"Clé manquante : {key}"
    assert s["alpha_EM_inv"] == 137
    assert s["K_CS"] == K_CS


def test_mass_ratios_pdg_match_known_values():
    """Les ratios PDG sont les valeurs canoniques."""
    md = mass_ratios_pdg()
    assert abs(md["m_mu/m_e_PDG"] - 206.77) < 1.0
    assert abs(md["m_tau/m_mu_PDG"] - 16.82) < 0.5


def test_v5p24_mass_predictions_underestimate_hierarchy():
    """Verdict honnête : ratios prédits sous-estimés (gen 2 et 3)."""
    mp = mass_ratios_predicted()
    md = mass_ratios_pdg()
    # ratio prédit (~28.9) << PDG (~206.8) — facteur ~7
    ratio_factor = md["m_mu/m_e_PDG"] / mp["m_g2/m_g1_predicted"]
    assert ratio_factor > 5, (
        f"V5.24 sous-estime m_μ/m_e d'un facteur {ratio_factor:.2f} "
        f"(documenté §3 V5.24, limite asymptotique V4)."
    )
