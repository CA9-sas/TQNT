"""Tests V5.37 — splittings radiaux quarkonium depuis Schrödinger χ-field.

Référence : theory/v5p37_radial_splittings.py
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.v5p37_radial_splittings import (
    QUARKONIUM_ASSIGNMENTS,
    QUARKONIUM_FAMILIES,
    V5P37_PAIRS,
    DEFAULT_N_COARSE,
    DEFAULT_EPSILON,
    quarkonium_assignment,
    schrodinger_spectrum_on_knot,
    delta_m_radial,
    v5p37_radial_table,
    verdict_v5p37,
    clear_spectrum_cache,
)


# Paramètres "rapide" pour les tests pytest (mesh plus grossier).
FAST_N_COARSE = 10
FAST_EPSILON = 0.3
FAST_N_CURVE_POINTS = 120


@pytest.fixture(autouse=True)
def _clear_cache():
    """Vide le cache de spectre avant chaque test."""
    clear_spectrum_cache()
    yield
    clear_spectrum_cache()


# ---------------------------------------------------------------------------
# 1. Catalogue + assignations
# ---------------------------------------------------------------------------
def test_quarkonium_catalog_has_required_states():
    """Les quatre états centraux + extensions 3S sont dans le catalogue."""
    required = {"J/psi", "psi(2S)", "Upsilon(1S)", "Upsilon(2S)",
                "psi(3S)", "Upsilon(3S)"}
    assert required.issubset(QUARKONIUM_ASSIGNMENTS.keys())


def test_quarkonium_assignment_matches_v5_catalog():
    """Les assignations matchent docs/TQNT_catalog_V5_diffusion.xlsx onglet 3."""
    expected = {
        "J/psi":       ["3_1", "4_1", "6_3"],
        "psi(2S)":     ["6_1", "8_2"],
        "Upsilon(1S)": ["6_2", "7_1", "7_3"],
        "Upsilon(2S)": ["6_2", "6_2", "8_2"],
    }
    for name, knots in expected.items():
        cfg = quarkonium_assignment(name)
        assert cfg["knots"] == knots, (
            f"{name}: V5 catalogue attendu {knots}, "
            f"trouvé {cfg['knots']}"
        )
        assert cfg["mass_pdg_MeV"] > 0


def test_quarkonium_assignment_unknown_raises():
    """Une particule absente lève KeyError, pas un fallback silencieux."""
    with pytest.raises(KeyError):
        quarkonium_assignment("Upsilon(42S)")


# ---------------------------------------------------------------------------
# 2. Spectre Schrödinger sur lien
# ---------------------------------------------------------------------------
def test_schrodinger_spectrum_single_knot():
    """Le solveur tourne sur un seul nœud et renvoie des eigenvalues finies."""
    res = schrodinger_spectrum_on_knot(
        ["3_1"], n_levels=3,
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert res["n_components"] == 1
    evs = res["eigenvalues"]
    assert len(evs) == 3
    assert np.all(np.isfinite(evs))
    # Doivent être triés croissant
    assert np.all(np.diff(evs) >= -1e-9)


def test_schrodinger_spectrum_multi_knot():
    """Le solveur tourne sur un lien multi-composant (J/psi link)."""
    res = schrodinger_spectrum_on_knot(
        ["3_1", "4_1", "6_3"], n_levels=3,
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert res["n_components"] == 3
    evs = res["eigenvalues"]
    assert len(evs) == 3
    assert np.all(np.isfinite(evs))


def test_schrodinger_spectrum_levels_bound_below():
    """Le potentiel attractif −λρ/d² donne E_1 < 0."""
    res = schrodinger_spectrum_on_knot(
        ["3_1", "4_1", "6_3"], n_levels=2,
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert res["eigenvalues"][0] < 0, (
        f"Etat fondamental attendu négatif, obtenu {res['eigenvalues'][0]}"
    )


# ---------------------------------------------------------------------------
# 3. delta_m_radial
# ---------------------------------------------------------------------------
def test_delta_m_radial_charmonium_psi2S():
    """ψ(2S)/J/ψ : structure de la sortie + delta_m_pdg cohérent."""
    res = delta_m_radial(
        "J/psi", "psi(2S)",
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    # PDG
    assert abs(res["delta_m_pdg_MeV"] - 589.2) < 1.0
    # Liens V5 différents
    assert res["knots_ground"] == ["3_1", "4_1", "6_3"]
    assert res["knots_excited"] == ["6_1", "8_2"]
    assert not res["same_link"]
    # Les deux interp sont calculées et finies
    for key in ("interp_A", "interp_B"):
        assert np.isfinite(res[key]["delta_m_v5p37_MeV"])
        assert res[key]["delta_m_v5p37_MeV"] > 0
        assert np.isfinite(res[key]["log_ratio"])


def test_delta_m_radial_bottomonium_upsilon2S():
    """Υ(2S)/Υ(1S) : structure et lien V5 distincts."""
    res = delta_m_radial(
        "Upsilon(1S)", "Upsilon(2S)",
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert abs(res["delta_m_pdg_MeV"] - 562.96) < 1.0
    assert res["knots_ground"] == ["6_2", "7_1", "7_3"]
    assert res["knots_excited"] == ["6_2", "6_2", "8_2"]
    assert not res["same_link"]


def test_delta_m_radial_3S_uses_same_link():
    """ψ(3S)/ψ(2S) partage le lien V5 6_1+8_2 → same_link=True."""
    res = delta_m_radial(
        "psi(2S)", "psi(3S)",
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert res["same_link"], (
        "ψ(3S) et ψ(2S) partagent le même lien V5 (6_1+8_2)"
    )
    # interp_A et interp_B identiques quand same_link
    assert res["interp_A"]["delta_m_v5p37_MeV"] == pytest.approx(
        res["interp_B"]["delta_m_v5p37_MeV"], rel=1e-9
    )
    # n_excited == 3 pour ψ(3S)
    assert res["interp_A"]["n_excited"] == 3


# ---------------------------------------------------------------------------
# 4. Table V5.37 + verdict
# ---------------------------------------------------------------------------
def test_v5p37_table_covers_all_pairs():
    """v5p37_radial_table renvoie une ligne par paire de V5P37_PAIRS."""
    rows = v5p37_radial_table(
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    assert len(rows) == len(V5P37_PAIRS)
    labels = {r["label"] for r in rows}
    expected_labels = {p[0] for p in V5P37_PAIRS}
    assert labels == expected_labels


def test_verdict_v5p37_well_formed():
    """Le verdict honnête expose les bons champs et un label connu."""
    rows = v5p37_radial_table(
        n_coarse=FAST_N_COARSE, epsilon=FAST_EPSILON,
        n_curve_points=FAST_N_CURVE_POINTS,
    )
    v = verdict_v5p37(rows)
    assert v["verdict"] in {"REPRODUIT", "PARTIEL", "ECHEC"}
    assert v["n_total"] == len(rows)
    assert 0 <= v["n_within_threshold"] <= v["n_total"]
    assert np.isfinite(v["sum_abs_log_best"])
    assert np.isfinite(v["mean_abs_log_best"])
    assert len(v["best_interp_per_row"]) == len(rows)
    assert all(t in {"A", "B", "none"} for t in v["best_interp_per_row"])
