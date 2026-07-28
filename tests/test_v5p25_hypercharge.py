"""V5.25 hypercharge Y_W validation tests.

Cross-checks via `scripts.V5_validation.v5p25_hypercharge_67_PDG`:

  1. Catalogue covers all 67 PDG-assigned particles.
  2. Gell-Mann-Nishijima Q = T_3 + Y/2 holds on every PDG reference entry.
  3. `topological_hypercharge` returns the documented constants Y_lep=-1,
     Y_quark=+1/3 — it is a CONJECTURE that hard-codes SM values; this is
     pinned by the tests so the limitation is not silently broken later.
  4. V5 Y predictions match PDG exactly on un-flavoured sectors (leptons,
     vector/scalar isoscalar mesons, photon, Z, W, nucleons by sum-of-quarks).
  5. Flavoured-hadron deviations are explicitly documented and bounded.
"""
from __future__ import annotations

import pytest

from scripts.V5_validation.v5p25_hypercharge_67_PDG import (
    PDG_HYPERCHARGE,
    gell_mann_nishijima,
    load_pdg_table,
    parse_knot_string,
    v5_hypercharge_composite,
    v5_hypercharge_single_knot,
)
from theory.particle_classification import topological_hypercharge


# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    return load_pdg_table()


@pytest.fixture(scope="module")
def by_name(rows):
    return {r.name: r for r in rows}


def test_67_particles_loaded(rows):
    """All 67 catalogued PDG particles must have a hypercharge entry."""
    assert len(rows) == len(PDG_HYPERCHARGE) == 67


# -----------------------------------------------------------------------------
# Gell-Mann-Nishijima identity Q = T_3 + Y/2 on PDG numbers
# -----------------------------------------------------------------------------

def test_gell_mann_nishijima_all(rows):
    """Q_PDG = T_3 + Y_PDG/2 must hold for every catalogued particle."""
    misses = [r for r in rows if not r.gmn_ok]
    assert misses == [], (
        f"GMN broken on PDG side for "
        f"{[(r.name, r.t3_pdg, r.y_pdg, r.q_pdg, r.q_gmn_check) for r in misses]}"
    )


def test_gell_mann_nishijima_function():
    """Pure-function GMN sanity check."""
    assert gell_mann_nishijima(+0.5, 1/3) == pytest.approx(2/3)   # u_L quark
    assert gell_mann_nishijima(-0.5, 1/3) == pytest.approx(-1/3)  # d_L quark
    assert gell_mann_nishijima(-0.5, -1.0) == pytest.approx(-1.0) # electron_L
    assert gell_mann_nishijima(+0.5, -1.0) == pytest.approx(0.0)  # nu_e_L


# -----------------------------------------------------------------------------
# topological_hypercharge() current behaviour (conjecture — SM constants)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("knot", ["3_1", "5_1", "7_1", "4_1", "6_3", "7_2", "8_3"])
def test_topological_hypercharge_returns_sm_constants(knot):
    """Document that topological_hypercharge plugs in Y_lep=-1, Y_quark=+1/3
    regardless of knot. This is the documented limitation of V5.25.
    """
    info = topological_hypercharge(knot)
    assert info["Y_lepton"] == pytest.approx(-1.0)
    assert info["Y_quark"] == pytest.approx(1/3)


# -----------------------------------------------------------------------------
# V5 Y predictions vs PDG — sectors that should match exactly
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_y", [
    ("e⁻",  -1.0),
    ("μ⁻",  -1.0),
    ("τ⁻",  -1.0),
    ("γ",    0.0),
    ("W±",   0.0),
    ("Z⁰",   0.0),
])
def test_y_v5_match_leptons_and_gauge(by_name, name, expected_y):
    """Leptons and gauge bosons (photon, W, Z) must have Y_V5 = Y_PDG exactly."""
    r = by_name[name]
    assert r.y_v5 == pytest.approx(expected_y), (
        f"{name} ({r.cfg}): Y_V5={r.y_v5} expected {expected_y}"
    )
    assert r.y_match_exact


def test_y_v5_nucleon_baryon_doublet(by_name):
    """Proton and neutron: Y_V5 from 3 knots * Y_quark = +1, matches PDG Y_nucleon=+1."""
    for name in ("p", "n"):
        r = by_name[name]
        assert r.y_v5 == pytest.approx(1.0), (
            f"{name}: Y_V5={r.y_v5} expected +1 (sum 3 * 1/3)"
        )
        assert r.y_pdg == pytest.approx(1.0)
        assert r.y_match_exact


def test_y_v5_pion_isovector(by_name):
    """π± and π⁰: Y_PDG=0 for the isovector pion triplet, V5 also returns 0."""
    for name in ("π⁰", "π±"):
        r = by_name[name]
        assert r.y_v5 == pytest.approx(0.0)
        assert r.y_pdg == pytest.approx(0.0)
        assert r.y_match_exact


# -----------------------------------------------------------------------------
# V5 Y deviations — flavoured hadrons (s, c, b)
# -----------------------------------------------------------------------------

def test_strange_charm_bottom_mesons_outliers(by_name):
    """Mesons carrying NET strangeness/charm/bottom show Y_V5 ≠ Y_PDG because
    topological_hypercharge doesn't track flavour at the knot level. B_s⁰ is an
    accidental match (Y_s + Y_b̄ = -1 + 1 = 0 = V5 default for mesons).
    """
    truly_flavoured = ["K±", "K⁰", "K*(892)",
                       "D⁰", "D±", "D_s±",
                       "B±", "B⁰", "B_c±"]  # B_s⁰ excluded (accidental Y=0)
    deviating = [m for m in truly_flavoured if not by_name[m].y_match_exact]
    assert deviating == truly_flavoured, (
        f"Expected all NET-flavoured mesons to deviate, found exact matches in: "
        f"{set(truly_flavoured) - set(deviating)}"
    )


def test_top_quark_Y_deviation(by_name):
    """Top quark assigned as 3-knot entrelacs (sector=quark_R) — V5 composite
    returns 0 for quark_R sector (not single-knot). Y_PDG=+4/3 missed.
    """
    r = by_name["t"]
    assert r.y_pdg == pytest.approx(4/3)
    # Composite 3-knot with sector quark_R falls through to default 0 branch.
    assert r.y_v5 == pytest.approx(0.0)
    assert not r.y_match_exact


def test_higgs_Y_deviation(by_name):
    """Higgs doublet Y=+1 not reproduced by the bosonic Y=0 default."""
    r = by_name["H"]
    assert r.y_pdg == pytest.approx(1.0)
    assert r.y_v5 == pytest.approx(0.0)
    assert not r.y_match_exact


# -----------------------------------------------------------------------------
# Aggregate fraction — bound on un-flavoured exact matches
# -----------------------------------------------------------------------------

def test_y_v5_exact_match_lower_bound(rows):
    """At least 45% of catalogued particles get Y_V5 = Y_PDG exactly
    (essentially the un-flavoured sector + isovector light mesons + nucleons).
    """
    n = len(rows)
    n_exact = sum(1 for r in rows if r.y_match_exact)
    frac = n_exact / n
    assert frac >= 0.45, (
        f"Y_V5 exact match fraction {frac:.1%} below 45% bound"
    )


def test_gmn_v5_consistency_on_match_set(rows):
    """For particles where Y_V5 matches Y_PDG, V5 also satisfies GMN trivially."""
    for r in rows:
        if r.y_match_exact:
            q_v5 = gell_mann_nishijima(r.t3_pdg, r.y_v5)
            assert q_v5 == pytest.approx(r.q_pdg), (
                f"{r.name}: GMN with Y_V5 gives Q={q_v5}, PDG Q={r.q_pdg}"
            )


# -----------------------------------------------------------------------------
# Composite helpers
# -----------------------------------------------------------------------------

def test_parse_knot_string():
    assert parse_knot_string("3_1 + 7_2 + 7_2") == ["3_1", "7_2", "7_2"]


def test_v5_hypercharge_composite_baryon():
    """3-knot baryon: Y_V5 = 3 * Y_quark = +1.0."""
    y = v5_hypercharge_composite(["3_1", "7_2", "7_2"], "baryon")
    assert y == pytest.approx(1.0)


def test_v5_hypercharge_single_lepton():
    """Single-knot lepton: Y_V5 = -1."""
    y = v5_hypercharge_single_knot("3_1", "lepton_L")
    assert y == pytest.approx(-1.0)
