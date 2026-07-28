"""V5 PDG validation tests — spin + Q_phys across catalogued particles.

Uses the validation script `scripts.V5_validation.spin_charge_67_PDG` to
load the 67 PDG-assigned particles from the V5 catalogue and check that:

  * spin V5 (composition of (b(K_i)-1)/2 braid spins) matches PDG, and
  * Q_phys V5 (= Q_topo(Jones) + Lk) rounds to PDG charge.

Convention: per `docs/TQNT_catalog_V5_diffusion.xlsx` sheet "7. Méthodologie",
Q_phys = Q_topo + Lk where Q_topo is the Jones-derived charge (fractional in
general). The rounded match (within ±0.5) reproduces the integer PDG charge
for all 66/67 entries; the top quark is the only fractional outlier.
"""
from __future__ import annotations

import pytest

from scripts.V5_validation.spin_charge_67_PDG import (
    PDG_REF,
    composite_q_topo,
    load_pdg_table,
    parse_knot_string,
    spin_spectrum,
)


@pytest.fixture(scope="module")
def rows():
    return load_pdg_table()


@pytest.fixture(scope="module")
def by_name(rows):
    return {r.name: r for r in rows}


# --- Coverage --------------------------------------------------------------

def test_67_particles_loaded(rows):
    """All 67 entries from sheet '3. Particules PDG' must be loaded."""
    assert len(rows) == len(PDG_REF) == 67


# --- Spin (100% match expected) -------------------------------------------

def test_spin_all_match(rows):
    """Every catalogued particle's PDG spin lies in its V5 spin spectrum."""
    misses = [r for r in rows if not r.spin_match]
    assert misses == [], (
        f"V5 spin spectrum misses: "
        f"{[(r.name, r.spin_pdg, r.spin_spectrum_v5) for r in misses]}"
    )


@pytest.mark.parametrize("name", [
    "e⁻", "μ⁻", "p", "n", "W±", "Z⁰", "H", "π⁰", "J/ψ", "Ω⁻", "Λ", "Δ",
])
def test_spin_specific(by_name, name):
    """Sample particles must individually have their PDG spin in the V5 spectrum."""
    r = by_name[name]
    assert r.spin_match, (
        f"{name} ({r.cfg}): PDG J={r.spin_pdg} not in V5 spectrum "
        f"{r.spin_spectrum_v5}"
    )


# --- Q_phys (98.5% rounded match expected) --------------------------------

def test_q_phys_rounded_high_fraction(rows):
    """At least 95% of particles must have V5 Q_phys rounding to PDG charge."""
    ok = sum(1 for r in rows if r.q_match_rounded)
    frac = ok / len(rows)
    assert frac >= 0.95, f"Q_phys rounded match {frac:.1%} below 95%"


@pytest.mark.parametrize("name,expected_q", [
    ("e⁻", -1),
    ("μ⁻", -1),
    ("p", +1),
    ("n", 0),
    ("π±", +1),
    ("π⁰", 0),
    ("K±", +1),
    ("Σ⁻", -1),
    ("Σ_c⁺⁺", +2),
    ("Ω⁻", -1),
])
def test_q_phys_rounded_specific(by_name, name, expected_q):
    """Specific particles must produce the correct integer charge after rounding."""
    r = by_name[name]
    assert r.q_phys_rounded == expected_q, (
        f"{name} ({r.cfg}): Q_topo={r.q_topo_v5:+.3f} + Lk={r.lk_inferred} "
        f"= {r.q_phys_v5:+.3f}, rounds to {r.q_phys_rounded:+g}, "
        f"PDG expects {expected_q}"
    )


def test_top_quark_only_fractional_outlier(rows):
    """Documents that the top quark Q=+2/3 is the only fractional outlier."""
    fractional_outliers = [
        r for r in rows
        if abs(r.q_pdg - round(r.q_pdg)) > 0.05 and not r.q_match_exact
    ]
    names = {r.name for r in fractional_outliers}
    # top quark has Q=+2/3; Q_topo+Lk gives a Jones-residual not matching 2/3 exactly
    assert names == {"t"}, (
        f"Expected only the top quark as fractional outlier, found: {names}"
    )


# --- Composite helpers ----------------------------------------------------

def test_parse_knot_string():
    assert parse_knot_string("3_1 + 7_2 + 7_2") == ["3_1", "7_2", "7_2"]
    assert parse_knot_string("6_3 + 8_2 + 8_3") == ["6_3", "8_2", "8_3"]


def test_proton_composition(by_name):
    """Proton = 3_1 + 7_2 + 7_2 → spin 1/2 in spectrum, charge round → +1."""
    r = by_name["p"]
    assert r.cfg.replace(" ", "") == "3_1+7_2+7_2"
    assert 0.5 in r.spin_spectrum_v5
    assert r.q_phys_rounded == +1


def test_higgs_composition(by_name):
    """H = 6_3 + 8_2 + 8_3 → spin 0 in spectrum, charge round → 0."""
    r = by_name["H"]
    assert 0.0 in r.spin_spectrum_v5
    assert r.q_phys_rounded == 0


def test_q_topo_amphicheiral_zero():
    """Amphicheiral knots have Q_topo = 0 (4_1, 6_3, 8_3)."""
    for k in ("4_1", "6_3", "8_3"):
        q = composite_q_topo([k])
        assert abs(q) < 1e-9, f"Q_topo({k}) = {q} should be 0"


def test_spin_spectrum_two_trefoils():
    """Two spin-1/2 trefoils combine to spin {0, 1}."""
    s = spin_spectrum(["3_1", "3_1"])
    assert s == [0.0, 1.0]
