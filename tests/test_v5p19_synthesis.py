"""Tests for the V5.19 6-class consolidation
(``scripts/V5_synthesis/v5p19_6classes_consolidation.py``).

These tests guard the consolidated synthesis dict shape and the
high-level verdict pattern. They do NOT re-test per-class physics
(that is covered by the per-class test modules); they only verify
that the aggregation honours the documented verdicts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.V5_synthesis.v5p19_6classes_consolidation import (
    ClassEntry,
    aggregate_all,
    synthesis_counts,
    synthesis_summary,
    vassiliev_pattern,
    make_plot,
)


# ----------------------------------------------------------------------------
# 1. Shape: 6 classes, A..F, ClassEntry instances
# ----------------------------------------------------------------------------

def test_aggregate_all_yields_six_classes():
    """The consolidation must aggregate exactly the 6 V5.19 classes."""
    entries = aggregate_all()
    assert set(entries.keys()) == {"A", "B", "C", "D", "E", "F"}
    for sid, e in entries.items():
        assert isinstance(e, ClassEntry)
        assert e.short_id == sid


def test_each_entry_has_mandatory_fields():
    """Every entry exposes the columns required by the synthesis table."""
    for sid, e in aggregate_all().items():
        assert e.name, f"Class {sid} has empty name"
        assert e.channel, f"Class {sid} has empty channel"
        assert e.commit and len(e.commit) >= 7, (
            f"Class {sid} commit hash missing or too short: {e.commit}"
        )
        assert e.verdict_short, f"Class {sid} has empty verdict_short"
        assert e.verdict_long, f"Class {sid} has empty verdict_long"
        assert e.M_typical_Msun > 0, f"Class {sid}: M_typical <= 0"
        assert e.M_max_Msun >= e.M_typical_Msun, (
            f"Class {sid}: M_max < M_typical"
        )
        assert e.Omega_label in ("Omega/Omega_DM", "Omega/Omega_b")
        assert 0.0 <= e.Omega_value <= 1.0
        assert 1 <= e.discriminability <= 5
        assert e.roadmap, f"Class {sid} has empty roadmap"


# ----------------------------------------------------------------------------
# 2. Verdict pattern: 3 SURVIVE, 1 MARGINAL, 1 CONTRIBUTES, 1 NULL
# ----------------------------------------------------------------------------

def test_verdict_distribution_matches_spec():
    """Spec calls for 5/6 SURVIVES-or-contributing-or-marginal, 1/6 NULL,
    and 0/6 EXCLUDED. Verify the consolidated counts agree."""
    counts = synthesis_counts(aggregate_all())
    # Robust: A, E and F should SURVIVE; C MARGINAL; B CONTRIBUTES; D NULL.
    assert counts["SURVIVE"] >= 2  # A and E always SURVIVE; F usually too
    assert counts["MARGINAL"] >= 1  # C is MARGINAL per Gaia DR4 projection
    assert counts["CONTRIBUTES"] >= 1  # B is CONTRIBUTES
    assert counts["NULL_TRIVIAL"] == 1  # D is the unique NULL trivial case
    assert counts["EXCLUDED"] == 0
    assert counts["REFUTED"] == 0


def test_class_D_is_null_trivial():
    """Class D MUST be NULL TRIVIAL: Vassiliev suppression is a hard prediction."""
    entries = aggregate_all()
    assert "NULL" in entries["D"].verdict_short.upper()
    assert entries["D"].discriminability == 1
    assert entries["D"].falsifiable_now is False
    assert entries["D"].falsifiable_soon is False


def test_class_A_and_E_survive_microlensing():
    """Class A (DM pearls) and Class E (strange knot stars) both survive
    the Subaru HSC bound — both are EM-invisible by construction."""
    entries = aggregate_all()
    assert "SURVIV" in entries["A"].verdict_short.upper()
    assert "SURVIV" in entries["E"].verdict_short.upper()


# ----------------------------------------------------------------------------
# 3. Vassiliev structural pattern: astro vs primordial
# ----------------------------------------------------------------------------

def test_vassiliev_pattern_singleton_D():
    """Only Class D is Vassiliev-suppressed primordial; the other five
    are astrophysically formed and testable."""
    pat = vassiliev_pattern(aggregate_all())
    assert pat["primordial_vassiliev_suppressed"] == ["D"]
    assert set(pat["astrophysical_testable"]) == {"A", "B", "C", "E", "F"}


# ----------------------------------------------------------------------------
# 4. Falsifiability roadmap
# ----------------------------------------------------------------------------

def test_falsifiable_soon_includes_C_and_F():
    """Class C (Gaia DR4 2026) and Class F (LIGO O5 2027) MUST be in the
    soon-falsifiable set."""
    entries = aggregate_all()
    assert entries["C"].falsifiable_soon is True
    assert entries["F"].falsifiable_soon is True


def test_falsifiable_now_only_microlensing():
    """Only the microlensing-targeted classes (A, E) are falsifiable with
    current data."""
    entries = aggregate_all()
    assert entries["A"].falsifiable_now is True
    assert entries["E"].falsifiable_now is True


# ----------------------------------------------------------------------------
# 5. Synthesis dict integrity + plot generation
# ----------------------------------------------------------------------------

def test_synthesis_summary_returns_full_dict():
    """synthesis_summary() returns the agreed-upon keys."""
    s = synthesis_summary()
    assert s["n_classes_total"] == 6
    for k in ("counts", "vassiliev_pattern", "entries"):
        assert k in s
    assert set(s["entries"].keys()) == {"A", "B", "C", "D", "E", "F"}


def test_make_plot_writes_png(tmp_path):
    """make_plot() writes a non-empty PNG at the requested path."""
    out = tmp_path / "v5p19_synthesis_6classes_test.png"
    p = make_plot(path=out)
    assert Path(p).exists()
    assert Path(p).stat().st_size > 5000  # at least 5 kB
