"""Tests pour V5.6 scoping alternatives Wilson-loop multipath."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.V5_6_re_validation.wilson_loop_multipath_scoping import (
    M_GRID, M_REF, V5P3_BASE_SLOPE, WB_SLOPE,
    framing_envelope, envelope_1overc, n_fusion_paths,
    slope_framing, slope_1overc, slope_multipath,
    slope_of_log_log,
    main,
)
from theory.su2_modular import V5_LEVEL, conformal_weight


def test_slope_of_log_log_powerlaw():
    """Une power-law y = x^p doit renvoyer pente = p."""
    x = np.logspace(0, 3, 50)
    y = x ** 1.7
    s = slope_of_log_log(x, y)
    assert s == pytest.approx(1.7, rel=1e-3)


def test_framing_envelope_at_M_ref():
    """A M_h = M_ref, f_eff = f0, donc envelope = |cos(pi h_j f0)|."""
    j, f0, delta = 1.0, 0.0, 0.5
    env = framing_envelope(np.array([M_REF]), j, f0, delta)
    h_j = conformal_weight(j)
    assert env[0] == pytest.approx(abs(np.cos(np.pi * h_j * f0)), rel=1e-6)


def test_1overc_envelope_at_M_ref():
    """A M_h = M_ref, envelope = 1 (la correction s'annule)."""
    env = envelope_1overc(np.array([M_REF]), alpha_c=5.0)
    assert env[0] == pytest.approx(1.0, abs=1e-10)


def test_n_fusion_paths_monotone():
    """N_paths(j_max) doit etre monotone croissant en j_max."""
    js = [0.5, 1.0, 2.0, 5.0, 10.0]
    Ns = [n_fusion_paths(j) for j in js]
    assert all(Ns[i+1] >= Ns[i] for i in range(len(Ns) - 1))


def test_framing_can_produce_negative_contribution():
    """L'alternative (A) framing DOIT pouvoir produire un slope contribution
    significativement negatif. Sans cela, le scoping est vide."""
    _, contrib = slope_framing(j=5.0, f0=1.0, delta=-0.6)
    assert contrib < -0.2, f"contrib framing = {contrib:+.3f} pas assez negatif"


def test_main_returns_winner():
    """main() doit retourner un gagnant parmi A/B/C."""
    res = main(verbose=False)
    assert res["winner"] in {"A", "B", "C"}
    assert "winner_name" in res
    # Le gagnant doit etre plus proche de WB qu'aucun des autres
    d = res["distances"]
    assert d[res["winner"]] == min(d.values())


def test_v5p3_base_slope_constant():
    """Constante hardcoded doit matcher le re-fit b107195 (~ +0.573)."""
    assert V5P3_BASE_SLOPE == pytest.approx(0.573, abs=0.005)


def test_wb_target_constant():
    """Cible WB hardcoded : -0.10 (Wechsler-Bullock 2002)."""
    assert WB_SLOPE == pytest.approx(-0.10, abs=1e-6)
