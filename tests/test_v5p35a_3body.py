"""Tests V5.35a — Terme 3-corps dans l'inertie chirale.

10 tests pytest couvrant :
1. three_body_term — vide pour n<3
2. three_body_term — clés et invariants pour n=3
3. three_body_term — formule explicite (vérifiée à la main sur D⁰)
4. three_body_term — invariance permutation (K⁰ : 3×6_1)
5. chiral_inertia_v5p35a — γ_3=0 redonne V5.34 F3
6. chiral_inertia_v5p35a — γ_3>0 augmente strictement I_χ (n≥3)
7. delta_m_v5p35a — K⁰ : R_χ = 1 (calibration)
8. delta_m_v5p35a — B_s⁰ : I_3body = 0 (n=2)
9. calibrate_gamma_3 — γ_3* ∈ range et améliore (ou égale) V5.34
10. v5p35a_table — contient tous les mésons, K⁰ log=0
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.jones import braid_index
from theory.knot_catalog import crossing_number
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p33_mutual_EM_oscillation import effective_linking
from theory.v5p34_chiral_inertia import (
    chiral_inertia as chiral_inertia_v5p34,
    _clear_cache as _clear_cache_v5p34,
)
from theory.v5p35a_3body_inertia import (
    _clear_cache,
    calibrate_gamma_3,
    chiral_inertia_v5p35a,
    delta_m_v5p35a,
    omega_chirality_v5p35a,
    three_body_term,
    v5p35a_table,
)


# ---------------------------------------------------------------------------
# 1. three_body_term : retourne 0 pour n<3
# ---------------------------------------------------------------------------
def test_three_body_term_zero_for_n_less_than_3():
    """B_s⁰ (n=2) et un singleton (n=1) → I_3body = 0."""
    res2 = three_body_term(["6_3", "7_6"])
    assert res2["I_3body"] == 0.0
    assert res2["n_comp"] == 2
    assert res2["n_triplets"] == 0
    assert res2["triplets"] == []

    res1 = three_body_term(["3_1"])
    assert res1["I_3body"] == 0.0
    assert res1["n_comp"] == 1
    assert res1["n_triplets"] == 0


# ---------------------------------------------------------------------------
# 2. three_body_term : clés et invariants pour n=3
# ---------------------------------------------------------------------------
def test_three_body_term_keys_and_positivity_for_n3():
    """Pour B⁰ (n=3) le terme est > 0 et a tous les champs attendus."""
    res = three_body_term(["3_1", "6_2", "7_7"])
    for k in ("I_3body", "n_comp", "n_triplets", "triplets"):
        assert k in res
    assert res["n_comp"] == 3
    # 1 triplet : C(3,3) = 1
    assert res["n_triplets"] == 1
    assert res["I_3body"] > 0
    # Le triplet a tous les champs internes
    t = res["triplets"][0]
    for k in ("i", "j", "k", "knots", "K_ijk", "c_term", "lk_term", "contrib"):
        assert k in t


# ---------------------------------------------------------------------------
# 3. three_body_term : formule explicite (D⁰)
# ---------------------------------------------------------------------------
def test_three_body_term_matches_formula_for_D0():
    """Vérifie I_3body = b₁b₂b₃ × √(c₁c₂c₃) × |Lk₁₂·Lk₂₃·Lk₁₃|^{1/3}
    sur D⁰ = 3_1 + 5_1 + 7_2 (1 seul triplet)."""
    knots = ["3_1", "5_1", "7_2"]
    cs = [crossing_number(k) for k in knots]
    bs = [braid_index(k) for k in knots]
    lk01 = effective_linking(knots[0], knots[1])["lk_eff"]
    lk12 = effective_linking(knots[1], knots[2])["lk_eff"]
    lk02 = effective_linking(knots[0], knots[2])["lk_eff"]
    lk_prod = abs(lk01 * lk12 * lk02)
    expected = (
        bs[0] * bs[1] * bs[2]
        * np.sqrt(cs[0] * cs[1] * cs[2])
        * (lk_prod ** (1.0 / 3.0))
    )
    res = three_body_term(knots)
    assert res["I_3body"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. three_body_term : invariance par permutation (K⁰ : 3×6_1)
# ---------------------------------------------------------------------------
def test_three_body_term_permutation_invariant():
    """K⁰ = 3 × 6_1 : permuter les composantes ne change pas I_3body."""
    res_abc = three_body_term(["6_1", "6_1", "6_1"])
    # Avec composantes identiques permuter ne change rien — testons
    # avec des knots distincts pour un vrai test de permutation.
    res_xyz = three_body_term(["3_1", "5_1", "7_2"])
    res_yxz = three_body_term(["5_1", "3_1", "7_2"])
    res_zyx = three_body_term(["7_2", "5_1", "3_1"])
    assert res_xyz["I_3body"] == pytest.approx(res_yxz["I_3body"])
    assert res_xyz["I_3body"] == pytest.approx(res_zyx["I_3body"])
    # K⁰ : 1 triplet, non nul
    assert res_abc["n_triplets"] == 1
    assert res_abc["I_3body"] > 0


# ---------------------------------------------------------------------------
# 5. chiral_inertia_v5p35a : γ_3 = 0 ↔ V5.34 F3
# ---------------------------------------------------------------------------
def test_chiral_inertia_v5p35a_gamma0_equals_v5p34_F3():
    """γ_3 = 0 → V5.35a = V5.34 F3 strictement (additif)."""
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        knots = cfg["knots"]
        v34 = chiral_inertia_v5p34(knots, formula="F3_coupling", kappa=1.0)
        v35a = chiral_inertia_v5p35a(knots, gamma_3=0.0)
        assert v35a["I_chi"] == pytest.approx(v34["I_chi"])
        assert v35a["I_chi_v5p34"] == pytest.approx(v34["I_chi"])


# ---------------------------------------------------------------------------
# 6. chiral_inertia_v5p35a : γ_3 > 0 augmente I_χ (n≥3)
# ---------------------------------------------------------------------------
def test_chiral_inertia_v5p35a_gamma_increases_inertia_for_n3():
    """γ_3 > 0 → I_χ strictement augmenté pour n≥3 ; inchangé pour n=2."""
    # n=3 : B⁰
    base_B = chiral_inertia_v5p35a(["3_1", "6_2", "7_7"], gamma_3=0.0)
    inc_B = chiral_inertia_v5p35a(["3_1", "6_2", "7_7"], gamma_3=1.0)
    assert inc_B["I_chi"] > base_B["I_chi"]
    # n=2 : B_s⁰ — inchangé
    base_Bs = chiral_inertia_v5p35a(["6_3", "7_6"], gamma_3=0.0)
    inc_Bs = chiral_inertia_v5p35a(["6_3", "7_6"], gamma_3=1.0)
    assert inc_Bs["I_chi"] == pytest.approx(base_Bs["I_chi"])


# ---------------------------------------------------------------------------
# 7. delta_m_v5p35a : K⁰ a R_χ = 1 (normalisation)
# ---------------------------------------------------------------------------
def test_delta_m_v5p35a_K0_R_chi_equals_one():
    """K⁰ est la référence de normalisation : R_χ ≡ 1 pour tout γ_3."""
    _clear_cache()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for gamma in (0.0, 0.5, 2.0):
        _clear_cache()
        res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], gamma_3=gamma)
        assert res["R_chi"] == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 8. delta_m_v5p35a : B_s⁰ inchangé par γ_3 (n=2 → I_3body=0)
# ---------------------------------------------------------------------------
def test_delta_m_v5p35a_Bs0_independent_of_gamma_3():
    """B_s⁰ a n=2 → I_3body=0 ; mais Δm DÉPEND de γ_3 via la
    normalisation R_χ qui change avec ω_χ(K⁰). On vérifie plutôt
    que I_χ(B_s) lui-même est invariant en γ_3."""
    _clear_cache()
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    I0 = chiral_inertia_v5p35a(cfg["knots"], gamma_3=0.0)["I_chi"]
    I1 = chiral_inertia_v5p35a(cfg["knots"], gamma_3=1.0)["I_chi"]
    I2 = chiral_inertia_v5p35a(cfg["knots"], gamma_3=3.0)["I_chi"]
    assert I0 == pytest.approx(I1)
    assert I0 == pytest.approx(I2)
    # Et le champ I_3body est bien à zéro
    _clear_cache()
    res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], gamma_3=1.5)
    assert res["I_3body"] == 0.0


# ---------------------------------------------------------------------------
# 9. calibrate_gamma_3 : optimum dans la plage et améliore (ou égale) V5.34
# ---------------------------------------------------------------------------
def test_calibrate_gamma_3_does_not_degrade_v5p34_F3():
    """L'optimum γ_3* doit donner Σ|log| ≤ Σ|log|(V5.34 F3) puisque
    γ_3 = 0 est inclus dans la plage de recherche."""
    # Baseline V5.34 F3 (sans 3-corps) : Σ|log| ≈ 0.820
    _clear_cache_v5p34()
    s_v534 = 0.0
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        # γ_3=0 dans V5.35a redonne V5.34 F3
        _clear_cache()
        res = delta_m_v5p35a(cfg["knots"], cfg["mass_pdg"], gamma_3=0.0)
        s_v534 += abs(np.log10(res["delta_m_MeV"] / DELTA_M_PDG[name]))
    # Calibration
    res = calibrate_gamma_3(gamma_range=(0.0, 5.0), n_grid=2001)
    assert 0.0 <= res["gamma_3"] <= 5.0
    # Doit améliorer ou égaler (à 1e-3 dex près pour la grille)
    assert res["sum_abs_log"] <= s_v534 + 1e-3


# ---------------------------------------------------------------------------
# 10. v5p35a_table : tous les mésons, K⁰ a log=0
# ---------------------------------------------------------------------------
def test_v5p35a_table_complete_and_K0_zero():
    """v5p35a_table doit contenir tous les mésons et K⁰ a log=0
    (calibration via R_χ)."""
    _clear_cache()
    rows = v5p35a_table(gamma_3=0.5)
    names = {r["name"] for r in rows}
    assert names == set(MESON_OSCILLATION_ASSIGNMENTS.keys())
    by_name = {r["name"]: r for r in rows}
    assert by_name["K0"]["log_ratio_v5p35a"] == pytest.approx(0.0, abs=1e-9)
    # Chaque ligne a les bons champs
    for r in rows:
        for k in (
            "delta_m_v5p35a_MeV", "delta_m_v5p30c_MeV", "delta_m_pdg_MeV",
            "log_ratio_v5p35a", "I_chi", "I_chi_v5p34", "I_3body",
            "gamma_3", "n_comp",
        ):
            assert k in r
        assert r["delta_m_v5p35a_MeV"] > 0
        assert r["gamma_3"] == pytest.approx(0.5)
