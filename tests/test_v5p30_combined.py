"""Tests V5.30 — Combinaison V5.29a (frustrés) × V5.29c (Kashaev).

Vérifie :
1.  Structure de la formule V5.30 (factorisation N_paths × F_total × exp(-S)
    × exp(κ_K · vol_tot))
2.  Réduction aux limites :
    - κ_K → 0 → V5.30 = V5.29a (V5.29c devient trivial)
    - Pas d'amphichiraux → V5.30 = V5.29c (F_total = 1)
3.  Calibration globale de κ_K (scan χ²_log)
4.  K⁰ reste calibré exactement par construction
5.  Hiérarchie K < D < B⁰ < B_s⁰ reproduite par V5.30
6.  Σ|log_ratio| V5.30 < V5.28, V5.29a, V5.29c (meilleur cumulé)
7.  Ratio Bs/B PDG ≈ 35 : test honnête de l'écart résiduel
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    delta_m_chiral_oscillation,
    delta_m_chiral_oscillation_v29c,
    KAPPA_K_DEFAULT,
)
from theory.v5p29_frustrated_amphichiral import (
    delta_m_v5p29a,
)
from theory.v5p30_combined_oscillation import (
    KAPPA_K_V5p30_DEFAULT,
    matrix_element_v5p30,
    delta_m_v5p30,
    calibrate_kappa_K_global,
    v5p30_table,
)


# --------------------------------------------------------------------------
# 1. Structure de la formule V5.30
# --------------------------------------------------------------------------
def test_matrix_element_factorisation_v5p30():
    """⟨K|H|K*⟩_V5.30 = N_paths^V5.29a × exp(-S_min) × exp(κ_K · vol_tot)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    me = matrix_element_v5p30(cfg["knots"], cfg["mass_pdg"], 0.611)
    expected = me["N_paths"] * np.exp(-me["S_min"]) * me["P_Kashaev"]
    rel = abs(me["matrix_element"] - expected) / expected
    assert rel < 1e-10, f"Factorisation V5.30 cassée : rel = {rel}"


def test_delta_m_v5p30_equals_2_prefactor_matrix_element():
    """Δm V5.30 = 2 × prefactor × matrix_element (cohérence facteur 2)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["B0"]
    res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"])
    expected = 2.0 * res["prefactor_MeV"] * res["matrix_element"]
    rel = abs(res["delta_m_MeV"] - expected) / expected
    assert rel < 1e-12


# --------------------------------------------------------------------------
# 2. Réduction aux limites
# --------------------------------------------------------------------------
def test_v5p30_reduces_to_v5p29a_when_kappa_zero():
    """κ_K = 0 → V5.30(L) = V5.29a(L) ∀L (P_Kashaev devient 1 et la
    renormalisation sur K⁰ est commune)."""
    for name in ("K0", "D0", "B0", "Bs0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        r_v530 = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], 0.0)
        r_v29a = delta_m_v5p29a(cfg["knots"], cfg["mass_pdg"])
        rel = abs(r_v530["delta_m_MeV"] - r_v29a["delta_m_MeV"]) / r_v29a[
            "delta_m_MeV"
        ]
        assert rel < 1e-9, (
            f"V5.30(κ=0)({name}) ≠ V5.29a({name}) ; rel = {rel}"
        )


def test_v5p30_reduces_to_v5p29c_when_no_amphichiral():
    """Sur K⁰, D⁰, B⁰ (aucun amphichiral, F_total = 1), V5.30 = V5.29c
    pour le même κ_K (les deux modèles ne diffèrent que par F_total)."""
    kappa = 0.611
    for name in ("K0", "D0", "B0"):
        cfg = MESON_OSCILLATION_ASSIGNMENTS[name]
        r_v530 = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], kappa)
        r_v29c = delta_m_chiral_oscillation_v29c(
            cfg["knots"], cfg["mass_pdg"], kappa
        )
        rel = abs(r_v530["delta_m_MeV"] - r_v29c["delta_m_MeV"]) / r_v29c[
            "delta_m_MeV"
        ]
        assert rel < 1e-9, (
            f"V5.30({name}) doit égaler V5.29c({name}) pour κ={kappa} "
            f"sans amphichiral ; rel = {rel}"
        )


# --------------------------------------------------------------------------
# 3. Calibration K⁰
# --------------------------------------------------------------------------
def test_K0_calibration_v5p30():
    """K⁰ calibré exactement à Δm PDG pour tout κ_K (préfacteur absorbe)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for kappa in (0.0, 0.5, 0.611, 1.0, 1.5):
        res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"], kappa)
        rel = abs(res["delta_m_MeV"] - DELTA_M_PDG["K0"]) / DELTA_M_PDG["K0"]
        assert rel < 1e-9, f"K0 calibration drift V5.30 à κ={kappa}: {rel}"


def test_F_total_K0_is_one():
    """K⁰ = (6_1)^3 : aucun amphichiral, F_total = 1."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"])
    assert res["F_total"] == 1
    assert res["n_amphichiral"] == 0


def test_F_total_Bs0_includes_63_factor():
    """B_s⁰ = 6_3 + 7_6 : 6_3 amphichiral → F_total ≥ 6! × 4 = 2880."""
    from math import factorial
    cfg = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]
    res = delta_m_v5p30(cfg["knots"], cfg["mass_pdg"])
    assert res["n_amphichiral"] == 1
    assert res["F_total"] == factorial(6) * 4  # 2880


# --------------------------------------------------------------------------
# 4. Calibration globale κ_K
# --------------------------------------------------------------------------
def test_calibrate_kappa_K_global_finds_finite_minimum():
    """Le scan global trouve un κ_K fini dans la fenêtre [-1, 2]."""
    cal = calibrate_kappa_K_global(
        kappa_range=(-1.0, 2.0), n_grid=401
    )
    assert -1.0 < cal["kappa_K"] < 2.0
    assert np.isfinite(cal["chi2_log"])


def test_calibrate_kappa_K_global_positive():
    """κ_K optimal doit être positif (P_Kashaev amplifie B, B_s)."""
    cal = calibrate_kappa_K_global(
        kappa_range=(-1.0, 2.0), n_grid=401
    )
    assert cal["kappa_K"] > 0.0


def test_default_kappa_K_close_to_calibration():
    """KAPPA_K_V5p30_DEFAULT doit être proche du résultat du scan global."""
    cal = calibrate_kappa_K_global(
        kappa_range=(-1.0, 2.0), n_grid=801
    )
    assert abs(KAPPA_K_V5p30_DEFAULT - cal["kappa_K"]) < 0.05, (
        f"KAPPA_K_V5p30_DEFAULT={KAPPA_K_V5p30_DEFAULT} doit être proche "
        f"du κ_K calibré {cal['kappa_K']:.4f}"
    )


# --------------------------------------------------------------------------
# 5. Hiérarchie K < D < B < Bs reproduite
# --------------------------------------------------------------------------
def test_v5p30_reproduces_hierarchy():
    """V5.30 doit reproduire K < D < B⁰ < B_s⁰ (vrai dans PDG)."""
    rows = v5p30_table()
    by_name = {r["name"]: r["delta_m_v5p30_MeV"] for r in rows}
    assert by_name["K0"] < by_name["D0"] < by_name["B0"] < by_name["Bs0"], (
        f"V5.30 ne reproduit pas la hiérarchie K<D<B<Bs : "
        f"K={by_name['K0']:.2e}, D={by_name['D0']:.2e}, "
        f"B={by_name['B0']:.2e}, Bs={by_name['Bs0']:.2e}"
    )


# --------------------------------------------------------------------------
# 6. V5.30 améliore Σ|log_ratio|
# --------------------------------------------------------------------------
def test_v5p30_global_improves_sum_log_ratio_vs_v528():
    """Σ|log V5.30| < Σ|log V5.28| (amélioration globale)."""
    rows = v5p30_table()
    s_528 = sum(abs(r["log_ratio_v528"]) for r in rows)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows)
    assert s_530 < s_528, (
        f"V5.30 doit améliorer Σ|log| vs V5.28 ; "
        f"V5.28={s_528:.3f}, V5.30={s_530:.3f}"
    )


def test_v5p30_global_improves_sum_log_ratio_vs_v29a():
    """V5.30 doit aussi améliorer V5.29a (ajout du Kashaev positif)."""
    rows = v5p30_table()
    s_29a = sum(abs(r["log_ratio_v529a"]) for r in rows)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows)
    assert s_530 < s_29a, (
        f"V5.30 doit améliorer Σ|log| vs V5.29a ; "
        f"V5.29a={s_29a:.3f}, V5.30={s_530:.3f}"
    )


def test_v5p30_global_improves_sum_log_ratio_vs_v29c():
    """V5.30 doit aussi améliorer V5.29c (ajout des frustrés amphi)."""
    rows = v5p30_table()
    s_29c = sum(abs(r["log_ratio_v529c"]) for r in rows)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows)
    assert s_530 < s_29c, (
        f"V5.30 doit améliorer Σ|log| vs V5.29c ; "
        f"V5.29c={s_29c:.3f}, V5.30={s_530:.3f}"
    )


# --------------------------------------------------------------------------
# 7. Honnêteté : ratio Bs/B reste loin de PDG
# --------------------------------------------------------------------------
def test_Bs_over_B_ratio_exceeds_pdg():
    """V5.30 sur-amplifie B_s⁰ (F=2880 + κ_K·vol(6_3+7_6) grand),
    donc le ratio B_s/B excède le PDG. Test honnête de cette limite."""
    rows = v5p30_table()
    by_name = {r["name"]: r["delta_m_v5p30_MeV"] for r in rows}
    ratio_530 = by_name["Bs0"] / by_name["B0"]
    ratio_pdg = DELTA_M_PDG["Bs0"] / DELTA_M_PDG["B0"]
    # Ratio doit dépasser PDG (signature de l'overamplification connue)
    assert ratio_530 > ratio_pdg, (
        f"V5.30 doit sur-amplifier Bs/B (ratio > PDG) ; "
        f"V5.30={ratio_530:.2f}, PDG={ratio_pdg:.2f}"
    )


def test_v5p30_table_complete():
    """Table V5.30 contient les 4 mésons + tous les champs requis."""
    rows = v5p30_table()
    names = {r["name"] for r in rows}
    assert names == {"K0", "D0", "B0", "Bs0"}
    required = {
        "delta_m_v5p30_MeV", "delta_m_v529a_MeV",
        "delta_m_v529c_MeV", "delta_m_v528_MeV", "delta_m_pdg_MeV",
        "log_ratio_v5p30", "log_ratio_v529a", "log_ratio_v529c",
        "log_ratio_v528",
        "F_total", "vol_tot", "P_Kashaev", "N_paths", "kappa_K",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"Champs manquants pour {r['name']} : {missing}"
        for k in ("log_ratio_v5p30", "log_ratio_v528"):
            assert r[k] is not None and np.isfinite(r[k])
