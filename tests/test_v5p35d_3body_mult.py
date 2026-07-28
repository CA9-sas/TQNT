"""Tests V5.35d — 3-corps multiplicatif renormalisant.

Vérifie :
1. f(K⁰) ≡ 0 pour les 7 formes (calibration K⁰ préservée)
2. Le produit 3-corps est 1 pour n < 3 (B_s⁰ inchangé)
3. δ = 0 redonne V5.35b exactement
4. K⁰ reste calibré à <1% pour toute (δ, form)
5. ω_χ > 0 et fini pour des δ raisonnables
6. La calibration de δ ne dégrade pas V5.35b (meilleur δ ≤ V5.35b)
7. Au moins une des 7 formes ferme B⁰ sous 0.4 dex
8. Le tableau v5p35d_table est cohérent avec calibrate_delta
9. La symétrie (permutation des composantes du triplet) est respectée
10. Le facteur 3-corps est sensible au signe de δ
11. prod_norm est bien recentrée (f(K⁰) = 0 même avec valeur brute ≠ 0)
12. δ = 0 sur n_comp = 2 (Bs⁰) → facteur trivial 1.0
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p35b_complexity_asymmetry import (
    chiral_inertia_v5p35b,
    delta_m_v5p35b,
)
from theory.v5p35d_3body_multiplicative import (
    FORMS,
    f_triplet,
    three_body_mult_factor,
    chiral_inertia_v5p35d,
    omega_chirality_v5p35d,
    delta_m_v5p35d,
    calibrate_delta,
    calibrate_best_form,
    v5p35d_table,
)


# ---------------------------------------------------------------------------
# 1. f(K⁰) ≡ 0 pour les 7 formes
# ---------------------------------------------------------------------------
def test_f_K0_zero_all_forms():
    """K⁰ a c=(6,6,6) et b=(3,3,3) ⇒ f = 0 pour les 7 formes."""
    for form in FORMS:
        v = f_triplet(6, 6, 6, 3, 3, 3, form)
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"form={form}: f(K⁰) = {v} ≠ 0"
        )


# ---------------------------------------------------------------------------
# 2. three_body_mult_factor = 1 pour n < 3
# ---------------------------------------------------------------------------
def test_factor_unity_for_two_components():
    """Pour n < 3 (B_s⁰ : 2 composantes) le facteur 3-corps = 1."""
    Bs_knots = MESON_OSCILLATION_ASSIGNMENTS["Bs0"]["knots"]
    for form in FORMS:
        for delta in (-5.0, -1.0, 0.0, 1.0, 5.0):
            r = three_body_mult_factor(Bs_knots, delta, form)
            assert r["factor"] == pytest.approx(1.0, abs=1e-15)
            assert r["n_triplets"] == 0


# ---------------------------------------------------------------------------
# 3. δ = 0 redonne V5.35b
# ---------------------------------------------------------------------------
def test_delta_zero_matches_v5p35b():
    """δ = 0 ⇒ I_χ V5.35d ≡ I_χ V5.35b pour toutes les formes/mésons."""
    for name in ("K0", "D0", "B0", "Bs0"):
        knots = MESON_OSCILLATION_ASSIGNMENTS[name]["knots"]
        I_v35b = chiral_inertia_v5p35b(knots, beta=-1.3, form="rel")["I_chi"]
        for form in FORMS:
            I_v35d = chiral_inertia_v5p35d(knots, 0.0, form)["I_chi"]
            assert I_v35d == pytest.approx(I_v35b, rel=1e-10), (
                f"{name}, form={form}, δ=0: {I_v35d} vs {I_v35b}"
            )


# ---------------------------------------------------------------------------
# 4. K⁰ reste calibré à <1% pour toute (δ, form)
# ---------------------------------------------------------------------------
def test_K0_stays_calibrated():
    """Δm V5.35d(K⁰) = Δm PDG(K⁰) à <1% pour tout (δ, form)."""
    cfg = MESON_OSCILLATION_ASSIGNMENTS["K0"]
    for form in FORMS:
        for delta in (-5.0, -1.0, 0.0, 1.0, 5.0):
            res = delta_m_v5p35d(
                cfg["knots"], cfg["mass_pdg"], delta, form
            )
            ratio = res["delta_m_MeV"] / DELTA_M_PDG["K0"]
            assert abs(np.log10(ratio)) < 0.02, (
                f"K⁰ décalibré : form={form}, δ={delta}, "
                f"log_ratio={np.log10(ratio):+.4f}"
            )


# ---------------------------------------------------------------------------
# 5. ω_χ V5.35d > 0 et fini
# ---------------------------------------------------------------------------
def test_omega_positive_finite():
    """ω_χ V5.35d > 0 et fini pour des δ raisonnables."""
    for name in ("K0", "D0", "B0", "Bs0"):
        knots = MESON_OSCILLATION_ASSIGNMENTS[name]["knots"]
        for form in FORMS:
            for delta in (-2.0, -0.5, 0.0, 0.5, 2.0):
                oc = omega_chirality_v5p35d(knots, delta, form)
                assert oc["omega_chi"] > 0
                assert np.isfinite(oc["omega_chi"])
                assert oc["I_chi"] > 0


# ---------------------------------------------------------------------------
# 6. La calibration de δ ne dégrade pas V5.35b
# ---------------------------------------------------------------------------
def test_calibration_does_not_degrade_v5p35b():
    """Σ|log| V5.35d (meilleure forme) ≤ Σ|log| V5.35b (0.5633)
    car δ=0 est dans l'espace de recherche."""
    baseline = 0.5633
    tol = 0.01
    out = calibrate_best_form(delta_range=(-10.0, 50.0), n_grid=2001)
    best = out["best_form"]
    assert out[best]["sum_abs_log"] <= baseline + tol, (
        f"V5.35d ({best}) Σ|log|={out[best]['sum_abs_log']:.4f} > "
        f"V5.35b ({baseline}) : la calibration n'a pas pu battre δ=0"
    )


# ---------------------------------------------------------------------------
# 7. Au moins une forme ferme B⁰ sous 0.4 dex
# ---------------------------------------------------------------------------
def test_at_least_one_form_closes_B():
    """Au moins une des 7 formes ferme Σ|log| < 0.4 dex (cible V5.35d)."""
    out = calibrate_best_form(delta_range=(-10.0, 50.0), n_grid=2001)
    best = out["best_form"]
    s = out[best]["sum_abs_log"]
    log_B = out[best]["per_meson"]["B0"]["log_ratio"]
    assert s < 0.4, (
        f"Aucune forme ne ferme : meilleure ({best}) Σ|log|={s:.4f}"
    )
    assert abs(log_B) < 0.2, (
        f"B⁰ pas fermé : log_r={log_B:+.4f}"
    )


# ---------------------------------------------------------------------------
# 8. v5p35d_table cohérent avec calibrate_delta
# ---------------------------------------------------------------------------
def test_table_matches_calibration():
    """v5p35d_table(δ*, form) donne les mêmes Δm que calibrate_delta."""
    out = calibrate_best_form(delta_range=(-10.0, 50.0), n_grid=2001)
    best = out["best_form"]
    delta_best = out[best]["delta"]
    rows = v5p35d_table(delta_best, form=best)
    by_name = {r["name"]: r for r in rows}
    for name in ("K0", "D0", "B0", "Bs0"):
        dm_tab = by_name[name]["delta_m_v5p35d_MeV"]
        dm_cal = out[best]["per_meson"][name]["delta_m_MeV"]
        assert dm_tab == pytest.approx(dm_cal, rel=1e-10), (
            f"{name}: incohérence table/cal {dm_tab} vs {dm_cal}"
        )


# ---------------------------------------------------------------------------
# 9. Symétrie de permutation
# ---------------------------------------------------------------------------
def test_f_symmetric_under_permutation():
    """f_triplet est symétrique sous permutation des 3 composantes."""
    # B⁰ (c=3,6,7 / b=2,3,4) — toutes distinctes
    perms = [
        (3, 6, 7, 2, 3, 4),
        (6, 3, 7, 3, 2, 4),
        (7, 3, 6, 4, 2, 3),
        (3, 7, 6, 2, 4, 3),
        (7, 6, 3, 4, 3, 2),
        (6, 7, 3, 3, 4, 2),
    ]
    for form in FORMS:
        ref = f_triplet(*perms[0], form)
        for p in perms[1:]:
            v = f_triplet(*p, form)
            assert v == pytest.approx(ref, abs=1e-12), (
                f"form={form}: f({p}) = {v} ≠ f({perms[0]}) = {ref}"
            )


# ---------------------------------------------------------------------------
# 10. Sensibilité au signe de δ pour B⁰
# ---------------------------------------------------------------------------
def test_delta_sign_changes_factor():
    """Le facteur 3-corps de B⁰ change de sens avec le signe de δ
    (sauf pour les formes où f(B⁰) = 0)."""
    B_knots = MESON_OSCILLATION_ASSIGNMENTS["B0"]["knots"]
    for form in FORMS:
        f_p = three_body_mult_factor(B_knots, +0.5, form)["factor"]
        f_n = three_body_mult_factor(B_knots, -0.5, form)["factor"]
        f_0 = three_body_mult_factor(B_knots, 0.0, form)["factor"]
        # Le facteur à δ = 0 vaut 1
        assert f_0 == pytest.approx(1.0, abs=1e-12)
        # Si la forme est active sur B⁰ (f ≠ 0), les facteurs
        # à +δ et −δ doivent être distincts
        # var_c, seq_asym, prod_norm, minmax_mid, irreg_idx,
        # braid_asym, braid_prod ont tous f(B⁰) ≠ 0
        assert f_p != pytest.approx(f_n, abs=1e-12), (
            f"form={form}: facteur(B⁰, +0.5) = facteur(B⁰, -0.5) — "
            f"forme inactive sur B⁰ ?"
        )
        # f_p et f_n encadrent f_0 = 1
        assert (f_p - 1.0) * (f_n - 1.0) < 0, (
            f"form={form}: f(B⁰, +δ)-1 = {f_p-1} et f(B⁰, -δ)-1 = {f_n-1} "
            f"ne sont pas de signes opposés"
        )


# ---------------------------------------------------------------------------
# 11. prod_norm est recentrée
# ---------------------------------------------------------------------------
def test_prod_norm_recenter():
    """Pour « prod_norm », f(K⁰) = 0 par recentrage explicite."""
    v = f_triplet(6, 6, 6, 3, 3, 3, "prod_norm")
    assert v == pytest.approx(0.0, abs=1e-12)
    # Et f(B⁰) doit être ≠ 0 (sinon le levier disparaît)
    v_B = f_triplet(3, 6, 7, 2, 3, 4, "prod_norm")
    assert abs(v_B) > 1e-6


# ---------------------------------------------------------------------------
# 12. Test « seq_asym » spécifique : D⁰ donne f=0 par arithmétique
# ---------------------------------------------------------------------------
def test_seq_asym_D0_zero():
    """D⁰ (c=3,5,7) est arithmétique ⇒ seq_asym(D⁰) = 0,
    donc seq_asym agit *seulement* sur B⁰ (c=3,6,7) — propriété
    qui explique pourquoi cette forme atteint Σ|log| ≈ 0."""
    v_D = f_triplet(3, 5, 7, 2, 2, 3, "seq_asym")
    assert v_D == pytest.approx(0.0, abs=1e-12)
    v_B = f_triplet(3, 6, 7, 2, 3, 4, "seq_asym")
    assert v_B == pytest.approx(1.0 / 3.0, abs=1e-10)
