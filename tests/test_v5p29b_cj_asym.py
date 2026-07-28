"""Tests V5.29b — Couplage c_J asymétrique inter-composantes.

Cadre : V5.28 (`theory.topological_chiral_oscillation`) ajoute un terme
de couplage c_J asymétrique au S_min pour tester l'hypothèse que des
composantes à c_J très différents couplent mal.

    S_min/ℏ = κ_S × Σ_i c_i log(1 + c_J,i)             (V5.28)
            + κ_C × Σ_{i<j} f(c_J,i, c_J,j)            (V5.29b)

Ces tests vérifient :

1. l'invariance pour composantes identiques (couplage = 0)
2. la croissance monotone de l'asymétrie en fonction du spread c_J
3. la calibration de κ_C sur D⁰ (fermeture du +1.97 dex)
4. K⁰ reste calibré (préfacteur K⁰ inchangé)
5. les trois formes (linear / quadratic / logratio) donnent toutes la même
   conclusion qualitative : D⁰ fixé, B⁰ rouvert (honnêteté)
6. l'API delta_m_v5p29b est cohérente (Δm = 2 × pref × N × exp(-S))
"""
from __future__ import annotations

import numpy as np
import pytest

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    count_paths_K_to_Kstar,
)
from theory.v5p29_cj_asymmetric import (
    KAPPA_C,
    COUPLING_FORM_DEFAULT,
    asymmetry_coupling,
    action_min_v5p29b,
    delta_m_v5p29b,
    calibrate_kappa_C,
    v5p29b_oscillation_table,
    _reset_prefactor_cache,
)


# --------------------------------------------------------------------------
# 1. Couplage = 0 pour composantes identiques (K⁰)
# --------------------------------------------------------------------------
def test_asymmetry_zero_for_identical_components():
    """K⁰ (6_1 × 3) : composantes identiques → asymétrie c_J nulle."""
    coup = asymmetry_coupling(["6_1", "6_1", "6_1"], kappa_C=1.0,
                              form="linear")
    assert coup["asymmetry_raw"] == 0.0
    assert coup["delta_S"] == 0.0


def test_asymmetry_zero_single_component():
    """Un seul knot : pas de paire → asymétrie = 0."""
    coup = asymmetry_coupling(["3_1"], kappa_C=1.0, form="linear")
    assert coup["asymmetry_raw"] == 0.0


# --------------------------------------------------------------------------
# 2. Croissance avec le spread des c_J
# --------------------------------------------------------------------------
def test_D0_asymmetry_matches_hand_calculation():
    """D⁰ = 3_1 + 5_1 + 7_2 : c_J = (3.3, 5.5, 8.1).

    |Δc_J| pairs : 2.2 + 4.8 + 2.6 = 9.6.
    """
    coup = asymmetry_coupling(["3_1", "5_1", "7_2"], kappa_C=1.0,
                              form="linear")
    assert coup["asymmetry_raw"] == pytest.approx(9.6, abs=1e-3)


def test_B0_asymmetry_exceeds_D0():
    """B⁰ (3_1+6_2+7_7, c_J = 3.3,7.1,9.1) a une asymétrie LINÉAIRE
    supérieure à D⁰ (3.3,5.5,8.1) : 11.6 vs 9.6.

    C'est la propriété structurelle qui fait que V5.29b ne peut pas
    sauver D⁰ sans casser B⁰.
    """
    cD = asymmetry_coupling(["3_1", "5_1", "7_2"], kappa_C=1.0,
                            form="linear")
    cB = asymmetry_coupling(["3_1", "6_2", "7_7"], kappa_C=1.0,
                            form="linear")
    assert cB["asymmetry_raw"] > cD["asymmetry_raw"]


def test_asymmetry_monotone_with_spread():
    """Couplage linéaire monotone par rapport au spread.

    On compare (3_1, 7_7) avec spread c_J ≈ 9.1-3.3 = 5.8 versus
    (6_3, 7_6) avec spread c_J ≈ 8.9-7.3 = 1.6.
    """
    wide = asymmetry_coupling(["3_1", "7_7"], kappa_C=1.0, form="linear")
    narrow = asymmetry_coupling(["6_3", "7_6"], kappa_C=1.0, form="linear")
    assert wide["asymmetry_raw"] > narrow["asymmetry_raw"]


# --------------------------------------------------------------------------
# 3. Calibration de κ_C sur D⁰
# --------------------------------------------------------------------------
def test_calibrate_kappa_C_positive_finite():
    """κ_C calibré sur D⁰ est strictement positif et fini."""
    for form in ["linear", "quadratic", "logratio"]:
        kc = calibrate_kappa_C(form=form)
        assert kc > 0 and np.isfinite(kc), f"κ_C[{form}] = {kc}"


def test_calibrate_kappa_C_fits_D0_to_PDG():
    """Après calibration sur D⁰, Δm V5.29b(D⁰) colle à PDG à mieux que 0.1 dex."""
    for form in ["linear", "quadratic", "logratio"]:
        _reset_prefactor_cache()
        kc = calibrate_kappa_C(form=form)
        _reset_prefactor_cache()
        res = delta_m_v5p29b(["3_1", "5_1", "7_2"], 1864.84,
                             kappa_C=kc, form=form)
        log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["D0"])
        assert abs(log_ratio) < 0.1, (
            f"D⁰ V5.29b[{form}] log_ratio={log_ratio:+.3f} hors fenêtre"
        )


# --------------------------------------------------------------------------
# 4. K⁰ reste calibré (préfacteur inchangé)
# --------------------------------------------------------------------------
def test_K0_calibration_preserved():
    """K⁰ : composantes identiques → ΔS_couplage = 0 → préfacteur
    inchangé → Δm V5.29b = Δm PDG (par construction).
    """
    for form in ["linear", "quadratic", "logratio"]:
        _reset_prefactor_cache()
        kc = calibrate_kappa_C(form=form)
        _reset_prefactor_cache()
        res = delta_m_v5p29b(["6_1", "6_1", "6_1"], 497.611,
                             kappa_C=kc, form=form)
        rel = abs(res["delta_m_MeV"] - DELTA_M_PDG["K0"]) / DELTA_M_PDG["K0"]
        assert rel < 1e-9, f"K⁰ V5.29b[{form}] drift relatif {rel}"


# --------------------------------------------------------------------------
# 5. Honnêteté : V5.29b sauve D⁰ mais casse B⁰
# --------------------------------------------------------------------------
def test_V5p29b_breaks_B0_honest():
    """Avec κ_C calibré sur D⁰, le gap B⁰ se ROUVRE en V5.29b.

    Constat scientifiquement honnête : la pénalité c_J asymétrique pour
    B⁰ est plus grande que pour D⁰, donc B⁰ tombe plus bas.  V5.28
    avait B⁰ à -0.73 dex ; V5.29b le ramène en dessous de -2 dex.
    """
    for form in ["linear", "quadratic", "logratio"]:
        _reset_prefactor_cache()
        kc = calibrate_kappa_C(form=form)
        _reset_prefactor_cache()
        res = delta_m_v5p29b(["3_1", "6_2", "7_7"], 5279.66,
                             kappa_C=kc, form=form)
        log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["B0"])
        assert log_ratio < -2.0, (
            f"V5.29b[{form}] devrait casser B⁰ ; "
            f"log_ratio={log_ratio:+.2f}"
        )


def test_V5p29b_does_not_save_Bs0():
    """Bs⁰ reste sous-prédit en V5.29b (la pénalité c_J ne le sauve pas).

    Asymétrie Bs⁰ = 1.6 (petite) → ΔS_couplage faible → Bs⁰ reste
    proche de sa valeur V5.28 (-3.30 dex).
    """
    _reset_prefactor_cache()
    kc = calibrate_kappa_C(form="linear")
    _reset_prefactor_cache()
    res = delta_m_v5p29b(["6_3", "7_6"], 5366.92, kappa_C=kc, form="linear")
    log_ratio = np.log10(res["delta_m_MeV"] / DELTA_M_PDG["Bs0"])
    assert log_ratio < -3.0, f"Bs⁰ V5.29b log_ratio={log_ratio:+.2f}"


# --------------------------------------------------------------------------
# 6. API et formule canonique Δm = 2 × pref × N × exp(-S)
# --------------------------------------------------------------------------
def test_delta_m_formula_consistency():
    """Δm V5.29b = 2 × prefactor × N_paths × exp(-S_min)."""
    _reset_prefactor_cache()
    kc = calibrate_kappa_C(form="linear")
    _reset_prefactor_cache()
    knots = ["3_1", "5_1", "7_2"]
    res = delta_m_v5p29b(knots, 1864.84, kappa_C=kc, form="linear")
    expected = (
        2.0 * res["prefactor_MeV"] * res["N_paths"] * np.exp(-res["S_min"])
    )
    assert abs(res["delta_m_MeV"] - expected) / expected < 1e-12


def test_action_decomposes_into_additive_and_coupling():
    """S_min V5.29b = S_topo_additive + S_coupling (vérification numérique)."""
    knots = ["3_1", "5_1", "7_2"]
    act = action_min_v5p29b(knots, kappa_C=1.0, form="linear")
    assert act["S_min"] == pytest.approx(
        act["S_topo_additive"] + act["S_coupling"], abs=1e-12
    )


# --------------------------------------------------------------------------
# 7. Forme de couplage : invariances par échange
# --------------------------------------------------------------------------
def test_coupling_symmetric_under_permutation():
    """L'asymétrie pairwise est invariante par permutation des composantes."""
    c1 = asymmetry_coupling(["3_1", "5_1", "7_2"], kappa_C=1.0,
                            form="linear")
    c2 = asymmetry_coupling(["7_2", "3_1", "5_1"], kappa_C=1.0,
                            form="linear")
    assert c1["asymmetry_raw"] == pytest.approx(c2["asymmetry_raw"],
                                                abs=1e-12)


def test_KAPPA_C_default_is_calibrated_value():
    """La constante par défaut KAPPA_C ≈ valeur calibrée sur D⁰ (linéaire)."""
    kc_recompute = calibrate_kappa_C(form="linear")
    assert KAPPA_C == pytest.approx(kc_recompute, rel=1e-6)


# --------------------------------------------------------------------------
# 8. Table de comparaison complète
# --------------------------------------------------------------------------
def test_comparison_table_complete():
    """La table V5.29b contient les 4 mésons."""
    _reset_prefactor_cache()
    rows = v5p29b_oscillation_table(form="linear")
    assert {r["name"] for r in rows} == {"K0", "D0", "B0", "Bs0"}
    for r in rows:
        assert r["log_ratio_v5p29b"] is not None
        assert np.isfinite(r["log_ratio_v5p29b"])
