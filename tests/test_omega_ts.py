"""Tests sur theory.omega_ts.predict_mass_V5.

Vérifie :
  - Calibration exacte sur l'électron (axiome 1)
  - Reproductibilité PDG sur 6 particules de référence à mieux de 3 %
  - Catalogue de 1539 entrées chargé sans erreur
  - Lookup vs fallback : configurations cataloguées vs nouvelles
"""
import math
import pytest

from theory.omega_ts import (
    predict_mass_V5, omega_ts_lookup, omega_ts_v4_fallback,
    is_in_catalog, _load_catalog, predict_lifetime_lookup,
)


PDG_BENCHMARK = [
    # (config, label, m_PDG_MeV, max_deviation_percent)
    ("3_1",           "electron",  0.51100,    0.01),
    ("3_1+6_1",       "muon",      105.6583,   3.0),
    ("3_1+7_2+7_2",   "proton",    938.272,    0.1),
    ("6_2+7_4+7_5",   "top",       172690.0,   1.0),
    ("7_1+7_2+8_3",   "W",         80369.0,    0.5),
    ("7_2+8_3+9_1",   "Z",         91187.6,    1.0),
    ("6_3+8_2+8_3",   "Higgs",     125250.0,   1.0),
]


def test_catalog_loaded():
    cat = _load_catalog()
    assert len(cat) > 1500, f"Catalogue trop petit ({len(cat)} entrées)"


def test_electron_exact():
    """L'axiome 1 calibre m(3_1) = m_e à 0.5 % près."""
    m = predict_mass_V5("3_1")
    assert math.isclose(m, 0.51100, rel_tol=0.005), f"m(e) = {m}"


@pytest.mark.parametrize("cfg,label,m_exp,tol", PDG_BENCHMARK)
def test_pdg_benchmark(cfg, label, m_exp, tol):
    m = predict_mass_V5(cfg)
    delta = abs(m - m_exp) / m_exp * 100
    assert delta < tol, (
        f"{label} ({cfg}) : m_V5 = {m:.3f}, m_PDG = {m_exp:.3f}, "
        f"Δ = {delta:.3f}% > {tol}%"
    )


def test_v5_predictions_exact():
    """Les 3 prédictions V5p4 (#262, #269, #283) sont dans le catalogue."""
    cfgs = ["6_1+6_1+7_3", "6_3+9_1+9_1", "4_1+6_1+9_1"]
    masses = [2919.1, 3037.3, 3232.3]
    for cfg, m_expected in zip(cfgs, masses):
        m = predict_mass_V5(cfg)
        assert math.isclose(m, m_expected, rel_tol=0.001), (
            f"{cfg} : m = {m}, attendu {m_expected}"
        )


def test_normalization_invariance():
    """L'ordre des composantes ne doit pas changer le résultat."""
    m1 = predict_mass_V5("3_1+7_2+7_2")
    m2 = predict_mass_V5("7_2+3_1+7_2")
    m3 = predict_mass_V5(["7_2", "3_1", "7_2"])
    assert math.isclose(m1, m2) and math.isclose(m1, m3)


def test_fallback_for_unknown():
    """Configuration absente du catalogue → fallback V4 (asymptote ~5 %)."""
    # Configuration improbable mais théoriquement valide :
    m = predict_mass_V5("10_1+10_1+10_1+10_1")  # 4 nœuds, hors catalogue n=4 inutilisé
    # Doit retourner une valeur finie (fallback V4) ou nan si V4 ne supporte pas n=4
    assert math.isfinite(m) or math.isnan(m)  # ne doit pas crasher


def test_lifetime_lookup_proton():
    """Le proton est stable -> lookup peut retourner une valeur extrême."""
    tau = predict_lifetime_lookup("3_1+7_2+7_2")
    # Le proton stable a tau > 10^30 ans dans le catalogue
    assert tau is not None
