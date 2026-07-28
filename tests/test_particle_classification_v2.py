# tests/test_particle_classification_v2.py
import numpy as np
from theory.particle_classification import (
    STANDARD_MODEL_PARTICLES,
    KnotParticle,
    knot_mass_spectrum,
    find_best_knot_mapping,
)


def test_standard_model_has_27_particles():
    """P14 : 27 particules (gluon retiré, c'est l'onde résiduelle)."""
    assert len(STANDARD_MODEL_PARTICLES) == 27


def test_standard_model_has_fermions():
    fermions = [p for p in STANDARD_MODEL_PARTICLES if p["spin"] == 0.5]
    assert len(fermions) == 24


def test_standard_model_has_bosons():
    """P14 : 3 bosons (photon, W, Z). Le gluon n'est pas une particule."""
    bosons = [p for p in STANDARD_MODEL_PARTICLES if p["spin"] == 1.0]
    assert len(bosons) == 3


def test_knot_particle_creation():
    kp = KnotParticle(name="test", knot="3_1")
    assert kp.mass > 0
    assert kp.retraction > 0
    assert kp.knot == "3_1"


def test_knot_mass_spectrum_sorted():
    spectrum = knot_mass_spectrum()
    masses = [m for _, m in spectrum]
    assert masses == sorted(masses)


def test_knot_mass_spectrum_has_entries():
    spectrum = knot_mass_spectrum()
    assert len(spectrum) >= 15


def test_find_best_knot_mapping_returns_mapping():
    mapping = find_best_knot_mapping()
    assert len(mapping) > 0
    for knot_name, candidates in mapping.items():
        assert isinstance(knot_name, str)
        assert isinstance(candidates, list)
