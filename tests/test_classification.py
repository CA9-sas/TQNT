# tests/test_classification.py
import numpy as np
from theory.nodal_classification import (
    NodalParticle,
    PARTICLE_CATALOG,
    classify_particle,
    mass_spectrum,
)


def test_nodal_particle_creation():
    p = NodalParticle(name="test", n=1, l=0, m=0)
    assert p.name == "test"
    assert p.n == 1
    assert p.mass > 0


def test_particle_catalog_is_populated():
    assert len(PARTICLE_CATALOG) >= 5


def test_classify_returns_particle():
    p = classify_particle(n=1, l=0, m=0)
    assert p.n == 1
    assert p.l == 0
    assert p.m == 0


def test_charge_signs_in_catalog():
    positives = [p for p in PARTICLE_CATALOG if p.m > 0]
    negatives = [p for p in PARTICLE_CATALOG if p.m < 0]
    neutrals = [p for p in PARTICLE_CATALOG if p.m == 0]
    assert len(positives) > 0
    assert len(negatives) > 0
    assert len(neutrals) > 0
    for p in positives:
        assert p.charge > 0
    for p in negatives:
        assert p.charge < 0
    for p in neutrals:
        assert p.charge == 0.0


def test_mass_spectrum_is_sorted():
    spectrum = mass_spectrum()
    masses = [m for _, m in spectrum]
    assert masses == sorted(masses)


def test_antiparticle_symmetry():
    p_plus = classify_particle(n=1, l=1, m=1)
    p_minus = classify_particle(n=1, l=1, m=-1)
    assert np.isclose(p_plus.mass, p_minus.mass)
    assert np.isclose(p_plus.charge, -p_minus.charge)
