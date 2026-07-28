# tests/test_topological_properties.py
"""Tests pour la détermination topologique du spin et de la charge."""
import pytest
from theory.jones import (
    JONES_KNOT_NAMES,
    jones_polynomial,
    braid_index,
    topological_spin,
    topological_charge,
    jones_power_center,
    mirror_jones,
)
from theory.particle_classification import KnotParticle, knot_particle_table


# === Braid index ===

def test_braid_index_unknot():
    assert braid_index("0_1") == 1


def test_braid_index_trefoil():
    assert braid_index("3_1") == 2


def test_braid_index_figure_eight():
    assert braid_index("4_1") == 3


def test_braid_index_torus_knots():
    """Les nœuds toriques T(2,s) ont b=2."""
    for name in ["3_1", "5_1", "7_1", "9_1"]:
        assert braid_index(name) == 2, f"{name} should have braid index 2"


def test_braid_index_all_defined():
    """Tous les 20 nœuds ont un indice de tresse défini."""
    for name in JONES_KNOT_NAMES:
        b = braid_index(name)
        assert isinstance(b, int)
        assert b >= 1


def test_braid_index_unknown_raises():
    with pytest.raises(ValueError):
        braid_index("99_99")


# === Topological spin ===

def test_spin_unknot():
    assert topological_spin("0_1") == 0.0


def test_spin_trefoil():
    assert topological_spin("3_1") == 0.5


def test_spin_figure_eight():
    assert topological_spin("4_1") == 1.0


def test_spin_always_non_negative():
    for name in JONES_KNOT_NAMES:
        assert topological_spin(name) >= 0


def test_spin_multiple_of_half():
    """Le spin est toujours un multiple de 1/2."""
    for name in JONES_KNOT_NAMES:
        s = topological_spin(name)
        assert (s * 2) == int(s * 2), f"{name}: spin {s} not a multiple of 1/2"


# === Topological charge ===

def test_charge_amphicheiral_zero():
    """Les nœuds amphichiraux ont Q=0."""
    for name in ["4_1", "6_3", "8_3"]:
        assert topological_charge(name) == 0.0, f"{name} should have Q=0"


def test_charge_6_1():
    assert abs(topological_charge("6_1") - 1/3) < 1e-10


def test_charge_6_2():
    assert abs(topological_charge("6_2") - 2/3) < 1e-10


def test_charge_8_2():
    assert abs(topological_charge("8_2") - 1.0) < 1e-10


def test_charge_cpt_symmetry():
    """Q(K) + Q(K*) = 0 pour tous les nœuds (symétrie CPT)."""
    for name in JONES_KNOT_NAMES:
        poly = jones_polynomial(name)
        mirror = mirror_jones(poly)
        q_k = jones_power_center(poly)
        q_mirror = jones_power_center(mirror)
        assert abs(q_k + q_mirror) < 1e-10, f"{name}: Q(K)+Q(K*)={q_k+q_mirror} != 0"


def test_charge_unknot():
    assert topological_charge("0_1") == 0.0


# === KnotParticle ===

def test_knot_particle_derived_spin():
    kp = KnotParticle(name="test", knot="3_1")
    assert kp.spin == 0.5


def test_knot_particle_derived_charge():
    kp = KnotParticle(name="test", knot="6_1")
    assert abs(kp.charge - 1/3) < 1e-10


def test_knot_particle_mass_positive():
    kp = KnotParticle(name="test", knot="3_1")
    assert kp.mass > 0


def test_knot_particle_retraction_positive():
    kp = KnotParticle(name="test", knot="3_1")
    assert kp.retraction > 0


def test_knot_particle_is_fermion():
    kp = KnotParticle(name="test", knot="3_1")
    assert kp.is_fermion
    assert not kp.is_boson


def test_knot_particle_is_boson():
    kp = KnotParticle(name="test", knot="4_1")
    assert kp.is_boson
    assert not kp.is_fermion


def test_knot_particle_type():
    assert KnotParticle(name="s", knot="0_1").particle_type == "scalar"
    assert KnotParticle(name="f", knot="3_1").particle_type == "fermion"
    assert KnotParticle(name="g", knot="4_1").particle_type == "gauge_boson"
    assert KnotParticle(name="x", knot="6_3").particle_type == "gravitino"


def test_knot_particle_table_has_20():
    table = knot_particle_table()
    assert len(table) == 20


def test_knot_particle_table_all_valid():
    for kp in knot_particle_table():
        assert kp.spin >= 0
        assert kp.mass >= 0
        assert isinstance(kp.charge, float)
