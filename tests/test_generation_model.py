# tests/test_generation_model.py
"""Tests pour le modèle de générations fermioniques (Gell-Mann–Nishijima)."""
import pytest
from theory.particle_classification import (
    # Existing API — compatibility
    STANDARD_MODEL_PARTICLES,
    KnotParticle,
    knot_mass_spectrum,
    knot_particle_table,
    # Generation model
    IsospinSector,
    FermionState,
    FermionGeneration,
    CompositeParticle,
    BosonCorrespondence,
    GENERATION_1,
    GENERATION_2,
    GENERATION_3,
    FERMION_GENERATIONS,
    PROTON,
    NEUTRON,
    COMPOSITE_PARTICLES,
    BOSON_CORRESPONDENCES,
    generation_table,
    verify_gell_mann_nishijima,
    composite_charge_table,
)


# ===================================================================
# Gell-Mann–Nishijima : Q = I₃ + Y/2
# ===================================================================

class TestGellMannNishijima:
    """Vérification de la formule Q = I₃ + Y/2 pour les 12 fermions."""

    # --- Génération 1 ---

    def test_nu_e_charge(self):
        assert GENERATION_1.neutrino.charge == 0.0

    def test_electron_charge(self):
        assert GENERATION_1.charged_lepton.charge == -1.0

    def test_up_charge(self):
        assert abs(GENERATION_1.up_quark.charge - 2/3) < 1e-10

    def test_down_charge(self):
        assert abs(GENERATION_1.down_quark.charge - (-1/3)) < 1e-10

    # --- Génération 2 ---

    def test_nu_mu_charge(self):
        assert GENERATION_2.neutrino.charge == 0.0

    def test_muon_charge(self):
        assert GENERATION_2.charged_lepton.charge == -1.0

    def test_charm_charge(self):
        assert abs(GENERATION_2.up_quark.charge - 2/3) < 1e-10

    def test_strange_charge(self):
        assert abs(GENERATION_2.down_quark.charge - (-1/3)) < 1e-10

    # --- Génération 3 ---

    def test_nu_tau_charge(self):
        assert GENERATION_3.neutrino.charge == 0.0

    def test_tau_charge(self):
        assert GENERATION_3.charged_lepton.charge == -1.0

    def test_top_charge(self):
        assert abs(GENERATION_3.up_quark.charge - 2/3) < 1e-10

    def test_bottom_charge(self):
        assert abs(GENERATION_3.down_quark.charge - (-1/3)) < 1e-10

    # --- Verification globale ---

    def test_all_gn_match_sm(self):
        """Toutes les charges GN correspondent exactement aux charges SM."""
        results = verify_gell_mann_nishijima()
        for r in results:
            assert r["match"], f"{r['name']}: Q_GN={r['Q_GN']} != Q_SM={r['Q_SM']}"

    def test_verify_gn_returns_12_entries(self):
        results = verify_gell_mann_nishijima()
        assert len(results) == 12


# ===================================================================
# Structure des générations
# ===================================================================

class TestGenerationStructure:
    """Vérification de la structure 3 × 4 fermions."""

    def test_three_generations(self):
        assert len(FERMION_GENERATIONS) == 3

    def test_four_fermions_per_generation(self):
        for gen in FERMION_GENERATIONS:
            assert len(gen.fermions) == 4

    def test_two_leptons_per_generation(self):
        for gen in FERMION_GENERATIONS:
            leptons = [f for f in gen.fermions
                       if f.sector == IsospinSector.LEPTON]
            assert len(leptons) == 2

    def test_two_quarks_per_generation(self):
        for gen in FERMION_GENERATIONS:
            quarks = [f for f in gen.fermions
                      if f.sector == IsospinSector.QUARK]
            assert len(quarks) == 2

    def test_generation_numbers(self):
        for i, gen in enumerate(FERMION_GENERATIONS):
            assert gen.generation == i + 1

    def test_torus_knot_mapping(self):
        """Génération n → nœud torique T(2, 2n+1)."""
        expected = [(1, "3_1", 3), (2, "5_1", 5), (3, "7_1", 7)]
        for gen, (n, knot, s) in zip(FERMION_GENERATIONS, expected):
            assert gen.generation == n
            assert gen.knot == knot
            assert gen.torus_param == s

    def test_all_fermions_spin_half(self):
        for gen in FERMION_GENERATIONS:
            for f in gen.fermions:
                assert f.spin == 0.5

    def test_generation_mass_knot_positive(self):
        for gen in FERMION_GENERATIONS:
            assert gen.mass_knot > 0

    def test_generation_retraction_knot_positive(self):
        for gen in FERMION_GENERATIONS:
            assert gen.retraction_knot > 0


# ===================================================================
# Isospin et hypercharge
# ===================================================================

class TestIsospin:
    """Vérification des doublets d'isospin et hypercharges."""

    def test_lepton_hypercharge(self):
        for gen in FERMION_GENERATIONS:
            assert gen.neutrino.hypercharge == -1.0
            assert gen.charged_lepton.hypercharge == -1.0

    def test_quark_hypercharge(self):
        for gen in FERMION_GENERATIONS:
            assert abs(gen.up_quark.hypercharge - 1/3) < 1e-10
            assert abs(gen.down_quark.hypercharge - 1/3) < 1e-10

    def test_isospin_doublets(self):
        """Chaque doublet a I₃ = +1/2 et -1/2."""
        for gen in FERMION_GENERATIONS:
            # Doublet leptonique
            assert gen.neutrino.isospin_3 == +0.5
            assert gen.charged_lepton.isospin_3 == -0.5
            # Doublet quark
            assert gen.up_quark.isospin_3 == +0.5
            assert gen.down_quark.isospin_3 == -0.5

    def test_isospin_sectors(self):
        for gen in FERMION_GENERATIONS:
            assert gen.neutrino.sector == IsospinSector.LEPTON
            assert gen.charged_lepton.sector == IsospinSector.LEPTON
            assert gen.up_quark.sector == IsospinSector.QUARK
            assert gen.down_quark.sector == IsospinSector.QUARK


# ===================================================================
# Particules composites
# ===================================================================

class TestComposites:
    """Vérification des charges proton et neutron."""

    def test_proton_charge(self):
        assert abs(PROTON.charge - 1.0) < 1e-10

    def test_neutron_charge(self):
        assert abs(NEUTRON.charge) < 1e-10

    def test_proton_baryon_number(self):
        assert PROTON.baryon_number == 1

    def test_neutron_baryon_number(self):
        assert NEUTRON.baryon_number == 1

    def test_proton_constituents(self):
        """Proton = u + u + d."""
        names = [f.name for f in PROTON.constituents]
        assert names.count("up") == 2
        assert names.count("down") == 1

    def test_neutron_constituents(self):
        """Neutron = u + d + d."""
        names = [f.name for f in NEUTRON.constituents]
        assert names.count("up") == 1
        assert names.count("down") == 2

    def test_composite_particles_list(self):
        assert len(COMPOSITE_PARTICLES) == 2

    def test_composite_charge_table(self):
        table = composite_charge_table()
        assert len(table) == 2
        proton_row = table[0]
        assert abs(proton_row["charge"] - 1.0) < 1e-10
        assert proton_row["baryon_number"] == 1


# ===================================================================
# Correspondances bosoniques
# ===================================================================

class TestBosonCorrespondences:
    """Vérification des correspondances nœuds → bosons."""

    def test_four_boson_correspondences(self):
        """P14 : 4 bosons (photon + Z + W+ + W-), le gluon n'est pas une particule."""
        assert len(BOSON_CORRESPONDENCES) == 4

    def test_photon_is_pure_wave(self):
        """P14 : le photon est une onde pure, pas un nœud."""
        photon = next(b for b in BOSON_CORRESPONDENCES
                      if b.boson_name == "photon")
        assert photon.sm_charge == 0.0
        assert photon.is_pure_wave
        assert photon.knot == "none"

    def test_w_plus_charge(self):
        w_plus = next(b for b in BOSON_CORRESPONDENCES
                      if b.boson_name == "W+ boson")
        assert w_plus.sm_charge == 1.0
        assert w_plus.knot == "8_2"
        assert not w_plus.is_mirror

    def test_w_minus_is_mirror(self):
        w_minus = next(b for b in BOSON_CORRESPONDENCES
                       if b.boson_name == "W- boson")
        assert w_minus.sm_charge == -1.0
        assert w_minus.is_mirror

    def test_neutral_bosons_classification(self):
        """P14 : Z⁰ est un nœud 4_1, photon est onde pure, gluon absent."""
        neutral = [b for b in BOSON_CORRESPONDENCES if b.sm_charge == 0]
        z_boson = next(b for b in neutral if b.boson_name == "Z boson")
        assert z_boson.knot == "4_1"
        assert not z_boson.is_pure_wave
        pure_waves = [b for b in neutral if b.is_pure_wave]
        assert len(pure_waves) == 1  # photon only
        assert pure_waves[0].boson_name == "photon"

    def test_no_gluon_in_correspondences(self):
        """P14 : le gluon n'est pas une particule, c'est l'onde résiduelle."""
        names = [b.boson_name for b in BOSON_CORRESPONDENCES]
        assert "gluon" not in names


# ===================================================================
# Propriétés algébriques
# ===================================================================

class TestAlgebraicProperties:
    """Vérification des propriétés algébriques du modèle."""

    def test_charge_sum_per_generation(self):
        """Somme des charges par génération = 0 + (-1) + 2/3 + (-1/3) = -2/3."""
        for gen in FERMION_GENERATIONS:
            total = sum(f.charge for f in gen.fermions)
            assert abs(total - (-2/3)) < 1e-10, \
                f"Gen {gen.generation}: sum={total}, expected -2/3"

    def test_total_charge_all_fermions(self):
        """Somme totale sur 3 générations = 3 × (-2/3) = -2."""
        total = sum(f.charge for gen in FERMION_GENERATIONS
                    for f in gen.fermions)
        assert abs(total - (-2.0)) < 1e-10

    def test_generation_table_has_12_rows(self):
        table = generation_table()
        assert len(table) == 12

    def test_generation_table_charges(self):
        table = generation_table()
        charges = [row["Q_GN"] for row in table]
        expected = [0, -1, 2/3, -1/3] * 3
        for q, e in zip(charges, expected):
            assert abs(q - e) < 1e-10


# ===================================================================
# Compatibilité avec le code existant
# ===================================================================

class TestCompatibility:
    """Vérification que le code existant n'est pas cassé."""

    def test_standard_model_particles_unchanged(self):
        """P14 : 27 particules (gluon retiré, c'est l'onde résiduelle)."""
        assert len(STANDARD_MODEL_PARTICLES) == 27

    def test_knot_particle_still_works(self):
        kp = KnotParticle(name="test", knot="3_1")
        assert kp.spin == 0.5
        assert kp.mass > 0

    def test_knot_mass_spectrum_still_works(self):
        spectrum = knot_mass_spectrum()
        assert len(spectrum) >= 15
        masses = [m for _, m in spectrum]
        assert masses == sorted(masses)

    def test_knot_particle_table_still_20(self):
        table = knot_particle_table()
        assert len(table) == 20
