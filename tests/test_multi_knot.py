"""Tests pour le modele algebrique multi-noeuds Fibonacci."""
import numpy as np
import pytest


class TestFibonacci:
    def test_fibonacci_first_values(self):
        from theory.multi_knot import fibonacci
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(3) == 2
        assert fibonacci(4) == 3
        assert fibonacci(5) == 5
        assert fibonacci(6) == 8

    def test_fibonacci_zero_raises(self):
        from theory.multi_knot import fibonacci
        with pytest.raises(ValueError):
            fibonacci(0)

    def test_fibonacci_negative_raises(self):
        from theory.multi_knot import fibonacci
        with pytest.raises(ValueError):
            fibonacci(-1)


class TestStableDistances:
    def test_stable_distances_values(self):
        from theory.multi_knot import stable_distances
        from theory.postulates import R_0
        dists = stable_distances(6)
        assert len(dists) == 6
        assert dists[0] == pytest.approx(R_0 / 1)  # F(1) = 1
        assert dists[1] == pytest.approx(R_0 / 1)  # F(2) = 1
        assert dists[2] == pytest.approx(R_0 / 2)  # F(3) = 2
        assert dists[3] == pytest.approx(R_0 / 3)  # F(4) = 3
        assert dists[4] == pytest.approx(R_0 / 5)  # F(5) = 5
        assert dists[5] == pytest.approx(R_0 / 8)  # F(6) = 8

    def test_stable_distances_decreasing(self):
        from theory.multi_knot import stable_distances
        dists = stable_distances(10)
        # After the first two (both d_0/1), strictly decreasing
        for i in range(2, len(dists)):
            assert dists[i] < dists[i - 1]


class TestMultiKnotLink:
    def test_single_knot_mass(self):
        """Single-knot link mass equals base knot mass + small correction."""
        from theory.multi_knot import MultiKnotLink
        from theory.retraction import retraction_factor, knot_mass
        from theory.jones import jones_polynomial
        from theory.postulates import LAMBDA_LINK, R_0

        poly = jones_polynomial("3_1")
        base_mass = knot_mass(poly)
        rho = retraction_factor(poly)

        link = MultiKnotLink(knot_names=["3_1"], distance=R_0)
        expected = base_mass + LAMBDA_LINK * rho / R_0
        assert link.mass == pytest.approx(expected)

    def test_mass_increases_as_distance_decreases(self):
        """Closer knots = higher mass."""
        from theory.multi_knot import MultiKnotLink
        link_far = MultiKnotLink(knot_names=["3_1", "3_1"], distance=10.0)
        link_near = MultiKnotLink(knot_names=["3_1", "3_1"], distance=2.0)
        assert link_near.mass > link_far.mass

    def test_charge_increases_with_distance(self):
        """Further knots = amplified charge."""
        from theory.multi_knot import MultiKnotLink
        link_near = MultiKnotLink(knot_names=["3_1", "5_1"], distance=1.0)
        link_far = MultiKnotLink(knot_names=["3_1", "5_1"], distance=10.0)
        assert abs(link_far.charge) >= abs(link_near.charge)

    def test_charge_zero_for_amphicheiral_pair(self):
        """Two amphicheiral knots (Q=0 each) have Q_link = 0 at any distance."""
        from theory.multi_knot import MultiKnotLink
        link = MultiKnotLink(knot_names=["4_1", "4_1"], distance=5.0)
        assert link.charge == pytest.approx(0.0)

    def test_inter_knot_distance(self):
        from theory.multi_knot import MultiKnotLink
        link = MultiKnotLink(knot_names=["3_1", "5_1"], distance=3.0)
        assert link.inter_knot_distance == 3.0

    def test_n_components(self):
        from theory.multi_knot import MultiKnotLink
        link = MultiKnotLink(knot_names=["3_1", "5_1", "7_1"], distance=5.0)
        assert link.n_components == 3


class TestLinkMassSpectrum:
    def test_spectrum_returns_entries(self):
        from theory.multi_knot import link_mass_spectrum
        spectrum = link_mass_spectrum()
        assert len(spectrum) > 0

    def test_spectrum_sorted_by_mass(self):
        from theory.multi_knot import link_mass_spectrum
        spectrum = link_mass_spectrum()
        masses = [entry["mass"] for entry in spectrum]
        assert masses == sorted(masses)

    def test_spectrum_contains_single_knots(self):
        """Single-knot entries should appear in the spectrum."""
        from theory.multi_knot import link_mass_spectrum
        spectrum = link_mass_spectrum()
        names = [entry["description"] for entry in spectrum]
        assert any("3_1" in n for n in names)

    def test_spectrum_contains_multi_knots(self):
        """Multi-knot entries should appear in the spectrum."""
        from theory.multi_knot import link_mass_spectrum
        spectrum = link_mass_spectrum()
        assert any(entry["n_components"] > 1 for entry in spectrum)

    def test_spectrum_entry_has_required_keys(self):
        from theory.multi_knot import link_mass_spectrum
        entry = link_mass_spectrum()[0]
        assert "description" in entry
        assert "mass" in entry
        assert "charge" in entry
        assert "n_components" in entry
        assert "distance" in entry


class TestCornellPotential:
    def test_potential_at_equilibrium_matches_formula(self):
        """U(d*) should equal M_base + rho*F(k)/R_0."""
        from theory.multi_knot import cornell_potential, cornell_equilibrium

        eq = cornell_equilibrium(["3_1"], 1)
        U_star = cornell_potential(["3_1"], eq["d_star"], 1)
        assert U_star == pytest.approx(eq["mass_raw"], rel=1e-10)

    def test_equilibrium_distance_is_R0_over_Fk(self):
        """d* must equal R_0 / F(k) for any config and k."""
        from theory.multi_knot import cornell_equilibrium, fibonacci
        from theory.postulates import R_0

        for config, k in [
            (["3_1"], 1),
            (["3_1", "3_1"], 7),
            (["7_1"], 9),
            (["5_1", "5_1"], 15),
            (["5_1"], 32),
        ]:
            eq = cornell_equilibrium(config, k)
            expected_d = R_0 / fibonacci(k)
            assert eq["d_star"] == pytest.approx(expected_d, rel=1e-10), \
                f"config={config}, k={k}"

    def test_potential_has_minimum_at_d_star(self):
        """U(d) should be minimized at d*."""
        from theory.multi_knot import cornell_potential, cornell_equilibrium

        eq = cornell_equilibrium(["3_1", "3_1"], 7)
        d_star = eq["d_star"]

        d_near = np.linspace(d_star * 0.5, d_star * 2.0, 1000)
        U_near = cornell_potential(["3_1", "3_1"], d_near, 7)
        d_min = d_near[np.argmin(U_near)]

        assert d_min == pytest.approx(d_star, rel=1e-2)

    def test_potential_diverges_at_small_d(self):
        """U(d) -> infinity as d -> 0 (compression quantique)."""
        from theory.multi_knot import cornell_potential, cornell_equilibrium

        eq = cornell_equilibrium(["3_1", "3_1"], 7)
        U_tiny = cornell_potential(["3_1", "3_1"], eq["d_star"] * 0.01, 7)
        assert U_tiny > eq["mass_raw"] * 5

    def test_potential_grows_at_large_d(self):
        """U(d) -> infinity as d -> infinity (confinement)."""
        from theory.multi_knot import cornell_potential, cornell_equilibrium

        eq = cornell_equilibrium(["3_1", "3_1"], 7)
        U_far = cornell_potential(["3_1", "3_1"], eq["d_star"] * 100, 7)
        assert U_far > eq["mass_raw"] * 5

    def test_potential_accepts_array(self):
        """cornell_potential should work with numpy arrays."""
        from theory.multi_knot import cornell_potential

        d = np.linspace(1.0, 10.0, 50)
        U = cornell_potential(["3_1"], d, 1)
        assert U.shape == (50,)
        assert np.all(np.isfinite(U))

    def test_mass_raw_recovers_fibonacci_formula(self):
        """mass_raw = M_base + lambda*rho*F(k)/R_0."""
        from theory.multi_knot import cornell_equilibrium, fibonacci
        from theory.jones import jones_polynomial
        from theory.retraction import retraction_factor, knot_mass
        from theory.postulates import R_0, LAMBDA_LINK

        config = ["5_1", "5_1"]
        k = 15
        eq = cornell_equilibrium(config, k)

        rho = sum(retraction_factor(jones_polynomial(n)) for n in config)
        M_base = sum(knot_mass(jones_polynomial(n)) for n in config)
        expected = M_base + LAMBDA_LINK * rho * fibonacci(k) / R_0

        assert eq["mass_raw"] == pytest.approx(expected, rel=1e-10)

    def test_higher_k_gives_higher_mass(self):
        """Larger k -> larger F(k) -> larger mass."""
        from theory.multi_knot import cornell_equilibrium

        eq_low = cornell_equilibrium(["3_1", "3_1"], 5)
        eq_high = cornell_equilibrium(["3_1", "3_1"], 10)
        assert eq_high["mass_raw"] > eq_low["mass_raw"]

    def test_higher_k_gives_smaller_distance(self):
        """Larger k -> smaller equilibrium distance."""
        from theory.multi_knot import cornell_equilibrium

        eq_low = cornell_equilibrium(["3_1", "3_1"], 5)
        eq_high = cornell_equilibrium(["3_1", "3_1"], 10)
        assert eq_high["d_star"] < eq_low["d_star"]

    def test_second_derivative_positive(self):
        """U''(d*) > 0 confirms a stable minimum."""
        from theory.multi_knot import cornell_equilibrium

        eq = cornell_equilibrium(["7_1"], 9)
        # U''(d) = 2A/d^3, always positive
        d2U = 2 * eq["A"] / eq["d_star"]**3
        assert d2U > 0

    def test_equilibrium_dict_has_all_keys(self):
        from theory.multi_knot import cornell_equilibrium

        eq = cornell_equilibrium(["3_1"], 1)
        for key in ["d_star", "mass_raw", "A", "sigma_k",
                     "rho_total", "M_base", "k", "F_k"]:
            assert key in eq, f"Missing key: {key}"


class TestFibonacciMinima:
    def test_find_fibonacci_minima_returns_results(self):
        """find_fibonacci_minima runs and returns a list of dicts."""
        from theory.multi_knot import find_fibonacci_minima
        results = find_fibonacci_minima(
            knot_names=["3_1", "3_1"],
            n_coarse=8,
            n_eigenvalues=3,
            n_distances=8,
        )
        assert len(results) > 0
        assert "distance" in results[0]
        assert "ground_energy" in results[0]

    def test_fibonacci_minima_energy_varies_with_distance(self):
        """Ground state energy should change as distance varies."""
        from theory.multi_knot import find_fibonacci_minima
        results = find_fibonacci_minima(
            knot_names=["3_1", "3_1"],
            n_coarse=8,
            n_eigenvalues=3,
            n_distances=5,
        )
        energies = [r["ground_energy"] for r in results]
        # Not all the same
        assert max(energies) - min(energies) > 1e-6


class TestConstructionWavelength:
    def test_single_knot_matches_component(self):
        """Pour n=1, λ_link == λ de la composante."""
        from theory.multi_knot import link_construction_wavelength
        from theory.knot_catalog import construction_wavelength
        result = link_construction_wavelength(["3_1"], k=1)
        expected = construction_wavelength("3_1")
        assert result["lambda_max"] == pytest.approx(expected)
        assert result["lambda_min"] == pytest.approx(expected)

    def test_link_leq_components(self):
        """λ_max(link) ≤ toutes les λ_max composantes."""
        from theory.multi_knot import link_construction_wavelength
        result = link_construction_wavelength(["3_1", "5_1"], k=5)
        for details in result["component_wavelengths"].values():
            assert result["lambda_max"] <= details["lambda_max"] + 1e-12

    def test_high_k_binding_is_inter_knot(self):
        """k élevé → d* très petit → contrainte inter-nœuds."""
        from theory.multi_knot import link_construction_wavelength
        result = link_construction_wavelength(["3_1", "5_1"], k=25)
        assert result["binding_constraint"] == "inter-knot"

    def test_low_k_binding_is_component(self):
        """k=1 → d*=R_0=10 → contrainte par une composante."""
        from theory.multi_knot import link_construction_wavelength
        result = link_construction_wavelength(["3_1", "5_1"], k=1)
        assert result["binding_constraint"] != "inter-knot"

    def test_component_lambda_max_leq_lambda_min(self):
        """λ_max(K_i) ≤ λ_min(K_i) pour chaque composante."""
        from theory.multi_knot import link_construction_wavelength
        result = link_construction_wavelength(["3_1", "5_1", "7_1"], k=10)
        for details in result["component_wavelengths"].values():
            assert details["lambda_max"] <= details["lambda_min"] + 1e-12

    def test_component_lambda_max_equals_dstar_when_binding(self):
        """En régime inter-nœuds, λ_max(K_i) = d* pour toutes les composantes."""
        from theory.multi_knot import link_construction_wavelength
        result = link_construction_wavelength(["3_1", "5_1"], k=25)
        d_star = result["d_star"]
        for details in result["component_wavelengths"].values():
            assert details["lambda_max"] == pytest.approx(d_star)

    def test_sm_table_13_entries(self):
        """sm_wavelength_table retourne 13 entrées avec λ > 0 et détails composantes."""
        from theory.multi_knot import sm_wavelength_table
        table = sm_wavelength_table()
        assert len(table) == 13
        for row in table:
            assert row["lambda_max"] > 0
            assert row["lambda_min"] >= row["lambda_max"]
            assert "component_wavelengths" in row
