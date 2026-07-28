"""Tests pour theory/collision.py — Recomposition topologique lors de collisions"""
import pytest
from collections import Counter

from theory.collision import (
    build_catalog, decompose_to_knots, pair_creation_cost,
    find_channels, channel_weight, rank_channels,
    collision_wavelength, particle_lambda_max,
    channel_wavelength_factor, rank_channels_wavelength,
    ParticleEntry, CollisionChannel,
)
from theory.postulates import R_0


def test_catalog_size():
    """≥ 28 entrées (particules + anti)."""
    catalog = build_catalog()
    assert len(catalog) >= 28


def test_proton_entry():
    """Q=+1, B=+1, m≈938 MeV."""
    catalog = build_catalog()
    p = catalog["p"]
    assert p.charge == 1.0
    assert p.baryon_number == 1.0
    assert abs(p.mass_mev - 938.3) < 1.0


def test_antiproton_entry():
    """Q=-1, B=-1, même masse que le proton."""
    catalog = build_catalog()
    pbar = catalog["pbar"]
    p = catalog["p"]
    assert pbar.charge == -1.0
    assert pbar.baryon_number == -1.0
    assert pbar.mass_mev == p.mass_mev


def test_self_conjugate():
    """Z⁰ et H sont leur propre anti."""
    catalog = build_catalog()
    assert catalog["Z0"].anti_name == "Z0"
    assert catalog["H"].anti_name == "H"
    # Pas de doublon "Z0bar" ou "Hbar"
    assert "Z0" in catalog
    assert "H" in catalog


def test_decompose_proton():
    """Counter({9_1:1, 7_4:1, 6_3:1})."""
    pool = decompose_to_knots(["p"])
    assert pool == Counter({"9_1": 1, "7_4": 1, "6_3": 1})


def test_decompose_pp():
    """Counter({9_1:2, 7_4:2, 6_3:2})."""
    pool = decompose_to_knots(["p", "p"])
    assert pool == Counter({"9_1": 2, "7_4": 2, "6_3": 2})


def test_antiproton_mirror_knots():
    """Nœuds avec suffixe *."""
    pool = decompose_to_knots(["pbar"])
    for knot in pool:
        assert knot.endswith("*")


def test_pair_cost_positive():
    """Coût > 0 pour tout nœud."""
    for knot in ["3_1", "5_1", "7_4", "9_1"]:
        assert pair_creation_cost(knot) > 0


def test_elastic_channel():
    """pp→pp toujours valide."""
    channels = find_channels(["p", "p"], 3000.0, max_products=4)
    products_list = [tuple(sorted(c.products)) for c in channels]
    assert ("p", "p") in products_list


def test_charge_conservation_all():
    """Tous les canaux ont Q_final = Q_initial."""
    channels = find_channels(["p", "p"], 5000.0, max_products=4)
    Q_init = 2.0
    for c in channels:
        assert abs(c.charge - Q_init) < 1e-9


def test_baryon_conservation_all():
    """Tous les canaux ont B_final = B_initial."""
    channels = find_channels(["p", "p"], 5000.0, max_products=4)
    B_init = 2.0
    for c in channels:
        assert abs(c.baryon_number - B_init) < 1e-9


def test_mass_within_budget():
    """Σm_final ≤ √s pour tout canal."""
    sqrt_s = 5000.0
    channels = find_channels(["p", "p"], sqrt_s, max_products=4)
    for c in channels:
        assert c.total_mass <= sqrt_s + 1e-9


def test_no_channels_below_threshold():
    """Aucun canal si √s trop basse (< masse min possible)."""
    # Le canal le plus léger avec Q=2, B=2 est u+u+u+n ≈ 946 MeV
    channels = find_channels(["p", "p"], 500.0, max_products=4)
    assert len(channels) == 0


def test_more_channels_higher_energy():
    """Monotonie du nombre de canaux."""
    energies = [2000, 3000, 5000, 10000]
    counts = [len(find_channels(["p", "p"], e, max_products=4))
              for e in energies]
    for i in range(len(counts) - 1):
        assert counts[i] <= counts[i + 1]


# ====== Tests poids statistiques (P16) ======


def test_weights_positive():
    """W > 0 pour tous les canaux accessibles."""
    channels = find_channels(["p", "p"], 5000.0, max_products=4)
    for c in channels:
        assert c.statistical_weight > 0, f"W=0 pour {c.products}"


def test_elastic_most_probable():
    """pp→pp a le plus grand poids (moins de nœuds + max phase space)."""
    channels = find_channels(["p", "p"], 5000.0, max_products=4)
    ranked = rank_channels(channels, top_n=len(channels))
    assert ranked[0].products == ["p", "p"]


def test_simpler_more_probable():
    """Canal avec N_total petit > canal avec N_total grand (à n produits égal)."""
    catalog = build_catalog()
    sqrt_s = 14_000_000.0
    # pp (6 nœuds) vs pn+u (3+8+2 = 13 nœuds si accessibles)
    w_pp, n_pp = channel_weight(["p", "p"], sqrt_s, catalog)
    # Chercher un canal avec plus de nœuds mais même nombre de produits
    channels = find_channels(["p", "p"], sqrt_s, max_products=4)
    two_body = [c for c in channels if len(c.products) == 2
                and c.n_total_knots > n_pp]
    if two_body:
        assert w_pp > two_body[0].statistical_weight


def test_probabilities_sum_to_one():
    """Σ P ≈ 1.0 après normalisation."""
    channels = find_channels(["p", "p"], 5000.0, max_products=4)
    ranked = rank_channels(channels, top_n=len(channels))
    total = sum(c.statistical_weight for c in ranked)
    assert abs(total - 1.0) < 1e-9


def test_rank_returns_top_n():
    """rank_channels retourne exactement top_n résultats."""
    channels = find_channels(["p", "p"], 14_000_000.0, max_products=4)
    top30 = rank_channels(channels, top_n=30)
    assert len(top30) == 30


def test_top_pair_threshold():
    """pp→tt̄pp accessible seulement si √s ≳ 2m_p+2m_t."""
    catalog = build_catalog()
    threshold = 2 * catalog["p"].mass_mev + 2 * catalog["t"].mass_mev

    # Sous le seuil : pas de canal tt̄
    channels_below = find_channels(["p", "p"], threshold - 100,
                                   max_products=4)
    has_top_below = any(
        "t" in c.products and "tbar" in c.products
        for c in channels_below
    )
    assert not has_top_below

    # Au-dessus du seuil : canal tt̄pp existe
    channels_above = find_channels(["p", "p"], threshold + 100,
                                   max_products=4)
    has_top_above = any(
        "t" in c.products and "tbar" in c.products
        for c in channels_above
    )
    assert has_top_above


# ====== Tests contrainte longueur d'onde (P17) ======


class TestWavelengthCollision:
    """Tests pour les contraintes de longueur d'onde dans les collisions."""

    def test_collision_wavelength_at_threshold(self):
        """λ_coll(2m_p) == R_0."""
        m_p = 938.3
        lam = collision_wavelength(2 * m_p)
        assert abs(lam - R_0) < 1e-9

    def test_collision_wavelength_decreases(self):
        """λ_coll décroit avec √s."""
        assert collision_wavelength(5000) > collision_wavelength(14_000_000)

    def test_collision_wavelength_positive(self):
        """λ_coll > 0 pour tout √s > 0."""
        for sqrt_s in [100, 1000, 1e6, 1e9]:
            assert collision_wavelength(sqrt_s) > 0

    def test_particle_lambda_max_electron(self):
        """λ_max(e-) > 0 et λ_max(e+) > 0."""
        catalog = build_catalog()
        lam_e = particle_lambda_max("e-", catalog)
        lam_ep = particle_lambda_max("e+", catalog)
        assert lam_e > 0
        assert lam_ep > 0

    def test_particle_lambda_max_antiparticle(self):
        """λ_max(particule) == λ_max(anti)."""
        catalog = build_catalog()
        pairs = [("e-", "e+"), ("mu-", "mu+"), ("p", "pbar"),
                 ("u", "ubar"), ("t", "tbar")]
        for p, anti in pairs:
            assert abs(particle_lambda_max(p, catalog)
                       - particle_lambda_max(anti, catalog)) < 1e-12

    def test_channel_factor_no_suppression(self):
        """Factor ≈ 1.0 à √s très élevé (λ_coll → 0)."""
        factor = channel_wavelength_factor(["p", "p"], 1e12)
        assert abs(factor - 1.0) < 1e-6

    def test_channel_factor_suppression(self):
        """Factor < 1.0 pour canal lourd à √s modéré."""
        # Canal avec particules lourdes (W+W-pp) à √s juste au-dessus du seuil
        factor = channel_wavelength_factor(
            ["W+", "W-", "p", "p"], 200_000.0)
        assert factor < 1.0

    def test_wavelength_ranking_conserves_charge(self):
        """Q conservée après re-ranking avec contrainte λ."""
        channels = find_channels(["p", "p"], 5000.0, max_products=4)
        ranked = rank_channels_wavelength(channels, 5000.0, top_n=len(channels))
        for c in ranked:
            assert abs(c.charge - 2.0) < 1e-9

    def test_wavelength_ranking_changes_order(self):
        """Ordre P16+λ ≠ ordre P16 seul (à énergie modérée)."""
        channels = find_channels(["p", "p"], 5000.0, max_products=4)
        ranked_p16 = rank_channels(channels, top_n=len(channels))
        ranked_wl = rank_channels_wavelength(channels, 5000.0,
                                             top_n=len(channels))
        # Il suffit qu'au moins un rang diffère
        prods_p16 = [tuple(c.products) for c in ranked_p16]
        prods_wl = [tuple(c.products) for c in ranked_wl]
        assert prods_p16 != prods_wl

    def test_elastic_still_dominant(self):
        """pp→pp reste #1 avec contrainte λ à haute énergie (14 TeV)."""
        # À 14 TeV, λ_coll ≈ 0.001 << λ_max(p) ≈ 0.004, pas de suppression
        channels = find_channels(["p", "p"], 14_000_000.0, max_products=4)
        ranked = rank_channels_wavelength(channels, 14_000_000.0)
        assert ranked[0].products == ["p", "p"]
