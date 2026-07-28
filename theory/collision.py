"""Théorie Nodale — Recomposition topologique lors de collisions

Lors d'une collision proton-proton, les entrelacs (9₁+7₄+6₃)×2 se brisent
et l'énergie du champ χ permet la création de nouveaux nœuds. Les nœuds
se recomposent en nouvelles particules selon les lois de conservation :
charge électrique (Gell-Mann-Nishijima), nombre baryonique, et énergie.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from theory.multi_knot import (
    SM_ASSIGNMENTS, ETA, cornell_equilibrium, fibonacci,
    link_construction_wavelength,
)
from theory.chi_field import jones_rho
from theory.postulates import R_0


# --- Charges SM explicites (Gell-Mann-Nishijima) ---

_SM_CHARGES: dict[str, float] = {
    "e-": -1, "mu-": -1, "tau-": -1,
    "u": 2/3, "c": 2/3, "t": 2/3,
    "d": -1/3, "s": -1/3, "b": -1/3,
    "nu_e": 0, "nu_mu": 0, "nu_tau": 0,
    "W+": 1, "Z0": 0, "H": 0,
    "p": 1, "n": 0,
}

# --- Nombres baryoniques ---

_SM_BARYON: dict[str, float] = {
    "e-": 0, "mu-": 0, "tau-": 0,
    "u": 1/3, "c": 1/3, "t": 1/3,
    "d": 1/3, "s": 1/3, "b": 1/3,
    "nu_e": 0, "nu_mu": 0, "nu_tau": 0,
    "W+": 0, "Z0": 0, "H": 0,
    "p": 1, "n": 1,
}

# --- Anti-noms ---

_ANTI_NAMES: dict[str, str] = {
    "e-": "e+", "mu-": "mu+", "tau-": "tau+",
    "u": "ubar", "c": "cbar", "t": "tbar",
    "d": "dbar", "s": "sbar", "b": "bbar",
    "nu_e": "nu_e_bar", "nu_mu": "nu_mu_bar", "nu_tau": "nu_tau_bar",
    "W+": "W-",
    "Z0": "Z0",   # auto-conjugué
    "H": "H",     # auto-conjugué
    "p": "pbar", "n": "nbar",
}

_SELF_CONJUGATE = {"Z0", "H"}

# --- P16 : suppression par nœud de formation ---
LAMBDA_FORMATION = 1.0  # paramètre de suppression topologique par nœud

# --- P17 : suppression par longueur d'onde de construction ---
ALPHA_WAVELENGTH = 2.0  # exposant de suppression λ

# --- Configurations nucléoniques ---

_NUCLEON_CONFIGS = {
    "p": {"knots": ["9_1", "7_4", "6_3"], "k": 18, "mass_sm": 938.3},
    "n": {"knots": ["9_1", "7_4", "6_3"], "k": 18, "mass_sm": 939.6},
}


@dataclass
class ParticleEntry:
    """Entrée du catalogue de particules."""
    name: str
    anti_name: str
    knots: list[str]
    k: int
    mass_mev: float
    charge: float
    baryon_number: float
    is_anti: bool = False


@dataclass
class CollisionChannel:
    """Canal de collision avec produits finaux."""
    products: list[str]
    total_mass: float
    charge: float
    baryon_number: float
    threshold_sqrt_s: float
    n_total_knots: int = 0
    statistical_weight: float = 0.0


def build_catalog() -> dict[str, ParticleEntry]:
    """Catalogue de ~28 particules (13 SM + nucléons + antiparticules).

    Les charges sont celles du Modèle Standard (Gell-Mann-Nishijima).
    Les particules auto-conjuguées (Z⁰, H) ne sont pas dupliquées.
    """
    catalog: dict[str, ParticleEntry] = {}

    # Particules SM depuis SM_ASSIGNMENTS
    for p in SM_ASSIGNMENTS:
        name = p["name"]
        anti = _ANTI_NAMES[name]
        entry = ParticleEntry(
            name=name,
            anti_name=anti,
            knots=list(p["knots"]),
            k=p["k"],
            mass_mev=p["mass_sm"],
            charge=_SM_CHARGES[name],
            baryon_number=_SM_BARYON[name],
        )
        catalog[name] = entry

        if name not in _SELF_CONJUGATE:
            catalog[anti] = ParticleEntry(
                name=anti,
                anti_name=name,
                knots=list(p["knots"]),
                k=p["k"],
                mass_mev=p["mass_sm"],
                charge=-_SM_CHARGES[name],
                baryon_number=-_SM_BARYON[name],
                is_anti=True,
            )

    # Neutrinos (P18 : reliquats ondulatoires, masse = inertie rotationnelle)
    # Masses indicatives (m = ℏ/(2Rc), R non encore déterminé ab initio)
    _NU_MASSES = {"nu_e": 0.0, "nu_mu": 0.0}
    for flavor, m in _NU_MASSES.items():
        anti = _ANTI_NAMES[flavor]
        # knots=[] car le neutrino n'a pas de nœud propre
        catalog[flavor] = ParticleEntry(
            name=flavor, anti_name=anti, knots=[], k=0,
            mass_mev=m, charge=0.0, baryon_number=0.0,
        )
        catalog[anti] = ParticleEntry(
            name=anti, anti_name=flavor, knots=[], k=0,
            mass_mev=m, charge=0.0, baryon_number=0.0, is_anti=True,
        )

    # Nucléons
    for nuc_name, cfg in _NUCLEON_CONFIGS.items():
        anti = _ANTI_NAMES[nuc_name]
        catalog[nuc_name] = ParticleEntry(
            name=nuc_name,
            anti_name=anti,
            knots=list(cfg["knots"]),
            k=cfg["k"],
            mass_mev=cfg["mass_sm"],
            charge=_SM_CHARGES[nuc_name],
            baryon_number=_SM_BARYON[nuc_name],
        )
        catalog[anti] = ParticleEntry(
            name=anti,
            anti_name=nuc_name,
            knots=list(cfg["knots"]),
            k=cfg["k"],
            mass_mev=cfg["mass_sm"],
            charge=-_SM_CHARGES[nuc_name],
            baryon_number=-_SM_BARYON[nuc_name],
            is_anti=True,
        )

    return catalog


def decompose_to_knots(names: list[str]) -> Counter:
    """Pool de nœuds des particules initiales.

    Les antiparticules contribuent des nœuds miroirs K* (suffixe *).
    """
    catalog = build_catalog()
    pool: Counter = Counter()
    for name in names:
        entry = catalog[name]
        for k in entry.knots:
            if entry.is_anti:
                pool[k + "*"] += 1
            else:
                pool[k] += 1
    return pool


def pair_creation_cost(knot_name: str) -> float:
    """Coût de création d'une paire K+K* = 2×η×ρ_J.

    Ce coût est négligeable (~1-7 MeV) ; l'énergie principale va dans F(k).
    """
    clean = knot_name.rstrip("*")
    rho = jones_rho(clean)
    return 2 * ETA * rho


def channel_weight(
    products: list[str],
    sqrt_s: float,
    catalog: dict[str, ParticleEntry] | None = None,
) -> tuple[float, int]:
    """Poids statistique topologique d'un canal (P16).

    W = Φ_n × exp(−λ_f × N_total)

    Φ_n = (√s − Σm_i)^(3n/2 − 5/2)  (espace de phase n-corps simplifié)
    N_total = nombre total de nœuds individuels dans tous les produits

    Parameters
    ----------
    products : noms des particules finales
    sqrt_s : énergie dans le centre de masse (MeV)
    catalog : catalogue de particules (construit si None)

    Returns
    -------
    (weight, n_total_knots)
    """
    if catalog is None:
        catalog = build_catalog()

    n = len(products)
    total_mass = sum(catalog[p].mass_mev for p in products)
    n_total = sum(len(catalog[p].knots) for p in products)

    # Espace de phase n-corps simplifié (fraction adimensionnée)
    available = sqrt_s - total_mass
    if available <= 0 or n < 2:
        return (0.0, n_total)
    fraction = available / sqrt_s  # 0 < fraction < 1
    exponent_phase = 3 * n / 2 - 5 / 2
    phi_n = fraction ** exponent_phase

    # Suppression exponentielle par nombre de nœuds
    weight = phi_n * math.exp(-LAMBDA_FORMATION * n_total)
    return (weight, n_total)


def rank_channels(
    channels: list[CollisionChannel],
    top_n: int = 30,
) -> list[CollisionChannel]:
    """Trie les canaux par poids décroissant et normalise les probabilités.

    Parameters
    ----------
    channels : liste de canaux (avec statistical_weight calculé)
    top_n : nombre de canaux à retourner

    Returns
    -------
    Top N canaux triés par poids décroissant, avec probabilités normalisées.
    """
    total_w = sum(c.statistical_weight for c in channels)
    if total_w > 0:
        for c in channels:
            c.statistical_weight = c.statistical_weight / total_w

    ranked = sorted(channels, key=lambda c: c.statistical_weight, reverse=True)
    return ranked[:top_n]


def collision_wavelength(sqrt_s: float) -> float:
    """Résolution spatiale de la collision.

    λ_coll = R₀ × 2m_p / √s

    Au seuil (√s = 2m_p) : λ_coll = R₀ (échelle du proton).
    Au LHC (14 TeV)       : λ_coll ≈ 0.00134.

    Parameters
    ----------
    sqrt_s : énergie dans le centre de masse (MeV), doit être > 0

    Returns
    -------
    λ_coll en unités modèle
    """
    m_p = 938.3  # MeV
    return R_0 * 2 * m_p / sqrt_s


def particle_lambda_max(
    name: str,
    catalog: dict[str, ParticleEntry] | None = None,
) -> float:
    """λ_max de construction d'une particule (ou antiparticule).

    Récupère la configuration (knots, k) depuis SM_ASSIGNMENTS ou
    _NUCLEON_CONFIGS, puis appelle link_construction_wavelength().
    Les antiparticules ont le même λ_max que leur conjuguée.

    Parameters
    ----------
    name : nom de la particule (e.g. "e-", "pbar", "mu+")
    catalog : catalogue (construit si None)

    Returns
    -------
    λ_max en unités modèle
    """
    if catalog is None:
        catalog = build_catalog()

    entry = catalog[name]
    # Pour une antiparticule, utiliser la config de la particule
    lookup_name = entry.anti_name if entry.is_anti else name

    # Chercher dans SM_ASSIGNMENTS
    for p in SM_ASSIGNMENTS:
        if p["name"] == lookup_name:
            wl = link_construction_wavelength(p["knots"], p["k"])
            return wl["lambda_max"]

    # Chercher dans _NUCLEON_CONFIGS
    if lookup_name in _NUCLEON_CONFIGS:
        cfg = _NUCLEON_CONFIGS[lookup_name]
        wl = link_construction_wavelength(cfg["knots"], cfg["k"])
        return wl["lambda_max"]

    # Neutrinos (P18) : reliquats ondulatoires, pas de contrainte topologique
    _WAVE_REMNANT = {"nu_e", "nu_mu"}
    if lookup_name in _WAVE_REMNANT:
        return float('inf')

    raise KeyError(f"Configuration introuvable pour {lookup_name}")


def channel_wavelength_factor(
    products: list[str],
    sqrt_s: float,
    catalog: dict[str, ParticleEntry] | None = None,
) -> float:
    """Facteur de suppression par longueur d'onde pour un canal.

    S_i = min(1, λ_max_i / λ_coll) ^ ALPHA_WAVELENGTH
    Retourne Π_i S_i.

    Quand λ_coll > λ_max_i, la collision est trop molle pour résoudre
    les croisements topologiques de la particule i → suppression.
    À haute énergie (λ_coll → 0), pas de suppression (S_i → 1).

    Parameters
    ----------
    products : noms des particules finales
    sqrt_s : énergie dans le centre de masse (MeV)
    catalog : catalogue (construit si None)

    Returns
    -------
    Facteur multiplicatif ∈ (0, 1]
    """
    if catalog is None:
        catalog = build_catalog()

    lam_coll = collision_wavelength(sqrt_s)
    factor = 1.0
    for name in products:
        lam_max = particle_lambda_max(name, catalog)
        ratio = min(1.0, lam_max / lam_coll)
        factor *= ratio ** ALPHA_WAVELENGTH
    return factor


def rank_channels_wavelength(
    channels: list[CollisionChannel],
    sqrt_s: float,
    top_n: int = 30,
) -> list[CollisionChannel]:
    """Trie les canaux avec contrainte P16+λ.

    Recalcule chaque poids : w_new = w_old × channel_wavelength_factor.
    Normalise et trie par poids décroissant.

    Parameters
    ----------
    channels : liste de canaux (avec statistical_weight P16 calculé)
    sqrt_s : énergie dans le centre de masse (MeV)
    top_n : nombre de canaux à retourner

    Returns
    -------
    Top N canaux triés par poids P16+λ décroissant, probabilités normalisées.
    """
    import copy
    catalog = build_catalog()
    ranked = []
    for c in channels:
        c2 = copy.copy(c)
        wl_factor = channel_wavelength_factor(c.products, sqrt_s, catalog)
        c2.statistical_weight = c.statistical_weight * wl_factor
        ranked.append(c2)

    total_w = sum(c.statistical_weight for c in ranked)
    if total_w > 0:
        for c in ranked:
            c.statistical_weight = c.statistical_weight / total_w

    ranked.sort(key=lambda c: c.statistical_weight, reverse=True)
    return ranked[:top_n]


def find_channels(
    initial: list[str],
    sqrt_s: float,
    max_products: int = 6,
) -> list[CollisionChannel]:
    """Recherche des canaux de collision accessibles par backtracking.

    Parameters
    ----------
    initial : noms des particules initiales (e.g. ["p", "p"])
    sqrt_s : énergie dans le centre de masse (MeV)
    max_products : nombre maximal de particules finales

    Returns
    -------
    Liste de CollisionChannel triée par seuil croissant.
    """
    catalog = build_catalog()

    Q_target = sum(catalog[n].charge for n in initial)
    B_target = sum(catalog[n].baryon_number for n in initial)

    # Tri par masse (break si budget dépassé)
    particles = sorted(catalog.values(), key=lambda p: (p.mass_mev, p.name))
    n_particles = len(particles)
    masses = [p.mass_mev for p in particles]
    charges = [p.charge for p in particles]
    baryons = [p.baryon_number for p in particles]
    pnames = [p.name for p in particles]

    # Lookup (3*Q, 3*B) → indices pour optimisation dernier niveau
    qb_lookup: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n_particles):
        key = (round(charges[i] * 3), round(baryons[i] * 3))
        qb_lookup[key].append(i)

    channels: list[tuple[int, ...]] = []

    def _backtrack(
        start: int,
        current: list[int],
        mass_sum: float,
        Q_sum: float,
        B_sum: float,
    ) -> None:
        remaining = max_products - len(current)
        Q_def = Q_target - Q_sum
        B_def = B_target - B_sum

        # Canal valide ?
        if len(current) >= 1 and abs(Q_def) < 1e-9 and abs(B_def) < 1e-9:
            channels.append(tuple(current))

        if remaining == 0:
            return

        # Élagage : déficit irréparable
        if Q_def > remaining + 1e-9 or Q_def < -remaining - 1e-9:
            return
        if B_def > remaining + 1e-9 or B_def < -remaining - 1e-9:
            return

        # Dernier niveau : lookup direct
        if remaining == 1:
            q_key = round(Q_def * 3)
            b_key = round(B_def * 3)
            for i in qb_lookup.get((q_key, b_key), []):
                if i >= start and mass_sum + masses[i] <= sqrt_s + 1e-9:
                    current.append(i)
                    channels.append(tuple(current))
                    current.pop()
            return

        for i in range(start, n_particles):
            new_mass = mass_sum + masses[i]
            if new_mass > sqrt_s + 1e-9:
                break
            current.append(i)
            _backtrack(i, current, new_mass,
                       Q_sum + charges[i], B_sum + baryons[i])
            current.pop()

    _backtrack(0, [], 0.0, 0.0, 0.0)

    # Construire les résultats (pas de doublons grâce à l'ordre canonique)
    result = []
    for ch in channels:
        prods = [pnames[i] for i in ch]
        total_m = sum(masses[i] for i in ch)
        Q = sum(charges[i] for i in ch)
        B = sum(baryons[i] for i in ch)
        w, n_knots = channel_weight(prods, sqrt_s, catalog)
        result.append(CollisionChannel(
            products=prods,
            total_mass=total_m,
            charge=Q,
            baryon_number=B,
            threshold_sqrt_s=total_m,
            n_total_knots=n_knots,
            statistical_weight=w,
        ))

    result.sort(key=lambda c: c.threshold_sqrt_s)
    return result
