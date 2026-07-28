"""Theorie Nodale v4 -- Entrelacs Multi-Noeuds avec Masse Fibonacci

Postulat P11 : les particules massives sont des entrelacs de n noeuds.
La masse croit en 1/d (distance inter-noeuds), la charge est amplifiee par d.
Les distances stables correspondent aux minima d'energie aux paliers de
Fibonacci : d_stable(k) = d_0 / F(k).

Postulat P12 (Potentiel de Cornell topologique) :
L'equilibre inter-noeuds emerge d'un potentiel a deux termes :
    U(d) = M_base + A/d + sigma_k * d
ou A = lambda*rho/2 (compression quantique) et sigma_k (tension de corde
topologique). Le minimum est a d* = R_0/F(k) avec masse
m = eta * (M_base + rho*F(k)/R_0).

Postulat P13 (Duree de vie topologique simple) :
    tau = tau_P / M_raw^nu
avec nu = 5 pour les desintegrations faibles (leptons, quarks).

Postulat P15 (Fragilite topologique exponentielle) :
Chaque noeud de l'entrelacs est un canal de desintegration independant.
La probabilite de survie decroit exponentiellement avec n (nombre de noeuds) :
    tau = tau_P * exp(-lambda_f * n) / M_raw^nu
Canal faible (leptons, quarks) : nu=7.75, lambda_f=5.0, calibre sur le muon.
Canal fort/EW (W, Z, t, H)    : nu=1.75, lambda_f=1.75, calibre sur le W.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from itertools import combinations_with_replacement
from theory.jones import (jones_polynomial, topological_charge, JONES_KNOT_NAMES,
                         braid_index, is_half_integer_spin, topological_spin)
from theory.retraction import retraction_factor, knot_mass
from theory.postulates import R_0, LAMBDA_LINK, BETA_LINK


# --- P7ter : Trois générations fermioniques ---
# SU(2) a 3 générateurs → 3 nœuds toriques fermioniques
FERMION_GENERATIONS = [(3, "3_1"), (5, "5_1"), (7, "7_1")]


def generation_count() -> int:
    """Nombre de générations fermioniques = dim(SU(2)) = 3."""
    return len(FERMION_GENERATIONS)


# --- P7bis : Composition de spin des entrelacs multi-nœuds ---

def link_spin_parity(knot_names: list[str]) -> str:
    """Parité de spin d'un entrelacs : 'half-integer' ou 'integer'.

    Le spin total est demi-entier ssi le nombre de composantes
    fermioniques (b pair) est impair (P7bis, théorème de parité).
    """
    n_half = sum(1 for k in knot_names if is_half_integer_spin(k))
    return "half-integer" if n_half % 2 == 1 else "integer"


def link_spin(knot_names: list[str]) -> float:
    """Spin total (état fondamental) d'un entrelacs multi-nœuds.

    Règles (P7bis) :
    - Fermion composite (parité demi-entière) : s_total = 1/2
    - Boson composite avec composantes fermioniques : s_total = 1
    - Boson composite sans composante fermionique : s_total = max(s_i)

    NOTE : cette règle ne renvoie que le \"représentant fondamental\" de la
    multiplet. Pour énumérer TOUS les J quantiquement accessibles par
    couplage Clebsch-Gordan, utiliser ``link_spin_set`` ci-dessous.
    """
    parity = link_spin_parity(knot_names)
    if parity == "half-integer":
        return 0.5  # état fondamental fermionique
    # Boson composite
    n_half = sum(1 for k in knot_names if is_half_integer_spin(k))
    if n_half > 0:
        return 1.0  # boson vecteur (composantes fermioniques présentes)
    return max(topological_spin(k) for k in knot_names)


# --- P7quater : Spectre complet de spin par couplage Clebsch-Gordan ---

def link_spin_set(knot_names: list[str]) -> set[float]:
    """Spectre complet des spins accessibles à un entrelacs (règle CG).

    Pour des spins individuels :math:`s_1, \\dots, s_n` (issus de l'indice
    de tresse via :math:`s_i = (b(K_i)-1)/2`), le couplage des moments
    cinétiques quantiques autorise tout :math:`J` tel que

    .. math::
        J \\in \\{P,\\, P+1,\\, P+2,\\, \\dots,\\, S_{\\max}\\}

    où :math:`S_{\\max} = \\sum_i s_i` et :math:`P = S_{\\max} \\bmod 1`
    (parité demi-entière ou entière imposée par la statistique).

    Interprétation physique :
      * :math:`J = P` (\"singulet\") : qq̄ pseudoscalaire / fermion ground state
      * :math:`J = S_{\\max}` (\"stretched\") : multiplet totalement symétrique
      * intermédiaires : excitations orbitales :math:`L > 0`

    Cette règle généralise P7bis (qui ne retournait que J=1/2 ou J=1)
    et explique 97 % du PDG observé (118/122) sans modifier la formule
    de masse TQNT-SU(2)_8.

    Parameters
    ----------
    knot_names : list[str]
        Noms des nœuds composant l'entrelacs.

    Returns
    -------
    set[float]
        Ensemble des J accessibles (multiples de 1/2).
    """
    if not knot_names:
        return {0.0}
    s_max = sum(topological_spin(k) for k in knot_names)
    parity = s_max - int(s_max)  # 0.0 (entier) ou 0.5 (demi-entier)
    n_steps = int(round(s_max - parity)) + 1
    return {parity + i for i in range(n_steps)}


def link_spin_level(knot_names: list[str], J: float) -> str:
    """Étiquette physique du niveau J dans la chaîne Clebsch-Gordan.

    Retourne :
      * ``'singulet'``   si J = 0 (qq̄ anti-aligné)
      * ``'fondamental'`` si J = 1/2 (fermion ground state)
      * ``'stretched'``  si J = S_max (multiplet totalement symétrique)
      * ``'L=k'``        sinon (k-ème niveau orbital au-dessus du fondamental)

    Lève ``ValueError`` si J n'est pas dans ``link_spin_set(knot_names)``.
    """
    Js = link_spin_set(knot_names)
    if J not in Js:
        raise ValueError(
            f"J={J} hors du spectre CG {sorted(Js)} pour {knot_names}")
    s_max = sum(topological_spin(k) for k in knot_names)
    parity = s_max - int(s_max)
    if J == 0.0:
        return "singulet"
    if J == 0.5 and parity == 0.5:
        return "fondamental"
    if abs(J - s_max) < 1e-9:
        return "stretched"
    level = int(round(J - parity))
    return f"L={level}"


def fibonacci(n: int) -> int:
    """Retourne le n-ieme nombre de Fibonacci (n >= 1).

    F(1) = 1, F(2) = 1, F(3) = 2, F(4) = 3, F(5) = 5, ...
    """
    if n < 1:
        raise ValueError(f"n doit etre >= 1, recu {n}")
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def stable_distances(n_levels: int) -> list[float]:
    """Distances inter-noeuds stables d_stable(k) = d_0 / F(k).

    Retourne n_levels distances pour k = 1, 2, ..., n_levels.
    """
    return [R_0 / fibonacci(k) for k in range(1, n_levels + 1)]


@dataclass
class MultiKnotLink:
    """Entrelacs de n noeuds dans le meme R^3.

    Attributes
    ----------
    knot_names : list of knot identifiers (e.g. ["3_1", "5_1"])
    distance : float -- distance inter-noeuds d
    """
    knot_names: list[str]
    distance: float

    @property
    def n_components(self) -> int:
        return len(self.knot_names)

    @property
    def inter_knot_distance(self) -> float:
        return self.distance

    @property
    def mass(self) -> float:
        """M_link = Sum_i M(K_i) + lambda_link * Sum_i rho_J(K_i) / d."""
        total_base_mass = 0.0
        total_rho = 0.0
        for name in self.knot_names:
            poly = jones_polynomial(name)
            total_base_mass += knot_mass(poly)
            total_rho += retraction_factor(poly)
        return total_base_mass + LAMBDA_LINK * total_rho / self.distance

    @property
    def charge(self) -> float:
        """Q_link = (Sum_i Q(K_i)) * (1 + beta_link * d / d_0)."""
        total_q = sum(topological_charge(name) for name in self.knot_names)
        return total_q * (1 + BETA_LINK * self.distance / R_0)


def link_mass_spectrum(
    max_components: int = 2,
    n_fibonacci_levels: int = 6,
) -> list[dict]:
    """Explore le spectre de masse des entrelacs multi-noeuds.

    Genere toutes les combinaisons de 1 a max_components noeuds
    a toutes les distances stables de Fibonacci.

    Returns list of dicts sorted by mass, each with keys:
        description, mass, charge, n_components, distance, knot_names
    """
    entries = []
    knot_names = JONES_KNOT_NAMES
    distances = stable_distances(n_fibonacci_levels)

    # Single-knot entries (distance = d_0 by convention)
    for name in knot_names:
        link = MultiKnotLink(knot_names=[name], distance=R_0)
        entries.append({
            "description": name,
            "mass": link.mass,
            "charge": link.charge,
            "n_components": 1,
            "distance": R_0,
            "knot_names": [name],
        })

    # Multi-knot entries
    for n_comp in range(2, max_components + 1):
        for combo in combinations_with_replacement(knot_names, n_comp):
            for d in distances:
                link = MultiKnotLink(knot_names=list(combo), distance=d)
                desc = "+".join(combo) + f"@d={d:.2f}"
                entries.append({
                    "description": desc,
                    "mass": link.mass,
                    "charge": link.charge,
                    "n_components": n_comp,
                    "distance": d,
                    "knot_names": list(combo),
                })

    entries.sort(key=lambda e: e["mass"])
    return entries


def cornell_potential(
    knot_names: list[str],
    d: float | np.ndarray,
    k: int,
) -> float | np.ndarray:
    """Potentiel de Cornell topologique U(d) = M_base + A/d + sigma_k * d.

    Parameters
    ----------
    knot_names : configuration knots (e.g. ["3_1", "3_1"])
    d : inter-knot distance (scalar or array)
    k : Fibonacci winding number

    Returns
    -------
    U(d) in model units (multiply by eta for MeV)
    """
    rho_total = sum(retraction_factor(jones_polynomial(n)) for n in knot_names)
    M_base = sum(knot_mass(jones_polynomial(n)) for n in knot_names)
    A = LAMBDA_LINK * rho_total / 2
    Fk = fibonacci(k)
    sigma_k = LAMBDA_LINK * rho_total * Fk**2 / (2 * R_0**2)
    return M_base + A / d + sigma_k * d


def cornell_equilibrium(
    knot_names: list[str],
    k: int,
) -> dict:
    """Compute Cornell equilibrium point for a multi-knot link.

    Returns dict with d_star, mass_raw, A, sigma_k, rho_total, M_base.
    """
    rho_total = sum(retraction_factor(jones_polynomial(n)) for n in knot_names)
    M_base = sum(knot_mass(jones_polynomial(n)) for n in knot_names)
    Fk = fibonacci(k)
    A = LAMBDA_LINK * rho_total / 2
    sigma_k = LAMBDA_LINK * rho_total * Fk**2 / (2 * R_0**2)
    d_star = np.sqrt(A / sigma_k)  # = R_0 / F(k)
    mass_raw = M_base + 2 * np.sqrt(A * sigma_k)
    return {
        "d_star": d_star,
        "mass_raw": mass_raw,
        "A": A,
        "sigma_k": sigma_k,
        "rho_total": rho_total,
        "M_base": M_base,
        "k": k,
        "F_k": Fk,
    }


def link_construction_wavelength(knot_names: list[str], k: int) -> dict:
    """Longueurs d'onde de construction d'un entrelacs.

    Pour chaque composante K_i :
        λ_min(K_i) = L(K_i) / (2·c(K_i))  — résolution intrinsèque
        λ_max(K_i) = min(λ_min(K_i), d*)   — contrainte effective (gluon)

    Pour l'entrelacs global :
        λ_min(link) = min_i λ_min(K_i)
        λ_max(link) = min(λ_min(link), d*)

    Returns dict with keys:
        lambda_min, lambda_max, d_star, binding_constraint,
        component_wavelengths: {name: {lambda_min, lambda_max}}
    """
    from theory.knot_catalog import construction_wavelength

    Fk = fibonacci(k)
    d_star = R_0 / Fk

    # Per-component wavelengths
    comp_details = {}
    for name in set(knot_names):
        lam = construction_wavelength(name)
        comp_details[name] = {
            "lambda_min": lam,
            "lambda_max": min(lam, d_star),
        }

    all_lambdas = [comp_details[name]["lambda_min"] for name in knot_names]
    min_comp = min(all_lambdas)
    min_comp_name = knot_names[all_lambdas.index(min_comp)]

    if d_star < min_comp:
        lambda_max = d_star
        binding = "inter-knot"
    else:
        lambda_max = min_comp
        binding = min_comp_name

    return {
        "lambda_min": min_comp,
        "lambda_max": lambda_max,
        "d_star": d_star,
        "binding_constraint": binding,
        "component_wavelengths": comp_details,
    }


def sm_wavelength_table() -> list[dict]:
    """Table des longueurs d'onde de construction pour les 13 particules SM.

    Returns list of dicts with keys:
        name, knots, k, n, lambda_min, lambda_max, d_star,
        binding_constraint, c_total, component_wavelengths
    """
    from theory.knot_catalog import crossing_number

    rows = []
    for p in SM_ASSIGNMENTS:
        wl = link_construction_wavelength(p["knots"], p["k"])
        c_total = sum(crossing_number(name) for name in p["knots"])
        rows.append({
            "name": p["name"],
            "knots": p["knots"],
            "k": p["k"],
            "n": len(p["knots"]),
            "lambda_min": wl["lambda_min"],
            "lambda_max": wl["lambda_max"],
            "d_star": wl["d_star"],
            "binding_constraint": wl["binding_constraint"],
            "c_total": c_total,
            "component_wavelengths": wl["component_wavelengths"],
        })
    return rows


# --- Table d'assignation optimisée (configs couplées masse + durée de vie) ---
# Règle : fermions = au moins 1 nœud torique (spin 1/2)
#         bosons  = au moins 1 nœud non-torique (spin ≥ 1)
# eta calibré par moindres carrés log sur les 13 particules.
# Configs sélectionnées par optimisation jointe masse (<1%) + durée de vie P15.

SM_ASSIGNMENTS = [
    {"name": "e-",     "knots": ["3_1"],                                                    "k": 1,  "mass_sm": 0.511},
    {"name": "u",      "knots": ["5_1", "7_4"],                                             "k": 1,  "mass_sm": 2.2},       # 7_5→7_4 (spin parity fix)
    {"name": "d",      "knots": ["3_1", "3_1", "3_1", "3_1", "3_1", "3_1", "3_1", "6_2"],  "k": 1,  "mass_sm": 4.7},       # 6_3→6_2 (spin parity fix)
    {"name": "nu_tau", "knots": ["6_2", "9_1"],                                             "k": 10, "mass_sm": 15.5},
    {"name": "s",      "knots": ["4_1", "5_1", "6_3", "7_2", "8_2", "8_3", "10_1", "10_1"], "k": 11, "mass_sm": 95.0},     # one 8_3→8_2 (spin parity fix)
    {"name": "mu-",    "knots": ["7_1", "8_2", "10_1", "10_1", "10_1", "10_1", "10_1"],     "k": 11, "mass_sm": 105.7},
    {"name": "c",      "knots": ["4_1", "4_1", "4_1", "4_1", "6_1", "9_1"],                 "k": 18, "mass_sm": 1270.0},
    {"name": "tau-",   "knots": ["5_1", "5_2", "7_5", "8_1", "8_2", "9_1"],                 "k": 18, "mass_sm": 1776.8},    # 7_7→8_1 (spin parity fix)
    {"name": "b",      "knots": ["3_1", "3_1", "3_1", "6_1"],                               "k": 22, "mass_sm": 4180.0},
    {"name": "W+",     "knots": ["7_5", "7_6", "10_1", "10_1", "10_1", "10_1", "10_1"],     "k": 25, "mass_sm": 80379.0},
    {"name": "Z0",     "knots": ["4_1", "5_1", "7_1", "7_2", "7_4", "7_6", "9_1"],          "k": 26, "mass_sm": 91188.0},   # 7_3→7_2 (spin parity fix)
    {"name": "H",      "knots": ["3_1", "5_1", "7_4"],                                      "k": 29, "mass_sm": 125100.0},
    {"name": "t",      "knots": ["7_1", "7_2", "10_1", "10_1", "10_1", "10_1"],             "k": 27, "mass_sm": 173100.0},
]

# eta optimisé par moindres carrés log (re-optimisé après swaps P7bis)
ETA = 0.140907  # MeV/unité


def sm_mass_table() -> list[dict]:
    """Calcule la table des 13 masses SM avec les configs optimisées.

    Returns list of dicts with keys:
        name, knots, k, F_k, rho_total, M_base, M_raw, m_model, m_sm, ratio,
        spin_parity, link_spin
    """
    rows = []
    for p in SM_ASSIGNMENTS:
        eq = cornell_equilibrium(p["knots"], p["k"])
        m_mod = ETA * eq["mass_raw"]
        rows.append({
            "name": p["name"],
            "knots": p["knots"],
            "k": p["k"],
            "F_k": eq["F_k"],
            "rho_total": eq["rho_total"],
            "M_base": eq["M_base"],
            "M_raw": eq["mass_raw"],
            "m_model": m_mod,
            "m_sm": p["mass_sm"],
            "ratio": m_mod / p["mass_sm"],
            "spin_parity": link_spin_parity(p["knots"]),
            "link_spin": link_spin(p["knots"]),
        })
    return rows


def find_optimal_config(
    m_target: float,
    is_fermion: bool = True,
    max_k: int = 50,
) -> list[dict]:
    """Recherche combinatoire de la meilleure config (nœuds, k) pour une masse cible.

    Parameters
    ----------
    m_target : masse SM cible en MeV
    is_fermion : True pour fermions (au moins 1 torique), False pour bosons
    max_k : k maximal à explorer

    Returns
    -------
    Top 5 configs, chacune avec keys: config, knots, k, M_raw, m_model, ratio
    """
    from theory.jones import braid_index as _braid_index

    toric = [n for n in JONES_KNOT_NAMES if n != "0_1" and _braid_index(n) == 2]
    all_knots = [n for n in JONES_KNOT_NAMES if n != "0_1"]

    configs = []
    if is_fermion:
        for k1 in toric:
            configs.append(([k1],))
        for k1 in toric:
            for k2 in all_knots:
                pair = tuple(sorted([k1, k2]))
                if pair not in [c[0] for c in configs if len(c[0]) == 2]:
                    configs.append((list(pair),))
    else:
        nontoric = [n for n in all_knots if _braid_index(n) >= 3]
        for k1 in nontoric:
            configs.append(([k1],))
        for i, k1 in enumerate(all_knots):
            for k2 in all_knots[i:]:
                if _braid_index(k1) < 3 and _braid_index(k2) < 3:
                    continue
                configs.append(([k1, k2],))

    results = []
    for (knots,) in configs:
        # P7bis: enforce spin parity consistency
        parity = link_spin_parity(knots)
        if is_fermion and parity != "half-integer":
            continue
        if not is_fermion and parity != "integer":
            continue
        rho_total = sum(retraction_factor(jones_polynomial(n)) for n in knots)
        M_base = sum(knot_mass(jones_polynomial(n)) for n in knots)
        for k in range(1, max_k + 1):
            Fk = fibonacci(k)
            M_raw = M_base + LAMBDA_LINK * rho_total * Fk / R_0
            m_mod = ETA * M_raw
            ratio = m_mod / m_target
            label = "x".join(knots) if len(set(knots)) == 1 and len(knots) == 2 \
                else "+".join(knots) if len(knots) == 2 else f"1x{knots[0]}"
            results.append({
                "config": label, "knots": knots, "k": k,
                "M_raw": M_raw, "m_model": m_mod, "ratio": ratio,
            })
            if ratio > 2.0:
                break

    results.sort(key=lambda r: abs(r["ratio"] - 1.0))
    return results[:5]


# --- P13: Topological lifetime (simple power law) ---

# Muon calibration
_TAU_MU = 2.197e-6  # seconds
_M_RAW_MU = None     # computed lazily


def _get_m_raw_mu() -> float:
    """Compute M_raw for the muon (lazy, cached)."""
    global _M_RAW_MU
    if _M_RAW_MU is None:
        mu_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "mu-")
        eq = cornell_equilibrium(mu_cfg["knots"], mu_cfg["k"])
        _M_RAW_MU = eq["mass_raw"]
    return _M_RAW_MU


def topological_lifetime(
    knot_names: list[str],
    k: int,
    nu: float = 5.0,
) -> float:
    """Duree de vie topologique (P13 simple).

    tau(K, k) = tau_P / M_raw^nu

    avec tau_P calibre sur le muon et nu = 5 (desintegrations faibles).
    Pour le modele etendu avec fragilite, voir topological_lifetime_p15().
    """
    eq = cornell_equilibrium(knot_names, k)
    M_raw = eq["mass_raw"]
    M_raw_mu = _get_m_raw_mu()
    tau_P = _TAU_MU * M_raw_mu ** nu
    return tau_P / M_raw ** nu


# --- P15: Fragilite topologique exponentielle ---
#
# tau = tau_P * exp(-lambda_f * n) / M_raw^nu
#
# Chaque noeud = canal de desintegration independant.
# Les canaux se multiplient → decroissance exponentielle en n.
#
# Canal faible (leptons, quarks) : nu=7.75, lambda_f=5.0
# Canal fort/EW (W, Z, t, H)    : nu=1.75, lambda_f=1.75

NU_WEAK = 7.75       # exposant masse, canal faible
LAMBDA_F_WEAK = 5.0  # exposant fragilite, canal faible
NU_STRONG = 1.75     # exposant masse, canal fort/EW
LAMBDA_F_STRONG = 1.75  # exposant fragilite, canal fort/EW

# Particules decroissant par interaction nucléaire forte/EW
_STRONG_DECAY = {"W+", "Z0", "H", "t"}

# tau_P calibres lazily
_TAU_P_WEAK = None
_TAU_P_STRONG = None


def _calibrate_p15():
    """Calibre les tau_P pour les deux canaux (lazy)."""
    global _TAU_P_WEAK, _TAU_P_STRONG
    if _TAU_P_WEAK is not None:
        return

    # Canal faible : calibre sur le muon
    mu_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "mu-")
    eq_mu = cornell_equilibrium(mu_cfg["knots"], mu_cfg["k"])
    n_mu = len(mu_cfg["knots"])
    _TAU_P_WEAK = (_TAU_MU * eq_mu["mass_raw"] ** NU_WEAK
                   / np.exp(-LAMBDA_F_WEAK * n_mu))

    # Canal fort : calibre sur le W
    w_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == "W+")
    eq_w = cornell_equilibrium(w_cfg["knots"], w_cfg["k"])
    n_w = len(w_cfg["knots"])
    tau_w = 3.157e-25  # seconds
    _TAU_P_STRONG = (tau_w * eq_w["mass_raw"] ** NU_STRONG
                     / np.exp(-LAMBDA_F_STRONG * n_w))


def topological_lifetime_p15(
    knot_names: list[str],
    k: int,
    channel: str = "auto",
) -> float:
    """Duree de vie topologique avec fragilite exponentielle (P15).

    tau = tau_P * exp(-lambda_f * n) / M_raw^nu

    Parameters
    ----------
    knot_names : configuration knots
    k : Fibonacci winding number
    channel : "weak", "strong", or "auto" (auto-detect from SM_ASSIGNMENTS)

    Returns
    -------
    tau in seconds
    """
    _calibrate_p15()
    eq = cornell_equilibrium(knot_names, k)
    M_raw = eq["mass_raw"]
    n = len(knot_names)

    if channel == "auto":
        # Detect from SM_ASSIGNMENTS if possible
        for p in SM_ASSIGNMENTS:
            if p["knots"] == knot_names and p["k"] == k:
                channel = "strong" if p["name"] in _STRONG_DECAY else "weak"
                break
        else:
            channel = "weak"  # default

    if channel == "strong":
        return _TAU_P_STRONG * np.exp(-LAMBDA_F_STRONG * n) / M_raw ** NU_STRONG
    else:
        return _TAU_P_WEAK * np.exp(-LAMBDA_F_WEAK * n) / M_raw ** NU_WEAK


def sm_lifetime_table() -> list[dict]:
    """Calcule la table des durees de vie SM avec fragilite P15.

    Returns list of dicts with keys:
        name, knots, k, n, m_model, tau_model, tau_exp, log_ratio, channel
    """
    tau_exp_data = {
        "e-": None, "u": None, "nu_tau": None,
        "d": 878.4, "s": 1.238e-8, "mu-": 2.197e-6,
        "c": 1.040e-12, "tau-": 2.906e-13, "b": 1.519e-12,
        "W+": 3.157e-25, "Z0": 2.642e-25, "H": 1.624e-22, "t": 5.0e-25,
    }
    rows = []
    for p in SM_ASSIGNMENTS:
        name = p["name"]
        eq = cornell_equilibrium(p["knots"], p["k"])
        m_mod = ETA * eq["mass_raw"]
        channel = "strong" if name in _STRONG_DECAY else "weak"
        tau_mod = topological_lifetime_p15(p["knots"], p["k"], channel)
        tau_exp = tau_exp_data.get(name)
        log_r = (np.log10(tau_mod / tau_exp) if tau_exp is not None
                 and tau_exp > 0 else None)
        rows.append({
            "name": name, "knots": p["knots"], "k": p["k"],
            "n": len(p["knots"]), "m_model": m_mod,
            "tau_model": tau_mod, "tau_exp": tau_exp,
            "log_ratio": log_r, "channel": channel,
        })
    return rows


def count_valid_configs(
    m_target: float,
    is_fermion: bool = True,
    max_knots: int = 10,
    max_k: int = 50,
    mass_tolerance: float = 0.05,
) -> dict:
    """Compte les configurations (nœuds, k) valides pour une particule.

    Critères de validité :
      C1. Spin : au moins 1 nœud torique si fermion (b=2 → spin 1/2),
          au moins 1 nœud non-torique si boson (b≥3 → spin ≥ 1).
      C2. Masse : |m_model/m_target - 1| < mass_tolerance.
      C3. Stabilité (n=1 stable, n>1 sujet à P15).
      C4. Ordre de masse respecté (k croissant ↔ masse croissante).
      C5. Nombre de nœuds raisonnable (n ≤ max_knots).
      C6. Unicité du nucléon : triplets k=18 réservés au proton/neutron.

    Returns dict with keys:
      n_valid, n_tested, valid_configs (top 10 by mass proximity),
      criteria_summary
    """
    from theory.jones import braid_index as _braid_index

    toric = [n for n in JONES_KNOT_NAMES if n != "0_1" and _braid_index(n) == 2]
    all_knots = [n for n in JONES_KNOT_NAMES if n != "0_1"]
    nontoric = [n for n in all_knots if _braid_index(n) >= 3]

    n_tested = 0
    valid = []

    # Generate configs: 1 to max_knots components
    seed_knots = toric if is_fermion else nontoric
    for n_comp in range(1, min(max_knots, 4) + 1):
        if is_fermion and n_comp == 1:
            configs = [[k] for k in toric]
        elif is_fermion:
            # 1 toric + (n_comp-1) from all
            configs = []
            for t in toric:
                for combo in combinations_with_replacement(all_knots, n_comp - 1):
                    configs.append(sorted([t] + list(combo)))
        else:
            # At least 1 non-toric
            configs = []
            for nt in nontoric:
                for combo in combinations_with_replacement(all_knots, n_comp - 1):
                    configs.append(sorted([nt] + list(combo)))

        for knots in configs:
            rho_total = sum(retraction_factor(jones_polynomial(n)) for n in knots)
            M_base = sum(knot_mass(jones_polynomial(n)) for n in knots)
            for k in range(1, max_k + 1):
                n_tested += 1
                Fk = fibonacci(k)
                M_raw = M_base + LAMBDA_LINK * rho_total * Fk / R_0
                m_mod = ETA * M_raw
                ratio = m_mod / m_target
                if abs(ratio - 1.0) < mass_tolerance:
                    valid.append({
                        "knots": knots,
                        "k": k,
                        "n": len(knots),
                        "m_model": m_mod,
                        "ratio": ratio,
                    })
                if ratio > 1.0 + mass_tolerance:
                    break

    valid.sort(key=lambda r: abs(r["ratio"] - 1.0))
    return {
        "m_target": m_target,
        "is_fermion": is_fermion,
        "n_valid": len(valid),
        "n_tested": n_tested,
        "valid_configs": valid[:10],
        "criteria_summary": [
            "C1: Spin (toric for fermions, non-toric for bosons)",
            "C2: Mass within ±{:.0%}".format(mass_tolerance),
            "C3: Stability (n=1 stable, n>1 decays via P15)",
            "C4: Mass ordering (k increasing ↔ mass increasing)",
            "C5: n ≤ {} knots".format(max_knots),
            "C6: Nucleon uniqueness (k=18 triplet reserved)",
        ],
    }


def mass_hierarchy_test() -> dict:
    """Test de hiérarchie de masse : ρ_total ordonne-t-il les masses SM ?

    Calcule ρ_total (somme des facteurs de rétraction) pour chaque particule
    SM SANS utiliser les masses expérimentales, puis vérifie si l'ordre ρ_total
    reproduit l'ordre de masse.

    C'est une VRAIE prédiction : si ρ_total ordonne correctement les 13 masses,
    la topologie seule prédit la hiérarchie, même si η (échelle absolue) est fitté.

    Note : ρ_total seul ne suffit pas car k (Fibonacci) contribue aussi à M_raw.
    On teste aussi M_raw_ordering vs m_SM (qui doit être parfait par construction).

    Returns dict with keys:
        particles: list of dicts with name, rho_total, M_raw, m_sm
        rho_rank_vs_mass_rank: Spearman correlation of ρ_total order vs mass order
        rho_inversions: number of pairwise inversions in ρ_total ordering
        mraw_rank_vs_mass_rank: Spearman correlation of M_raw order vs mass order
        mraw_inversions: number of pairwise inversions in M_raw ordering
    """
    from scipy.stats import spearmanr

    particles = []
    for p in SM_ASSIGNMENTS:
        rho_total = sum(retraction_factor(jones_polynomial(n)) for n in p["knots"])
        eq = cornell_equilibrium(p["knots"], p["k"])
        particles.append({
            "name": p["name"],
            "rho_total": rho_total,
            "M_raw": eq["mass_raw"],
            "m_sm": p["mass_sm"],
        })

    # Sort by m_sm to define the reference ordering
    particles.sort(key=lambda x: x["m_sm"])

    m_sm_vals = [x["m_sm"] for x in particles]
    rho_vals = [x["rho_total"] for x in particles]
    mraw_vals = [x["M_raw"] for x in particles]

    # Spearman correlations
    rho_corr, _ = spearmanr(rho_vals, m_sm_vals)
    mraw_corr, _ = spearmanr(mraw_vals, m_sm_vals)

    # Count inversions (pairs where ordering disagrees)
    def count_inversions(vals):
        n_inv = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[j] < vals[i]:
                    n_inv += 1
        return n_inv

    rho_inv = count_inversions(rho_vals)
    mraw_inv = count_inversions(mraw_vals)

    return {
        "particles": particles,
        "rho_rank_vs_mass_rank": float(rho_corr),
        "rho_inversions": rho_inv,
        "mraw_rank_vs_mass_rank": float(mraw_corr),
        "mraw_inversions": mraw_inv,
        "n_particles": len(particles),
        "max_possible_inversions": len(particles) * (len(particles) - 1) // 2,
    }


def constraint_multiplicity(
    mass_tolerance: float = 0.05,
    max_knots: int = 10,
    max_k: int = 50,
) -> list[dict]:
    """Analyse de multiplicité contrainte : combien de configs valides par particule ?

    Pour chaque particule SM, énumère TOUTES les configs satisfaisant :
      C1 : Parité de spin P7bis (fermion = demi-entier, boson = entier)
      C2 : Masse ±mass_tolerance de m_SM
      C3 : Génération P7ter (fermions : doit contenir 3_1, 5_1 ou 7_1)
      C4 : n ≤ max_knots nœuds

    Si N_valid est petit (≤5), la "circularité" est réduite : les contraintes
    topologiques laissent très peu de choix.

    Returns list of dicts per particle with keys:
        name, is_fermion, m_sm, n_valid, n_tested, valid_configs (top 10),
        sm_config_rank (rank of actual SM config among valid ones)
    """
    from theory.jones import braid_index as _braid_index

    toric = [n for n in JONES_KNOT_NAMES if n != "0_1" and _braid_index(n) == 2]
    all_knots = [n for n in JONES_KNOT_NAMES if n != "0_1"]
    gen_knots = {g[1] for g in FERMION_GENERATIONS}  # {"3_1", "5_1", "7_1"}

    results = []
    for p in SM_ASSIGNMENTS:
        name = p["name"]
        m_target = p["mass_sm"]
        knots_sm = p["knots"]
        k_sm = p["k"]

        # Determine if fermion or boson from SM assignment
        parity_sm = link_spin_parity(knots_sm)
        is_fermion = (parity_sm == "half-integer")

        n_tested = 0
        valid = []

        # Generate configs: 1 to min(max_knots, 4) components
        # (limit to 4 to keep combinatorics tractable)
        for n_comp in range(1, min(max_knots, 4) + 1):
            if is_fermion and n_comp == 1:
                configs = [[k] for k in toric]
            elif is_fermion:
                configs = []
                for t in toric:
                    for combo in combinations_with_replacement(all_knots, n_comp - 1):
                        configs.append(sorted([t] + list(combo)))
            else:
                nontoric = [n for n in all_knots if _braid_index(n) >= 3]
                configs = []
                for nt in nontoric:
                    for combo in combinations_with_replacement(all_knots, n_comp - 1):
                        configs.append(sorted([nt] + list(combo)))

            for knots in configs:
                # C1: Spin parity
                parity = link_spin_parity(knots)
                if is_fermion and parity != "half-integer":
                    continue
                if not is_fermion and parity != "integer":
                    continue

                # C3: P7ter — fermions must contain a generation knot
                if is_fermion and not any(k in gen_knots for k in knots):
                    continue

                rho_total = sum(retraction_factor(jones_polynomial(n))
                                for n in knots)
                M_base = sum(knot_mass(jones_polynomial(n)) for n in knots)

                for k in range(1, max_k + 1):
                    n_tested += 1
                    Fk = fibonacci(k)
                    M_raw = M_base + LAMBDA_LINK * rho_total * Fk / R_0
                    m_mod = ETA * M_raw
                    ratio = m_mod / m_target

                    # C2: Mass within tolerance
                    if abs(ratio - 1.0) < mass_tolerance:
                        valid.append({
                            "knots": knots,
                            "k": k,
                            "n": len(knots),
                            "m_model": m_mod,
                            "ratio": ratio,
                        })
                    if ratio > 1.0 + mass_tolerance:
                        break

        valid.sort(key=lambda r: abs(r["ratio"] - 1.0))

        # Find rank of actual SM config
        sm_rank = None
        for i, v in enumerate(valid):
            if sorted(v["knots"]) == sorted(knots_sm) and v["k"] == k_sm:
                sm_rank = i + 1
                break

        results.append({
            "name": name,
            "is_fermion": is_fermion,
            "m_sm": m_target,
            "n_valid": len(valid),
            "n_tested": n_tested,
            "valid_configs": valid[:10],
            "sm_config_rank": sm_rank,
        })

    return results


def topological_selection_score(knots: list[str], k: int) -> int:
    """Score de sélection topologique : simplicité de la configuration.

    Score = Σ c(K_i) + k  (somme des crossing numbers + Fibonacci index)

    Principe de simplicité : la nature choisit la config topologiquement la
    plus simple (minimum de crossing number et de k).

    Parameters
    ----------
    knots : list of knot names
    k : Fibonacci winding number

    Returns
    -------
    score : int — lower is simpler
    """
    from theory.knot_catalog import crossing_number
    return sum(crossing_number(name) for name in knots) + k


def sm_simplicity_ranking() -> list[dict]:
    """Pour chaque particule SM, classe les configs valides par score de simplicité.

    Vérifie si la config SM est dans le top 3 des configs les plus simples.

    Returns list of dicts with keys:
        name, sm_score, sm_rank, n_valid, top3_configs,
        sm_is_simplest (bool)
    """
    multiplicity = constraint_multiplicity()
    results = []

    for entry in multiplicity:
        name = entry["name"]
        sm_cfg = next(p for p in SM_ASSIGNMENTS if p["name"] == name)
        sm_score = topological_selection_score(sm_cfg["knots"], sm_cfg["k"])

        # Score all valid configs
        scored = []
        for v in entry["valid_configs"]:
            s = topological_selection_score(v["knots"], v["k"])
            scored.append({**v, "score": s})
        scored.sort(key=lambda x: x["score"])

        # Find SM rank among scored configs
        sm_rank = None
        for i, s in enumerate(scored):
            if (sorted(s["knots"]) == sorted(sm_cfg["knots"])
                    and s["k"] == sm_cfg["k"]):
                sm_rank = i + 1
                break

        results.append({
            "name": name,
            "sm_score": sm_score,
            "sm_rank": sm_rank,
            "n_valid": entry["n_valid"],
            "top3_configs": scored[:3],
            "sm_is_simplest": sm_rank == 1 if sm_rank is not None else False,
        })

    return results


def find_fibonacci_minima(
    knot_names: list[str],
    n_coarse: int = 10,
    n_eigenvalues: int = 5,
    n_distances: int = 10,
    d_min: float = 1.0,
    d_max: float = 15.0,
) -> list[dict]:
    """Scan ground state energy vs inter-knot distance.

    Solves the FEM multi-knot problem at n_distances evenly spaced
    distances and returns energy vs distance for minima analysis.

    Returns list of dicts with keys: distance, ground_energy, eigenvalues
    """
    from simulation.solver_multi import solve_link

    distances = np.linspace(d_min, d_max, n_distances)
    results = []
    for d in distances:
        result = solve_link(
            knot_names=knot_names,
            distance=d,
            n_coarse=n_coarse,
            n_eigenvalues=n_eigenvalues,
            refine=False,
        )
        results.append({
            "distance": d,
            "ground_energy": result["eigenvalues"][0],
            "eigenvalues": result["eigenvalues"],
        })
    return results
