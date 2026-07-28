"""
Theorie Nodale v2 -- Classification des Particules par Noeuds Topologiques

Maps the 28 fundamental particles (12 fermions + 12 antifermions + 4 gauge
bosons) to topological knots via their Jones polynomials and retraction
factors.
"""
from __future__ import annotations
from dataclasses import dataclass
from theory.jones import (
    JONES_KNOT_NAMES as KNOT_NAMES,
    jones_polynomial,
    topological_spin,
    topological_charge,
)
from theory.retraction import retraction_factor, knot_mass

STANDARD_MODEL_PARTICLES = [
    # Leptons (6)
    {"name": "nu_e",    "type": "fermion", "spin": 0.5, "charge": 0,    "mass_mev": 0.0000022},
    {"name": "nu_mu",   "type": "fermion", "spin": 0.5, "charge": 0,    "mass_mev": 0.17},
    {"name": "nu_tau",  "type": "fermion", "spin": 0.5, "charge": 0,    "mass_mev": 15.5},
    {"name": "e-",      "type": "fermion", "spin": 0.5, "charge": -1,   "mass_mev": 0.511},
    {"name": "mu-",     "type": "fermion", "spin": 0.5, "charge": -1,   "mass_mev": 105.7},
    {"name": "tau-",    "type": "fermion", "spin": 0.5, "charge": -1,   "mass_mev": 1776.8},
    # Quarks (6)
    {"name": "up",      "type": "fermion", "spin": 0.5, "charge": 2/3,  "mass_mev": 2.2},
    {"name": "down",    "type": "fermion", "spin": 0.5, "charge": -1/3, "mass_mev": 4.7},
    {"name": "charm",   "type": "fermion", "spin": 0.5, "charge": 2/3,  "mass_mev": 1270.0},
    {"name": "strange", "type": "fermion", "spin": 0.5, "charge": -1/3, "mass_mev": 95.0},
    {"name": "top",     "type": "fermion", "spin": 0.5, "charge": 2/3,  "mass_mev": 173100.0},
    {"name": "bottom",  "type": "fermion", "spin": 0.5, "charge": -1/3, "mass_mev": 4180.0},
    # Anti-leptons (6)
    {"name": "nu_e_bar",    "type": "antifermion", "spin": 0.5, "charge": 0,    "mass_mev": 0.0000022},
    {"name": "nu_mu_bar",   "type": "antifermion", "spin": 0.5, "charge": 0,    "mass_mev": 0.17},
    {"name": "nu_tau_bar",  "type": "antifermion", "spin": 0.5, "charge": 0,    "mass_mev": 15.5},
    {"name": "e+",          "type": "antifermion", "spin": 0.5, "charge": 1,    "mass_mev": 0.511},
    {"name": "mu+",         "type": "antifermion", "spin": 0.5, "charge": 1,    "mass_mev": 105.7},
    {"name": "tau+",        "type": "antifermion", "spin": 0.5, "charge": 1,    "mass_mev": 1776.8},
    # Anti-quarks (6)
    {"name": "anti-up",      "type": "antifermion", "spin": 0.5, "charge": -2/3, "mass_mev": 2.2},
    {"name": "anti-down",    "type": "antifermion", "spin": 0.5, "charge": 1/3,  "mass_mev": 4.7},
    {"name": "anti-charm",   "type": "antifermion", "spin": 0.5, "charge": -2/3, "mass_mev": 1270.0},
    {"name": "anti-strange", "type": "antifermion", "spin": 0.5, "charge": 1/3,  "mass_mev": 95.0},
    {"name": "anti-top",     "type": "antifermion", "spin": 0.5, "charge": -2/3, "mass_mev": 173100.0},
    {"name": "anti-bottom",  "type": "antifermion", "spin": 0.5, "charge": 1/3,  "mass_mev": 4180.0},
    # Gauge bosons (3) — P14 : le gluon n'est pas une particule,
    # c'est le reliquat d'onde non-nodifié du champ χ.
    {"name": "photon",  "type": "boson", "spin": 1.0, "charge": 0,  "mass_mev": 0.0},
    {"name": "W",       "type": "boson", "spin": 1.0, "charge": 1,  "mass_mev": 80379.0},
    {"name": "Z",       "type": "boson", "spin": 1.0, "charge": 0,  "mass_mev": 91188.0},
]


@dataclass
class KnotParticle:
    """Particule définie uniquement par son nœud — spin et charge dérivés."""
    name: str
    knot: str

    @property
    def spin(self) -> float:
        """Spin dérivé de l'indice de tresse : s(K) = (b(K) - 1) / 2."""
        return topological_spin(self.knot)

    @property
    def charge(self) -> float:
        """Charge topologique Q(K) = (p_min + p_max) / span(V_K)."""
        return topological_charge(self.knot)

    @property
    def retraction(self) -> float:
        poly = jones_polynomial(self.knot)
        return retraction_factor(poly)

    @property
    def mass(self) -> float:
        poly = jones_polynomial(self.knot)
        return knot_mass(poly)

    @property
    def is_fermion(self) -> bool:
        """Fermion si spin est demi-entier (1/2, 3/2, ...)."""
        return (self.spin % 1) == 0.5

    @property
    def is_boson(self) -> bool:
        """Boson si spin est entier (0, 1, 2, ...)."""
        return (self.spin % 1) == 0.0

    @property
    def particle_type(self) -> str:
        """Type de particule déduit du spin."""
        s = self.spin
        if s == 0:
            return "scalar"
        elif s == 0.5:
            return "fermion"
        elif s == 1.0:
            return "gauge_boson"
        elif s == 1.5:
            return "gravitino"
        else:
            return f"spin_{s}"


def knot_mass_spectrum() -> list[tuple[str, float]]:
    entries = []
    for name in KNOT_NAMES:
        poly = jones_polynomial(name)
        m = knot_mass(poly)
        entries.append((name, m))
    return sorted(entries, key=lambda x: x[1])


def find_best_knot_mapping() -> dict[str, list[dict]]:
    import numpy as np
    spectrum = knot_mass_spectrum()
    particles_sorted = sorted(STANDARD_MODEL_PARTICLES, key=lambda p: p["mass_mev"])

    mass_groups: list[list[dict]] = []
    current_group: list[dict] = [particles_sorted[0]]

    for i in range(1, len(particles_sorted)):
        prev_mass = max(particles_sorted[i - 1]["mass_mev"], 1e-10)
        curr_mass = max(particles_sorted[i]["mass_mev"], 1e-10)
        if np.log10(curr_mass) - np.log10(prev_mass) < 1.0:
            current_group.append(particles_sorted[i])
        else:
            mass_groups.append(current_group)
            current_group = [particles_sorted[i]]
    mass_groups.append(current_group)

    mapping: dict[str, list[dict]] = {}
    for i, group in enumerate(mass_groups):
        if i < len(spectrum):
            knot_name = spectrum[i][0]
        else:
            knot_name = f"unknown_knot_{i}"
        mapping[knot_name] = group

    return mapping


def knot_particle_table() -> list[KnotParticle]:
    """Table des 20 nœuds avec propriétés dérivées topologiquement."""
    return [KnotParticle(name=name, knot=name) for name in KNOT_NAMES]


def find_sm_candidates() -> dict[str, list[dict]]:
    """Association nœud → particules SM compatibles (par spin et signe de charge)."""
    particles = knot_particle_table()
    result: dict[str, list[dict]] = {}
    for kp in particles:
        compatible = []
        for sm in STANDARD_MODEL_PARTICLES:
            if abs(kp.spin - sm["spin"]) > 1e-10:
                continue
            # Compatibilité de signe de charge
            kq = kp.charge
            sq = sm["charge"]
            if kq == 0 and sq == 0:
                compatible.append(sm)
            elif kq > 0 and sq > 0:
                compatible.append(sm)
            elif kq < 0 and sq < 0:
                compatible.append(sm)
        if compatible:
            result[kp.knot] = compatible
    return result


# =====================================================================
# Modèle de générations — Correspondances exactes nœuds → particules SM
# =====================================================================
#
# Les nœuds toriques T(2,s) (b=2, spin 1/2) hébergent chacun une
# GÉNÉRATION de 4 fermions. Les charges viennent de la formule de
# Gell-Mann–Nishijima : Q = I₃ + Y/2.
# Les bosons gardent la charge topologique directe Q(K).

from enum import Enum


class IsospinSector(Enum):
    """Secteur d'isospin faible — détermine l'hypercharge Y."""
    LEPTON = "lepton"   # Y = -1
    QUARK = "quark"     # Y = +1/3


_HYPERCHARGE = {
    IsospinSector.LEPTON: -1.0,
    IsospinSector.QUARK: 1 / 3,
}


@dataclass
class FermionState:
    """État propre fermionique sur le complément d'un nœud torique.

    La charge électrique est dérivée de Gell-Mann–Nishijima :
        Q = I₃ + Y/2
    """
    name: str
    symbol: str
    sector: IsospinSector
    isospin_3: float       # I₃ = +1/2 ou -1/2
    hypercharge: float     # Y

    @property
    def charge(self) -> float:
        """Charge électrique Q = I₃ + Y/2 (Gell-Mann–Nishijima)."""
        return self.isospin_3 + self.hypercharge / 2

    @property
    def spin(self) -> float:
        return 0.5


@dataclass
class FermionGeneration:
    """Génération fermionique associée à un nœud torique T(2,s).

    Chaque nœud torique héberge 4 fermions : neutrino, lepton chargé,
    quark up-type, quark down-type.
    """
    generation: int
    knot: str
    torus_param: int       # s dans T(2,s) : 3, 5 ou 7

    neutrino: FermionState
    charged_lepton: FermionState
    up_quark: FermionState
    down_quark: FermionState

    @property
    def fermions(self) -> list[FermionState]:
        """Les 4 fermions de cette génération."""
        return [self.neutrino, self.charged_lepton,
                self.up_quark, self.down_quark]

    @property
    def mass_knot(self) -> float:
        """Masse topologique du nœud porteur."""
        poly = jones_polynomial(self.knot)
        return knot_mass(poly)

    @property
    def retraction_knot(self) -> float:
        """Facteur de rétractation du nœud porteur."""
        poly = jones_polynomial(self.knot)
        return retraction_factor(poly)


@dataclass
class CompositeParticle:
    """Particule composite (baryon) = combinaison de fermions."""
    name: str
    symbol: str
    constituents: list[FermionState]

    @property
    def charge(self) -> float:
        """Charge totale = somme des charges des constituants."""
        return sum(f.charge for f in self.constituents)

    @property
    def baryon_number(self) -> int:
        """Nombre baryonique = nombre de quarks / 3."""
        n_quarks = sum(1 for f in self.constituents
                       if f.sector == IsospinSector.QUARK)
        return n_quarks // 3


@dataclass
class BosonCorrespondence:
    """Correspondance nœud → boson de jauge (charge topologique directe).

    P14 : le photon et le gluon sont des ondes pures (is_pure_wave=True),
    sans structure de nœud. Les bosons massifs (W±, Z⁰) sont des nœuds.
    """
    knot: str
    boson_name: str
    symbol: str
    sm_charge: float
    sm_mass_mev: float
    is_mirror: bool = False
    is_pure_wave: bool = False


# --- Génération 1 : 3_1 = T(2,3) ---

_nu_e = FermionState("electron neutrino", "νe", IsospinSector.LEPTON,
                      isospin_3=+0.5, hypercharge=-1.0)
_e = FermionState("electron", "e⁻", IsospinSector.LEPTON,
                   isospin_3=-0.5, hypercharge=-1.0)
_u = FermionState("up", "u", IsospinSector.QUARK,
                   isospin_3=+0.5, hypercharge=1/3)
_d = FermionState("down", "d", IsospinSector.QUARK,
                   isospin_3=-0.5, hypercharge=1/3)

GENERATION_1 = FermionGeneration(
    generation=1, knot="3_1", torus_param=3,
    neutrino=_nu_e, charged_lepton=_e, up_quark=_u, down_quark=_d,
)

# --- Génération 2 : 5_1 = T(2,5) ---

_nu_mu = FermionState("muon neutrino", "νμ", IsospinSector.LEPTON,
                       isospin_3=+0.5, hypercharge=-1.0)
_mu = FermionState("muon", "μ⁻", IsospinSector.LEPTON,
                    isospin_3=-0.5, hypercharge=-1.0)
_c = FermionState("charm", "c", IsospinSector.QUARK,
                   isospin_3=+0.5, hypercharge=1/3)
_s = FermionState("strange", "s", IsospinSector.QUARK,
                   isospin_3=-0.5, hypercharge=1/3)

GENERATION_2 = FermionGeneration(
    generation=2, knot="5_1", torus_param=5,
    neutrino=_nu_mu, charged_lepton=_mu, up_quark=_c, down_quark=_s,
)

# --- Génération 3 : 7_1 = T(2,7) ---

_nu_tau = FermionState("tau neutrino", "ντ", IsospinSector.LEPTON,
                        isospin_3=+0.5, hypercharge=-1.0)
_tau = FermionState("tau", "τ⁻", IsospinSector.LEPTON,
                     isospin_3=-0.5, hypercharge=-1.0)
_t = FermionState("top", "t", IsospinSector.QUARK,
                   isospin_3=+0.5, hypercharge=1/3)
_b = FermionState("bottom", "b", IsospinSector.QUARK,
                   isospin_3=-0.5, hypercharge=1/3)

GENERATION_3 = FermionGeneration(
    generation=3, knot="7_1", torus_param=7,
    neutrino=_nu_tau, charged_lepton=_tau, up_quark=_t, down_quark=_b,
)

FERMION_GENERATIONS = [GENERATION_1, GENERATION_2, GENERATION_3]

# --- Composites (1ère génération) ---

PROTON = CompositeParticle("proton", "p", [_u, _u, _d])
NEUTRON = CompositeParticle("neutron", "n", [_u, _d, _d])
COMPOSITE_PARTICLES = [PROTON, NEUTRON]

# --- Correspondances bosoniques (charge topologique directe) ---

# P14 : le photon est une onde pure (pas de nœud), bosons massifs = nœuds.
# Le gluon n'est PAS une particule : c'est le reliquat d'onde libre du champ χ
# qui n'est pas nodifié autour des nœuds. La « force nucléaire forte » est l'interaction
# via cette onde résiduelle partagée entre nœuds.
BOSON_CORRESPONDENCES = [
    BosonCorrespondence("none", "photon", "γ", 0.0, 0.0,
                        is_pure_wave=True),
    BosonCorrespondence("4_1", "Z boson", "Z⁰", 0.0, 91188.0),
    BosonCorrespondence("8_2", "W+ boson", "W⁺", 1.0, 80379.0),
    BosonCorrespondence("8_2", "W- boson", "W⁻", -1.0, 80379.0,
                        is_mirror=True),
]


def topological_hypercharge(knot_name: str) -> dict:
    """Conjecture l'hypercharge Y depuis la structure des branches du A-polynôme.

    Pour les nœuds toriques T(2,s) (spin 1/2), le A-polynôme factorise en
    une branche abélienne (L=1) et une branche non-abélienne (L = -M^(2s)).
    La conjecture est :
      - Branche abélienne unique → leptons : Y = -1
      - Branche non-abélienne avec 1 racine → quarks : Y = +1/3
        (la structure de la racine en M^(2s) encode 3 couleurs via s=3,5,7)

    ATTENTION : ceci est une CONJECTURE, pas une preuve.
    Les valeurs Y = -1 et Y = +1/3 coïncident avec le SM mais
    ne sont pas rigoureusement dérivées du A-polynôme.

    Returns dict with keys: knot, n_roots, Y_lepton, Y_quark, conjecture_note
    """
    from theory.a_polynomial import a_polynomial_branches, A_POLY_KNOT_NAMES

    result = {
        "knot": knot_name,
        "conjecture_note": "Motivation topologique, pas dérivation rigoureuse",
    }

    if knot_name not in A_POLY_KNOT_NAMES:
        result["n_roots"] = None
        result["Y_lepton"] = -1.0
        result["Y_quark"] = 1 / 3
        result["note"] = "A-polynomial non disponible, valeurs SM utilisées"
        return result

    branches = a_polynomial_branches(knot_name)
    n_roots = branches["n_nonabelian_roots"]
    result["n_roots"] = n_roots

    # Conjecture: branche abélienne → leptons (Y=-1),
    # branche non-abélienne → quarks (Y=+1/3, 3 couleurs)
    result["Y_lepton"] = -1.0
    result["Y_quark"] = 1 / 3

    return result


def generation_table() -> list[dict]:
    """Table résumant les 3 générations × 4 fermions = 12 fermions."""
    rows = []
    for gen in FERMION_GENERATIONS:
        for f in gen.fermions:
            rows.append({
                "generation": gen.generation,
                "knot": gen.knot,
                "torus": f"T(2,{gen.torus_param})",
                "name": f.name,
                "symbol": f.symbol,
                "sector": f.sector.value,
                "I3": f.isospin_3,
                "Y": f.hypercharge,
                "Q_GN": f.charge,
                "spin": f.spin,
            })
    return rows


def verify_gell_mann_nishijima() -> list[dict]:
    """Vérifie Q = I₃ + Y/2 pour chaque fermion et compare au SM."""
    sm_charges = {
        "electron neutrino": 0, "electron": -1, "up": 2/3, "down": -1/3,
        "muon neutrino": 0, "muon": -1, "charm": 2/3, "strange": -1/3,
        "tau neutrino": 0, "tau": -1, "top": 2/3, "bottom": -1/3,
    }
    results = []
    for gen in FERMION_GENERATIONS:
        for f in gen.fermions:
            q_gn = f.charge
            q_sm = sm_charges[f.name]
            results.append({
                "name": f.name,
                "Q_GN": q_gn,
                "Q_SM": q_sm,
                "match": abs(q_gn - q_sm) < 1e-10,
            })
    return results


def composite_charge_table() -> list[dict]:
    """Table des particules composites avec décomposition des charges."""
    rows = []
    for cp in COMPOSITE_PARTICLES:
        rows.append({
            "name": cp.name,
            "symbol": cp.symbol,
            "constituents": " + ".join(f.symbol for f in cp.constituents),
            "charge": cp.charge,
            "baryon_number": cp.baryon_number,
        })
    return rows
