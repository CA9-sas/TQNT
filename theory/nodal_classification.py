"""
Theorie Nodale -- Classification des Particules

Each particle is defined by a nodal triplet (n, l, m).
"""
from dataclasses import dataclass
from theory.postulates import nodal_mass, nodal_energy, nodal_charge


@dataclass
class NodalParticle:
    name: str
    n: int
    l: int
    m: int

    @property
    def mass(self) -> float:
        return nodal_mass(self.n, self.l)

    @property
    def energy(self) -> float:
        return nodal_energy(self.n, self.l)

    @property
    def charge(self) -> float:
        return nodal_charge(self.m)

    @property
    def nodal_signature(self) -> str:
        return f"({self.n},{self.l},{self.m})"


PARTICLE_CATALOG = [
    NodalParticle("neutrino", n=1, l=0, m=0),
    NodalParticle("electron", n=1, l=1, m=-1),
    NodalParticle("positron", n=1, l=1, m=1),
    NodalParticle("photon-like", n=1, l=1, m=0),
    NodalParticle("muon-", n=2, l=1, m=-1),
    NodalParticle("muon+", n=2, l=1, m=1),
    NodalParticle("heavy-neutral", n=2, l=0, m=0),
    NodalParticle("quark-like", n=3, l=1, m=-1),
    NodalParticle("quark-like+", n=3, l=1, m=1),
    NodalParticle("massive-neutral", n=3, l=2, m=0),
]


def classify_particle(n: int, l: int, m: int) -> NodalParticle:
    for p in PARTICLE_CATALOG:
        if p.n == n and p.l == l and p.m == m:
            return p
    return NodalParticle(f"unknown-{n}{l}{m}", n=n, l=l, m=m)


def mass_spectrum() -> list[tuple[str, float]]:
    return sorted(
        [(p.nodal_signature + " " + p.name, p.mass) for p in PARTICLE_CATALOG],
        key=lambda x: x[1],
    )
