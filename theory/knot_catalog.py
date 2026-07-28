"""
Théorie Nodale v2 — Catalogue de Noeuds

Parametric curves for knots (3D embeddings) and a summary info table.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from theory.jones import (
    JONES_KNOT_NAMES as KNOT_NAMES,
    jones_polynomial, jones_determinant, jones_span, poly_to_str,
    braid_index, topological_spin, topological_charge,
)
from theory.retraction import retraction_factor, knot_mass


def _unknot_curve(n_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points)
    return np.column_stack([np.cos(t), np.sin(t), np.zeros(n_points)])


def _trefoil_curve(n_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.column_stack([x, y, z])


def _figure_eight_curve(n_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points)
    x = (2 + np.cos(2 * t)) * np.cos(3 * t)
    y = (2 + np.cos(2 * t)) * np.sin(3 * t)
    z = np.sin(4 * t)
    return np.column_stack([x, y, z])


def _torus_knot_curve(p: int, q: int, n_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points)
    r = 2 + np.cos(q * t)
    x = r * np.cos(p * t)
    y = r * np.sin(p * t)
    z = -np.sin(q * t)
    return np.column_stack([x, y, z])


def _harmonic_curve(coeffs: dict, n_points: int) -> np.ndarray:
    """Courbe harmonique de Fourier paramétrique.

    *coeffs* contient les clés 'x', 'y', 'z', chacune étant une liste
    de tuples ``(amplitude, fréquence, phase)``.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    xyz = np.zeros((n_points, 3))
    for axis_idx, key in enumerate(("x", "y", "z")):
        for amp, freq, phase in coeffs[key]:
            xyz[:, axis_idx] += amp * np.cos(freq * t + phase) if phase != 0 else amp * np.cos(freq * t)
            # Utiliser sin quand la phase est np.pi/2 (optimisation lisibilité)
    return xyz


def _fourier_curve(x_terms, y_terms, z_terms, n_points: int) -> np.ndarray:
    """Courbe harmonique — chaque terme est (amp, freq, phase).

    x(t) = Σ amp·sin(freq·t + phase)  (sin pour x/y, sin pour z)
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = sum(a * np.sin(f * t + p) for a, f, p in x_terms)
    y = sum(a * np.cos(f * t + p) for a, f, p in y_terms)
    z = sum(a * np.sin(f * t + p) for a, f, p in z_terms)
    return np.column_stack([x, y, z])


# --- Courbes harmoniques distinctes pour les 13 nœuds non-toriques ---

def _curve_6_1(n: int) -> np.ndarray:
    """Nœud 6_1 (stevedore) — courbe harmonique à 3 fréquences."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (2.0, 2, 0), (0.5, 3, np.pi/4)],
        y_terms=[(1.0, 1, 0), (-2.0, 2, 0), (0.5, 3, np.pi/3)],
        z_terms=[(-1.0, 3, 0), (0.8, 4, 0)],
        n_points=n,
    )


def _curve_6_2(n: int) -> np.ndarray:
    """Nœud 6_2 — harmoniques décalées en phase."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (1.5, 2, 0), (1.0, 3, np.pi/6)],
        y_terms=[(1.0, 1, 0), (-1.5, 2, 0), (0.5, 3, np.pi/2)],
        z_terms=[(1.0, 3, 0), (-0.5, 5, 0)],
        n_points=n,
    )


def _curve_6_3(n: int) -> np.ndarray:
    """Nœud 6_3 (amphichiral) — paramétrisation symétrique."""
    return _fourier_curve(
        x_terms=[(2.0, 1, np.pi/2), (1.0, 2, np.pi/2), (0.5, 3, np.pi/2)],
        y_terms=[(2.0, 1, 0), (-1.0, 2, 0), (0.5, 3, 0)],
        z_terms=[(1.0, 3, 0), (-1.0, 4, 0)],
        n_points=n,
    )


def _curve_7_2(n: int) -> np.ndarray:
    """Nœud 7_2 — harmoniques 1,2,3,4."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (2.0, 2, 0), (0.3, 4, np.pi/5)],
        y_terms=[(1.0, 1, 0), (-2.0, 2, 0), (0.3, 4, np.pi/4)],
        z_terms=[(-1.0, 3, 0), (0.5, 5, 0)],
        n_points=n,
    )


def _curve_7_3(n: int) -> np.ndarray:
    """Nœud 7_3 — amplitude modulée."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (1.8, 2, 0), (0.6, 3, np.pi/3)],
        y_terms=[(1.0, 1, 0), (-1.8, 2, 0), (0.3, 3, np.pi/6)],
        z_terms=[(-1.0, 3, 0), (0.4, 6, 0)],
        n_points=n,
    )


def _curve_7_4(n: int) -> np.ndarray:
    """Nœud 7_4 — fréquences 1,2,3 renforcées."""
    return _fourier_curve(
        x_terms=[(1.5, 1, 0), (1.0, 2, 0), (0.8, 3, np.pi/4)],
        y_terms=[(1.5, 1, 0), (-1.0, 2, 0), (0.4, 3, np.pi/3)],
        z_terms=[(1.0, 4, 0), (-0.6, 5, 0)],
        n_points=n,
    )


def _curve_7_5(n: int) -> np.ndarray:
    """Nœud 7_5 — composante z complexe."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (2.0, 2, 0), (0.5, 4, np.pi/6)],
        y_terms=[(1.0, 1, 0), (-2.0, 2, 0), (0.3, 4, np.pi/4)],
        z_terms=[(-1.0, 3, 0), (0.7, 5, np.pi/2)],
        n_points=n,
    )


def _curve_7_6(n: int) -> np.ndarray:
    """Nœud 7_6 — dominante basse fréquence."""
    return _fourier_curve(
        x_terms=[(2.0, 1, 0), (0.4, 2, np.pi/3), (1.0, 3, np.pi/2)],
        y_terms=[(2.0, 1, 0), (0.4, 2, np.pi/4), (-1.0, 3, 0)],
        z_terms=[(1.0, 2, 0), (-0.5, 5, 0)],
        n_points=n,
    )


def _curve_7_7(n: int) -> np.ndarray:
    """Nœud 7_7 — le plus complexe des 7-croisements."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (1.5, 2, 0), (1.0, 3, np.pi/2)],
        y_terms=[(1.0, 1, 0), (-1.5, 2, 0), (-1.0, 3, 0)],
        z_terms=[(-1.0, 4, 0), (0.5, 5, 0)],
        n_points=n,
    )


def _curve_8_1(n: int) -> np.ndarray:
    """Nœud 8_1 — 4 harmoniques en x/y, 3 en z."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (2.0, 2, 0), (0.3, 3, np.pi/5), (0.2, 4, np.pi/3)],
        y_terms=[(1.0, 1, 0), (-2.0, 2, 0), (0.3, 3, np.pi/4), (0.2, 4, np.pi/6)],
        z_terms=[(-1.0, 3, 0), (0.5, 5, 0), (-0.3, 7, 0)],
        n_points=n,
    )


def _curve_8_2(n: int) -> np.ndarray:
    """Nœud 8_2 — harmoniques équilibrées."""
    return _fourier_curve(
        x_terms=[(1.5, 1, 0), (1.5, 2, 0), (0.5, 4, np.pi/4)],
        y_terms=[(1.5, 1, 0), (-1.5, 2, 0), (0.5, 4, np.pi/3)],
        z_terms=[(1.0, 3, 0), (-0.4, 5, 0), (0.3, 6, np.pi/2)],
        n_points=n,
    )


def _curve_8_3(n: int) -> np.ndarray:
    """Nœud 8_3 (amphichiral) — paramétrisation symétrique haute fréquence."""
    return _fourier_curve(
        x_terms=[(2.0, 1, np.pi/2), (1.0, 3, np.pi/2), (0.4, 5, np.pi/2)],
        y_terms=[(2.0, 1, 0), (-1.0, 3, 0), (0.4, 5, 0)],
        z_terms=[(1.0, 2, 0), (-1.0, 4, 0), (0.3, 6, 0)],
        n_points=n,
    )


def _curve_10_1(n: int) -> np.ndarray:
    """Nœud 10_1 — richesse harmonique maximale (4 fréquences en z)."""
    return _fourier_curve(
        x_terms=[(1.0, 1, 0), (2.0, 2, 0), (0.4, 3, np.pi/5), (0.2, 5, np.pi/4)],
        y_terms=[(1.0, 1, 0), (-2.0, 2, 0), (0.4, 3, np.pi/4), (0.2, 5, np.pi/3)],
        z_terms=[(-1.0, 3, 0), (0.5, 5, 0), (-0.3, 7, 0), (0.2, 9, 0)],
        n_points=n,
    )


_CURVE_REGISTRY: dict[str, callable] = {
    "0_1": lambda n: _unknot_curve(n),
    "3_1": lambda n: _trefoil_curve(n),
    "4_1": lambda n: _figure_eight_curve(n),
    "5_1": lambda n: _torus_knot_curve(2, 5, n),
    "5_2": lambda n: _torus_knot_curve(3, 2, n),
    "6_1": lambda n: _curve_6_1(n),
    "6_2": lambda n: _curve_6_2(n),
    "6_3": lambda n: _curve_6_3(n),
    "7_1": lambda n: _torus_knot_curve(2, 7, n),
    "7_2": lambda n: _curve_7_2(n),
    "7_3": lambda n: _curve_7_3(n),
    "7_4": lambda n: _curve_7_4(n),
    "7_5": lambda n: _curve_7_5(n),
    "7_6": lambda n: _curve_7_6(n),
    "7_7": lambda n: _curve_7_7(n),
    "8_1": lambda n: _curve_8_1(n),
    "8_2": lambda n: _curve_8_2(n),
    "8_3": lambda n: _curve_8_3(n),
    "9_1": lambda n: _torus_knot_curve(2, 9, n),
    "10_1": lambda n: _curve_10_1(n),
}


def knot_curve(knot_name: str, n_points: int = 500) -> np.ndarray:
    """Retourne la courbe 3D du nœud *knot_name*.

    Les 20 nœuds du catalogue possèdent chacun une paramétrisation
    harmonique distincte. Le fallback T(2,c) n'est conservé que pour
    d'éventuels nœuds ajoutés sans courbe explicite.
    """
    if knot_name in _CURVE_REGISTRY:
        return _CURVE_REGISTRY[knot_name](n_points)
    crossings = int(knot_name.split("_")[0])
    return _torus_knot_curve(2, crossings, n_points)


def distance_to_knot(point: np.ndarray, curve: np.ndarray) -> float:
    diffs = curve - point[np.newaxis, :]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.min(dists))


def crossing_number(knot_name: str) -> int:
    """Nombre de croisements du nœud (extrait du nom).

    Pour les nœuds alternants du catalogue, c = span du Jones polynomial.
    """
    return int(knot_name.split("_")[0])


def arc_length(knot_name: str, n_points: int = 2000) -> float:
    """Longueur d'arc de la courbe paramétrique du nœud.

    Somme des longueurs de segments sur n_points échantillons.
    """
    pts = knot_curve(knot_name, n_points)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def construction_wavelength(knot_name: str) -> float:
    """Longueur d'onde maximale de construction (critère de Nyquist topologique).

    λ_max(K) = L(K) / (2·c(K))

    Au-delà de cette longueur d'onde, le champ χ ne peut pas résoudre
    les c croisements du nœud. Retourne float('inf') pour l'unknot (c=0).
    """
    c = crossing_number(knot_name)
    if c == 0:
        return float('inf')
    L = arc_length(knot_name)
    return L / (2 * c)


@dataclass
class KnotInfo:
    name: str
    degree: int
    determinant: int
    retraction: float
    mass: float
    polynomial_str: str
    braid_idx: int = 0
    spin: float = 0.0
    charge: float = 0.0


def knot_info_table() -> list[KnotInfo]:
    infos = []
    for name in KNOT_NAMES:
        poly = jones_polynomial(name)
        deg = jones_span(poly)
        det = jones_determinant(poly)
        rho = retraction_factor(poly)
        m = knot_mass(poly)
        pstr = poly_to_str(poly)
        b = braid_index(name)
        s = topological_spin(name)
        q = topological_charge(name)
        infos.append(KnotInfo(name, deg, det, rho, m, pstr, b, s, q))
    return sorted(infos, key=lambda k: k.retraction)
