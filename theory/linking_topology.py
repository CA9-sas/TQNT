"""Théorie Nodale — Topologie des rencontres onde-onde

Calcule le linking number entre deux ondes circulaires (unknots 0₁) via
l'intégrale de Gauss, et cartographie les topologies résultantes dans
l'espace des paramètres de rencontre (θ, b, φ).

L'intégrale de Gauss est un invariant topologique :
    Lk = 1/(4π) ∮∮ (γ₁−γ₂)·(dγ₁×dγ₂) / |γ₁−γ₂|³

Les topologies possibles :
  - ℓ=0  : non lié (unlinked)
  - |ℓ|=1 : lien de Hopf (entrelacs minimal)
  - |ℓ|=2 : lien de Solomon
  - |ℓ|≥3 : entrelacs supérieurs
"""
from __future__ import annotations

import numpy as np


# ── Intégrale de Gauss ──────────────────────────────────────────────

def gauss_linking_integral(curve1: np.ndarray, curve2: np.ndarray) -> int:
    """Linking number via l'intégrale de Gauss (vectorisée).

    Lk = 1/(4π) ∮∮ (γ₁−γ₂)·(dγ₁×dγ₂) / |γ₁−γ₂|³

    Parameters
    ----------
    curve1 : array (N, 3) — points de la première courbe fermée
    curve2 : array (M, 3) — points de la deuxième courbe fermée

    Returns
    -------
    ℓ ∈ ℤ — linking number (arrondi à l'entier le plus proche)
    """
    # Tangent vectors (finite differences, closed curve)
    dg1 = np.roll(curve1, -1, axis=0) - curve1  # (N, 3)
    dg2 = np.roll(curve2, -1, axis=0) - curve2  # (M, 3)

    # Midpoints of each segment
    mid1 = curve1 + 0.5 * dg1  # (N, 3)
    mid2 = curve2 + 0.5 * dg2  # (M, 3)

    # Difference vectors: r_{ij} = mid1_i - mid2_j  → (N, M, 3)
    r = mid1[:, np.newaxis, :] - mid2[np.newaxis, :, :]

    # |r|³
    r_norm = np.linalg.norm(r, axis=2)  # (N, M)
    r_norm_cubed = r_norm ** 3
    # Avoid division by zero
    r_norm_cubed = np.where(r_norm_cubed < 1e-30, 1e-30, r_norm_cubed)

    # Cross product dγ₁ × dγ₂ → (N, M, 3)
    cross = np.cross(dg1[:, np.newaxis, :], dg2[np.newaxis, :, :])

    # Dot product (γ₁−γ₂) · (dγ₁×dγ₂) → (N, M)
    integrand = np.sum(r * cross, axis=2) / r_norm_cubed

    lk = np.sum(integrand) / (4.0 * np.pi)
    return int(np.round(lk))


# ── Boucle circulaire ──────────────────────────────────────────────

def circular_loop(R: float, center: np.ndarray, normal: np.ndarray,
                  n_points: int = 200) -> np.ndarray:
    """Boucle circulaire de rayon R, centrée en `center`, orientée selon `normal`.

    Parameters
    ----------
    R : rayon du cercle
    center : array (3,) — centre du cercle
    normal : array (3,) — vecteur normal au plan du cercle
    n_points : nombre de points sur le cercle

    Returns
    -------
    array (n_points, 3) — points du cercle
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Build orthonormal basis in the plane perpendicular to normal
    # Pick a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = (center[np.newaxis, :]
              + R * np.cos(t)[:, np.newaxis] * e1[np.newaxis, :]
              + R * np.sin(t)[:, np.newaxis] * e2[np.newaxis, :])
    return points


# ── Géométrie de la rencontre ──────────────────────────────────────

def encounter_geometry(R_nu: float, R_gamma: float, theta: float,
                       b: float, phi: float,
                       n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Construit deux boucles en configuration de rencontre.

    - Boucle 1 (neutrino) : rayon R_nu, dans le plan xy, centrée à l'origine
    - Boucle 2 (photon) : rayon R_gamma, inclinée de θ par rapport à z,
      décalée de b dans le plan, tournée de φ autour de l'axe z

    Parameters
    ----------
    R_nu : rayon de la boucle neutrino
    R_gamma : rayon de la boucle photon
    theta : angle d'inclinaison de la boucle photon [0, π]
    b : paramètre d'impact (décalage dans le plan)
    phi : angle azimutal de rotation [0, 2π]

    Returns
    -------
    (curve1, curve2) : tuple de arrays (N, 3)
    """
    # Boucle 1 : neutrino dans le plan xy
    normal1 = np.array([0.0, 0.0, 1.0])
    center1 = np.array([0.0, 0.0, 0.0])
    curve1 = circular_loop(R_nu, center1, normal1, n_points)

    # Boucle 2 : photon inclinée et décalée
    # Normal de la boucle 2 : inclinée de theta par rapport à z, dans le plan xz
    normal2 = np.array([np.sin(theta), 0.0, np.cos(theta)])

    # Centre décalé de b dans la direction (cos(phi), sin(phi), 0)
    center2 = np.array([b * np.cos(phi), b * np.sin(phi), 0.0])

    curve2 = circular_loop(R_gamma, center2, normal2, n_points)
    return curve1, curve2


# ── Classification de la rencontre ─────────────────────────────────

def classify_encounter(R_nu: float, R_gamma: float, theta: float,
                       b: float, phi: float) -> dict:
    """Calcule le linking number et classifie la topologie de la rencontre.

    Parameters
    ----------
    R_nu, R_gamma : rayons des deux boucles
    theta : angle d'inclinaison [0, π]
    b : paramètre d'impact
    phi : angle azimutal [0, 2π]

    Returns
    -------
    dict avec clés : linking_number, linked, topology
    """
    curve1, curve2 = encounter_geometry(R_nu, R_gamma, theta, b, phi)
    lk = gauss_linking_integral(curve1, curve2)
    abs_lk = abs(lk)

    if abs_lk == 0:
        topology = "unlinked"
    elif abs_lk == 1:
        topology = "hopf"
    elif abs_lk == 2:
        topology = "solomon"
    else:
        topology = "higher"

    return {
        "linking_number": lk,
        "linked": abs_lk > 0,
        "topology": topology,
    }


# ── Scan de l'espace des paramètres ────────────────────────────────

def scan_encounter_space(R_nu: float, R_gamma: float,
                         n_theta: int = 50, n_b: int = 50,
                         n_phi: int = 12) -> dict:
    """Scan sur la grille (θ, b, φ) et statistiques de linking.

    Parameters
    ----------
    R_nu, R_gamma : rayons des deux boucles
    n_theta : nombre de points en θ ∈ [0, π]
    n_b : nombre de points en b ∈ [0, 3·max(R)]
    n_phi : nombre de points en φ ∈ [0, 2π)

    Returns
    -------
    dict avec clés :
      - theta_grid, b_grid, phi_grid : arrays 1D des valeurs scannées
      - linking_map : array (n_theta, n_b, n_phi) de linking numbers
      - fraction_linked : fraction de configurations avec ℓ ≠ 0
      - distribution : dict {|ℓ| : count}
    """
    R_max = max(R_nu, R_gamma)
    b_max = 3.0 * R_max

    theta_grid = np.linspace(0, np.pi, n_theta)
    b_grid = np.linspace(0, b_max, n_b)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    linking_map = np.zeros((n_theta, n_b, n_phi), dtype=int)

    for i, theta in enumerate(theta_grid):
        for j, b_val in enumerate(b_grid):
            for k, phi_val in enumerate(phi_grid):
                curve1, curve2 = encounter_geometry(
                    R_nu, R_gamma, theta, b_val, phi_val, n_points=200
                )
                linking_map[i, j, k] = gauss_linking_integral(curve1, curve2)

    total = linking_map.size
    abs_map = np.abs(linking_map)
    fraction_linked = float(np.count_nonzero(abs_map)) / total

    # Distribution of |ℓ|
    unique, counts = np.unique(abs_map, return_counts=True)
    distribution = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "theta_grid": theta_grid,
        "b_grid": b_grid,
        "phi_grid": phi_grid,
        "linking_map": linking_map,
        "fraction_linked": fraction_linked,
        "distribution": distribution,
    }


# ── Section efficace effective ─────────────────────────────────────

def linking_cross_section_effective(R_nu: float, R_gamma: float,
                                    n_samples: int = 5000) -> float:
    """Section efficace effective par Monte Carlo sur θ, b, φ.

    σ_eff = (fraction liée) × π·(R_nu + R_gamma)²

    On échantillonne uniformément θ ∈ [0, π] (pondéré par sin θ),
    b ∈ [0, b_max], φ ∈ [0, 2π).

    Parameters
    ----------
    R_nu, R_gamma : rayons des deux boucles
    n_samples : nombre de configurations Monte Carlo

    Returns
    -------
    σ_eff (mêmes unités que R²)
    """
    R_max = max(R_nu, R_gamma)
    b_max = 3.0 * R_max

    rng = np.random.default_rng(42)

    # Uniform sampling with proper measure
    # θ uniform in [0, π] (we weight by sin θ for solid angle)
    theta = np.arccos(1.0 - 2.0 * rng.random(n_samples))  # uniform on sphere
    # Clip to [0, π]
    theta = np.clip(theta, 0, np.pi)
    b = b_max * np.sqrt(rng.random(n_samples))  # uniform in area: P(b) ∝ b
    phi = 2.0 * np.pi * rng.random(n_samples)

    n_linked = 0
    for i in range(n_samples):
        curve1, curve2 = encounter_geometry(
            R_nu, R_gamma, theta[i], b[i], phi[i], n_points=200
        )
        lk = gauss_linking_integral(curve1, curve2)
        if lk != 0:
            n_linked += 1

    fraction = n_linked / n_samples
    sigma_geom = np.pi * (R_nu + R_gamma) ** 2
    return fraction * sigma_geom
