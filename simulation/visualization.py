# simulation/visualization.py
"""
Théorie Nodale — Visualisation

Plotting utilities for nodal wave theory:
- Radial wave functions and their nodes
- Spherical harmonic angular distributions
- Spacetime density maps
- Mass spectrum charts
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory.wave_functions import radial_function, spherical_harmonic, count_radial_nodes
from theory.nodal_classification import mass_spectrum, PARTICLE_CATALOG
from theory.emergent_spacetime import spacetime_density, emergent_metric_factor
from theory.postulates import KAPPA


def plot_radial_wave(n: int, l: int, r_max: float = 20.0, save_path: str | None = None):
    """Plot radial wave function R_n,l(r) showing nodes."""
    r = np.linspace(0.01, r_max, 2000)
    R = radial_function(r, n, l, r_max)
    nodes = count_radial_nodes(r, R)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Wave function
    ax1.plot(r, R, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.set_title(f"Fonction radiale R(r) — n={n}, l={l} — {nodes} noeud(s)")
    ax1.set_xlabel("r")
    ax1.set_ylabel("R(r)")
    ax1.grid(True, alpha=0.3)

    # Mark nodes
    sign_changes = np.where(np.diff(np.sign(R)))[0]
    for idx in sign_changes:
        ax1.axvline(x=r[idx], color='r', linestyle='--', alpha=0.7, label="noeud" if idx == sign_changes[0] else "")
    if len(sign_changes) > 0:
        ax1.legend()

    # Density |R|²r²
    density = np.abs(R) ** 2 * r**2
    ax2.plot(r, density, 'r-', linewidth=1.5)
    ax2.fill_between(r, density, alpha=0.3, color='red')
    ax2.set_title(f"Densité radiale |R(r)|² r²")
    ax2.set_xlabel("r")
    ax2.set_ylabel("|R|² r²")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_angular_distribution(l: int, m: int, save_path: str | None = None):
    """Plot |Y_l^m|² angular distribution in polar coordinates."""
    theta = np.linspace(0, np.pi, 300)
    phi_fixed = 0.0
    phi_arr = np.full_like(theta, phi_fixed)

    Y = spherical_harmonic(theta, phi_arr, l, m)
    prob = np.abs(Y) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(theta, prob, 'b-', linewidth=2)
    ax.plot(-theta, prob, 'b-', linewidth=2)
    ax.set_title(f"Distribution angulaire |Y_{l}^{m}|²", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mass_spectrum(save_path: str | None = None):
    """Plot predicted mass spectrum of nodal particles."""
    spectrum = mass_spectrum()
    names = [s[0] for s in spectrum]
    masses = [s[1] for s in spectrum]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if 'neutral' in n or 'neutrino' in n or 'photon' in n
              else 'blue' if '-' in n or 'electron' in n
              else 'red'
              for n in names]
    bars = ax.barh(range(len(names)), masses, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Masse nodale (unités naturelles)")
    ax.set_title("Spectre de masse — Théorie Nodale")
    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Neutre (m=0)'),
        Patch(facecolor='blue', alpha=0.8, label='Charge - (m<0)'),
        Patch(facecolor='red', alpha=0.8, label='Charge + (m>0)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_emergent_spacetime(n: int, l: int, m: int, r_max: float = 15.0, save_path: str | None = None):
    """Plot emergent spacetime density and metric around a particle."""
    from theory.wave_functions import nodal_wave

    r = np.linspace(0.01, r_max, 1000)
    theta = np.full_like(r, np.pi / 2)
    phi = np.zeros_like(r)

    psi = nodal_wave(r, theta, phi, t=0.0, n=n, l=l, m=m, r_max=r_max)
    rho = spacetime_density(psi)
    metric = emergent_metric_factor(psi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(r, rho, 'purple', linewidth=1.5)
    ax1.fill_between(r, rho, alpha=0.2, color='purple')
    ax1.set_title(f"Densité d'espace-temps |Ψ|² — particule ({n},{l},{m})")
    ax1.set_xlabel("r")
    ax1.set_ylabel("|Ψ|²")
    ax1.grid(True, alpha=0.3)

    ax2.plot(r, metric, 'darkred', linewidth=1.5)
    ax2.axhline(y=1.0, color='k', linewidth=0.5, linestyle='--', label="métrique plate")
    ax2.set_title(f"Facteur métrique (1 + κ|Ψ|²) — courbure émergente")
    ax2.set_xlabel("r")
    ax2.set_ylabel("g/η")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Phase 2: Schrödinger solver visualization ---

def plot_density_cut(result: dict, state_index: int = 0,
                     output_path: str | None = None, z_cut: float = 0.0):
    """Plot |Psi|^2 in a 2D cut at z = z_cut."""
    points = result["points"]
    boundary = result["boundary"]
    free_mask = ~boundary
    eigvec = result["eigenvectors"][:, state_index]

    psi_full = np.zeros(len(points))
    psi_full[free_mask] = eigvec
    density = psi_full**2

    tol = np.ptp(points[:, 2]) / 20
    near_z = np.abs(points[:, 2] - z_cut) < tol
    if near_z.sum() < 10:
        tol *= 3
        near_z = np.abs(points[:, 2] - z_cut) < tol

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(points[near_z, 0], points[near_z, 1],
                    c=density[near_z], cmap='hot', s=10, edgecolors='none')
    plt.colorbar(sc, ax=ax, label="|Ψ|²")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{result['knot_name']} — |Ψ|² (état {state_index}, z≈{z_cut:.1f})")
    ax.set_aspect('equal')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_energy_spectrum(result: dict, output_path: str | None = None):
    """Plot the energy spectrum (bar chart of eigenvalues)."""
    eigs = result["eigenvalues"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(eigs)), eigs, color='steelblue')
    ax.set_xlabel("État n")
    ax.set_ylabel("Énergie E_n")
    ax.set_title(f"{result['knot_name']} — Spectre d'énergie ({len(eigs)} niveaux)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_E0_rho(results: list[dict], output_path: str | None = None):
    """Plot E_0 vs retraction factor for multiple knots."""
    rhos = [r["retraction_factor"] for r in results]
    E0s = [r["eigenvalues"][0] for r in results]
    names = [r["knot_name"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rhos, E0s, s=60, zorder=5)
    for name, rho, e0 in zip(names, rhos, E0s):
        ax.annotate(name, (rho, e0), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    if len(rhos) > 2:
        coeffs = np.polyfit(rhos, E0s, 1)
        x_fit = np.linspace(min(rhos), max(rhos), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', alpha=0.7,
                label=f"Régression: E₀ = {coeffs[0]:.2f}ρ + {coeffs[1]:.2f}")
        ss_res = sum((e - np.polyval(coeffs, r))**2 for r, e in zip(rhos, E0s))
        ss_tot = sum((e - np.mean(E0s))**2 for e in E0s)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.legend(title=f"R² = {r_sq:.3f}")

    ax.set_xlabel("Facteur de rétractation ρ(K)")
    ax.set_ylabel("Énergie fondamentale E₀")
    ax.set_title("Corrélation E₀ vs ρ(K)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_3d_isosurface(result: dict, state_index: int = 0,
                        threshold_pct: float = 0.3,
                        output_path: str | None = None):
    """Plot 3D scatter of |Psi|^2 above threshold, with knot curve overlay."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    points = result["points"]
    boundary = result["boundary"]
    free_mask = ~boundary
    eigvec = result["eigenvectors"][:, state_index]

    psi_full = np.zeros(len(points))
    psi_full[free_mask] = eigvec
    density = psi_full**2

    threshold = threshold_pct * np.max(density)
    mask = density > threshold
    if mask.sum() == 0:
        mask = density > 0.01 * np.max(density)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    d_masked = density[mask]
    d_norm = d_masked / d_masked.max() if d_masked.max() > 0 else d_masked

    ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
               c=d_norm, cmap='hot', s=5, alpha=0.5, edgecolors='none')

    from theory.knot_catalog import knot_curve
    try:
        curve = knot_curve(result["knot_name"], 300)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-',
                linewidth=2, alpha=0.8, label="Noeud K")
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{result['knot_name']} — Isosurface |Ψ|² (état {state_index})")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close(fig)
