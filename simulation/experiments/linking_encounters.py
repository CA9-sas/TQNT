"""
Théorie Nodale — Rencontres onde-onde : topologies résultantes

Simulation Monte Carlo des encounters entre deux ondes circulaires (unknots 0₁).
L'intégrale de Gauss donne le linking number pour chaque configuration (θ, b, φ).

Génère 4 figures dans output/ :
  1. Carte de linking number (θ, b/R) pour φ=0
  2. Distribution des topologies (histogramme)
  3. Section efficace effective vs ratio R_γ/R_ν
  4. Visualisations 3D de rencontres typiques
"""
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from theory.linking_topology import (
    gauss_linking_integral,
    circular_loop,
    encounter_geometry,
    classify_encounter,
    scan_encounter_space,
    linking_cross_section_effective,
)


def plot_linking_map(R_nu, R_gamma, n_theta=50, n_b=50):
    """Figure 1 : Carte de linking number (θ, b/R) pour φ fixe."""
    R_max = max(R_nu, R_gamma)
    b_max = 3.0 * R_max

    theta_grid = np.linspace(0, np.pi, n_theta)
    b_grid = np.linspace(0, b_max, n_b)

    # Scan pour plusieurs valeurs de φ, moyenne des |ℓ|
    phi_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lk_map = np.zeros((n_theta, n_b))

    for phi in phi_values:
        for i, theta in enumerate(theta_grid):
            for j, b_val in enumerate(b_grid):
                curve1, curve2 = encounter_geometry(
                    R_nu, R_gamma, theta, b_val, phi, n_points=200
                )
                lk_map[i, j] += abs(gauss_linking_integral(curve1, curve2))
    lk_map /= len(phi_values)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(
        b_grid / R_max, np.degrees(theta_grid), lk_map,
        cmap="RdYlBu_r", shading="auto", vmin=0, vmax=2,
    )
    ax.set_xlabel("b / R_max", fontsize=13)
    ax.set_ylabel("θ (degrés)", fontsize=13)
    ax.set_title(
        f"Linking number moyen |ℓ| — R_γ/R_ν = {R_gamma / R_nu:.1f}",
        fontsize=14,
    )
    cb = fig.colorbar(im, ax=ax, label="|ℓ| moyen")
    fig.tight_layout()
    fig.savefig("output/linking_encounter_map.png", dpi=150)
    plt.close(fig)
    print("  → output/linking_encounter_map.png")


def plot_topology_distribution(R_nu, R_gamma, n_theta=50, n_b=50, n_phi=12):
    """Figure 2 : Distribution des topologies."""
    result = scan_encounter_space(R_nu, R_gamma, n_theta, n_b, n_phi)

    dist = result["distribution"]
    total = sum(dist.values())

    labels = []
    counts = []
    for lk_val in sorted(dist.keys()):
        if lk_val == 0:
            labels.append("Non lié\n(ℓ=0)")
        elif lk_val == 1:
            labels.append("Hopf\n(|ℓ|=1)")
        elif lk_val == 2:
            labels.append("Solomon\n(|ℓ|=2)")
        else:
            labels.append(f"|ℓ|={lk_val}")
        counts.append(dist[lk_val])

    fractions = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#cccccc", "#4a90d9", "#e74c3c", "#2c3e50"]
    bars = ax.bar(
        range(len(labels)), fractions,
        color=colors[: len(labels)], edgecolor="black", linewidth=0.8,
    )

    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{frac:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Fraction (%)", fontsize=13)
    ax.set_title(
        f"Distribution des topologies — R_γ/R_ν = {R_gamma / R_nu:.1f}",
        fontsize=14,
    )
    ax.set_ylim(0, max(fractions) * 1.2)
    fig.tight_layout()
    fig.savefig("output/linking_topology_distribution.png", dpi=150)
    plt.close(fig)
    print("  → output/linking_topology_distribution.png")

    return result


def plot_sigma_vs_ratio(R_nu_base=1.0, n_ratios=20, n_samples=2000):
    """Figure 3 : Section efficace effective vs ratio R_γ/R_ν."""
    ratios = np.logspace(-1, 1, n_ratios)
    sigma_norm = np.zeros(n_ratios)

    for i, ratio in enumerate(ratios):
        R_gamma = R_nu_base * ratio
        sigma = linking_cross_section_effective(
            R_nu_base, R_gamma, n_samples=n_samples,
        )
        sigma_geom = np.pi * (R_nu_base + R_gamma) ** 2
        sigma_norm[i] = sigma / sigma_geom if sigma_geom > 0 else 0
        print(f"    ratio={ratio:.2f}  σ_eff/σ_geom={sigma_norm[i]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(ratios, sigma_norm, "o-", color="#e74c3c", linewidth=2, markersize=6)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="R_γ = R_ν")
    ax.set_xlabel("R_γ / R_ν", fontsize=13)
    ax.set_ylabel("σ_eff / π(R₁+R₂)²", fontsize=13)
    ax.set_title(
        "Section efficace effective normalisée vs ratio de rayons",
        fontsize=14,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("output/linking_sigma_vs_ratio.png", dpi=150)
    plt.close(fig)
    print("  → output/linking_sigma_vs_ratio.png")


def plot_encounter_3d():
    """Figure 4 : Visualisations 3D de rencontres typiques."""
    configs = [
        {"label": "Non lié (ℓ=0)", "theta": np.pi / 2, "b": 3.0, "phi": 0.0},
        {"label": "Hopf (|ℓ|=1)", "theta": np.pi / 2, "b": 1.0, "phi": np.pi / 2},
        {"label": "Solomon (|ℓ|=2)", "theta": np.pi / 4, "b": 0.3, "phi": np.pi / 4},
        {"label": "Triple (|ℓ|≥3)", "theta": np.pi / 6, "b": 0.2, "phi": np.pi / 3},
    ]

    fig = plt.figure(figsize=(14, 12))

    for idx, cfg in enumerate(configs):
        c1, c2 = encounter_geometry(
            1.0, 1.0, cfg["theta"], cfg["b"], cfg["phi"], n_points=200
        )
        lk = gauss_linking_integral(c1, c2)

        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        # Close the curves for plotting
        c1_closed = np.vstack([c1, c1[0]])
        c2_closed = np.vstack([c2, c2[0]])
        ax.plot(c1_closed[:, 0], c1_closed[:, 1], c1_closed[:, 2],
                color="#4a90d9", linewidth=2.5, label="ν (neutrino)")
        ax.plot(c2_closed[:, 0], c2_closed[:, 1], c2_closed[:, 2],
                color="#e74c3c", linewidth=2.5, label="γ (photon)")
        ax.set_title(f"{cfg['label']}\nℓ = {lk}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Set equal aspect
        all_pts = np.vstack([c1, c2])
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
        mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.suptitle(
        "Rencontres onde-onde : topologies résultantes",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("output/linking_encounter_3d.png", dpi=150)
    plt.close(fig)
    print("  → output/linking_encounter_3d.png")


def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 72)
    print("  RENCONTRES ONDE-ONDE : TOPOLOGIES RÉSULTANTES")
    print("=" * 72)

    R_nu = 1.0
    R_gamma = 1.0

    # ── Figure 1 : Carte de linking number ──────────────────────────
    print("\n  [1/4] Carte de linking number (θ, b/R)...")
    plot_linking_map(R_nu, R_gamma, n_theta=50, n_b=50)

    # ── Figure 2 : Distribution des topologies ──────────────────────
    print("\n  [2/4] Distribution des topologies...")
    result = plot_topology_distribution(R_nu, R_gamma, n_theta=50, n_b=50, n_phi=12)

    # ── Tableau récapitulatif ───────────────────────────────────────
    dist = result["distribution"]
    total = sum(dist.values())
    print("\n  " + "─" * 40)
    print(f"  {'Topologie':<20} {'Count':>8} {'Fraction':>10}")
    print("  " + "─" * 40)
    for lk_val in sorted(dist.keys()):
        if lk_val == 0:
            name = "Non lié (ℓ=0)"
        elif lk_val == 1:
            name = "Hopf (|ℓ|=1)"
        elif lk_val == 2:
            name = "Solomon (|ℓ|=2)"
        else:
            name = f"|ℓ|={lk_val}"
        print(f"  {name:<20} {dist[lk_val]:>8d} {dist[lk_val] / total:>9.1%}")
    print("  " + "─" * 40)
    print(f"  Fraction liée totale : {result['fraction_linked']:.1%}")

    # ── Figure 3 : σ_eff vs ratio ──────────────────────────────────
    print("\n  [3/4] Section efficace vs ratio R_γ/R_ν...")
    plot_sigma_vs_ratio(R_nu, n_ratios=15, n_samples=1000)

    # ── Section efficace effective pour R_γ = R_ν ──────────────────
    sigma_eff = linking_cross_section_effective(R_nu, R_gamma, n_samples=2000)
    sigma_geom = np.pi * (R_nu + R_gamma) ** 2
    print(f"\n  σ_eff (R_γ=R_ν) = {sigma_eff:.4f} R²")
    print(f"  σ_geom = π(R₁+R₂)² = {sigma_geom:.4f} R²")
    print(f"  σ_eff / σ_geom = {sigma_eff / sigma_geom:.4f}")

    # ── Figure 4 : Visualisations 3D ──────────────────────────────
    print("\n  [4/4] Visualisations 3D de rencontres typiques...")
    plot_encounter_3d()

    print("\n" + "=" * 72)
    print("  Terminé — 4 figures dans output/")
    print("=" * 72)


if __name__ == "__main__":
    main()
