"""
Experience : Probabilite de formation des noeuds

Partie 1 : Table Boltzmann (61 noeuds avec P(K), poids, rang)
Partie 2 : Resume asymptotique (Z_total, coverage, Omega_DM/Omega_b)
Partie 3 : Monte Carlo (probabilite de nouage vs longueur)
Partie 4 : Graphiques dans output/knot_probability/
"""
from __future__ import annotations

import math
import os

import numpy as np

# --- matplotlib setup (Agg backend) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.knot_listing import generate_full_listing
from theory.knot_probability import (
    ALPHA_EFF,
    boltzmann_weight,
    listing_partition_function,
    asymptotic_partition_function,
    normalized_probabilities,
    dark_matter_ratio,
    monte_carlo_knot_statistics,
    knotting_probability_vs_length,
)
from theory.postulates import BETA_TOPO, C_STAR


def main():
    output_dir = "output/knot_probability"
    os.makedirs(output_dir, exist_ok=True)

    # =================================================================
    # Partie 1 : Table Boltzmann
    # =================================================================
    print("=" * 70)
    print("PARTIE 1 : Distribution de Boltzmann topologique (P18)")
    print("=" * 70)

    listing = generate_full_listing(max_crossing=44, n_points=1000)
    lp = listing_partition_function(listing, BETA_TOPO)
    np_result = normalized_probabilities(listing, BETA_TOPO, c_max=500)

    print(f"\nNombre de noeuds dans le listing : {len(listing)}")
    print(f"beta_topo = {BETA_TOPO}")
    print(f"Z_listing = {lp['Z_listing']:.6f}")
    print(f"Coverage (Z_listing/Z_total) = {np_result['coverage']:.4f}")

    # Table triee par poids decroissant
    ranked = sorted(
        zip(listing, lp["weights"], np_result["probabilities"]),
        key=lambda x: -x[1],
    )

    print(f"\n{'Rang':>4} {'Noeud':<12} {'c':>3} {'rho_J':>6} "
          f"{'w(K)':>10} {'P(K)':>12} {'log10 P':>8}")
    print("-" * 65)
    for rank, (knot, w, p) in enumerate(ranked[:20], 1):
        log_p = math.log10(p) if p > 0 else float("-inf")
        print(f"{rank:4d} {knot['name']:<12} {knot['crossing_number']:3d} "
              f"{knot['rho_J']:6.2f} {w:10.6f} {p:12.6e} {log_p:8.2f}")

    print(f"\n... (affichage des 20 premiers sur {len(listing)})")

    # =================================================================
    # Partie 2 : Resume asymptotique
    # =================================================================
    print("\n" + "=" * 70)
    print("PARTIE 2 : Fonction de partition asymptotique")
    print("=" * 70)

    ap = asymptotic_partition_function(BETA_TOPO, c_max=500)
    dm = dark_matter_ratio(BETA_TOPO, c_max=500)

    print(f"\nalpha_ST = 2.68 (Sundberg-Thistlethwaite)")
    print(f"alpha_eff = alpha_ST * exp(-beta * 1.1) = {ALPHA_EFF:.4f}")
    print(f"  => Serie convergente (alpha_eff < 1)")
    print(f"\nZ_total = {ap['Z_total']:.4f}")
    print(f"c dominant = {ap['c_dominant']}")
    print(f"\nSeuil matiere visible/noire : c* = {C_STAR}")
    print(f"P_visible (c <= {C_STAR}) = {np_result['P_visible']:.4f}")
    print(f"P_dark (c > {C_STAR}) = {np_result['P_dark']:.4f}")
    print(f"\nOmega_DM / Omega_b = {dm['ratio']:.3f}")
    print(f"  (Planck 2018 : 5.45 +/- 0.28)")

    # =================================================================
    # Partie 3 : Monte Carlo
    # =================================================================
    print("\n" + "=" * 70)
    print("PARTIE 3 : Monte Carlo polygones aleatoires")
    print("=" * 70)

    print("\nP(noue) vs nombre d'aretes :")
    print(f"{'N':>6} {'P(noue)':>10} {'P(unknot)':>10}")
    print("-" * 30)

    mc_result = knotting_probability_vs_length(
        n_edges_list=[30, 50, 100, 200, 300],
        n_trials=1000,
        seed=42,
    )
    for n, pk in zip(mc_result["n_edges"], mc_result["p_knotted"]):
        print(f"{n:6d} {pk:10.4f} {1-pk:10.4f}")

    print(f"\nFit Deguchi-Tsurusaki : N0 = {mc_result['fit_N0']:.0f}")
    print(f"  (Reference litterature : N0 ~ 300)")

    # Detail pour N = 100
    print("\nDetail Monte Carlo pour N = 100 :")
    stats_100 = monte_carlo_knot_statistics(n_edges=100, n_trials=2000, seed=42)
    print(f"  Essais : {stats_100['n_trials']}")
    print(f"  Unknot : {stats_100['unknot_fraction']:.3f}")
    print(f"  Noue : {stats_100['knotted_fraction']:.3f}")
    print(f"\n  Types de noeuds detectes :")
    for name, frac in list(stats_100["knot_fractions"].items())[:10]:
        print(f"    {name:<15} {frac:.4f}")

    # =================================================================
    # Partie 4 : Graphiques
    # =================================================================
    print("\n" + "=" * 70)
    print("PARTIE 4 : Graphiques")
    print("=" * 70)

    # --- Figure 1 : Barres log P(K) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [k["name"] for k in listing]
    crossings = [k["crossing_number"] for k in listing]
    log_probs = [
        math.log10(p) if p > 0 else -30
        for p in np_result["probabilities"]
    ]
    colors = ["tab:blue" if c <= C_STAR else "tab:red" for c in crossings]
    ax.bar(range(len(names)), log_probs, color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("log10 P(K)")
    ax.set_title("Distribution de Boltzmann topologique P18 — 61 noeuds")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="tab:blue", label=f"Visible (c <= {C_STAR})"),
        Patch(facecolor="tab:red", label=f"Dark (c > {C_STAR})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    path1 = os.path.join(output_dir, "boltzmann_probabilities.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  -> {path1}")

    # --- Figure 2 : P(noue) vs N avec fit ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = mc_result["n_edges"]
    pks = mc_result["p_knotted"]
    ax.plot(ns, pks, "ko-", markersize=8, label="Monte Carlo")
    # Fit exponentiel
    N0 = mc_result["fit_N0"]
    n_fit = np.linspace(10, 600, 200)
    p_fit = 1 - np.exp(-n_fit / N0)
    ax.plot(n_fit, p_fit, "r--", label=f"Fit : 1 - exp(-N/{N0:.0f})")
    ax.set_xlabel("Nombre d'aretes N")
    ax.set_ylabel("P(noue)")
    ax.set_title("Probabilite de nouage — polygones aleatoires")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(output_dir, "knotting_vs_length.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  -> {path2}")

    # --- Figure 3 : Z(c) par nombre de croisements ---
    fig, ax = plt.subplots(figsize=(10, 5))
    Z_by_c = ap["Z_by_crossing"]
    c_range = np.arange(1, len(Z_by_c) + 1)
    # Limiter a c <= 100 pour lisibilite
    mask = c_range <= 100
    ax.semilogy(c_range[mask], Z_by_c[mask], "b-", linewidth=1.5)
    ax.axvline(x=C_STAR, color="red", linestyle="--", alpha=0.7,
               label=f"c* = {C_STAR}")
    ax.set_xlabel("Nombre de croisements c")
    ax.set_ylabel("Z(c) = N(c) * exp(-beta * 1.1c)")
    ax.set_title("Fonction de partition par croisement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(output_dir, "partition_by_crossing.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  -> {path3}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
