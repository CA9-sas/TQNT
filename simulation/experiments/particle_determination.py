# simulation/experiments/particle_determination.py
"""
Expérience : Détermination topologique du spin et de la charge.

Dérive spin = (b(K)-1)/2 et charge Q = (p_min+p_max)/span(V_K) pour les 20 nœuds,
vérifie la symétrie CPT, et identifie les candidats SM.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from theory.jones import (
    JONES_KNOT_NAMES,
    jones_polynomial,
    braid_index,
    topological_spin,
    topological_charge,
    mirror_jones,
    jones_power_center,
    is_amphicheiral,
    jones_retraction_factor,
)
from theory.retraction import knot_mass
from theory.particle_classification import (
    knot_particle_table,
    find_sm_candidates,
    FERMION_GENERATIONS,
    COMPOSITE_PARTICLES,
    BOSON_CORRESPONDENCES,
    generation_table,
    verify_gell_mann_nishijima,
    composite_charge_table,
)

OUTPUT_DIR = "output/particle_determination"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    particles = knot_particle_table()

    # === Table 1: Propriétés topologiques ===
    print("=" * 100)
    print("THEORIE NODALE v3 -- Determination Topologique du Spin et de la Charge")
    print("=" * 100)
    print(f"{'Noeud':<7} {'b(K)':<5} {'Spin':<6} {'Q(K)':<8} {'rho_J':<8} "
          f"{'Masse':<8} {'Type':<12} {'Amphichiral'}")
    print("-" * 100)
    for kp in particles:
        poly = jones_polynomial(kp.knot)
        amph = is_amphicheiral(poly)
        print(f"{kp.knot:<7} {braid_index(kp.knot):<5} {kp.spin:<6.1f} "
              f"{kp.charge:<8.4f} {kp.retraction:<8.2f} {kp.mass:<8.2f} "
              f"{kp.particle_type:<12} {'oui' if amph else 'non'}")
    print("=" * 100)

    # === Table 2: Correspondance SM ===
    print("\n")
    print("=" * 100)
    print("CORRESPONDANCE NOEUDS -> PARTICULES SM (par spin et signe de charge)")
    print("=" * 100)
    sm_map = find_sm_candidates()
    for knot_name, candidates in sm_map.items():
        kp = next(p for p in particles if p.knot == knot_name)
        c_names = ", ".join(c["name"] for c in candidates)
        print(f"  {knot_name:<7} (s={kp.spin:.1f}, Q={kp.charge:+.4f}) -> {c_names}")
    print("=" * 100)

    # === Graphiques ===
    spins = [kp.spin for kp in particles]
    charges = [kp.charge for kp in particles]
    masses = [kp.mass for kp in particles]
    names = [kp.knot for kp in particles]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution du spin
    ax = axes[0, 0]
    spin_vals = sorted(set(spins))
    spin_counts = [spins.count(s) for s in spin_vals]
    colors_spin = ['#2196F3' if s == 0.5 else '#FF9800' if s == 1.0 else '#4CAF50' if s == 0 else '#9C27B0'
                   for s in spin_vals]
    bars = ax.bar([str(s) for s in spin_vals], spin_counts, color=colors_spin, alpha=0.85)
    ax.set_xlabel("Spin s(K) = (b(K)-1)/2")
    ax.set_ylabel("Nombre de noeuds")
    ax.set_title("Distribution du spin topologique")
    for bar, count in zip(bars, spin_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', fontsize=10)

    # 2. Distribution de la charge
    ax = axes[0, 1]
    charge_sorted = sorted(set(charges))
    ax.hist(charges, bins=max(len(charge_sorted), 10), color='#E91E63', alpha=0.75, edgecolor='black')
    ax.set_xlabel("Charge Q(K) = (p_min+p_max)/span")
    ax.set_ylabel("Nombre de noeuds")
    ax.set_title("Distribution de la charge topologique")
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Q=0 (neutre)')
    ax.axvline(x=1/3, color='blue', linestyle=':', alpha=0.5, label='Q=1/3')
    ax.axvline(x=2/3, color='green', linestyle=':', alpha=0.5, label='Q=2/3')
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5, label='Q=1')
    ax.legend(fontsize=7)

    # 3. Charge vs Masse
    ax = axes[1, 0]
    scatter_colors = ['#2196F3' if s == 0.5 else '#FF9800' if s == 1.0 else '#4CAF50' if s == 0 else '#9C27B0'
                      for s in spins]
    ax.scatter(masses, charges, c=scatter_colors, s=80, edgecolors='black', linewidths=0.5, zorder=5)
    for name, m, q in zip(names, masses, charges):
        ax.annotate(name, (m, q), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Masse (unites naturelles)")
    ax.set_ylabel("Charge Q(K)")
    ax.set_title("Charge topologique vs Masse")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # 4. Vérification CPT : Q(K) + Q(K*) = 0
    ax = axes[1, 1]
    q_sums = []
    knot_labels = []
    for name in JONES_KNOT_NAMES:
        poly = jones_polynomial(name)
        mirror = mirror_jones(poly)
        q_k = jones_power_center(poly)
        q_mirror = jones_power_center(mirror)
        q_sums.append(q_k + q_mirror)
        knot_labels.append(name)

    ax.bar(range(len(q_sums)), q_sums, color='#673AB7', alpha=0.8)
    ax.set_xticks(range(len(knot_labels)))
    ax.set_xticklabels(knot_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Q(K) + Q(K*)")
    ax.set_title("Verification CPT : Q(K) + Q(K*) = 0")
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "particle_determination.png"), dpi=150)
    plt.close(fig)

    # === Table 3: Modèle de générations ===
    print("\n")
    print("=" * 100)
    print("MODELE DE GENERATIONS -- Fermions via Gell-Mann-Nishijima (Q = I3 + Y/2)")
    print("=" * 100)
    print(f"{'Gen':<5} {'Noeud':<7} {'Torus':<8} {'Particule':<20} "
          f"{'I3':<7} {'Y':<7} {'Q_GN':<8} {'Q_SM':<8} {'OK'}")
    print("-" * 100)
    gt = generation_table()
    gn = verify_gell_mann_nishijima()
    for row, check in zip(gt, gn):
        ok = "✓" if check["match"] else "✗"
        print(f"{row['generation']:<5} {row['knot']:<7} {row['torus']:<8} "
              f"{row['name']:<20} {row['I3']:<+7.1f} {row['Y']:<+7.2f} "
              f"{row['Q_GN']:<+8.4f} {check['Q_SM']:<+8.4f} {ok}")
    print("=" * 100)

    # === Composites ===
    print("\n")
    print("=" * 70)
    print("PARTICULES COMPOSITES")
    print("=" * 70)
    for row in composite_charge_table():
        print(f"  {row['name']:<10} ({row['symbol']}) = {row['constituents']:<12} "
              f"-> Q = {row['charge']:+.4f}, B = {row['baryon_number']}")
    print("=" * 70)

    # === Correspondances bosoniques ===
    print("\n")
    print("=" * 80)
    print("CORRESPONDANCES BOSONIQUES (charge topologique directe)")
    print("=" * 80)
    print(f"{'Noeud':<7} {'Boson':<12} {'Symbole':<6} {'Q_SM':<8} "
          f"{'M (MeV)':<12} {'Miroir'}")
    print("-" * 80)
    for b in BOSON_CORRESPONDENCES:
        print(f"{b.knot:<7} {b.boson_name:<12} {b.symbol:<6} {b.sm_charge:<+8.1f} "
              f"{b.sm_mass_mev:<12.0f} {'oui' if b.is_mirror else 'non'}")
    print("=" * 80)

    # Summary
    print(f"\nFormules :")
    print(f"  Spin     : s(K) = (b(K) - 1) / 2")
    print(f"  Charge   : Q(K) = (p_min + p_max) / span(V_K)  [bosons]")
    print(f"  Charge   : Q = I3 + Y/2  (Gell-Mann-Nishijima)  [fermions]")
    print(f"\nVerification CPT : max|Q(K)+Q(K*)| = {max(abs(s) for s in q_sums):.2e}")
    print(f"Verification GN  : {sum(1 for r in gn if r['match'])}/12 charges exactes")
    print(f"\nGraphiques sauvegardes dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
