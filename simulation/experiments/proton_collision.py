"""
Expérience : Recomposition topologique lors de collisions pp

Lors d'une collision proton-proton à haute énergie, les entrelacs
(9₁+7₄+6₃)×2 se brisent et l'énergie du champ χ crée de nouveaux
nœuds qui se recomposent en particules. Ce script énumère les canaux
de collision accessibles en fonction de √s et met en évidence les
processus classiques du LHC.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt

from theory.collision import (
    build_catalog, find_channels, rank_channels,
    collision_wavelength, particle_lambda_max,
    channel_wavelength_factor, rank_channels_wavelength,
)

OUTPUT_DIR = "output/proton_collision"

# Processus LHC célèbres à identifier
LHC_PROCESSES = [
    ("pp (élastique)", ["p", "p"]),
    ("pp + e⁻e⁺", ["e+", "e-", "p", "p"]),
    ("pp + μ⁻μ⁺", ["mu+", "mu-", "p", "p"]),
    ("pp + W⁺W⁻", ["W+", "W-", "p", "p"]),
    ("pp + Z⁰", ["Z0", "p", "p"]),
    ("pp + H", ["H", "p", "p"]),
    ("pp + Z⁰H", ["H", "Z0", "p", "p"]),
    ("pp + tt̄", ["p", "p", "t", "tbar"]),
]


def _channel_threshold(products: list[str]) -> float:
    """Seuil √s d'un canal = somme des masses."""
    catalog = build_catalog()
    return sum(catalog[p].mass_mev for p in products)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    catalog = build_catalog()
    m_p = catalog["p"].mass_mev

    print("=" * 76)
    print("  Recomposition topologique : collisions pp")
    print("=" * 76)

    # ====== Table 1 : Processus LHC célèbres ======
    print("\n  TABLE 1 : Processus LHC célèbres")
    print("-" * 60)
    print(f"  {'Processus':<24} {'Seuil √s (GeV)':>16} {'Produits':>16}")
    print("-" * 60)

    lhc_thresholds = []
    for label, prods in LHC_PROCESSES:
        thr = _channel_threshold(prods)
        lhc_thresholds.append((label, thr, prods))
        print(f"  {label:<24} {thr/1000:>14.2f}   {len(prods):>8}")

    # ====== Scan nombre de canaux vs √s ======
    print("\n  TABLE 2 : Nombre de canaux vs √s")
    print("-" * 50)
    print(f"  {'√s (GeV)':>12} {'N canaux (≤4 prod.)':>22}")
    print("-" * 50)

    # Energies de scan (en MeV)
    scan_energies = [
        1000, 1500, 1900, 2000, 3000, 5000,
        10_000, 50_000, 100_000, 200_000,
        500_000, 1_000_000,
    ]

    scan_counts = []
    for sqrt_s in scan_energies:
        channels = find_channels(["p", "p"], sqrt_s, max_products=4)
        n = len(channels)
        scan_counts.append(n)
        print(f"  {sqrt_s/1000:>10.1f}   {n:>14}")

    # ====== Canaux accessibles à 14 TeV ======
    sqrt_s_14tev = 14_000_000.0  # MeV
    print(f"\n  Canaux accessibles à √s = 14 TeV (max 4 produits) :")
    channels_14 = find_channels(["p", "p"], sqrt_s_14tev, max_products=4)
    print(f"  Total : {len(channels_14)} canaux")

    # Afficher les 20 premiers et 10 derniers par seuil
    print(f"\n  {'#':<5} {'Seuil (GeV)':>12}  Produits")
    print("  " + "-" * 55)
    show = channels_14[:20]
    for i, ch in enumerate(show):
        prods_str = " + ".join(ch.products)
        print(f"  {i+1:<5} {ch.threshold_sqrt_s/1000:>10.2f}   {prods_str}")
    if len(channels_14) > 30:
        print("  ...")
        for i, ch in enumerate(channels_14[-5:]):
            idx = len(channels_14) - 5 + i + 1
            prods_str = " + ".join(ch.products)
            print(f"  {idx:<5} {ch.threshold_sqrt_s/1000:>10.2f}   {prods_str}")

    # ====== Table 3 : Top 30 canaux par probabilité ======
    top30 = rank_channels(channels_14, top_n=30)
    print("\n  TABLE 3 : Top 30 canaux par probabilité (P16)")
    print("-" * 80)
    print(f"  {'Rang':<5} {'Produits':<30} {'Seuil (GeV)':>12} "
          f"{'N_nœuds':>8} {'P (%)':>10}")
    print("-" * 80)
    for i, ch in enumerate(top30):
        prods_str = " + ".join(ch.products)
        pct = ch.statistical_weight * 100
        print(f"  {i+1:<5} {prods_str:<30} "
              f"{ch.threshold_sqrt_s/1000:>10.2f}   "
              f"{ch.n_total_knots:>6}   {pct:>8.4f}")

    # ====== Table 4 : Top 30 canaux par probabilité P16+λ ======
    lam_coll = collision_wavelength(sqrt_s_14tev)
    top30_wl = rank_channels_wavelength(channels_14, sqrt_s_14tev, top_n=30)
    print(f"\n  TABLE 4 : Top 30 canaux par probabilité (P16+λ)")
    print(f"  λ_coll = {lam_coll:.6f} (à 14 TeV)")
    print("-" * 100)
    print(f"  {'Rang':<5} {'Produits':<30} {'λ_max_min':>10} "
          f"{'S_λ':>10} {'P (%)':>10}")
    print("-" * 100)
    for i, ch in enumerate(top30_wl):
        prods_str = " + ".join(ch.products)
        # λ_max_min = min des λ_max des produits
        lam_maxs = [particle_lambda_max(p, catalog) for p in ch.products]
        lam_max_min = min(lam_maxs)
        s_lam = channel_wavelength_factor(ch.products, sqrt_s_14tev, catalog)
        pct = ch.statistical_weight * 100
        print(f"  {i+1:<5} {prods_str:<30} {lam_max_min:>10.4f} "
              f"{s_lam:>10.6f} {pct:>8.4f}")

    # ====== Table 5 : Comparaison P16 vs P16+λ ======
    # Re-rank P16 pour avoir les rangs
    all_p16 = rank_channels(channels_14, top_n=len(channels_14))
    p16_rank = {}
    for i, c in enumerate(all_p16):
        key = tuple(sorted(c.products))
        p16_rank[key] = i + 1

    all_wl = rank_channels_wavelength(channels_14, sqrt_s_14tev,
                                      top_n=len(channels_14))
    wl_rank = {}
    for i, c in enumerate(all_wl):
        key = tuple(sorted(c.products))
        wl_rank[key] = i + 1

    print(f"\n  TABLE 5 : Comparaison P16 vs P16+λ (Top 30 P16+λ)")
    print("-" * 80)
    print(f"  {'Rang P16':>8} {'Rang P16+λ':>10} {'Δrang':>8}  {'Produits':<40}")
    print("-" * 80)
    for ch in top30_wl:
        key = tuple(sorted(ch.products))
        r_p16 = p16_rank.get(key, 0)
        r_wl = wl_rank.get(key, 0)
        delta = r_p16 - r_wl
        prods_str = " + ".join(ch.products)
        sign = "+" if delta > 0 else ""
        print(f"  {r_p16:>8} {r_wl:>10} {sign}{delta:>7}  {prods_str:<40}")

    # ====== Bilan ======
    print("\n" + "=" * 76)
    print("  BILAN")
    print("=" * 76)
    print(f"\n  Proton : entrelacs 9₁+7₄+6₃, masse = {m_p:.1f} MeV")
    print(f"  Collision pp : Q_init = +2, B_init = +2")
    print(f"  Conservation : charge (Gell-Mann-Nishijima) + nombre baryonique")
    print(f"  Canaux à 14 TeV (≤4 prod.) : {len(channels_14)}")
    print(f"  Canal dominant P16   : {' + '.join(top30[0].products)} "
          f"(P = {top30[0].statistical_weight*100:.2f}%)")
    print(f"  Canal dominant P16+λ : {' + '.join(top30_wl[0].products)} "
          f"(P = {top30_wl[0].statistical_weight*100:.2f}%)")
    print(f"  λ_coll(14 TeV) = {lam_coll:.6f}")

    # ====== Graphiques ======
    _make_plots(scan_energies, scan_counts, lhc_thresholds, top30,
                top30_wl, sqrt_s_14tev)
    print(f"\n  Graphiques sauvegardés dans {OUTPUT_DIR}/")


def _make_plots(scan_energies, scan_counts, lhc_thresholds, top_channels,
                top_channels_wl=None, sqrt_s=None):
    """Génère les graphiques de collision."""

    # 1. Nombre de canaux vs √s
    fig, ax = plt.subplots(figsize=(10, 6))
    nonzero = [(e, n) for e, n in zip(scan_energies, scan_counts) if n > 0]
    if nonzero:
        energies_gev = [e / 1000 for e, _ in nonzero]
        counts = [n for _, n in nonzero]
        ax.plot(energies_gev, counts, 'o-', color='steelblue',
                markersize=6, linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("√s (GeV)")
    ax.set_ylabel("Nombre de canaux (≤ 4 produits)")
    ax.set_title("Ouverture progressive des canaux pp")
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "channels_vs_sqrt_s.png"), dpi=150)
    plt.close(fig)

    # 2. Seuils des processus LHC
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [t[0] for t in lhc_thresholds]
    thresholds_gev = [t[1] / 1000 for t in lhc_thresholds]

    # Trier par seuil
    sorted_data = sorted(zip(thresholds_gev, labels))
    thresholds_sorted = [d[0] for d in sorted_data]
    labels_sorted = [d[1] for d in sorted_data]

    y_pos = np.arange(len(labels_sorted))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels_sorted)))
    ax.barh(y_pos, thresholds_sorted, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Seuil √s (GeV)")
    ax.set_title("Seuils des processus LHC — Théorie Nodale")
    ax.set_xscale('log')

    # Annotations
    for i, (thr, label) in enumerate(zip(thresholds_sorted, labels_sorted)):
        ax.text(thr * 1.1, i, f"{thr:.0f} GeV", va='center', fontsize=8)

    ax.axvline(x=14000, color='red', linestyle='--', alpha=0.7,
               label='LHC 14 TeV')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x', which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lhc_thresholds.png"), dpi=150)
    plt.close(fig)

    # 3. Top 30 canaux par probabilité (P16)
    fig, ax = plt.subplots(figsize=(12, 10))
    top_labels = [" + ".join(ch.products) for ch in reversed(top_channels)]
    top_probs = [ch.statistical_weight * 100 for ch in reversed(top_channels)]
    top_knots = [ch.n_total_knots for ch in reversed(top_channels)]

    y_pos = np.arange(len(top_labels))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_labels)))
    bars = ax.barh(y_pos, top_probs, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=7)
    ax.set_xlabel("Probabilité (%)")
    ax.set_title("Top 30 canaux pp par probabilité topologique (P16, 14 TeV)")

    # Annotations : nombre de nœuds
    for i, (prob, nk) in enumerate(zip(top_probs, top_knots)):
        if prob > 0.01:
            ax.text(prob + 0.1, i, f"N={nk}", va='center', fontsize=7)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top30_probability.png"), dpi=150)
    plt.close(fig)

    # 4. Comparaison P16 vs P16+λ (barh côte à côte)
    if top_channels_wl is not None:
        # Trouver les canaux communs (union des top 20 de chaque)
        n_show = min(20, len(top_channels), len(top_channels_wl))
        # Utiliser les top N de P16+λ comme référence
        wl_labels = [" + ".join(ch.products) for ch in top_channels_wl[:n_show]]

        # Retrouver les probas P16 pour ces mêmes canaux
        p16_probs_map = {}
        for ch in top_channels:
            key = tuple(sorted(ch.products))
            p16_probs_map[key] = ch.statistical_weight * 100

        p16_vals = []
        wl_vals = []
        for ch in top_channels_wl[:n_show]:
            key = tuple(sorted(ch.products))
            p16_vals.append(p16_probs_map.get(key, 0.0))
            wl_vals.append(ch.statistical_weight * 100)

        fig, ax = plt.subplots(figsize=(12, 10))
        y_pos = np.arange(n_show)
        bar_h = 0.35
        ax.barh(y_pos + bar_h / 2, list(reversed(p16_vals)),
                height=bar_h, color='steelblue', alpha=0.8, label='P16')
        ax.barh(y_pos - bar_h / 2, list(reversed(wl_vals)),
                height=bar_h, color='darkorange', alpha=0.8, label='P16+λ')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(reversed(wl_labels)), fontsize=7)
        ax.set_xlabel("Probabilité (%)")
        ax.set_title("Comparaison P16 vs P16+λ (14 TeV)")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "p16_vs_p16_lambda.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    run()
