#!/usr/bin/env python3
"""
GO/NO-GO : Comparaison Q(3₁) vs Q(0₁)
=========================================
Simule le confinement d'un électron dans :
  - un canal en forme de trèfle 3₁ (nœud topologique)
  - un canal en forme d'anneau 0₁ (nœud trivial, référence)
de MÊME longueur et même section, et compare :
  - les spectres d'énergie (E_0, E_1, ...)
  - le gap ΔE = E_1 - E_0 (lié au facteur Q)
  - la localisation de |Ψ|² (pic/moyenne)
  - l'énergie de confinement

Si Q(3₁) > Q(0₁) : le trèfle piège mieux → la topologie a un
  effet mesurable → GO pour le prototype.
Si Q(3₁) ≈ Q(0₁) : pas d'effet topologique → la batterie topo
  est un condensateur classique → le bénéfice est uniquement
  l'anti-fuite (barrière CS), pas le piégeage.

Usage :
  python3 -m simulation.experiments.compare_trefoil_vs_ring
"""

import numpy as np
import time
import os
import sys

# Ajouter le répertoire racine au path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from theory.knot_catalog import knot_curve
from theory.schrodinger import (generate_mesh, assemble_system,
                                 generate_mesh_multi, assemble_system_multi)
from scipy.spatial import cKDTree

OUTPUT = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT, exist_ok=True)


def run_knot(knot_name, n_coarse=25, n_eig=15, epsilon=0.15):
    """Résout le problème de Schrödinger pour un nœud donné.

    Returns dict with eigenvalues, eigenvectors, points, stats.
    """
    print(f"\n{'='*60}")
    print(f"Résolution du nœud {knot_name}")
    print(f"{'='*60}")

    t0 = time.time()

    # Courbe du nœud
    curve = knot_curve(knot_name, n_points=500)

    # Maillage
    points, tets, boundary = generate_mesh(
        curve, n_coarse=n_coarse, epsilon=epsilon, refine=True
    )
    free_mask = ~boundary
    n_free = free_mask.sum()
    print(f"  Points: {len(points)}, Tets: {len(tets)}, Free DOFs: {n_free}")

    # KDTree pour les distances
    knot_tree = cKDTree(curve)

    # Assemblage et résolution
    H, M = assemble_system(
        points, tets, boundary, knot_name, knot_tree,
        energy=0.0, epsilon=epsilon
    )

    from theory.schrodinger import solve_eigenproblem
    eigenvalues, eigenvectors = solve_eigenproblem(H, M, n_eigenvalues=n_eig)

    elapsed = time.time() - t0
    print(f"  Eigenvalues: {len(eigenvalues)}")
    print(f"  Temps: {elapsed:.1f}s")

    # Statistiques sur |Ψ|²
    stats = []
    for i in range(min(n_eig, len(eigenvalues))):
        psi = np.zeros(len(points))
        psi[free_mask] = eigenvectors[:, i]
        psi2 = psi**2
        psi2_mean = psi2[free_mask].mean()
        psi2_max = psi2[free_mask].max()
        ratio = psi2_max / psi2_mean if psi2_mean > 0 else 0
        stats.append({
            'E': eigenvalues[i],
            'psi2_mean': psi2_mean,
            'psi2_max': psi2_max,
            'ratio': ratio,
        })

    return {
        'knot_name': knot_name,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'points': points,
        'tets': tets,
        'boundary': boundary,
        'free_mask': free_mask,
        'stats': stats,
        'elapsed': elapsed,
    }


def compare_and_plot(results_trefoil, results_ring):
    """Compare les résultats et génère les graphiques."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ev_t = results_trefoil['eigenvalues']
    ev_r = results_ring['eigenvalues']
    stats_t = results_trefoil['stats']
    stats_r = results_ring['stats']

    n_show = min(len(ev_t), len(ev_r), 12)

    # Normaliser : E_n - E_0
    E0_t = ev_t[0]
    E0_r = ev_r[0]
    dE_t = ev_t[:n_show] - E0_t
    dE_r = ev_r[:n_show] - E0_r

    # Gap
    gap_t = ev_t[1] - ev_t[0] if len(ev_t) > 1 else 0
    gap_r = ev_r[1] - ev_r[0] if len(ev_r) > 1 else 0

    # Bande
    bw_t = ev_t[n_show-1] - ev_t[0]
    bw_r = ev_r[n_show-1] - ev_r[0]

    # Ratio pic/moyenne du fondamental
    ratio_t = stats_t[0]['ratio'] if stats_t else 0
    ratio_r = stats_r[0]['ratio'] if stats_r else 0

    # Affichage
    print(f"\n{'='*60}")
    print("COMPARAISON TRÈFLE 3₁ vs ANNEAU 0₁")
    print(f"{'='*60}")
    print(f"\n{'':25s} {'3₁ (trèfle)':>15s} {'0₁ (anneau)':>15s} {'Ratio':>10s}")
    print('-' * 68)
    print(f"{'E₀ (fondamental)':25s} {E0_t:>15.4f} {E0_r:>15.4f} {E0_t/E0_r:>10.3f}")
    print(f"{'Gap ΔE = E₁-E₀':25s} {gap_t:>15.4f} {gap_r:>15.4f} {gap_t/gap_r if gap_r else float('inf'):>10.3f}")
    print(f"{'Bande (E_n-E₀)':25s} {bw_t:>15.4f} {bw_r:>15.4f} {bw_t/bw_r if bw_r else float('inf'):>10.3f}")
    print(f"{'pic/moy |Ψ₀|²':25s} {ratio_t:>15.1f} {ratio_r:>15.1f} {ratio_t/ratio_r if ratio_r else float('inf'):>10.3f}")
    print(f"{'DOFs libres':25s} {results_trefoil['free_mask'].sum():>15d} {results_ring['free_mask'].sum():>15d}")

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if gap_r > 0:
        Q_ratio = gap_t / gap_r
        print(f"\n  Gap(3₁) / Gap(0₁) = {Q_ratio:.3f}")
        if Q_ratio > 1.1:
            print(f"  → Le trèfle a un gap {(Q_ratio-1)*100:.0f}% PLUS GRAND que l'anneau")
            print(f"  → EFFET TOPOLOGIQUE DÉTECTÉ → GO pour le prototype")
        elif Q_ratio > 0.9:
            print(f"  → Pas de différence significative (ratio {Q_ratio:.2f})")
            print(f"  → PAS d'effet topologique sur le gap")
            print(f"  → Le stockage est purement capacitif")
        else:
            print(f"  → Le trèfle a un gap PLUS PETIT ({(1-Q_ratio)*100:.0f}%)")
            print(f"  → L'anneau confine MIEUX (géométrie plus simple)")

    if ratio_r > 0:
        loc_ratio = ratio_t / ratio_r
        print(f"\n  Localisation |Ψ₀|² : pic/moy(3₁) / pic/moy(0₁) = {loc_ratio:.2f}")
        if loc_ratio > 1.2:
            print(f"  → L'électron est PLUS LOCALISÉ dans le trèfle")
            print(f"  → Effet de focalisation topologique")
        else:
            print(f"  → Localisation similaire")

    # --- Graphiques ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison trèfle 3₁ vs anneau 0₁ — GO/NO-GO',
                 fontsize=14, fontweight='bold')

    # 1. Spectre d'énergie
    ax = axes[0, 0]
    x = np.arange(n_show)
    w = 0.35
    ax.bar(x - w/2, dE_t, w, label='3₁ (trèfle)', color='#E53935', alpha=0.8)
    ax.bar(x + w/2, dE_r, w, label='0₁ (anneau)', color='#1E88E5', alpha=0.8)
    ax.set_xlabel('Niveau n')
    ax.set_ylabel('E_n - E_0')
    ax.set_title('Spectre d\'énergie')
    ax.legend()

    # 2. Gap et bande
    ax = axes[0, 1]
    metrics = ['Gap ΔE', 'Bande', '|E₀|']
    vals_t = [gap_t, bw_t, abs(E0_t)]
    vals_r = [gap_r, bw_r, abs(E0_r)]
    x = np.arange(len(metrics))
    ax.bar(x - w/2, vals_t, w, label='3₁', color='#E53935', alpha=0.8)
    ax.bar(x + w/2, vals_r, w, label='0₁', color='#1E88E5', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Métriques de confinement')
    ax.legend()

    # 3. Localisation |Ψ|²
    ax = axes[1, 0]
    ratios_t = [s['ratio'] for s in stats_t[:n_show]]
    ratios_r = [s['ratio'] for s in stats_r[:n_show]]
    ax.plot(ratios_t, 'o-', color='#E53935', label='3₁', markersize=5)
    ax.plot(ratios_r, 's-', color='#1E88E5', label='0₁', markersize=5)
    ax.set_xlabel('Niveau n')
    ax.set_ylabel('pic/moy |Ψ_n|²')
    ax.set_title('Localisation de la fonction d\'onde')
    ax.legend()

    # 4. Tableau résumé
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['', '3₁ (trèfle)', '0₁ (anneau)', 'Ratio'],
        ['E₀', f'{E0_t:.4f}', f'{E0_r:.4f}', f'{E0_t/E0_r:.3f}'],
        ['Gap ΔE', f'{gap_t:.4f}', f'{gap_r:.4f}',
         f'{gap_t/gap_r:.3f}' if gap_r else '∞'],
        ['Bande', f'{bw_t:.4f}', f'{bw_r:.4f}',
         f'{bw_t/bw_r:.3f}' if bw_r else '∞'],
        ['pic/moy', f'{ratio_t:.1f}', f'{ratio_r:.1f}',
         f'{ratio_t/ratio_r:.2f}' if ratio_r else '∞'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    # Coloriser l'en-tête
    for j in range(4):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_text_props(fontweight='bold')

    verdict = "GO" if gap_r > 0 and gap_t/gap_r > 1.1 else "À CONFIRMER"
    color = '#4CAF50' if verdict == "GO" else '#FF9800'
    ax.text(0.5, -0.1, f'VERDICT : {verdict}',
            transform=ax.transAxes, ha='center', fontsize=16,
            fontweight='bold', color=color)

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'compare_trefoil_vs_ring.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique : {path}")
    plt.close()


def run_entrelacs(n_comp=3, n_coarse=22, n_eig=15, epsilon=0.15):
    """Résout pour un entrelacs de n trèfles 3₁ entrelacés.

    Chaque composante est un trèfle 3₁ tourné de 40° × i.
    """
    print(f"\n{'='*60}")
    print(f"Résolution de l'entrelacs {n_comp}×3₁")
    print(f"{'='*60}")

    t0 = time.time()

    # Générer n courbes de trèfle tournées
    curves = []
    knot_names = []
    for i in range(n_comp):
        angle = i * 40 * np.pi / 180  # rotation en radians
        base = knot_curve('3_1', n_points=500)
        # Rotation autour de l'axe z
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = base.copy()
        rotated[:, 0] = base[:, 0] * cos_a - base[:, 1] * sin_a
        rotated[:, 1] = base[:, 0] * sin_a + base[:, 1] * cos_a
        curves.append(rotated)
        knot_names.append('3_1')

    # Maillage multi-nœud
    points, tets, boundary = generate_mesh_multi(
        curves, n_coarse=n_coarse, epsilon=epsilon, refine=True
    )
    free_mask = ~boundary
    n_free = free_mask.sum()
    print(f"  Composantes: {n_comp}")
    print(f"  Points: {len(points)}, Tets: {len(tets)}, Free DOFs: {n_free}")

    # Assemblage multi-nœud
    H, M = assemble_system_multi(
        points, tets, boundary, knot_names, curves,
        energy=0.0, epsilon=epsilon
    )

    from theory.schrodinger import solve_eigenproblem
    eigenvalues, eigenvectors = solve_eigenproblem(H, M, n_eigenvalues=n_eig)

    elapsed = time.time() - t0
    print(f"  Eigenvalues: {len(eigenvalues)}")
    print(f"  Temps: {elapsed:.1f}s")

    # Stats
    stats = []
    for i in range(min(n_eig, len(eigenvalues))):
        psi = np.zeros(len(points))
        psi[free_mask] = eigenvectors[:, i]
        psi2 = psi**2
        psi2_mean = psi2[free_mask].mean()
        psi2_max = psi2[free_mask].max()
        ratio = psi2_max / psi2_mean if psi2_mean > 0 else 0
        stats.append({
            'E': eigenvalues[i],
            'psi2_mean': psi2_mean,
            'psi2_max': psi2_max,
            'ratio': ratio,
        })

    return {
        'knot_name': f'{n_comp}×3₁',
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'points': points,
        'tets': tets,
        'boundary': boundary,
        'free_mask': free_mask,
        'stats': stats,
        'elapsed': elapsed,
    }


def compare_three(res_trefoil, res_ring, res_entrelacs):
    """Compare 3 configurations : 1×3₁, 0₁, 3×3₁."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configs = [
        ('0₁ (anneau)', res_ring, '#1E88E5'),
        ('1×3₁ (trèfle)', res_trefoil, '#E53935'),
        ('3×3₁ (entrelacs)', res_entrelacs, '#2E7D32'),
    ]

    n_show = min(*(len(r['eigenvalues']) for _, r, _ in configs), 12)

    print(f"\n{'='*70}")
    print("COMPARAISON 0₁ vs 1×3₁ vs 3×3₁")
    print(f"{'='*70}")
    print(f"\n{'':25s}", end='')
    for name, _, _ in configs:
        print(f"{name:>18s}", end='')
    print()
    print('-' * 79)

    rows = []
    for label, key in [('E₀', lambda r: r['eigenvalues'][0]),
                        ('Gap ΔE', lambda r: r['eigenvalues'][1]-r['eigenvalues'][0]),
                        ('Bande', lambda r: r['eigenvalues'][n_show-1]-r['eigenvalues'][0]),
                        ('pic/moy |Ψ₀|²', lambda r: r['stats'][0]['ratio']),
                        ('DOFs', lambda r: float(r['free_mask'].sum()))]:
        vals = [key(r) for _, r, _ in configs]
        print(f"{label:25s}", end='')
        for v in vals:
            print(f"{v:>18.2f}", end='')
        print()
        rows.append((label, vals))

    # Ratios vs anneau
    print(f"\n{'Ratios vs anneau':25s}", end='')
    for name, _, _ in configs:
        print(f"{name:>18s}", end='')
    print()
    print('-' * 79)
    for label, vals in rows:
        if vals[0] != 0 and label != 'DOFs':
            print(f"{label:25s}", end='')
            for v in vals:
                print(f"{v/vals[0]:>18.3f}", end='')
            print()

    # Verdict
    gap_ring = rows[1][1][0]
    gap_trefoil = rows[1][1][1]
    gap_entrelacs = rows[1][1][2]
    loc_ring = rows[3][1][0]
    loc_trefoil = rows[3][1][1]
    loc_entrelacs = rows[3][1][2]

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"\n  Gap(3×3₁) / Gap(0₁) = {gap_entrelacs/gap_ring:.3f}")
    if gap_entrelacs/gap_ring > 1.15:
        print(f"  → L'entrelacs a un gap {(gap_entrelacs/gap_ring-1)*100:.0f}% PLUS GRAND")
        print(f"  → EFFET TOPOLOGIQUE DE L'ENTRELACS → GO")
    elif gap_entrelacs/gap_ring > 1.05:
        print(f"  → Gain modeste ({(gap_entrelacs/gap_ring-1)*100:.0f}%)")
    else:
        print(f"  → Pas de gain significatif sur le gap")

    print(f"\n  Localisation |Ψ₀|² :")
    print(f"    0₁  : {loc_ring:.1f}")
    print(f"    1×3₁: {loc_trefoil:.1f} (×{loc_trefoil/loc_ring:.0f} vs anneau)")
    print(f"    3×3₁: {loc_entrelacs:.1f} (×{loc_entrelacs/loc_ring:.0f} vs anneau)")

    # Graphique
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison 0₁ vs 1×3₁ vs 3×3₁ — GO/NO-GO entrelacs',
                 fontsize=14, fontweight='bold')

    # 1. Spectre
    ax = axes[0, 0]
    x = np.arange(n_show)
    w = 0.25
    for i, (name, r, color) in enumerate(configs):
        ev = r['eigenvalues'][:n_show] - r['eigenvalues'][0]
        ax.bar(x + (i-1)*w, ev, w, label=name, color=color, alpha=0.8)
    ax.set_xlabel('Niveau n')
    ax.set_ylabel('E_n - E_0')
    ax.set_title('Spectre d\'énergie')
    ax.legend(fontsize=8)

    # 2. Métriques
    ax = axes[0, 1]
    metrics = ['Gap ΔE', 'Bande', '|E₀|']
    for i, (name, r, color) in enumerate(configs):
        vals = [r['eigenvalues'][1]-r['eigenvalues'][0],
                r['eigenvalues'][n_show-1]-r['eigenvalues'][0],
                abs(r['eigenvalues'][0])]
        ax.bar(np.arange(3) + (i-1)*w, vals, w, label=name,
               color=color, alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(metrics)
    ax.set_title('Métriques de confinement')
    ax.legend(fontsize=8)

    # 3. Localisation
    ax = axes[1, 0]
    for name, r, color in configs:
        ratios = [s['ratio'] for s in r['stats'][:n_show]]
        ax.plot(ratios, 'o-', color=color, label=name, markersize=4)
    ax.set_xlabel('Niveau n')
    ax.set_ylabel('pic/moy |Ψ_n|²')
    ax.set_title('Localisation de la fonction d\'onde')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # 4. Tableau
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [['', '0₁', '1×3₁', '3×3₁', 'Ratio 3×3₁/0₁']]
    for label, vals in rows:
        if label == 'DOFs':
            table_data.append([label, f'{vals[0]:.0f}', f'{vals[1]:.0f}',
                              f'{vals[2]:.0f}', ''])
        else:
            r = vals[2]/vals[0] if vals[0] != 0 else 0
            table_data.append([label, f'{vals[0]:.2f}', f'{vals[1]:.2f}',
                              f'{vals[2]:.2f}', f'{r:.2f}'])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(5):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'compare_trefoil_vs_ring_vs_entrelacs.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique : {path}")
    plt.close()


def main():
    print("=" * 60)
    print("GO/NO-GO : Q(3₁) vs Q(0₁) vs Q(3×3₁)")
    print("Simulation FEM — nœud simple, anneau, entrelacs")
    print("=" * 60)

    # Paramètres communs
    n_coarse = 25
    n_eig = 15
    epsilon = 0.15

    # Résoudre le trèfle 3₁
    res_trefoil = run_knot('3_1', n_coarse=n_coarse, n_eig=n_eig,
                            epsilon=epsilon)

    # Résoudre l'anneau 0₁
    res_ring = run_knot('0_1', n_coarse=n_coarse, n_eig=n_eig,
                         epsilon=epsilon)

    # Résoudre l'entrelacs 3×3₁
    res_entrelacs = run_entrelacs(n_comp=3, n_coarse=22, n_eig=n_eig,
                                   epsilon=epsilon)

    # Comparer les 3
    compare_three(res_trefoil, res_ring, res_entrelacs)

    # Tableau des niveaux
    print(f"\n{'='*60}")
    print("Niveaux d'énergie détaillés")
    print(f"{'='*60}")
    print(f"\n{'n':>3s} {'E_n(3₁)':>12s} {'E_n(0₁)':>12s} {'ΔE_n(3₁)':>12s} {'ΔE_n(0₁)':>12s}")
    print('-' * 55)
    n_show = min(len(res_trefoil['eigenvalues']),
                 len(res_ring['eigenvalues']), 12)
    for i in range(n_show):
        Et = res_trefoil['eigenvalues'][i]
        Er = res_ring['eigenvalues'][i]
        dEt = Et - res_trefoil['eigenvalues'][0]
        dEr = Er - res_ring['eigenvalues'][0]
        print(f"{i:>3d} {Et:>12.4f} {Er:>12.4f} {dEt:>12.4f} {dEr:>12.4f}")


if __name__ == '__main__':
    main()
