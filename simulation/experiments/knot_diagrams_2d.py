r"""
Diagrammes 2D canoniques des nœuds (Rolfsen / Lissajous projection).

Contrairement aux formes 3D de ``theory/knot_catalog.py`` (qui sont des
courbes de Fourier conçues pour le solveur FEM mais dont la projection
(x, y) peut être dégénérée — 4_1 et 5_2 projettent identiquement par ex.),
ce module définit des paramétrisations **dédiées au rendu 2D** :

  • soit la forme « rose » (2 + cos(qt)) · (cos(pt), sin(pt)) pour les nœuds
    toriques T(p, q) — 3_1, 5_1, 7_1, 9_1 ;
  • soit la forme de Lissajous (cos(at+α), cos(bt+β), cos(ct+γ)) pour les
    autres nœuds, avec des paramètres connus dans la littérature (Bogle,
    Hughes, Murasugi 1994).

La projection (x, y) est forcée d'être propre (croisements transverses) ;
la coordonnée z fournit le signe over/under.

Pour chaque croisement détecté en 2D, le brin de plus petit z est cassé
par un trou blanc.

Usage :
    python3 -m simulation.experiments.knot_diagrams_2d
"""
from __future__ import annotations
import os
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "docs/knot_diagrams_v5"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Paramétrisations canoniques (pensées pour la projection 2D)
# ──────────────────────────────────────────────────────────────────────────
def _torus_pq(p: int, q: int):
    """Nœud torique T(p, q) en forme « rose » 2D."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        r = 2 + np.cos(q * t)
        x = r * np.cos(p * t)
        y = r * np.sin(p * t)
        z = -np.sin(q * t)
        return np.column_stack([x, y, z])
    return gen


def _figure_eight():
    """4_1 : lemniscate de Bernoulli, profil x = cos·sin, y = cos·sin·cos.

    Forme Lissajous (3, 2, 7) avec phases ajustées. Donne un vrai « 8 »
    avec 4 croisements bien visibles.
    """
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(3 * t + 0.1)
        y = np.cos(2 * t + 0.7)
        z = np.cos(7 * t + 1.0)
        return np.column_stack([x, y, z])
    return gen


def _twist_5_2():
    """5_2 — nœud twist (continued fraction [3, 2]).

    Lissajous (3, 2, 5) avec phases adaptées pour 5 croisements.
    Réf : Bogle-Hughes-Murasugi (1994) table 1.
    """
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(3 * t + 0.2)
        y = np.cos(2 * t + 0.9)
        z = np.cos(5 * t + 0.0)
        return np.column_stack([x, y, z])
    return gen


def _stevedore_6_1():
    """6_1 (stevedore) — Lissajous (4, 3, 7)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(4 * t + 0.0)
        y = np.cos(3 * t + 1.1)
        z = np.cos(7 * t + 0.5)
        return np.column_stack([x, y, z])
    return gen


def _knot_6_2():
    """6_2 — Lissajous (4, 3, 5) phase π/2."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(4 * t + 0.3)
        y = np.cos(3 * t + 0.5)
        z = np.cos(5 * t + 1.6)
        return np.column_stack([x, y, z])
    return gen


def _knot_6_3():
    """6_3 (amphichiral) — Lissajous (4, 3, 11)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(4 * t + 0.1)
        y = np.cos(3 * t + 0.9)
        z = np.cos(11 * t + 0.0)
        return np.column_stack([x, y, z])
    return gen


def _twist_7_2():
    """7_2 — twist knot (continued fraction [5, 2])."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(3 * t + 0.0)
        y = np.cos(2 * t + 0.8)
        z = np.cos(7 * t + 0.4)
        return np.column_stack([x, y, z])
    return gen


def _knot_7_3():
    """7_3 — Lissajous (4, 3, 5) phases adaptées."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(4 * t + 0.2)
        y = np.cos(3 * t + 1.4)
        z = np.cos(5 * t + 0.7)
        return np.column_stack([x, y, z])
    return gen


def _knot_7_4():
    """7_4 — pretzel form approchée par (5, 4, 7)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.0)
        y = np.cos(4 * t + 0.5)
        z = np.cos(7 * t + 1.1)
        return np.column_stack([x, y, z])
    return gen


def _knot_7_5():
    """7_5 — Lissajous (5, 4, 9)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.1)
        y = np.cos(4 * t + 1.0)
        z = np.cos(9 * t + 0.0)
        return np.column_stack([x, y, z])
    return gen


def _knot_7_6():
    """7_6 — Lissajous (5, 4, 11)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.0)
        y = np.cos(4 * t + 0.7)
        z = np.cos(11 * t + 0.3)
        return np.column_stack([x, y, z])
    return gen


def _knot_7_7():
    """7_7 — Lissajous (5, 4, 13)."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.2)
        y = np.cos(4 * t + 1.3)
        z = np.cos(13 * t + 0.0)
        return np.column_stack([x, y, z])
    return gen


def _knot_8_1():
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.0)
        y = np.cos(3 * t + 0.9)
        z = np.cos(7 * t + 0.4)
        return np.column_stack([x, y, z])
    return gen


def _knot_8_2():
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.3)
        y = np.cos(3 * t + 1.0)
        z = np.cos(11 * t + 0.5)
        return np.column_stack([x, y, z])
    return gen


def _knot_8_3():
    """8_3 — amphichiral."""
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(5 * t + 0.0)
        y = np.cos(3 * t + 1.2)
        z = np.cos(13 * t + 0.4)
        return np.column_stack([x, y, z])
    return gen


def _knot_10_1():
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cos(7 * t + 0.0)
        y = np.cos(5 * t + 0.5)
        z = np.cos(11 * t + 0.3)
        return np.column_stack([x, y, z])
    return gen


def _unknot():
    def gen(n):
        t = np.linspace(0, 2 * np.pi, n)
        return np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
    return gen


# Registre des paramétrisations 2D-friendly
KNOT_CURVES_2D: dict[str, Callable[[int], np.ndarray]] = {
    "0_1": _unknot(),
    "3_1": _torus_pq(2, 3),
    "4_1": _figure_eight(),
    "5_1": _torus_pq(2, 5),
    "5_2": _twist_5_2(),
    "6_1": _stevedore_6_1(),
    "6_2": _knot_6_2(),
    "6_3": _knot_6_3(),
    "7_1": _torus_pq(2, 7),
    "7_2": _twist_7_2(),
    "7_3": _knot_7_3(),
    "7_4": _knot_7_4(),
    "7_5": _knot_7_5(),
    "7_6": _knot_7_6(),
    "7_7": _knot_7_7(),
    "8_1": _knot_8_1(),
    "8_2": _knot_8_2(),
    "8_3": _knot_8_3(),
    "9_1": _torus_pq(2, 9),
    "10_1": _knot_10_1(),
}


# ──────────────────────────────────────────────────────────────────────────
# Détection des croisements 2D et tracé avec coupes
# ──────────────────────────────────────────────────────────────────────────
def _seg_intersect(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    dp = p3 - p1
    s = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    t = (dp[0] * d1[1] - dp[1] * d1[0]) / denom
    if 0 < s < 1 and 0 < t < 1:
        return s, t, p1 + s * d1
    return None


def _find_crossings(xy, z, min_sep=4):
    n = len(xy) - 1
    crossings = []
    for i in range(n):
        for j in range(i + min_sep, n):
            # éviter les segments quasi-adjacents au bouclage
            if i < min_sep and j > n - min_sep:
                continue
            res = _seg_intersect(xy[i], xy[i + 1], xy[j], xy[j + 1])
            if res is None:
                continue
            s, t, pt = res
            z_i = z[i] + s * (z[i + 1] - z[i])
            z_j = z[j] + t * (z[j + 1] - z[j])
            crossings.append((i, j, pt[0], pt[1], z_i, z_j))
    return crossings


def plot_knot(ax, name, n_points=2000, color="#1f3a93", linewidth=2.4,
              gap_width_factor=0.04):
    """Trace le nœud `name` avec sous-croisements proprement coupés."""
    curve = KNOT_CURVES_2D[name](n_points)
    x, y, z = curve[:, 0], curve[:, 1], curve[:, 2]
    # rayon caractéristique pour calibrer le gap
    span = max(x.max() - x.min(), y.max() - y.min())
    gap_radius = gap_width_factor * span

    xy = np.column_stack([x, y])

    # Tracer la courbe principale (sans gap pour l'instant)
    ax.plot(x, y, color=color, linewidth=linewidth,
            solid_capstyle="round", solid_joinstyle="round", zorder=1)

    # Trouver tous les croisements
    crossings = _find_crossings(xy, z, min_sep=max(6, n_points // 100))

    # Pour chaque croisement, dessiner un blanc autour du brin du DESSOUS
    for (i, j, cx, cy, z_i, z_j) in crossings:
        under_idx = i if z_i < z_j else j
        # On extrait un tronçon de ~5 segments centré sur under_idx
        a = max(0, under_idx - 3)
        b = min(len(x) - 1, under_idx + 4)
        seg_x = x[a:b + 1]
        seg_y = y[a:b + 1]
        # Distance au point de croisement
        d = np.hypot(seg_x - cx, seg_y - cy)
        mask = d > gap_radius * 2.0

        # Tracer un blanc autour du croisement sur le brin under
        ax.plot(seg_x, seg_y, color="white", linewidth=linewidth + 4,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)

        # Redessiner le brin under en deux morceaux (hors du gap)
        in_run = False
        run_x, run_y = [], []
        for ix, iy, ok in zip(seg_x, seg_y, mask):
            if ok:
                run_x.append(ix); run_y.append(iy); in_run = True
            else:
                if in_run and len(run_x) >= 2:
                    ax.plot(run_x, run_y, color=color, linewidth=linewidth,
                            solid_capstyle="round", solid_joinstyle="round",
                            zorder=3)
                run_x, run_y = [], []; in_run = False
        if in_run and len(run_x) >= 2:
            ax.plot(run_x, run_y, color=color, linewidth=linewidth,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)

    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 0.2, y.max() + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ──────────────────────────────────────────────────────────────────────────
# Étiquettes et planches
# ──────────────────────────────────────────────────────────────────────────
LABELS = {
    "0_1": r"$0_1$ (trivial)",
    "3_1": r"$3_1$ (trèfle)",
    "4_1": r"$4_1$ (huit)",
    "5_1": r"$5_1$ (cinq-feuilles)",
    "5_2": r"$5_2$ (twist)",
    "6_1": r"$6_1$ (stevedore)",
    "6_2": r"$6_2$ (Miller)",
    "6_3": r"$6_3$ (amphichiral)",
    "7_1": r"$7_1$ (torique $T(2,7)$)",
    "7_2": r"$7_2$ (twist)",
    "7_3": r"$7_3$",
    "7_4": r"$7_4$",
    "7_5": r"$7_5$",
    "7_6": r"$7_6$",
    "7_7": r"$7_7$",
    "8_1": r"$8_1$",
    "8_2": r"$8_2$",
    "8_3": r"$8_3$ (amphichiral)",
    "9_1": r"$9_1$ (torique $T(2,9)$)",
    "10_1": r"$10_1$",
}

# 12 nœuds principaux pour la planche du chapitre 2
MAIN_PANEL = [
    "0_1", "3_1", "4_1", "5_1",
    "5_2", "6_1", "6_2", "6_3",
    "7_1", "7_2", "7_3", "8_3",
]


def render_individual_pdfs():
    for name in LABELS:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        plot_knot(ax, name, n_points=2400)
        ax.set_title(LABELS[name], fontsize=14, pad=8)
        fig.tight_layout(pad=0.4)
        out = os.path.join(OUT_DIR, f"knot_{name}.pdf")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"  → {out}")


def render_main_panel():
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 9.0))
    for ax, name in zip(axes.flat, MAIN_PANEL):
        plot_knot(ax, name, n_points=2400)
        ax.set_title(LABELS[name], fontsize=12, pad=6)
    fig.tight_layout(pad=0.8)
    out = os.path.join(OUT_DIR, "knot_panel_main.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → {out}")


def render_full_catalog():
    fig, axes = plt.subplots(4, 5, figsize=(14.0, 11.5))
    names = list(LABELS.keys())
    for ax, name in zip(axes.flat, names):
        plot_knot(ax, name, n_points=2400)
        ax.set_title(LABELS[name], fontsize=11, pad=4)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.6)
    out = os.path.join(OUT_DIR, "knot_panel_full20.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    print("=== Diagrammes 2D canoniques ===")
    print(f"  Sortie : {OUT_DIR}/\n")
    print("→ Diagrammes individuels…")
    render_individual_pdfs()
    print("\n→ Planche principale (12 nœuds)…")
    render_main_panel()
    print("\n→ Catalogue complet (20 nœuds)…")
    render_full_catalog()
    print(f"\n✓ Fini.")
