"""Galerie des 20 nœuds du catalogue — planches et vues individuelles.

Génère :
  - output/knot_gallery/catalog_20knots.png  (planche 4×5)
  - output/knot_gallery/{name}.png            (vues individuelles haute résolution)

Style : courbe 3D colorée par z (coolwarm), annotations ρ_J, spin, charge.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from theory.jones import (
    JONES_KNOT_NAMES as KNOT_NAMES,
    jones_polynomial, jones_determinant, jones_span,
    topological_spin, topological_charge,
)
from theory.retraction import retraction_factor
from theory.knot_catalog import knot_curve

OUT_DIR = "output/knot_gallery"


def _plot_knot_3d(ax, curve, title_lines, linewidth=1.2):
    """Trace une courbe 3D colorée par z sur un Axes3D."""
    z = curve[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)

    for i in range(len(curve) - 1):
        ax.plot(
            curve[i:i+2, 0], curve[i:i+2, 1], curve[i:i+2, 2],
            color=plt.cm.coolwarm(z_norm[i]),
            linewidth=linewidth,
        )

    ax.set_title("\n".join(title_lines), fontsize=9, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.set_box_aspect([1, 1, 0.4])
    ax.view_init(elev=25, azim=45)


def _knot_annotations(name):
    """Retourne (ρ_J, spin, charge) formatés."""
    poly = jones_polynomial(name)
    rho = retraction_factor(poly)
    s = topological_spin(name)
    q = topological_charge(name)
    return rho, s, q


def run():
    """Génère la galerie complète des 20 nœuds."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Planche 4×5 ---
    nrows, ncols = 4, 5
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "Galerie des 20 nœuds  —  Théorie Quantique des Nœuds",
        fontsize=18, fontweight="bold", y=0.98,
    )

    for idx, name in enumerate(KNOT_NAMES):
        curve = knot_curve(name, n_points=2000)
        rho, s, q = _knot_annotations(name)

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        title_lines = [
            f"${name.replace('_', '_{') + '}'}$",
            f"$\\rho_J = {rho:.1f}$,  $s = {s}$,  $Q = {q:.2f}$",
        ]
        _plot_knot_3d(ax, curve, title_lines)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_catalog = os.path.join(OUT_DIR, "catalog_20knots.png")
    plt.savefig(out_catalog, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Planche sauvegardée : {out_catalog}")

    # --- Vues individuelles haute résolution ---
    for name in KNOT_NAMES:
        curve = knot_curve(name, n_points=3000)
        rho, s, q = _knot_annotations(name)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        title_lines = [
            f"Nœud ${name.replace('_', '_{') + '}'}$",
            f"$\\rho_J = {rho:.1f}$,  spin $= {s}$,  charge $= {q:.2f}$",
        ]
        _plot_knot_3d(ax, curve, title_lines, linewidth=1.5)

        plt.tight_layout()
        out_i = os.path.join(OUT_DIR, f"{name}.png")
        plt.savefig(out_i, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_i}")


if __name__ == "__main__":
    run()
