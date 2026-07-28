"""Visualisation des nœuds toriques de matière noire T(2,p), p = 19..37.

Génère une planche de 10 nœuds toriques (2 rangées × 5 colonnes)
représentant les candidats de matière noire avec c > 18.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from theory.knot_catalog import _torus_knot_curve
from theory.jones import jones_retraction_factor


def torus_jones_rho(p: int) -> float:
    """ρ_J pour T(2,p) = span + γ·det = (p-1) + 0.1·p = 1.1p - 1."""
    return 1.1 * p - 1


def run():
    """Dessine les 10 nœuds toriques de matière noire T(2,p), p impair de 19 à 37."""
    p_values = list(range(19, 38, 2))  # 19, 21, 23, 25, 27, 29, 31, 33, 35, 37
    n_knots = len(p_values)
    ncols, nrows = 5, 2

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        "Nœuds toriques de matière noire  $T(2,p)$,  $c > 18$",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for idx, p in enumerate(p_values):
        curve = _torus_knot_curve(2, p, n_points=2000)

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        # Colorer par position z pour donner du relief
        z = curve[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)

        # Tracer par segments colorés
        for i in range(len(curve) - 1):
            ax.plot(
                curve[i:i+2, 0], curve[i:i+2, 1], curve[i:i+2, 2],
                color=plt.cm.coolwarm(z_norm[i]),
                linewidth=1.2,
            )

        rho = torus_jones_rho(p)
        mass_mev = 0.1408 * rho
        ax.set_title(
            f"$T(2,{p})$\n$\\rho_J = {rho:.1f}$,  $m = {mass_mev:.1f}$ MeV",
            fontsize=9,
        )

        # Nettoyer les axes
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

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = "output/dark_matter_torus_knots.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Planche sauvegardée : {out}")

    # Aussi générer des vues individuelles haute résolution
    for p in p_values:
        curve = _torus_knot_curve(2, p, n_points=3000)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        z = curve[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)

        for i in range(len(curve) - 1):
            ax.plot(
                curve[i:i+2, 0], curve[i:i+2, 1], curve[i:i+2, 2],
                color=plt.cm.coolwarm(z_norm[i]),
                linewidth=1.5,
            )

        rho = torus_jones_rho(p)
        mass_mev = 0.1408 * rho
        ax.set_title(
            f"Nœud torique $T(2,{p})$  —  matière noire\n"
            f"$c = {p}$,  $\\rho_J = {rho:.1f}$,  $m = {mass_mev:.1f}$ MeV",
            fontsize=12,
        )
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

        plt.tight_layout()
        out_i = f"output/dark_matter_T2_{p}.png"
        plt.savefig(out_i, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_i}")


if __name__ == "__main__":
    run()
