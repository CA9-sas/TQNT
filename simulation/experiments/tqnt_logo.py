r"""
Génération du logo TQNT (Théorie Quantique des Nœuds Topologiques).

Produit plusieurs variantes du logo basé sur le nœud trèfle 3_1
(le plus simple nœud non-trivial, identifié à l'électron dans la TQNT) :

  - tqnt_logo_trefoil.png       : trèfle seul, bleu foncé sur transparent
  - tqnt_logo_full.png          : trèfle + texte "TQNT" sous le motif
  - tqnt_logo_compact.svg       : version vectorielle scalable
  - tqnt_logo_white.png         : version blanche pour fond sombre
  - tqnt_favicon_64.png         : 64×64 px favicon
  - tqnt_favicon_256.png        : 256×256 px favicon haute résolution
  - tqnt_social_card.png        : 1200×630 px carte Twitter/Open Graph

Sortie : docs/tqnt_logos/

Usage : python3 -m simulation.experiments.tqnt_logo
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from simulation.experiments.knot_diagrams_2d import plot_knot

OUT_DIR = "docs/tqnt_logos"
os.makedirs(OUT_DIR, exist_ok=True)

# Couleur principale TQNT : bleu nuit profond (cohérent avec hyperref)
COLOR_PRIMARY = "#1f3a93"
COLOR_LIGHT = "#FFFFFF"


# ──────────────────────────────────────────────────────────────────────────
# Variante 1 : trèfle seul, fond transparent
# ──────────────────────────────────────────────────────────────────────────
def render_trefoil_solo(filename: str, color: str = COLOR_PRIMARY,
                        size: float = 3.0, dpi: int = 300,
                        transparent: bool = True):
    """Trèfle 3_1 sur fond transparent, format carré."""
    fig, ax = plt.subplots(figsize=(size, size))
    plot_knot(ax, "3_1", n_points=2400, color=color, linewidth=4.5)
    ax.set_title("")
    fig.tight_layout(pad=0.0)
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=dpi,
                bbox_inches="tight", pad_inches=0.05,
                transparent=transparent)
    plt.close(fig)
    print(f"  ✓ {filename}")


# ──────────────────────────────────────────────────────────────────────────
# Variante 2 : trèfle + texte "TQNT" en bas
# ──────────────────────────────────────────────────────────────────────────
def render_trefoil_with_text(filename: str, text: str = "TQNT",
                              color: str = COLOR_PRIMARY,
                              size: float = 3.5, dpi: int = 300):
    """Trèfle 3_1 + texte 'TQNT' centré dessous."""
    fig, ax = plt.subplots(figsize=(size, size * 1.15))
    plot_knot(ax, "3_1", n_points=2400, color=color, linewidth=4.5)
    ax.set_title("")

    # ajouter texte TQNT sous le nœud
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.text(
        (xlim[0] + xlim[1]) / 2,
        ylim[0] - 0.20 * span,
        text,
        ha="center", va="top",
        fontsize=42, fontweight="bold",
        color=color,
        family="sans-serif",
    )
    # élargir la zone visible pour inclure le texte
    ax.set_ylim(ylim[0] - 0.45 * span, ylim[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.0)
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=dpi,
                bbox_inches="tight", pad_inches=0.1,
                transparent=True)
    plt.close(fig)
    print(f"  ✓ {filename}")


# ──────────────────────────────────────────────────────────────────────────
# Variante 3 : carte sociale 1200×630 (Twitter, Open Graph, LinkedIn)
# ──────────────────────────────────────────────────────────────────────────
def render_social_card(filename: str = "tqnt_social_card.png"):
    """Carte sociale 1200×630 (proportions Twitter / Open Graph standard)."""
    # Format Twitter : 1200×630 px → 12 × 6.3 inches à 100 dpi
    fig, ax = plt.subplots(figsize=(12, 6.3))
    fig.patch.set_facecolor("white")

    # Diviser horizontalement : trèfle à gauche, texte à droite
    # 1) Tracer le trèfle dans la portion gauche
    ax_trefoil = fig.add_axes([0.02, 0.05, 0.40, 0.90])
    plot_knot(ax_trefoil, "3_1", n_points=2400,
              color=COLOR_PRIMARY, linewidth=8)
    ax_trefoil.set_title("")

    # 2) Cacher l'axe principal
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 3) Ajouter le texte à droite
    fig.text(0.52, 0.72, "TQNT",
             fontsize=72, fontweight="bold",
             color=COLOR_PRIMARY, family="sans-serif")
    fig.text(0.52, 0.58,
             "Quantum Theory of",
             fontsize=24, color="#333333", family="sans-serif")
    fig.text(0.52, 0.50,
             "Topological Knots",
             fontsize=24, color="#333333", family="sans-serif")
    fig.text(0.52, 0.30,
             "github.com/CA9-sas/TQNT",
             fontsize=18, color="#777777", family="monospace")
    fig.text(0.52, 0.22,
             "doi:10.5281/zenodo.19131538",
             fontsize=18, color="#777777", family="monospace")
    fig.text(0.52, 0.10,
             "Lilian Cariou — CA9 SAS, France",
             fontsize=14, color="#999999", style="italic",
             family="sans-serif")

    fig.savefig(os.path.join(OUT_DIR, filename), dpi=100,
                bbox_inches="tight", pad_inches=0.0,
                facecolor="white")
    plt.close(fig)
    print(f"  ✓ {filename}  (1200×630 px)")


# ──────────────────────────────────────────────────────────────────────────
# Variante 4 : favicon (multi-résolutions)
# ──────────────────────────────────────────────────────────────────────────
def render_favicon(filename: str, size_px: int = 256):
    """Favicon carré format PNG (utilisé pour Zenodo community logo,
    favicon GitHub, avatar Twitter, etc.)"""
    size_in = size_px / 100.0  # ~2.56" pour 256px à 100dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in))

    # Fond carré bleu nuit
    fig.patch.set_facecolor(COLOR_PRIMARY)
    ax.set_facecolor(COLOR_PRIMARY)

    # Trèfle blanc sur fond
    plot_knot(ax, "3_1", n_points=2000,
              color=COLOR_LIGHT, linewidth=6)
    ax.set_title("")

    # Pas de marges
    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=100,
                bbox_inches="tight", pad_inches=0.05,
                facecolor=COLOR_PRIMARY)
    plt.close(fig)
    print(f"  ✓ {filename}  ({size_px}×{size_px} px)")


# ──────────────────────────────────────────────────────────────────────────
# Variante 5 : SVG vectoriel (scalable, qualité optimale)
# ──────────────────────────────────────────────────────────────────────────
def render_svg(filename: str = "tqnt_logo.svg",
               with_text: bool = True):
    """Version SVG (vectorielle, infiniment scalable)."""
    fig, ax = plt.subplots(figsize=(3, 3.5 if with_text else 3))
    plot_knot(ax, "3_1", n_points=2400,
              color=COLOR_PRIMARY, linewidth=4.5)
    ax.set_title("")

    if with_text:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        ax.text(
            (xlim[0] + xlim[1]) / 2,
            ylim[0] - 0.20 * span,
            "TQNT",
            ha="center", va="top",
            fontsize=42, fontweight="bold",
            color=COLOR_PRIMARY,
            family="sans-serif",
        )
        ax.set_ylim(ylim[0] - 0.45 * span, ylim[1])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.0)
    fig.savefig(os.path.join(OUT_DIR, filename),
                bbox_inches="tight", pad_inches=0.1,
                transparent=True, format="svg")
    plt.close(fig)
    print(f"  ✓ {filename}  (vector)")


# ──────────────────────────────────────────────────────────────────────────
def main():
    print("=== Génération du logo TQNT ===")
    print(f"  Sortie : {OUT_DIR}/\n")

    # 1) Trèfle seul, transparent (pour superposition)
    render_trefoil_solo("tqnt_logo_trefoil.png",
                        color=COLOR_PRIMARY,
                        size=3.0, dpi=300, transparent=True)

    # 2) Trèfle + texte TQNT
    render_trefoil_with_text("tqnt_logo_full.png",
                             text="TQNT",
                             color=COLOR_PRIMARY,
                             size=3.5, dpi=300)

    # 3) Version blanche pour fond sombre
    render_trefoil_solo("tqnt_logo_white.png",
                        color=COLOR_LIGHT,
                        size=3.0, dpi=300, transparent=True)

    # 4) Favicon 64×64
    render_favicon("tqnt_favicon_64.png", size_px=64)

    # 5) Favicon 256×256 (Zenodo community logo)
    render_favicon("tqnt_favicon_256.png", size_px=256)

    # 6) Carte sociale (Twitter / Open Graph / LinkedIn)
    render_social_card("tqnt_social_card.png")

    # 7) SVG vectoriel (scalable infini)
    render_svg("tqnt_logo.svg", with_text=True)
    render_svg("tqnt_logo_compact.svg", with_text=False)

    print(f"\n✓ Tous les logos générés dans {OUT_DIR}/")
    print("\nUsage recommandé :")
    print("  - tqnt_logo_full.png       → README GitHub, headers de pages")
    print("  - tqnt_logo_trefoil.png    → icône/avatar simple")
    print("  - tqnt_logo_white.png      → fond sombre / dark mode")
    print("  - tqnt_favicon_64.png      → favicon site web (16/32/64)")
    print("  - tqnt_favicon_256.png     → community Zenodo (limite 256 KB)")
    print("  - tqnt_social_card.png     → Twitter/Open Graph/LinkedIn cards")
    print("  - tqnt_logo.svg            → impression haute résolution")
    print("  - tqnt_logo_compact.svg    → embed inline HTML / impression")


if __name__ == "__main__":
    main()
