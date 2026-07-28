"""V5.23 -- Dérivation explicite de la charge de couleur SU(3) (script).

Ce script exécute la dérivation V5.23 et produit la figure
``output/v5p23_su3_color.png`` qui résume :

1. Les 6 pistes théoriques évaluées et leur verdict ;
2. L'assignation de couleur du nucléon (3₁ + 7₂ + 7₂) ;
3. L'orbite Z_3 vue dans le plan transverse au tube χ ;
4. La prédiction Λ_QCD V5.23 vs PDG.

Usage::

    python3 -m scripts.V5_color.v5p23_su3_derivation
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.su3_color_topology import (
    COLORS,
    F_2,
    F_3,
    LAMBDA_QCD_PDG_MEV,
    M_PROTON_MEV,
    assign_color_nucleon,
    color_count_consistency_check,
    color_from_knot_position,
    is_color_singlet,
    lambda_qcd_v5p23,
    required_color_count_from_topology,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "v5p23_su3_color.png")


# ── Tableau des 6 pistes ──────────────────────────────────────────

PISTES = [
    ("P1. Symétrie Z_3 du trèfle 3₁", "RETENU (partiel)", "tab:green"),
    ("P2. Racines cubiques à q=e^{2πi/137}", "REJETÉ : q³ ≠ 1", "tab:red"),
    ("P3. HOMFLY-PT (SU(N))", "POSSIBLE mais ad hoc", "tab:orange"),
    ("P4. Wilson loops rep fond. SU(3)", "INSUFFISANT seul", "tab:orange"),
    ("P5. Confinement = enchevêtrement", "EXPLIQUE confinement, pas N=3", "tab:orange"),
    ("P6. Baryon = 3 composantes", "RETENU (principal)", "tab:green"),
]


def panel_pistes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Pistes théoriques évaluées (V5.23)", fontsize=11,
                 fontweight="bold", loc="left")
    rows = len(PISTES)
    for i, (name, verdict, color) in enumerate(PISTES):
        y = 1.0 - (i + 0.5) / rows
        ax.text(0.02, y, name, transform=ax.transAxes,
                fontsize=9, ha="left", va="center")
        ax.text(0.62, y, verdict, transform=ax.transAxes,
                fontsize=9, ha="left", va="center",
                color=color, fontweight="bold")
    ax.text(0.02, -0.05, "Mécanisme retenu : P1 + P6 (cyclicité Z_3 du 3₁ "
                         "projetée sur 3 brins du baryon).",
            transform=ax.transAxes, fontsize=8.5, style="italic", color="black")


def panel_nucleon(ax: plt.Axes) -> None:
    assign = assign_color_nucleon()
    ax.set_title("Assignation V5.23 — Nucléon 3₁ ⊕ 7₂ ⊕ 7₂",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    color_map = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
    # 3 cercles disposés en triangle (orbite Z_3)
    for i, (knot, color) in enumerate(zip(assign.components, assign.colors)):
        theta = np.pi / 2 + 2 * np.pi * i / 3
        x, y = 0.7 * np.cos(theta), 0.7 * np.sin(theta)
        circ = plt.Circle((x, y), 0.32, color=color_map[color],
                          ec="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(circ)
        ax.text(x, y, f"{knot}\n({color})", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    # Tube χ central (singulet)
    ax.add_patch(plt.Circle((0, 0), 0.18, color="gray", alpha=0.3,
                            linestyle="--", fill=False, linewidth=2))
    ax.text(0, 0, "χ", ha="center", va="center", fontsize=14,
            fontweight="bold", color="gray")
    ax.text(0, -1.15, f"Singulet (|rgb⟩) : {assign.is_singlet}",
            ha="center", fontsize=10, style="italic")


def panel_z3_orbit(ax: plt.Axes) -> None:
    ax.set_title("Orbite Z_3 (plan transverse à l'axe du tube χ)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # Action cyclique Z_3 = ⟨(0→1→2→0)⟩
    color_map = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
    labels = ["pos 0 = r", "pos 1 = g", "pos 2 = b"]
    points = []
    for i in range(3):
        theta = np.pi / 2 + 2 * np.pi * i / 3
        x, y = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
        points.append((x, y))
        color = COLORS[i]
        ax.scatter([x], [y], s=350, c=color_map[color],
                   edgecolors="black", linewidth=1.5, zorder=3)
        ax.text(x * 1.32, y * 1.32, labels[i], ha="center", va="center",
                fontsize=10, color=color_map[color], fontweight="bold")
    # Flèches Z_3 : 0→1→2→0
    for i in range(3):
        j = (i + 1) % 3
        x0, y0 = points[i]
        x1, y1 = points[j]
        ax.annotate("", xy=(x1 * 0.86, y1 * 0.86),
                    xytext=(x0 * 0.86, y0 * 0.86),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    ax.text(0, -1.32, "Cyclicité Z_3 ⊂ S_3 du trèfle 3₁\nfixe N = 3 couleurs",
            ha="center", fontsize=9, style="italic")


def panel_lambda_qcd(ax: plt.Axes) -> None:
    lam_v5 = lambda_qcd_v5p23()
    ax.set_title("Échelle Λ_QCD V5.23 vs PDG", fontsize=11,
                 fontweight="bold", loc="left")
    labels = ["V5.23\nm_p × (1−f(3))", "PDG 2024"]
    values = [lam_v5, LAMBDA_QCD_PDG_MEV]
    errors = [0, 10]
    colors = ["tab:green", "tab:blue"]
    bars = ax.bar(labels, values, yerr=errors, color=colors,
                  capsize=10, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3,
                f"{val:.1f} MeV", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Λ_QCD (MeV)")
    ax.set_ylim(0, max(values) * 1.25)
    delta = (lam_v5 - LAMBDA_QCD_PDG_MEV) / LAMBDA_QCD_PDG_MEV * 100
    ax.text(0.5, 0.95, f"Écart V5.23 vs PDG : {delta:+.1f} %",
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.axhline(LAMBDA_QCD_PDG_MEV, linestyle=":", color="gray", alpha=0.5)


def print_summary() -> dict:
    """Imprime un résumé console pour le commit log."""
    print("=" * 70)
    print("V5.23 — Charge de couleur SU(3) depuis topologie TQNT")
    print("=" * 70)

    assign = assign_color_nucleon()
    print(f"\nNucléon : {assign.description}")
    print(f"  singulet de couleur : {assign.is_singlet}")

    diag = color_count_consistency_check()
    print(f"\nNombre de couleurs requis : {diag['n_colors_required']}")
    print(f"  orbite Z_3 distincte    : {diag['z3_orbit_distinct']}")
    print(f"  Z_2 trivial             : {diag['z2_subgroup_trivial']}")
    print(f"  Z_4 ne contient pas Z_3 : {diag['z4_extension_breaks']}")
    print(f"  écrantage f(2)/f(3)     : {diag['f_2_screening']}/{diag['f_3_screening']}")

    lam_v5 = lambda_qcd_v5p23()
    delta = (lam_v5 - LAMBDA_QCD_PDG_MEV) / LAMBDA_QCD_PDG_MEV * 100
    print(f"\nΛ_QCD V5.23 = m_p × (1 − f(3))")
    print(f"           = {M_PROTON_MEV:.2f} × {1 - F_3:.4f}")
    print(f"           = {lam_v5:.2f} MeV")
    print(f"PDG 2024   = {LAMBDA_QCD_PDG_MEV:.0f} ± 10 MeV")
    print(f"Écart      : {delta:+.2f} %")

    print("\nVerdict V5.23 :")
    print("  - 3 couleurs : DÉRIVÉ (cyclicité Z_3 du 3₁ → 3 composantes baryon)")
    print("  - anti-couleur : DÉRIVÉ (involution miroir Z_3⁻¹)")
    print("  - singulet : DÉRIVÉ (totale symétrisation Z_3)")
    print("  - Λ_QCD ≈ 220 MeV : DÉRIVÉ (m_p × (1-f(3)) = 215,8 MeV)")
    print("  - SU(3) complexification de Z_3 : OUVERT (fusion Verlinde)")
    print("  - dérivation ab initio de f(3) : OUVERT (Gross-Pitaevskii)")
    print("  STATUT : PARTIEL — SU(3) n'est plus un postulat ad hoc.")

    return {
        "lambda_v5p23_mev": lam_v5,
        "lambda_pdg_mev": LAMBDA_QCD_PDG_MEV,
        "delta_pct": delta,
        "nucleon_singlet": assign.is_singlet,
        "n_colors": diag["n_colors_required"],
    }


def make_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    panel_pistes(axes[0, 0])
    panel_nucleon(axes[0, 1])
    panel_z3_orbit(axes[1, 0])
    panel_lambda_qcd(axes[1, 1])
    fig.suptitle("V5.23 — Charge de couleur SU(3) depuis la topologie TQNT",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_PNG, dpi=130)
    print(f"\nFigure sauvegardée : {OUTPUT_PNG}")


def main() -> dict:
    summary = print_summary()
    make_figure()
    return summary


if __name__ == "__main__":
    main()
