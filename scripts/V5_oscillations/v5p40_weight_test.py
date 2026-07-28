"""V5.40 - Test du weight system v_4 explicite + verification delta exact.

Genere output/v5p40_weights.png avec :
  1. Coefficients w_sl2(D) pour les 5 classes principales de diagrammes
  2. delta predit vs k (limite classique vs quantique k=135)
  3. Dimension primitive P_4 + facteur 4 explicite

Reference : theory/v5p40_weight_system_v4.py
            theory/v5p39_vassiliev_v4.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p38_delta_derivation import delta_first_principles
from theory.v5p40_weight_system_v4 import (
    CASIMIR_SPIN_HALF,
    chord_diagrams_4_chord_3_circle,
    delta_from_weight_system,
    factor_4_status,
    primitive_dimension,
    summary_report,
    verify_casimir_consistency,
    weight_v4_su2,
)


def print_report() -> None:
    print(summary_report())


def print_status() -> None:
    print()
    print("=" * 88)
    print("V5.40 - Statut honnete du facteur 4")
    print("=" * 88)
    status = factor_4_status()
    for key, value in status.items():
        if isinstance(value, list):
            print(f"  {key:32s} :")
            for item in value:
                print(f"                                       - {item}")
        else:
            print(f"  {key:32s} : {value}")


def make_plot(out_path: str = "output/v5p40_weights.png") -> None:
    diagrams = chord_diagrams_4_chord_3_circle()
    weights = [weight_v4_su2(D) for D in diagrams]
    labels = ["D_K\n(Kontsevich)", "D_B\n(Boucles)", "D_T\n(Triangle)",
              "D_S\n(Split)", "D_X\n(Crossing)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    # Panel 1 : Weight system coefficients
    weight_floats = [float(w) for w in weights]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax1.bar(labels, weight_floats, color=colors)
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel(r"$w_{\mathfrak{sl}_2, V_{1/2}}(D)$")
    ax1.set_title(
        "V5.40 - Weight system $\\mathfrak{sl}_2$ pour 5 classes de diagrammes\n"
        "ordre 4 sur 3 cercles (Casimir = $3/4$, dim $V_{1/2} = 2$)"
    )
    for i, (D, w_frac, w_f) in enumerate(zip(diagrams, weights, weight_floats)):
        ax1.text(
            i, w_f,
            f" {w_frac}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
        ax1.text(
            i, -0.15,
            f"(intra={D.intra_chord_count},\ninter={D.inter_chord_count})",
            ha="center", va="top",
            fontsize=8, color="gray",
        )
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(-0.3, max(weight_floats) * 1.2)

    # Panel 2 : delta universel (classique vs quantique)
    ks = np.logspace(1, 4, 50)
    deltas_quantum = [-4.0 * np.log(2.0 * np.cos(np.pi / (k + 2))) for k in ks]
    ax2.plot(ks, deltas_quantum, "-", color="#1f77b4", lw=2,
             label=r"$\delta(k) = -4\ln d_{1/2}(k)$ (quantique)")
    ax2.axhline(
        -4.0 * np.log(2.0), color="#d62728", ls="--", lw=1.5,
        label=r"limite $k \to \infty$ : $-\ln 16 = -4\ln 2$ (classique)",
    )
    ax2.axhline(
        delta_first_principles(), color="#2ca02c", ls=":", lw=1.5,
        label=f"$\\delta$(V5.38, k=135) = {delta_first_principles():+.6f}",
    )
    ax2.axvline(135, color="k", ls=":", alpha=0.6, label="k = 135 (TQNT)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Chern-Simons level k")
    ax2.set_ylabel(r"$\delta$")
    ax2.set_title(
        r"V5.40 - $\delta = -\dim P_4 \cdot \ln d_{1/2}(k)$"
        "\n"
        r"avec $\dim P_4 = 4$ (Vassiliev primitif 3-link mod 4T)"
    )
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Panel 3 : decomposition factor 4 = dim P_4
    res = delta_from_weight_system()
    components = {
        "Casimir $C_2$": float(CASIMIR_SPIN_HALF),
        "dim $V_{1/2}$": 2.0,
        "$d_{1/2}(135)$": float(np.exp(-res.delta_quantum / 4)),
        "$d_{1/2}(\\infty)$": 2.0,
    }
    ax3.bar(list(components.keys()), list(components.values()),
            color=["#9467bd", "#8c564b", "#1f77b4", "#d62728"])
    ax3.set_ylabel("valeur")
    ax3.set_title(
        f"V5.40 - Composants du calcul explicite\n"
        f"facteur 4 = dim $P_4$ = {primitive_dimension()} "
        f"(VERDICT : {factor_4_status()['verdict']})"
    )
    for i, (label, v) in enumerate(components.items()):
        ax3.text(i, v, f" {v:.6f}", ha="center", va="bottom", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.set_ylim(0, max(components.values()) * 1.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegarde : {out_path}")


def main() -> None:
    print()
    print(
        "V5.40 - Coefficients explicites weight system v_4 pour 3-link\n"
        "    factor 4 derive comme dim P_4 (Bar-Natan / Chmutov-Lando)\n"
    )
    print_report()
    print_status()
    make_plot()


if __name__ == "__main__":
    main()
