#!/usr/bin/env python3
"""
V5.10b — Validation PN-1 précession périhélie sur les 6 planètes.

Reproduit la précession GR du périhélie de Mercure, Vénus, Terre, Mars,
Jupiter, Saturne à partir de la formule fermée PN-1 :

    Δφ = 6 π G M / (c² a (1 - e²))

V5 hérite ce résultat par sa dérivation V5.8 de l'équation d'Einstein
depuis le χ-field. La validation ne teste PAS une spécificité V5,
seulement la cohérence V5 ↔ GR en régime faible champ + 1PN.

Output :
  - output/v5p10b_perihelion_precession.png

Lancement :
    PYTHONPATH=. python3 scripts/V5_10b_PN_QNM/validate_perihelion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.post_newtonian_PN1 import (
    PLANETS,
    M_SUN,
    perihelion_precession_arcsec_per_century,
    validate_all_planets,
)


def main() -> int:
    print("=" * 72)
    print("V5.10b — VALIDATION PN-1 PÉRIHÉLIE (6 planètes)")
    print("=" * 72)
    print(f"{'Planet':<10} {'a (AU)':>8} {'e':>9} {'Predicted':>14} {'Observed':>14} {'Err %':>8}")
    print("-" * 72)
    results = validate_all_planets()
    for r in results:
        print(
            f"{r['name']:<10} {r['a_AU']:>8.4f} {r['e']:>9.5f}"
            f" {r['predicted_arcsec_per_century']:>14.4f}"
            f" {r['observed_arcsec_per_century']:>14.4f}"
            f" {100.0 * r['relative_error']:>7.2f}"
        )
    print("-" * 72)
    print("Unités : secondes d'arc par siècle.")
    print("Réf. observed : Will, Living Rev. Relativ. 17, 4 (2014).")
    print()
    print(f"Mercure : prédit {results[0]['predicted_arcsec_per_century']:.4f} ″/siècle,")
    print(f"observé  42,98 ″/siècle, écart {100.0 * results[0]['relative_error']:.3f} %.")
    print()
    print("Résultat GR standard, hérité par V5 via la dérivation V5.8.")
    print("Aucune spécificité TQNT revendiquée ici.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r["name"] for r in results]
    pred = np.array([r["predicted_arcsec_per_century"] for r in results])
    obs = np.array([r["observed_arcsec_per_century"] for r in results])
    err = 100.0 * np.array([r["relative_error"] for r in results])

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.18, obs, width=0.36, label="Observé (Will 2014)",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + 0.18, pred, width=0.36, label="Prédit PN-1",
           color="#d62728", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("Précession périhélie (″/siècle)")
    ax.set_title("PN-1 vs observation\n(échelle log)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.bar(names, err, color="#2ca02c", alpha=0.85)
    ax.set_ylabel("Écart relatif (%)")
    ax.set_title("Écart prédit / observé")
    ax.set_ylim(0, max(1.0, 1.2 * err.max()))
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(err):
        ax.text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=8)

    fig.suptitle("V5.10b — PN-1 précession périhélie (GR standard, hérité V5)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v5p10b_perihelion_precession.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
