"""V5.36 — Test runner pour la dérivation first-principles de β et δ.

Évalue 10 candidats pour chaque paramètre, identifie le meilleur
candidat first-principles, vérifie la cohérence avec la fermeture
V5.35d (Σ|log| ≈ 0,01 dex).

Sortie console : table des 20 candidats triés par erreur relative,
verdict honnête (STRONG / ACCEPTABLE / WEAK / NONE).

Sortie graphique : output/v5p36_first_principles.png — barplots
horizontaux des écarts en pourcent, séparant first-principles
(barres pleines) de candidats heuristiques (barres hachurées).

Référence : theory/v5p36_first_principles_beta_delta.py
            theory/v5p35b_complexity_asymmetry.py
            theory/v5p35d_3body_multiplicative.py
            theory/su2_modular.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p36_first_principles_beta_delta import (
    BETA_CALIBRATED,
    DELTA_CALIBRATED,
    candidate_beta_formulas,
    candidate_delta_formulas,
    rank_candidates,
    summary_report,
    validate_against_calibrated_values,
)


OUTPUT_DIR = "output"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "v5p36_first_principles.png")


def _plot_block(ax, candidates, target, title):
    """Barplot horizontal des écarts relatifs (%)."""
    candidates = list(rank_candidates(candidates))
    names = [c.name for c in candidates]
    errs = [c.relative_error * 100 for c in candidates]
    fp = [c.is_first_principles for c in candidates]
    colors = ["#1b6f3a" if f else "#9b5d4c" for f in fp]
    hatches = ["" if f else "//" for f in fp]

    bars = ax.barh(
        range(len(names)),
        errs,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("écart relatif à la valeur calibrée (%)")
    ax.axvline(0.3, color="green", lw=0.7, ls=":", label="STRONG (<0,3 %)")
    ax.axvline(1.0, color="orange", lw=0.7, ls=":", label="ACCEPTABLE (<1 %)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)


def make_plot(out_path: str = OUTPUT_FIG) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _plot_block(
        axes[0],
        candidate_beta_formulas(),
        BETA_CALIBRATED,
        f"β candidats (cible {BETA_CALIBRATED:+.4f})",
    )
    _plot_block(
        axes[1],
        candidate_delta_formulas(),
        DELTA_CALIBRATED,
        f"δ candidats (cible {DELTA_CALIBRATED:+.4f})",
    )
    fig.suptitle(
        "V5.36 — dérivation first-principles β (V5.35b) et δ (V5.35d) — "
        "vert = first-principles, rouge hachuré = heuristique",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    print(summary_report())
    print()
    res = validate_against_calibrated_values()
    for tag in ("beta", "delta"):
        best = res[tag]["best_first_principles"]
        verdict = res[tag]["verdict"]
        if best is None:
            print(f"[{tag}] AUCUN candidat first-principles : verdict NONE.")
        else:
            print(
                f"[{tag}] Best FP : {best.name} | {best.formula} | "
                f"err = {best.relative_error*100:.2f} % | verdict {verdict}"
            )
            print(f"        Interprétation : {best.interpretation}")
    out = make_plot()
    print()
    print(f"Figure sauvegardée : {out}")


if __name__ == "__main__":
    main()
