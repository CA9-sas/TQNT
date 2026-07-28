"""V5.36a — Test runner : c_J(3_1) depuis SU(2)_135 par 10 méthodes.

Évalue 10 définitions candidates de ``c_J(3_1)`` :

* M1 — span + γ·|V_K(-1)| = 3,3 (canonique V5)
* M2 — |V_K(q²)| (Wilson loop spin-1/2)
* M3 — |V_K(q)| / d_{1/2}(k=135) (RT normalisé)
* M4 — Mahler measure
* M5 — Σ|a_n| (norme L1)
* M6 — √Σa_n² (norme L2)
* M7 — span + 2/3 (heuristique, sans interprétation TQNT)
* M8 — exp(-β_calibré) (tautologique, contrôle)
* M9 — span · d_{1/2} / √3 (test SU(2)_135 direct)
* M10 — Kashaev limit (torus knot, vol=0 → 1)

Cible : c_J = e^{1,30} ≈ 3,669 pour β = -ln(c_J) = -1,30 (V5.35b calibré).

Sortie graphique : output/v5p36a_cJ.png — barplots horizontaux des écarts
relatifs à la cible (vert : first-principles, rouge hachuré : heuristique).

Verdict honnête : aucune méthode first-principles SU(2)_135 ne reproduit
la cible 3,669 à <1 %.  Le code donne 3,30 (écart 8 %).  β reste calibré.

Référence : theory/v5p36a_cJ_first_principles.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p36a_cJ_first_principles import (
    BETA_CALIBRATED,
    TARGET_C_J_3_1,
    all_candidates,
    best_first_principles,
    rank_candidates,
    summary_report,
    verdict,
)


OUTPUT_DIR = "output"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "v5p36a_cJ.png")


def make_plot(out_path: str = OUTPUT_FIG) -> str:
    """Génère la figure des écarts c_J et β pour les 10 méthodes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cands = rank_candidates()
    names = [c.name for c in cands]
    errs_cJ = [c.relative_error_cJ * 100 for c in cands]
    errs_beta = [c.relative_error_beta * 100 for c in cands]
    fp = [c.is_first_principles for c in cands]
    colors = ["#1b6f3a" if f else "#9b5d4c" for f in fp]
    hatches = ["" if f else "//" for f in fp]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel gauche : écart sur c_J ---
    ax = axes[0]
    bars = ax.barh(range(len(names)), errs_cJ, color=colors,
                   edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("écart relatif à c_J = e^{1,30} ≈ 3,669 (%)")
    ax.axvline(1.0, color="green", lw=0.7, ls=":", label="STRONG (<1 %)")
    ax.axvline(5.0, color="orange", lw=0.7, ls=":", label="ACCEPTABLE (<5 %)")
    ax.set_title(f"c_J(3_1) — 10 méthodes (cible {TARGET_C_J_3_1:.4f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    # Annotation des valeurs
    for i, c in enumerate(cands):
        ax.text(min(errs_cJ[i] + 1, ax.get_xlim()[1] * 0.95),
                i, f"{c.value:.3f}", va="center", fontsize=8)

    # --- Panel droit : écart sur β = -ln(c_J) ---
    ax = axes[1]
    cands_b = sorted(cands, key=lambda c: c.relative_error_beta)
    names_b = [c.name for c in cands_b]
    errs_b = [c.relative_error_beta * 100 for c in cands_b]
    fp_b = [c.is_first_principles for c in cands_b]
    colors_b = ["#1b6f3a" if f else "#9b5d4c" for f in fp_b]
    hatches_b = ["" if f else "//" for f in fp_b]
    bars = ax.barh(range(len(names_b)), errs_b, color=colors_b,
                   edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches_b):
        bar.set_hatch(h)
    ax.set_yticks(range(len(names_b)))
    ax.set_yticklabels(names_b)
    ax.invert_yaxis()
    ax.set_xlabel("écart à β_calibré = -1,30 (%)")
    ax.axvline(0.3, color="green", lw=0.7, ls=":", label="STRONG (<0,3 %)")
    ax.axvline(1.0, color="orange", lw=0.7, ls=":", label="ACCEPTABLE (<1 %)")
    ax.set_title(f"β = -ln(c_J) (cible {BETA_CALIBRATED:+.4f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    for i, c in enumerate(cands_b):
        ax.text(min(errs_b[i] + 1, ax.get_xlim()[1] * 0.95),
                i, f"{c.beta:+.3f}", va="center", fontsize=8)

    fig.suptitle(
        "V5.36a — c_J(3_1) depuis SU(2)_135 : 10 méthodes (vert = first-principles, "
        "rouge hachuré = heuristique/tautologique)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    print(summary_report())
    print()
    out = make_plot()
    print(f"Figure sauvegardée : {out}")
    print()
    v = verdict()
    best = best_first_principles()
    print(f"VERDICT V5.36a : {v}")
    if best is not None:
        print(
            f"  Best FP : {best.name} ({best.formula}) "
            f"c_J = {best.value:.4f}, β = {best.beta:+.4f}, "
            f"écart β = {best.relative_error_beta * 100:.2f} %"
        )
    print()
    print(
        "HONNÊTE : la cible c_J = 3,669 ne sort d'aucune définition "
        "first-principles SU(2)_135 ; le code canonique (M1) donne 3,30 "
        "(β = -1,19, écart 8 %). β reste calibré en V5.35d."
    )


if __name__ == "__main__":
    main()
