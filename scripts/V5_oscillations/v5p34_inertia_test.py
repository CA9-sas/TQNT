"""V5.34 — Inertie chirale collective + force centripète — test runner.

Δm V5.34(L) = Δm V5.30c(L) × R_χ(L),
    R_χ      = ω_χ(L) / ω_χ(K⁰),
    ω_χ²     = F_torque / I_χ,
    F_torque = Σ_{i≠j} |τ_ij|   (couplage V5.33),
    I_χ      = formule candidate (5 variantes).

Sortie : output/v5p34_inertia.png
Référence : theory/v5p34_chiral_inertia.py
            theory/v5p33_mutual_EM_oscillation.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p34_chiral_inertia import (
    FORMULAS,
    compare_5_formulas,
    v5p34_table,
)


def print_compare(results):
    print("=" * 88)
    print("V5.34 — Comparaison des 5 formules d'inertie chirale I_χ")
    print("=" * 88)
    print(
        f"{'Formule':<14} {'param':>10} {'#param':>7} "
        f"{'Σ|log| (dex)':>14} {'B_s/B':>9}"
    )
    print("-" * 88)
    for s in results["summary"]:
        print(
            f"{s['formula']:<14} {s['kappa']:>+10.4f} {s['n_params']:>7d} "
            f"{s['sum_abs_log']:>14.3f} {s['ratio_Bs_B']:>9.2f}"
        )
    print(f"\nMeilleure formule : {results['best_formula']}")
    print()


def print_best_details(results):
    best = results["best_formula"]
    block = results[best]
    print("=" * 88)
    print(f"V5.34 — Détails de la meilleure formule : {best}")
    print("=" * 88)
    if "kappa" in block:
        print(
            f"paramètre calibré : κ* = {block['kappa']:+.4f} "
            f"({block.get('n_params', 0)} ddl libre)"
        )
    print(
        f"Σ|log| = {block['sum_abs_log']:.3f} dex  "
        f"|  ratio B_s/B = {block['ratio_Bs_B']:.2f} (PDG = 35.05)"
    )
    print()
    print(
        f"{'Méson':<6} {'Δm V5.34':>12} {'Δm PDG':>12} "
        f"{'log_ratio':>11} {'R_χ':>9} {'F_torque':>10} {'I_χ':>14}"
    )
    print("-" * 88)
    per = block["per_meson"]
    for name in ("K0", "D0", "B0", "Bs0"):
        p = per[name]
        print(
            f"{name:<6} {p['delta_m_MeV']:>12.3e} {p['delta_m_pdg_MeV']:>12.3e} "
            f"{p['log_ratio']:>+11.3f} {p['R_chi']:>9.3f} "
            f"{p['F_torque']:>10.3f} {p['I_chi']:>14.4e}"
        )
    print()
    return best, block


def print_history(results):
    """Σ|log| comparé à V5.30c (1.05), V5.31 (1.46), V5.33 (1.06)."""
    print("=" * 88)
    print("V5.34 — Plafond Σ|log| par version")
    print("=" * 88)
    history = [
        ("V5.30c", 1.051, "préfacteur vibratoire scalaire"),
        ("V5.31",  1.46,  "chiralité asymétrique"),
        ("V5.33",  1.06,  "EM mutuelle (λ_max du couplage)"),
    ]
    best_label = results["best_formula"]
    history.append(
        (
            f"V5.34 ({best_label})",
            results[best_label]["sum_abs_log"],
            "inertie chirale + force centripète",
        )
    )
    for tag, sval, desc in history:
        print(f"  {tag:<18} Σ|log| = {sval:.3f} dex   ({desc})")
    print()


def verdict(results):
    print("=" * 88)
    print("Verdict V5.34 (honnête)")
    print("=" * 88)
    best = results["best_formula"]
    s = results[best]["sum_abs_log"]
    rBsB = results[best]["ratio_Bs_B"]
    closes_ratio = 25.0 <= rBsB <= 50.0
    if s < 0.5 and closes_ratio:
        label = "BREAKTHROUGH"
        print(
            "VERDICT : BREAKTHROUGH — V5.34 ferme à la fois Σ|log| < 0.5 dex\n"
            f"   et le ratio B_s/B (≈ 35) ; la meilleure formule est {best}\n"
            "   avec UN paramètre nouveau au maximum."
        )
    elif s < 1.0:
        label = "PARTIAL"
        print(
            f"VERDICT : PARTIAL — V5.34 ({best}) descend Σ|log| à {s:.3f} dex,\n"
            f"   sous le plafond V5.30c/V5.33 (~1.05 dex), mais le ratio\n"
            f"   B_s/B = {rBsB:.1f} reste loin de la valeur PDG (35).\n"
            "   L'inertie chirale réduit l'écart absolu sans expliquer\n"
            "   la pente B_s/B."
        )
        if best == "F5_kashaev":
            print(
                "\n   ATTENTION : la formule gagnante est F5 (volume Kashaev\n"
                "   exponentiel) — son succès est partiellement importé du\n"
                "   volume hyperbolique tabulé.  Légitime mais à signaler."
            )
    elif s < 1.1:
        label = "NEUTRAL"
        print(
            f"VERDICT : NEUTRAL — V5.34 ({best}) donne Σ|log| = {s:.3f} dex,\n"
            f"   essentiellement équivalent à V5.30c (1.05) et V5.33 (1.06).\n"
            "   L'inertie chirale n'apporte pas d'amélioration significative."
        )
    else:
        label = "FAILURE"
        print(
            f"VERDICT : FAILURE — V5.34 ({best}) donne Σ|log| = {s:.3f} dex,\n"
            f"   au-dessus du plafond V5.30c/V5.33 (~1.05 dex).\n"
            "   L'inertie chirale collective ne suffit pas."
        )
    print()
    return label


def make_plot(results, out_path: str = "output/v5p34_inertia.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Panneau 1 : Σ|log| par formule
    names = [s["formula"] for s in results["summary"]]
    sums = [s["sum_abs_log"] for s in results["summary"]]
    kappas = [s["kappa"] for s in results["summary"]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax1.bar(names, sums, color=colors)
    ax1.axhline(1.05, color="k", ls="--", lw=1, alpha=0.5, label="V5.30c (1.05)")
    ax1.axhline(0.5, color="g", ls=":", lw=1, alpha=0.5, label="seuil 0.5 dex")
    ax1.set_ylabel(r"$\Sigma|\log_{10}(\Delta m / \Delta m_{\rm PDG})|$ (dex)")
    ax1.set_title("V5.34 — Σ|log| par formule I_χ")
    ax1.legend(fontsize=9)
    for bar, k in zip(bars, kappas):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"κ={k:+.2f}",
            ha="center",
            fontsize=8,
        )
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panneau 2 : Δm par méson pour la meilleure formule
    best = results["best_formula"]
    per = results[best]["per_meson"]
    mesons = ["K0", "D0", "B0", "Bs0"]
    dm_v534 = [per[m]["delta_m_MeV"] for m in mesons]
    dm_pdg = [per[m]["delta_m_pdg_MeV"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35
    ax2.bar(x - w / 2, dm_v534, w, label=f"V5.34 ({best})", color="#9467bd")
    ax2.bar(x + w / 2, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesons)
    ax2.set_ylabel(r"$\Delta m$ (MeV)")
    ax2.set_title(f"V5.34 — Δm meilleure formule ({best})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print(
        "V5.34 — Inertie chirale collective + force centripète\n"
        "Δm V5.34(L) = Δm V5.30c(L) × ω_χ(L)/ω_χ(K⁰),\n"
        "    ω_χ² = F_torque(L) / I_χ(L)\n"
        f"    I_χ ∈ {{{', '.join(FORMULAS)}}}\n"
    )
    results = compare_5_formulas()
    print_compare(results)
    best, _ = print_best_details(results)
    print_history(results)
    label = verdict(results)
    make_plot(results)
    # Pour usage en pipeline
    print(f"Label final : {label}")
    return label


if __name__ == "__main__":
    main()
