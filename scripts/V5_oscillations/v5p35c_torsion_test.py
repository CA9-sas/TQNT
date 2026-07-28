"""V5.35c — Torsion (writhe signé) vs linking dans l'inertie chirale — runner.

Compare Δm V5.35c (writhe signé inter-composantes, zéro paramètre
libre nouveau) à Δm V5.34 F3_coupling (Lk_eff, Σ|log| = 0,82 dex).

Hypothèse minimale : toutes les composantes chirales d'un même méson
sont alignées (sign +1).  Les amphi (6_1, 6_3) ont writhe = 0 → coupling
chiral inter-composantes nul pour ces liens.

Sortie : output/v5p35c_torsion.png
Référence : theory/v5p35c_torsion_writhe.py
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
from theory.v5p35c_torsion_writhe import (
    _WRITHE_TABLE,
    knot_writhe,
    summary_v5p35c,
    v5p35c_table,
)


SIGMA_LOG_V5p34 = 0.82  # référence V5.34 F3_coupling
SIGMA_LOG_V5p30c = 1.051
SIGMA_LOG_V5p33 = 1.06


def print_writhe_table():
    print("=" * 88)
    print("V5.35c — Table writhe utilisée (Rolfsen / KnotInfo)")
    print("=" * 88)
    used = set()
    for cfg in MESON_OSCILLATION_ASSIGNMENTS.values():
        used.update(cfg["knots"])
    print(f"{'knot':<8} {'|w|':>6} {'chiral?':>9}")
    print("-" * 30)
    for k in sorted(used):
        w = knot_writhe(k)
        chiral = "chiral" if w != 0 else "amphi"
        print(f"{k:<8} {w:>6d} {chiral:>9}")
    print()


def print_per_meson(summary):
    print("=" * 88)
    print("V5.35c — Δm par méson (writhe signé)")
    print("=" * 88)
    print(
        f"{'Méson':<6} {'Knots':<22} {'Δm V5.35c':>12} {'Δm PDG':>12} "
        f"{'log_ratio':>10} {'I_self':>10} {'I_inter':>10} {'signs':>10}"
    )
    print("-" * 100)
    for name in ("K0", "D0", "B0", "Bs0"):
        r = summary["per_meson"][name]
        knots_str = "+".join(r["knots"])
        signs_str = "[" + ",".join(f"{s:+d}" for s in r["writhe_signs"]) + "]"
        print(
            f"{name:<6} {knots_str:<22} {r['delta_m_v5p35c_MeV']:>12.3e} "
            f"{r['delta_m_pdg_MeV']:>12.3e} {r['log_ratio']:>+10.3f} "
            f"{r['I_self']:>10.2f} {r['I_inter']:>10.2f} {signs_str:>10}"
        )
    print()
    print(
        f"Σ|log|        = {summary['sum_abs_log']:.3f} dex"
        f"   (V5.34 = {SIGMA_LOG_V5p34:.3f},  V5.30c = {SIGMA_LOG_V5p30c:.3f})"
    )
    print(
        f"ratio B_s/B  = {summary['ratio_Bs_B']:.2f}"
        f"   (PDG = 35.05)"
    )
    print()


def verdict(summary):
    s = summary["sum_abs_log"]
    rBsB = summary["ratio_Bs_B"]
    closes_ratio = 25.0 <= rBsB <= 50.0
    log_B0 = summary["per_meson"]["B0"]["log_ratio"]
    print("=" * 88)
    print("Verdict V5.35c (honnête)")
    print("=" * 88)
    if s < 0.5 and closes_ratio:
        label = "BREAKTHROUGH"
        print(
            f"VERDICT : BREAKTHROUGH — Σ|log| = {s:.3f} dex < 0.5 et le ratio\n"
            f"   B_s/B = {rBsB:.2f} est dans [25, 50].  Le writhe signé\n"
            "   capture l'asymétrie chirale absente du |Lk_eff|."
        )
    elif s < SIGMA_LOG_V5p34 - 0.05:
        label = "IMPROVEMENT"
        print(
            f"VERDICT : IMPROVEMENT — Σ|log| = {s:.3f} dex < V5.34 ({SIGMA_LOG_V5p34}).\n"
            f"   Le writhe signé améliore V5.34, mais ratio B_s/B = {rBsB:.1f}\n"
            f"   reste {'OK' if closes_ratio else 'loin'} de 35.\n"
            f"   log(B⁰) = {log_B0:+.3f} (cible : ≈ 0)."
        )
    elif s < SIGMA_LOG_V5p34 + 0.05:
        label = "NEUTRAL"
        print(
            f"VERDICT : NEUTRAL — Σ|log| = {s:.3f} dex ≈ V5.34 ({SIGMA_LOG_V5p34}).\n"
            "   Le writhe signé ne change pas la qualité globale ;\n"
            "   il redistribue les résidus."
        )
    else:
        label = "REGRESSION"
        print(
            f"VERDICT : REGRESSION — Σ|log| = {s:.3f} dex > V5.34 ({SIGMA_LOG_V5p34}).\n"
            "   Le |Lk_eff| de V5.34 capturait bien quelque chose qui\n"
            "   disparaît quand on isole le writhe : amphichiraux\n"
            "   contribuent réellement à l'inertie chirale par leur\n"
            "   contact géométrique, pas seulement par leur signe."
        )
    print()
    return label


def make_plot(summary, out_path: str = "output/v5p35c_torsion.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    mesons = ["K0", "D0", "B0", "Bs0"]
    per = summary["per_meson"]
    dm_v5p35c = [per[m]["delta_m_v5p35c_MeV"] for m in mesons]
    dm_pdg = [per[m]["delta_m_pdg_MeV"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35

    ax1.bar(x - w / 2, dm_v5p35c, w, color="#9467bd", label="V5.35c (writhe)")
    ax1.bar(x + w / 2, dm_pdg, w, color="#ff7f0e", label="PDG")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mesons)
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title("V5.35c — Δm vs PDG")
    ax1.grid(True, axis="y", which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    log_r = [per[m]["log_ratio"] for m in mesons]
    colors = ["#1f77b4" if abs(v) < 0.5 else "#d62728" for v in log_r]
    ax2.bar(mesons, log_r, color=colors)
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(0.5, color="g", ls=":", lw=1, alpha=0.5)
    ax2.axhline(-0.5, color="g", ls=":", lw=1, alpha=0.5)
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{V5.35c} / \Delta m_{PDG})$ (dex)")
    ax2.set_title(
        f"V5.35c — résidus  (Σ|log| = {summary['sum_abs_log']:.3f}, "
        f"V5.34 ref = {SIGMA_LOG_V5p34})"
    )
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print(
        "V5.35c — Torsion writhe vs linking dans l'inertie chirale\n"
        "Δm V5.35c(L) = Δm V5.30c(L) × ω_χ(L)/ω_χ(K⁰),\n"
        "    ω_χ² = F_torque / I_χ V5.35c,\n"
        "    I_χ  = Σ c³/b² + Σ_{i<j} √(c_i c_j) b_i b_j × w_ij_signed,\n"
        "    w_ij_signed = ± 1 chiral × chiral, 0 si amphi.\n"
    )
    print_writhe_table()
    summary = summary_v5p35c()
    print_per_meson(summary)
    label = verdict(summary)
    make_plot(summary)
    print(f"Label final : {label}")
    return label


if __name__ == "__main__":
    main()
