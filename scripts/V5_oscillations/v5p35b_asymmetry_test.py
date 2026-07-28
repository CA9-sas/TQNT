"""V5.35b — Asymétrie de complexité dans le coupling V5.34 — test runner.

    J_ij V5.35b = J_ij V5.34 × A_ij(c_i, c_j; β)
    A_ij rel   = 1 + β × |c_i − c_j| / max(c_i, c_j)
    A_ij ratio = (max(c_i, c_j) / min(c_i, c_j))^β

Δm V5.35b(L) = Δm V5.30c(L) × ω_χ(L) / ω_χ(K⁰)
    ω_χ² = F_torque / I_χ V5.35b
    I_χ V5.35b = Σ I_i + Σ_{i<j} J_ij V5.35b × |Lk_eff|

UN paramètre libre β.  Calibration sur {K⁰, D⁰, B⁰, B_s⁰}.

Sortie : output/v5p35b_asymmetry.png
Référence : theory/v5p35b_complexity_asymmetry.py
            theory/v5p34_chiral_inertia.py
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
from theory.v5p35b_complexity_asymmetry import (
    FORMS,
    compare_forms,
)


def print_compare(results):
    print("=" * 88)
    print("V5.35b — Comparaison des deux formes d'asymétrie A_ij")
    print("=" * 88)
    print(
        f"{'Form':<8} {'β*':>10} {'Σ|log| (dex)':>14} {'B_s/B':>9}"
        f"  {'log_r K⁰':>10}  {'log_r D⁰':>10}  {'log_r B⁰':>10}  {'log_r B_s':>10}"
    )
    print("-" * 88)
    for form in FORMS:
        block = results[form]
        per = block["per_meson"]
        print(
            f"{form:<8} {block['beta']:>+10.4f} {block['sum_abs_log']:>14.3f} "
            f"{block['ratio_Bs_B']:>9.2f}  "
            f"{per['K0']['log_ratio']:>+10.3f}  "
            f"{per['D0']['log_ratio']:>+10.3f}  "
            f"{per['B0']['log_ratio']:>+10.3f}  "
            f"{per['Bs0']['log_ratio']:>+10.3f}"
        )
    print(f"\nForme retenue : {results['best_form']}")
    print()


def print_details(results, form: str):
    block = results[form]
    print("=" * 88)
    print(f"V5.35b — Détails forme {form}")
    print("=" * 88)
    print(
        f"β* = {block['beta']:+.4f}  |  Σ|log| = {block['sum_abs_log']:.3f} dex  "
        f"|  ratio B_s/B = {block['ratio_Bs_B']:.2f} (PDG = 35.05)"
    )
    print()
    print(
        f"{'Méson':<6} {'Δm V5.35b':>12} {'Δm PDG':>12} "
        f"{'log_ratio':>11} {'R_χ':>9} {'F':>9} {'I_χ':>11} {'I_int':>11}"
    )
    print("-" * 88)
    per = block["per_meson"]
    for name in ("K0", "D0", "B0", "Bs0"):
        p = per[name]
        print(
            f"{name:<6} {p['delta_m_MeV']:>12.3e} {p['delta_m_pdg_MeV']:>12.3e} "
            f"{p['log_ratio']:>+11.3f} {p['R_chi']:>9.3f} "
            f"{p['F_torque']:>9.3f} {p['I_chi']:>11.3e} {p['I_int']:>11.3e}"
        )
    print()


def print_history(results):
    print("=" * 88)
    print("V5.35b — Plafond Σ|log| par version")
    print("=" * 88)
    history = [
        ("V5.30c",        1.051, "préfacteur vibratoire scalaire"),
        ("V5.31",         1.460, "chiralité asymétrique"),
        ("V5.33",         1.060, "EM mutuelle (λ_max)"),
        ("V5.34 (F3)",    0.820, "inertie chirale F3_coupling"),
    ]
    best = results["best_form"]
    history.append(
        (
            f"V5.35b ({best})",
            results[best]["sum_abs_log"],
            "asymétrie complexité × J_ij",
        )
    )
    for tag, s, desc in history:
        print(f"  {tag:<18} Σ|log| = {s:.3f} dex   ({desc})")
    print()


def verdict(results):
    print("=" * 88)
    print("Verdict V5.35b (honnête)")
    print("=" * 88)
    best = results["best_form"]
    s = results[best]["sum_abs_log"]
    rBsB = results[best]["ratio_Bs_B"]
    log_B = results[best]["per_meson"]["B0"]["log_ratio"]
    sigma_34 = 0.820
    if s < 0.5 and abs(log_B) < 0.2:
        label = "BREAKTHROUGH"
        print(
            f"VERDICT : BREAKTHROUGH — V5.35b ({best}) ferme à la fois\n"
            f"   Σ|log| < 0.5 dex et le déficit B⁰ < 0.2 dex avec un\n"
            f"   seul paramètre β = {results[best]['beta']:+.3f}."
        )
    elif s < sigma_34 - 0.05:
        label = "PARTIAL"
        print(
            f"VERDICT : PARTIAL — V5.35b ({best}) descend Σ|log| de\n"
            f"   {sigma_34:.3f} (V5.34) à {s:.3f} dex.\n"
            f"   B⁰ log_r = {log_B:+.3f} — l'asymétrie corrige K⁰/D⁰/B_s⁰\n"
            f"   mais le déficit B⁰ persiste partiellement.\n"
            f"   Le levier J_ij × A_ij capture une partie mais pas tout."
        )
    elif s < sigma_34 + 0.05:
        label = "NEUTRAL"
        print(
            f"VERDICT : NEUTRAL — V5.35b ({best}) donne Σ|log| = {s:.3f},\n"
            f"   essentiellement équivalent à V5.34 ({sigma_34:.3f}).\n"
            f"   L'asymétrie de complexité n'apporte pas d'amélioration nette."
        )
    else:
        label = "FAILURE"
        print(
            f"VERDICT : FAILURE — V5.35b ({best}) donne Σ|log| = {s:.3f},\n"
            f"   au-dessus de V5.34 (0.820).  L'asymétrie dégrade."
        )
    print(f"   ratio B_s/B = {rBsB:.2f} (PDG = 35.05).")
    print()
    return label


def make_plot(results, out_path: str = "output/v5p35b_asymmetry.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Panneau 1 : Σ|log| par forme + référence V5.34
    forms_lbl = list(FORMS) + ["V5.34 (F3)"]
    sums = [results[f]["sum_abs_log"] for f in FORMS] + [0.820]
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
    bars = ax1.bar(forms_lbl, sums, color=colors)
    ax1.axhline(0.5, color="g", ls=":", lw=1, alpha=0.6, label="seuil 0.5 dex")
    ax1.axhline(0.820, color="k", ls="--", lw=1, alpha=0.6, label="V5.34 (0.820)")
    ax1.set_ylabel(r"$\Sigma|\log_{10}(\Delta m / \Delta m_{\rm PDG})|$ (dex)")
    ax1.set_title("V5.35b — Σ|log| par forme A_ij vs V5.34")
    ax1.legend(fontsize=9)
    for bar, f in zip(bars[:-1], FORMS):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"β={results[f]['beta']:+.2f}",
            ha="center",
            fontsize=9,
        )
    ax1.grid(True, axis="y", alpha=0.3)

    # Panneau 2 : Δm par méson pour la meilleure forme
    best = results["best_form"]
    per = results[best]["per_meson"]
    mesons = ["K0", "D0", "B0", "Bs0"]
    dm_v535b = [per[m]["delta_m_MeV"] for m in mesons]
    dm_pdg = [per[m]["delta_m_pdg_MeV"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35
    ax2.bar(x - w / 2, dm_v535b, w, label=f"V5.35b ({best})", color="#1f77b4")
    ax2.bar(x + w / 2, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesons)
    ax2.set_ylabel(r"$\Delta m$ (MeV)")
    ax2.set_title(f"V5.35b — Δm meilleure forme ({best})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print(
        "V5.35b — Asymétrie de complexité dans le coupling inertiel V5.34\n"
        "    J_ij V5.35b = J_ij V5.34 × A_ij(c_i, c_j ; β)\n"
        "    A_ij rel   = 1 + β × |c_i − c_j| / max(c_i, c_j)\n"
        "    A_ij ratio = (max / min)^β\n"
    )
    results = compare_forms()
    print_compare(results)
    print_details(results, results["best_form"])
    print_history(results)
    label = verdict(results)
    make_plot(results)
    print(f"Label final : {label}")
    return label


if __name__ == "__main__":
    main()
