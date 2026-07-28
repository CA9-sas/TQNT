"""V5.35d — 3-corps multiplicatif renormalisant — test runner.

    I_χ V5.35d(L) = I_χ V5.35b(L) × Π_{i<j<k} [1 + δ × f(c, b)]

Sept formes candidates f : var_c, seq_asym, prod_norm, minmax_mid,
irreg_idx, braid_asym, braid_prod.

Toutes vérifient f(K⁰) = 0 (composantes identiques 6_1 → triplet
homogène) — la calibration K⁰ est strictement préservée.

Baseline V5.35b : Σ|log| = 0.563 dex (B⁰ = -0.563 résidu).
Cible V5.35d : Σ|log| < 0.4 dex et |B⁰| < 0.2 dex.

Sortie : output/v5p35d_3body_mult.png
Référence : theory/v5p35d_3body_multiplicative.py
            theory/v5p35b_complexity_asymmetry.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p35d_3body_multiplicative import (
    FORMS,
    calibrate_best_form,
    v5p35d_table,
)


BASELINE_V5p35b = 0.5633
BASELINE_V5p34 = 0.820


def print_compare(results):
    print("=" * 100)
    print("V5.35d — Comparaison des 7 formes f(c, b) du 3-corps multiplicatif")
    print("=" * 100)
    print(
        f"{'Form':<12} {'δ*':>9} {'Σ|log|':>10} {'B_s/B':>9}  "
        f"{'log K⁰':>9}  {'log D⁰':>9}  {'log B⁰':>9}  {'log B_s⁰':>10}"
    )
    print("-" * 100)
    for form in FORMS:
        b = results[form]
        per = b["per_meson"]
        print(
            f"{form:<12} {b['delta']:>+9.3f} {b['sum_abs_log']:>10.4f} "
            f"{b['ratio_Bs_B']:>9.2f}  "
            f"{per['K0']['log_ratio']:>+9.3f}  "
            f"{per['D0']['log_ratio']:>+9.3f}  "
            f"{per['B0']['log_ratio']:>+9.3f}  "
            f"{per['Bs0']['log_ratio']:>+10.3f}"
        )
    print(f"\nForme retenue : {results['best_form']}")
    print()


def print_details(results, form: str):
    b = results[form]
    print("=" * 100)
    print(f"V5.35d — Détails forme {form}")
    print("=" * 100)
    print(
        f"δ* = {b['delta']:+.4f}  |  Σ|log| = {b['sum_abs_log']:.4f} dex  "
        f"|  ratio B_s/B = {b['ratio_Bs_B']:.2f} (PDG = 35.05)"
    )
    print()
    print(
        f"{'Méson':<6} {'Δm V5.35d':>12} {'Δm PDG':>12} {'log_ratio':>11} "
        f"{'R_χ':>8} {'fact_3b':>9} {'I_χ V5.35b':>13} {'I_χ V5.35d':>13}"
    )
    print("-" * 100)
    per = b["per_meson"]
    for name in ("K0", "D0", "B0", "Bs0"):
        p = per[name]
        print(
            f"{name:<6} {p['delta_m_MeV']:>12.4e} "
            f"{p['delta_m_pdg_MeV']:>12.4e} "
            f"{p['log_ratio']:>+11.4f} {p['R_chi']:>8.3f} "
            f"{p['factor_3body']:>9.3f} "
            f"{p['I_chi_v5p35b']:>13.3e} {p['I_chi']:>13.3e}"
        )
    print()


def print_history(results):
    print("=" * 100)
    print("V5.35d — Plafond Σ|log| par version")
    print("=" * 100)
    history = [
        ("V5.30c",        1.051, "préfacteur vibratoire scalaire"),
        ("V5.31",         1.460, "chiralité asymétrique"),
        ("V5.33",         1.060, "EM mutuelle (λ_max)"),
        ("V5.34 (F3)",    0.820, "inertie chirale F3_coupling"),
        ("V5.35b (rel)",  BASELINE_V5p35b, "asymétrie complexité × J_ij"),
    ]
    best = results["best_form"]
    history.append(
        (
            f"V5.35d ({best})",
            results[best]["sum_abs_log"],
            "3-corps multiplicatif renormalisant",
        )
    )
    for tag, s, desc in history:
        print(f"  {tag:<22} Σ|log| = {s:.4f} dex   ({desc})")
    print()


def verdict(results):
    print("=" * 100)
    print("Verdict V5.35d (honnête)")
    print("=" * 100)
    best = results["best_form"]
    s = results[best]["sum_abs_log"]
    rBsB = results[best]["ratio_Bs_B"]
    log_B = results[best]["per_meson"]["B0"]["log_ratio"]
    if s < 0.1 and abs(log_B) < 0.05:
        label = "BREAKTHROUGH"
        print(
            f"VERDICT : BREAKTHROUGH — V5.35d ({best}) ferme Σ|log| à\n"
            f"   {s:.4f} dex (B⁰ log_r = {log_B:+.4f}) avec un seul\n"
            f"   paramètre δ = {results[best]['delta']:+.4f}.\n"
            f"   La structure 3-corps multiplicative était le bon levier."
        )
    elif s < 0.4 and abs(log_B) < 0.2:
        label = "SUCCESS"
        print(
            f"VERDICT : SUCCESS — V5.35d ({best}) atteint Σ|log| = {s:.4f}\n"
            f"   dex (B⁰ log_r = {log_B:+.4f}).  Net amélioration sur\n"
            f"   V5.35b ({BASELINE_V5p35b:.4f})."
        )
    elif s < BASELINE_V5p35b - 0.05:
        label = "PARTIAL"
        print(
            f"VERDICT : PARTIAL — V5.35d ({best}) descend Σ|log| de\n"
            f"   {BASELINE_V5p35b:.4f} (V5.35b) à {s:.4f} dex mais\n"
            f"   B⁰ reste à {log_B:+.4f} dex.  Amélioration partielle."
        )
    elif s < BASELINE_V5p35b + 0.05:
        label = "NEUTRAL"
        print(
            f"VERDICT : NEUTRAL — V5.35d ({best}) donne Σ|log| = {s:.4f},\n"
            f"   équivalent à V5.35b ({BASELINE_V5p35b:.4f}).\n"
            f"   Le 3-corps multiplicatif n'apporte rien.\n"
            f"   Recommandation : reformuler V5.36."
        )
    else:
        label = "FAILURE"
        print(
            f"VERDICT : FAILURE — V5.35d ({best}) donne Σ|log| = {s:.4f},\n"
            f"   au-dessus de V5.35b ({BASELINE_V5p35b:.4f}).\n"
            f"   Recommandation : reformuler V5.36."
        )
    print(f"   ratio B_s/B = {rBsB:.2f} (PDG = 35.05).")
    print()
    return label


def make_plot(results, out_path: str = "output/v5p35d_3body_mult.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # Panneau 1 : Σ|log| par forme
    forms_lbl = list(FORMS) + ["V5.35b", "V5.34"]
    sums = [results[f]["sum_abs_log"] for f in FORMS] + [
        BASELINE_V5p35b, BASELINE_V5p34
    ]
    best = results["best_form"]
    colors = [
        "#1f77b4" if f != best else "#2ca02c" for f in FORMS
    ] + ["#7f7f7f", "#999999"]
    bars = ax1.bar(forms_lbl, sums, color=colors)
    ax1.axhline(0.4, color="g", ls=":", lw=1, alpha=0.6, label="seuil 0.4 dex")
    ax1.axhline(
        BASELINE_V5p35b, color="k", ls="--", lw=1, alpha=0.6,
        label=f"V5.35b ({BASELINE_V5p35b:.3f})"
    )
    ax1.set_ylabel(
        r"$\Sigma|\log_{10}(\Delta m / \Delta m_{\rm PDG})|$ (dex)"
    )
    ax1.set_title("V5.35d — Σ|log| par forme f vs V5.35b/V5.34")
    ax1.legend(fontsize=9)
    for bar, f in zip(bars[: len(FORMS)], FORMS):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"δ={results[f]['delta']:+.2f}",
            ha="center",
            fontsize=8,
            rotation=30,
        )
    ax1.grid(True, axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Panneau 2 : Δm par méson pour la meilleure forme
    per = results[best]["per_meson"]
    mesons = ["K0", "D0", "B0", "Bs0"]
    dm_v535d = [per[m]["delta_m_MeV"] for m in mesons]
    dm_pdg = [per[m]["delta_m_pdg_MeV"] for m in mesons]
    x = np.arange(len(mesons))
    w = 0.35
    ax2.bar(
        x - w / 2, dm_v535d, w,
        label=f"V5.35d ({best}, δ={results[best]['delta']:+.3f})",
        color="#2ca02c",
    )
    ax2.bar(x + w / 2, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesons)
    ax2.set_ylabel(r"$\Delta m$ (MeV)")
    ax2.set_title(f"V5.35d — Δm meilleure forme ({best})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print(
        "V5.35d — 3-corps multiplicatif renormalisant\n"
        "    I_χ V5.35d(L) = I_χ V5.35b(L) × Π_{i<j<k} [1 + δ × f(c, b)]\n"
        "    f(K⁰) ≡ 0 par construction (calibration K⁰ préservée)\n"
    )
    # Calibration grossière sur la grille demandée + raffinement local
    print("Calibration grossière (n_grid=6001) ...")
    results = calibrate_best_form(delta_range=(-10.0, 50.0), n_grid=6001)

    # Raffinement local autour du meilleur δ pour chaque forme
    print("Raffinement local des δ ...")
    from theory.v5p35d_3body_multiplicative import calibrate_delta
    for form in FORMS:
        d0 = results[form]["delta"]
        # Fenêtre ±10 % de l'intervalle initial autour de d0
        win = 1.0  # large window in absolute units
        refined = calibrate_delta(
            form, delta_range=(d0 - win, d0 + win), n_grid=4001
        )
        if refined["sum_abs_log"] < results[form]["sum_abs_log"]:
            results[form] = refined
    # Recompute best
    best = min(FORMS, key=lambda f: results[f]["sum_abs_log"])
    results["best_form"] = best

    print_compare(results)
    print_details(results, results["best_form"])
    print_history(results)
    label = verdict(results)
    make_plot(results)
    print(f"Label final : {label}")
    return label


if __name__ == "__main__":
    main()
