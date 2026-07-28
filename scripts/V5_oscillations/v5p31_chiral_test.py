"""V5.31 — Spectre vibratoire chiral asymétrique — test runner.

Étend V5.30c (préfacteur vibratoire scalaire) par un facteur chiral
par composante distinguant amphichiraux et chiraux :

    Δm V5.31(L) = Δm V5.30c(L) × (P_chiral(L) / P_chiral(K⁰))
    P_chiral(L) = Π_i exp(β · σ(K_i))
    σ(amphichiral) = -1, σ(chiral) = |jones_power_center(V_K)|

1 seul paramètre nouveau (β) au-dessus de V5.30c, calibré sur le ratio
B_s/B = 35 (PDG).  K⁰ contribue zéro par construction.

Sortie : output/v5p31_chiral.png
Référence : theory/v5p31_chiral_vibrational.py,
            theory/v5p30c_vibrational.py

Verdict honnête (cf. consigne) :
* SUCCESS : tous |log_ratio| < 0.5 dex ET ratio Bs/B ∈ [30, 40]
            ET Σ|log| < V5.30c
* PARTIAL : ferme le ratio mais dégrade un absolu, OU améliore Σ|log|
            sans fermer le ratio
* FAILURE : V5.31 dégrade tout (Σ|log| > V5.30c ET ratio loin de 35)
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
from theory.v5p30_combined_oscillation import KAPPA_K_V5p30_DEFAULT
from theory.v5p30c_vibrational import (
    ALPHA_VIB_DEFAULT,
    v5p30c_table,
)
from theory.v5p31_chiral_vibrational import (
    BETA_CHIRAL_DEFAULT,
    calibrate_beta_ratio,
    chirality_signature,
    v5p31_table,
)


def print_signatures():
    print("=" * 80)
    print("V5.31 — Signatures de chiralité σ(K) par composante")
    print("=" * 80)
    seen = set()
    print(f"{'Knot':<6} {'is_amphi':<10} {'power_center':>14} {'sigma':>10}")
    print("-" * 50)
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        for k in cfg["knots"]:
            if k in seen:
                continue
            seen.add(k)
            sig = chirality_signature(k)
            print(
                f"{k:<6} {str(sig['is_amphi']):<10} "
                f"{sig['power_center']:>+14.4f} {sig['sigma']:>+10.4f}"
            )
    print()


def print_calibration():
    print("=" * 80)
    print("V5.31 — Calibration β sur ratio B_s/B = 35 (PDG)")
    print("=" * 80)
    cal = calibrate_beta_ratio(
        target_ratio=35.0,
        beta_range=(-3.0, 3.0),
        n_grid=6001,
    )
    print(
        f"β* = {cal['beta']:.4f}  →  ratio_V5.31 = {cal['ratio_v531']:.3f}  "
        f"(target = {cal['target_ratio']:.1f}, "
        f"log_diff = {cal['log_diff_dex']:+.4f} dex)"
    )
    print(
        f"(α maintenu à V5.30c : {ALPHA_VIB_DEFAULT:.4f} ; "
        f"κ_K maintenu à V5.30 : {KAPPA_K_V5p30_DEFAULT:.4f})"
    )
    print()
    return cal["beta"]


def print_table(beta: float):
    rows = v5p31_table(beta=beta)
    print("=" * 132)
    print(
        f"V5.31 — Δm chiral vibrational (β = {beta:.4f})"
    )
    print("=" * 132)
    print(
        f"{'Méson':<6} {'Knots':<22} {'n_a':>4} {'σ_sum':>8} "
        f"{'R_chiral':>11} "
        f"{'Δm V5.31':>11} {'Δm V5.30c':>11} {'Δm PDG':>11} "
        f"{'log V5.31':>10} {'log V5.30c':>11}"
    )
    print("-" * 132)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<6} {knots_str:<22} "
            f"{r['n_amphi']:>4d} {r['sigma_sum']:>+8.3f} "
            f"{r['R_chiral']:>11.3e} "
            f"{r['delta_m_v5p31_MeV']:>11.3e} "
            f"{r['delta_m_v5p30c_MeV']:>11.3e} "
            f"{r['delta_m_pdg_MeV']:>11.3e} "
            f"{r['log_ratio_v5p31']:>+10.3f} "
            f"{r['log_ratio_v5p30c']:>+11.3f}"
        )
    print()
    return rows


def print_hierarchy(rows):
    by_name = {r["name"]: r["delta_m_v5p31_MeV"] for r in rows}
    pdg = {r["name"]: r["delta_m_pdg_MeV"] for r in rows}

    print("=" * 80)
    print("Hiérarchie K < D < B⁰ < B_s⁰")
    print("=" * 80)
    order_target = ["K0", "D0", "B0", "Bs0"]
    print(
        "PDG    : "
        + " < ".join(
            f"{n}({pdg[n]:.2e})"
            for n in sorted(order_target, key=lambda x: pdg[x])
        )
    )
    print(
        "V5.31  : "
        + " < ".join(
            f"{n}({by_name[n]:.2e})"
            for n in sorted(order_target, key=lambda x: by_name[x])
        )
    )
    order_v531 = sorted(order_target, key=lambda n: by_name[n])
    correct = order_v531 == order_target
    print(f"Ordre V5.31 : {'REPRODUIT' if correct else 'INCORRECT'}")

    ratio_v531 = by_name["Bs0"] / by_name["B0"]
    ratio_pdg = pdg["Bs0"] / pdg["B0"]
    print(
        f"Ratio B_s⁰/B⁰ : V5.31 = {ratio_v531:.2f}, "
        f"PDG = {ratio_pdg:.2f}  "
        f"(facteur d'écart : {ratio_v531 / ratio_pdg:.2f}×)"
    )
    print()
    return correct, ratio_v531, ratio_pdg


def print_sum_log_ratio(rows):
    s_531 = sum(abs(r["log_ratio_v5p31"]) for r in rows)
    s_530c = sum(abs(r["log_ratio_v5p30c"]) for r in rows)
    print("=" * 80)
    print("Cumul Σ|log_ratio| (somme des écarts absolus)")
    print("=" * 80)
    print(f"  V5.30c : Σ|log| = {s_530c:.3f} dex")
    print(f"  V5.31  : Σ|log| = {s_531:.3f} dex")
    delta = s_531 - s_530c
    print(
        f"  Δ      : {delta:+.3f} dex "
        f"({'amélioration' if delta < 0 else 'détérioration'})"
    )
    print()
    return s_530c, s_531


def make_plot(rows, beta: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    names = [r["name"] for r in rows]
    dm_531 = [r["delta_m_v5p31_MeV"] for r in rows]
    dm_530c = [r["delta_m_v5p30c_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.27

    ax1.bar(x - w, dm_530c, w, label="V5.30c", color="#1f77b4")
    ax1.bar(x, dm_531, w, label=f"V5.31 (β={beta:.3f})", color="#d62728")
    ax1.bar(x + w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ : V5.30c / V5.31 / PDG")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    log_531 = [r["log_ratio_v5p31"] for r in rows]
    log_530c = [r["log_ratio_v5p30c"] for r in rows]
    ax2.bar(x - w / 2, log_530c, w, label="V5.30c", color="#1f77b4")
    ax2.bar(
        x + w / 2,
        log_531,
        w,
        label=f"V5.31 (β={beta:.3f})",
        color="#d62728",
    )
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(0.5, color="g", ls="--", lw=0.8, alpha=0.5, label="±0.5 dex")
    ax2.axhline(-0.5, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$\log_{10}(\Delta m / \Delta m_{\rm PDG})$")
    ax2.set_title("Précision V5.31 vs V5.30c")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = "output/v5p31_chiral.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def verdict(
    rows,
    correct_hier,
    ratio_v531,
    ratio_pdg,
    s_530c,
    s_531,
    beta,
):
    print()
    print("=" * 80)
    print("Verdict V5.31 (honnête)")
    print("=" * 80)
    by_name = {r["name"]: r for r in rows}

    print("Δm V5.31 vs V5.30c par système (log_ratio) :")
    for n in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[n]
        status = (
            "amélioration"
            if abs(r["log_ratio_v5p31"]) < abs(r["log_ratio_v5p30c"])
            else "détérioration"
        )
        print(
            f"  {n:<4} : V5.30c = {r['log_ratio_v5p30c']:+.3f}, "
            f"V5.31 = {r['log_ratio_v5p31']:+.3f}  ({status})"
        )
    print()

    closes_absolutes = all(
        abs(by_name[n]["log_ratio_v5p31"]) < 0.5
        for n in ["K0", "D0", "B0", "Bs0"]
    )
    closes_ratio = 30.0 <= ratio_v531 <= 40.0
    improves_sum = s_531 < s_530c

    if closes_absolutes and closes_ratio and improves_sum:
        label = "SUCCESS"
        print(
            "VERDICT : SUCCESS — V5.31 ferme à la fois les absolus\n"
            "         (tous |log| < 0.5) et le ratio Bs/B ∈ [30,40],\n"
            f"         avec UN paramètre nouveau (β = {beta:.4f}).\n"
            f"         Σ|log| V5.30c = {s_530c:.3f} → V5.31 = {s_531:.3f}."
        )
    elif closes_ratio and not closes_absolutes:
        label = "PARTIAL"
        print(
            "VERDICT : PARTIAL — V5.31 ferme le ratio Bs/B "
            f"({ratio_v531:.1f} vs PDG = {ratio_pdg:.1f})\n"
            "         mais dégrade au moins un absolu :\n"
        )
        for n in ["K0", "D0", "B0", "Bs0"]:
            r = by_name[n]
            if abs(r["log_ratio_v5p31"]) >= 0.5:
                print(
                    f"           {n} : |log| = {abs(r['log_ratio_v5p31']):.3f}"
                    " ≥ 0.5"
                )
        print(
            f"         Σ|log| V5.30c = {s_530c:.3f} → V5.31 = {s_531:.3f}.\n"
            "Diagnostic : la formule chirale "
            "Π exp(β σ(K_i)) ne distingue pas\n"
            "la sur-amplification spécifique de B_s sans dégrader D⁰.\n"
            "V5 a une LIMITE structurelle ici : un seul paramètre β\n"
            "ne peut pas reconcilier ratio + absolus simultanément."
        )
    elif improves_sum and not closes_ratio:
        label = "PARTIAL"
        print(
            "VERDICT : PARTIAL — V5.31 améliore Σ|log| sans fermer le ratio."
        )
    else:
        label = "FAILURE"
        print(
            "VERDICT : FAILURE — V5.31 ne ferme ni les absolus ni le ratio\n"
            f"         et Σ|log| {s_530c:.3f} → {s_531:.3f}.\n"
            "Le mécanisme chiral asymétrique testé ici n'est pas suffisant ;\n"
            "V5 a une limite structurelle pour le système (B, B_s)."
        )
    print()
    return label


def main():
    print()
    print(
        "V5.31 — Spectre vibratoire chiral asymétrique\n"
        "Δm = Δm V5.30c × (P_chiral(L) / P_chiral(K⁰)),\n"
        "    P_chiral = Π_i exp(β · σ(K_i)),\n"
        "    σ(amphi) = -1, σ(chiral) = |jones_power_center(V_K)|\n"
    )

    print_signatures()
    beta_opt = print_calibration()
    rows = print_table(beta_opt)
    correct, ratio_531, ratio_pdg = print_hierarchy(rows)
    s_530c, s_531 = print_sum_log_ratio(rows)
    make_plot(rows, beta_opt)
    label = verdict(
        rows, correct, ratio_531, ratio_pdg, s_530c, s_531, beta_opt
    )
    print(f"Label final : {label}")


if __name__ == "__main__":
    main()
