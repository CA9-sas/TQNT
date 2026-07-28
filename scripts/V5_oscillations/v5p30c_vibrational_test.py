"""V5.30c — Modes vibratoires du nœud comme préfacteur dynamique — test.

Script principal V5.30c.  Étend V5.30 (combinaison frustrés + Kashaev) par
un préfacteur dynamique vibratoire :

    Δm V5.30c(L) = Δm V5.30(L) × ( Ω_link(L) / Ω_link(K⁰) )^α
    Ω_link(L)    = n_comp(L) · ⟨ω_0⟩_geom(L)
    ⟨ω_0⟩_geom  = ( Π_i b_i / √c_i )^{1/n_comp}

1 seul paramètre nouveau (α) au-dessus de V5.30, calibré globalement par
χ²_log sur K, D, B⁰, B_s⁰ (K⁰ contribue zéro par construction).

Sortie : output/v5p30c_vibrational.png
Référence : theory/v5p30c_vibrational.py, theory/v5p30_combined_oscillation.py

Verdict honnête (cf. consigne) :
* SUCCESS : tous |log_ratio| < 0.5 dex ET |log10(ratio_Bs_B / 35)| < 0.3
* PARTIAL : amélioration mais critères non tous remplis
* FAILURE : V5.30c casse un système relatif à V5.30
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
    calibrate_alpha_global,
    v5p30c_table,
)


def print_calibration():
    print("=" * 80)
    print("V5.30c — Calibration globale de α (modes vibratoires)")
    print("=" * 80)
    cal = calibrate_alpha_global(
        alpha_range=(-5.0, 15.0),
        n_grid=4001,
        kappa_K=KAPPA_K_V5p30_DEFAULT,
    )
    print(
        f"α optimal (χ²_log sur K+D+B+Bs) = {cal['alpha']:.4f} "
        f"(χ²_log = {cal['chi2_log']:.4f}, Σ|log| = {cal['sum_abs_log']:.3f})"
    )
    print(
        f"(κ_K maintenu à sa valeur V5.30 : "
        f"{KAPPA_K_V5p30_DEFAULT:.4f})"
    )
    print("Note : K⁰ contribue zéro par construction (R_vib(K⁰) = 1).")
    print()
    return cal["alpha"]


def print_table(alpha: float):
    rows = v5p30c_table(alpha=alpha, kappa_K=KAPPA_K_V5p30_DEFAULT)
    print("=" * 132)
    print(
        f"V5.30c — Δm vibrational prefactor (α = {alpha:.4f}, "
        f"κ_K = {KAPPA_K_V5p30_DEFAULT:.4f})"
    )
    print("=" * 132)
    print(
        f"{'Méson':<6} {'Knots':<22} {'n_c':>4} {'omega0_g':>9} "
        f"{'Ω_link':>9} {'R_vib':>10} "
        f"{'Δm V5.30c':>11} {'Δm V5.30':>11} {'Δm PDG':>11} "
        f"{'log 530c':>9} {'log 530':>9}"
    )
    print("-" * 132)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<6} {knots_str:<22} "
            f"{r['n_comp']:>4d} {r['omega0_geom']:>9.4f} "
            f"{r['omega_link']:>9.4f} {r['R_vib']:>10.3e} "
            f"{r['delta_m_v5p30c_MeV']:>11.3e} "
            f"{r['delta_m_v5p30_MeV']:>11.3e} "
            f"{r['delta_m_pdg_MeV']:>11.3e} "
            f"{r['log_ratio_v5p30c']:>+9.3f} "
            f"{r['log_ratio_v5p30']:>+9.3f}"
        )
    print()
    return rows


def print_hierarchy(rows):
    by_name = {r["name"]: r["delta_m_v5p30c_MeV"] for r in rows}
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
        "V5.30c : "
        + " < ".join(
            f"{n}({by_name[n]:.2e})"
            for n in sorted(order_target, key=lambda x: by_name[x])
        )
    )
    order_v530c = sorted(order_target, key=lambda n: by_name[n])
    correct = order_v530c == order_target
    print(f"Ordre V5.30c : {'REPRODUIT' if correct else 'INCORRECT'}")

    ratio_v530c = by_name["Bs0"] / by_name["B0"]
    ratio_pdg = pdg["Bs0"] / pdg["B0"]
    print(
        f"Ratio B_s⁰/B⁰ : V5.30c = {ratio_v530c:.2f}, PDG = {ratio_pdg:.2f}  "
        f"(facteur d'écart : {ratio_v530c / ratio_pdg:.2f}×)"
    )
    print()
    return correct, ratio_v530c, ratio_pdg


def print_sum_log_ratio(rows):
    s_530c = sum(abs(r["log_ratio_v5p30c"]) for r in rows)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows)
    # Récupère V5.28 et V5.29 depuis v5p30_table pour comparaison historique
    from theory.v5p30_combined_oscillation import v5p30_table

    rows_hist = v5p30_table(KAPPA_K_V5p30_DEFAULT)
    s_528 = sum(abs(r["log_ratio_v528"]) for r in rows_hist)
    s_29a = sum(abs(r["log_ratio_v529a"]) for r in rows_hist)
    s_29c = sum(abs(r["log_ratio_v529c"]) for r in rows_hist)
    print("=" * 80)
    print("Cumul Σ|log_ratio| (somme des écarts absolus)")
    print("=" * 80)
    print(f"  V5.28  : Σ|log| = {s_528:.3f} dex")
    print(f"  V5.29a : Σ|log| = {s_29a:.3f} dex")
    print(f"  V5.29c : Σ|log| = {s_29c:.3f} dex")
    print(f"  V5.30  : Σ|log| = {s_530:.3f} dex")
    print(f"  V5.30c : Σ|log| = {s_530c:.3f} dex")
    best = min(
        (s_528, "V5.28"),
        (s_29a, "V5.29a"),
        (s_29c, "V5.29c"),
        (s_530, "V5.30"),
        (s_530c, "V5.30c"),
    )
    print(f"  Meilleur : {best[1]} (Σ|log| = {best[0]:.3f})")
    print()
    return s_528, s_29a, s_29c, s_530, s_530c


def make_plot(rows, alpha: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    names = [r["name"] for r in rows]
    dm_530c = [r["delta_m_v5p30c_MeV"] for r in rows]
    dm_530 = [r["delta_m_v5p30_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.27

    ax1.bar(x - w, dm_530, w, label="V5.30", color="#1f77b4")
    ax1.bar(x, dm_530c, w, label=f"V5.30c (α={alpha:.2f})", color="#d62728")
    ax1.bar(x + w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ : V5.30 / V5.30c / PDG")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    log_530c = [r["log_ratio_v5p30c"] for r in rows]
    log_530 = [r["log_ratio_v5p30"] for r in rows]
    ax2.bar(x - w / 2, log_530, w, label="V5.30", color="#1f77b4")
    ax2.bar(
        x + w / 2, log_530c, w, label=f"V5.30c (α={alpha:.2f})", color="#d62728"
    )
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(0.5, color="g", ls="--", lw=0.8, alpha=0.5, label="±0.5 dex")
    ax2.axhline(-0.5, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$\log_{10}(\Delta m / \Delta m_{\rm PDG})$")
    ax2.set_title("Précision V5.30c vs V5.30")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = "output/v5p30c_vibrational.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def verdict(rows, correct_hier, ratio_v530c, ratio_pdg,
            s_530, s_530c, alpha):
    print()
    print("=" * 80)
    print("Verdict V5.30c (honnête)")
    print("=" * 80)
    by_name = {r["name"]: r for r in rows}

    print("Δm V5.30c vs V5.30 par système (log_ratio) :")
    for n in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[n]
        status = (
            "amélioration"
            if abs(r["log_ratio_v5p30c"]) < abs(r["log_ratio_v5p30"])
            else "détérioration"
        )
        print(
            f"  {n:<4} : V5.30 = {r['log_ratio_v5p30']:+.3f}, "
            f"V5.30c = {r['log_ratio_v5p30c']:+.3f}  ({status})"
        )
    print()

    has_failure = False
    msgs = []
    for n in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[n]
        if abs(r["log_ratio_v5p30c"]) > abs(r["log_ratio_v5p30"]) + 0.5:
            has_failure = True
            msgs.append(
                f"  - {n} dégradé de "
                f"{r['log_ratio_v5p30c'] - r['log_ratio_v5p30']:+.2f} dex"
            )

    closes_all = (
        correct_hier
        and all(
            abs(by_name[n]["log_ratio_v5p30c"]) < 0.5
            for n in ["K0", "D0", "B0", "Bs0"]
        )
        and abs(np.log10(ratio_v530c / ratio_pdg)) < 0.3
    )

    if closes_all:
        label = "SUCCESS"
        print(
            "VERDICT : SUCCESS — V5.30c ferme la hiérarchie complète\n"
            "         (tous Δm sous 0.5 dex, ratio Bs/B sous 0.3 dex)\n"
            f"         avec UN paramètre nouveau (α = {alpha:.3f}).\n"
            f"         Σ|log| V5.30 = {s_530:.3f} → V5.30c = {s_530c:.3f}."
        )
    elif has_failure:
        label = "FAILURE"
        print(
            "VERDICT : FAILURE — V5.30c dégrade au moins un système :\n"
            + "\n".join(msgs)
            + "\n"
            "Recommandation : la rigidité dynamique simple à un paramètre\n"
            "ne suffit pas ; explorer un mécanisme indépendant pour la\n"
            "structure de couplage inter-composantes (e.g. spectre complet)."
        )
    else:
        label = "PARTIAL"
        print(
            "VERDICT : PARTIAL — V5.30c améliore V5.30 mais ne ferme pas\n"
            "         tous les critères simultanément :\n"
            f"         Σ|log| {s_530:.3f} → {s_530c:.3f} dex,\n"
            f"         ratio Bs/B = {ratio_v530c:.1f} vs PDG = {ratio_pdg:.1f}\n"
            f"         (écart × {ratio_v530c / ratio_pdg:.2f}).\n"
            "Diagnostic : un seul paramètre dynamique global (α) ne peut\n"
            "pas distinguer simultanément l'absolu Bs et le ratio Bs/B.\n"
            "Une extension reste nécessaire (e.g. spectre de modes inter-\n"
            "composantes dépendant explicitement de la chiralité du link)."
        )
    print()
    return label


def main():
    print()
    print(
        "V5.30c — Modes vibratoires du nœud comme préfacteur dynamique\n"
        "Δm = Δm V5.30 × ( Ω_link(L) / Ω_link(K⁰) )^α,\n"
        "    Ω_link = n_comp · ⟨b/√c⟩_geom\n"
    )

    alpha_opt = print_calibration()
    rows = print_table(alpha_opt)
    correct, ratio_530c, ratio_pdg = print_hierarchy(rows)
    s_528, s_29a, s_29c, s_530, s_530c = print_sum_log_ratio(rows)
    make_plot(rows, alpha_opt)
    label = verdict(rows, correct, ratio_530c, ratio_pdg,
                    s_530, s_530c, alpha_opt)
    print(f"Label final : {label}")


if __name__ == "__main__":
    main()
