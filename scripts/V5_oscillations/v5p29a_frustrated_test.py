"""V5.29a — Chemins frustrés amphichirales : test sur K, D, B⁰, Bs⁰.

Compare V5.29a (N_paths × Π F_i) à V5.28 (N_paths nu) et au PDG.

Sortie : output/v5p29a_frustrated.png
Référence : theory/v5p29_frustrated_amphichiral.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p29_frustrated_amphichiral import (
    v5p29a_table,
    frustrated_factor,
    n_paths_with_frustration,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)


def print_frustration_factors():
    """Affiche F_i par composante pour chaque méson."""
    print("=" * 88)
    print("V5.29a — Facteurs de frustration F_i par composante amphichirale")
    print("=" * 88)
    print(
        f"{'Particule':<8} {'Composante':<10} {'amphi':<6} {'c':>3} "
        f"{'b':>3} {'F_i':>10} {'F_total':>10}"
    )
    print("-" * 88)
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        n_p = n_paths_with_frustration(cfg["knots"])
        first = True
        for f in n_p["F_per_component"]:
            print(
                f"{name if first else '':<8} {f['knot']:<10} "
                f"{'oui' if f['amphichiral'] else 'non':<6} "
                f"{f['c']:>3} {f['b']:>3} {f['F']:>10d} "
                f"{n_p['F_total']:>10d}" if first else
                f"{'':<8} {f['knot']:<10} "
                f"{'oui' if f['amphichiral'] else 'non':<6} "
                f"{f['c']:>3} {f['b']:>3} {f['F']:>10d} {'':>10}"
            )
            first = False
    print()


def print_table():
    """Affiche la table Δm V5.29a vs V5.28 vs PDG."""
    rows = v5p29a_table()
    print("=" * 110)
    print("V5.29a — Δm avec chemins frustrés amphichirales")
    print("=" * 110)
    print(
        f"{'Particule':<8} {'Knots':<22} {'n_amphi':>7} {'F_tot':>10} "
        f"{'Δm V5.29a':>11} {'Δm V5.28':>11} {'Δm PDG':>11} "
        f"{'log V5.29a':>10} {'log V5.28':>9}"
    )
    print("-" * 110)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<8} {knots_str:<22} "
            f"{r['n_amphichiral']:>7d} {r['F_total']:>10d} "
            f"{r['delta_m_v5p29a_MeV']:>11.2e} "
            f"{r['delta_m_v528_MeV']:>11.2e} "
            f"{r['delta_m_pdg_MeV']:>11.2e} "
            f"{r['log_ratio_v5p29a']:>+10.2f} "
            f"{r['log_ratio_v528']:>+9.2f}"
        )
    print()
    return rows


def make_plot(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    names = [r["name"] for r in rows]
    dm_29a = [r["delta_m_v5p29a_MeV"] for r in rows]
    dm_528 = [r["delta_m_v528_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.27

    # --- Plot 1 : Δm en log scale ---
    ax1.bar(x - w, dm_528, w, label="V5.28 (tunneling chiral)",
            color="#888888")
    ax1.bar(x,     dm_29a, w, label="V5.29a (frustrés)", color="#1f77b4")
    ax1.bar(x + w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("0", "⁰") for n in names])
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ V5.29a vs V5.28 vs PDG")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)
    for i, r in enumerate(rows):
        ax1.text(
            i, max(r["delta_m_v5p29a_MeV"], r["delta_m_pdg_MeV"]) * 5,
            f"{r['log_ratio_v5p29a']:+.2f}",
            ha="center", fontsize=9, color="darkblue", fontweight="bold",
        )

    # --- Plot 2 : log_ratio comparatif ---
    log_29a = [r["log_ratio_v5p29a"] for r in rows]
    log_528 = [r["log_ratio_v528"] for r in rows]
    ax2.bar(x - 0.2, log_528, 0.4, label="V5.28", color="#888888")
    ax2.bar(x + 0.2, log_29a, 0.4, label="V5.29a", color="#1f77b4")
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(1, color="g", ls="--", lw=0.8, alpha=0.5, label="±1 dex")
    ax2.axhline(-1, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("0", "⁰") for n in names])
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{\rm V5}/\Delta m_{\rm PDG})$")
    ax2.set_title("V5.29a vs V5.28 : amélioration (axe linéaire)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = "output/v5p29a_frustrated.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def print_verdict(rows):
    """Verdict honnête sur V5.29a."""
    print()
    print("=" * 76)
    print("Verdict V5.29a (honnête)")
    print("=" * 76)
    by_name = {r["name"]: r for r in rows}
    for name in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[name]
        delta = r["log_ratio_v5p29a"] - r["log_ratio_v528"]
        if abs(delta) < 1e-9:
            status = "inchangé"
        elif delta > 0:
            status = f"boost +{delta:.2f} dex"
        else:
            status = f"dégradé {delta:.2f} dex"
        print(
            f"- {name:<5}: log V5.28 = {r['log_ratio_v528']:+.2f}, "
            f"V5.29a = {r['log_ratio_v5p29a']:+.2f}  ({status})"
        )
    print()
    # Ratio Bs/B
    bs = by_name["Bs0"]["delta_m_v5p29a_MeV"]
    b0 = by_name["B0"]["delta_m_v5p29a_MeV"]
    ratio_29a = bs / b0
    bs_528 = by_name["Bs0"]["delta_m_v528_MeV"]
    b_528 = by_name["B0"]["delta_m_v528_MeV"]
    ratio_528 = bs_528 / b_528
    ratio_pdg = DELTA_M_PDG["Bs0"] / DELTA_M_PDG["B0"]
    print(
        f"Ratio Bs/B  : V5.28 = {ratio_528:.3f}, "
        f"V5.29a = {ratio_29a:.2f}, PDG = {ratio_pdg:.2f}"
    )
    print(
        f"  Amélioration ratio Bs/B : "
        f"log10 V5.29a/PDG = {np.log10(ratio_29a/ratio_pdg):+.2f} "
        f"(V5.28 était {np.log10(ratio_528/ratio_pdg):+.2f})"
    )
    print()
    # Cumul des |log_ratio|
    sum_abs_528 = sum(abs(r["log_ratio_v528"]) for r in rows)
    sum_abs_29a = sum(abs(r["log_ratio_v5p29a"]) for r in rows)
    print(
        f"Cumul |log_ratio| : V5.28 = {sum_abs_528:.2f}, "
        f"V5.29a = {sum_abs_29a:.2f}"
    )
    if sum_abs_29a < sum_abs_528:
        print(
            f"  → V5.29a améliore la précision globale "
            f"de {sum_abs_528 - sum_abs_29a:.2f} dex (somme)."
        )
    else:
        print(
            f"  → V5.29a dégrade la précision globale "
            f"de {sum_abs_29a - sum_abs_528:.2f} dex (somme)."
        )


def main():
    print()
    print(
        "V5.29a — Chemins frustrés amphichirales (boost Bs⁰)"
    )
    print(
        "F_i = c_i! × b_i pour composantes amphichirales (4_1, 6_3, 8_3)"
    )
    print()
    print_frustration_factors()
    rows = print_table()
    make_plot(rows)
    print_verdict(rows)


if __name__ == "__main__":
    main()
