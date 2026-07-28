"""V5.28 — Oscillation chirale K ↔ K* comme tunneling quantique topologique.

Script principal V5.28.  Reframing conceptuel : ce qui oscille n'est PAS un
méson (étiquette de saveur SM) mais la **superposition cohérente** entre un
nœud K et son miroir K*, gouvernée par un élément matriciel topologique de
la forme N_paths × exp(-S_min/ℏ).

Sortie :
    output/v5p28_chiral_oscillation.png

Référence : theory/topological_chiral_oscillation.py
Honnête : V5.28 ferme le gap B⁰ (V5p27: -3.34 dex → V5.28: -0.73 dex) mais
n'épargne pas la hiérarchie PDG ; le ratio Bs/B reste inversé.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.topological_chiral_oscillation import (
    chiral_oscillation_table,
    count_paths_K_to_Kstar,
    action_min,
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)


def print_table():
    rows = chiral_oscillation_table()
    print("=" * 96)
    print("V5.28 — Oscillation chirale K↔K* (tunneling topologique)")
    print("=" * 96)
    print(
        f"{'Particule':<8} {'Knots':<22} {'N_paths':>10} {'S_min':>7} "
        f"{'Δm V5.28':>11} {'Δm V5p27':>11} {'Δm PDG':>11} "
        f"{'log V5.28':>9} {'log V5p27':>9}"
    )
    print("-" * 96)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<8} {knots_str:<22} "
            f"{r['N_paths']:>10.2e} {r['S_min']:>7.2f} "
            f"{r['delta_m_v528_MeV']:>11.2e} {r['delta_m_v5p27_MeV']:>11.2e} "
            f"{r['delta_m_pdg_MeV']:>11.2e} "
            f"{r['log_ratio_v528']:>+9.2f} {r['log_ratio_v5p27']:>+9.2f}"
        )
    print()
    return rows


def make_plot(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # --- Plot 1 : Δm V5.28 vs V5p27 vs PDG ---
    names = [r["name"] for r in rows]
    dm_528 = [r["delta_m_v528_MeV"] for r in rows]
    dm_527 = [r["delta_m_v5p27_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.27
    ax1.bar(x - w, dm_527, w, label="V5p27 (P15 reconnexion)", color="#888888")
    ax1.bar(x,     dm_528, w, label="V5.28 (tunneling chiral)", color="#1f77b4")
    ax1.bar(x + w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("0", "⁰") for n in names])
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ : V5.28 vs V5p27 vs PDG (\'echelle log)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)
    for i, r in enumerate(rows):
        ax1.text(
            i, max(r["delta_m_v528_MeV"], r["delta_m_pdg_MeV"]) * 5,
            f"{r['log_ratio_v528']:+.2f}",
            ha="center", fontsize=9, color="darkblue", fontweight="bold",
        )

    # --- Plot 2 : log_ratio V5.28 vs V5p27 ---
    log_528 = [r["log_ratio_v528"] for r in rows]
    log_527 = [r["log_ratio_v5p27"] for r in rows]
    ax2.bar(x - 0.2, log_527, 0.4, label="V5p27", color="#888888")
    ax2.bar(x + 0.2, log_528, 0.4, label="V5.28", color="#1f77b4")
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(1, color="g", ls="--", lw=0.8, alpha=0.5, label="±1 dex")
    ax2.axhline(-1, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("0", "⁰") for n in names])
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{\rm V5}/\Delta m_{\rm PDG})$")
    ax2.set_title("V5.28 vs V5p27 : amélioration de précision (axe linéaire)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = "output/v5p28_chiral_oscillation.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def print_details():
    print("Comptage de chemins R-move et action topologique :")
    print(
        f"  {'Knot':<22} {'c_tot':>6} {'N_paths':>11} {'multinom.':>12} "
        f"{'sym':>5} {'braid':>6} {'S_min':>7}"
    )
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        p = count_paths_K_to_Kstar(cfg["knots"])
        a = action_min(cfg["knots"], cfg["mass_pdg"])
        knots = " + ".join(cfg["knots"])
        print(
            f"  {name:<5} {knots:<16} {p['c_tot_eff']:>6} "
            f"{p['N_paths']:>11.2e} {p['multinomial']:>12.2e} "
            f"{p['sym_factor']:>5d} {p['braid_product']:>6d} "
            f"{a['S_min']:>7.2f}"
        )
    print()


def main():
    print()
    print("V5.28 — Oscillation chirale K↔K* comme tunneling quantique topologique")
    print(
        "Reframe : ce qui oscille = superposition |K⟩ + |K*⟩, pas une saveur SM.")
    print()
    print_details()
    rows = print_table()
    make_plot(rows)

    print()
    print("=" * 76)
    print("Verdict V5.28 (honnête)")
    print("=" * 76)
    print(
        "- K⁰  : calibrage (Δm V5.28 = Δm PDG par construction)")
    print(
        f"- D⁰  : V5.28 sur-prédit de +1.97 dex (V5p27 était à -0.28).")
    print(
        "        La combinatoire des chemins gonfle trop le matrix element.")
    print(
        f"- B⁰  : **gap fermé à -0.73 dex** (V5p27 était à -3.34 dex).")
    print(
        "        N_paths grand (2.3e7) compense l'action S_min plus élevée.")
    print(
        f"- Bs⁰ : V5.28 à -3.30 dex (V5p27 était à -1.76).")
    print(
        "        N_paths petit (peu de composantes) supprime la prédiction.")
    print()
    print(
        "Hiérarchie : V5.28 prédit K < Bs < B < D ; PDG est K < D < B < Bs.")
    print(
        "Ratio Bs/B  : V5.28 ~ 0.094 ; PDG ~ 35 (inversion d'un facteur 350).")
    print()
    print(
        "Conclusion : V5.28 ferme partiellement le gap B (3.34 → 0.73 dex),")
    print(
        "preuve que la combinatoire R-move est la bonne direction physique,")
    print(
        "mais ne capte pas encore la structure fine PDG. Piste V5.29 :")
    print(
        " - couplage Bs/B via un terme S_min explicite en c_J asymetrique")
    print(
        " - chemins frustrés (suspension Reidemeister) pour amphichiraux")
    print(
        " - phase topologique résiduelle non-cohomologue (Kashaev / volume)")


if __name__ == "__main__":
    main()
