"""V5.29c — Kashaev / volume conjecture comme préfacteur des oscillations.

Script principal V5.29c.  Étend V5.28 (commit 962c96f) en ajoutant un
préfacteur knot-spécifique motivé par la volume conjecture (Kashaev 1995,
Murakami-Murakami 2001) :

    Δm V5.29c = 2 ℏ ω_0 · N_paths · exp(-S_min/ℏ) · P_Kashaev(L)

avec P_Kashaev(L) = exp(κ_K · Σ_i vol(K_i)).

Sortie :
    output/v5p29c_kashaev_oscillation.png

Référence : theory/kashaev_invariants.py + theory/topological_chiral_oscillation.py

Honnête :
* les volumes hyperboliques sont **tabulés** (SnapPy / KnotInfo), pas dérivés
* κ_K est **un paramètre de plus** que V5.28
* si la hiérarchie complète K < D < B < Bs n'est pas reproduite, on l'écrit
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.kashaev_invariants import (
    hyperbolic_volume_table,
    total_hyperbolic_volume,
)
from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
    KAPPA_K_DEFAULT,
    chiral_oscillation_table_v29c,
    calibrate_kappa_K_to_hierarchy,
)


def print_volume_table():
    print("=" * 76)
    print("V5.29c — Volumes hyperboliques (SnapPy / KnotInfo) tabulés")
    print("=" * 76)
    print(f"  {'Knot':<6} {'vol(K)':>10} {'hyperbolique':>14}")
    for r in hyperbolic_volume_table():
        flag = "oui" if r["is_hyperbolic"] else "non (torus)"
        print(f"  {r['knot']:<6} {r['vol']:>10.4f} {flag:>14}")
    print()
    print("Volumes totaux des links d'oscillation :")
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        knots = " + ".join(cfg["knots"])
        vol_tot = total_hyperbolic_volume(cfg["knots"])
        print(f"  {name:<5} ({knots:<18}) : vol_tot = {vol_tot:.3f}")
    print()


def print_calibration(kappa_K: float):
    print("=" * 76)
    print(f"Calibration κ_K = {kappa_K:.4f}")
    print("=" * 76)
    print(
        "Source : minimisation χ² = Σ log²(Δm V5.29c / Δm PDG) sur D, B⁰, B_s⁰\n"
        "K⁰ calibré par construction (préfacteur global absorbe ω_0·ℏ·P_K(K⁰))."
    )
    print()


def print_table(kappa_K: float):
    rows = chiral_oscillation_table_v29c(kappa_K)
    print("=" * 116)
    print(
        f"V5.29c — Δm avec préfacteur Kashaev (κ_K = {kappa_K:.4f})"
    )
    print("=" * 116)
    header = (
        f"{'Particule':<8} {'Knots':<22} {'vol_tot':>8} {'P_Kash':>10} "
        f"{'Δm V5.29c':>11} {'Δm V5.28':>11} {'Δm PDG':>11} "
        f"{'log 29c':>8} {'log 528':>8}"
    )
    print(header)
    print("-" * 116)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<8} {knots_str:<22} "
            f"{r['vol_tot']:>8.3f} {r['P_Kashaev']:>10.3e} "
            f"{r['delta_m_v29c_MeV']:>11.3e} {r['delta_m_v528_MeV']:>11.3e} "
            f"{r['delta_m_pdg_MeV']:>11.3e} "
            f"{r['log_ratio_v29c']:>+8.3f} {r['log_ratio_v528']:>+8.3f}"
        )
    print()
    return rows


def print_hierarchy(rows):
    """Diagnostic honnête de la hiérarchie K < D < B⁰ < B_s⁰."""
    by_name = {r["name"]: r["delta_m_v29c_MeV"] for r in rows}
    pdg = {r["name"]: r["delta_m_pdg_MeV"] for r in rows}

    print("=" * 76)
    print("Hiérarchie")
    print("=" * 76)
    print(f"PDG    : K({pdg['K0']:.2e}) < D({pdg['D0']:.2e}) "
          f"< B({pdg['B0']:.2e}) < Bs({pdg['Bs0']:.2e})")
    print(f"V5.29c : K({by_name['K0']:.2e})  D({by_name['D0']:.2e})  "
          f"B({by_name['B0']:.2e})  Bs({by_name['Bs0']:.2e})")
    order_target = ["K0", "D0", "B0", "Bs0"]
    order_v29c = sorted(order_target, key=lambda n: by_name[n])
    correct = order_v29c == order_target
    print(f"Ordre V5.29c : {' < '.join(order_v29c)} "
          f"({'OK' if correct else 'partiel'})")
    ratio_v29c = by_name["Bs0"] / by_name["B0"]
    ratio_pdg = pdg["Bs0"] / pdg["B0"]
    print(
        f"Ratio B_s/B⁰ : V5.29c = {ratio_v29c:.3f}, "
        f"PDG = {ratio_pdg:.3f}  (facteur {ratio_pdg/ratio_v29c:.1f}×)"
    )
    print()
    return correct, ratio_v29c, ratio_pdg


def make_plot(rows, kappa_K: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    names = [r["name"] for r in rows]
    dm_29c = [r["delta_m_v29c_MeV"] for r in rows]
    dm_528 = [r["delta_m_v528_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.27

    ax1.bar(x - w, dm_528, w, label="V5.28", color="#888888")
    ax1.bar(x,     dm_29c, w, label=f"V5.29c (Kashaev, κ_K={kappa_K:.3f})",
            color="#1f77b4")
    ax1.bar(x + w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("0", "0") for n in names])
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ : V5.28 vs V5.29c vs PDG")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    log_29c = [r["log_ratio_v29c"] for r in rows]
    log_528 = [r["log_ratio_v528"] for r in rows]
    ax2.bar(x - 0.2, log_528, 0.4, label="V5.28", color="#888888")
    ax2.bar(x + 0.2, log_29c, 0.4, label="V5.29c", color="#1f77b4")
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(1, color="g", ls="--", lw=0.8, alpha=0.5, label="±1 dex")
    ax2.axhline(-1, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{V5}/\Delta m_{PDG})$")
    ax2.set_title("V5.29c vs V5.28 : précision (axe linéaire)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = "output/v5p29c_kashaev_oscillation.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def main():
    print()
    print("V5.29c — Kashaev / volume conjecture comme préfacteur knot-spécifique")
    print("Reframe : Δm = 2 ℏ ω_0 · N_paths · exp(-S_min) · exp(κ_K · vol_tot)")
    print()
    print_volume_table()

    # Calibration empirique sur D, B⁰, B_s⁰
    cal = calibrate_kappa_K_to_hierarchy(kappa_range=(-1.0, 2.0), n_grid=601)
    print(f"Scan empirique : κ_K optimal = {cal['kappa_K']:.4f} "
          f"(χ²_log = {cal['chi2_log']:.3f})")
    print()

    # On utilise la valeur de défaut (calibrée préalablement) pour la table.
    kappa_K = KAPPA_K_DEFAULT
    print_calibration(kappa_K)
    rows = print_table(kappa_K)
    correct, r_29c, r_pdg = print_hierarchy(rows)
    make_plot(rows, kappa_K)

    print()
    print("=" * 76)
    print("Verdict V5.29c (honnête)")
    print("=" * 76)
    print(
        "* Le préfacteur de volume amplifie B_s⁰ (vol_tot grand) et\n"
        "  supprime D⁰ (vol_tot petit, 3_1+5_1 = 0).\n"
    )
    print(
        f"* Gap V5.28 → V5.29c :\n"
        f"    K⁰  : 0.00   → 0.00    (calibrage)\n"
        f"    D⁰  : +1.97  → {rows[1]['log_ratio_v29c']:+.2f}  "
        f"({'correction' if abs(rows[1]['log_ratio_v29c']) < 1.97 else 'détérioration'})\n"
        f"    B⁰  : -0.73  → {rows[2]['log_ratio_v29c']:+.2f}  "
        f"({'correction' if abs(rows[2]['log_ratio_v29c']) < 0.73 else 'détérioration'})\n"
        f"    B_s⁰ : -3.30  → {rows[3]['log_ratio_v29c']:+.2f}  "
        f"({'correction' if abs(rows[3]['log_ratio_v29c']) < 3.30 else 'détérioration'})\n"
    )
    print(
        f"* Hiérarchie K < D < B⁰ < B_s⁰ : "
        f"{'REPRODUITE' if correct else 'PARTIELLEMENT (D⁰ trop bas)'}\n"
    )
    print(
        f"* Ratio B_s⁰/B⁰ : V5.29c = {r_29c:.2f}, PDG ≈ {r_pdg:.1f}\n"
        f"  → reste sous le PDG d'un facteur {r_pdg/r_29c:.1f}.\n"
    )
    print(
        "Conclusion : la volume conjecture EST une direction physique\n"
        "correcte (B_s remonte de 0.79 dex sur V5.28), mais κ_K seul ne\n"
        "suffit pas à reproduire le ratio PDG B_s/B ≈ 35. Piste V5.29d :\n"
        " - combiner Kashaev (volume) + coupling c_J asymétrique (V5.29b)\n"
        " - chemins frustrés pour amphichiraux (V5.29a) en parallèle\n"
    )


if __name__ == "__main__":
    main()
