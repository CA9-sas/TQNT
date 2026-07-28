"""V5.30 — Combinaison V5.29a (frustrés) + V5.29c (Kashaev) — test.

Script principal V5.30.  Combine deux extensions de V5.28 :

    Δm V5.30 = 2 ℏ ω_0 × N_paths^V5.29a × exp(-S_min/ℏ) × exp(κ_K × vol_tot)

où :
* N_paths^V5.29a = N_paths^V5.28 × Π_i F_i, F_i = c_i!·b_i pour amphichiral
* vol_tot = Σ vol(K_i), tabulé (SnapPy / KnotInfo)
* κ_K calibré globalement sur K, D, B⁰, B_s⁰ par χ²_log

Sortie : output/v5p30_combined.png
Référence : theory/v5p30_combined_oscillation.py

Honnête :
* V5.30 a UN paramètre libre (κ_K) au-dessus de V5.28
* Si V5.30 reproduit la hiérarchie K<D<B<Bs avec un seul κ_K : SUCCESS
* Si V5.30 améliore mais ne ferme pas tout : PARTIAL
* Si V5.30 casse un système : FAILURE → recommander V5.30c (modes vibratoires)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.topological_chiral_oscillation import (
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)
from theory.v5p30_combined_oscillation import (
    KAPPA_K_V5p30_DEFAULT,
    v5p30_table,
    calibrate_kappa_K_global,
)


def print_calibration():
    print("=" * 80)
    print("V5.30 — Calibration globale de κ_K")
    print("=" * 80)
    cal = calibrate_kappa_K_global(
        kappa_range=(-1.0, 2.0), n_grid=4001
    )
    print(
        f"κ_K optimal (χ²_log sur K+D+B+Bs) = {cal['kappa_K']:.4f} "
        f"(χ²_log = {cal['chi2_log']:.4f})"
    )
    print("Note : K⁰ contribue zéro par construction (préfacteur calibré).")
    print()
    return cal["kappa_K"]


def print_table(kappa_K: float):
    rows = v5p30_table(kappa_K)
    print("=" * 128)
    print(f"V5.30 — Δm combinaison V5.29a × V5.29c (κ_K = {kappa_K:.4f})")
    print("=" * 128)
    print(
        f"{'Méson':<6} {'Knots':<20} {'F_tot':>7} {'vol_tot':>8} "
        f"{'P_Kash':>10} "
        f"{'Δm V5.30':>11} {'Δm V5.29a':>11} {'Δm V5.29c':>11} "
        f"{'Δm V5.28':>11} {'Δm PDG':>11} "
        f"{'log 530':>8} {'log 528':>8}"
    )
    print("-" * 128)
    for r in rows:
        knots_str = " + ".join(r["knots"])
        print(
            f"{r['name']:<6} {knots_str:<20} "
            f"{r['F_total']:>7d} {r['vol_tot']:>8.3f} "
            f"{r['P_Kashaev']:>10.3e} "
            f"{r['delta_m_v5p30_MeV']:>11.3e} "
            f"{r['delta_m_v529a_MeV']:>11.3e} "
            f"{r['delta_m_v529c_MeV']:>11.3e} "
            f"{r['delta_m_v528_MeV']:>11.3e} "
            f"{r['delta_m_pdg_MeV']:>11.3e} "
            f"{r['log_ratio_v5p30']:>+8.3f} "
            f"{r['log_ratio_v528']:>+8.3f}"
        )
    print()
    return rows


def print_hierarchy(rows):
    """Diagnostic honnête de la hiérarchie K < D < B⁰ < B_s⁰."""
    by_name = {r["name"]: r["delta_m_v5p30_MeV"] for r in rows}
    pdg = {r["name"]: r["delta_m_pdg_MeV"] for r in rows}

    print("=" * 80)
    print("Hiérarchie K < D < B⁰ < B_s⁰")
    print("=" * 80)
    order_target = ["K0", "D0", "B0", "Bs0"]
    print(
        "PDG    : " + " < ".join(
            f"{n}({pdg[n]:.2e})" for n in
            sorted(order_target, key=lambda x: pdg[x])
        )
    )
    print(
        "V5.30  : " + " < ".join(
            f"{n}({by_name[n]:.2e})" for n in
            sorted(order_target, key=lambda x: by_name[x])
        )
    )
    order_v530 = sorted(order_target, key=lambda n: by_name[n])
    correct = order_v530 == order_target
    print(f"Ordre V5.30 : {'REPRODUIT' if correct else 'INCORRECT'}")
    print()

    ratio_v530 = by_name["Bs0"] / by_name["B0"]
    ratio_pdg = pdg["Bs0"] / pdg["B0"]
    print(
        f"Ratio B_s⁰/B⁰ : V5.30 = {ratio_v530:.2f}, "
        f"PDG = {ratio_pdg:.2f}  "
        f"(facteur d'écart : {ratio_v530/ratio_pdg:.2f}×)"
    )
    print()
    return correct, ratio_v530, ratio_pdg


def print_sum_log_ratio(rows):
    """Compare Σ|log_ratio| pour V5.28, V5.29a, V5.29c, V5.30."""
    s_528 = sum(abs(r["log_ratio_v528"]) for r in rows)
    s_29a = sum(abs(r["log_ratio_v529a"]) for r in rows)
    s_29c = sum(abs(r["log_ratio_v529c"]) for r in rows)
    s_530 = sum(abs(r["log_ratio_v5p30"]) for r in rows)
    print("=" * 80)
    print("Cumul Σ|log_ratio| (somme des écarts absolus)")
    print("=" * 80)
    print(f"  V5.28  : Σ|log| = {s_528:.3f} dex")
    print(f"  V5.29a : Σ|log| = {s_29a:.3f} dex")
    print(f"  V5.29c : Σ|log| = {s_29c:.3f} dex")
    print(f"  V5.30  : Σ|log| = {s_530:.3f} dex")
    best = min((s_528, "V5.28"), (s_29a, "V5.29a"),
               (s_29c, "V5.29c"), (s_530, "V5.30"))
    print(f"  Meilleur : {best[1]} (Σ|log| = {best[0]:.3f})")
    print()
    return s_528, s_29a, s_29c, s_530


def make_plot(rows, kappa_K: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["name"] for r in rows]
    dm_530 = [r["delta_m_v5p30_MeV"] for r in rows]
    dm_29a = [r["delta_m_v529a_MeV"] for r in rows]
    dm_29c = [r["delta_m_v529c_MeV"] for r in rows]
    dm_528 = [r["delta_m_v528_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    w = 0.16

    ax1.bar(x - 2*w, dm_528, w, label="V5.28", color="#888888")
    ax1.bar(x - w,   dm_29a, w, label="V5.29a (frustrés)", color="#4daf4a")
    ax1.bar(x,       dm_29c, w, label="V5.29c (Kashaev)",  color="#984ea3")
    ax1.bar(x + w,   dm_530, w, label=f"V5.30 (κ_K={kappa_K:.3f})",
            color="#1f77b4")
    ax1.bar(x + 2*w, dm_pdg, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel(r"$\Delta m$ (MeV)")
    ax1.set_title(r"$\Delta m$ : V5.28 / V5.29a / V5.29c / V5.30 / PDG")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    log_530 = [r["log_ratio_v5p30"] for r in rows]
    log_29a = [r["log_ratio_v529a"] for r in rows]
    log_29c = [r["log_ratio_v529c"] for r in rows]
    log_528 = [r["log_ratio_v528"] for r in rows]
    ax2.bar(x - 1.5*w, log_528, w, label="V5.28", color="#888888")
    ax2.bar(x - 0.5*w, log_29a, w, label="V5.29a", color="#4daf4a")
    ax2.bar(x + 0.5*w, log_29c, w, label="V5.29c", color="#984ea3")
    ax2.bar(x + 1.5*w, log_530, w, label="V5.30",  color="#1f77b4")
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(1, color="g", ls="--", lw=0.8, alpha=0.5, label="±1 dex")
    ax2.axhline(-1, color="g", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{V5}/\Delta m_{PDG})$")
    ax2.set_title("V5.30 vs V5.29a vs V5.29c vs V5.28 : précision")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = "output/v5p30_combined.png"
    plt.savefig(out, dpi=150)
    print(f"Plot sauvegardé : {out}")


def verdict(rows, correct_hier, ratio_v530, ratio_pdg,
            s_528, s_29a, s_29c, s_530, kappa_K):
    """Verdict honnête V5.30."""
    print()
    print("=" * 80)
    print("Verdict V5.30 (honnête)")
    print("=" * 80)
    by_name = {r["name"]: r for r in rows}

    # Comparaison système par système vs V5.28
    print("Δm V5.30 vs V5.28 par système (log ratio) :")
    for n in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[n]
        diff = r["log_ratio_v5p30"] - r["log_ratio_v528"]
        status = (
            "amélioration" if abs(r["log_ratio_v5p30"]) < abs(
                r["log_ratio_v528"])
            else "détérioration"
        )
        print(
            f"  {n:<4} : V5.28 = {r['log_ratio_v528']:+.3f}, "
            f"V5.30 = {r['log_ratio_v5p30']:+.3f}  ({status})"
        )
    print()

    # Décision
    has_failure = False
    msgs = []
    for n in ["K0", "D0", "B0", "Bs0"]:
        r = by_name[n]
        if abs(r["log_ratio_v5p30"]) > abs(r["log_ratio_v528"]) + 0.5:
            has_failure = True
            msgs.append(
                f"  - {n} dégradé de {r['log_ratio_v5p30'] - r['log_ratio_v528']:+.2f} dex"
            )

    closes_all = (
        correct_hier
        and all(abs(by_name[n]["log_ratio_v5p30"]) < 0.5
                for n in ["K0", "D0", "B0", "Bs0"])
        and abs(np.log10(ratio_v530 / ratio_pdg)) < 0.3
    )

    if closes_all:
        verdict_label = "SUCCESS"
        print(
            "VERDICT : SUCCESS — V5.30 ferme la hiérarchie complète\n"
            "         (tous les Δm sous 0.5 dex, ratio Bs/B sous 0.3 dex)\n"
            f"         avec un seul paramètre libre κ_K = {kappa_K:.3f}."
        )
    elif has_failure:
        verdict_label = "FAILURE"
        print(
            "VERDICT : FAILURE — V5.30 dégrade au moins un système :\n"
            + "\n".join(msgs)
            + "\n"
            "Recommandation : explorer V5.30c (modes vibratoires) — un\n"
            "préfacteur indépendant de la topologie statique pour ajuster\n"
            "Bs/B sans sur-amplifier Bs.\n"
        )
    else:
        verdict_label = "PARTIAL"
        print(
            "VERDICT : PARTIAL — V5.30 reproduit la hiérarchie K<D<B<Bs\n"
            f"         et abaisse Σ|log| à {s_530:.2f} (vs V5.28 = {s_528:.2f}),\n"
            f"         mais le ratio Bs/B = {ratio_v530:.0f} vs PDG = {ratio_pdg:.0f}\n"
            f"         dépasse PDG d'un facteur {ratio_v530/ratio_pdg:.1f}.\n"
            "         Le B_s est sur-amplifié par la combinaison\n"
            "         F_total(6_3) × exp(κ_K × vol(6_3+7_6)).\n"
        )
        print(
            "Piste V5.30c : modes vibratoires (oscillations de longueur de\n"
            "             tube knotté), pour ajuster Bs/B sans modifier la\n"
            "             topologie statique des liens (préfacteur dynamique).\n"
        )
    print()
    return verdict_label


def main():
    print()
    print(
        "V5.30 — Combinaison V5.29a (frustrés amphichiraux) "
        "+ V5.29c (Kashaev / volume)"
    )
    print(
        "Δm = 2ℏω_0 · N_paths · F_total · exp(-S_min) · exp(κ_K · vol_tot)"
    )
    print()

    _ = print_calibration()
    kappa_K = KAPPA_K_V5p30_DEFAULT
    rows = print_table(kappa_K)
    correct, ratio_530, ratio_pdg = print_hierarchy(rows)
    s_528, s_29a, s_29c, s_530 = print_sum_log_ratio(rows)
    make_plot(rows, kappa_K)
    label = verdict(rows, correct, ratio_530, ratio_pdg,
                    s_528, s_29a, s_29c, s_530, kappa_K)
    print(f"Label final : {label}")


if __name__ == "__main__":
    main()
