"""V5p27 (nouveau) — Oscillations K⁰-K̄⁰, B⁰-B̄⁰, D⁰-D̄⁰ via P15.

Script principal V5p27.  Génère :
  - Table des Δm V5 vs PDG (K0, D0, B0, Bs0)
  - Plot output/v5p27_oscillations.png

Cadre CLAUDE.md règle 0 : la « saveur » n'est PAS un nombre quantique
fondamental TQNT ; les vrais phénomènes physiques (oscillations K-K̄,
etc.) sont des reconnexions topologiques P15.

Note : la matrice CKM (mélange de saveurs) et la phase δ_CP (qui
confondait la chiralité Dirac avec l'opération miroir topologique) ont
été retirées : ce sont des artefacts du SM (artefact dérivé d'un
artefact).  En TQNT, la chiralité = opération miroir V_K*(t) =
V_K(t⁻¹), pas la handedness L/R des spineurs.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.p15_topological_reconnection import (
    oscillation_comparison_table,
    chirality_sigma,
    is_link_amphichiral,
    topological_distance_to_mirror,
    MESON_OSCILLATION_ASSIGNMENTS,
    DELTA_M_PDG,
)


def print_oscillation_table():
    print("=" * 80)
    print("V5p27 — Oscillations mésons neutres (Δm V5 vs PDG)")
    print("=" * 80)
    print(f"{'Particule':<8} {'Knots V5':<28} {'Δm V5 (MeV)':>14} "
          f"{'Δm PDG (MeV)':>14} {'log10(V5/PDG)':>14}")
    print("-" * 80)
    rows = oscillation_comparison_table()
    for r in rows:
        knots_str = " + ".join(r["knots"])
        log_str = (
            f"{r['log_ratio']:+.2f}" if r["log_ratio"] is not None else "—"
        )
        print(
            f"{r['name']:<8} {knots_str:<28} "
            f"{r['delta_m_v5_MeV']:>14.3e} {r['delta_m_pdg_MeV']:>14.3e} "
            f"{log_str:>14}"
        )
    print()
    return rows


def make_plot(rows):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # --- Plot : Δm V5 vs PDG ---
    names = [r["name"] for r in rows]
    dm_v5 = [r["delta_m_v5_MeV"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, dm_v5, width, label="V5 (P15)", color="#1f77b4")
    ax.bar(x + width / 2, dm_pdg, width, label="PDG", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("0", "⁰") for n in names])
    ax.set_ylabel(r"$\Delta m$ (MeV)")
    ax.set_title("Oscillations mésons neutres (V5 calibré sur K⁰)")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for i, (v5, pdg) in enumerate(zip(dm_v5, dm_pdg)):
        log = np.log10(v5 / pdg)
        ax.text(
            i, max(v5, pdg) * 3, f"{log:+.2f}",
            ha="center", fontsize=9, color="darkred",
        )

    plt.tight_layout()
    out_path = "output/v5p27_oscillations.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot sauvegardé : {out_path}")


def main():
    print()
    print("V5p27 (NOUVEAU) — Reconnexion topologique P15 et oscillations")
    print("CLAUDE.md règle 0 : saveur = artefact SM, pas un concept TQNT.")
    print()

    # Identifications topologiques
    print("Identifications topologiques V5 (catalogue diffusion) :")
    for name, cfg in MESON_OSCILLATION_ASSIGNMENTS.items():
        d = topological_distance_to_mirror(cfg["knots"])
        amphi = is_link_amphichiral(cfg["knots"])
        chir = sum(chirality_sigma(n) for n in cfg["knots"])
        print(
            f"  {name:<5}: {' + '.join(cfg['knots']):<28}  "
            f"d(L,L*)={d:>3}  amphi={amphi}  Σσ={chir:+d}"
        )
    print()

    rows = print_oscillation_table()
    make_plot(rows)

    print()
    print("=== Conclusion honnête ===")
    print("- K⁰ calibre Γ_P_osc (Δm V5 = Δm PDG par construction)")
    print("- D⁰ : V5 prédit dans facteur 0.52 (log = -0.28) — bonne précision")
    print("- B⁰ : V5 sous-prédit (~3000×, log = -3.34) — limite du modèle")
    print("- Bs⁰ : V5 sous-prédit (~60×, log = -1.76) — limite du modèle")
    print()
    print("Le mécanisme P15 capte l'échelle des oscillations légères (K, D)")
    print("mais sous-prédit B/Bs. Une extension P15 avec dépendance en c_J")
    print("est nécessaire pour les mésons lourds (V5.28 future).")


if __name__ == "__main__":
    main()
