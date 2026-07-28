"""V5.24 -- Dérivation explicite des 3 générations fermioniques (script).

Ce script exécute la dérivation V5.24 et produit la figure
``output/v5p24_generations.png`` qui résume :

1. Les 6 pistes topologiques évaluées et leur verdict ;
2. Le spectre T(2,q) admissible (q ∈ {3,5,7}) sous contrainte P11 ;
3. Les masses V4 bare prédites vs. PDG (échelle log) ;
4. Verdict : N_gen=3 dérivé, hiérarchie des masses partielle.

Usage::

    python3 -m scripts.V5_generations.v5p24_3_generations_derivation
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.generations_topology import (
    M_E_PDG_MEV,
    M_MU_PDG_MEV,
    M_TAU_PDG_MEV,
    PISTES_V5p24,
    Q_MAX_PLASTICITY,
    TORUS_FAMILY_KNOTS,
    TORUS_FAMILY_Q,
    admissible_torus_orders,
    mass_ratios_pdg,
    mass_ratios_predicted,
    mass_v4_bare,
    n_generations_predicted,
    v5p24_summary,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "v5p24_generations.png")


# ── Panel 1 : 6 pistes ─────────────────────────────────────────────

VERDICT_COLORS = {
    "RETENU": "tab:green",
    "PARTIEL": "tab:olive",
    "INSUFFISANT": "tab:orange",
    "REJETE": "tab:red",
}


def panel_pistes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Six pistes V5.24 pour dériver $N_{\\rm gen} = 3$",
                 fontsize=11, fontweight="bold", loc="left")
    rows = len(PISTES_V5p24)
    for i, p in enumerate(PISTES_V5p24):
        y = 1.0 - (i + 0.5) / (rows + 0.5)
        color = VERDICT_COLORS.get(p.verdict, "black")
        ax.text(0.02, y, f"{p.label}. {p.name}", transform=ax.transAxes,
                fontsize=9, ha="left", va="center")
        ax.text(0.72, y, p.verdict, transform=ax.transAxes,
                fontsize=9, ha="left", va="center",
                color=color, fontweight="bold")
    ax.text(0.02, -0.01,
            "Mécanisme retenu : P5 (spectre T(2,2g+1) + plasticité P11).",
            transform=ax.transAxes, fontsize=8.5, style="italic")


# ── Panel 2 : spectre T(2,q) admissible ────────────────────────────

def panel_spectrum(ax: plt.Axes) -> None:
    ax.set_title("Spectre T(2,q) admissible (V5.24)",
                 fontsize=11, fontweight="bold", loc="left")
    qs_all = list(range(3, 12, 2))  # 3,5,7,9,11
    admissible = set(admissible_torus_orders())
    q_max = Q_MAX_PLASTICITY
    colors = ["tab:green" if q in admissible else "tab:red" for q in qs_all]
    labels = [f"T(2,{q})\n= {q}_1" if q in {3, 5, 7} else f"T(2,{q})\ninstable"
              for q in qs_all]
    ys = [1.0] * len(qs_all)
    ax.bar(range(len(qs_all)), ys, color=colors, alpha=0.7,
           edgecolor="black", linewidth=1.2)
    for i, lab in enumerate(labels):
        ax.text(i, 0.5, lab, ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.axvline(x=2.5, color="black", linestyle="--", alpha=0.5,
               label=f"barrière plasticité (q_max={q_max})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.5, -0.10,
            f"Trois nœuds toriques fermioniques retenus : {{3_1, 5_1, 7_1}}.",
            transform=ax.transAxes, fontsize=9, ha="center", style="italic")


# ── Panel 3 : masses V4 vs PDG (log) ───────────────────────────────

def panel_masses(ax: plt.Axes) -> None:
    ax.set_title("Masses prédites V4 bare vs PDG (échelle log)",
                 fontsize=11, fontweight="bold", loc="left")
    gens = [1, 2, 3]
    m_v5 = [mass_v4_bare(g) for g in gens]
    m_pdg = [M_E_PDG_MEV, M_MU_PDG_MEV, M_TAU_PDG_MEV]
    labels = [f"gen {g}\n({TORUS_FAMILY_KNOTS[g-1]})" for g in gens]
    x = np.arange(len(gens))
    width = 0.35
    ax.bar(x - width / 2, m_v5, width, label="V5.24 (V4 bare)",
           color="tab:blue", edgecolor="black")
    ax.bar(x + width / 2, m_pdg, width, label="PDG",
           color="tab:gray", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Masse (MeV)")
    ax.legend(loc="upper left", fontsize=9)
    for xi, mv, mp in zip(x, m_v5, m_pdg):
        ratio = mp / mv
        ax.text(xi, max(mv, mp) * 1.3, f"×{ratio:.1f}",
                ha="center", fontsize=8, color="darkred")
    ax.text(0.5, -0.18,
            "Génération 1 calibrée exactement (m_e = 0.512 MeV) ;\n"
            "ratios gen 2/3 sous-prédits ×7 et ×4 (limite asymptotique V4).",
            transform=ax.transAxes, fontsize=8.5, ha="center", style="italic")


# ── Panel 4 : verdict ──────────────────────────────────────────────

def panel_verdict(ax: plt.Axes) -> None:
    ax.set_axis_off()
    s = v5p24_summary()
    mp = mass_ratios_predicted()
    md = mass_ratios_pdg()
    ax.set_title("Verdict V5.24", fontsize=11, fontweight="bold", loc="left")
    lines = [
        f"$N_{{\\rm gen}}^{{V5.24}}$ = {s['n_generations_predicted']}  "
        f"(PDG : 3)  ✓ dérivé",
        f"Mécanisme : {s['mechanism']}",
        "",
        "Hiérarchie des masses (V4 bare) :",
        f"  $m_e$ prédit = {s['m_g1_MeV']:.3f} MeV (PDG 0.511)  ✓ calibrée",
        f"  $m_\\mu/m_e$ prédit = {mp['m_g2/m_g1_predicted']:.1f}  "
        f"(PDG {md['m_mu/m_e_PDG']:.1f})  ✗ sous-prédit ×7",
        f"  $m_\\tau/m_\\mu$ prédit = {mp['m_g3/m_g2_predicted']:.1f}  "
        f"(PDG {md['m_tau/m_mu_PDG']:.1f})  ✗ sous-prédit ×4",
        "",
        "Verdict honnête : 3 générations dérivé (RIGOUREUX) ;",
        "hiérarchie verticale prédite seulement en ordre de magnitude.",
        "Forme canonique V5 ($m = \\Omega_{ts}(L)$) requise pour quantitatif.",
    ]
    for i, line in enumerate(lines):
        y = 0.95 - i * 0.085
        ax.text(0.02, y, line, transform=ax.transAxes,
                fontsize=9, ha="left", va="center")


# ── Assemblage ──────────────────────────────────────────────────────

def main() -> str:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    panel_pistes(fig.add_subplot(gs[0, 0]))
    panel_spectrum(fig.add_subplot(gs[0, 1]))
    panel_masses(fig.add_subplot(gs[1, 0]))
    panel_verdict(fig.add_subplot(gs[1, 1]))
    fig.suptitle("V5.24 — Dérivation des 3 générations fermioniques",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] V5.24 figure écrite : {OUTPUT_PNG}")
    s = v5p24_summary()
    print(f"  N_gen prédit V5.24    = {s['n_generations_predicted']}")
    print(f"  Mécanisme retenu      = {s['mechanism']}")
    print(f"  Verdict global        = {s['verdict']}")
    print(f"  m_e prédit (gen 1)    = {s['m_g1_MeV']:.4f} MeV  "
          f"(PDG {M_E_PDG_MEV:.4f})")
    print(f"  m_g2 prédit (5_1)     = {s['m_g2_MeV']:.2f} MeV  "
          f"(PDG μ {M_MU_PDG_MEV:.2f})")
    print(f"  m_g3 prédit (7_1)     = {s['m_g3_MeV']:.2f} MeV  "
          f"(PDG τ {M_TAU_PDG_MEV:.2f})")
    return OUTPUT_PNG


if __name__ == "__main__":
    main()
