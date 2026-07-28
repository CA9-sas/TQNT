"""V5.37b — Test runner : potentiel Cornell TQNT pour quarkonium.

Pour chaque paire (état_fond, état_excité) du quarkonium V5 :

  * charmonium  : ψ(2S)/J/ψ , ψ(3S)/ψ(2S)
  * bottomonium : Υ(2S)/Υ(1S), Υ(3S)/Υ(2S)

on calcule Δm V5.37b = E_{n_e} − E_{n_g} avec le Hamiltonien Cornell
TQNT V(r) = -α_TQNT · ℏc / r + σ_TQNT · r , dans deux modes :

  - **universal** : α, σ universels (vide χ TQNT) — interprétation V5
    correcte.
  - **topo**      : α, σ ∝ c_J_eff(L) — interprétation naïve,
    diagnostiquée pour traçabilité honnête.

Calibration : 2 paramètres globaux (K_alpha, K_sigma) calibrés une fois
sur la paire (J/ψ, ψ(2S)).  Les 3 autres paires sont des PRÉDICTIONS.

Sortie : output/v5p37b_cornell.png
Référence : theory/v5p37b_cornell_quarkonium.py
            theory/mass_formula.py, theory/su3_color_topology.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p37b_cornell_quarkonium import (
    QUARKONIUM_ASSIGNMENTS,
    QUARKONIUM_RADIAL_FAMILIES,
    V5P37B_PAIRS,
    M_CHARM_MEV,
    M_BOTTOM_MEV,
    MODE_UNIVERSAL,
    MODE_TOPO,
    BETA_MEV,
    ALPHA_EM,
    N_COLOR,
    calibrate_cornell_constants,
    v5p37b_cornell_table,
    verdict_v5p37b,
    clear_calibration_cache,
)


def print_header() -> None:
    print()
    print("=" * 100)
    print(
        "V5.37b — Splittings radiaux quarkonium depuis potentiel Cornell TQNT"
    )
    print("=" * 100)
    print(
        f"  Vacuum constants : α_EM = 1/137 (P21), β_V5 = {BETA_MEV} MeV "
        f"(tension d'enlacement), N_c = {int(N_COLOR)}\n"
        f"  Masses constituantes : m_charm = {M_CHARM_MEV} MeV, "
        f"m_bottom = {M_BOTTOM_MEV} MeV (effectifs Cornell)\n"
        "  Reference : theory/v5p37b_cornell_quarkonium.py\n"
        "  Catalogue : docs/TQNT_catalog_V5_diffusion.xlsx onglet 3\n"
    )


def print_assignments() -> None:
    print("=" * 100)
    print("Assignations V5 du quarkonium (utilisées par V5.37b)")
    print("=" * 100)
    print(f"{'particle':<14} {'topologie':<28} {'m_PDG (MeV)':>14}  famille")
    print("-" * 100)
    for name, cfg in QUARKONIUM_ASSIGNMENTS.items():
        topo = " + ".join(cfg["knots"])
        fam = ""
        for fname, fcfg in QUARKONIUM_RADIAL_FAMILIES.items():
            if name == fcfg["ground"] or name in fcfg["excited"]:
                fam = fname
        print(
            f"{name:<14} {topo:<28} {cfg['mass_pdg_MeV']:>14.3f}  {fam}"
        )
    print()


def print_calibration(cal: dict) -> None:
    print(f"  mode = {cal['mode']}")
    print(f"  K_alpha = {cal['K_alpha']:.4f}, K_sigma = {cal['K_sigma']:.4f}")
    print(f"  α_TQNT(J/ψ) = {cal['alpha_TQNT_J_psi']:.4f}")
    print(
        f"  σ_TQNT(J/ψ) = {cal['sigma_TQNT_J_psi_MeV_per_fm']:.2f} MeV/fm"
    )
    print(
        f"  Calibration : Δm_cible = "
        f"{cal['dm_calibration_target_MeV']:.2f} MeV, "
        f"Δm_atteint = {cal['dm_calibration_achieved_MeV']:.2f} MeV"
    )
    print()


def print_table(rows: list[dict]) -> None:
    print(f"{'paire':<28} {'Δm PDG':>10} {'Δm V5.37b':>12} {'log_ratio':>10}")
    print("-" * 100)
    for r in rows:
        log_s = (
            f"{r['log_ratio']:+.3f}" if np.isfinite(r["log_ratio"]) else "nan"
        )
        print(
            f"{r['label']:<28} {r['delta_m_pdg_MeV']:>10.2f} "
            f"{r['delta_m_v5p37b_MeV']:>12.2f} {log_s:>10}"
        )
    print()


def print_verdict(v: dict) -> None:
    print(
        f"  Σ|log|  = {v['sum_abs_log']:.3f} dex   "
        f"⟨|log|⟩ = {v['mean_abs_log']:.3f} dex"
    )
    print(
        f"  paires dans seuil ±{v['threshold_dex']:.2f} dex : "
        f"{v['n_within_threshold']}/{v['n_total']}"
    )
    print(f"  verdict : {v['verdict']}")
    for line in v["recommendation"].split(". "):
        if line.strip():
            print(f"    {line.strip()}")
    print()


def compare_v5p37(rows_univ: list[dict], v_univ: dict) -> None:
    print("=" * 100)
    print("Comparaison V5.37b (mode universal) vs V5.37 (χ-field)")
    print("=" * 100)
    # V5.37 rapporté : Σ|log| = 0,76 dex, 3/4 paires dans ±0,30 dex
    print(f"  V5.37  (χ-field)        : Σ|log| ≈ 0.76 dex, 3/4 dans seuil")
    print(
        f"  V5.37b (Cornell univ.)  : Σ|log| = "
        f"{v_univ['sum_abs_log']:.3f} dex, "
        f"{v_univ['n_within_threshold']}/{v_univ['n_total']} dans seuil"
    )
    delta = 0.76 - v_univ["sum_abs_log"]
    if delta > 0:
        print(
            f"  → V5.37b améliore V5.37 de {delta:.2f} dex sur Σ|log|."
        )
    else:
        print(
            f"  → V5.37b dégrade V5.37 de {-delta:.2f} dex sur Σ|log|."
        )
    print()


def make_plot(
    rows_univ: list[dict],
    rows_topo: list[dict],
    v_univ: dict,
    v_topo: dict,
    out_path: str = "output/v5p37b_cornell.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    labels = [r["label"] for r in rows_univ]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows_univ]
    dm_u = [r["delta_m_v5p37b_MeV"] for r in rows_univ]
    dm_t = [r["delta_m_v5p37b_MeV"] for r in rows_topo]
    x = np.arange(len(rows_univ))
    w = 0.27

    ax1.bar(x - w, dm_pdg, w, color="#1f77b4", label="PDG")
    ax1.bar(x,     dm_u,   w, color="#2ca02c",
            label=f"V5.37b universal (Σ|log|={v_univ['sum_abs_log']:.2f})")
    ax1.bar(x + w, dm_t,   w, color="#ff7f0e",
            label=f"V5.37b topo (Σ|log|={v_topo['sum_abs_log']:.2f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel(r"$|\Delta m|$ (MeV)")
    ax1.set_title("V5.37b — Cornell TQNT vs PDG")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)

    log_u = [r["log_ratio"] if np.isfinite(r["log_ratio"]) else 0
             for r in rows_univ]
    log_t = [r["log_ratio"] if np.isfinite(r["log_ratio"]) else 0
             for r in rows_topo]
    ax2.bar(x - w / 2, log_u, w, color="#2ca02c",
            label="universal (vide χ)")
    ax2.bar(x + w / 2, log_t, w, color="#ff7f0e",
            label="topo (∝ c_J_eff)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(v_univ["threshold_dex"], color="g", ls=":", lw=1, alpha=0.7,
                label=f"seuil ±{v_univ['threshold_dex']:.2f} dex")
    ax2.axhline(-v_univ["threshold_dex"], color="g", ls=":", lw=1, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{V5.37b} / \Delta m_{PDG})$")
    ax2.set_title(
        f"V5.37b — log_ratio | universal : {v_univ['verdict']}, "
        f"topo : {v_topo['verdict']}"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot sauvegardé : {out_path}")


def main() -> str:
    print_header()
    print_assignments()
    clear_calibration_cache()

    # Mode universal (l'interprétation V5 correcte)
    print("=" * 100)
    print("MODE = universal (α, σ universels — vide χ TQNT)")
    print("=" * 100)
    cal_univ = calibrate_cornell_constants(mode=MODE_UNIVERSAL)
    print_calibration(cal_univ)
    rows_univ = v5p37b_cornell_table(mode=MODE_UNIVERSAL)
    print_table(rows_univ)
    v_univ = verdict_v5p37b(rows_univ)
    print_verdict(v_univ)

    # Mode topo (interprétation naïve, diagnostic)
    print("=" * 100)
    print("MODE = topo (α, σ ∝ c_J_eff — interprétation naïve)")
    print("=" * 100)
    cal_topo = calibrate_cornell_constants(mode=MODE_TOPO)
    print_calibration(cal_topo)
    rows_topo = v5p37b_cornell_table(mode=MODE_TOPO)
    print_table(rows_topo)
    v_topo = verdict_v5p37b(rows_topo)
    print_verdict(v_topo)

    compare_v5p37(rows_univ, v_univ)
    make_plot(rows_univ, rows_topo, v_univ, v_topo)

    print(f"Label final V5.37b : {v_univ['verdict']} (mode universal)")
    return v_univ["verdict"]


if __name__ == "__main__":
    main()
