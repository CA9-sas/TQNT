"""V5.37 — Test runner : splittings radiaux du quarkonium depuis Schrödinger.

Pour chaque paire (état_fond, état_excité) du quarkonium V5 :

  * charmonium  : ψ(2S)/J/ψ , ψ(3S)/ψ(2S)
  * bottomonium : Υ(2S)/Υ(1S), Υ(3S)/Υ(2S)

on calcule deux Δm V5.37 :

  - **interp_A** (intra-lien) : sur le lien fondamental, lit E_2 − E_1
    et applique Δm_A = m_fond × (E_2 − E_1)/|E_1|.  Image SM classique
    où n=2 est l'excitation radiale du même puits.
  - **interp_B** (inter-lien) : sur le lien fondamental et sur le lien
    excité du catalogue V5 (différents en général), lit E_1 sur chacun
    et applique Δm_B = m_fond × |E_1[L'] − E_1[L]| / |E_1[L]|.  Image
    V5 où ψ(2S) est un autre lien.

Aucun paramètre n'est calibré : log_ratio est purement diagnostic.

Sortie : output/v5p37_radial.png
Référence : theory/v5p37_radial_splittings.py
            theory/schrodinger.py + simulation/solver.py
            docs/TQNT_catalog_V5_diffusion.xlsx onglet 3
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p37_radial_splittings import (
    QUARKONIUM_ASSIGNMENTS,
    V5P37_PAIRS,
    DEFAULT_N_COARSE,
    DEFAULT_EPSILON,
    DEFAULT_N_CURVE_POINTS,
    v5p37_radial_table,
    verdict_v5p37,
    clear_spectrum_cache,
)


def print_header() -> None:
    print()
    print("=" * 100)
    print(
        "V5.37 — Splittings radiaux quarkonium depuis Schrödinger χ-field"
    )
    print("=" * 100)
    print(
        f"Solveur FEM : n_coarse={DEFAULT_N_COARSE}  epsilon={DEFAULT_EPSILON}"
        f"  n_curve_points={DEFAULT_N_CURVE_POINTS}\n"
        "Reference   : theory/v5p37_radial_splittings.py +\n"
        "              theory/schrodinger.py (assemble_system_multi)\n"
        "Catalogue   : docs/TQNT_catalog_V5_diffusion.xlsx onglet 3\n"
    )


def print_assignments() -> None:
    print("=" * 100)
    print("Assignations V5 du quarkonium")
    print("=" * 100)
    print(f"{'particle':<14} {'topologie':<28} {'m_PDG (MeV)':>14}  note")
    print("-" * 100)
    for name, cfg in QUARKONIUM_ASSIGNMENTS.items():
        topo = " + ".join(cfg["knots"])
        note = ""
        if cfg.get("lowconf"):
            note = "[LOWCONF] " + cfg.get("note", "")
        print(
            f"{name:<14} {topo:<28} {cfg['mass_pdg_MeV']:>14.3f}  {note}"
        )
    print()


def print_table(rows: list[dict]) -> None:
    print("=" * 100)
    print(
        "Δm V5.37 sur les 4 paires radiales (interp A : intra-lien, "
        "interp B : inter-lien)"
    )
    print("=" * 100)
    print(
        f"{'paire':<24} {'Δm PDG':>10} "
        f"{'Δm_A (MeV)':>12} {'log_A':>8} "
        f"{'Δm_B (MeV)':>12} {'log_B':>8} "
        f"{'same_link':>9}"
    )
    print("-" * 100)
    for r in rows:
        a = r["interp_A"]
        b = r["interp_B"]
        log_A = a["log_ratio"]
        log_B = b["log_ratio"]
        log_A_s = f"{log_A:+.3f}" if np.isfinite(log_A) else "nan"
        log_B_s = f"{log_B:+.3f}" if np.isfinite(log_B) else "nan"
        print(
            f"{r['label']:<24} {r['delta_m_pdg_MeV']:>10.2f} "
            f"{a['delta_m_v5p37_MeV']:>12.3f} {log_A_s:>8} "
            f"{b['delta_m_v5p37_MeV']:>12.3f} {log_B_s:>8} "
            f"{str(r['same_link']):>9}"
        )
    print()


def print_spectra(rows: list[dict]) -> None:
    print("=" * 100)
    print("Spectres Schrödinger χ-field (4 niveaux par lien)")
    print("=" * 100)
    seen: set[tuple] = set()
    for r in rows:
        for key, lvls in (
            (tuple(r["knots_ground"]), r["E_ground_levels"]),
            (tuple(r["knots_excited"]), r["E_excited_levels"]),
        ):
            if key in seen:
                continue
            seen.add(key)
            label = " + ".join(key)
            evs = " ".join(f"{e:+.3f}" for e in lvls[:4])
            print(f"  {label:<24} E = [ {evs} ]")
    print()


def print_verdict(v: dict, rows: list[dict]) -> None:
    print("=" * 100)
    print("Verdict V5.37 (honnête)")
    print("=" * 100)
    print(
        f"  Σ|log| (meilleure interprétation par paire) = "
        f"{v['sum_abs_log_best']:.3f} dex"
    )
    print(
        f"  ⟨|log|⟩ (meilleure interprétation par paire) = "
        f"{v['mean_abs_log_best']:.3f} dex"
    )
    print(
        f"  paires dans seuil ±{v['threshold_dex']:.2f} dex : "
        f"{v['n_within_threshold']}/{v['n_total']}"
    )
    print(f"  meilleure interp par paire : {v['best_interp_per_row']}")
    print(f"  verdict     : {v['verdict']}")
    print(f"  recommandation :")
    for line in v["recommendation"].split(". "):
        if line.strip():
            print(f"    {line.strip()}")
    print()


def make_plot(rows: list[dict], v: dict,
              out_path: str = "output/v5p37_radial.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    labels = [r["label"] for r in rows]
    dm_pdg = [r["delta_m_pdg_MeV"] for r in rows]
    dm_A = [r["interp_A"]["delta_m_v5p37_MeV"] for r in rows]
    dm_B = [r["interp_B"]["delta_m_v5p37_MeV"] for r in rows]
    x = np.arange(len(rows))
    w = 0.27

    ax1.bar(x - w, dm_pdg, w, color="#1f77b4", label="PDG")
    ax1.bar(x,     dm_A,   w, color="#2ca02c", label="V5.37 interp_A (intra-lien)")
    ax1.bar(x + w, dm_B,   w, color="#ff7f0e", label="V5.37 interp_B (inter-lien)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel(r"$|\Delta m|$ (MeV)")
    ax1.set_title("V5.37 — Δm radiaux vs PDG")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    log_A = [r["interp_A"]["log_ratio"] if np.isfinite(r["interp_A"]["log_ratio"])
             else 0 for r in rows]
    log_B = [r["interp_B"]["log_ratio"] if np.isfinite(r["interp_B"]["log_ratio"])
             else 0 for r in rows]
    ax2.bar(x - w / 2, log_A, w, color="#2ca02c", label="interp_A")
    ax2.bar(x + w / 2, log_B, w, color="#ff7f0e", label="interp_B")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(v["threshold_dex"], color="g", ls=":", lw=1, alpha=0.7,
                label=f"seuil ±{v['threshold_dex']:.2f} dex")
    ax2.axhline(-v["threshold_dex"], color="g", ls=":", lw=1, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\log_{10}(\Delta m_{V5.37} / \Delta m_{PDG})$")
    ax2.set_title(
        f"V5.37 — log_ratio | verdict : {v['verdict']} | "
        f"Σ|log|={v['sum_abs_log_best']:.2f}"
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
    clear_spectrum_cache()
    rows = v5p37_radial_table()
    print_table(rows)
    print_spectra(rows)
    v = verdict_v5p37(rows)
    print_verdict(v, rows)
    make_plot(rows, v)
    print(f"Label final : {v['verdict']}")
    return v["verdict"]


if __name__ == "__main__":
    main()
