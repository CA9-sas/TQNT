"""V5.35e — Test runner : extension de V5.35d a K_S/K_L, quarkonium, XYZ.

Pour chaque systeme :
* K_S - K_L                    (oscillation chirale, meme nœuds)
* K_S oscillation (CP+)        (Δm V5.35d standard sur K0)
* psi(2S) - J/psi              (splitting radial charmonium)
* Y(2S) - Y(1S)                (splitting radial bottomonium)
* Z_c(3900) - X(3872)          (memes nœuds, J^PC distincts)

on calcule Δm V5.35d generalisee, log_ratio vs PDG, et le verdict
d'universalite.

Sortie : output/v5p35e_extended.png
Reference : theory/v5p35e_extended_systems.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.v5p35e_extended_systems import (
    EXTENDED_TOPOLOGY,
    DELTA_M_EXTENDED_PDG_MEV,
    V5p35D_DELTA,
    V5p35D_FORM,
    V5p35D_BETA,
    V5p35D_FORM_B,
    omega_chi_v5p35d,
    delta_m_v5p35d_generalized,
    v5p35e_extended_table,
    verdict_v5p35e,
)


def print_header() -> None:
    print()
    print("=" * 100)
    print("V5.35e — Extension de V5.35d a K_S/K_L, quarkonium, XYZ exotiques")
    print("=" * 100)
    print(
        f"Parametres calibres V5.35d  : δ = {V5p35D_DELTA:+.3f}  "
        f"forme = {V5p35D_FORM!r}\n"
        f"Parametres calibres V5.35b  : β = {V5p35D_BETA:+.3f}  "
        f"forme = {V5p35D_FORM_B!r}\n"
    )


def print_topology() -> None:
    print("=" * 100)
    print("Topologies V5 (source : docs/TQNT_catalog_V5_diffusion.xlsx)")
    print("=" * 100)
    print(f"{'particle':<14} {'topologie':<30} {'m_PDG (MeV)':>14}  note")
    print("-" * 100)
    for name, cfg in EXTENDED_TOPOLOGY.items():
        topo = " + ".join(cfg["knots"])
        lc = " [LOWCONF]" if cfg.get("lowconf") else ""
        print(
            f"{name:<14} {topo:<30} {cfg['mass_pdg']:>14.3f}  "
            f"{cfg['note']}{lc}"
        )
    print()


def print_table(rows: list[dict]) -> None:
    print("=" * 100)
    print(
        "Δm V5.35d (generalisee) vs PDG sur 5 systemes etendus"
    )
    print("=" * 100)
    print(
        f"{'systeme':<26} {'mode':<22} {'appl?':<6} "
        f"{'Δm V5.35d':>13} {'Δm PDG':>13} {'log10 ratio':>13}"
    )
    print("-" * 100)
    for r in rows:
        dm_v = r["delta_m_v5p35d_MeV"]
        dm_p = r["delta_m_pdg_MeV"]
        lr = r["log_ratio"]
        dm_v_s = (
            f"{dm_v:.4e}" if dm_v != 0.0 or r["applicable"]
            else "0 (none)"
        )
        dm_p_s = f"{dm_p:.4e}" if np.isfinite(dm_p) else "n/a"
        lr_s = f"{lr:+.4f}" if np.isfinite(lr) else "n/a"
        print(
            f"{r['label']:<26} {r['mode']:<22} {str(r['applicable']):<6} "
            f"{dm_v_s:>13} {dm_p_s:>13} {lr_s:>13}"
        )
    print()
    print("Notes par systeme :")
    for r in rows:
        print(f"  * {r['label']:<26} : {r['note']}")
    print()


def print_omega_chi(rows: list[dict]) -> None:
    print("=" * 100)
    print("Frequences chirales ω_χ V5.35d par particule (debug)")
    print("=" * 100)
    print(
        f"{'particle':<14} {'topologie':<30} "
        f"{'n_comp':>6} {'ω_χ':>14} {'fact_3body':>11}"
    )
    print("-" * 100)
    seen = set()
    for name in EXTENDED_TOPOLOGY:
        if name in seen:
            continue
        seen.add(name)
        try:
            r = omega_chi_v5p35d(name)
        except Exception as e:
            print(f"{name:<14} (erreur : {e})")
            continue
        topo = " + ".join(r["knots"])
        print(
            f"{name:<14} {topo:<30} {r['n_comp']:>6} "
            f"{r['omega_chi']:>14.6e} {r['factor_3body']:>11.4f}"
        )
    print()


def print_verdict(v: dict) -> None:
    print("=" * 100)
    print("Verdict V5.35e (honnete)")
    print("=" * 100)
    print(f"  applicables : {v['applicable_count']}/{v['total_count']}")
    if np.isfinite(v["mean_log_applicable"]):
        print(
            f"  <|log_ratio|> sur applicables : "
            f"{v['mean_log_applicable']:.4f} dex"
        )
    print(f"  universalite : {v['universality']}")
    print(f"  recommandation :")
    for line in v["recommendation"].split(". "):
        if line.strip():
            print(f"    {line.strip()}")
    print()


def make_plot(rows: list[dict],
              out_path: str = "output/v5p35e_extended.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # --- Panneau 1 : Δm V5.35d vs PDG (echelle log) ---
    labels = [r["label"] for r in rows]
    dm_v = [abs(r["delta_m_v5p35d_MeV"]) for r in rows]
    dm_p = [r["delta_m_pdg_MeV"] for r in rows]
    # Remplace 0 par 1e-25 pour pouvoir tracer en log.
    dm_v_plot = [max(x, 1e-25) for x in dm_v]
    dm_p_plot = [max(x if np.isfinite(x) else 1e-25, 1e-25) for x in dm_p]
    x = np.arange(len(rows))
    w = 0.35
    bars1 = ax1.bar(
        x - w / 2, dm_v_plot, w, label="V5.35d generalisee",
        color=["#2ca02c" if r["applicable"] else "#d62728" for r in rows],
    )
    ax1.bar(x + w / 2, dm_p_plot, w, label="PDG", color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel(r"$|\Delta m|$ (MeV)")
    ax1.set_title("V5.35e — Δm V5.35d (vert: applicable, rouge: non-applicable)")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    # Annoter les bars avec mode
    for bar, r in zip(bars1, rows):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 1e-25) * 1.5,
            r["mode"].replace("_", "\n"),
            ha="center", va="bottom", fontsize=7,
        )

    # --- Panneau 2 : log_ratio par systeme ---
    lr = [r["log_ratio"] if np.isfinite(r["log_ratio"]) else 0 for r in rows]
    finite = [np.isfinite(r["log_ratio"]) for r in rows]
    colors = []
    for f, r in zip(finite, rows):
        if not f:
            colors.append("#7f7f7f")
        elif r["applicable"] and abs(r["log_ratio"]) < 0.1:
            colors.append("#2ca02c")
        elif r["applicable"]:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")
    ax2.bar(x, lr, color=colors)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(0.1, color="g", ls=":", lw=1, alpha=0.6,
                label="seuil ±0.1 dex")
    ax2.axhline(-0.1, color="g", ls=":", lw=1, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("log10(Δm V5.35d / Δm PDG)")
    ax2.set_title("V5.35e — log_ratio (gris : non applicable, vert : ±0.1)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot sauvegarde : {out_path}")


def main() -> str:
    print_header()
    print_topology()
    rows = v5p35e_extended_table()
    print_table(rows)
    print_omega_chi(rows)
    v = verdict_v5p35e(rows)
    print_verdict(v)
    make_plot(rows)
    print(f"Label final : {v['universality']}")
    return v["universality"]


if __name__ == "__main__":
    main()
