#!/usr/bin/env python3
"""V5.21 -- Dérivation de Δc_min ≥ 22 et rachat (ou réfutation) de V5.20.

Calcule Δc_min pour chacun des 4 mécanismes de sélection topologique,
sur les 15 paires croisées et 6 paires propres des 6 STRICT
amphichiraux V5 §13.7 :
  1) Q_phys / χ-balance
  2) Linking number (Kanenobu–Murakami)
  3) Vassiliev (Bar-Natan)
  4) SU(2)_135 fusion modulaire

Imprime un tableau et rend un verdict :
  Δc_min_overall ≥ 22 → V5.20 redeemed (Fermi-LAT à 10^23σ couvert)
  Δc_min_overall <  22 → V5.20 REFUTÉ (Classe E doit être abandonnée)

Génère également ``output/v5p21_dc_selection.png``.

Run :
    PYTHONPATH=. python3 scripts/V5_pearl_internal/dc_22_derivation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.dc_selection_rule import (
    AMPHICHIRAL_DATA,
    DELTA_C_MIN_OVERALL,
    all_amphichiral_pairs_delta_c_min,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p21_dc_selection.png"


# ---------------------------------------------------------------------------
# 1. Tableau ASCII
# ---------------------------------------------------------------------------

def print_table(pairs: list, title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    header = (
        f"{'pair':36s} {'c_i':>4} {'c_j':>4} "
        f"{'ΔcQ':>5} {'ΔcLk':>5} {'ΔcVa':>5} {'ΔcS2':>5} "
        f"{'MIN':>5}  bind"
    )
    print(header)
    print("-" * 96)
    for e in pairs:
        name = f"{e['name_i']} + {e['name_j']}"
        binding_short = {
            "Q_phys_min_crossings": "Q",
            "Lk_min_crossings": "Lk",
            "vassiliev_min": "Vass",
            "su2_135_min": "SU2",
        }[e["binding_mechanism"]]
        print(
            f"{name:36s} {e['c_i']:>4} {e['c_j']:>4} "
            f"{e['Q_phys_min_crossings']:>5} {e['Lk_min_crossings']:>5} "
            f"{e['vassiliev_min']:>5} {e['su2_135_min']:>5} "
            f"{e['delta_c_min_total']:>5}  {binding_short}"
        )


# ---------------------------------------------------------------------------
# 2. Figure
# ---------------------------------------------------------------------------

def make_figure(pairs: list, out_path: Path) -> None:
    """Bar chart des 4 mécanismes par paire + ligne horizontale à 22."""
    n = len(pairs)
    x = np.arange(n)
    w = 0.20
    dc_Q = [e["Q_phys_min_crossings"] for e in pairs]
    dc_Lk = [e["Lk_min_crossings"] for e in pairs]
    dc_V = [e["vassiliev_min"] for e in pairs]
    dc_S = [e["su2_135_min"] for e in pairs]
    dc_min = [e["delta_c_min_total"] for e in pairs]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5 * w, dc_Q, w, label=r"$\Delta c_Q$ (χ-balance)",
           color="#4c72b0")
    ax.bar(x - 0.5 * w, dc_Lk, w, label=r"$\Delta c_{Lk}$ (Kanenobu-Murakami)",
           color="#55a868")
    ax.bar(x + 0.5 * w, dc_V, w, label=r"$\Delta c_{\rm Vass}$ (Bar-Natan)",
           color="#c44e52")
    ax.bar(x + 1.5 * w, dc_S, w, label=r"$\Delta c_{SU(2)_{135}}$",
           color="#8172b2")

    # Min (curve)
    ax.plot(x, dc_min, "ko-", lw=1.5, markersize=6,
            label=r"$\Delta c_{\min} = \max$ des 4")

    # Horizontal Fermi threshold
    ax.axhline(22, color="red", linestyle="--", lw=2,
               label="Seuil Fermi-LAT : Δc ≥ 22")

    ax.set_xticks(x)
    labels = [f"{e['name_i']}\n+\n{e['name_j']}" for e in pairs]
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel(r"$\Delta c$ (croisements additionnels)")
    ax.set_title("V5.21 — Règle de sélection Δc par paire d'amphichiraux STRICT V5 §13.7\n"
                 f"Δc$_{{\\min}}$ absolu = {DELTA_C_MIN_OVERALL}  ⇒  "
                 + ("V5.20 RACHETÉ" if DELTA_C_MIN_OVERALL >= 22
                    else "V5.20 RÉFUTÉ"))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(40, max(dc_min) + 5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[fig] Saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 96)
    print("V5.21 — Dérivation de la règle de sélection Δc_min pour les perles DM")
    print("=" * 96)
    print()
    print("6 STRICT amphichiraux V5 §13.7 (CLAUDE.md DM) :")
    for a in AMPHICHIRAL_DATA:
        print(f"  {a.name:14s}  m = {a.mass_GeV:6.2f} GeV   c = {a.c:>2}   "
              f"Q_phys = {a.Q_phys}   h_link = {a.h:.4f}")

    pairs = all_amphichiral_pairs_delta_c_min()
    print_table(pairs["cross_pairs"],
                "Δc_min des 15 paires CROISÉES (C(6,2) = 15)")
    print_table(pairs["self_pairs"],
                "Δc_min des 6 paires PROPRES (auto-link)")

    # All entries combined for overall min
    all_entries = pairs["cross_pairs"] + pairs["self_pairs"]
    overall = min(e["delta_c_min_total"] for e in all_entries)
    binders = [e for e in all_entries if e["delta_c_min_total"] == overall]

    print()
    print("=" * 96)
    print("VERDICT V5.21")
    print("=" * 96)
    print(f"  Δc_min absolu (sur 21 paires) = {overall}")
    print(f"  Paires qui saturent la borne  : "
          + ", ".join(f"({b['name_i']} + {b['name_j']})" for b in binders))
    print(f"  Mécanisme dominant            : "
          + binders[0]["binding_mechanism"])
    print()

    if overall >= 22:
        print(f"  *** V5.20 REDEEMED ***")
        print(f"  Δc_min = {overall} ≥ 22 ⇒ suppression Vassiliev α^Δc ≤ α^22 ~ 1.0e-47")
        print(f"  ⇒ Cascade gamma sous le bruit Fermi-LAT EDGRB ; V5.20 compatible.")
        print(f"  ⇒ La signature 6-raies Classe E reste falsifiable (Fermi + CTA).")
    else:
        print(f"  *** V5.20 REFUTED by Fermi-LAT at 10^23 σ ***")
        print(f"  Δc_min = {overall} < 22 ⇒ suppression Vassiliev insuffisante.")
        print(f"  ⇒ V5 doit abandonner l'enchevêtrement spontané Classe E.")
        print(f"  ⇒ Le 6-raies γ signature n'est plus prédite par V5.21.")
        print(f"  ⇒ La masse DM doit être recalibrée sur les autres canaux (P22, P18).")

    print()
    print(f"DELTA_C_MIN_OVERALL constant (theory/dc_selection_rule.py) = {DELTA_C_MIN_OVERALL}")
    assert overall == DELTA_C_MIN_OVERALL, "Inconsistency"

    # Figure (on cross-pairs only, lisible)
    make_figure(pairs["cross_pairs"], OUTPUT_PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
