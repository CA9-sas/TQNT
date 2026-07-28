#!/usr/bin/env python3
"""V5.22 -- Tentative derivation of Δc_min ≥ 30 (Fermi line search threshold).

Evaluates the 5 V5.22 candidate mechanisms on top of V5.21 selection rules
for all 21 pairs of STRICT amphichirals (V5 §13.7), prints a verdict, and
saves a comparison figure to ``output/v5p22_dc_30_derivation.png``.

Run :
    PYTHONPATH=. python3 scripts/V5_pearl_internal/v5p22_dc_30_derivation.py

Verdict :
    Δc_min^V5.22_honest ≥ 30 → V5.20 line search REDEEMED
    Δc_min^V5.22_honest <  30 → F3 falsification OPEN (V5 partially refuted)
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

from theory.dc_higher_order_selection import (
    DELTA_C_MIN_V5P22_HONEST,
    DELTA_C_MIN_V5P22_UPPER,
    FERMI_LINE_SEARCH_THRESHOLD,
    all_amphichiral_pairs_v5p22,
    evaluate_all_mechanisms,
)
from theory.dc_selection_rule import AMPHICHIRAL_DATA, DELTA_C_MIN_OVERALL

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p22_dc_30_derivation.png"


def print_mechanism_per_pair(pairs: list) -> None:
    """Detailed breakdown of the 5 V5.22 mechanisms per pair."""
    print()
    print("=" * 110)
    print("V5.22 -- 5 mechanisms per pair (Δc_additional)")
    print("=" * 110)
    header = (
        f"{'pair':36s} "
        f"{'M1prim':>7} {'M2RT':>7} {'M3Q2':>7} {'M4fram':>7} {'M5loop':>7} "
        f"{'axiSum':>7} {'allSum':>7}"
    )
    print(header)
    print("-" * 110)
    for p in pairs:
        ms = {m['name']: m for m in p['v5p22_mechanisms']}
        name = f"{p['name_i']} + {p['name_j']}"
        print(
            f"{name:36s} "
            f"{ms['primitive_vassiliev']['delta_c_additional']:>7} "
            f"{ms['rt_color_decomposition']['delta_c_additional']:>7} "
            f"{ms['quadrupole_Q2']['delta_c_additional']:>7} "
            f"{ms['framing_anomaly_differential']['delta_c_additional']:>7} "
            f"{ms['one_loop_conformal_weight']['delta_c_additional']:>7} "
            f"{p['v5p22_axiomatic_sum']:>7} "
            f"{p['v5p22_all_sum']:>7}"
        )


def print_combined_verdict(pairs: list) -> None:
    print()
    print("=" * 110)
    print("V5.22 -- Combined Δc_min (V5.21 + V5.22)")
    print("=" * 110)
    header = (
        f"{'pair':36s} "
        f"{'v21':>5} {'+axi':>5} {'v22h':>5} {'+all':>5} {'v22M':>5}  binding"
    )
    print(header)
    print("-" * 110)
    for p in pairs:
        name = f"{p['name_i']} + {p['name_j']}"
        print(
            f"{name:36s} "
            f"{p['v5p21_total']:>5} "
            f"{p['v5p22_axiomatic_sum']:>5} "
            f"{p['delta_c_min_v5p22_honest']:>5} "
            f"{p['v5p22_all_sum']:>5} "
            f"{p['delta_c_min_v5p22_max']:>5}  "
            f"{p['binding_mechanism']}"
        )


def make_figure(pairs: list, out_path: Path) -> None:
    """Stacked bar chart of V5.21 baseline + V5.22 additions."""
    n = len(pairs)
    x = np.arange(n)

    v21 = np.array([p["v5p21_total"] for p in pairs], dtype=float)
    add_h = np.array([p["v5p22_axiomatic_sum"] for p in pairs], dtype=float)
    add_max = np.array([p["v5p22_all_sum"] for p in pairs], dtype=float)
    total_honest = np.array([p["delta_c_min_v5p22_honest"] for p in pairs],
                            dtype=float)
    total_max = np.array([p["delta_c_min_v5p22_max"] for p in pairs],
                         dtype=float)

    fig, ax = plt.subplots(figsize=(15, 7))

    # V5.21 baseline (Vassiliev or Lk binding)
    ax.bar(x, v21, 0.65, label="V5.21 baseline (Δc_min^V5.21)",
           color="#4c72b0")
    # V5.22 axiomatic addition (honest)
    ax.bar(x, add_h, 0.65, bottom=v21,
           label="V5.22 axiomatic addition (honest)", color="#dd8452")
    # V5.22 untrusted addition
    untrusted = add_max - add_h
    ax.bar(x, untrusted, 0.65, bottom=v21 + add_h,
           label="V5.22 non-axiomatic addition (upper bound, UNTRUSTED)",
           color="#8c8c8c", alpha=0.5, hatch="//")

    # Thresholds
    ax.axhline(22, color="orange", linestyle=":", lw=1.5,
               label="V5.21 EDGRB threshold (Δc ≥ 22)")
    ax.axhline(FERMI_LINE_SEARCH_THRESHOLD, color="red", linestyle="--", lw=2,
               label=f"Fermi line search threshold (Δc ≥ {FERMI_LINE_SEARCH_THRESHOLD})")

    labels = [f"{p['name_i']}\n+\n{p['name_j']}" for p in pairs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_ylabel(r"$\Delta c$ (crossings)")
    verdict = ("V5.20 lines REDEEMED" if DELTA_C_MIN_V5P22_HONEST
               >= FERMI_LINE_SEARCH_THRESHOLD else "F3 FALSIFICATION OPEN")
    ax.set_title(
        "V5.22 -- Tentative derivation of Δc_min ≥ 30 for amphichiral DM pearls\n"
        f"V5.21 = {DELTA_C_MIN_OVERALL}  |  V5.22 honest = {DELTA_C_MIN_V5P22_HONEST}  |  "
        f"V5.22 upper bound = {DELTA_C_MIN_V5P22_UPPER}  ⇒  {verdict}"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(50, float(total_max.max()) + 5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[fig] Saved → {out_path}")


def main() -> int:
    print("=" * 110)
    print("V5.22 -- Tentative derivation of Δc_min ≥ 30 for amphichiral DM pearls")
    print("=" * 110)
    print()
    print(f"V5.21 baseline (theory/dc_selection_rule.py)  : Δc_min = {DELTA_C_MIN_OVERALL}")
    print(f"Fermi line search threshold (commit 06ad81f)  : Δc_min ≥ "
          f"{FERMI_LINE_SEARCH_THRESHOLD}")
    print(f"Tension to bridge                              : "
          f"+{FERMI_LINE_SEARCH_THRESHOLD - DELTA_C_MIN_OVERALL} crossings\n")

    print("V5 amphichirals (V5 §13.7) :")
    for a in AMPHICHIRAL_DATA:
        print(f"  {a.name:14s}  m = {a.mass_GeV:6.2f} GeV  c = {a.c:>2}  "
              f"Q_phys = {a.Q_phys}")

    # Detailed mechanism breakdown for one canonical pair
    print()
    print("=" * 110)
    print("Detailed analysis : 6_3+6_3 self-link (lightest amphichiral pair)")
    print("=" * 110)
    a = AMPHICHIRAL_DATA[0]
    ms = evaluate_all_mechanisms(a, a)
    for m in ms:
        flag = "AXIOM" if m.axiomatic else "non-ax"
        print(f"  [{flag}]  {m.name:32s}  Δc_additional = {m.delta_c_additional:>3}")
        print(f"             rationale : {m.rationale}")

    pairs = all_amphichiral_pairs_v5p22(honest=True)
    print_mechanism_per_pair(pairs)
    print_combined_verdict(pairs)

    # Overall verdict
    print()
    print("=" * 110)
    print("VERDICT V5.22")
    print("=" * 110)
    print(f"  Δc_min^V5.21        = {DELTA_C_MIN_OVERALL}")
    print(f"  Δc_min^V5.22 honest = {DELTA_C_MIN_V5P22_HONEST}")
    print(f"  Δc_min^V5.22 upper  = {DELTA_C_MIN_V5P22_UPPER}"
          "  (UNTRUSTED, all 5 mechanisms)")
    print(f"  Threshold (Fermi)   = {FERMI_LINE_SEARCH_THRESHOLD}\n")

    if DELTA_C_MIN_V5P22_HONEST >= FERMI_LINE_SEARCH_THRESHOLD:
        print(f"  *** V5.22 REDEEMS V5.20 LINES ***")
        print(f"  Honest Δc_min = {DELTA_C_MIN_V5P22_HONEST} ≥ {FERMI_LINE_SEARCH_THRESHOLD}")
        print(f"  ⇒ Fermi-LAT line search compatible.")
        print(f"  ⇒ V5.20 Class E six-line signature stays predictive at CTA depth.")
    else:
        print(f"  *** V5.22 INSUFFICIENT ***")
        print(f"  Honest Δc_min = {DELTA_C_MIN_V5P22_HONEST} <"
              f" {FERMI_LINE_SEARCH_THRESHOLD}")
        print(f"  ⇒ F3 falsification (Fermi line search) REMAINS OPEN.")
        print(f"  ⇒ V5.20 Class E line predictions excluded by Fermi-LAT 2015.")
        print(f"  ⇒ Pivot recommended : cascade smear (continuous spectrum) or"
              f" abandon Class E.")
        print()
        print(f"  Honest itemisation of V5.22 mechanisms :")
        print(f"    M1 primitive Vassiliev (axiomatic)   : +2")
        print(f"    M2 RT color decomposition            : non-axiomatic")
        print(f"    M3 quadrupole Q^(2)                  : not in V5, ≡0")
        print(f"    M4 framing anomaly differential      : Δh<1/2 absorbed")
        print(f"    M5 1-loop conformal weight           : order-of-magnitude")
        print(f"  ⇒ ONLY +2 honestly available, falling 4 crossings short.")

    print()
    print(f"output : {OUTPUT_PNG}")
    make_figure(pairs, OUTPUT_PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
