"""V5.26 PDG validation — derivation of Y_W from A-polynomial structure.

Honest audit of the 6 pistes proposed to extract the weak hypercharge from
the A-polynomial of each knot, evaluated against all 67 PDG particles
catalogued in `docs/TQNT_catalog_V5_diffusion.xlsx`.

CONTEXT
-------
V5.25 (commit b0d5a3a) flagged that `topological_hypercharge` returns plain
SM constants Y_lepton=-1 and Y_quark=+1/3 regardless of the knot. V5.26
attempts to derive Y_W from A-polynomial features.

CONCLUSION (negative, honest)
-----------------------------
Y_W is *not* extractable from any single A-polynomial feature for the
following structural reason:

  Y_PDG of a hadron depends on flavour content (S, C, B'), which carries
  global U(3)_flavour quantum numbers absent from any topological invariant.

V5.26 reproduces the isolated-knot doublet hypercharges (e- and the
photon-like neutral knots) but cannot in principle account for flavoured
mesons / baryons whose Y is set by quark content beyond topology.

Outputs:
  output/v5p26_hypercharge_a_poly.csv
  output/v5p26_hypercharge_a_poly.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.hypercharge_a_polynomial import (
    evaluate_pistes,
    extract_a_poly_features,
    hypercharge_v5p26,
    validate_against_pdg,
)
from theory.a_polynomial import A_POLY_KNOT_NAMES
from theory.jones import JONES_KNOT_NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output"
OUT_CSV = OUT_DIR / "v5p26_hypercharge_a_poly.csv"
OUT_PNG = OUT_DIR / "v5p26_hypercharge_a_poly.png"


def print_features_table() -> list[dict]:
    """Print and return the A-poly / Jones features for all 20 V5 knots."""
    print("=== V5.26 feature extraction on 20 V5 knots ===")
    print(
        f"{'knot':6s} {'src':14s} {'n_roots':>7s} {'jspan':>5s} {'jdet':>5s} "
        f"{'mahler':>7s} {'newton_vert':>11s} {'max_slope':>9s}"
    )
    rows: list[dict] = []
    for k in JONES_KNOT_NAMES:
        f = extract_a_poly_features(k)
        rows.append(f)
        nr = f["n_roots"] if f["n_roots"] is not None else "-"
        print(
            f"{k:6s} {f['feature_source']:14s} {str(nr):>7s} "
            f"{f['jones_span']:>5d} {f['jones_det']:>5.1f} "
            f"{f['mahler_log']:>7.3f} {f['newton_n_vertices']:>11d} "
            f"{f['max_slope']:>9.2f}"
        )
    return rows


def print_pistes_evaluation() -> None:
    print("\n=== V5.26 — Evaluation des 6 pistes theoriques ===")
    for p in evaluate_pistes():
        print(f"\n[{p.name}] {p.description}")
        print(f"  verdict : {p.verdict}")
        print(f"  notes   : {p.notes}")


def write_csv(stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "config", "sector", "T3_PDG", "Y_PDG", "Q_PDG",
            "Y_V5p26", "residual", "exact_match", "within_1/6_grid",
        ])
        for r in stats["per_particle"]:
            w.writerow([
                r["name"], r["config"], r["sector"],
                f"{r['t3_pdg']:+g}", f"{r['y_pdg']:+g}", f"{r['q_pdg']:+g}",
                f"{r['y_v5p26']:+.4f}", f"{r['residual']:+.4f}",
                int(r["exact"]), int(r["within_1_6"]),
            ])


def plot_summary(stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = stats["per_particle"]
    n = stats["n"]
    n_exact = stats["n_exact"]
    n_within = stats["n_within_1_6"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left : Y_V5.26 vs Y_PDG scatter, coloured by sector
    ax = axes[0, 0]
    sector_colours = {
        "lepton_L": "#1f77b4",
        "boson": "#9467bd",
        "quark_R": "#8c564b",
        "meson": "#2ca02c",
        "baryon": "#ff7f0e",
        "exotic": "#d62728",
    }
    for s, c in sector_colours.items():
        xs = [r["y_pdg"] for r in rows if r["sector"] == s]
        ys = [r["y_v5p26"] for r in rows if r["sector"] == s]
        ax.scatter(xs, ys, c=c, label=s, s=60, edgecolor="black", linewidth=0.4)
    ymin = min(min(r["y_pdg"] for r in rows), min(r["y_v5p26"] for r in rows)) - 0.5
    ymax = max(max(r["y_pdg"] for r in rows), max(r["y_v5p26"] for r in rows)) + 0.5
    ax.plot([ymin, ymax], [ymin, ymax], "k--", lw=0.7, alpha=0.6, label="y=x ideal")
    ax.set_xlabel("Y_PDG (Q = T_3 + Y/2)")
    ax.set_ylabel("Y_V5.26 (sum n_roots-classifier)")
    ax.set_title(
        f"V5.26 hypercharge — {n_exact}/{n} exact ({100*n_exact/n:.1f}%)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Top-right : per-sector bar chart
    ax = axes[0, 1]
    sectors = list(stats["per_sector"].keys())
    pct = [100 * stats["per_sector"][s]["exact"] / stats["per_sector"][s]["n"]
           for s in sectors]
    counts = [stats["per_sector"][s]["n"] for s in sectors]
    colours = [sector_colours.get(s, "#777") for s in sectors]
    bars = ax.bar(sectors, pct, color=colours, edgecolor="black", linewidth=0.5)
    for bar, pc, ct in zip(bars, pct, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0,
                f"{pc:.0f}%\n(n={ct})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% Y exact match")
    ax.set_title("V5.26 per-sector accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left : feature scatter (n_roots / jones_span proxy vs Y_PDG)
    ax = axes[1, 0]
    feats20 = {k: extract_a_poly_features(k) for k in JONES_KNOT_NAMES}
    # Aggregate per-particle: average jones_span of constituent knots
    import re
    pattern = re.compile(r"\d+_\d+")
    avg_spans = []
    y_pdgs = []
    secs = []
    for r in rows:
        knots = pattern.findall(r["config"])
        spans = [feats20[k]["jones_span"] for k in knots if k in feats20]
        if spans:
            avg_spans.append(sum(spans) / len(spans))
            y_pdgs.append(r["y_pdg"])
            secs.append(r["sector"])
    for s, c in sector_colours.items():
        xs = [avg_spans[i] for i in range(len(secs)) if secs[i] == s]
        ys = [y_pdgs[i] for i in range(len(secs)) if secs[i] == s]
        ax.scatter(xs, ys, c=c, label=s, s=40, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean Jones span of constituent knots")
    ax.set_ylabel("Y_PDG")
    ax.set_title("No clean correlation: jones structure vs Y_PDG\n"
                 "(flavour Y is not in topology)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right : pistes verdict summary
    ax = axes[1, 1]
    ax.axis("off")
    pistes = evaluate_pistes()
    table_data = [["Piste", "Verdict", "Mecanisme"]]
    for p in pistes:
        short = p.description if len(p.description) < 45 else p.description[:42] + "..."
        table_data.append([p.name, p.verdict, short])
    table = ax.table(cellText=table_data, loc="center", cellLoc="left",
                     colWidths=[0.18, 0.18, 0.64])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor("#cccccc")
                cell.set_text_props(weight="bold")
            elif j == 1:
                v = table_data[i][1]
                cell.set_facecolor("#c8e6c9" if v == "retenue" else
                                   "#ffe0b2" if v == "partielle" else
                                   "#ffcdd2")
    ax.set_title("V5.26 — 6 pistes evaluated")

    fig.suptitle(
        f"V5.26 weak hypercharge from A-polynomial structure — {n} PDG particles\n"
        f"Exact: {n_exact}/{n} ({100*n_exact/n:.1f}%)  |  "
        f"Within 1/6 grid: {n_within}/{n} ({100*n_within/n:.1f}%)  |  "
        f"Honest verdict: TOPOLOGICAL Y cannot encode flavour S/C/B'",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def main() -> dict:
    print_features_table()
    print_pistes_evaluation()

    stats = validate_against_pdg()
    write_csv(stats)
    plot_summary(stats)

    print("\n=== V5.26 final statistics ===")
    print(f"Total particles      : {stats['n']}")
    print(f"Y exact match        : {stats['n_exact']}/{stats['n']} "
          f"({100*stats['n_exact']/stats['n']:.1f}%)")
    print(f"Y within 1/6 grid    : {stats['n_within_1_6']}/{stats['n']} "
          f"({100*stats['n_within_1_6']/stats['n']:.1f}%)")
    print("\nPer sector :")
    for s, d in stats["per_sector"].items():
        pct = 100 * d["exact"] / d["n"]
        print(f"  {s:10s}  exact = {d['exact']}/{d['n']}  ({pct:.0f}%)")

    print("\nVerdict V5.26 :")
    print("  - n_roots(A_K) discrimines correctly torus vs hyperbolic (5/5 knots OK)")
    print("  - Single-knot leptons (electron = 3_1) : Y match exact")
    print("  - Composite hadrons : flavour-dependent Y NOT extractable from topology")
    print("  - HONEST CONCLUSION: derivation PARTIAL, not complete.")
    print(f"\nOutputs : {OUT_CSV}\n          {OUT_PNG}")
    return stats


if __name__ == "__main__":
    main()
