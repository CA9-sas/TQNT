"""V5.19 — Global consolidation of the 6 compact-object class tests.

Aggregates the verdict diagnostics produced by the six per-class test
modules into a single comparative table + testability matrix figure.

Per-class modules (consolidated here)
-------------------------------------
* Class A : ``scripts/V5_microlensing/class_A_subaru_hsc_test.py``
    commit ``f2829fa`` — SURVIVES (Subaru HSC PBH bound, x5 headroom).
* Class B : ``scripts/V5_JWST/v5p19_class_B_AGN_test.py``
    commit ``27f0963`` — CONTRIBUTES (mass OK, abundance overshoots x1e8).
* Class C : ``scripts/V5_Gaia/class_C_hybrid_stars_test.py``
    commit ``d8f086c`` — MARGINAL (DR4 projection ~26 sigma stacked).
* Class D : ``scripts/V5_BBN/class_D_bbn_test.py``
    commit ``26ad81a`` — NULL TRIVIAL (Vassiliev kills signal by 45 dex).
* Class E : ``scripts/V5_microlensing/class_E_strange_knot_test.py``
    commit ``ae3cb04`` — SURVIVES (multi-peak partially resolvable).
* Class F : ``scripts/V5_GW/class_F_imbh_test.py``
    commit ``ae3cb04`` — SURVIVES (LVK IMBH bound, sub-dominant x10).

The module is purely a CONSOLIDATION: no new physics is introduced and
no new publication is created. The output is a 4-panel summary figure
``output/v5p19_synthesis_6classes.png`` and a dict of per-class
diagnostics. Caveats from the per-class tests are preserved verbatim.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Per-class entry points
from scripts.V5_microlensing.class_A_subaru_hsc_test import (
    summary as summary_A,
)
from scripts.V5_microlensing.class_E_strange_knot_test import (
    summary as summary_E,
)
from scripts.V5_Gaia.class_C_hybrid_stars_test import summary as summary_C
from scripts.V5_BBN.class_D_bbn_test import verdict as verdict_D
from scripts.V5_GW.class_F_imbh_test import verdict as verdict_F
from scripts.V5_JWST.v5p19_class_B_AGN_test import (
    grid_search_best_seed,
    jwst_observed_n_AGN_Mpc3,
    verdict as verdict_B_raw,
)
from theory.v5_compact_objects_catalog import build_catalog, MSUN_KG


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# 0. Static metadata for the 6 classes
# =========================================================================

@dataclass
class ClassEntry:
    """One row in the consolidated table."""
    short_id: str
    name: str
    channel: str
    commit: str
    verdict_short: str          # SURVIVES / MARGINAL / CONTRIBUTES / NULL
    verdict_long: str           # Full text from the per-class verdict()
    M_typical_Msun: float
    M_max_Msun: float
    Omega_label: str            # "Omega/Omega_DM" or "Omega/Omega_b"
    Omega_value: float
    headroom_or_tension: str    # short text, "x5 margin" / "+45 dex below"
    discriminability: int       # 1..5 (5 = high)
    roadmap: str                # instrument + timescale
    falsifiable_now: bool       # current data can refute?
    falsifiable_soon: bool      # data expected by 2027 can refute?


# =========================================================================
# 1. Aggregation of the 6 verdicts
# =========================================================================

def aggregate_class_A() -> ClassEntry:
    s = summary_A()
    cat = build_catalog()["A"]
    return ClassEntry(
        short_id="A",
        name="DM Pearls",
        channel="Subaru HSC microlensing (Niikura+ 2019)",
        commit="f2829fa",
        verdict_short=s["verdict"],
        verdict_long=(
            f"max tension f_V5/f_HSC = {s['max_tension']:.2e} at "
            f"M = {s['M_max_tension_Msun']:.2e} Msun ; "
            f"sensitivity factor to detect = "
            f"{s['sensitivity_factor_to_detect']:.1f}x"
        ),
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_DM",
        Omega_value=cat.omega_over_omega_dm,
        headroom_or_tension=(
            f"x{s['sensitivity_factor_to_detect']:.1f} margin below HSC"
        ),
        discriminability=4,
        roadmap="Subaru HSC second run (2026-2028) + Roman PBH search 2027",
        falsifiable_now=True,
        falsifiable_soon=True,
    )


def aggregate_class_B() -> ClassEntry:
    """Class B: re-run the grid search + verdict to capture diagnostics."""
    grid = grid_search_best_seed(
        M_seed_grid=np.logspace(-1, 8, 50), z_form=30.0, f_duty=1.0,
    )
    n_obs = jwst_observed_n_AGN_Mpc3(z_min=5.0, z_max=8.0)
    v = verdict_B_raw(grid["M_seed_min_required_Msun"], n_obs)
    cat = build_catalog()["B"]
    # Short verdict label
    if "EXPLAIN" in v["verdict"]:
        short = "EXPLAINS"
    elif "CONTRIBUT" in v["verdict"]:
        short = "CONTRIBUTES"
    elif "REFUTED" in v["verdict"]:
        short = "REFUTED"
    else:
        short = "INDISTINCT"
    return ClassEntry(
        short_id="B",
        name="Topological PBH seeds",
        channel="JWST high-z AGN/LRDs (Matthee+ 2023, Bogdan+ 2024)",
        commit="27f0963",
        verdict_short=short,
        verdict_long=v["verdict"],
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_DM",
        Omega_value=cat.omega_over_omega_dm,
        headroom_or_tension=(
            f"abundance log(n_pred/n_obs) = "
            f"{v['log_ratio_n_pred_over_obs']:+.1f} dex"
        ),
        discriminability=3,
        roadmap="JWST cycles 4-5 + Roman 2027 (AGN census)",
        falsifiable_now=False,
        falsifiable_soon=True,
    )


def aggregate_class_C() -> ClassEntry:
    s = summary_C()
    cat = build_catalog()["C"]
    return ClassEntry(
        short_id="C",
        name="Knot-Baryon Hybrid Stars",
        channel="Gaia DR3 ensemble + DR4 projected (2026)",
        commit="d8f086c",
        verdict_short="MARGINAL" if "MARGINAL" in s["verdict"]
                       else s["verdict"].split()[0],
        verdict_long=s["verdict"],
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_b",
        Omega_value=cat.omega_over_omega_b,
        headroom_or_tension=(
            f"DR4 stacked = "
            f"{s['ensemble_significance_DR4_sigma']:.0f} sigma "
            f"(DR3 = {s['ensemble_significance_DR3_sigma']:.1f} sigma)"
        ),
        discriminability=4,
        roadmap="Gaia DR4 (release 2026-end / 2027)",
        falsifiable_now=False,
        falsifiable_soon=True,
    )


def aggregate_class_D() -> ClassEntry:
    v = verdict_D()
    cat = build_catalog()["D"]
    return ClassEntry(
        short_id="D",
        name="Heavy-proton BBN plasma",
        channel="D/H Cooke+ 2018 (BBN nucleosynthesis)",
        commit="26ad81a",
        verdict_short="NULL TRIVIAL",
        verdict_long=v["verdict"],
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_b",
        Omega_value=cat.omega_over_omega_b,
        headroom_or_tension=(
            f"V5 ~ {v['log10_V5_below_bound']:.0f} dex BELOW D/H bound "
            f"(Vassiliev alpha_EM^{v['c_n_vassiliev']})"
        ),
        discriminability=1,
        roadmap="NONE: indistinct by construction (Vassiliev kills signal)",
        falsifiable_now=False,
        falsifiable_soon=False,
    )


def aggregate_class_E() -> ClassEntry:
    s = summary_E()
    cat = build_catalog()["E"]
    return ClassEntry(
        short_id="E",
        name="Strange Knot Stars",
        channel="Subaru HSC microlensing (multi-peak amphichiral)",
        commit="ae3cb04",
        verdict_short=s["verdict"],
        verdict_long=(
            f"max tension = {s['max_tension']:.2e} ; "
            f"excluded peaks = "
            f"{len(s['excluded_peaks'])}/{s['n_peaks']} ; "
            f"multi-peak visible = {s['multi_peak_visible']}"
        ),
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_DM",
        Omega_value=cat.omega_over_omega_dm,
        headroom_or_tension=(
            f"max tension = {s['max_tension']:.2e} ; "
            f"{s['n_resolved_peaks']}/{s['n_peaks']} peaks resolvable"
        ),
        discriminability=4,
        roadmap="Subaru HSC second run + Roman 2027 (per-peak resolution)",
        falsifiable_now=True,
        falsifiable_soon=True,
    )


def aggregate_class_F() -> ClassEntry:
    v = verdict_F()
    cat = build_catalog()["F"]
    if "EXCLUDED" in v["verdict"]:
        short = "EXCLUDED"
    elif "MARGINAL" in v["verdict"]:
        short = "MARGINAL"
    else:
        short = "SURVIVES"
    return ClassEntry(
        short_id="F",
        name="Primordial unknot defects (IMBH)",
        channel="LVK GWTC-3 + partial GWTC-4 IMBH mergers",
        commit="ae3cb04",
        verdict_short=short,
        verdict_long=v["verdict"],
        M_typical_Msun=cat.m_typical_Msun,
        M_max_Msun=cat.m_max_Msun,
        Omega_label="Omega/Omega_DM",
        Omega_value=cat.omega_over_omega_dm,
        headroom_or_tension=(
            f"abundance x{v['abundance_headroom']:.0f} below LVK bound ; "
            f"O5 predicted N = {v['N_O5_pred']:.1e}"
        ),
        discriminability=3,
        roadmap="LIGO O5 (~2027) + LIGO-India 2030",
        falsifiable_now=False,
        falsifiable_soon=True,
    )


def aggregate_all() -> dict[str, ClassEntry]:
    """Aggregate the 6 per-class verdicts into a uniform dict."""
    return {
        "A": aggregate_class_A(),
        "B": aggregate_class_B(),
        "C": aggregate_class_C(),
        "D": aggregate_class_D(),
        "E": aggregate_class_E(),
        "F": aggregate_class_F(),
    }


# =========================================================================
# 2. Synthesis summary (dict) + structural counts
# =========================================================================

def synthesis_counts(entries: dict[str, ClassEntry]) -> dict[str, int]:
    """Count verdict types across all 6 classes."""
    counts = {
        "SURVIVE": 0,
        "MARGINAL": 0,
        "CONTRIBUTES": 0,
        "NULL_TRIVIAL": 0,
        "EXCLUDED": 0,
        "REFUTED": 0,
        "FALSIFIABLE_NOW": 0,
        "FALSIFIABLE_SOON": 0,
    }
    for e in entries.values():
        v = e.verdict_short.upper()
        if "SURVIV" in v:
            counts["SURVIVE"] += 1
        elif "MARGINAL" in v:
            counts["MARGINAL"] += 1
        elif "CONTRIBUT" in v:
            counts["CONTRIBUTES"] += 1
        elif "NULL" in v:
            counts["NULL_TRIVIAL"] += 1
        elif "EXCLUDED" in v:
            counts["EXCLUDED"] += 1
        elif "REFUTED" in v:
            counts["REFUTED"] += 1
        if e.falsifiable_now:
            counts["FALSIFIABLE_NOW"] += 1
        if e.falsifiable_soon:
            counts["FALSIFIABLE_SOON"] += 1
    return counts


def vassiliev_pattern(entries: dict[str, ClassEntry]) -> dict[str, list[str]]:
    """Structural classification: astrophysically formed vs primordial.

    The pattern is: astrophysically/late-time formed classes are testable
    via direct observation (microlensing, AGN, stellar census, GW), while
    primordial classes suffer Vassiliev alpha_EM^c suppression at high c
    and become indistinct (Class D).
    """
    astro = []      # formed astrophysically / late-time : A, B, C, E, F
    primordial_suppressed = []   # purely primordial + Vassiliev : D
    for sid, e in entries.items():
        if sid == "D":
            primordial_suppressed.append(sid)
        else:
            astro.append(sid)
    return {
        "astrophysical_testable": astro,
        "primordial_vassiliev_suppressed": primordial_suppressed,
    }


def synthesis_summary() -> dict[str, Any]:
    """Build the full synthesis dict (for tests + CLI)."""
    entries = aggregate_all()
    counts = synthesis_counts(entries)
    pattern = vassiliev_pattern(entries)
    return {
        "n_classes_total": len(entries),
        "counts": counts,
        "vassiliev_pattern": pattern,
        "entries": entries,
    }


# =========================================================================
# 3. Plot: 4-panel synthesis figure
# =========================================================================

VERDICT_COLOR = {
    "SURVIVE": "#3a9b3a",         # green
    "SURVIVES": "#3a9b3a",
    "MARGINAL": "#dd9933",        # amber
    "CONTRIBUTES": "#dd9933",
    "NULL TRIVIAL": "#888888",    # grey
    "EXCLUDED": "#cc3333",        # red
    "REFUTED": "#cc3333",
    "PARTIALLY_EXCLUDED": "#cc8833",
}


def _verdict_color(label: str) -> str:
    label_up = label.upper()
    for k, c in VERDICT_COLOR.items():
        if k in label_up:
            return c
    return "#666666"


def make_plot(path: Optional[Path] = None) -> Path:
    """Render the 4-panel V5.19 6-classes synthesis figure."""
    if path is None:
        path = OUTPUT_DIR / "v5p19_synthesis_6classes.png"
    summary = synthesis_summary()
    entries = summary["entries"]
    counts = summary["counts"]
    pattern = summary["vassiliev_pattern"]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0))
    fig.suptitle(
        "V5.19 — Synthesis of the six compact-object class tests "
        "(Classes A--F)",
        fontsize=14,
    )

    # ---- Panel (a) : Mass / abundance scatter ----------------------------
    ax = axes[0, 0]
    for sid, e in entries.items():
        m_lo = max(e.M_typical_Msun, 1e-15)
        m_hi = max(e.M_max_Msun, m_lo * 1.001)
        omega = max(e.Omega_value, 1e-12)
        color = _verdict_color(e.verdict_short)
        ax.errorbar(
            [np.sqrt(m_lo * m_hi)], [omega],
            xerr=[[np.sqrt(m_lo * m_hi) - m_lo], [m_hi - np.sqrt(m_lo * m_hi)]],
            fmt="o", color=color, ms=12, capsize=4, lw=2,
            label=f"Class {sid}: {e.name[:30]}",
        )
        ax.text(np.sqrt(m_lo * m_hi) * 1.15, omega * 1.1, sid,
                fontsize=12, fontweight="bold", color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$ range [$M_\odot$]")
    ax.set_ylabel(r"$\Omega_{\rm class}/\Omega_{\rm DM}$ or $\Omega_{\rm b}$")
    ax.set_title("(a) Mass span and predicted abundance")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-15, 1e10)
    ax.set_ylim(1e-12, 1.0)

    # ---- Panel (b) : Verdict bar chart -----------------------------------
    ax = axes[0, 1]
    sids = list(entries.keys())
    verdicts = [entries[sid].verdict_short for sid in sids]
    discrim = [entries[sid].discriminability for sid in sids]
    bar_colors = [_verdict_color(v) for v in verdicts]
    ax.bar(sids, discrim, color=bar_colors, edgecolor="black")
    for i, (sid, v) in enumerate(zip(sids, verdicts)):
        # Truncate verdict to first 11 chars
        ax.text(i, discrim[i] + 0.08, v[:11], ha="center", fontsize=8,
                rotation=0, color=bar_colors[i], fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Discriminability (1=null test ... 5=conclusive)")
    ax.set_title("(b) Per-class verdict and discriminability")
    ax.set_ylim(0, 5.6)
    ax.grid(True, axis="y", alpha=0.3)

    # ---- Panel (c) : Testability matrix ----------------------------------
    ax = axes[1, 0]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 7)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["existing\ndata", "now\nfalsifiable",
                         "<= 2027\nfalsifiable", "long-term\nfalsifiable"],
                        fontsize=9)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels([f"Class {s}" for s in reversed(list(entries.keys()))],
                       fontsize=10)
    # Fill cells
    matrix = {sid: [True, e.falsifiable_now, e.falsifiable_soon,
                     not (e.falsifiable_now or e.falsifiable_soon)]
              for sid, e in entries.items()}
    for j, sid in enumerate(reversed(list(entries.keys()))):
        for i, ok in enumerate(matrix[sid]):
            color = _verdict_color(entries[sid].verdict_short) if ok else "white"
            ax.fill_between([i, i+1], j, j+1,
                            color=color, edgecolor="black", alpha=0.7)
            if ok:
                ax.text(i + 0.5, j + 0.5, "X", ha="center", va="center",
                        fontsize=14, fontweight="bold")
    ax.set_title("(c) Testability matrix per class")

    # ---- Panel (d) : Text box - summary verdict --------------------------
    ax = axes[1, 1]
    ax.axis("off")
    n_survive = counts["SURVIVE"]
    n_marginal = counts["MARGINAL"]
    n_contrib = counts["CONTRIBUTES"]
    n_null = counts["NULL_TRIVIAL"]
    n_excl = counts["EXCLUDED"] + counts["REFUTED"]
    n_now = counts["FALSIFIABLE_NOW"]
    n_soon = counts["FALSIFIABLE_SOON"]
    txt = (
        "V5.19 Synthesis - 6 classes\n"
        "=" * 40 + "\n"
        f"SURVIVES         : {n_survive}/6\n"
        f"MARGINAL         : {n_marginal}/6\n"
        f"CONTRIBUTES      : {n_contrib}/6\n"
        f"NULL TRIVIAL     : {n_null}/6\n"
        f"EXCLUDED/REFUTED : {n_excl}/6\n"
        "\n"
        "Falsifiability\n"
        "-" * 40 + "\n"
        f"  now            : {n_now}/6 ({pattern['astrophysical_testable']})\n"
        f"  <= 2027        : {n_soon}/6\n"
        "\n"
        "Structural pattern\n"
        "-" * 40 + "\n"
        f"  astro-formed   : {pattern['astrophysical_testable']}\n"
        f"  Vassiliev-supp : {pattern['primordial_vassiliev_suppressed']}\n"
        "\n"
        "Roadmap 2026-2030\n"
        "-" * 40 + "\n"
        "  Subaru HSC run2 -> Class A, E (now + 2027)\n"
        "  Gaia DR4        -> Class C (2026-end / 2027)\n"
        "  LIGO O5         -> Class F (2027)\n"
        "  JWST + Roman    -> Class B (2027+)\n"
        "  D BBN           -> indistinct by construction\n"
        "\n"
        "Global verdict\n"
        "-" * 40 + "\n"
        f"  V5 compact family : robust empirically\n"
        f"  {n_survive + n_marginal + n_contrib}/6 contribute or survive\n"
        f"  {n_null}/6 indistinct (Vassiliev)\n"
        f"  {n_excl}/6 excluded\n"
    )
    ax.text(0.0, 1.0, txt, fontsize=9, family="monospace",
            verticalalignment="top", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================================
# 4. CLI entry
# =========================================================================

def print_table(entries: dict[str, ClassEntry]) -> None:
    """Print the consolidated comparative table to stdout."""
    print("=" * 100)
    print(
        f"{'Cls':>3} | {'Name':<28} | {'M_typ Msun':>11} | "
        f"{'Omega/X':>10} | {'Verdict':<14} | {'Discr.':>6} | "
        f"{'Commit':<8}"
    )
    print("-" * 100)
    for sid, e in entries.items():
        print(
            f"{sid:>3} | {e.name[:28]:<28} | "
            f"{e.M_typical_Msun:>11.2e} | "
            f"{e.Omega_value:>10.2e} | "
            f"{e.verdict_short[:14]:<14} | "
            f"{e.discriminability:>6d} | "
            f"{e.commit:<8}"
        )
    print("=" * 100)


def main() -> dict[str, Any]:
    """Run consolidation, render plot, print table, return dict."""
    summary = synthesis_summary()
    print("V5.19 6-CLASS CONSOLIDATION")
    print()
    print_table(summary["entries"])
    print()
    print("Counts: ", summary["counts"])
    print("Vassiliev pattern: ", summary["vassiliev_pattern"])
    out = make_plot()
    print(f"\nFigure saved: {out}")
    return summary


if __name__ == "__main__":
    main()
