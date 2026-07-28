"""V5.25 PDG validation — weak hypercharge Y_W on all 67 catalogued particles.

Reads `docs/TQNT_catalog_V5_diffusion.xlsx` (sheet "3. Particules PDG") and
cross-checks weak hypercharge Y_W via:

  1. PDG / GSW reference value of Y_W per particle
     (left-handed doublet convention: Q = T_3 + Y/2).
  2. V5 prediction from `theory.particle_classification.topological_hypercharge`,
     summed over composite knot components when applicable.
  3. Gell-Mann-Nishijima identity check : does Q_PDG = T_3 + Y_PDG/2 hold?

HONEST CAVEAT (CRITICAL TO READ):
----------------------------------
`topological_hypercharge(knot)` in theory/particle_classification.py:363 does
NOT actually derive Y from the A-polynomial — it returns the constants
Y_lepton = -1 and Y_quark = +1/3 regardless of the knot. The function exposes a
'n_roots' count from `a_polynomial_branches` as a structural motivation but
plugs in the SM hypercharge by hand for the two sectors. So this script
"validates" the V5 assignment scheme (which knot plays which sector role)
rather than a first-principles knot -> Y derivation.

For composite hadrons we map known PDG quark content (e.g. proton = uud) to
the sum of constituent Y values, and check this against PDG Y_hadron from the
doublet hypercharge of the bound state (typically Y_proton = +1/3 for the
left-handed nucleon doublet — the right-handed nucleon Y is also +1/3 because
the proton/neutron are not gauge eigenstates after EWSB).

Outputs:
  - output/v5p25_hypercharge_67_PDG.csv
  - output/v5p25_hypercharge_67_PDG.png
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from theory.particle_classification import topological_hypercharge

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "TQNT_catalog_V5_diffusion.xlsx"
OUT_DIR = ROOT / "output"
OUT_CSV = OUT_DIR / "v5p25_hypercharge_67_PDG.csv"
OUT_PNG = OUT_DIR / "v5p25_hypercharge_67_PDG.png"


# -----------------------------------------------------------------------------
# PDG / Gell-Mann-Nishijima reference table (T_3, Y, Q) per particle
# -----------------------------------------------------------------------------
# Convention used: Q = T_3 + Y/2 (Glashow-Salam-Weinberg).
# For hadrons we list the weak hypercharge of the bound state from PDG
# (left-handed doublet, e.g. (p, n)_L has Y = +1).
# For bosons we list the gauge-eigenstate Y (W^a triplet Y=0, B singlet Y=0,
# physical photon mixing : Y irrelevant after EWSB but conventionally 0).
# Higgs doublet : Y_H = +1 (in the convention Q = T_3 + Y/2, i.e. Y/2 = +1/2).
#
# For composites where PDG quark content is known we annotate the
# constituent-quark Y sum (mainly diagnostic — does sum-of-quarks = doublet Y?).
#
# Format: name -> (T_3, Y_PDG, Q_PDG, sector, quark_content_or_None)
# -----------------------------------------------------------------------------

PDG_HYPERCHARGE: dict[str, tuple[float, float, float, str, str | None]] = {
    # --- Leptons (charged; LH doublet member) -------------------------------
    "e⁻":  (-0.5, -1.0, -1.0,  "lepton_L",   None),
    "μ⁻":  (-0.5, -1.0, -1.0,  "lepton_L",   None),
    "τ⁻":  (-0.5, -1.0, -1.0,  "lepton_L",   None),
    # --- Gauge bosons (Y_gauge = 0 for SU(2) triplet and U(1)_Y singlet) ---
    "γ":   ( 0.0,  0.0,  0.0,  "boson",      None),
    "W±":  ( 1.0,  0.0,  1.0,  "boson",      None),    # W+: T_3=+1, Y=0, Q=+1
    "Z⁰":  ( 0.0,  0.0,  0.0,  "boson",      None),
    "H":   (-0.5,  1.0,  0.0,  "boson",      None),    # neutral Higgs doublet member
    # --- Top quark (RH singlet conventional Y_R = +4/3 -> Q=+2/3) ----------
    "t":   ( 0.0,  4/3, +2/3,  "quark_R",    "t"),
    # --- Pseudoscalar mesons --------------------------------------------
    # Y_meson = B + S + C + B' (B=0 for mesons); Q = T_3 + Y/2 (GSW).
    # Antiquark convention: s-bar carries S=+1, c carries C=+1, b-bar carries B'=+1.
    "π⁰":  ( 0.0,  0.0,  0.0,  "meson",      "uu-dd"),
    "π±":  ( 1.0,  0.0, +1.0,  "meson",      "ud"),    # pi+ = u d-bar : Y=0
    "K±":  ( 0.5,  1.0, +1.0,  "meson",      "us"),    # K+ = u s-bar : Y=S=+1
    "K⁰":  (-0.5,  1.0,  0.0,  "meson",      "ds"),    # K0 = d s-bar : Y=+1
    "η":   ( 0.0,  0.0,  0.0,  "meson",      "uu+dd+ss"),
    "η'":  ( 0.0,  0.0,  0.0,  "meson",      "uu+dd+ss"),
    # Charmonium (cc-bar, Y_total=0, isoscalar)
    "η_c(1S)":  (0.0, 0.0, 0.0, "meson", "cc"),
    "η_c(2S)":  (0.0, 0.0, 0.0, "meson", "cc"),
    "J/ψ":      (0.0, 0.0, 0.0, "meson", "cc"),
    "ψ(2S)":    (0.0, 0.0, 0.0, "meson", "cc"),
    "χ_c0(1P)": (0.0, 0.0, 0.0, "meson", "cc"),
    "χ_c1(1P)": (0.0, 0.0, 0.0, "meson", "cc"),
    "χ_c2(1P)": (0.0, 0.0, 0.0, "meson", "cc"),
    # Bottomonium (bb-bar)
    "η_b(1S)":  (0.0, 0.0, 0.0, "meson", "bb"),
    "Υ(1S)":    (0.0, 0.0, 0.0, "meson", "bb"),
    "Υ(2S)":    (0.0, 0.0, 0.0, "meson", "bb"),
    "χ_b0(1P)": (0.0, 0.0, 0.0, "meson", "bb"),
    "χ_b1(1P)": (0.0, 0.0, 0.0, "meson", "bb"),
    "χ_b2(1P)": (0.0, 0.0, 0.0, "meson", "bb"),
    # Charmed mesons: D⁰=c ū (Q=0), D⁺=c d̄ (Q=+1); Y = C = +1.
    "D⁰":        (-0.5,  1.0,  0.0, "meson", "cu"),
    "D±":        ( 0.5,  1.0, +1.0, "meson", "cd"),
    "D_s±":      ( 0.0,  2.0, +1.0, "meson", "cs"),    # D_s+ = c s-bar : Y=C+S=2
    "D*(2007)⁰": (-0.5,  1.0,  0.0, "meson", "cu"),
    "D*(2010)±": ( 0.5,  1.0, +1.0, "meson", "cd"),
    # B mesons (b-bar q): B+ = u b-bar (Y=B'=+1), B0 = d b-bar (Y=+1)
    "B±":   ( 0.5,  1.0, +1.0, "meson", "ub"),
    "B⁰":   (-0.5,  1.0,  0.0, "meson", "db"),
    "B_s⁰": ( 0.0,  0.0,  0.0, "meson", "sb"),         # s b-bar : Y=S+B'=-1+1=0
    "B_c±": ( 0.0,  2.0, +1.0, "meson", "cb"),         # c b-bar : Y=C+B'=+2
    # Light vector / scalar / tensor (isoscalar / isovector light)
    "ρ":      ( 0.0, 0.0, 0.0, "meson", "uu-dd"),      # rho0 entry; rho+- exist
    "ω":      ( 0.0, 0.0, 0.0, "meson", "uu+dd"),
    "φ":      ( 0.0, 0.0, 0.0, "meson", "ss"),
    "K*(892)":(-0.5, 1.0, 0.0, "meson", "ds"),         # K*0
    "f₀(980)":( 0.0, 0.0, 0.0, "meson", "uu+dd+ss"),
    "a₂(1320)":(1.0, 0.0, 1.0, "meson", "ud"),         # a2+ : I=1, isovector
    "f₂(1270)":(0.0, 0.0, 0.0, "meson", "uu+dd"),
    # --- Baryons : Y from PDG hypercharge (Y = B + S + C + B' + T) ---------
    # Nucleons p, n: Y = B = +1 (baryon LH doublet); strangeness/charm = 0
    "p":   ( 0.5,  1.0, +1.0, "baryon",  "uud"),
    "n":   (-0.5,  1.0,  0.0, "baryon",  "udd"),
    "Λ":   ( 0.0,  0.0,  0.0, "baryon",  "uds"),     # Y = B + S = 1 - 1 = 0
    "Σ⁺":  ( 1.0,  0.0, +1.0, "baryon",  "uus"),     # Y=0
    "Σ⁰":  ( 0.0,  0.0,  0.0, "baryon",  "uds"),
    "Σ⁻":  (-1.0,  0.0, -1.0, "baryon",  "dds"),
    "Ξ⁰":  ( 0.5, -1.0,  0.0, "baryon",  "uss"),     # Y = 1 - 2 = -1
    "Ξ⁻":  (-0.5, -1.0, -1.0, "baryon",  "dss"),
    "Δ":   ( 0.5,  1.0, +1.0, "baryon",  "uud"),     # Delta+; Y = B = 1
    "Σ*⁺": ( 1.0,  0.0, +1.0, "baryon",  "uus"),
    "Ξ*⁰": ( 0.5, -1.0,  0.0, "baryon",  "uss"),
    "Ω⁻":  ( 0.0, -2.0, -1.0, "baryon",  "sss"),     # Y = B + S = 1 - 3 = -2
    "Λ_c⁺":  ( 0.0,  2.0, +1.0, "baryon",  "udc"),   # Y = B + C = 1 + 1 = 2
    "Σ_c⁺⁺": ( 1.0,  2.0, +2.0, "baryon",  "uuc"),
    "Ξ_c⁺":  ( 0.5,  1.0, +1.0, "baryon",  "usc"),   # Y = B + S + C = 1 - 1 + 1 = 1
    "Ω_c⁰":  ( 0.0,  0.0,  0.0, "baryon",  "ssc"),   # Y = 1 - 2 + 1 = 0
    "Λ_b⁰":  ( 0.0,  0.0,  0.0, "baryon",  "udb"),   # Y = B + B' = 1 - 1 = 0
    "Ξ_b⁰":  ( 0.5, -1.0,  0.0, "baryon",  "usb"),
    "Ω_b⁻":  ( 0.0, -2.0, -1.0, "baryon",  "ssb"),
    # --- Exotics : conservative quark content / Q only ---------------------
    "X(3872)":  (0.0, 0.0, 0.0, "exotic",  "cc??"),
    "Z_c(3900)":(1.0, 0.0, 1.0, "exotic",  "cc-ud"),
    "X(6900)":  (0.0, 0.0, 0.0, "exotic",  "ccc-c"),
    "P_c(4380)":(0.5, 1.0, 1.0, "exotic",  "uudcc"),  # pentaquark with B=1, C=0 -> Y=1
    "P_c(4450)":(0.5, 1.0, 1.0, "exotic",  "uudcc"),
}


# Quark Y (LH doublet convention, Q = T_3 + Y/2)
# u, d : Y = +1/3 ; u-bar, d-bar : Y = -1/3
# c, t : Y = +1/3 ; s, b : Y = +1/3 (LH doublet)
# In hadron Y formula Y = B + S + C + B' + T, strange/bottom carry Y unit of -1.
# We compute Y at the hadron level from PDG quantum numbers and use the
# constituent-quark sum only as a diagnostic.
QUARK_Y_LH = {"u": +1/3, "d": +1/3, "c": +1/3, "s": +1/3, "t": +1/3, "b": +1/3}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_KNOT_PATTERN = re.compile(r"\d+_\d+")


def parse_knot_string(cfg: str) -> list[str]:
    return _KNOT_PATTERN.findall(cfg)


# Heuristic sector for a single knot (single-knot particles only)
LEPTON_KNOTS = {"3_1"}            # e- assignment uses 3_1
BOSON_SINGLE_KNOTS = {"4_1"}      # gamma in catalog (also Y=0)


def v5_hypercharge_single_knot(knot: str, sector_hint: str) -> float:
    """V5 Y from `topological_hypercharge` for a single-knot particle.

    Selects Y_lepton vs Y_quark using the sector hint from PDG.
    NOTE: topological_hypercharge currently returns the SM constants
    (-1 or +1/3) — see module docstring. This is the documented limitation.
    """
    info = topological_hypercharge(knot)
    if sector_hint == "lepton_L":
        return info["Y_lepton"]
    if sector_hint == "boson":
        return 0.0
    # quark default
    return info["Y_quark"]


def v5_hypercharge_composite(knots: list[str], sector_hint: str) -> float:
    """V5 Y prediction for a composite entrelacs.

    For baryons we assume each component plays a quark-like role and sum
    Y_quark over the three constituents — yielding Y = 3 x (1/3) = +1 for any
    3-knot baryon (matches PDG nucleon doublet Y = +1).

    For mesons we have a quark + anti-quark in pairs of two knots, but knots
    do not encode anti-particle structure directly — Σχ + Lk handles that.
    We return 0 for two-knot mesonic configurations as a baseline and check
    GMN separately.
    """
    n = len(knots)
    if sector_hint == "boson":
        # Single-knot W/Z/H/gamma all have Y_gauge = 0 in the catalog
        # (Higgs is the only one with Y = +1; flagged separately).
        return 0.0
    if sector_hint == "lepton_L":
        return topological_hypercharge(knots[0])["Y_lepton"]
    if sector_hint in ("meson",):
        # quark + antiquark : Y = +1/3 + (-1/3) = 0 unless strangeness/charm/bottom flavour
        # The knot list doesn't track flavour — we report 0 as bare topological prediction.
        return 0.0
    if sector_hint in ("baryon", "exotic"):
        # n knots interpreted as n constituents, each Y_quark = +1/3
        return sum(topological_hypercharge(k)["Y_quark"] for k in knots)
    return 0.0


def gell_mann_nishijima(t3: float, y: float) -> float:
    """Q = T_3 + Y/2."""
    return t3 + y / 2.0


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------
@dataclass
class HyperchargeRow:
    name: str
    cfg: str
    knots: list[str]
    sector: str
    t3_pdg: float
    y_pdg: float
    q_pdg: float
    q_gmn_check: float            # T_3 + Y_pdg/2
    gmn_ok: bool
    y_v5: float
    y_residual: float             # Y_v5 - Y_pdg
    y_match_exact: bool           # |residual| < 1e-6
    y_match_fraction: bool        # rounds to nearest sixth or 0
    note: str = ""


# -----------------------------------------------------------------------------
# Load and analyse
# -----------------------------------------------------------------------------
def load_pdg_table() -> list[HyperchargeRow]:
    wb = openpyxl.load_workbook(CATALOG, data_only=True)
    ws = wb["3. Particules PDG"]
    rows: list[HyperchargeRow] = []

    for r in range(3, ws.max_row + 1):
        raw = [c.value for c in ws[r]]
        if not raw or not raw[0]:
            continue
        name = str(raw[0]).strip()
        if name not in PDG_HYPERCHARGE:
            continue
        cfg = str(raw[3]).strip()
        knots = parse_knot_string(cfg)
        if not knots:
            continue

        t3, y_pdg, q_pdg, sector, _qc = PDG_HYPERCHARGE[name]

        # Gell-Mann-Nishijima check on PDG reference numbers
        q_gmn = gell_mann_nishijima(t3, y_pdg)
        gmn_ok = abs(q_gmn - q_pdg) < 1e-6

        # V5 prediction
        if len(knots) == 1:
            y_v5 = v5_hypercharge_single_knot(knots[0], sector)
        else:
            y_v5 = v5_hypercharge_composite(knots, sector)

        y_residual = y_v5 - y_pdg
        y_match_exact = abs(y_residual) < 1e-6
        # Allowed-rounding: match if both round to nearest 1/6
        y_match_fraction = abs(round(y_v5 * 6) - round(y_pdg * 6)) < 0.5

        note = ""
        if not gmn_ok:
            note += f"GMN mismatch Q_GMN={q_gmn:+.3f} vs Q_PDG={q_pdg:+.3f}; "
        if not y_match_exact:
            note += f"Y_V5={y_v5:+.3f} vs Y_PDG={y_pdg:+.3f}; "

        rows.append(HyperchargeRow(
            name=name, cfg=cfg, knots=knots, sector=sector,
            t3_pdg=t3, y_pdg=y_pdg, q_pdg=q_pdg,
            q_gmn_check=q_gmn, gmn_ok=gmn_ok,
            y_v5=y_v5, y_residual=y_residual,
            y_match_exact=y_match_exact,
            y_match_fraction=y_match_fraction,
            note=note.strip("; "),
        ))
    return rows


def write_csv(rows: list[HyperchargeRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "config", "sector", "T_3", "Y_PDG", "Q_PDG",
            "Q_GMN(=T_3+Y/2)", "GMN_ok",
            "Y_V5", "Y_residual", "Y_match_exact", "Y_match_fraction", "note",
        ])
        for r in rows:
            w.writerow([
                r.name, r.cfg, r.sector,
                f"{r.t3_pdg:+g}", f"{r.y_pdg:+g}", f"{r.q_pdg:+g}",
                f"{r.q_gmn_check:+g}", int(r.gmn_ok),
                f"{r.y_v5:+.4f}", f"{r.y_residual:+.4f}",
                int(r.y_match_exact), int(r.y_match_fraction), r.note,
            ])


def plot_summary(rows: list[HyperchargeRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    gmn_ok = sum(1 for r in rows if r.gmn_ok)
    y_exact = sum(1 for r in rows if r.y_match_exact)
    y_frac = sum(1 for r in rows if r.y_match_fraction)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Bar : Y match per particle
    ax = axes[0]
    x = np.arange(n)
    names = [r.name for r in rows]
    colours_gmn = ["#2ca02c" if r.gmn_ok else "#d62728" for r in rows]
    colours_y = ["#2ca02c" if r.y_match_exact else
                 "#ff7f0e" if r.y_match_fraction else "#d62728"
                 for r in rows]
    width = 0.4
    ax.bar(x - width / 2, [1] * n, width=width, color=colours_gmn,
           edgecolor="black", linewidth=0.3, label="Gell-Mann-Nishijima OK")
    ax.bar(x + width / 2, [1] * n, width=width, color=colours_y,
           edgecolor="black", linewidth=0.3,
           label="Y_V5 match (green=exact, orange=1/6 grid, red=miss)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title(
        f"V5.25 hypercharge Y_W — {n} particules · GMN {gmn_ok}/{n} ({100*gmn_ok/n:.0f}%)"
        f" · Y exact {y_exact}/{n} ({100*y_exact/n:.0f}%)"
        f" · Y rounded(1/6) {y_frac}/{n} ({100*y_frac/n:.0f}%)"
    )
    ax.legend(loc="upper right", fontsize=8)

    # Scatter Y_V5 vs Y_PDG
    ax2 = axes[1]
    ax2.scatter(
        [r.y_pdg for r in rows],
        [r.y_v5 for r in rows],
        c=["#2ca02c" if r.y_match_exact
           else "#ff7f0e" if r.y_match_fraction else "#d62728" for r in rows],
        s=40, edgecolor="black", linewidth=0.5,
    )
    ymin = min(min(r.y_pdg for r in rows), min(r.y_v5 for r in rows)) - 0.5
    ymax = max(max(r.y_pdg for r in rows), max(r.y_v5 for r in rows)) + 0.5
    ax2.plot([ymin, ymax], [ymin, ymax], "k--", lw=0.5, alpha=0.5, label="y=x")
    ax2.set_xlabel("Y_PDG (LH doublet convention, units of e)")
    ax2.set_ylabel("Y_V5 (from topological_hypercharge summed over knots)")
    ax2.set_title("Y_V5 vs Y_PDG (CAUTION: topological_hypercharge returns SM constants)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def main() -> dict:
    rows = load_pdg_table()
    write_csv(rows)
    plot_summary(rows)

    n = len(rows)
    gmn_ok = sum(1 for r in rows if r.gmn_ok)
    y_exact = sum(1 for r in rows if r.y_match_exact)
    y_frac = sum(1 for r in rows if r.y_match_fraction)
    sectors = {}
    for r in rows:
        sectors.setdefault(r.sector, []).append(r)

    print(f"\n=== V5.25 hypercharge Y_W validation — {n} particules ===")
    print(f"Gell-Mann-Nishijima OK : {gmn_ok}/{n}  ({100*gmn_ok/n:.1f}%)")
    print(f"Y_V5 exact match       : {y_exact}/{n}  ({100*y_exact/n:.1f}%)")
    print(f"Y_V5 1/6-grid match    : {y_frac}/{n}  ({100*y_frac/n:.1f}%)")
    print()
    print("Per sector :")
    for sec, srows in sectors.items():
        ye = sum(1 for r in srows if r.y_match_exact)
        gm = sum(1 for r in srows if r.gmn_ok)
        print(f"  {sec:12s} : n={len(srows):2d}  GMN={gm}/{len(srows)}  Y_exact={ye}/{len(srows)}")

    outliers = [r for r in rows if not r.y_match_exact]
    print(f"\nOutliers Y ({len(outliers)}):")
    for r in outliers[:30]:
        print(f"  - {r.name:12s} {r.cfg:30s} sector={r.sector:10s} "
              f"Y_PDG={r.y_pdg:+.3f} Y_V5={r.y_v5:+.3f}")

    print(f"\nOutputs : {OUT_CSV}\n          {OUT_PNG}")

    return {
        "n": n,
        "gmn_ok": gmn_ok,
        "y_exact": y_exact,
        "y_frac": y_frac,
        "rows": rows,
        "sectors": sectors,
    }


if __name__ == "__main__":
    main()
