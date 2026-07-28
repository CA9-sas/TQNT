#!/usr/bin/env python3
"""V5.19 — Test Class E (Strange Knot Stars) vs Subaru HSC microlensing.

Class E "Strange Knot Stars" is the 6-amphichiral STRICT sub-population of
TQNT-V5 dark matter (catalogue `TQNT_catalog_V5_diffusion.xlsx`, onglet 5).
Unlike Class A "DM Pearls" — which collapses the whole DM knot zoo into a
single broad log-normal centered on the geometric-mean Chandrasekhar mass —
Class E retains the DISCRETE structure: 6 amphichiral species, each with
its own topological Chandrasekhar mass M_Ch = M_Pl^3 / m_knot^2 and its own
abundance weight set by the V5 P18 condensation penalty α^(c - c_min)
(c = crossing number, α = 1.20 from V5 §22 Big-Bang condensation kinetics).

The expected signature is a MULTI-PEAK microlensing mass function, with
six log-normal components instead of one. The question this test answers:

  1. Are the 6 peaks individually below the published Subaru HSC + OGLE +
     EROS f_PBH(M) upper limits ?
  2. Is the multi-peak pattern actually RESOLVABLE by Subaru HSC
     (separation between peaks vs experimental log-M resolution) ?

V5 PREDICTION (catalogue V5)
----------------------------
Six STRICT amphichirals, each with its Chandrasekhar mass:

  name              c     m_knot (GeV)    M_Ch (M_⊙)     log10 M_Ch
  ------------------------------------------------------------
  6_3+6_3          12         1.55        6.79e-01       -0.17
  6_3+6_3+6_3      18         4.70        7.38e-02       -1.13
  4_1+8_3          12        16.01        6.36e-03       -2.20
  8_3+8_3          16        17.24        5.49e-03       -2.26
  6_3+8_3          14        26.48        2.33e-03       -2.63
  4_1+4_1+8_3      16        51.78        6.08e-04       -3.22

Total Ω_E / Ω_DM = 0.10 (10% of DM is in STRICT amphichirals).

Abundance weight per amphichiral:

  Ω_i ∝ α^(c_i - c_min)        with α = 1.20  (V5 P18 condensation)
  c_min = 12

This makes the c=12 species (6_3+6_3, 4_1+8_3) the most abundant; the
c=18 species (6_3+6_3+6_3) the least abundant.

References
----------
* CLAUDE.md V5 spec (Class E section)
* `docs/TQNT_detaillee_V5_FR.tex` §13.7 (catalogue 117 candidats, 6 STRICT)
* Niikura+ 2019 Nature Astron. 3, 524 (arXiv:1701.02151) — Subaru HSC
* Smyth+ 2020 PRD 101, 063005 (arXiv:1910.01285) — combined OGLE+HSC+GAIA
* Tisserand+ 2007 A&A 469, 387 — EROS-2 MACHO limits

Honest disclaimers (per CLAUDE.md V5 guidelines)
------------------------------------------------
* Published microlensing limits are derived for MONOCHROMATIC mass functions;
  applying them peak-by-peak is conservative (extended distributions give
  slightly weaker bounds; Carr+ 2017).
* The abundance-weight formula α^(c - c_min) is the V5 §22 condensation
  penalty (P18); the Vassiliev-suppression formula σ_DM-N ∝ α_EM^(c-18)
  in CLAUDE.md applies to direct-detection cross-sections, NOT to
  cosmic-abundance weights.
* The amphichirals' Chandrasekhar mass uses M_Pl^3 / m_χ^2 with the same
  conventions as Class A.

Run :
    PYTHONPATH=. python3 scripts/V5_microlensing/class_E_strange_knot_test.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the HSC limit table + interpolator from Class A so we are not
# duplicating data: any future revision to the table propagates everywhere.
sys.path.insert(0, str(ROOT / "scripts" / "V5_microlensing"))
import class_A_subaru_hsc_test as classA  # noqa: E402

from theory.v5_compact_objects_catalog import (  # noqa: E402
    GEV_KG,
    MPL_KG,
    MSUN_KG,
    chandrasekhar_mass_topological,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# 1. Catalogue of the 6 STRICT amphichirals
# =========================================================================
#
# Each entry: (label, crossing_number_c_total, m_knot_GeV).
# Crossing numbers are SUM of constituent prime knots:
#   3_1  -> 3 (not amphichiral, skipped),
#   4_1  -> 4,  6_3 -> 6,  8_3 -> 8.
# Masses are the catalogue V5 values reproduced in CLAUDE.md.

AMPHICHIRALS: List[Tuple[str, int, float]] = [
    ("6_3+6_3",       12,  1.55),
    ("6_3+6_3+6_3",   18,  4.70),
    ("4_1+8_3",       12, 16.01),
    ("8_3+8_3",       16, 17.24),
    ("6_3+8_3",       14, 26.48),
    ("4_1+4_1+8_3",   16, 51.78),
]

# V5 P18 condensation penalty per crossing (V5 §22).
# Lightest crossing has weight 1; heavier crossings get suppressed by alpha**(c-c_min).
ALPHA_P18 = 1.0 / 1.20

# Total dark-matter fraction in Class E (V5 catalogue §13.7).
OMEGA_E_OVER_OMEGA_DM = 0.10

# Default per-peak log-normal width.  We deliberately pick a narrower spread
# than Class A's 0.3 dex because a STRICT amphichiral has a UNIQUE
# Chandrasekhar mass — the only spread comes from finite-T condensation
# scatter, hence ~0.1 dex (cf. V5 §22).
DEFAULT_SIGMA_PEAK_DEX = 0.1


# =========================================================================
# 2. Class E parameters
# =========================================================================

@dataclass(frozen=True)
class V5AmphichiralPeak:
    """One log-normal peak in the Class E mass function."""
    label: str
    crossing: int
    m_knot_GeV: float
    log_M_Ch: float          # log10(M_Ch / M_sun) — center of the log-normal
    weight: float            # normalised so sum(weights) == 1
    f_i: float               # Ω_i / Ω_DM contribution
    sigma_log_M: float       # peak width in dex


@dataclass(frozen=True)
class V5ClassE:
    """V5 Class E "Strange Knot Stars" multi-peak parameters."""
    f_total: float                       # Omega_E / Omega_DM
    peaks: Tuple[V5AmphichiralPeak, ...]
    sigma_log_M: float

    @property
    def n_peaks(self) -> int:
        return len(self.peaks)


def _amphichiral_chandrasekhar_Msun(m_knot_GeV: float) -> float:
    """M_Ch (in M_⊙) for one amphichiral with knot mass m_knot (GeV)."""
    return chandrasekhar_mass_topological(m_knot_GeV) / MSUN_KG


def v5_class_E_parameters(
    sigma_log_M: float = DEFAULT_SIGMA_PEAK_DEX,
    f_total: float = OMEGA_E_OVER_OMEGA_DM,
    alpha_penalty: float = ALPHA_P18,
) -> V5ClassE:
    """Build the Class E multi-peak descriptor.

    Parameters
    ----------
    sigma_log_M : float
        Per-peak log-normal width in dex (default 0.1, see module docstring).
    f_total : float
        Ω_E / Ω_DM (default 0.10 per V5 catalogue).
    alpha_penalty : float
        Per-crossing P18 condensation penalty (default 1/1.20).
    """
    if sigma_log_M <= 0:
        raise ValueError("sigma_log_M must be positive")
    if not (0.0 < f_total <= 1.0):
        raise ValueError("f_total must lie in (0, 1]")
    if not (0.0 < alpha_penalty <= 1.0):
        raise ValueError("alpha_penalty must lie in (0, 1]")

    c_min = min(c for _, c, _ in AMPHICHIRALS)
    raw_weights = [alpha_penalty ** (c - c_min) for _, c, _ in AMPHICHIRALS]
    total_w = float(sum(raw_weights))
    norm_weights = [w / total_w for w in raw_weights]

    peaks: List[V5AmphichiralPeak] = []
    for (label, c, m_gev), w in zip(AMPHICHIRALS, norm_weights):
        M_Ch_Msun = _amphichiral_chandrasekhar_Msun(m_gev)
        peaks.append(
            V5AmphichiralPeak(
                label=label,
                crossing=c,
                m_knot_GeV=m_gev,
                log_M_Ch=math.log10(M_Ch_Msun),
                weight=w,
                f_i=f_total * w,
                sigma_log_M=sigma_log_M,
            )
        )
    return V5ClassE(
        f_total=f_total,
        peaks=tuple(peaks),
        sigma_log_M=sigma_log_M,
    )


# =========================================================================
# 3. Multi-peak mass function f(M)
# =========================================================================

def f_v5_class_E(
    M_Msun,
    params: V5ClassE | None = None,
) -> np.ndarray:
    """V5 Class E f(M) — sum of 6 log-normal peaks.

    Each peak contributes:
        f_i * (1 / (sigma * sqrt(2pi))) * exp(-(log10(M/M_Ch))^2 / (2 sigma^2))
    truncated above the largest M_Ch.
    """
    if params is None:
        params = v5_class_E_parameters()
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    log_M = np.log10(np.maximum(M_arr, 1e-30))
    total = np.zeros_like(M_arr, dtype=float)
    M_max_global = 10.0 ** max(p.log_M_Ch for p in params.peaks)
    for p in params.peaks:
        norm = 1.0 / (p.sigma_log_M * math.sqrt(2 * math.pi))
        kernel = np.exp(
            -0.5 * ((log_M - p.log_M_Ch) / p.sigma_log_M) ** 2
        )
        total = total + p.f_i * norm * kernel
    # Truncate above the heaviest M_Ch (global topological cutoff).
    # Use a generous 3-sigma buffer above the heaviest peak before going to 0.
    log_cut = math.log10(M_max_global) + 3.0 * params.sigma_log_M
    total = np.where(log_M > log_cut, 0.0, total)
    return total


def per_peak_f(
    M_Msun,
    params: V5ClassE | None = None,
) -> dict:
    """Return per-peak f_i(M) curves, useful for plotting decomposition."""
    if params is None:
        params = v5_class_E_parameters()
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    log_M = np.log10(np.maximum(M_arr, 1e-30))
    out = {}
    for p in params.peaks:
        norm = 1.0 / (p.sigma_log_M * math.sqrt(2 * math.pi))
        kernel = np.exp(
            -0.5 * ((log_M - p.log_M_Ch) / p.sigma_log_M) ** 2
        )
        out[p.label] = p.f_i * norm * kernel
    return out


# =========================================================================
# 4. Tension / verdict against HSC limits
# =========================================================================

def f_hsc_limit(M_Msun) -> np.ndarray:
    """Re-export Class A's HSC interpolation for convenience."""
    return classA.f_hsc_limit(M_Msun)


def tension_ratio(
    M_Msun,
    params: V5ClassE | None = None,
) -> np.ndarray:
    """f_V5_Class_E(M) / f_HSC_limit(M). >1 ⇒ EXCLUDED at this M."""
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    f_v5 = f_v5_class_E(M_arr, params=params)
    f_lim = f_hsc_limit(M_arr)
    return np.where(f_lim > 0, f_v5 / f_lim, np.inf)


def per_peak_tension(
    params: V5ClassE | None = None,
) -> List[dict]:
    """Tension at the centroid of each peak (M = M_Ch_i)."""
    if params is None:
        params = v5_class_E_parameters()
    out: List[dict] = []
    for p in params.peaks:
        M_c = 10.0 ** p.log_M_Ch
        f_v5_peak = float(f_v5_class_E(M_c, params=params)[0])
        f_lim = float(f_hsc_limit(M_c)[0])
        ratio = f_v5_peak / f_lim if f_lim > 0 else float("inf")
        out.append(
            {
                "label": p.label,
                "crossing": p.crossing,
                "m_knot_GeV": p.m_knot_GeV,
                "M_Ch_Msun": M_c,
                "log_M_Ch": p.log_M_Ch,
                "weight": p.weight,
                "f_i": p.f_i,
                "f_V5_at_peak": f_v5_peak,
                "f_HSC_limit_at_peak": f_lim,
                "ratio": ratio,
                "excluded": ratio > 1.0,
            }
        )
    return out


def verdict(params: V5ClassE | None = None) -> dict:
    """Global decision for Class E.

    A peak is considered EXCLUDED if its centroid tension > 1.
    Class E SURVIVES if no peak is excluded.  Class E is PARTIALLY
    EXCLUDED if at least one peak is excluded; we list which ones.
    """
    if params is None:
        params = v5_class_E_parameters()
    M_grid = np.logspace(-11.0, 0.0, 4400)
    f_v5 = f_v5_class_E(M_grid, params=params)
    f_lim = f_hsc_limit(M_grid)
    ratio = np.where(f_lim > 0, f_v5 / f_lim, np.inf)
    max_t = float(np.max(ratio))
    M_max_t = float(M_grid[int(np.argmax(ratio))])
    dlogM = (np.log10(M_grid[-1]) - np.log10(M_grid[0])) / (len(M_grid) - 1)
    n_excluded = float(np.sum(ratio > 1.0) * dlogM)

    per_peak = per_peak_tension(params=params)
    excluded_peaks = [p["label"] for p in per_peak if p["excluded"]]
    surviving_peaks = [p["label"] for p in per_peak if not p["excluded"]]

    if not excluded_peaks:
        v = "SURVIVES"
    elif len(excluded_peaks) == len(per_peak):
        v = "EXCLUDED"
    else:
        v = "PARTIALLY_EXCLUDED"

    return {
        "max_tension": max_t,
        "M_max_tension": M_max_t,
        "n_excluded_decades": n_excluded,
        "n_peaks": len(per_peak),
        "n_excluded_peaks": len(excluded_peaks),
        "excluded_peaks": excluded_peaks,
        "surviving_peaks": surviving_peaks,
        "verdict": v,
        "per_peak": per_peak,
    }


# =========================================================================
# 5. Multi-peak resolvability against Subaru HSC
# =========================================================================
#
# Subaru HSC dense-cadence monitoring of M31 (Niikura+ 2019) constrains
# f_PBH(M) on a grid finer than 1 dex.  We take a CONSERVATIVE
# experimental log-M resolution of 0.5 dex (the actual spacing of the
# Niikura table is comparable).  Two adjacent peaks are RESOLVABLE if
#
#   |delta log_M_Ch| > R_HSC                          (separated)
#   AND  combined sigma_eff = sqrt(sigma^2 + R_HSC^2) < |delta log_M_Ch|/2
#
# Otherwise they blend into a single broad feature indistinguishable
# from a Class-A-like log-normal.

R_HSC_DEX_DEFAULT = 0.5


def peak_separations(
    params: V5ClassE | None = None,
) -> List[float]:
    """Differences |log_M_Ch_{i+1} - log_M_Ch_i| for sorted peaks."""
    if params is None:
        params = v5_class_E_parameters()
    log_centers = sorted(p.log_M_Ch for p in params.peaks)
    return [log_centers[i + 1] - log_centers[i]
            for i in range(len(log_centers) - 1)]


def resolvability_report(
    params: V5ClassE | None = None,
    R_hsc_dex: float = R_HSC_DEX_DEFAULT,
) -> dict:
    """Quantify which adjacent peak pairs are resolvable by Subaru HSC.

    Returns dict with:
      - separations              : list of |delta log_M| between sorted peaks
      - is_resolvable_pairs      : bool list for each adjacent pair
      - n_resolved_peaks         : effective number of distinguishable peaks
      - multi_peak_visible       : True if at least 2 peaks resolved
    """
    if params is None:
        params = v5_class_E_parameters()
    seps = peak_separations(params=params)
    sigma_eff = math.sqrt(params.sigma_log_M ** 2 + R_hsc_dex ** 2)
    is_resolv = [s > 2.0 * sigma_eff for s in seps]
    # Cluster adjacent peaks that are NOT resolvable into one feature.
    n_clusters = 1
    for r in is_resolv:
        if r:
            n_clusters += 1
    return {
        "R_HSC_dex": R_hsc_dex,
        "sigma_eff_dex": sigma_eff,
        "separations_dex": seps,
        "is_resolvable_pairs": is_resolv,
        "n_resolved_peaks": n_clusters,
        "n_total_peaks": len(params.peaks),
        "multi_peak_visible": n_clusters >= 2,
    }


# =========================================================================
# 6. Comparison with Class A (single log-normal)
# =========================================================================

def f_v5_class_A_renormalised(
    M_Msun,
    f_total: float = OMEGA_E_OVER_OMEGA_DM,
) -> np.ndarray:
    """Class A log-normal renormalised to Class E's f_total.

    This is the "blended" curve that would be measured if Subaru HSC
    cannot resolve the 6 individual peaks and Class E hides as Class A.
    """
    params_A = classA.v5_class_A_parameters()
    # Rebuild with the requested f_total
    from class_A_subaru_hsc_test import V5ClassA  # local import
    params_eff = V5ClassA(
        f_total=f_total,
        log_M_typical=params_A.log_M_typical,
        sigma_log_M=params_A.sigma_log_M,
        M_max_Msun=params_A.M_max_Msun,
    )
    return classA.f_v5_class_A(M_Msun, params=params_eff)


# =========================================================================
# 7. Plotting
# =========================================================================

def make_plot(path: Path | None = None) -> Path:
    """Render the 4-panel figure to path (default output dir)."""
    if path is None:
        path = OUTPUT_DIR / "v5p19_class_E_microlensing.png"
    params = v5_class_E_parameters()
    M_grid = np.logspace(-5.0, 1.0, 3000)
    f_lim = f_hsc_limit(M_grid)
    f_v5_E = f_v5_class_E(M_grid, params=params)
    per_peak = per_peak_f(M_grid, params=params)
    f_v5_A_blended = f_v5_class_A_renormalised(M_grid)
    ratio = np.where(f_lim > 0, f_v5_E / f_lim, np.inf)
    vdict = verdict(params=params)
    res = resolvability_report(params=params)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"V5.19 Class E (Strange Knot Stars) vs Subaru HSC microlensing  "
        f"— verdict: {vdict['verdict']}",
        fontsize=14,
    )

    colors = ["tab:red", "tab:orange", "tab:olive",
              "tab:green", "tab:cyan", "tab:purple"]

    # (a) Multi-peak f(M) + HSC limit
    ax = axes[0, 0]
    ax.loglog(M_grid, f_lim, "k-", lw=2, label="Subaru HSC + OGLE + EROS 95% CL")
    ax.loglog(M_grid, f_v5_E, "b-", lw=2.5, label="Class E total")
    for i, (label, curve) in enumerate(per_peak.items()):
        ax.loglog(M_grid, curve, ls="--", lw=1.2,
                  color=colors[i % len(colors)], label=label, alpha=0.85)
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$f(M) = d\Omega/d\log_{10}M / \Omega_{DM}$")
    ax.set_title("(a) 6-peak multi-modal mass function")
    ax.set_xlim(1e-5, 10)
    ax.set_ylim(1e-6, 2.0)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # (b) Class E vs Class A blended
    ax = axes[0, 1]
    ax.loglog(M_grid, f_v5_A_blended, "g-", lw=2,
              label=r"Class A blended ($f_{tot}=0.10$)")
    ax.loglog(M_grid, f_v5_E, "b-", lw=2.5, label="Class E (this work)")
    ax.loglog(M_grid, f_lim, "k--", lw=1.5, alpha=0.6, label="HSC limit")
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$f(M)$")
    ax.set_title("(b) Class E vs Class A (same $f_{total}=0.10$)")
    ax.set_xlim(1e-5, 10)
    ax.set_ylim(1e-6, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (c) Tension ratio
    ax = axes[1, 0]
    ax.loglog(M_grid, np.maximum(ratio, 1e-12), color="purple", lw=2)
    ax.axhline(1.0, color="red", ls="--", label="exclusion threshold")
    ax.fill_between(
        M_grid, 1e-12, np.maximum(ratio, 1e-12),
        where=(ratio > 1.0), color="red", alpha=0.2,
        label="excluded region",
    )
    # Mark each peak centroid
    for p, c in zip(params.peaks, colors):
        ax.axvline(10.0 ** p.log_M_Ch, ls=":", color=c, alpha=0.7,
                   lw=1, label=f"{p.label}")
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$f_{V5}/f_{HSC}$")
    ax.set_title(
        f"(c) Tension (max = {vdict['max_tension']:.2g})  — "
        f"{vdict['n_excluded_peaks']}/{vdict['n_peaks']} peaks excluded"
    )
    ax.set_xlim(1e-5, 10)
    ax.set_ylim(1e-4, 1e2)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # (d) Resolvability bar chart
    ax = axes[1, 1]
    sorted_peaks = sorted(params.peaks, key=lambda p: p.log_M_Ch)
    log_centers = [p.log_M_Ch for p in sorted_peaks]
    labels = [p.label for p in sorted_peaks]
    ax.bar(range(len(log_centers)), log_centers,
           color=[colors[AMPHICHIRALS.index((p.label, p.crossing, p.m_knot_GeV))]
                  for p in sorted_peaks],
           alpha=0.7)
    for i, (lab, lc) in enumerate(zip(labels, log_centers)):
        ax.text(i, lc + 0.1 if lc > 0 else lc - 0.3, lab,
                rotation=45, fontsize=7, ha="center")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(labels))], fontsize=8)
    ax.set_ylabel(r"$\log_{10}(M_{Ch}/M_\odot)$")
    title = (
        f"(d) Peak ladder. Subaru HSC res. R={res['R_HSC_dex']} dex  →  "
        f"{res['n_resolved_peaks']}/{res['n_total_peaks']} resolved"
    )
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================================
# 8. Compact summary
# =========================================================================

def summary() -> dict:
    """Compact dict for CLI / tests."""
    params = v5_class_E_parameters()
    v = verdict(params=params)
    res = resolvability_report(params=params)
    return {
        "f_total": params.f_total,
        "n_peaks": params.n_peaks,
        "sigma_log_M_per_peak": params.sigma_log_M,
        "per_peak": v["per_peak"],
        "max_tension": v["max_tension"],
        "M_max_tension_Msun": v["M_max_tension"],
        "n_excluded_decades": v["n_excluded_decades"],
        "n_excluded_peaks": v["n_excluded_peaks"],
        "excluded_peaks": v["excluded_peaks"],
        "surviving_peaks": v["surviving_peaks"],
        "verdict": v["verdict"],
        "R_HSC_dex": res["R_HSC_dex"],
        "sigma_eff_dex": res["sigma_eff_dex"],
        "n_resolved_peaks": res["n_resolved_peaks"],
        "multi_peak_visible": res["multi_peak_visible"],
        "separations_dex": res["separations_dex"],
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 78)
    print("V5.19 Class E (Strange Knot Stars) vs Subaru HSC microlensing")
    print("=" * 78)
    print(f"\nClass E parameters:")
    print(f"  Omega_E / Omega_DM     = {s['f_total']:.3f}")
    print(f"  N peaks                = {s['n_peaks']}")
    print(f"  sigma per peak         = {s['sigma_log_M_per_peak']:.2f} dex")
    print(f"\nPer-peak masses, abundances, and tensions:")
    print(f"  {'name':<14} {'c':>3} {'M_Ch (Msun)':>14} "
          f"{'f_i':>10} {'f_lim':>10} {'ratio':>10} {'status':>10}")
    for p in s["per_peak"]:
        status = "EXCLUDED" if p["excluded"] else "OK"
        print(f"  {p['label']:<14} {p['crossing']:>3} "
              f"{p['M_Ch_Msun']:>14.4e} "
              f"{p['f_i']:>10.3e} {p['f_HSC_limit_at_peak']:>10.3e} "
              f"{p['ratio']:>10.3e} {status:>10}")
    print(f"\nVerdict:")
    print(f"  max tension            = {s['max_tension']:.3e}  "
          f"@ M = {s['M_max_tension_Msun']:.3e} Msun")
    print(f"  excluded peaks         = {s['n_excluded_peaks']}/{s['n_peaks']}"
          f"  ({s['excluded_peaks']})")
    print(f"  surviving peaks        = {s['surviving_peaks']}")
    print(f"  VERDICT                = {s['verdict']}")
    print(f"\nResolvability (R_HSC = {s['R_HSC_dex']} dex):")
    print(f"  separations (dex)      = "
          + ", ".join(f"{x:.2f}" for x in s["separations_dex"]))
    print(f"  effective sigma        = {s['sigma_eff_dex']:.2f} dex")
    print(f"  n resolved features    = {s['n_resolved_peaks']}/{s['n_peaks']}")
    print(f"  multi-peak visible     = {s['multi_peak_visible']}")
    path = make_plot()
    print(f"\nFigure written to: {path}")
