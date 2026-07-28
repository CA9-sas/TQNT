#!/usr/bin/env python3
"""V5.19 - Test Class A (DM Pearls) vs Subaru HSC microlensing constraints.

Compare the V5.19 Class A prediction (topological Chandrasekhar DM pearls
catalogued in `theory/v5_compact_objects_catalog.py`) against the published
microlensing upper limits on the PBH-like fraction
f_PBH(M) = Omega_PBH / Omega_DM from Subaru Hyper Suprime-Cam (HSC) M31
monitoring and complementary OGLE / EROS surveys.

References
----------
* Niikura et al. 2019 (Nature Astronomy 3, 524-534), arXiv:1701.02151
  Subaru HSC dense-cadence monitoring of M31.
* Sugiyama et al. 2020 (MNRAS 493, 3632), arXiv:1905.06066
  Suprime-Cam complementary constraints, extending toward planetary masses.
* Smyth et al. 2020 (PRD 101, 063005), arXiv:1910.01285
  Combined OGLE + GAIA + Subaru analysis.
* Carr & Kuhnel 2020 (ARNPS 70, 355) - PBH microlensing constraints review.

V5 PREDICTION (catalogue v5_compact_objects_catalog.py)
------------------------------------------------------
Class A "DM Pearls":
  - M_typical = 2.03e-3 Msun (geometric mean / 10 of the topological
    Chandrasekhar range from the 6 STRICT amphichiral DM knot masses)
  - M_max     = 6.79e-1 Msun (Chandrasekhar of the lightest DM knot, 1.55 GeV)
  - Omega_class_A / Omega_DM = 0.05 (5% of DM in pearl form)
  - EM-invisible: detectable ONLY via gravitational microlensing.

The mass function is assumed log-normal centered on M_typical with
sigma_log10M = 0.3 dex (representative width of the Chandrasekhar range
collapsed to a single STRICT amphichiral knot mass).

The test:
  1. Builds an interpolation f_HSC_limit(M) from a hardcoded compilation
     of Subaru HSC / OGLE / EROS published 95% CL upper limits.
  2. Evaluates f_V5(M) = 0.05 * dN/dlog10M (log-normal) and compares to the
     interpolated limit at every M in [1e-11, 1e0] Msun.
  3. Reports the tension ratio f_V5 / f_HSC_limit at M = M_typical and
     across the full Class A mass range.
  4. Outputs `output/v5p19_class_A_subaru_microlensing.png` with 4 panels:
     (a) f_HSC_limit(M) data,
     (b) V5 Class A f(M) prediction (with mass-function spread),
     (c) ratio f_V5 / f_HSC_limit (tension),
     (d) future projection: sensitivity factor required to detect Class A.

Honest disclaimers (per CLAUDE.md V5 guidelines)
------------------------------------------------
* The Subaru HSC published limits assume MONOCHROMATIC PBH mass functions;
  for extended (log-normal) distributions the effective bound is somewhat
  weakened (Carr+ 2017). We do NOT attempt to undo this for the catalogue
  log-normal width 0.3 dex; the comparison is conservative.
* The mass-function width sigma = 0.3 dex is a default plausibility choice;
  the catalogue does NOT derive it from first principles for Class A.
* Subaru HSC constraints below M < 1e-10 Msun are heavily affected by wave
  optics (finite-source effects) and saturate; we extrapolate flat in this
  region as is conventional.

Run :
    PYTHONPATH=. python3 scripts/V5_microlensing/class_A_subaru_hsc_test.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.v5_compact_objects_catalog import (
    class_A_dm_pearls,
    OMEGA_DM,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# 1. Hardcoded Subaru HSC + complementary microlensing 95% CL upper limits
# =========================================================================
#
# Each entry is (log10(M / Msun), f_PBH_limit_95CL, source_tag).
#
# Sources (tags) :
#   N19  Niikura+ 2019 (Subaru HSC M31, Nature Astron. 3, 524)
#   N19s Niikura+ 2019 (Subaru HSC, planetary mass extrapolation)
#   Sm20 Smyth+ 2020 PRD 101, 063005 (OGLE+Subaru+GAIA combined)
#   OG   OGLE-IV constraints (Niikura+ 2019 Table A1)
#   EROS EROS-2 constraints (Tisserand+ 2007, used as legacy >1e-1 Msun)
#
# The grid is log-log; we interpolate piecewise-linearly in log10(M).
# Reasonable rounding to two significant figures.

HSC_LIMITS = [
    # (log10 M/Msun, f_PBH_95CL, source)
    (-11.0, 1.0,    "N19s"),  # finite-source wave optics, no useful limit
    (-10.5, 5.0e-1, "N19s"),
    (-10.0, 1.0e-1, "N19"),   # ~10% upper bound emerges
    ( -9.0, 5.0e-2, "N19"),
    ( -8.0, 3.0e-2, "N19"),   # best limit (~3%) around 10^-8 Msun
    ( -7.0, 4.0e-2, "N19"),
    ( -6.0, 5.0e-2, "N19"),
    ( -5.0, 1.0e-1, "N19"),
    ( -4.0, 2.0e-1, "Sm20"),
    ( -3.0, 3.0e-1, "Sm20"),  # *** in Class A window ***
    ( -2.0, 5.0e-1, "OG"),
    ( -1.0, 1.0e0,  "EROS"),  # essentially unconstrained at MACHO scale
    (  0.0, 1.0e0,  "EROS"),
]


def hsc_limit_interpolator() -> tuple[np.ndarray, np.ndarray]:
    """Return (log10_M_arr, f_limit_arr) sorted for piecewise-linear interp."""
    arr = np.array([(x[0], x[1]) for x in HSC_LIMITS], dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return arr[:, 0], arr[:, 1]


def f_hsc_limit(M_Msun: float | np.ndarray) -> np.ndarray:
    """Subaru HSC + complementary 95% CL upper limit on f_PBH at mass M_Msun.

    Piecewise-linear in log10(M) - log10(f), clamped at f=1 outside the
    instrumented window.
    """
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    log_M = np.log10(np.maximum(M_arr, 1e-30))
    x, y = hsc_limit_interpolator()
    # piecewise-linear in log-log space
    log_y = np.interp(log_M, x, np.log10(y))
    f = 10.0 ** log_y
    return np.minimum(f, 1.0)


# =========================================================================
# 2. V5 Class A predicted f(M)
# =========================================================================

@dataclass(frozen=True)
class V5ClassA:
    f_total: float            # Omega_class_A / Omega_DM
    log_M_typical: float      # log10(M_typical / Msun)
    sigma_log_M: float        # dex spread of the log-normal mass function
    M_max_Msun: float          # absolute maximum from Chandrasekhar / m_chi^2


def v5_class_A_parameters(sigma_log_M: float = 0.3) -> V5ClassA:
    """Build the V5 Class A parameters from the catalogue.

    sigma_log_M is the default log-normal width: 0.3 dex matches the typical
    mass scatter from the 6 STRICT amphichiral knots collapsed to a single
    log-normal.
    """
    cls_A = class_A_dm_pearls()
    return V5ClassA(
        f_total=cls_A.omega_over_omega_dm,
        log_M_typical=math.log10(cls_A.m_typical_Msun),
        sigma_log_M=sigma_log_M,
        M_max_Msun=cls_A.m_max_Msun,
    )


def f_v5_class_A(
    M_Msun: float | np.ndarray,
    params: V5ClassA | None = None,
) -> np.ndarray:
    """V5 Class A f(M) = f_total * dN/dlog10(M) (log-normal).

    This is the effective constraint quantity that microlensing surveys bound:
    the integrated dark-matter fraction "in pearls around mass M per dex".

    Returns the log-normal weighted f(M) so it can be compared directly to
    the published monochromatic f_PBH upper limit (the standard convention
    in extended-mass-function microlensing tests).
    """
    if params is None:
        params = v5_class_A_parameters()
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    log_M = np.log10(np.maximum(M_arr, 1e-30))
    # log-normal in log10 M: dN / dlog10M = (1/(sigma sqrt(2pi))) exp(-(...)
    norm = 1.0 / (params.sigma_log_M * math.sqrt(2 * math.pi))
    kernel = np.exp(
        -0.5 * ((log_M - params.log_M_typical) / params.sigma_log_M) ** 2
    )
    f = params.f_total * norm * kernel
    # Hard truncation above M_max (topological Chandrasekhar cutoff)
    f = np.where(M_arr > params.M_max_Msun, 0.0, f)
    return f


# =========================================================================
# 3. Tension / compatibility metrics
# =========================================================================

def tension_ratio(
    M_Msun: float | np.ndarray,
    params: V5ClassA | None = None,
) -> np.ndarray:
    """f_V5(M) / f_HSC_limit(M). >1 means V5 EXCLUDED at this M."""
    M_arr = np.atleast_1d(np.asarray(M_Msun, dtype=float))
    f_v5 = f_v5_class_A(M_arr, params=params)
    f_lim = f_hsc_limit(M_arr)
    # safe division
    return np.where(f_lim > 0, f_v5 / f_lim, np.inf)


def verdict(params: V5ClassA | None = None) -> dict:
    """Decision over the Class A mass range.

    Returns a dict with keys:
      - max_tension          : max f_V5/f_limit over [1e-11, 1e0] Msun
      - M_max_tension        : the M at which tension peaks
      - n_excluded_decades   : nb of dex over which V5 > limit (extent)
      - ratio_at_M_typical   : f_V5/f_limit at M_typical
      - verdict              : "SURVIVES" / "MARGINAL" / "EXCLUDED"
    """
    if params is None:
        params = v5_class_A_parameters()
    M_grid = np.logspace(-11.0, 0.0, 4400)
    f_v5 = f_v5_class_A(M_grid, params=params)
    f_lim = f_hsc_limit(M_grid)
    ratio = np.where(f_lim > 0, f_v5 / f_lim, np.inf)
    # count decades where V5 exceeds the limit
    dlogM = (np.log10(M_grid[-1]) - np.log10(M_grid[0])) / (len(M_grid) - 1)
    n_excluded = float(np.sum(ratio > 1.0) * dlogM)
    M_typ = 10.0 ** params.log_M_typical
    ratio_typ = float(f_v5_class_A(M_typ, params=params)[0]
                      / f_hsc_limit(M_typ)[0])
    max_t = float(np.max(ratio))
    M_max_t = float(M_grid[int(np.argmax(ratio))])
    if max_t <= 1.0:
        v = "SURVIVES"
    elif max_t <= 2.0:
        v = "MARGINAL"
    else:
        v = "EXCLUDED"
    return {
        "max_tension": max_t,
        "M_max_tension": M_max_t,
        "n_excluded_decades": n_excluded,
        "ratio_at_M_typical": ratio_typ,
        "M_typical_Msun": M_typ,
        "f_V5_at_M_typical": float(f_v5_class_A(M_typ, params=params)[0]),
        "f_limit_at_M_typical": float(f_hsc_limit(M_typ)[0]),
        "verdict": v,
    }


def sensitivity_factor_to_detect(
    params: V5ClassA | None = None,
) -> float:
    """Factor by which Subaru HSC sensitivity must improve to DETECT V5.

    Defined as min over M of (f_HSC_limit / f_V5). A factor of 1 means
    V5 sits right on the current bound; >1 means future surveys must
    tighten the limit by that factor to begin probing the prediction.
    """
    if params is None:
        params = v5_class_A_parameters()
    M_grid = np.logspace(-11.0, 0.0, 4400)
    f_v5 = f_v5_class_A(M_grid, params=params)
    f_lim = f_hsc_limit(M_grid)
    # only meaningful where the prediction is nonzero
    mask = f_v5 > 1e-12
    if not np.any(mask):
        return float("inf")
    return float(np.min(f_lim[mask] / f_v5[mask]))


# =========================================================================
# 4. Plotting
# =========================================================================

def make_plot(path: Path | None = None) -> Path:
    """Render the 4-panel figure to path (default output dir)."""
    if path is None:
        path = OUTPUT_DIR / "v5p19_class_A_subaru_microlensing.png"
    params = v5_class_A_parameters()
    M_grid = np.logspace(-11.0, 0.0, 4400)
    f_lim = f_hsc_limit(M_grid)
    f_v5 = f_v5_class_A(M_grid, params=params)
    ratio = np.where(f_lim > 0, f_v5 / f_lim, np.inf)
    sens = sensitivity_factor_to_detect(params=params)
    vdict = verdict(params=params)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "V5.19 Class A (DM Pearls) vs Subaru HSC microlensing",
        fontsize=14,
    )

    # (a) f_HSC_limit(M)
    ax = axes[0, 0]
    pts = np.array([(x[0], x[1]) for x in HSC_LIMITS])
    ax.loglog(M_grid, f_lim, "k-", lw=2, label="Interp. 95% CL upper limit")
    ax.scatter(10.0 ** pts[:, 0], pts[:, 1], color="red", s=30, zorder=5,
               label="HSC/OGLE/EROS data")
    ax.axvline(10.0 ** params.log_M_typical, color="blue", ls=":",
               label=f"M_typ = {10.0 ** params.log_M_typical:.2e} Msun")
    ax.set_xlabel("M [Msun]")
    ax.set_ylabel("f_PBH = Omega_PBH / Omega_DM (95% CL)")
    ax.set_title("(a) Subaru HSC + complementary limits")
    ax.set_ylim(1e-3, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (b) V5 Class A f(M)
    ax = axes[0, 1]
    ax.loglog(M_grid, f_v5, "b-", lw=2, label="V5 Class A f(M)")
    ax.loglog(M_grid, f_lim, "k--", lw=1, alpha=0.6, label="HSC limit")
    ax.axhline(params.f_total, color="green", ls=":", alpha=0.6,
               label=f"Omega_A/Omega_DM = {params.f_total}")
    ax.set_xlabel("M [Msun]")
    ax.set_ylabel("f(M) = dOmega_A/dlog10M (per dex) / Omega_DM")
    ax.set_title("(b) V5 Class A log-normal mass function")
    ax.set_ylim(1e-6, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (c) tension ratio
    ax = axes[1, 0]
    ax.loglog(M_grid, np.maximum(ratio, 1e-12), "purple", lw=2)
    ax.axhline(1.0, color="red", ls="--", label="exclusion threshold")
    ax.axvline(10.0 ** params.log_M_typical, color="blue", ls=":")
    ax.fill_between(
        M_grid, 1e-12, np.maximum(ratio, 1e-12),
        where=(ratio > 1.0), color="red", alpha=0.2,
        label="excluded region",
    )
    ax.set_xlabel("M [Msun]")
    ax.set_ylabel("f_V5 / f_HSC_limit")
    ax.set_title(
        f"(c) Tension ratio (max = {vdict['max_tension']:.2g}) - "
        f"verdict: {vdict['verdict']}"
    )
    ax.set_ylim(1e-4, 1e2)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (d) future sensitivity required
    ax = axes[1, 1]
    # Plot f_v5 vs current limit and the dashed line showing required
    # sensitivity-factor improvement
    ax.loglog(M_grid, f_v5, "b-", lw=2, label="V5 Class A (prediction)")
    ax.loglog(M_grid, f_lim, "k-", lw=1, alpha=0.5, label="Subaru HSC limit")
    ax.loglog(M_grid, f_lim / max(sens, 1.0), "g--", lw=2,
              label=f"Required sens. x{sens:.1f}")
    ax.axvline(10.0 ** params.log_M_typical, color="blue", ls=":")
    ax.set_xlabel("M [Msun]")
    ax.set_ylabel("f(M)")
    ax.set_title(
        f"(d) Sensitivity x{sens:.1f} required to detect V5 Class A"
    )
    ax.set_ylim(1e-6, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================================
# 5. CLI summary
# =========================================================================

def summary() -> dict:
    """Compact dict for CLI / tests."""
    params = v5_class_A_parameters()
    v = verdict(params=params)
    return {
        "M_typical_Msun": v["M_typical_Msun"],
        "f_total": params.f_total,
        "sigma_log_M": params.sigma_log_M,
        "M_max_Msun": params.M_max_Msun,
        "f_HSC_limit_at_M_typical": v["f_limit_at_M_typical"],
        "f_V5_at_M_typical": v["f_V5_at_M_typical"],
        "ratio_at_M_typical": v["ratio_at_M_typical"],
        "max_tension": v["max_tension"],
        "M_max_tension_Msun": v["M_max_tension"],
        "n_excluded_decades": v["n_excluded_decades"],
        "sensitivity_factor_to_detect":
            sensitivity_factor_to_detect(params=params),
        "verdict": v["verdict"],
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 78)
    print("V5.19 Class A (DM Pearls) vs Subaru HSC microlensing")
    print("=" * 78)
    print(f"\nV5 Class A catalogue parameters:")
    print(f"  M_typical            = {s['M_typical_Msun']:.3e} Msun")
    print(f"  M_max                = {s['M_max_Msun']:.3e} Msun")
    print(f"  Omega_A / Omega_DM   = {s['f_total']:.3f}")
    print(f"  sigma_log10(M)       = {s['sigma_log_M']:.2f} dex")
    print(f"\nMicrolensing comparison at M = M_typical:")
    print(f"  f_HSC_limit(M_typ)   = {s['f_HSC_limit_at_M_typical']:.3e}")
    print(f"  f_V5(M_typ)          = {s['f_V5_at_M_typical']:.3e}")
    print(f"  ratio f_V5 / f_lim   = {s['ratio_at_M_typical']:.3e}")
    print(f"\nGlobal verdict:")
    print(f"  max tension          = {s['max_tension']:.3e} "
          f"@ M = {s['M_max_tension_Msun']:.3e} Msun")
    print(f"  excluded over        = {s['n_excluded_decades']:.2f} dex")
    print(f"  sensitivity to det.  = x{s['sensitivity_factor_to_detect']:.1f}")
    print(f"  VERDICT              = {s['verdict']}")
    path = make_plot()
    print(f"\nFigure written to: {path}")
