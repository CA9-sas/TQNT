"""
NICER mass-radius posterior loader for TQNT-V5.

Loads M-R posterior samples from /mnt/data/data_recherche/nicer_mr/.
Each pulsar sub-folder contains a `posterior_samples.csv`
(columns: mass_Msun, radius_km) downsampled to 10 000 points, plus a
`source.txt` documenting Zenodo DOI and the original paper.

Pulsars with archives too large to ingest (Riley 2019/2021) are returned
as Gaussian-approx summary dicts based on the published median +/- 1 sigma.

Entry points:
    load_pulsar(name)  -> DataFrame or dict
    load_all()         -> dict[name, DataFrame]
    predict_radius_V5(mass, case='B') -> np.nan (stub)

Run as __main__ to dump a summary table and an overlaid M-R credible-region
plot to output/nicer_mr_tqnt.png.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------ paths
DATA_ROOT = Path("/mnt/data/data_recherche/nicer_mr")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

# Map convenient short names to sub-folder names
_PULSAR_DIRS = {
    "J0030_Miller":    "J0030_Miller2019",
    "J0030_Riley":     "J0030_Riley2019",
    "J0740_Miller":    "J0740_Miller2021",
    "J0740_Riley":     "J0740_Riley2021",
    "J0437_Choudhury": "J0437_Choudhury2024",
    # back-compat aliases
    "J0030":           "J0030_Miller2019",
    "J0740":           "J0740_Miller2021",
    "J0437":           "J0437_Choudhury2024",
}

# Published headline values used when a full posterior CSV is unavailable
# (Riley 2019/2021 — archives 854 MB / 8 GB, see source.txt files)
_SUMMARY_FALLBACK = {
    "J0030_Riley2019": dict(
        M_median=1.34, M_68_lo=1.18, M_68_hi=1.49,
        R_median=12.71, R_68_lo=11.52, R_68_hi=13.85,
        note="Riley+2019 ApJL 887 L21, ST+PST model (Gaussian approx).",
    ),
    "J0740_Riley2021": dict(
        M_median=2.072, M_68_lo=2.006, M_68_hi=2.139,
        R_median=12.39, R_68_lo=11.41, R_68_hi=13.69,
        note="Riley+2021 ApJL 918 L27 (Gaussian approx).",
    ),
}


# ----------------------------------------------------------- core loaders
def _summarise(M: np.ndarray, R: np.ndarray) -> dict:
    """Compute median + 68% interval for unweighted samples."""
    M_med = float(np.median(M))
    R_med = float(np.median(R))
    M_lo, M_hi = (float(x) for x in np.percentile(M, [16, 84]))
    R_lo, R_hi = (float(x) for x in np.percentile(R, [16, 84]))
    return dict(
        M_median=M_med, M_68_lo=M_lo, M_68_hi=M_hi,
        R_median=R_med, R_68_lo=R_lo, R_68_hi=R_hi,
    )


def load_pulsar(name: str):
    """Return a DataFrame (mass_Msun, radius_km) plus summary stats,
    or a summary dict if only published values are available.

    The returned DataFrame has `.attrs` set with all summary keys.
    """
    if name not in _PULSAR_DIRS:
        raise KeyError(f"Unknown pulsar {name!r}. "
                       f"Known names: {sorted(_PULSAR_DIRS)}")

    sub = _PULSAR_DIRS[name]
    folder = DATA_ROOT / sub
    csv_path = folder / "posterior_samples.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        summary = _summarise(df["mass_Msun"].to_numpy(),
                             df["radius_km"].to_numpy())
        summary["source_dir"] = str(folder)
        summary["has_posterior"] = True
        df.attrs.update(summary)
        return df

    # Fallback: Gaussian-approx summary only
    if sub in _SUMMARY_FALLBACK:
        out = dict(_SUMMARY_FALLBACK[sub])
        out["source_dir"] = str(folder)
        out["has_posterior"] = False
        return out

    raise FileNotFoundError(
        f"No posterior_samples.csv at {csv_path} and no fallback summary.")


def load_all() -> dict:
    """Load every pulsar sub-folder. Returns dict keyed by short name."""
    out = {}
    for short, sub in _PULSAR_DIRS.items():
        if short in {"J0030", "J0740", "J0437"}:
            continue  # skip aliases
        try:
            out[short] = load_pulsar(short)
        except FileNotFoundError as exc:
            warnings.warn(f"Could not load {short}: {exc}")
    return out


# ----------------------------------------------------- TQNT-V5 prediction
def predict_radius_V5(mass_Msun, case: str = "B"):
    """Stub: TQNT-V5 prediction of neutron-star radius as a function of mass.

    Not yet implemented. Returns NaN and emits a warning.

    Parameters
    ----------
    mass_Msun : float or np.ndarray
        Gravitational mass in solar masses.
    case : str
        'A' = B_ext approx 0 (pure Omega_ts), 'B' = B_ext != 0 (P23
        broken-symmetry correction). All NICER pulsars are Case B because
        of their magnetospheres.
    """
    warnings.warn(
        "Not implemented - V5 NS radius prediction needs Omega_ts + P23 with "
        "B_ext != 0 magnetosphere model. See V5 sections 15-16.",
        stacklevel=2,
    )
    arr = np.atleast_1d(np.asarray(mass_Msun, dtype=float))
    return np.full_like(arr, np.nan) if arr.size > 1 else np.nan


# ------------------------------------------------------------- plot utils
def _kde_credible_contours(M, R, levels=(0.95, 0.68), grid=120, pad=0.15):
    """Build a KDE on (R, M) and return contour levels enclosing
    the requested credible mass fractions (highest-density regions)."""
    M = np.asarray(M); R = np.asarray(R)
    R_span = np.ptp(R); M_span = np.ptp(M)
    R_lo, R_hi = R.min() - pad * R_span, R.max() + pad * R_span
    M_lo, M_hi = M.min() - pad * M_span, M.max() + pad * M_span
    rr = np.linspace(R_lo, R_hi, grid)
    mm = np.linspace(M_lo, M_hi, grid)
    RR, MM = np.meshgrid(rr, mm)
    kde = gaussian_kde(np.vstack([R, M]))
    Z = kde(np.vstack([RR.ravel(), MM.ravel()])).reshape(RR.shape)
    Zflat = Z.ravel(); order = np.argsort(Zflat)[::-1]
    cum = np.cumsum(Zflat[order]); cum /= cum[-1]
    contour_levels = []
    for lvl in levels:
        idx = np.searchsorted(cum, lvl)
        idx = min(idx, len(order) - 1)
        contour_levels.append(Zflat[order[idx]])
    return RR, MM, Z, sorted(contour_levels)


def _plot_summary_ellipse(ax, summary, color, label, ls="--"):
    """Plot a Gaussian-approx 1-sigma and 2-sigma ellipse for a pulsar
    that only has summary statistics (no posterior CSV)."""
    from matplotlib.patches import Ellipse
    M0 = summary["M_median"]; R0 = summary["R_median"]
    dM = 0.5 * (summary["M_68_hi"] - summary["M_68_lo"])
    dR = 0.5 * (summary["R_68_hi"] - summary["R_68_lo"])
    for k, alpha in ((1.0, 0.45), (2.0, 0.20)):
        e = Ellipse((R0, M0), width=2 * k * dR, height=2 * k * dM,
                    edgecolor=color, facecolor=color, alpha=alpha,
                    linestyle=ls, linewidth=1.4)
        ax.add_patch(e)
    ax.plot(R0, M0, marker="x", color=color, mew=2, ms=8,
            label=f"{label} (summary)")


# ------------------------------------------------------------------ main
def _print_summary():
    print("=" * 72)
    print("NICER M-R posteriors — summary (TQNT-V5 input)")
    print("=" * 72)
    print(f"{'Pulsar':22s} {'M_med':>7s} {'+1s':>6s} {'-1s':>6s} "
          f"{'R_med':>7s} {'+1s':>6s} {'-1s':>6s} {'N':>7s}")
    for short in ("J0030_Miller", "J0030_Riley",
                  "J0740_Miller", "J0740_Riley", "J0437_Choudhury"):
        obj = load_pulsar(short)
        if isinstance(obj, pd.DataFrame):
            s = obj.attrs; N = len(obj)
        else:
            s = obj; N = 0
        Mm = s["M_median"]; dMp = s["M_68_hi"] - Mm; dMn = Mm - s["M_68_lo"]
        Rm = s["R_median"]; dRp = s["R_68_hi"] - Rm; dRn = Rm - s["R_68_lo"]
        print(f"{short:22s} {Mm:7.3f} {dMp:+6.3f} {-dMn:+6.3f} "
              f"{Rm:7.2f} {dRp:+6.2f} {-dRn:+6.2f} {N:7d}")
    print("=" * 72)


def _make_plot(outpath: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    # colour/label per pulsar (distinct hues)
    cfg = [
        ("J0030_Miller",    "tab:blue",   "J0030+0451 Miller+2019"),
        ("J0030_Riley",     "tab:cyan",   "J0030+0451 Riley+2019"),
        ("J0740_Miller",    "tab:red",    "J0740+6620 Miller+2021"),
        ("J0740_Riley",     "tab:orange", "J0740+6620 Riley+2021"),
        ("J0437_Choudhury", "tab:green",  "J0437-4715 Choudhury+2024"),
    ]

    for short, color, label in cfg:
        obj = load_pulsar(short)
        if isinstance(obj, pd.DataFrame):
            R = obj["radius_km"].to_numpy()
            M = obj["mass_Msun"].to_numpy()
            RR, MM, Z, lvls = _kde_credible_contours(M, R)
            ax.contourf(RR, MM, Z, levels=[lvls[0], lvls[1], Z.max()],
                        colors=[color, color], alpha=0.18)
            ax.contour(RR, MM, Z, levels=lvls,
                       colors=color, linewidths=1.4)
            ax.plot(np.median(R), np.median(M),
                    marker="o", color=color, ms=6, label=label)
        else:
            _plot_summary_ellipse(ax, obj, color, label)

    ax.set_xlabel("Equatorial radius $R_\\mathrm{eq}$ [km]", fontsize=12)
    ax.set_ylabel(r"Gravitational mass $M$ [$M_\odot$]", fontsize=12)
    ax.set_title("NICER mass-radius posteriors — TQNT-V5 input "
                 "(Case B, $B_\\mathrm{ext}\\neq 0$)", fontsize=12)
    ax.set_xlim(8.5, 18.0)
    ax.set_ylim(1.0, 2.4)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    fig.text(0.99, 0.01,
             "filled contours = 68% / 95% HPD from MCMC; "
             "dashed ellipses = Gaussian summary",
             ha="right", va="bottom", fontsize=8, color="0.4")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {outpath}")


if __name__ == "__main__":
    _print_summary()
    out_png = OUTPUT_DIR / "nicer_mr_tqnt.png"
    _make_plot(out_png)
    # sanity-check the V5 stub
    val = predict_radius_V5(1.4)
    assert np.isnan(val), "predict_radius_V5 should return NaN until V5 is wired"
    print("predict_radius_V5(1.4) =", val, " (expected NaN)")
