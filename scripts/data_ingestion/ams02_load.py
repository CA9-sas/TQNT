#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMS-02 cosmic-ray lepton & hadron loader for the TQNT-V5 falsification test.

Loads the AMS-02 positron fraction (Phys. Rep. 894, 1-116, 2021) and the
antiproton-to-proton flux ratio (PRL 117, 091103, 2016) from
    /mnt/data/data_recherche/ams02/positron_fraction.csv
    /mnt/data/data_recherche/ams02/antiproton_proton_ratio.csv
and produces a quick log-log inspection plot at
    /home/euk9/projets/physique_nouvelle/output/ams02_v5.png

Pas de venv : /usr/bin/python3 + numpy + pandas + matplotlib (backend Agg).

TQNT-V5 prediction
------------------
TQNT-V5 predicts ZERO dark-matter annihilation channel (Chern-Simons
barrier at k = 135 keeps the prime topological knots stable, no
breakable R-parity-like symmetry). Therefore:
    * the observed AMS-02 positron excess above ~10 GeV is predicted to
      be entirely astrophysical (pulsars: Geminga, Monogem, ... +
      secondary p-p production);
    * any DM-annihilation interpretation that fits AMS-02 better than
      the pulsar + secondary background falsifies TQNT-V5.

The function ``v5_prediction`` returns a dict making this explicit; the
function ``secondary_background_model`` is a stub (GALPROP / DRAGON
required externally) that returns NaN with a warning.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless obligatoire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/data/data_recherche/ams02")
PF_CSV = DATA_ROOT / "positron_fraction.csv"
PBARP_CSV = DATA_ROOT / "antiproton_proton_ratio.csv"
OUTPUT_PNG = Path(
    "/home/euk9/projets/physique_nouvelle/output/ams02_v5.png"
)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_positron_fraction() -> pd.DataFrame:
    """Return the AMS-02 positron fraction e+/(e+ + e-) vs kinetic energy.

    Source : AMS Collaboration, Phys. Rep. 894, 1-116 (2021),
    DOI 10.1016/j.physrep.2020.09.003 (INSPIRE 1841796).
    Mirrored via CRDB v4.1 (Maurin et al., EPJC 83, 971 (2023)).

    Returns
    -------
    DataFrame with columns
        E_min_GeV, E_max_GeV, E_GeV, positron_fraction, sigma_stat, sigma_syst, ads_ref
    """
    if not PF_CSV.exists():
        raise FileNotFoundError(
            f"Positron fraction CSV missing: {PF_CSV}. "
            "Re-run the data fetch step (see /mnt/data/data_recherche/ams02/README.md)."
        )
    df = pd.read_csv(PF_CSV)
    expected = {
        "E_min_GeV", "E_max_GeV", "E_GeV", "positron_fraction",
        "sigma_stat", "sigma_syst",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"positron_fraction.csv missing columns: {missing}")
    return df


def load_pbar_p() -> pd.DataFrame:
    """Return the AMS-02 antiproton-to-proton flux ratio vs magnetic rigidity.

    Source : AMS Collaboration, PRL 117, 091103 (2016),
    DOI 10.1103/PhysRevLett.117.091103 (INSPIRE 1484387).
    Mirrored via CRDB v4.1.

    Returns
    -------
    DataFrame with columns
        R_min_GV, R_max_GV, R_GV, pbar_p_ratio, sigma_stat, sigma_syst, ads_ref
    """
    if not PBARP_CSV.exists():
        raise FileNotFoundError(
            f"pbar/p CSV missing: {PBARP_CSV}. "
            "Re-run the data fetch step (see /mnt/data/data_recherche/ams02/README.md)."
        )
    df = pd.read_csv(PBARP_CSV)
    expected = {
        "R_min_GV", "R_max_GV", "R_GV", "pbar_p_ratio",
        "sigma_stat", "sigma_syst",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"antiproton_proton_ratio.csv missing columns: {missing}")
    return df


# ----------------------------------------------------------------------------
# TQNT-V5 prediction & secondary-background stub
# ----------------------------------------------------------------------------
def v5_prediction() -> dict:
    """Encode the TQNT-V5 verdict for any DM-annihilation interpretation.

    Returns
    -------
    dict
        * dm_annihilation_channel : 0.0
        * excess_attributable_to_DM : False
        * reason : string
    """
    return {
        "dm_annihilation_channel": 0.0,
        "excess_attributable_to_DM": False,
        "reason": (
            "V5 knot DM has no annihilation channel "
            "(CS barrier k=135 stable)"
        ),
    }


def secondary_background_model(rigidity_GV):
    """Stub for the secondary p-p antiproton/positron production background.

    A faithful background needs a numerical cosmic-ray transport solver
    (GALPROP, DRAGON, USINE, PICARD, ...). This function exists only to
    expose the right interface to downstream analysis scripts; it does
    NOT model anything.

    Parameters
    ----------
    rigidity_GV : array_like
        Magnetic rigidity in GV (or kinetic energy in GeV) at which the
        secondary background flux is evaluated.

    Returns
    -------
    ndarray
        Array of NaN, same shape as ``rigidity_GV``.
    """
    warnings.warn(
        "secondary p-p production model not implemented; "
        "use GALPROP or DRAGON predictions externally",
        UserWarning,
        stacklevel=2,
    )
    arr = np.asarray(rigidity_GV, dtype=float)
    return np.full(arr.shape, np.nan)


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def make_inspection_figure(
    pf: pd.DataFrame,
    pbarp: pd.DataFrame,
    out_path: Path = OUTPUT_PNG,
) -> Path:
    """Two-panel log-log inspection of the two AMS-02 datasets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left panel : positron fraction
    ax = axes[0]
    sigma_tot = np.sqrt(pf["sigma_stat"] ** 2 + pf["sigma_syst"] ** 2)
    ax.errorbar(
        pf["E_GeV"],
        pf["positron_fraction"],
        yerr=sigma_tot,
        xerr=[pf["E_GeV"] - pf["E_min_GeV"], pf["E_max_GeV"] - pf["E_GeV"]],
        fmt="o",
        ms=4,
        lw=1,
        color="#1f77b4",
        ecolor="#1f77b4",
        label=f"AMS-02 e+/(e++e-)  Phys.Rep. 894 (2021)  n={len(pf)}",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Kinetic energy  $E_k$  [GeV]")
    ax.set_ylabel(r"Positron fraction  $e^+/(e^+ + e^-)$")
    ax.set_title("AMS-02 positron fraction")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(
        0.98, 0.04,
        "TQNT-V5: excess attributed to pulsars,\n"
        "not DM annihilation (CS barrier k=135)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7d6", ec="#cc9900", alpha=0.9),
    )

    # Right panel : pbar/p
    ax = axes[1]
    sigma_tot = np.sqrt(pbarp["sigma_stat"] ** 2 + pbarp["sigma_syst"] ** 2)
    ax.errorbar(
        pbarp["R_GV"],
        pbarp["pbar_p_ratio"],
        yerr=sigma_tot,
        xerr=[pbarp["R_GV"] - pbarp["R_min_GV"], pbarp["R_max_GV"] - pbarp["R_GV"]],
        fmt="s",
        ms=4,
        lw=1,
        color="#d62728",
        ecolor="#d62728",
        label=f"AMS-02 pbar/p  PRL 117, 091103 (2016)  n={len(pbarp)}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Magnetic rigidity  $R$  [GV]")
    ax.set_ylabel(r"$\bar p / p$  flux ratio")
    ax.set_title("AMS-02 antiproton-to-proton flux ratio")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(
        0.04, 0.96,
        "TQNT-V5: no DM annihilation -> pbar/p\n"
        "must match secondary p-p only",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7d6", ec="#cc9900", alpha=0.9),
    )

    fig.suptitle(
        "AMS-02 cosmic-ray data vs TQNT-V5 (no DM annihilation channel)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Résumé
# ----------------------------------------------------------------------------
def summarize(pf: pd.DataFrame, pbarp: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== AMS-02 datasets summary ===")
    lines.append(
        f"  positron fraction     n_bins={len(pf):3d}   "
        f"E = {pf['E_GeV'].min():.3f} ... {pf['E_GeV'].max():.2f} GeV   "
        f"(bin edges {pf['E_min_GeV'].min():.3f} ... {pf['E_max_GeV'].max():.0f} GeV)"
    )
    lines.append(
        f"  source ref: {pf['ads_ref'].iloc[0]}  "
        "(AMS Coll., Phys. Rep. 894, 1-116, 2021)"
    )
    lines.append(
        f"  pbar / p ratio        n_bins={len(pbarp):3d}   "
        f"R = {pbarp['R_GV'].min():.2f} ... {pbarp['R_GV'].max():.1f} GV   "
        f"(bin edges {pbarp['R_min_GV'].min():.2f} ... {pbarp['R_max_GV'].max():.0f} GV)"
    )
    lines.append(
        f"  source ref: {pbarp['ads_ref'].iloc[0]}  "
        "(AMS Coll., PRL 117, 091103, 2016)"
    )
    lines.append("")
    lines.append("=== TQNT-V5 prediction ===")
    pred = v5_prediction()
    for k, v in pred.items():
        lines.append(f"  {k:30s} : {v}")
    lines.append("")
    lines.append(
        "  secondary_background_model(R) returns NaN -- requires external"
    )
    lines.append("  GALPROP / DRAGON transport prediction to finish the falsification test.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Entrée
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _ = argv  # placeholder
    pf = load_positron_fraction()
    pbarp = load_pbar_p()
    out_path = make_inspection_figure(pf, pbarp)
    print(summarize(pf, pbarp))
    print()
    print(f"Figure written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
