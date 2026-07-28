"""PDG 2024 mass/width loader and TQNT-V5 lifetime comparator (stub).

Source : `mass_width_2024.mcd` (https://pdg.lbl.gov/2024/mcdata/).

Parse the fixed-width MCD format (FORTRAN
`(BN, 4I8, 2(1X,E18.0, 1X,E8.0, 1X,E8.0), 1X,A21)`) into a tidy
pandas.DataFrame, convert width Gamma -> lifetime tau = hbar / Gamma, and
expose hooks for a TQNT-V5 prediction.

Notes on TQNT-V5
----------------
The V5 lifetime law is topological:

    tau_V5 = tau_P * exp(-lambda * n_crossings(K)) / M_raw**nu

where K is the knot assigned to the particle (see
`docs/TQNT_detaillee_V5_FR.tex` §15, postulates P13/P15) and
n_crossings(K) is the crossing number of K. M_raw is the raw topological
mass m = alpha_0 * U_EM * c_J**3.55 (no plasticity correction).

The knot -> particle assignment is **not** yet wired in this script.
`predict_lifetime_V5` returns NaN with a warning until the assignment
table is connected.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

HBAR_GEV_S = 6.582_119_569e-25  # GeV . s, exact (CODATA 2018)

DEFAULT_MCD_PATH = "/mnt/data/data_recherche/pdg/2024/mass_width_2024.mcd"

# Fixed-width column slices (1-indexed in the header -> 0-indexed slices)
_COL_ID = [(0, 8), (8, 16), (16, 24), (24, 32)]
_COL_MASS = (33, 51)
_COL_MASS_ERRP = (52, 60)
_COL_MASS_ERRN = (61, 69)
_COL_WIDTH = (70, 88)
_COL_WIDTH_ERRP = (89, 97)
_COL_WIDTH_ERRN = (98, 106)
_COL_NAME = (107, 128)


# ----------------------------------------------------------------------------
# Category assignment by PDG MC ID
# ----------------------------------------------------------------------------

def _categorize(pdg_id: int, name: str) -> str:
    """Return one of: quark, lepton, boson, meson, baryon, exotic.

    Uses the PDG Monte Carlo numbering scheme (Workman et al. 2024, §44).
    """
    pid = abs(int(pdg_id))
    if pid == 0:
        return "exotic"
    # Quarks: 1..8 (u, d, s, c, b, t, b', t')
    if 1 <= pid <= 8:
        return "quark"
    # Leptons: 11..18
    if 11 <= pid <= 18:
        return "lepton"
    # Gauge / Higgs / SM-extension bosons: 21..37
    if pid in (21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37):
        return "boson"
    # PDG numbering for hadrons:
    #   - mesons: 3-digit, last digit = 2J+1 odd
    #   - baryons: 4-digit
    #   - 5/6/7-digit IDs prefix excited/exotic states
    # The last three digits encode quark content; baryons (4-digit) vs mesons
    # (3-digit) can be told by digit count of the *full* PDG ID modulo prefix.
    # Concretely: extract the "core" id (mod 10000) and look at its width.
    core = pid % 10000  # strip the radial/orbital excitation prefix
    if 100 <= core <= 999:
        return "meson"
    if 1000 <= core <= 9999:
        return "baryon"
    return "exotic"


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

def _parse_float(token: str) -> float:
    token = token.strip()
    if not token:
        return np.nan
    try:
        return float(token)
    except ValueError:
        return np.nan


def _parse_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _split_name_charges(field: str) -> tuple[str, list[str]]:
    """Split `name <whitespace> charge_states` from the trailing column.

    Charge states are right-justified, name is left-justified. Charge states
    are separated by commas (no space inside). The minimal heuristic: take
    the last whitespace-separated token as the charge list, and the rest as
    the name.
    """
    s = field.rstrip()
    if not s:
        return "", []
    parts = s.rsplit(maxsplit=1)
    if len(parts) == 1:
        # Only a name, no explicit charges (unusual)
        return parts[0].strip(), []
    name, charges = parts
    charges_list = [c.strip() for c in charges.split(",") if c.strip()]
    return name.strip(), charges_list


def load_pdg2024(path: str = DEFAULT_MCD_PATH) -> pd.DataFrame:
    """Load PDG 2024 mass_width MCD file into a tidy DataFrame.

    One row per (particle, charge state). Columns:

    pdg_id, name, charge, mass_MeV, mass_unc_low, mass_unc_high,
    width_MeV, width_unc, lifetime_s, lifetime_unc_low, lifetime_unc_high,
    decay_modes_count, category.

    `decay_modes_count` is not provided by the MCD file -> filled with 0.
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("*"):
                continue
            # Pad to at least 128 chars to safely slice
            line = raw.rstrip("\n")
            if len(line) < 128:
                line = line + " " * (128 - len(line))

            ids = [_parse_int(line[a:b]) for (a, b) in _COL_ID]
            ids = [i for i in ids if i is not None]

            mass_gev = _parse_float(line[_COL_MASS[0]:_COL_MASS[1]])
            mass_errp = _parse_float(line[_COL_MASS_ERRP[0]:_COL_MASS_ERRP[1]])
            mass_errn = _parse_float(line[_COL_MASS_ERRN[0]:_COL_MASS_ERRN[1]])
            width_gev = _parse_float(line[_COL_WIDTH[0]:_COL_WIDTH[1]])
            width_errp = _parse_float(line[_COL_WIDTH_ERRP[0]:_COL_WIDTH_ERRP[1]])
            width_errn = _parse_float(line[_COL_WIDTH_ERRN[0]:_COL_WIDTH_ERRN[1]])

            name, charges = _split_name_charges(line[_COL_NAME[0]:_COL_NAME[1]])

            # Lifetime tau = hbar / Gamma (only meaningful for Gamma > 0)
            if np.isfinite(width_gev) and width_gev > 0:
                lifetime_s = HBAR_GEV_S / width_gev
                # Asymmetric error propagation: dtau/dGamma = -hbar/Gamma^2
                # so a positive dGamma shrinks tau (lower error on tau) and
                # a negative dGamma lengthens it (upper error on tau).
                if np.isfinite(width_errp) and width_errp > 0:
                    lifetime_unc_low = HBAR_GEV_S * width_errp / width_gev**2
                else:
                    lifetime_unc_low = np.nan
                if np.isfinite(width_errn) and width_errn > 0:
                    lifetime_unc_high = HBAR_GEV_S * width_errn / width_gev**2
                else:
                    lifetime_unc_high = np.nan
            else:
                lifetime_s = np.nan
                lifetime_unc_low = np.nan
                lifetime_unc_high = np.nan

            # Iterate over charge states. If len(charges) != len(ids), zip
            # with the shorter one. Stable particles like 'gamma' may have
            # one ID and one charge "0".
            n_states = max(len(ids), len(charges), 1)
            for k in range(n_states):
                pid = ids[k] if k < len(ids) else (ids[0] if ids else None)
                ch = charges[k] if k < len(charges) else (
                    charges[0] if charges else ""
                )
                rows.append({
                    "pdg_id": pid,
                    "name": name,
                    "charge": ch,
                    "mass_MeV": mass_gev * 1e3 if np.isfinite(mass_gev) else np.nan,
                    # Uncertainties are stored as positive magnitudes; the
                    # MCD already encodes the sign in the column name
                    # (errp = upper, errn = lower).
                    "mass_unc_low": abs(mass_errn) * 1e3 if np.isfinite(mass_errn) else np.nan,
                    "mass_unc_high": abs(mass_errp) * 1e3 if np.isfinite(mass_errp) else np.nan,
                    "width_MeV": width_gev * 1e3 if np.isfinite(width_gev) else np.nan,
                    "width_unc": ((width_errp + width_errn) / 2.0) * 1e3
                                  if (np.isfinite(width_errp) and np.isfinite(width_errn))
                                  else np.nan,
                    "lifetime_s": lifetime_s,
                    "lifetime_unc_low": lifetime_unc_low,
                    "lifetime_unc_high": lifetime_unc_high,
                    "decay_modes_count": 0,  # not provided by MCD; fill from RPP API if needed
                })

    df = pd.DataFrame(rows)
    # Quantum numbers (J, P, I, G) are NOT provided in mass_width_2024.mcd;
    # they live in the full RPP database. We add NaN placeholders so callers
    # see the contract.
    for col in ("J", "P", "I", "G"):
        df[col] = np.nan
    df["category"] = [
        _categorize(pid if pid is not None else 0, name)
        for pid, name in zip(df["pdg_id"], df["name"])
    ]
    return df


# ----------------------------------------------------------------------------
# TQNT-V5 prediction stub
# ----------------------------------------------------------------------------

def _load_pdg_name_to_config() -> dict[str, str]:
    """Carte nom_PDG -> config_V5 depuis l'onglet 3 du catalogue V5."""
    import openpyxl
    xlsx = os.path.join(os.path.dirname(__file__), "..", "..",
                        "docs", "TQNT_catalog_V5_diffusion.xlsx")
    xlsx = os.path.abspath(xlsx)
    if not os.path.exists(xlsx):
        return {}
    try:
        wb = openpyxl.load_workbook(xlsx, data_only=True)
    except Exception:
        return {}
    if "3. Particules PDG" not in wb.sheetnames:
        return {}
    ws = wb["3. Particules PDG"]
    mapping = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] is None or row[3] is None:
            continue
        # Normaliser le nom PDG (enlever charge, etc., pour matcher PDG MCD)
        name = str(row[0]).strip()
        config = str(row[3]).strip()
        mapping[name] = config
        # Variantes ASCII pour matcher mass_width_2024.mcd
        for replaced, asci in [
            ("⁺", "+"), ("⁻", "-"), ("⁰", "0"), ("±", "±"),
            ("μ", "mu"), ("τ", "tau"), ("Λ", "Lambda"),
            ("Σ", "Sigma"), ("Ξ", "Xi"), ("Ω", "Omega"),
            ("γ", "gamma"), ("ν", "nu"), ("π", "pi"),
            ("ρ", "rho"), ("φ", "phi"), ("ω", "omega"),
            ("η", "eta"), ("ψ", "psi"), ("χ", "chi"),
            ("Υ", "Upsilon"),
        ]:
            name = name.replace(replaced, asci)
        mapping[name] = config
    return mapping


def predict_lifetime_V5(particle_row: pd.Series) -> float:
    """Predict TQNT-V5 lifetime (s) via Ω_ts catalogue lookup.

    Uses the V5 catalogue (onglet 2 "Catalogue unifié", colonne τ TQNT)
    via `theory.omega_ts.predict_lifetime_lookup`. For particles assigned
    in the catalogue, returns the calibrated τ_TQNT (seconds).

    Returns NaN if no assignment found.
    """
    try:
        from theory.omega_ts import predict_lifetime_lookup
    except ImportError:
        warnings.warn("theory.omega_ts not importable", stacklevel=2)
        return np.nan

    name = particle_row.get("name", "")
    mapping = _load_pdg_name_to_config_cached()
    if not mapping:
        return np.nan
    config = mapping.get(str(name).strip())
    if config is None:
        # Try without charge sign
        for suffix in ("+", "-", "0", "±"):
            stripped = str(name).strip().rstrip(suffix)
            config = mapping.get(stripped)
            if config is not None:
                break
    if config is None:
        return np.nan
    tau = predict_lifetime_lookup(config)
    return tau if tau is not None else np.nan


# Cache the mapping to avoid re-reading xlsx on every prediction
_NAME_TO_CFG_CACHE: dict[str, str] | None = None

def _load_pdg_name_to_config_cached() -> dict[str, str]:
    global _NAME_TO_CFG_CACHE
    if _NAME_TO_CFG_CACHE is None:
        _NAME_TO_CFG_CACHE = _load_pdg_name_to_config()
    return _NAME_TO_CFG_CACHE


# ----------------------------------------------------------------------------
# Chi^2 comparator
# ----------------------------------------------------------------------------

def compute_chi2(
    df: pd.DataFrame,
    prediction_fn: Callable[[pd.Series], float],
    sigma_log_floor: float = 0.05,
) -> dict:
    """Compare log10(tau_predicted) to log10(tau_measured) row by row.

    Returns a dict with keys:
    - n_used  : number of rows with finite measured + predicted lifetimes
    - chi2    : sum of ((log10 tau_pred - log10 tau_meas) / sigma_log)^2
    - chi2_ndf: chi2 / n_used (rough goodness-of-fit per particle)
    - residuals : np.ndarray of (log10 tau_pred - log10 tau_meas)
    - sigma_log : np.ndarray of per-row uncertainties in log10 space

    `sigma_log_floor` is a minimum uncertainty in log10 tau (default 0.05
    dex ~ 12 %) applied when the propagated PDG uncertainty is smaller or
    unavailable. This prevents a single ultra-precise measurement from
    swamping the global chi^2.
    """
    measured = df["lifetime_s"].to_numpy(dtype=float)
    predicted = np.array([prediction_fn(row) for _, row in df.iterrows()],
                         dtype=float)

    # Uncertainty in log10 tau : sigma_log ~ sigma_tau / (tau * ln 10)
    # Use the larger of the two asymmetric uncertainties.
    sigma_tau = np.fmax(
        df["lifetime_unc_low"].to_numpy(dtype=float),
        df["lifetime_unc_high"].to_numpy(dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_log = sigma_tau / (measured * np.log(10.0))
    sigma_log = np.where(np.isfinite(sigma_log) & (sigma_log > sigma_log_floor),
                         sigma_log, sigma_log_floor)

    ok = np.isfinite(measured) & (measured > 0) & np.isfinite(predicted) & (predicted > 0)
    if not ok.any():
        return {
            "n_used": 0,
            "chi2": np.nan,
            "chi2_ndf": np.nan,
            "residuals": np.array([]),
            "sigma_log": np.array([]),
        }

    residuals = np.log10(predicted[ok]) - np.log10(measured[ok])
    sig = sigma_log[ok]
    chi2 = float(np.sum((residuals / sig) ** 2))
    n = int(ok.sum())
    return {
        "n_used": n,
        "chi2": chi2,
        "chi2_ndf": chi2 / n,
        "residuals": residuals,
        "sigma_log": sig,
    }


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------

_CATEGORY_COLORS = {
    "lepton":  "#1f77b4",
    "boson":   "#d62728",
    "meson":   "#2ca02c",
    "baryon":  "#ff7f0e",
    "exotic":  "#9467bd",
}


def _plot_tau_vs_mass(df: pd.DataFrame, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["lifetime_s"] > 0) & (df["mass_MeV"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat, color in _CATEGORY_COLORS.items():
        s = sub[sub["category"] == cat]
        if s.empty:
            continue
        ax.scatter(
            s["mass_MeV"], s["lifetime_s"],
            s=24, c=color, alpha=0.75, edgecolors="black", linewidths=0.3,
            label=f"{cat} (n={len(s)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mass $m$ (MeV)")
    ax.set_ylabel(r"Lifetime $\tau = \hbar/\Gamma$ (s)")
    ax.set_title("PDG 2024 (Workman et al., Phys. Rev. D 110, 030001) — "
                 r"$\tau$ vs $m$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load PDG 2024 mass/width and report a tau-vs-m scatter."
    )
    parser.add_argument("--mcd", default=DEFAULT_MCD_PATH,
                        help="Path to mass_width_2024.mcd")
    parser.add_argument(
        "--plot",
        default="/home/euk9/projets/physique_nouvelle/output/pdg2024_tau_vs_M.png",
        help="Path for the scatter plot output (matplotlib Agg).",
    )
    args = parser.parse_args(argv)

    df = load_pdg2024(args.mcd)
    n_total = len(df)
    print(f"Loaded {n_total} (particle, charge-state) rows from {args.mcd}")

    # Category breakdown
    cat_counts = df["category"].value_counts().sort_index()
    print("\nBreakdown by category:")
    for cat, n in cat_counts.items():
        print(f"  {cat:8s} : {n}")

    # tau range
    tau = df["lifetime_s"].dropna()
    tau_pos = tau[tau > 0]
    if not tau_pos.empty:
        print(f"\nLifetime tau range : {tau_pos.min():.3e} s "
              f"to {tau_pos.max():.3e} s "
              f"(n = {len(tau_pos)} particles with finite Gamma)")
    else:
        print("\nNo finite-width particles in this dataset.")

    # Mass range
    m = df["mass_MeV"].dropna()
    m_pos = m[m > 0]
    if not m_pos.empty:
        print(f"Mass range          : {m_pos.min():.3e} MeV "
              f"to {m_pos.max():.3e} MeV")

    os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    _plot_tau_vs_mass(df, args.plot)
    print(f"\nScatter plot written to {args.plot}")

    # Demonstration: chi^2 against the V5 stub (will return n_used=0 since
    # predict_lifetime_V5 is a stub).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_chi2(df, predict_lifetime_V5)
    print(f"\ncompute_chi2 vs stub V5 prediction : n_used = {result['n_used']} "
          "(stub returns NaN -> expected zero usable rows)")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
