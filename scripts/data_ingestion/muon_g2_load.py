#!/usr/bin/env python3
"""Muon g-2 (Fermilab E989) data loader and TQNT-V5 comparator.

Source: D. P. Aguillard et al. (Muon g-2 Collaboration), Phys. Rev. Lett. 135,
245001 (2025); arXiv:2506.03069.

Theory reference: Muon g-2 Theory Initiative White Paper 2025 (BMW-lattice
dominated HVP). The data-driven e+e- evaluation is currently inconsistent
across experiments (CMD-3 vs KLOE/BaBar) and is not used as the official SM.

TQNT-V5 stance:
  - g-2 is unambiguously a "Case B" observable (B_ext = 1.45 T storage field).
  - The prediction requires Omega_ts(muon) AND a P23 plasticity correction
    eps(n=1) * g(B). Neither is numerically calibrated yet, so we return None
    with an explanatory message rather than fabricate a number.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/mnt/data/data_recherche/muon_g2")
OUTPUT_DIR = Path("/home/euk9/projets/physique_nouvelle/output")


def load_e989() -> pd.DataFrame:
    """Load the E989 measurement table (BNL, FNAL Runs, world avg, SM)."""
    df = pd.read_csv(DATA_DIR / "e989_final.csv")
    return df


def tqnt_v5_prediction() -> dict:
    """V5 prediction for a_mu.

    Returns dict with optional numeric value + diagnostic. We intentionally
    return None for the value until Omega_ts(muon) and the P23 broken-symmetry
    correction under B_ext = 1.45 T are calibrated.
    """
    return {
        "a_mu_V5_case_B_P23": None,
        "B_ext_storage_T": 1.45,
        "case": "B (broken-symmetry under B_ext)",
        "needs": [
            "Omega_ts(muon) numerical calibration",
            "P23 eps(n=1) * g(B=1.45 T) correction function",
        ],
        "target": "BMW-lattice consistent SM (~116592033e-11)",
        "reason": (
            "V5 prescribes Case B with P23 plasticity under the 1.45 T storage "
            "field. Neither Omega_ts(mu) nor eps(n=1)*g(B) is calibrated yet, "
            "so no point estimate can be reported honestly. V5 EXPECTS a value "
            "consistent with BMW-lattice and therefore with E989 final."
        ),
    }


def compare_v5_to_data(df: pd.DataFrame) -> dict:
    """Compute tensions: experiment vs BMW-lattice SM, vs old data-driven SM.

    Returns dict with tension in sigma for each theory baseline plus the V5
    qualitative stance.
    """
    exp = df[df["Run"] == "WorldAvg_BNL_FNAL"].iloc[0]
    bmw = df[df["Run"] == "SM_TI2025_lattice"].iloc[0]
    eedat = df[df["Run"] == "SM_eeData_2020"].iloc[0]

    def tension(a_exp, sig_exp, a_th, sig_th):
        delta = a_exp - a_th
        sigma = math.sqrt(sig_exp**2 + sig_th**2)
        return delta, sigma, delta / sigma

    d_bmw, s_bmw, n_bmw = tension(
        exp["a_mu_value_x1e11"],
        exp["a_mu_unc_total_x1e11"],
        bmw["a_mu_value_x1e11"],
        bmw["a_mu_unc_total_x1e11"],
    )
    d_ee, s_ee, n_ee = tension(
        exp["a_mu_value_x1e11"],
        exp["a_mu_unc_total_x1e11"],
        eedat["a_mu_value_x1e11"],
        eedat["a_mu_unc_total_x1e11"],
    )

    v5 = tqnt_v5_prediction()
    return {
        "experiment_a_mu_x1e11": float(exp["a_mu_value_x1e11"]),
        "experiment_unc_x1e11": float(exp["a_mu_unc_total_x1e11"]),
        "vs_BMW_lattice": {
            "delta_x1e11": d_bmw,
            "combined_sigma_x1e11": s_bmw,
            "tension_nsigma": n_bmw,
        },
        "vs_data_driven_ee_2020": {
            "delta_x1e11": d_ee,
            "combined_sigma_x1e11": s_ee,
            "tension_nsigma": n_ee,
        },
        "V5_prediction": v5,
        "V5_target_alignment": "BMW-lattice",
        "V5_status_vs_experiment": (
            "Indeterminate numerically — V5 lacks calibration for "
            "Omega_ts(mu) and P23 in B_ext. Qualitatively COMPATIBLE since "
            "V5 targets BMW-lattice baseline which itself is ~0.6 sigma."
        ),
    }


def _plot(df: pd.DataFrame, comp: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = [
        ("BNL E821 (2006)", "BNL_E821", "tab:gray"),
        ("FNAL Run-1 (2021)", "FNAL_Run1", "tab:olive"),
        ("FNAL Run-1+2+3 (2023)", "FNAL_Run1_2_3", "tab:cyan"),
        ("FNAL Run-4+5+6 (2025)", "FNAL_Run4_5_6", "tab:blue"),
        ("FNAL combined (2025)", "FNAL_combined", "tab:blue"),
        ("World avg BNL+FNAL", "WorldAvg_BNL_FNAL", "black"),
        ("SM Theory Init. 2025 (BMW)", "SM_TI2025_lattice", "tab:green"),
        ("SM data-driven e+e- (2020)", "SM_eeData_2020", "tab:red"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (label, key, color) in zip(ys, rows):
        r = df[df["Run"] == key].iloc[0]
        x = r["a_mu_value_x1e11"]
        e = r["a_mu_unc_total_x1e11"]
        ax.errorbar([x], [y], xerr=[e], fmt="o", color=color, capsize=4, label=label)
        ax.text(x, y + 0.2, f"  {x:.1f} ± {e:.1f}", fontsize=8, color=color)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(r"$a_\mu \times 10^{11}$")
    ax.set_title(
        "Muon g-2: E989 final 2025 vs SM (TQNT-V5 targets BMW-lattice)\n"
        f"Exp vs BMW: {comp['vs_BMW_lattice']['tension_nsigma']:+.2f} σ   |   "
        f"Exp vs e+e- 2020: {comp['vs_data_driven_ee_2020']['tension_nsigma']:+.2f} σ"
    )
    ax.axvline(comp["experiment_a_mu_x1e11"], color="black", lw=0.5, ls="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "muon_g2_v5.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    df = load_e989()
    comp = compare_v5_to_data(df)

    print("=" * 70)
    print("Muon g-2 / E989 final 2025 — TQNT-V5 status")
    print("=" * 70)
    print(f"World average a_mu = {comp['experiment_a_mu_x1e11']:.1f} ± "
          f"{comp['experiment_unc_x1e11']:.1f}  (× 1e-11)")
    print()
    print(f"vs BMW-lattice SM 2025  : Δ = "
          f"{comp['vs_BMW_lattice']['delta_x1e11']:+.1f} × 1e-11, "
          f"tension = {comp['vs_BMW_lattice']['tension_nsigma']:+.2f} σ")
    print(f"vs e+e- data-driven 2020 : Δ = "
          f"{comp['vs_data_driven_ee_2020']['delta_x1e11']:+.1f} × 1e-11, "
          f"tension = {comp['vs_data_driven_ee_2020']['tension_nsigma']:+.2f} σ")
    print()
    print("TQNT-V5 prediction:")
    for k, v in comp["V5_prediction"].items():
        print(f"  {k}: {v}")
    print()
    print("V5 status vs experiment:")
    print(f"  {comp['V5_status_vs_experiment']}")
    print()
    out = _plot(df, comp)
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
