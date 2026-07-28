#!/usr/bin/env python3
"""KATRIN m_beta endpoint loader and TQNT-V5 comparator.

Source: M. Aker et al. (KATRIN Collaboration), Science 388, 180-185 (2025);
arXiv:2406.13516. 259 measurement days, KNM1-5, 36 million signal electrons.

TQNT-V5 stance (memory `neutrinos-are-free-chi-waves.md`):
  - Neutrinos = free chi waves, NO topological knot.
  - Dirac character (not Majorana) -> 0nu beta-beta rate = 0 exactly.
  - Sum m_nu ~ 66 meV (close to DESI ~70 meV bound).
  - Beta-decay endpoint still observable (neutrinos carry energy & spin).

m_beta vs Sum m_nu:
  m_beta^2 = sum_i |U_ei|^2 * m_i^2  (incoherent sum, electron neutrino flavor).
  For normal hierarchy with light lightest mass, m_beta ~ 8-12 meV when
  Sum m_nu ~ 66 meV. Well below KATRIN final 0.45 eV limit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/mnt/data/data_recherche/katrin")
OUTPUT_DIR = Path("/home/euk9/projets/physique_nouvelle/output")

# Normal hierarchy PMNS-derived |U_ei|^2 approximations (PDG 2024).
# Used to convert Sum m_nu -> m_beta under a normal-hierarchy assumption.
UE1_SQ = 0.681   # |U_e1|^2
UE2_SQ = 0.297   # |U_e2|^2
UE3_SQ = 0.0222  # |U_e3|^2
# Mass-squared splittings (eV^2)
DM21_SQ = 7.42e-5
DM31_SQ = 2.51e-3


def load_katrin() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "katrin_2024.csv")
    return df


def _mbeta_from_sum(sum_mnu_eV: float) -> float:
    """Convert Sum m_nu (eV) -> m_beta (eV) assuming normal hierarchy.

    Solve for m1 from Sum m_nu = m1 + sqrt(m1^2 + dm21^2) + sqrt(m1^2 + dm31^2).
    """
    lo, hi = 0.0, sum_mnu_eV
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = (
            mid
            + np.sqrt(mid * mid + DM21_SQ)
            + np.sqrt(mid * mid + DM31_SQ)
        )
        if s > sum_mnu_eV:
            hi = mid
        else:
            lo = mid
    m1 = 0.5 * (lo + hi)
    m2 = np.sqrt(m1 * m1 + DM21_SQ)
    m3 = np.sqrt(m1 * m1 + DM31_SQ)
    mbeta_sq = UE1_SQ * m1 * m1 + UE2_SQ * m2 * m2 + UE3_SQ * m3 * m3
    return float(np.sqrt(mbeta_sq))


def tqnt_v5_prediction() -> dict:
    sum_mnu_meV = 66.0
    mbeta_eV = _mbeta_from_sum(sum_mnu_meV * 1e-3)
    return {
        "sigma_m_nu_meV": sum_mnu_meV,
        "is_Dirac": True,
        "beta_decay_observable": True,
        "neutrinoless_doublebeta_rate": 0.0,
        "m_beta_NH_eV": mbeta_eV,
        "m_beta_NH_meV": mbeta_eV * 1e3,
        "hierarchy_assumed": "normal",
        "notes": (
            "V5 neutrinos = free chi waves, no knot. Dirac character -> "
            "0nubetabeta = 0 exactly. m_beta derived from Sum m_nu = 66 meV "
            "under normal hierarchy and PDG PMNS mixing."
        ),
    }


def compare_v5_to_data(df: pd.DataFrame) -> dict:
    v5 = tqnt_v5_prediction()
    last = df[df["Run"] == "KNM1_5"].iloc[0]
    katrin_limit_eV = float(last["m_beta_eV_upper_limit_90CL"])
    katrin_final_sens_eV = float(
        df[df["Run"] == "KATRIN_final_design_sensitivity"].iloc[0][
            "m_beta_eV_upper_limit_90CL"
        ]
    )
    mbeta_v5_eV = v5["m_beta_NH_eV"]
    compatible = mbeta_v5_eV < katrin_limit_eV
    # "Decades to falsify": how many orders of magnitude below the limit V5 sits.
    decades_below_current = float(np.log10(katrin_limit_eV / mbeta_v5_eV))
    decades_below_design = float(np.log10(katrin_final_sens_eV / mbeta_v5_eV))

    return {
        "katrin_limit_eV_90CL": katrin_limit_eV,
        "katrin_design_sensitivity_eV": katrin_final_sens_eV,
        "V5_m_beta_eV": mbeta_v5_eV,
        "V5_sigma_m_nu_meV": v5["sigma_m_nu_meV"],
        "compatible_with_KATRIN_final": compatible,
        "decades_below_current_limit": decades_below_current,
        "decades_below_design_sensitivity": decades_below_design,
        "falsifiability": (
            "KATRIN cannot reach V5: m_beta ~ {:.1f} meV, design ~200 meV. "
            "Project 8 / cryo-bolometer mu-Hz tritium would be needed to "
            "test m_beta < 50 meV.".format(mbeta_v5_eV * 1e3)
        ),
    }


def _plot(df: pd.DataFrame, comp: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = [
        ("KATRIN KNM1 (2019)", "KNM1", "tab:gray"),
        ("KATRIN KNM1+2 (2022)", "KNM1_2", "tab:olive"),
        ("KATRIN KNM1-5 (2025)", "KNM1_5", "tab:blue"),
        ("KATRIN design (1000 d)", "KATRIN_final_design_sensitivity", "tab:cyan"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (label, key, color) in zip(ys, rows):
        r = df[df["Run"] == key].iloc[0]
        lim = r["m_beta_eV_upper_limit_90CL"]
        ax.plot([0, lim], [y, y], color=color, lw=3, label=label)
        ax.plot([lim], [y], "v", color=color)
        ax.text(lim, y + 0.15, f"  < {lim:.2f} eV (90% CL)", fontsize=9, color=color)

    v5 = comp["V5_m_beta_eV"]
    ax.axvline(v5, color="red", lw=2, ls="--",
               label=f"TQNT-V5 m_beta = {v5*1e3:.1f} meV (NH, Σmν=66 meV)")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 2.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(r"$m_\beta$ (eV)")
    title = (
        "KATRIN m_β endpoint vs TQNT-V5\n"
        f"V5 sits ~{comp['decades_below_current_limit']:.1f} decades below current limit, "
        f"~{comp['decades_below_design_sensitivity']:.1f} below design"
    )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3, which="both")
    plt.tight_layout()
    out = OUTPUT_DIR / "katrin_v5.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    df = load_katrin()
    comp = compare_v5_to_data(df)
    print("=" * 70)
    print("KATRIN m_beta endpoint 2025 (Science 388, 180) — TQNT-V5 status")
    print("=" * 70)
    print(f"KATRIN final limit (KNM1-5): m_beta < "
          f"{comp['katrin_limit_eV_90CL']:.2f} eV @ 90% CL")
    print(f"KATRIN design sensitivity     : m_beta < "
          f"{comp['katrin_design_sensitivity_eV']:.2f} eV")
    print()
    print(f"TQNT-V5 Sum m_nu             : {comp['V5_sigma_m_nu_meV']:.1f} meV")
    print(f"TQNT-V5 m_beta (NH derived)  : {comp['V5_m_beta_eV']*1e3:.1f} meV")
    print(f"Compatible with KATRIN final : {comp['compatible_with_KATRIN_final']}")
    print(f"Decades below current limit  : {comp['decades_below_current_limit']:.2f}")
    print(f"Decades below design sens.   : {comp['decades_below_design_sensitivity']:.2f}")
    print()
    print(f"Falsifiability: {comp['falsifiability']}")
    print()
    out = _plot(df, comp)
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
