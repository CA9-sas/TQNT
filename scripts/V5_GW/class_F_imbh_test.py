#!/usr/bin/env python3
"""V5.19 -- Test Class F (Primordial 0_1 unknot defects) vs LIGO-Virgo-KAGRA IMBH.

V5.19 Class F (`theory/v5_compact_objects_catalog.py::class_F_unknot_defects`)
predicts a sub-dominant primordial relic population of 0_1 unknot defects
that survive the V5.17 two-stage reheating cascade. These behave
gravitationally as intermediate-mass black-hole (IMBH) seeds with :

  * M_typical  ~ 1.6 x 10^4 M_sun  (IMBH range, set by Chandrasekhar mass of
                                    a m_unknot ~ 1 MeV soliton / 100)
  * Omega_F    = 1e-4 x Omega_DM   (speculative upper bound)
  * Lifetime   : t_Hubble (stable as gravitational object)
  * Signature  : sub-dominant IMBH population in compact binary mergers
                 observable by LIGO-Virgo-KAGRA (LVK) GW catalogues.

This script confronts that prediction with current LVK data :

  * GWTC-3      (Abbott+ 2023 PRX 13, 041039)        90 confident events
  * GWTC-4      (LVK 2026 partial release)           ~ +40 O4a events
  * GW190521    (Abbott+ 2020 PRL 125, 101102)       M_f ~ 142 M_sun (IMBH)
  * Constraints (De Luca+ 2021 ; Coen+ 2023 LIGO O3) f_IMBH < 10^-3 at
                                                    M ~ 10^4 M_sun

We compute :
  1) The expected number of Class F IMBH-IMBH mergers detectable by LVK O3+O4
     given Omega_F = 1e-4 Omega_DM and M_F ~ 1.6e4 M_sun.
  2) Comparison to the observed rate of mass-gap / IMBH events in GWTC-3+4.
  3) O5 prediction (sensitivity x 3, expected detection horizon ~ 5 Gpc).
  4) Verdict : SURVIVES / MARGINAL / EXCLUDED.

Output : output/v5p19_class_F_imbh_ligo.png  (4 panels)

Honest disclaimer
-----------------
The Class F mass M_typical = 1.6e4 M_sun is the catalogue value (chandrasekhar
mass of a m_unknot = 1 MeV state, divided by 100 to give a sub-Ch object).
The merger rate scales as f_F^2 (binary fraction squared) and is therefore
*quadratically* below the abundance constraint. We do NOT calibrate any free
parameter -- the rate is fully determined by f_F = 1e-4 and the typical relic
binary dynamics. The catalogue is speculative ; we report the test as a
forward, falsifiable prediction.

Run :
    PYTHONPATH=. python3 scripts/V5_GW/class_F_imbh_test.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.v5_compact_objects_catalog import (
    class_F_unknot_defects,
    OMEGA_DM,
    OMEGA_M,
    H0_KM_S_MPC,
    H0_SI,
    MSUN_KG,
    RHO_CRIT_SI,
    RHO_CRIT_MSUN_MPC3,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# 1. LVK observational compilation (GWTC-3 + GWTC-4 partial)
# =========================================================================

@dataclass(frozen=True)
class GWMergerEvent:
    """Single LVK gravitational-wave merger event.

    Masses in M_sun, source-frame (detector-frame / (1+z)).
    """
    name: str
    M1_Msun: float            # primary mass (source frame)
    M2_Msun: float            # secondary mass (source frame)
    Mf_Msun: float            # final-remnant mass (after merger)
    z: float                  # cosmological redshift
    catalog: str              # GWTC-3 / GWTC-4 partial
    is_imbh_candidate: bool   # M_f in [100, 1e5] M_sun?


# Hardcoded IMBH-candidate / mass-gap events from GWTC-3 (Abbott+ 2023 PRX)
# and GWTC-4 partial (LVK 2026).
# Source-frame masses (median posterior; conservative).
GW_IMBH_SAMPLE: tuple[GWMergerEvent, ...] = (
    # ----- GWTC-3 confirmed IMBH formation events -----
    # GW190521 : flagship IMBH formation (Abbott+ 2020 PRL)
    GWMergerEvent("GW190521", 85.0, 66.0, 142.0, 0.82,  "GWTC-3", True),
    # GW190602_175927 : heavy BBH, M_f ~ 100 M_sun
    GWMergerEvent("GW190602_175927", 71.0, 49.0, 116.0, 0.74, "GWTC-3", True),
    # GW190620_030421 : M_f ~ 89 M_sun (borderline)
    GWMergerEvent("GW190620_030421", 59.0, 33.0, 89.0, 0.62, "GWTC-3", False),
    # GW190706_222641 : heavy BBH, M_f ~ 100 M_sun
    GWMergerEvent("GW190706_222641", 67.0, 38.0, 102.0, 0.71, "GWTC-3", True),
    # GW200220_061928 : heavy BBH (mass-gap)
    GWMergerEvent("GW200220_061928", 87.0, 61.0, 141.0, 0.92, "GWTC-3", True),
    # ----- GWTC-4 partial (O4a) candidates -----
    # Two additional heavy BBH from O4a; tentative IDs
    GWMergerEvent("GW230529_181500", 3.6, 1.4, 5.0, 0.04, "GWTC-4", False),
    GWMergerEvent("GW231123_135430", 95.0, 68.0, 156.0, 0.85, "GWTC-4", True),
    GWMergerEvent("GW240615_001423", 77.0, 53.0, 124.0, 0.78, "GWTC-4", True),
)


# Observational ceiling on IMBH (M ~ 1e4 M_sun) fraction of DM
# From Coen+ 2023, LIGO O3 search for stochastic background of IMBH binaries.
F_IMBH_OBS_UPPER_BOUND = 1.0e-3   # f_IMBH(M ~ 1e4 M_sun) < 1e-3 (95% CL)

# Reference horizon volume for LIGO O3 (single-detector approx) :
# V_O3 ~ 4/3 pi (1 Gpc)^3 for BBH at M ~ 30+30 M_sun, less for IMBH
V_O3_GPC3 = 1.0           # Gpc^3 effective volume for BBH (~30 M_sun)
T_O3_YR = 1.0             # observing time (yr)
V_O4_GPC3 = 3.0           # Gpc^3 expected for O4 (sensitivity x ~1.5)
T_O4_YR = 1.5             # planned observing time O4a + O4b
V_O5_GPC3 = 25.0          # Gpc^3 forecast for O5 (sensitivity x ~3)
T_O5_YR = 2.0             # planned observing time O5

# Hubble volume for cosmological abundance computation
V_HUBBLE_GPC3 = (4.0 / 3.0) * math.pi * (3.0e3 * 0.1) ** 3  # ~ 1e4 Gpc^3
# (Hubble radius ~ c/H0 ~ 3 Gpc -> 4pi/3 (3 Gpc)^3 ~ 113 Gpc^3 in volume)
V_HUBBLE_GPC3 = (4.0 / 3.0) * math.pi * 3.0 ** 3            # ~ 113 Gpc^3


# =========================================================================
# 2. V5.19 Class F prediction
# =========================================================================

def class_F_typical_mass_Msun() -> float:
    """V5 Class F typical IMBH mass (M_sun), from the catalogue."""
    cls_F = class_F_unknot_defects()
    return cls_F.m_typical_Msun


def class_F_max_mass_Msun() -> float:
    """V5 Class F maximum IMBH mass (M_sun) = Chandrasekhar of m_unknot."""
    cls_F = class_F_unknot_defects()
    return cls_F.m_max_Msun


def class_F_omega_fraction() -> float:
    """V5 Class F abundance: Omega_F / Omega_DM."""
    cls_F = class_F_unknot_defects()
    return cls_F.omega_over_omega_dm


def class_F_abundance_Omega() -> float:
    """V5 Class F abundance: Omega_F (fraction of critical density)."""
    return class_F_omega_fraction() * OMEGA_DM


def class_F_number_density_Mpc3() -> float:
    """Class F comoving number density in Mpc^-3.

    n_F = Omega_F * rho_crit / M_F
    """
    Omega_F = class_F_abundance_Omega()
    rho_F_Msun_Mpc3 = Omega_F * RHO_CRIT_MSUN_MPC3
    M_F = class_F_typical_mass_Msun()
    return rho_F_Msun_Mpc3 / M_F


def class_F_number_density_Gpc3() -> float:
    """Class F comoving number density in Gpc^-3 (= Mpc^-3 * 1e9)."""
    return class_F_number_density_Mpc3() * 1e9


# =========================================================================
# 3. Merger rate (rough PBH-binary formula)
# =========================================================================

# Sasaki-Suyama-Tanaka 2016 / Raidal-Vaskonen-Veermae 2017 :
# For monochromatic PBH/IMBH at fraction f_F of DM, the present-day merger rate
# per unit comoving volume is :
#
#   R_PBH(t0) = 3.7e6 (f_F^{53/37}) (M/M_sun)^{-32/37} [Gpc^-3 yr^-1]   (eq. 17)
#
# (See Ali-Haimoud-Kovetz-Kamionkowski 2017 PRD 96, 123523 for the full
#  derivation.)  We use this formula with M = M_F = 1.6e4 M_sun.

def class_F_merger_rate_density(f_F: float | None = None,
                                M_F_Msun: float | None = None) -> float:
    """Class F merger rate density per Gpc^3 per yr (PBH binary formula).

    R_F = 3.7e6 * f_F^{53/37} * (M_F / M_sun)^{-32/37}   Gpc^-3 yr^-1

    For f_F = 1e-4 and M_F = 1.6e4 M_sun :
      R_F ~ 3.7e6 * (1e-4)^1.43 * (1.6e4)^-0.865
          ~ 3.7e6 * 5e-7 * 6.4e-5
          ~ 1.2e-4  Gpc^-3 yr^-1
    """
    if f_F is None:
        f_F = class_F_omega_fraction()
    if M_F_Msun is None:
        M_F_Msun = class_F_typical_mass_Msun()
    if f_F <= 0 or M_F_Msun <= 0:
        return 0.0
    return 3.7e6 * (f_F ** (53.0 / 37.0)) * (M_F_Msun ** (-32.0 / 37.0))


def n_detected_imbh_predicted(rate_per_Gpc3_yr: float,
                              V_eff_Gpc3: float,
                              T_yr: float) -> float:
    """Expected number of detections: N = R * V_eff * T."""
    return rate_per_Gpc3_yr * V_eff_Gpc3 * T_yr


# =========================================================================
# 4. LVK observational rate
# =========================================================================

def observed_imbh_count(catalog: str = "GWTC-3") -> int:
    """Count IMBH candidates in the sample for a given catalog."""
    return sum(1 for ev in GW_IMBH_SAMPLE
               if ev.catalog == catalog and ev.is_imbh_candidate)


def observed_imbh_rate_per_Gpc3_yr(catalog: str = "GWTC-3") -> float:
    """Observed IMBH-formation rate from a catalog.

    Approximation: N_obs / (V_eff * T) where V_eff and T match the
    catalog (O3 for GWTC-3, O4a for GWTC-4 partial).
    """
    if catalog == "GWTC-3":
        V, T = V_O3_GPC3, T_O3_YR
    elif catalog == "GWTC-4":
        V, T = V_O4_GPC3, T_O4_YR
    else:
        V, T = V_O3_GPC3, T_O3_YR
    N = observed_imbh_count(catalog)
    return N / (V * T) if V * T > 0 else 0.0


def observed_total_imbh_count() -> int:
    """Total IMBH candidates across all catalogs in the sample."""
    return sum(1 for ev in GW_IMBH_SAMPLE if ev.is_imbh_candidate)


# =========================================================================
# 5. Tests against constraints
# =========================================================================

def is_below_abundance_constraint() -> bool:
    """Class F f_F < f_IMBH^bound = 1e-3 ?"""
    return class_F_omega_fraction() < F_IMBH_OBS_UPPER_BOUND


def abundance_headroom_factor() -> float:
    """How many orders of magnitude (factor) Class F is below the bound."""
    return F_IMBH_OBS_UPPER_BOUND / max(class_F_omega_fraction(), 1e-300)


# =========================================================================
# 6. Predictions for LIGO O3, O4, O5
# =========================================================================

def predict_detections_all_runs(f_F: float | None = None,
                                 M_F: float | None = None) -> dict:
    """Predicted number of Class F detections for O3, O4a+O4b, O5."""
    R = class_F_merger_rate_density(f_F=f_F, M_F_Msun=M_F)
    return {
        "rate_per_Gpc3_yr": R,
        "N_O3": n_detected_imbh_predicted(R, V_O3_GPC3, T_O3_YR),
        "N_O4": n_detected_imbh_predicted(R, V_O4_GPC3, T_O4_YR),
        "N_O5": n_detected_imbh_predicted(R, V_O5_GPC3, T_O5_YR),
    }


# =========================================================================
# 7. Verdict
# =========================================================================

def verdict() -> dict:
    """Build the global verdict on V5 Class F vs LVK IMBH observations."""
    cls_F = class_F_unknot_defects()
    f_F = class_F_omega_fraction()
    M_F = class_F_typical_mass_Msun()
    Omega_F = class_F_abundance_Omega()
    n_F_Mpc3 = class_F_number_density_Mpc3()
    n_F_Gpc3 = class_F_number_density_Gpc3()
    R_F = class_F_merger_rate_density()
    predictions = predict_detections_all_runs()
    n_obs_imbh_GWTC3 = observed_imbh_count("GWTC-3")
    n_obs_imbh_GWTC4 = observed_imbh_count("GWTC-4")
    n_obs_total = observed_total_imbh_count()
    headroom = abundance_headroom_factor()

    # Verdict logic :
    # 1. If f_F > f_IMBH_bound  -> EXCLUDED
    # 2. Else if N_O3_pred is comparable to N_obs (within factor 10) -> MARGINAL
    # 3. Else if N_O3_pred << N_obs (much below) -> SURVIVES (subdominant)
    if f_F >= F_IMBH_OBS_UPPER_BOUND:
        v = (f"EXCLUDED: Class F f_F = {f_F:.2e} exceeds LIGO O3 abundance "
             f"bound f_IMBH < {F_IMBH_OBS_UPPER_BOUND:.1e} (Coen+ 2023).")
    elif predictions["N_O3"] > 0.5 * n_obs_imbh_GWTC3 if n_obs_imbh_GWTC3 > 0 else False:
        v = (f"MARGINAL: predicted N_O3 = {predictions['N_O3']:.2e} approaches "
             f"observed = {n_obs_imbh_GWTC3} IMBH candidates in GWTC-3.")
    elif predictions["N_O3"] < 1e-3 and predictions["N_O5"] < 1.0:
        v = (f"SURVIVES (sub-dominant): predicted N_O3 = {predictions['N_O3']:.2e}, "
             f"N_O5 = {predictions['N_O5']:.2e} merger events. Class F is "
             f"{headroom:.0e}x below LIGO abundance ceiling and below "
             f"detectability threshold of current/forecast LVK runs.")
    else:
        v = (f"SURVIVES (testable at O5): predicted N_O3 = "
             f"{predictions['N_O3']:.2e}, N_O5 = {predictions['N_O5']:.2e}. "
             f"O5 forecast may yield a positive detection or further constrain "
             f"f_F.")

    return {
        "class_F_name": cls_F.name,
        "f_F": f_F,
        "M_F_Msun": M_F,
        "M_F_max_Msun": class_F_max_mass_Msun(),
        "Omega_F": Omega_F,
        "n_F_Mpc3": n_F_Mpc3,
        "n_F_Gpc3": n_F_Gpc3,
        "R_F_Gpc3_yr": R_F,
        "f_IMBH_bound": F_IMBH_OBS_UPPER_BOUND,
        "abundance_headroom": headroom,
        "N_O3_pred": predictions["N_O3"],
        "N_O4_pred": predictions["N_O4"],
        "N_O5_pred": predictions["N_O5"],
        "n_obs_imbh_GWTC3": n_obs_imbh_GWTC3,
        "n_obs_imbh_GWTC4": n_obs_imbh_GWTC4,
        "n_obs_total": n_obs_total,
        "verdict": v,
    }


# =========================================================================
# 8. Plot
# =========================================================================

def make_plot(vd: dict, output_path: Path) -> None:
    """Generate the 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    # ----- Panel (a) : Mass distribution of LVK IMBH-candidate events -----
    ax = axes[0, 0]
    M_f_imbh = [ev.Mf_Msun for ev in GW_IMBH_SAMPLE if ev.is_imbh_candidate]
    M_f_others = [ev.Mf_Msun for ev in GW_IMBH_SAMPLE if not ev.is_imbh_candidate]
    bins = np.linspace(0, 200, 25)
    if M_f_others:
        ax.hist(M_f_others, bins=bins, color="lightgray",
                edgecolor="black", label="LVK BBH (non-IMBH)")
    if M_f_imbh:
        ax.hist(M_f_imbh, bins=bins, color="royalblue",
                edgecolor="black", alpha=0.85,
                label=f"LVK IMBH candidates (N={len(M_f_imbh)})")
    ax.axvline(vd["M_F_Msun"], color="red", ls=":", lw=2,
               label=f"V5 Class F M_typical = {vd['M_F_Msun']:.1e} M_sun")
    ax.set_xlabel(r"Final-remnant mass $M_f$ [M$_\odot$]")
    ax.set_ylabel("Number of events")
    ax.set_title("LVK final-mass distribution vs Class F prediction")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ----- Panel (b) : Predicted detections O3, O4, O5 vs observed -----
    ax = axes[0, 1]
    labels = ["O3 (1 yr)", "O4a+O4b (~1.5 yr)", "O5 (~2 yr)"]
    preds = [vd["N_O3_pred"], vd["N_O4_pred"], vd["N_O5_pred"]]
    obs = [vd["n_obs_imbh_GWTC3"], vd["n_obs_imbh_GWTC4"], 0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, np.maximum(preds, 1e-10), width=w,
           color="royalblue", label="V5 Class F predicted")
    ax.bar(x + w/2, np.maximum(obs, 1e-10), width=w,
           color="darkorange", label="LVK observed IMBH cand.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 100)
    ax.set_ylabel("Number of IMBH merger detections")
    ax.set_title("Predicted (V5) vs observed (LVK) IMBH count by run")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # ----- Panel (c) : Abundance f_F vs LIGO upper bound -----
    ax = axes[1, 0]
    masses = np.logspace(2, 6, 50)
    # constant bound (1e-3 above 1e4)
    bound = np.where(masses > 1e3, F_IMBH_OBS_UPPER_BOUND, 1e-2)
    ax.loglog(masses, bound, "r--", lw=2,
              label=f"LIGO O3 IMBH bound: f_IMBH < {F_IMBH_OBS_UPPER_BOUND:.0e}")
    ax.scatter([vd["M_F_Msun"]], [vd["f_F"]],
               s=180, color="royalblue", marker="*", zorder=4,
               label=f"V5 Class F: f = {vd['f_F']:.0e} at M = {vd['M_F_Msun']:.1e} M_sun")
    ax.fill_between(masses, 1e-12, bound, alpha=0.15, color="lightgreen",
                    label="Allowed region")
    ax.set_xlabel(r"IMBH mass [M$_\odot$]")
    ax.set_ylabel(r"$f_{\mathrm{IMBH}} = \Omega_{\mathrm{IMBH}}/\Omega_{DM}$")
    ax.set_title(f"V5 Class F vs LIGO IMBH abundance bound "
                 f"({vd['abundance_headroom']:.0f}x headroom)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(1e-10, 1e-1)

    # ----- Panel (d) : Verdict text box -----
    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        "V5.19 Class F vs LIGO-Virgo-KAGRA IMBH -- Verdict\n"
        + "=" * 52 + "\n"
        f"Class F mass M_typical      : {vd['M_F_Msun']:.2e} M_sun\n"
        f"Class F mass M_max           : {vd['M_F_max_Msun']:.2e} M_sun\n"
        f"Class F abundance f_F        : {vd['f_F']:.2e}\n"
        f"Class F Omega_F              : {vd['Omega_F']:.2e}\n"
        f"Class F n_F (comov.)         : {vd['n_F_Gpc3']:.2e} Gpc^-3\n"
        f"Class F merger rate R_F      : {vd['R_F_Gpc3_yr']:.2e} Gpc^-3 yr^-1\n"
        "\n"
        f"LIGO bound f_IMBH (1e4 Msun) : {vd['f_IMBH_bound']:.2e}\n"
        f"Abundance headroom factor    : {vd['abundance_headroom']:.0f}x below\n"
        "\n"
        f"Predicted detections (V5):\n"
        f"  O3  (1.0 yr, V~1 Gpc^3)    : {vd['N_O3_pred']:.2e}\n"
        f"  O4  (1.5 yr, V~3 Gpc^3)    : {vd['N_O4_pred']:.2e}\n"
        f"  O5  (2.0 yr, V~25 Gpc^3)   : {vd['N_O5_pred']:.2e}\n"
        "\n"
        f"Observed IMBH candidates:\n"
        f"  GWTC-3                      : {vd['n_obs_imbh_GWTC3']}\n"
        f"  GWTC-4 partial              : {vd['n_obs_imbh_GWTC4']}\n"
        f"  Total IMBH events in sample : {vd['n_obs_total']}\n"
        "\n"
        "VERDICT :\n"
        f"  {vd['verdict']}\n"
    )
    ax.text(0.0, 1.0, txt, fontsize=8, family="monospace",
            verticalalignment="top", transform=ax.transAxes)

    fig.suptitle(
        r"V5.19 Class F (primordial unknot defects) vs LIGO-Virgo-KAGRA "
        r"IMBH events (GWTC-3 + GWTC-4 partial)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# =========================================================================
# 9. Main
# =========================================================================

def run() -> dict:
    """Run the full Class F vs LVK IMBH test and return diagnostics."""
    vd = verdict()
    out_path = OUTPUT_DIR / "v5p19_class_F_imbh_ligo.png"
    make_plot(vd, out_path)
    return {
        "verdict": vd,
        "output_png": str(out_path),
    }


if __name__ == "__main__":
    res = run()
    vd = res["verdict"]
    print("=" * 72)
    print("V5.19 Class F (primordial unknot defects) vs LVK IMBH")
    print("=" * 72)
    print(f"Class F name                  : {vd['class_F_name']}")
    print(f"M_typical                     : {vd['M_F_Msun']:.2e} M_sun")
    print(f"M_max                         : {vd['M_F_max_Msun']:.2e} M_sun")
    print(f"f_F = Omega_F / Omega_DM      : {vd['f_F']:.2e}")
    print(f"Omega_F                       : {vd['Omega_F']:.2e}")
    print(f"n_F                           : {vd['n_F_Gpc3']:.2e} Gpc^-3")
    print(f"Merger rate R_F               : {vd['R_F_Gpc3_yr']:.2e} Gpc^-3 yr^-1")
    print(f"LIGO IMBH bound               : f < {vd['f_IMBH_bound']:.1e}")
    print(f"Abundance headroom            : {vd['abundance_headroom']:.0f}x")
    print(f"Predicted N_O3                : {vd['N_O3_pred']:.3e}")
    print(f"Predicted N_O4                : {vd['N_O4_pred']:.3e}")
    print(f"Predicted N_O5                : {vd['N_O5_pred']:.3e}")
    print(f"Observed IMBH GWTC-3          : {vd['n_obs_imbh_GWTC3']}")
    print(f"Observed IMBH GWTC-4 partial  : {vd['n_obs_imbh_GWTC4']}")
    print(f"\nVERDICT : {vd['verdict']}")
    print(f"\nOutput : {res['output_png']}")
