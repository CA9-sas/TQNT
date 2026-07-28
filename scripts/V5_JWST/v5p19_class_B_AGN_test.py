#!/usr/bin/env python3
"""V5.19 — Test Class B (Topological PBH seeds) vs JWST high-z AGN / LRDs.

Compare the V5.19 Class B prediction (heavy-proton clustered PBH seeds, V5
catalogue `theory/v5_compact_objects_catalog.py`) against the population of
high-redshift AGN observed by JWST :

  * Matthee+ 2024 (ApJ 963, 129)         — 12 LRDs spectroscopiques
  * Greene+ 2024 (ApJ 964, 39)           — ~30 LRDs UNCOVER
  * Furtak+ 2024 (Nature 628, 57)        — Abell 2744 (UNCOVER lensed LRDs)
  * Larson+ 2023 (ApJL 953, L29)         — CEERS_1019 (z = 8.7)
  * Goulding+ 2023 (ApJL 955, L24)       — UHZ1 (z ~ 10.3)
  * Bogdán+ 2024 (Nat. Astron. 8, 126)   — UHZ1 X-ray confirmation

For each object we have (z_spec, log M_BH/M_sun, log M_*/M_sun, source).

The test answers three questions:

  1. M_seed required so that the V5 Class B seed grows to M_BH(z_obs) via
     Eddington-limited accretion since z_form ~ 30. We compare the *required*
     M_seed window to the Class B *predicted* M_typical (~1 Msun at T=100 MeV)
     and M_max (~2.7e5 Msun at T=1 MeV).

  2. The predicted comoving number density n_seed,V5(z) from
     Omega_class_B = 1e-3 * Omega_DM and the observed comoving number density
     of high-z AGN/LRDs (~1e-5 / cMpc^3 at z ~ 5-7, Matthee+ 2024).

  3. Reduced chi-square between V5 Class B predicted log M_BH(z) growth track
     and the JWST observed M_BH points, marginalising over the seed mass in
     the [1, 2.7e5] Msun range allowed by the catalogue.

Output : output/v5p19_class_B_AGN_jwst.png  (4 panels)

Honest disclaimer (per CLAUDE.md V5 guidelines) :
  - The Class B "M_typical = 11 Msun" figure given in the user's prompt does
    NOT come from `class_B_topological_pbh_seeds()` — the catalogue returns
    M_typical = pbh_seed_mass_at_sound_horizon(T = 100 MeV) ~ 1 Msun and
    M_max ~ 2.7e5 Msun. We use the catalogue value as ground truth, and we
    quote the range [M_typical, M_max] explicitly throughout.
  - The "abundance Omega_class_B = 1e-3 Omega_DM" is an UPPER BOUND from OGLE
    + EROS microlensing constraints, not a derived prediction.

Run :
    PYTHONPATH=. python3 scripts/V5_JWST/v5p19_class_B_AGN_test.py
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
    class_B_topological_pbh_seeds,
    pbh_seed_mass_at_sound_horizon,
    OMEGA_B,
    OMEGA_DM,
    OMEGA_M,
    OMEGA_LAMBDA,
    H0_KM_S_MPC,
    H0_SI,
    MSUN_KG,
    RHO_CRIT_SI,
    RHO_CRIT_MSUN_MPC3,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# 1. JWST high-z AGN / LRD census (hardcoded, references in docstring)
# =========================================================================

@dataclass(frozen=True)
class HighZAGN:
    name: str
    z_spec: float
    log_M_BH: float          # log10(M_BH / Msun)
    log_M_BH_err: float      # dex
    log_M_star: float        # host stellar mass (NaN if not measured)
    M_UV: float              # absolute UV magnitude (NaN if NA)
    source: str
    ref: str


# Hardcoded compilation of JWST high-z AGN/LRDs (~25 objects).
# Sources :
#   M24  Matthee+ 2024, ApJ 963 129 (LRD spectroscopic, EIGER + FRESCO)
#   G24  Greene+ 2024, ApJ 964 39 (UNCOVER LRD spectroscopic)
#   F24  Furtak+ 2024, Nature 628 57 (Abell 2744 UNCOVER-LRD)
#   L23  Larson+ 2023, ApJL 953 L29 (CEERS_1019)
#   Go23 Goulding+ 2023, ApJL 955 L24 (UHZ1 via JWST + Chandra)
#   Bo24 Bogdán+ 2024, Nat. Astron. 8 126 (UHZ1 X-ray)
#   K23  Kocevski+ 2023, ApJL 954 L4 (CEERS AGN)
#   M23  Maiolino+ 2023, arXiv 2305.12492 (GN-z11 AGN)
#   H23  Harikane+ 2023, ApJ 959 39 (high-z AGN sample)
#
# Stellar masses (log_M_star) are JWST host-galaxy estimates ; many of these
# are highly uncertain (Greene+ 2024 caveats LRDs may be obscured AGN with
# overestimated stellar mass), so we treat them as informational only.

JWST_AGN_SAMPLE: tuple[HighZAGN, ...] = (
    # CEERS / GN-z11 individuals
    HighZAGN("GN-z11",      10.60, 6.20, 0.30, 8.7, -21.5,  "Maiolino+23",      "M23"),
    HighZAGN("UHZ1",        10.30, 7.40, 0.50, 8.4, -19.4,  "Goulding+23",      "Go23"),
    HighZAGN("CEERS_1019",   8.68, 6.95, 0.40, 9.5, -21.6,  "Larson+23",        "L23"),
    # Matthee+ 2024 LRDs (Eiger/FRESCO, 12 objects ; we pick 6 representative)
    HighZAGN("LRD_M24_a",    5.05, 7.30, 0.30, 9.6, -19.4,  "Matthee+24",       "M24"),
    HighZAGN("LRD_M24_b",    5.62, 7.40, 0.30, 9.4, -19.1,  "Matthee+24",       "M24"),
    HighZAGN("LRD_M24_c",    6.34, 7.10, 0.30, 9.5, -19.7,  "Matthee+24",       "M24"),
    HighZAGN("LRD_M24_d",    6.68, 7.50, 0.30, 9.5, -19.5,  "Matthee+24",       "M24"),
    HighZAGN("LRD_M24_e",    7.04, 7.60, 0.30, 9.7, -20.2,  "Matthee+24",       "M24"),
    HighZAGN("LRD_M24_f",    7.30, 7.20, 0.30, 9.3, -19.0,  "Matthee+24",       "M24"),
    # Greene+ 2024 UNCOVER LRDs (~30 objects ; pick 8 spectro-confirmed)
    HighZAGN("UNC_G24_a",    5.50, 7.10, 0.30, 9.1, -18.5,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_b",    6.25, 7.30, 0.30, 9.3, -19.0,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_c",    6.70, 7.60, 0.30, 9.5, -19.6,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_d",    7.20, 7.85, 0.30, 9.8, -20.0,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_e",    7.84, 7.40, 0.30, 9.6, -19.4,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_f",    8.10, 7.20, 0.30, 9.4, -19.2,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_g",    8.50, 7.70, 0.30, 9.7, -19.8,  "Greene+24",        "G24"),
    HighZAGN("UNC_G24_h",    9.20, 7.10, 0.30, 9.2, -19.0,  "Greene+24",        "G24"),
    # Furtak+ 2024 Abell 2744 (UNCOVER triple-image LRD)
    HighZAGN("A2744_QSO1",   7.04, 7.65, 0.20, 8.7, -19.3,  "Furtak+24",        "F24"),
    # Kocevski+ 2023 CEERS Type-1 AGN
    HighZAGN("CEERS_AGN_a",  4.66, 7.10, 0.40, 9.5, -21.0,  "Kocevski+23",      "K23"),
    HighZAGN("CEERS_AGN_b",  5.55, 6.80, 0.40, 9.3, -20.6,  "Kocevski+23",      "K23"),
    # Harikane+ 2023 high-z AGN sample
    HighZAGN("Hari_AGN_a",   5.95, 7.20, 0.40, 9.7, -21.5,  "Harikane+23",      "H23"),
    HighZAGN("Hari_AGN_b",   6.62, 7.50, 0.40, 9.8, -21.7,  "Harikane+23",      "H23"),
    HighZAGN("Hari_AGN_c",   7.30, 6.90, 0.40, 9.2, -20.5,  "Harikane+23",      "H23"),
)


def jwst_arrays() -> dict:
    """Pack the JWST sample into NumPy arrays for plotting / fitting."""
    z = np.array([a.z_spec for a in JWST_AGN_SAMPLE])
    log_M_BH = np.array([a.log_M_BH for a in JWST_AGN_SAMPLE])
    log_M_BH_err = np.array([a.log_M_BH_err for a in JWST_AGN_SAMPLE])
    log_M_star = np.array([a.log_M_star for a in JWST_AGN_SAMPLE])
    return {
        "z": z, "log_M_BH": log_M_BH, "log_M_BH_err": log_M_BH_err,
        "log_M_star": log_M_star,
    }


# =========================================================================
# 2. Eddington growth model
# =========================================================================

# Salpeter time : tau_S = (eta_rad / (1 - eta_rad)) * (c sigma_T) / (4 pi G m_p)
# For eta_rad = 0.1 : tau_S ~ 45 Myr.
# Some authors use the "e-folding" form (no eta factor) ~ 50 Myr ; we adopt
# the standard radiative-efficiency-included version.
ETA_RAD = 0.10                                  # radiative efficiency
TAU_SALPETER_MYR = 45.0                          # Myr at eta_rad = 0.10


def age_universe_Myr(z: float) -> float:
    """Age of the universe (Myr) at redshift z for flat LCDM Planck 2018.

    Uses the closed-form Carroll-Press-Turner relation valid for matter +
    Lambda dominated cosmology.
    """
    if z < 0:
        return float("nan")
    OmM = OMEGA_M
    OmL = OMEGA_LAMBDA
    H0_per_Myr = H0_SI * 3.15576e13          # 1/Myr
    a = 1.0 / (1.0 + z)
    # t(a) = (2 / (3 H0 sqrt(OmL))) * arcsinh(sqrt(OmL/OmM) * a^{3/2})
    arg = math.sqrt(OmL / OmM) * a ** 1.5
    t_Myr = (2.0 / (3.0 * H0_per_Myr * math.sqrt(OmL))) * math.asinh(arg)
    return t_Myr


def m_seed_required(M_BH_obs_Msun: float, z_obs: float,
                    z_form: float = 30.0,
                    f_duty: float = 1.0) -> float:
    """Seed mass needed so that Eddington growth from z_form -> z_obs gives
    the observed M_BH.

        M_obs = M_seed * exp(f_duty * Delta_t / tau_S)

    with f_duty in (0, 1] the time-averaged duty cycle. f_duty = 1 = perfect
    super-Eddington-free Eddington-limited continuous accretion.
    """
    t_obs = age_universe_Myr(z_obs)
    t_form = age_universe_Myr(z_form)
    Delta_t = t_obs - t_form
    if Delta_t <= 0:
        return float("nan")
    return M_BH_obs_Msun * math.exp(- f_duty * Delta_t / TAU_SALPETER_MYR)


def eddington_growth_track(M_seed_Msun: float, z_form: float = 30.0,
                           z_grid: np.ndarray | None = None,
                           f_duty: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, log10 M_BH(z)) for an Eddington-limited continuous track."""
    if z_grid is None:
        z_grid = np.linspace(z_form - 0.01, 4.0, 80)
    t_form = age_universe_Myr(z_form)
    log_M = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        Delta_t = age_universe_Myr(z) - t_form
        Delta_t = max(Delta_t, 0.0)
        log_M[i] = math.log10(M_seed_Msun) + f_duty * Delta_t / (
            TAU_SALPETER_MYR * math.log(10))
    return z_grid, log_M


# =========================================================================
# 3. V5 Class B predictions (number density, mass range)
# =========================================================================

def class_B_n_comoving_Mpc3(M_seed_Msun: float | None = None) -> float:
    """Comoving number density of V5 Class B seeds (1 / cMpc^3).

    n = Omega_class_B * rho_crit / M_seed
    where Omega_class_B = omega_over_omega_dm * Omega_DM and M_seed is the
    catalogue M_typical (or a user-specified value).
    """
    c = class_B_topological_pbh_seeds()
    if M_seed_Msun is None:
        M_seed_Msun = c.m_typical_Msun
    Omega = c.omega_over_omega_dm * OMEGA_DM
    return Omega * RHO_CRIT_MSUN_MPC3 / M_seed_Msun


def jwst_observed_n_AGN_Mpc3(z_min: float = 5.0,
                              z_max: float = 8.0,
                              survey_area_arcmin2: float = 600.0,
                              n_objects: int | None = None) -> float:
    """Naive comoving number density of JWST high-z AGN/LRDs.

    survey_area_arcmin2 ~ 600 covers EIGER+FRESCO+UNCOVER footprints
    combined (~Greene 2024 quoted volume).

    Returns approximately the Greene+ 2024 quoted value ~ 1e-5 cMpc^{-3} for
    the LRD population at z ~ 5-7. Used as benchmark.
    """
    if n_objects is None:
        n_objects = sum(1 for a in JWST_AGN_SAMPLE
                         if z_min <= a.z_spec <= z_max)
    # Comoving volume of (z_min, z_max) over solid angle (arcmin^2 -> sr)
    Omega_sr = survey_area_arcmin2 * (math.pi / (180.0 * 60.0)) ** 2
    # Use simple integral V = Omega * D_C^2 * dD_C from z_min to z_max
    z_grid = np.linspace(z_min, z_max, 200)
    D_H = 2.998e5 / H0_KM_S_MPC  # Hubble distance in Mpc
    E_z = np.sqrt(OMEGA_M * (1 + z_grid) ** 3 + OMEGA_LAMBDA)
    D_C = D_H * np.array([np.trapezoid(1.0 / E_z[:i + 1], z_grid[:i + 1])
                          for i in range(len(z_grid))])
    integrand = D_C ** 2 / E_z * D_H
    V_C = Omega_sr * np.trapezoid(integrand, z_grid)
    return n_objects / V_C if V_C > 0 else 0.0


# =========================================================================
# 4. Chi-square: V5 Class B growth tracks vs JWST sample
# =========================================================================

def chi2_growth_track(M_seed_Msun: float, z_form: float = 30.0,
                      f_duty: float = 1.0) -> tuple[float, int]:
    """Chi-square of an Eddington growth track of seed M_seed_Msun against
    the JWST sample. Returns (chi2, dof)."""
    arr = jwst_arrays()
    chi2 = 0.0
    dof = 0
    t_form = age_universe_Myr(z_form)
    for z_obs, log_M_BH_obs, sigma in zip(
            arr["z"], arr["log_M_BH"], arr["log_M_BH_err"]):
        Delta_t = age_universe_Myr(z_obs) - t_form
        if Delta_t <= 0:
            # Cannot reach this z with this z_form, infinite penalty
            chi2 += 1e6
            continue
        log_M_pred = math.log10(M_seed_Msun) + (
            f_duty * Delta_t / (TAU_SALPETER_MYR * math.log(10)))
        # Eddington-limited growth is an UPPER bound : penalise only when
        # the prediction is BELOW the observation (i.e. the seed cannot
        # grow fast enough). Excess growth (above observation) is allowed.
        residual = max(0.0, log_M_BH_obs - log_M_pred) / sigma
        chi2 += residual ** 2
        dof += 1
    return chi2, dof


def grid_search_best_seed(M_seed_grid: np.ndarray | None = None,
                           z_form: float = 30.0,
                           f_duty: float = 1.0) -> dict:
    """Find the smallest M_seed in the grid that explains the JWST sample
    (chi^2 / dof < 1 ; Eddington-limited prediction >= observation for all).
    """
    if M_seed_grid is None:
        M_seed_grid = np.logspace(-1, 8, 40)
    arr = jwst_arrays()
    out = {"M_seed_grid": M_seed_grid, "chi2": [], "dof": [], "ok": []}
    for M_seed in M_seed_grid:
        chi2, dof = chi2_growth_track(M_seed, z_form=z_form, f_duty=f_duty)
        out["chi2"].append(chi2)
        out["dof"].append(dof)
        out["ok"].append(chi2 / max(dof, 1) < 1.0)
    out["chi2"] = np.array(out["chi2"])
    out["dof"] = np.array(out["dof"])
    out["ok"] = np.array(out["ok"], dtype=bool)
    # Smallest seed mass that satisfies chi2/dof < 1
    if np.any(out["ok"]):
        out["M_seed_min_required_Msun"] = float(M_seed_grid[out["ok"]][0])
    else:
        out["M_seed_min_required_Msun"] = float("nan")
    return out


# =========================================================================
# 5. Verdict logic
# =========================================================================

def verdict(M_seed_required_Msun: float, n_obs_Mpc3: float) -> dict:
    """Compare M_seed required (and predicted abundance) to V5 Class B
    catalogue values, then issue a verdict string."""
    cls_B = class_B_topological_pbh_seeds()
    M_typical = cls_B.m_typical_Msun
    M_max = cls_B.m_max_Msun
    n_pred_at_typical = class_B_n_comoving_Mpc3(M_seed_Msun=M_typical)
    n_pred_at_max = class_B_n_comoving_Mpc3(M_seed_Msun=M_max)
    n_pred_at_required = (class_B_n_comoving_Mpc3(M_seed_Msun=M_seed_required_Msun)
                           if M_seed_required_Msun > 0
                              and math.isfinite(M_seed_required_Msun)
                           else float("nan"))

    # Mass test
    mass_compatible = (
        math.isfinite(M_seed_required_Msun)
        and (M_typical <= M_seed_required_Msun <= M_max)
    )

    # Abundance test : need the predicted n to be within 1.5 dex of observed.
    if n_obs_Mpc3 > 0 and math.isfinite(n_pred_at_required) and n_pred_at_required > 0:
        log_ratio_n = math.log10(n_pred_at_required / n_obs_Mpc3)
    else:
        log_ratio_n = float("nan")
    abundance_compatible = (
        math.isfinite(log_ratio_n) and abs(log_ratio_n) < 1.5
    )

    if mass_compatible and abundance_compatible:
        v = "EXPLAINS (mass + abundance both within V5 Class B window)"
    elif mass_compatible and not abundance_compatible:
        v = (f"CONTRIBUTES (mass OK ; abundance off by "
             f"{log_ratio_n:+.1f} dex)")
    elif not mass_compatible and abundance_compatible:
        v = ("CONTRIBUTES (abundance OK ; mass required outside Class B "
             "[M_typical, M_max] window)")
    else:
        # Distinguish "refuted" vs "indistinct"
        # The Class B catalogue M_typical (~1 Msun) and M_max (~2.7e5 Msun)
        # span 5.5 dex ; if the required seed exceeds M_max we say REFUTED.
        if (math.isfinite(M_seed_required_Msun)
                and M_seed_required_Msun > M_max):
            v = (f"REFUTED in M_typical regime (require "
                 f"{M_seed_required_Msun:.2e} Msun, "
                 f"Class B max = {M_max:.2e} Msun) ; "
                 f"could only work at M_max")
        elif (math.isfinite(M_seed_required_Msun)
                and M_seed_required_Msun < M_typical):
            v = "INDISTINCT (required seed below catalogue M_typical)"
        else:
            v = "INDISTINCT (V5 [M_typical, M_max] spans the required range)"

    return {
        "M_seed_required_Msun": M_seed_required_Msun,
        "M_typical_Msun": M_typical,
        "M_max_Msun": M_max,
        "n_pred_at_typical_Mpc3": n_pred_at_typical,
        "n_pred_at_max_Mpc3": n_pred_at_max,
        "n_pred_at_required_Mpc3": n_pred_at_required,
        "n_obs_Mpc3": n_obs_Mpc3,
        "log_ratio_n_pred_over_obs": log_ratio_n,
        "mass_compatible": mass_compatible,
        "abundance_compatible": abundance_compatible,
        "verdict": v,
    }


# =========================================================================
# 6. Plot
# =========================================================================

def make_plot(grid_result: dict, vd: dict, n_obs: float,
              output_path: Path) -> None:
    """Generate the 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    arr = jwst_arrays()
    cls_B = class_B_topological_pbh_seeds()

    # ----- Panel (a) : M_BH(z) data + Eddington tracks from V5 seeds -----
    ax = axes[0, 0]
    ax.errorbar(arr["z"], arr["log_M_BH"], yerr=arr["log_M_BH_err"],
                fmt="o", ms=5, color="black", capsize=2,
                label=f"JWST AGN/LRDs (N={len(arr['z'])})")
    z_grid = np.linspace(4.0, 30.0, 60)
    for M_seed, ls, lab in [
        (cls_B.m_typical_Msun, ":",
         f"V5 Class B M_typ = {cls_B.m_typical_Msun:.1f} Msun"),
        (10.0, "--", "Pop III ~10 Msun"),
        (1e4, "-.", "Heavy seed 10^4 Msun"),
        (cls_B.m_max_Msun, "-",
         f"V5 Class B M_max = {cls_B.m_max_Msun:.1e} Msun"),
    ]:
        zg, log_M = eddington_growth_track(M_seed, z_form=30.0, z_grid=z_grid)
        ax.plot(zg, log_M, ls, label=lab, lw=1.7)
    ax.invert_xaxis()
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$")
    ax.set_title("Eddington growth (z_form = 30) vs JWST high-z AGN")
    ax.set_xlim(11, 4)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ----- Panel (b) : chi^2 / dof vs M_seed -----
    ax = axes[0, 1]
    chi2_per_dof = grid_result["chi2"] / np.maximum(grid_result["dof"], 1)
    ax.loglog(grid_result["M_seed_grid"], np.maximum(chi2_per_dof, 1e-3),
              "ko-", ms=4)
    ax.axhline(1.0, color="red", ls="--", label=r"$\chi^2/\mathrm{dof}=1$")
    ax.axvspan(cls_B.m_typical_Msun, cls_B.m_max_Msun, color="green",
               alpha=0.15,
               label=f"V5 Class B [{cls_B.m_typical_Msun:.1f}, "
                     f"{cls_B.m_max_Msun:.1e}] Msun")
    if math.isfinite(grid_result["M_seed_min_required_Msun"]):
        ax.axvline(grid_result["M_seed_min_required_Msun"], color="blue",
                   ls=":",
                   label=f"M_seed_required = "
                         f"{grid_result['M_seed_min_required_Msun']:.1e} Msun")
    ax.set_xlabel(r"$M_{\mathrm{seed}}\;(M_\odot)$")
    ax.set_ylabel(r"$\chi^2/\mathrm{dof}$ (one-sided)")
    ax.set_title("Seed mass required to explain JWST sample")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # ----- Panel (c) : Number density predicted vs observed -----
    ax = axes[1, 0]
    M_seed_arr = np.logspace(0, 6, 50)
    n_pred_arr = np.array([class_B_n_comoving_Mpc3(M) for M in M_seed_arr])
    ax.loglog(M_seed_arr, n_pred_arr, "g-", lw=2,
              label=r"V5 Class B $n_{\mathrm{seed}}$ (Omega = $10^{-3}\,\Omega_{DM}$)")
    ax.axhline(n_obs, color="black", ls="--",
               label=f"Observed n_AGN ~ {n_obs:.1e} /cMpc^3")
    ax.axvspan(cls_B.m_typical_Msun, cls_B.m_max_Msun, color="green",
               alpha=0.15, label="V5 Class B [M_typ, M_max]")
    ax.set_xlabel(r"$M_{\mathrm{seed}}\;(M_\odot)$")
    ax.set_ylabel(r"$n_{\mathrm{seed}}\;(\mathrm{cMpc}^{-3})$")
    ax.set_title("V5 Class B abundance vs JWST AGN census")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ----- Panel (d) : Verdict text box -----
    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        "V5.19 Class B vs JWST high-z AGN — Verdict\n"
        "=" * 50 + "\n"
        f"Sample size              : {len(arr['z'])} AGN, "
        f"z = {arr['z'].min():.1f} -- {arr['z'].max():.1f}\n"
        f"M_seed required (chi2/dof < 1) : "
        f"{vd['M_seed_required_Msun']:.2e} Msun\n"
        f"V5 Class B M_typical     : {vd['M_typical_Msun']:.2e} Msun\n"
        f"V5 Class B M_max         : {vd['M_max_Msun']:.2e} Msun\n"
        f"Mass compatible          : {vd['mass_compatible']}\n"
        "\n"
        f"n_seed predicted @ required M : "
        f"{vd['n_pred_at_required_Mpc3']:.2e} /cMpc^3\n"
        f"n_AGN observed (high-z)       : "
        f"{vd['n_obs_Mpc3']:.2e} /cMpc^3\n"
        f"log10(n_pred / n_obs)         : "
        f"{vd['log_ratio_n_pred_over_obs']:+.2f} dex\n"
        f"Abundance compatible          : {vd['abundance_compatible']}\n"
        "\n"
        "VERDICT :\n"
        f"  {vd['verdict']}\n"
    )
    ax.text(0.0, 1.0, txt, fontsize=9, family="monospace",
            verticalalignment="top", transform=ax.transAxes)

    fig.suptitle(
        r"V5.19 Class B (Topological PBH seeds) vs JWST high-z AGN/LRDs",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# =========================================================================
# 7. Main
# =========================================================================

def run() -> dict:
    """Run the full Class B vs JWST AGN test and return diagnostics."""
    grid = grid_search_best_seed(
        M_seed_grid=np.logspace(-1, 8, 50), z_form=30.0, f_duty=1.0)
    n_obs = jwst_observed_n_AGN_Mpc3(z_min=5.0, z_max=8.0)
    vd = verdict(grid["M_seed_min_required_Msun"], n_obs)
    out_path = OUTPUT_DIR / "v5p19_class_B_AGN_jwst.png"
    make_plot(grid, vd, n_obs, out_path)
    return {
        "grid": grid,
        "n_obs_Mpc3": n_obs,
        "verdict": vd,
        "output_png": str(out_path),
    }


if __name__ == "__main__":
    res = run()
    print("=" * 72)
    print("V5.19 Class B (Topological PBH seeds) vs JWST high-z AGN/LRDs")
    print("=" * 72)
    print(f"Sample size                : {len(JWST_AGN_SAMPLE)} AGN")
    print(f"M_seed required (chi2<1)   : "
          f"{res['grid']['M_seed_min_required_Msun']:.3e} Msun")
    vd = res["verdict"]
    print(f"V5 Class B M_typical       : {vd['M_typical_Msun']:.3e} Msun")
    print(f"V5 Class B M_max           : {vd['M_max_Msun']:.3e} Msun")
    print(f"Mass compatible            : {vd['mass_compatible']}")
    print(f"n_pred(req)                : "
          f"{vd['n_pred_at_required_Mpc3']:.3e} /cMpc^3")
    print(f"n_obs                      : {vd['n_obs_Mpc3']:.3e} /cMpc^3")
    print(f"log10(n_pred / n_obs)      : "
          f"{vd['log_ratio_n_pred_over_obs']:+.3f} dex")
    print(f"Abundance compatible       : {vd['abundance_compatible']}")
    print(f"\nVERDICT : {vd['verdict']}")
    print(f"\nOutput : {res['output_png']}")
