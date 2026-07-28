#!/usr/bin/env python3
"""V5.20 Class E -- Prediction of 6 discrete gamma-ray lines vs Fermi-LAT + CTA.

Cadre
-----
V5.20 Class E (commit 0208e72) predit 6 raies gamma discretes aux masses
des 6 STRICT amphichiraux du catalogue V5 (annihilation pearl-antipearl
ou desexcitation topologique : E_gamma = m_chi).

    Composition       m_chi (GeV)   n_brick   delta_c_E
    --------------    -----------   -------   ---------
    6_3+6_3              1.55         2          4
    6_3+6_3+6_3          4.70         3          6
    4_1+8_3             16.01         2          4
    8_3+8_3             17.24         2          4
    6_3+8_3             26.48         2          4
    4_1+4_1+8_3         51.78         3          6

Objectifs (analyse, pas de nouvelle publication) :
  1. Position energetique (connue : masses ci-dessus)
  2. Intensite relative phi_i = f(Omega_i, sigma_i)
  3. Largeur de raie naturelle Delta_E / E (vs largeur instrumentale)
  4. Comparaison aux Fermi-LAT line search limits (Ackermann+ 2015)
  5. Prevision CTA sensibilite (2027+)

Donnees observationnelles
--------------------------
  * Fermi-LAT line search (Ackermann+ 2015, PRD 91, 122002) :
      - GC R3 region, ~5.8 yr exposure
      - Limits phi_gamma_gamma ~ 1e-10 ph cm^-2 s^-1 pour M_DM ~ 10-300 GeV
      - Plus stringent vers 50-100 GeV : ~3-5 x 10^-11 ph cm^-2 s^-1
  * HESS-II line search a TeV (Abdalla+ 2018) : ~1e-12 ph cm^-2 s^-1
  * CTA forecast (Acharyya+ 2021, JCAP 01, 057) :
      - Sensibilite line ~ 1e-12 a 1 TeV (galactic halo, 500h)
      - x10 mieux que HESS dans 25 GeV - 100 TeV
      - Pour 10-100 GeV : ~1e-11 ph cm^-2 s^-1

Sous-condition Delta_c_min >= 22 (V5.21 test parallele) : on calcule les
deux scenarios (V5.20 baseline avec delta_c_E ~ 4-6 ET V5.21 stricte
delta_c >= 22).

Run :
    PYTHONPATH=. python3 scripts/V5_Fermi_LAT/class_E_six_lines_predictions.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (
    ALPHA_EM,
    DM_STRICT_AMPHI_GEV,
    GEV_KG,
    M_PEARL_TYPICAL_MSUN,
    M_SUN_KG,
    R_HUBBLE_M,
    SIGMA_QED,
    T_HUBBLE_S,
    V_HUBBLE_M3,
    n_events_over_universe_age,
    sigma_entanglement,
)

import class_E_entanglement_test as CE

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_class_E_six_lines.png"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class E abundance: 10% of Omega_DM, spread equally over 6 amphichirals
OMEGA_DM = 0.265
OMEGA_E_FRACTION = 0.10           # of Omega_DM
OMEGA_PER_AMPHI = OMEGA_E_FRACTION / 6.0  # equal split (Ω_i = 1/6 * Ω_E)

# Class A reference Delta c (V5.20 baseline)
DELTA_C_CLASS_A = 10

# V5.21 candidate selection rule : Delta c >= 22
DELTA_C_V5P21_MIN = 22

# Cosmological constants
H0_SI = 67.4e3 / 3.0857e22        # 1/s
G_NEWTON = 6.67430e-11
RHO_CRIT = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_NEWTON)  # kg/m^3

# Instrumental resolution (FWHM / E)
FERMI_DE_OVER_E = 0.12            # ~12% at 1-100 GeV (LAT PSF/eres)
CTA_DE_OVER_E = 0.07              # ~7% (CTA design goal)

# Natural width of the V5 line (topological transition: ~0)
# We model a delta function; instrumental resolution dominates.

# Fermi-LAT line search differential limits (Ackermann+ 2015)
# Approximate (E_GeV, phi_limit ph cm^-2 s^-1) digitised from Fig.4 R3 NFW
# (limit on gamma+gamma line flux from GC region)
FERMI_LINE_LIMITS_R3 = [
    (1.0,   3.0e-9),    # extrapolation low
    (3.0,   8.0e-10),
    (10.0,  1.5e-10),
    (30.0,  4.0e-11),
    (50.0,  3.0e-11),
    (100.0, 3.5e-11),
    (200.0, 5.0e-11),
    (300.0, 8.0e-11),
]

# CTA line search forecast (Acharyya+ 2021 + Hutten+ 2022)
# (E_GeV, phi_limit ph cm^-2 s^-1)  ~10x better than HESS
CTA_LINE_LIMITS = [
    (25.0,   3.0e-12),
    (50.0,   2.0e-12),
    (100.0,  1.5e-12),
    (300.0,  1.0e-12),
    (1000.0, 8.0e-13),
]


# ---------------------------------------------------------------------------
# 1. Line intensities (flux predicted at Earth)
# ---------------------------------------------------------------------------

def relative_intensity_per_line(M_pearl_Msun: float = M_PEARL_TYPICAL_MSUN,
                                use_v5p21_min_dc: bool = False
                                ) -> List[Dict]:
    """Compute predicted flux phi_i for each of the 6 Class E lines.

    Two scenarios :
      * use_v5p21_min_dc=False (V5.20 baseline) :
            delta_c_E_i = 2 * n_brick_i  (formula §13.6.4 amphichiral)
            -> 4 or 6 crossings additionnels.
      * use_v5p21_min_dc=True  (V5.21 strict selection) :
            delta_c >= 22 obligatoire -> impose delta_c_eff = 22
            (alpha^22 ~ 1e-47 suppression).

    Flux at Earth (volume Hubble, isotropic) :
        phi_line_i = N_events_i * 2 / (4 pi d_H^2 t_H)    [ph cm^-2 s^-1]
    where the factor 2 accounts for 2 photons per annihilation
    (knot + antiknot -> 2 gamma). Each amphichiral splits Omega_DM as
    Omega_i = Omega_E / 6, so N_events_i scales linearly with abundance.

    Returns list of dicts per amphichiral.
    """
    out = []
    abundance_rescale = (OMEGA_PER_AMPHI * 6.0) / (CE.OMEGA_A_FRACTION)
    # ^ n_events_over_universe_age uses OMEGA_A = 0.05*0.265 internally; we
    # rescale to Omega_per_amphi = (0.10/6)*0.265.
    # rescale factor per amphi = (OMEGA_PER_AMPHI / OMEGA_A_FRACTION)
    abundance_rescale = OMEGA_PER_AMPHI / CE.OMEGA_A_FRACTION

    for name, m_chi, c_brick, n_brick in CE.CLASS_E_CATALOG:
        delta_c_E_native = CE.delta_c_amphichiral(c_brick, n_brick)
        if use_v5p21_min_dc:
            delta_c_used = max(delta_c_E_native, DELTA_C_V5P21_MIN)
        else:
            delta_c_used = delta_c_E_native

        nev = n_events_over_universe_age(M_pearl_Msun, m_chi, delta_c_used)
        N_events_universe = nev["N_events_total_universe"] * abundance_rescale

        # Average flux at Earth (Hubble-volume isotropic emission)
        # phi = N_events * N_photons_per_event / (4 pi d_H^2 t_H)
        # d_H = R_Hubble_m
        # Convert m^2 -> cm^2
        d_H_cm = R_HUBBLE_M * 100.0     # cm
        t_H = T_HUBBLE_S
        N_photons_per_event = 2.0       # k + anti-k -> 2 gamma
        phi_line = (N_events_universe * N_photons_per_event
                    / (4.0 * math.pi * d_H_cm ** 2 * t_H))

        out.append({
            "name": name,
            "m_chi_GeV": m_chi,
            "n_brick": n_brick,
            "delta_c_E_native": delta_c_E_native,
            "delta_c_used": delta_c_used,
            "sigma_cm2": sigma_entanglement(delta_c_used) * 1e4,
            "N_events_universe": N_events_universe,
            "phi_line_ph_cm2_s": phi_line,
        })
    return out


# ---------------------------------------------------------------------------
# 2. Line widths (natural + instrumental)
# ---------------------------------------------------------------------------

def line_widths_GeV(E_GeV: float) -> Dict[str, float]:
    """Largeurs apparentes des raies au detecteur (FWHM, GeV).

    Largeur naturelle (transition discrete topologique) ~ 0.
    Largeur instrumentale dominante.
    """
    return {
        "natural_GeV": 0.0,
        "fermi_FWHM_GeV": FERMI_DE_OVER_E * E_GeV,
        "cta_FWHM_GeV": CTA_DE_OVER_E * E_GeV,
        "fermi_DE_E": FERMI_DE_OVER_E,
        "cta_DE_E": CTA_DE_OVER_E,
    }


# ---------------------------------------------------------------------------
# 3. Fermi-LAT and CTA limits (interpolation)
# ---------------------------------------------------------------------------

def fermi_limit_at_E(E_GeV: float) -> float:
    """Interpolation log-log de la limite Fermi-LAT line search (ph/cm^2/s)."""
    Es = np.array([p[0] for p in FERMI_LINE_LIMITS_R3])
    phis = np.array([p[1] for p in FERMI_LINE_LIMITS_R3])
    if E_GeV <= Es[0]:
        return float(phis[0])
    if E_GeV >= Es[-1]:
        return float(phis[-1])
    log_phi = np.interp(math.log10(E_GeV), np.log10(Es), np.log10(phis))
    return float(10.0 ** log_phi)


def cta_limit_at_E(E_GeV: float) -> float:
    """Interpolation log-log de la sensibilite CTA line search (ph/cm^2/s)."""
    Es = np.array([p[0] for p in CTA_LINE_LIMITS])
    phis = np.array([p[1] for p in CTA_LINE_LIMITS])
    if E_GeV < Es[0]:
        # Below 25 GeV: CTA does not cover -> return inf
        return float('inf')
    if E_GeV >= Es[-1]:
        return float(phis[-1])
    log_phi = np.interp(math.log10(E_GeV), np.log10(Es), np.log10(phis))
    return float(10.0 ** log_phi)


# ---------------------------------------------------------------------------
# 4. Per-line verdict
# ---------------------------------------------------------------------------

def per_line_verdict(use_v5p21_min_dc: bool = False) -> List[Dict]:
    """Pour chaque raie : flux predit, limite Fermi, limite CTA, status."""
    lines = relative_intensity_per_line(use_v5p21_min_dc=use_v5p21_min_dc)
    out = []
    for L in lines:
        E = L["m_chi_GeV"]
        phi = L["phi_line_ph_cm2_s"]
        fermi_lim = fermi_limit_at_E(E)
        cta_lim = cta_limit_at_E(E)
        widths = line_widths_GeV(E)

        # Status Fermi
        if phi > fermi_lim:
            fermi_status = "EXCLUDED"
            fermi_ratio = phi / fermi_lim
        else:
            fermi_status = "ALLOWED"
            fermi_ratio = phi / fermi_lim

        # Status CTA (2027+)
        if cta_lim == float('inf'):
            cta_status = "OUT-OF-BAND"
            cta_ratio = 0.0
        elif phi > cta_lim:
            cta_status = "DETECTABLE"
            cta_ratio = phi / cta_lim
        else:
            cta_status = "BELOW-SENSITIVITY"
            cta_ratio = phi / cta_lim

        out.append({
            **L,
            "fermi_limit": fermi_lim,
            "cta_limit": cta_lim,
            "fermi_ratio": fermi_ratio,
            "cta_ratio": cta_ratio,
            "fermi_status": fermi_status,
            "cta_status": cta_status,
            "fermi_FWHM_GeV": widths["fermi_FWHM_GeV"],
            "cta_FWHM_GeV": widths["cta_FWHM_GeV"],
        })
    return out


def critical_delta_c_per_line(M_pearl_Msun: float = M_PEARL_TYPICAL_MSUN
                               ) -> List[Dict]:
    """Pour chaque raie, le Delta_c minimal pour passer SOUS la limite Fermi.

    Solution analytique : phi propto sigma propto alpha^delta_c.
    phi(dc) = phi(dc=base) * alpha^(dc - base)
    -> dc_crit = base + log(phi_base / phi_lim) / log(1/alpha)
    """
    abundance_rescale = OMEGA_PER_AMPHI / CE.OMEGA_A_FRACTION
    log_inv_alpha = math.log(1.0 / ALPHA_EM)
    out = []
    for name, m_chi, c_brick, n_brick in CE.CLASS_E_CATALOG:
        # Use a small base dc to get phi_base
        base_dc = 4
        nev = n_events_over_universe_age(M_pearl_Msun, m_chi, base_dc)
        N_events_universe = nev["N_events_total_universe"] * abundance_rescale
        d_H_cm = R_HUBBLE_M * 100.0
        phi_base = (N_events_universe * 2.0
                    / (4.0 * math.pi * d_H_cm ** 2 * T_HUBBLE_S))
        phi_lim = fermi_limit_at_E(m_chi)
        if phi_base <= phi_lim:
            dc_crit = base_dc
        else:
            extra = math.log(phi_base / phi_lim) / log_inv_alpha
            dc_crit = base_dc + math.ceil(extra)
        out.append({
            "name": name,
            "m_chi_GeV": m_chi,
            "n_brick": n_brick,
            "delta_c_E_native": 2 * n_brick,
            "delta_c_critical_Fermi": int(dc_crit),
            "phi_base_dc4": phi_base,
            "fermi_limit": phi_lim,
        })
    return out


def global_verdict(use_v5p21_min_dc: bool = False) -> Dict:
    """Synthese : raies exclues, detectables CTA, statut global."""
    per_line = per_line_verdict(use_v5p21_min_dc)
    n_excluded_fermi = sum(1 for L in per_line if L["fermi_status"] == "EXCLUDED")
    n_detectable_cta = sum(1 for L in per_line if L["cta_status"] == "DETECTABLE")
    n_below_cta = sum(1 for L in per_line if L["cta_status"] == "BELOW-SENSITIVITY")
    n_out_of_band = sum(1 for L in per_line if L["cta_status"] == "OUT-OF-BAND")

    if n_excluded_fermi == 6:
        status = "CLASS E STRICTLY EXCLUDED"
        comment = (
            f"Les 6 raies depassent les limites Fermi-LAT line search ; "
            f"V5.20 Class E (delta_c amphi {use_v5p21_min_dc=}) est "
            f"deja exclu en 2026."
        )
    elif n_excluded_fermi >= 1:
        status = "PARTIALLY EXCLUDED"
        comment = (
            f"{n_excluded_fermi}/6 raies excluent V5.20 Class E ; "
            "scenario marginal."
        )
    else:
        status = "COMPATIBLE WITH FERMI"
        comment = (
            "Aucune raie ne depasse Fermi-LAT 2015. "
            f"{n_detectable_cta}/6 raies detectables CTA 2027+, "
            f"{n_below_cta}/6 sous sensibilite, {n_out_of_band}/6 "
            "hors bande (< 25 GeV)."
        )

    return {
        "use_v5p21_min_dc": use_v5p21_min_dc,
        "per_line": per_line,
        "n_excluded_fermi": n_excluded_fermi,
        "n_detectable_cta": n_detectable_cta,
        "n_below_cta": n_below_cta,
        "n_out_of_band_cta": n_out_of_band,
        "critical_delta_c": critical_delta_c_per_line(),
        "status": status,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

def make_plot(path: Path = OUTPUT_PNG) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel (a) : phi_i vs Fermi/CTA limits (V5.20 baseline) ----
    ax = axes[0, 0]
    res_v520 = per_line_verdict(use_v5p21_min_dc=False)
    Es = np.array([L["m_chi_GeV"] for L in res_v520])
    phis = np.array([L["phi_line_ph_cm2_s"] for L in res_v520])
    # Fermi & CTA limit curves
    E_grid = np.logspace(0, 3, 200)
    fermi_curve = np.array([fermi_limit_at_E(e) for e in E_grid])
    cta_curve = np.array([cta_limit_at_E(e) for e in E_grid])
    cta_mask = np.isfinite(cta_curve)
    ax.plot(E_grid, fermi_curve, 'k-', lw=1.5, label="Fermi-LAT line limit (Ackermann+ 2015)")
    ax.plot(E_grid[cta_mask], cta_curve[cta_mask], 'b--', lw=1.5, label="CTA forecast (Acharyya+ 2021)")
    # Predicted lines as vertical arrows
    ax.scatter(Es, phis, s=120, marker='v', color='red', edgecolors='k', zorder=5,
               label=r"V5.20 Class E ($\Delta c_E$ natif)")
    for L in res_v520:
        ax.annotate(L["name"], (L["m_chi_GeV"], L["phi_line_ph_cm2_s"]),
                    fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel(r"$\phi_\gamma$ (ph cm$^{-2}$ s$^{-1}$)")
    ax.set_title("(a) V5.20 Class E baseline vs Fermi-LAT + CTA")
    ax.set_ylim(1e-14, 1e10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (b) : V5.21 strict (delta_c >= 22) ----
    ax = axes[0, 1]
    res_v521 = per_line_verdict(use_v5p21_min_dc=True)
    phis_521 = np.array([L["phi_line_ph_cm2_s"] for L in res_v521])
    ax.plot(E_grid, fermi_curve, 'k-', lw=1.5, label="Fermi-LAT limit")
    ax.plot(E_grid[cta_mask], cta_curve[cta_mask], 'b--', lw=1.5, label="CTA forecast")
    ax.scatter(Es, phis_521, s=120, marker='v', color='green', edgecolors='k', zorder=5,
               label=r"V5.21 ($\Delta c_{\rm min}=22$)")
    for L in res_v521:
        ax.annotate(L["name"], (L["m_chi_GeV"], L["phi_line_ph_cm2_s"]),
                    fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel(r"$\phi_\gamma$ (ph cm$^{-2}$ s$^{-1}$)")
    ax.set_title(r"(b) V5.21 selection $\Delta c \geq 22$ vs Fermi-LAT + CTA")
    ax.set_ylim(1e-60, 1e10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (c) : ratio phi/Fermi par raie (les deux scenarios) ----
    ax = axes[1, 0]
    names = [L["name"] for L in res_v520]
    x = np.arange(len(names))
    width = 0.4
    r520 = [L["fermi_ratio"] for L in res_v520]
    r521 = [L["fermi_ratio"] for L in res_v521]
    ax.bar(x - width/2, r520, width, color='red', alpha=0.7,
           label=r"V5.20 baseline ($\Delta c_E$ natif)")
    ax.bar(x + width/2, r521, width, color='green', alpha=0.7,
           label=r"V5.21 strict ($\Delta c \geq 22$)")
    ax.axhline(1.0, color='k', ls='--', label='Limite Fermi (=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\phi_i / \phi_{\rm Fermi}$")
    ax.set_title("(c) Sur-prediction vs Fermi-LAT line search")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # ---- Panel (d) : SNR CTA + largeur de raie instrumentale ----
    ax = axes[1, 1]
    fwhm_fermi = [L["fermi_FWHM_GeV"] for L in res_v520]
    fwhm_cta = [L["cta_FWHM_GeV"] for L in res_v520]
    ax.scatter(Es, fwhm_fermi, s=80, marker='o', color='black', label='Fermi FWHM (12%)')
    ax.scatter(Es, fwhm_cta, s=80, marker='s', color='blue', label='CTA FWHM (7%)')
    for L in res_v520:
        ax.annotate(L["name"], (L["m_chi_GeV"], L["fermi_FWHM_GeV"]),
                    fontsize=6.5, xytext=(5, -10), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E_line (GeV)")
    ax.set_ylabel("FWHM (GeV)")
    ax.set_title("(d) Largeur instrumentale apparente (raies V5 ~ delta)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "V5.20 Class E -- 6 raies gamma discretes : prediction vs "
        "Fermi-LAT + CTA forecast",
        fontsize=13, weight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("V5.20 Class E -- 6 raies gamma vs Fermi-LAT + CTA")
    print("=" * 78)

    for label, use_v521 in [("V5.20 baseline", False),
                             ("V5.21 strict (dc>=22)", True)]:
        print(f"\n--- Scenario : {label} ---")
        v = global_verdict(use_v521)
        print(f"  Status      : {v['status']}")
        print(f"  N excluded Fermi : {v['n_excluded_fermi']}/6")
        print(f"  N detectable CTA : {v['n_detectable_cta']}/6")
        print(f"  N below CTA      : {v['n_below_cta']}/6")
        print(f"  N out-of-band CTA: {v['n_out_of_band_cta']}/6")
        print()
        print(f"  {'composition':18s} {'E (GeV)':>9s} {'dc':>4s} "
              f"{'phi pred':>14s} {'Fermi lim':>14s} {'ratio':>10s} "
              f"{'Fermi':>10s} {'CTA':>14s}")
        for L in v['per_line']:
            print(f"  {L['name']:18s} {L['m_chi_GeV']:9.2f} "
                  f"{L['delta_c_used']:4d} {L['phi_line_ph_cm2_s']:14.2e} "
                  f"{L['fermi_limit']:14.2e} {L['fermi_ratio']:10.2e} "
                  f"{L['fermi_status']:>10s} {L['cta_status']:>14s}")
        print(f"\n  Comment: {v['comment']}")

    print("\n--- Delta_c critique (par raie) pour passer SOUS Fermi-LAT ---")
    crit = critical_delta_c_per_line()
    print(f"  {'composition':18s} {'E (GeV)':>9s} {'dc native':>10s} "
          f"{'dc crit Fermi':>14s}")
    for c in crit:
        print(f"  {c['name']:18s} {c['m_chi_GeV']:9.2f} "
              f"{c['delta_c_E_native']:10d} {c['delta_c_critical_Fermi']:14d}")

    print("\n--- Plot ---")
    p = make_plot()
    print(f"  ecrit : {p}")


__all__ = [
    "OMEGA_E_FRACTION", "OMEGA_PER_AMPHI",
    "DELTA_C_CLASS_A", "DELTA_C_V5P21_MIN",
    "FERMI_DE_OVER_E", "CTA_DE_OVER_E",
    "FERMI_LINE_LIMITS_R3", "CTA_LINE_LIMITS",
    "relative_intensity_per_line",
    "line_widths_GeV",
    "fermi_limit_at_E", "cta_limit_at_E",
    "per_line_verdict", "global_verdict",
    "make_plot",
]
