#!/usr/bin/env python3
"""V5.20 Class A continuum vs Fermi-LAT (avec V5.21 selection Delta c >= 22).

Cadre
-----
Class A (DM Pearls) abrite une population de noeuds amphichiraux dont la
masse est ETALEE sur une fonction de masse log-normale (catalogue V5,
mu ~ 10 GeV, sigma ~ 0.5 dex). Chaque enchevetrement spontane (intra-
perle) emet une cascade gamma centree sur m_chi du noeud impliquee.

V5.20 baseline (delta_c = 10) etait deja exclu a > 1e23 sigma par
Fermi-LAT EDGRB (commit 23d5f88). V5.21 a derive Delta_c_min = 22
via Bar-Natan + Q_phys + Kanenobu-Murakami + SU(2)_135
(theory/dc_selection_rule.py, commit 8743f4f).

Question : Class A (continuum gamma, mass function complete) avec
Delta_c = 22 passe-t-il EDGRB, GCE, et line search ?

Difference avec le test single-mass deja en place (m_chi = 1.55 GeV
fixe) : ici on integre sur la mass function et chaque bin de masse
contribue une cascade dont le pic est a son propre m_chi -- le
spectre total est donc beaucoup plus large et la normalisation est
dominee par les noeuds les plus lourds (Gamma_pearl ~ N^2 sigma v_F / V
augmente avec m_chi par compaction n_F).

Donnees observationnelles
--------------------------
* EDGRB Ackermann+ 2015 (ApJ 799, 86) : I = I_0 (E/100 MeV)^(-2.32) exp(-E/279 GeV).
* Liang+ 2024 (ApJ updated EDGRB) : ~ x0.8 a 1 GeV (a pres de la meme normalisation,
  on adopte donc Ackermann pour l'analyse principale).
* GCE Hooper-Goodenough 2011 : pic 2 GeV, ROI |b|<20 |l|<20.
* Fermi-LAT line search Ackermann+ 2015 PRD 91, 122002 :
  ~3e-11 ph cm^-2 s^-1 a 50-100 GeV (gentle binning).

Approche :
  1. Mass function log-normale (mu_GeV, sigma_dex)
  2. Pour chaque bin, taux d'evenements / m^3 / s (Delta c = 22)
  3. Spectre log-normal centre sur m_chi du bin
  4. Somme ponderee --> dI/dE total
  5. Comparaison EDGRB / GCE / line search

Verdict :
  - HONNETE : si Class A continuum passe -> V5 sauvee
  - HONNETE : si Class A continuum deborde EDGRB malgre Delta_c = 22
    -> F3 ouverte aussi pour Class A.

Run :
    PYTHONPATH=. python3 scripts/V5_Fermi_LAT/class_A_continuum_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (
    ALPHA_EM,
    M_PEARL_TYPICAL_MSUN,
    R_HUBBLE_M,
    T_HUBBLE_S,
    V_HUBBLE_M3,
    n_events_over_universe_age,
)
from v5p20_gamma_diffuse_test import (
    EDGRB_ALPHA,
    EDGRB_E_CUT_GeV,
    EDGRB_I0_per_MeV,
    GEV_TO_MEV,
    edgrb_fermi,
    gce_hooper,
)
from class_E_six_lines_predictions import (
    FERMI_LINE_LIMITS_R3,
    fermi_limit_at_E,
)


OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_class_A_continuum_fermi.png"


# ---------------------------------------------------------------------------
# Class A mass function (default : log-normale centree mu=10 GeV, sigma=0.5 dex)
# ---------------------------------------------------------------------------

MU_GEV_DEFAULT = 10.0
SIGMA_DEX_DEFAULT = 0.5
M_MIN_GEV = 0.3
M_MAX_GEV = 60.0          # high-mass cutoff (TOV bissection fails > ~80 GeV)
N_MASS_BINS = 25

# Cascade : log-normale centree sur m_chi avec sigma 0.5 dex
SIGMA_LOG_CASCADE = 0.5
N_PHOTONS_PER_CASCADE = 10

DELTA_C_DEFAULT = 22

SR_FULL = 4.0 * math.pi


def mass_pdf(m_GeV: np.ndarray,
             mu: float = MU_GEV_DEFAULT,
             sigma_dex: float = SIGMA_DEX_DEFAULT) -> np.ndarray:
    """Densite de probabilite log-normale dp/dm (1/GeV)."""
    m = np.asarray(m_GeV, dtype=float)
    sig = sigma_dex * math.log(10.0)
    ln_mu = math.log(mu)
    return np.where(m > 0,
                    np.exp(-0.5 * ((np.log(np.maximum(m, 1e-300)) - ln_mu) / sig) ** 2)
                    / (m * sig * math.sqrt(2.0 * math.pi)),
                    0.0)


def mass_grid_and_pdf(mu: float = MU_GEV_DEFAULT,
                      sigma_dex: float = SIGMA_DEX_DEFAULT,
                      n_bins: int = N_MASS_BINS,
                      m_min: float = M_MIN_GEV,
                      m_max: float = M_MAX_GEV
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Grille log-uniforme et PDF normalisee (integrale = 1) sur [m_min, m_max]."""
    m_grid = np.logspace(math.log10(m_min), math.log10(m_max), n_bins)
    pdf = mass_pdf(m_grid, mu, sigma_dex)
    norm = np.trapezoid(pdf, m_grid)
    if norm > 0:
        pdf = pdf / norm
    return m_grid, pdf


# ---------------------------------------------------------------------------
# Cascade spectrum per knot mass (log-normale)
# ---------------------------------------------------------------------------

def cascade_dN_dE_bin(E_GeV: np.ndarray, m_chi_GeV: float,
                      sigma_log: float = SIGMA_LOG_CASCADE) -> np.ndarray:
    """Spectre photons normalise par cascade pour un noeud de masse m_chi.

    Pic log-normal centre a m_chi avec largeur sigma_log (dex).
    Normalisation : int dN/dE dE = 1.
    """
    E = np.asarray(E_GeV, dtype=float)
    ln_mu = math.log(m_chi_GeV)
    sig = sigma_log * math.log(10.0)
    return np.where(E > 0,
                    np.exp(-0.5 * ((np.log(np.maximum(E, 1e-300)) - ln_mu) / sig) ** 2)
                    / (E * sig * math.sqrt(2.0 * math.pi)),
                    0.0)


# ---------------------------------------------------------------------------
# Events rate per m^3 per s per mass bin
# ---------------------------------------------------------------------------

_RATE_CACHE: Dict[Tuple[float, int], float] = {}


def n_events_rate_per_m3_s(m_chi_GeV: float, delta_c: int) -> float:
    """N_events_total_universe / (V_Hubble * t_Hubble) pour un seul m_chi."""
    key = (round(m_chi_GeV, 6), int(delta_c))
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    res = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, m_chi_GeV, delta_c)
    rate = res["N_events_total_universe"] / (V_HUBBLE_M3 * T_HUBBLE_S)
    _RATE_CACHE[key] = rate
    return rate


# ---------------------------------------------------------------------------
# Class A continuum intensity (sum over mass bins)
# ---------------------------------------------------------------------------

def intensity_class_A_continuum(E_GeV: np.ndarray,
                                delta_c: int = DELTA_C_DEFAULT,
                                mu: float = MU_GEV_DEFAULT,
                                sigma_dex: float = SIGMA_DEX_DEFAULT,
                                n_bins: int = N_MASS_BINS,
                                N_photons: float = N_PHOTONS_PER_CASCADE
                                ) -> np.ndarray:
    """Intensite diffuse Class A continuum (ph cm^-2 s^-1 sr^-1 GeV^-1).

    Pour chaque bin de masse :
      Q_i(E) = rate_i * N_photons * f_i(E)        [ph m^-3 s^-1 GeV^-1]
    avec f_i(E) le spectre log-normale centre sur m_chi_i.

    Total :
      Q(E) = int rate(m) * N_photons * f(E,m) * p(m) dm

    Intensite isotrope ligne de vue ~ R_Hubble :
      dI/dE = R_H * Q(E) / (4 pi) / 1e4   [photons cm^-2 s^-1 sr^-1 GeV^-1]
    """
    E = np.asarray(E_GeV, dtype=float)
    m_grid, pdf = mass_grid_and_pdf(mu, sigma_dex, n_bins)
    rates = np.array([n_events_rate_per_m3_s(m, delta_c) for m in m_grid])
    # Q_per_E[k] = int over m of rate(m) * pdf(m) * f(E_k, m)
    Q = np.zeros_like(E)
    for k, Ek in enumerate(E):
        f_vals = np.array([float(cascade_dN_dE_bin(np.array([Ek]), m)[0])
                           for m in m_grid])
        Q[k] = np.trapezoid(rates * pdf * f_vals, m_grid) * N_photons
    dI_dE_m2 = R_HUBBLE_M * Q / SR_FULL
    return dI_dE_m2 / 1.0e4


# ---------------------------------------------------------------------------
# Continuum vs Fermi-LAT EDGRB
# ---------------------------------------------------------------------------

def ratio_continuum_vs_fermi(E_GeV: np.ndarray,
                             delta_c: int = DELTA_C_DEFAULT,
                             mu: float = MU_GEV_DEFAULT,
                             sigma_dex: float = SIGMA_DEX_DEFAULT) -> np.ndarray:
    """Ratio I_classA(E) / I_Fermi(E)."""
    return (intensity_class_A_continuum(E_GeV, delta_c, mu, sigma_dex)
            / edgrb_fermi(E_GeV))


def verdict_continuum(delta_c: int = DELTA_C_DEFAULT,
                      mu: float = MU_GEV_DEFAULT,
                      sigma_dex: float = SIGMA_DEX_DEFAULT) -> Dict:
    """Verdict global Class A continuum vs Fermi EDGRB (1-100 GeV)."""
    E_test = np.array([1.0])
    I_V5_1 = float(intensity_class_A_continuum(E_test, delta_c, mu, sigma_dex)[0])
    I_F_1 = float(edgrb_fermi(E_test)[0])

    # Integration sur 1-100 GeV
    E_int = np.logspace(0, 2, 60)
    I_V5_int = intensity_class_A_continuum(E_int, delta_c, mu, sigma_dex)
    F_V5 = float(np.trapezoid(I_V5_int, E_int))
    F_F = float(np.trapezoid(edgrb_fermi(E_int), E_int))
    ratio_int = F_V5 / F_F

    # ratio max sur 0.1-100 GeV (resolution plus large)
    E_wide = np.logspace(-1, 2, 50)
    ratios = (intensity_class_A_continuum(E_wide, delta_c, mu, sigma_dex)
              / edgrb_fermi(E_wide))
    ratio_max = float(np.max(ratios))
    E_at_max = float(E_wide[int(np.argmax(ratios))])

    # GCE comparison (1-10 GeV)
    E_gce = np.logspace(0, 1, 30)
    F_gce_V5 = float(np.trapezoid(
        intensity_class_A_continuum(E_gce, delta_c, mu, sigma_dex), E_gce))
    F_gce_obs = float(np.trapezoid(gce_hooper(E_gce), E_gce))
    ratio_gce = F_gce_V5 / F_gce_obs

    # Line search check : a chaque m_chi du mass function, peak-flux integre
    # sur largeur instrumentale Fermi 12% vs limite line search
    # On approxime ici par dI/dE * FWHM_inst, avec FWHM = 0.12 m_chi
    m_grid, pdf = mass_grid_and_pdf(mu, sigma_dex)
    line_search_status = []
    for m, p in zip(m_grid, pdf):
        I_at_m = float(intensity_class_A_continuum(np.array([m]),
                                                   delta_c, mu, sigma_dex)[0])
        fwhm = 0.12 * m
        phi_line = I_at_m * fwhm * SR_FULL  # ph cm^-2 s^-1 (full sky approx)
        phi_lim = fermi_limit_at_E(m)
        line_search_status.append({
            "m_GeV": float(m),
            "pdf": float(p),
            "phi_line_approx": phi_line,
            "phi_limit_Fermi": phi_lim,
            "ratio": phi_line / phi_lim if phi_lim > 0 else float("inf"),
        })
    line_ratio_max = max(L["ratio"] for L in line_search_status)

    if ratio_int > 1e3:
        status = "STRICTLY EXCLUDED"
        comment = (
            f"Class A continuum (mu={mu} GeV, sigma={sigma_dex} dex) avec "
            f"Delta_c={delta_c} sur-predit EDGRB par {ratio_int:.2e}. "
            f"F3 OUVERTE pour Class A : V5.21 Delta_c=22 INSUFFISANT."
        )
    elif ratio_int > 1.0:
        status = "EXCLUDED"
        comment = (
            f"Class A continuum deborde EDGRB d'un facteur {ratio_int:.1f}. "
            f"F3 OUVERTE."
        )
    elif ratio_int > 1e-2:
        status = "MARGINAL"
        comment = (
            f"Class A continuum contribue {ratio_int*100:.1f}% de EDGRB. "
            f"Compatible mais limite (Liang+ 2024 contraint a ~30%)."
        )
    elif ratio_int > 1e-6:
        status = "COMPATIBLE"
        comment = (
            f"Class A continuum ~ {ratio_int*100:.2e}% de EDGRB. "
            f"V5 SAUVEE par Delta_c={delta_c}."
        )
    else:
        status = "DEEPLY SUB-DOMINANT"
        comment = (
            f"Class A continuum ~ {ratio_int:.2e} de EDGRB : invisible. "
            f"V5 SAUVEE largement."
        )

    return {
        "delta_c": delta_c,
        "mu_GeV": mu,
        "sigma_dex": sigma_dex,
        "I_V5_1GeV": I_V5_1,
        "I_Fermi_1GeV": I_F_1,
        "ratio_1GeV": I_V5_1 / I_F_1,
        "F_V5_1to100GeV": F_V5,
        "F_Fermi_1to100GeV": F_F,
        "ratio_integrated_1_100GeV": ratio_int,
        "ratio_max_wide": ratio_max,
        "E_at_ratio_max_GeV": E_at_max,
        "ratio_gce_1_10GeV": ratio_gce,
        "line_search_ratio_max": line_ratio_max,
        "line_search_details": line_search_status,
        "status": status,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# Plot 4 panneaux
# ---------------------------------------------------------------------------

def make_plot(delta_c_values=(10, 22, 30, 50),
              mu: float = MU_GEV_DEFAULT,
              sigma_dex: float = SIGMA_DEX_DEFAULT,
              out_path: Path = OUTPUT_PNG) -> Path:
    """Plot 4 panneaux : continuum vs EDGRB / GCE / line / mass function."""
    E = np.logspace(-1, math.log10(500.0), 80)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- (a) E^2 dN/dE continuum vs EDGRB ---
    ax = axes[0, 0]
    E2F = E ** 2 * edgrb_fermi(E)
    ax.plot(E, E2F, 'k-', lw=2.0, label='Fermi-LAT EDGRB (Ackermann+ 2015)')
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(delta_c_values)))
    for c, dc in zip(colors, delta_c_values):
        I = intensity_class_A_continuum(E, dc, mu, sigma_dex)
        ax.plot(E, E ** 2 * I, color=c, lw=1.5,
                label=f'Class A continuum $\\Delta c$={dc}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$E^2 dN/dE$ (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    ax.set_title(f'(a) Class A continuum ($\\mu$={mu} GeV, $\\sigma$={sigma_dex} dex)\n'
                 'vs Fermi-LAT EDGRB')
    ax.set_ylim(1e-18, 1e15)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

    # --- (b) ratio I_V5 / I_Fermi ---
    ax = axes[0, 1]
    for c, dc in zip(colors, delta_c_values):
        r = ratio_continuum_vs_fermi(E, dc, mu, sigma_dex)
        ax.plot(E, r, color=c, lw=1.5, label=f'$\\Delta c$={dc}')
    ax.axhline(1.0, color='r', ls='--', label='compatibilite EDGRB')
    ax.axhline(0.3, color='orange', ls=':', label='Liang+ 2024 ~30%')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$I_{Class\ A}(E) / I_{Fermi}(E)$')
    ax.set_title('(b) Ratio Class A continuum / Fermi-LAT')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # --- (c) vs GCE (1-10 GeV) ---
    ax = axes[1, 0]
    E_gce = np.logspace(-0.5, 1.5, 40)
    ax.plot(E_gce, E_gce ** 2 * gce_hooper(E_gce), 'g-', lw=2.0,
            label='GCE Hooper+Goodenough 2011')
    for c, dc in zip(colors, delta_c_values):
        I = intensity_class_A_continuum(E_gce, dc, mu, sigma_dex)
        ax.plot(E_gce, E_gce ** 2 * I, color=c, lw=1.5,
                label=f'Class A $\\Delta c$={dc}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$E^2 dN/dE$ (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    ax.set_title('(c) Class A continuum vs Galactic Center Excess')
    ax.set_ylim(1e-18, 1e10)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

    # --- (d) mass function + ratio integre vs Delta_c ---
    ax = axes[1, 1]
    dc_grid = np.arange(10, 60, 2)
    E_int = np.logspace(0, 2, 40)
    F_F = float(np.trapezoid(edgrb_fermi(E_int), E_int))
    ratios = []
    for dc in dc_grid:
        F_V5 = float(np.trapezoid(
            intensity_class_A_continuum(E_int, dc, mu, sigma_dex), E_int))
        ratios.append(F_V5 / F_F)
    ratios = np.array(ratios)
    ax.semilogy(dc_grid, ratios, 'b-', lw=1.5,
                label='Class A continuum / Fermi (1-100 GeV)')
    ax.axhline(1.0, color='r', ls='--', label='Exclusion (>1)')
    ax.axhline(0.1, color='orange', ls=':', label='Sub-dominant (<10%)')
    # dc critique
    above = np.where(ratios > 1.0)[0]
    if len(above) > 0 and above[-1] + 1 < len(dc_grid):
        dc_crit = dc_grid[above[-1] + 1]
        ax.axvline(dc_crit, color='g', ls='-.', lw=1.2,
                   label=f'$\\Delta c_{{crit}}^{{Class\\ A}}$ = {dc_crit}')
    ax.axvline(22, color='purple', ls='--', lw=1.0,
               label=r'V5.21 $\Delta c_{\min}$=22')
    ax.set_xlabel(r'$\Delta c$ (crossings Vassiliev)')
    ax.set_ylabel(r'$F_{Class\ A}/F_{Fermi}$ (1-100 GeV)')
    ax.set_title('(d) Bound exclusion sur $\\Delta c$ (Class A continuum)')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        'V5.20 Class A CONTINUUM vs Fermi-LAT (mass function complete) -- '
        f'$\\mu$={mu} GeV, $\\sigma$={sigma_dex} dex',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("V5.20 Class A CONTINUUM vs Fermi-LAT (V5.21 Delta c = 22)")
    print("=" * 78)
    print(f"\nMass function : log-normale mu={MU_GEV_DEFAULT} GeV, "
          f"sigma={SIGMA_DEX_DEFAULT} dex")
    print(f"M_pearl = {M_PEARL_TYPICAL_MSUN} M_sun, "
          f"{N_MASS_BINS} bins sur [{M_MIN_GEV}, {M_MAX_GEV}] GeV")
    print()
    for dc in [10, 22, 30, 50]:
        v = verdict_continuum(delta_c=dc)
        print(f"--- Delta_c = {dc} ---")
        print(f"  I_V5(1 GeV)     = {v['I_V5_1GeV']:.3e}")
        print(f"  I_Fermi(1 GeV)  = {v['I_Fermi_1GeV']:.3e}")
        print(f"  ratio @ 1 GeV   = {v['ratio_1GeV']:.3e}")
        print(f"  ratio int 1-100 = {v['ratio_integrated_1_100GeV']:.3e}")
        print(f"  ratio max wide  = {v['ratio_max_wide']:.3e} (at "
              f"{v['E_at_ratio_max_GeV']:.2f} GeV)")
        print(f"  ratio GCE 1-10  = {v['ratio_gce_1_10GeV']:.3e}")
        print(f"  line search max = {v['line_search_ratio_max']:.3e}")
        print(f"  STATUS : {v['status']}")
        print(f"  {v['comment']}")
        print()
    p = make_plot()
    print(f"Plot ecrit : {p}")


__all__ = [
    "MU_GEV_DEFAULT", "SIGMA_DEX_DEFAULT", "DELTA_C_DEFAULT",
    "M_MIN_GEV", "M_MAX_GEV", "N_MASS_BINS",
    "mass_pdf", "mass_grid_and_pdf",
    "cascade_dN_dE_bin", "n_events_rate_per_m3_s",
    "intensity_class_A_continuum", "ratio_continuum_vs_fermi",
    "verdict_continuum", "make_plot",
]
