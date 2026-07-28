#!/usr/bin/env python3
"""V5.20 -- Test gamma diffus vs Fermi-LAT EDGRB.

Prediction V5.20 (cf. theory/dm_pearl_internal.py, commit 67e0489) :
  * Enchevetrement spontane interne aux DM pearls (Class A) ~ 1.9e92
    evenements sur l'univers observable pour Delta_c = 10.
  * Chaque event emet une cascade gamma ~ 1.5 GeV par desintegration
    du link n=2 (P15 instability).
  * Signature observable possible : exces dans le fond diffus
    extra-galactique gamma (EDGRB) mesure par Fermi-LAT.

Donnees observationnelles :
  * EDGRB Ackermann+ 2015 (ApJ 799, 86) :
      I(E) = I_0 (E/100 MeV)^(-alpha) exp(-E/E_cut)
      I_0 ~ 1.12e-7 cm^-2 s^-1 sr^-1 MeV^-1
      alpha ~ 2.32 (single power law)
      E_cut ~ 279 GeV
  * GCE (Galactic Center Excess) Hooper+ Goodenough 2011 :
      peak ~ 2 GeV, ~ 1e-6 cm^-2 s^-1 (integrated 1-100 GeV)
      ROI : |b| < 20 deg, |l| < 20 deg

Calcul V5.20 :

  N_events_univ = 1.9e92 (Delta_c = 10, baseline V5.20)
  E_cascade ~ 1.5 GeV / event
  N_photons_per_cascade ~ 10 (analogue fragmentation hadronique)

  Luminosite gamma totale dans le volume Hubble :
    L_gamma_tot = N_events_univ * E_cascade / t_Hubble
                ~ 1.9e92 * 1.5 GeV / 4.35e17 s
                ~ 6.5e74 GeV/s = 1.0e66 erg/s
                ~ 1.0e83 ph/s

  Densite numerique de photons GeV produits :
    n_phot_rate = N_events_univ * N_photons / (V_Hubble * t_Hubble)

  Intensite diffuse (isotrope) :
    I_diff = c * n_phot_rate / (4 pi)  [photons cm^-2 s^-1 sr^-1]
    dN/dE = I_diff * f_E(E)            avec f_E(E) le spectre normalise

Spectre par cascade : modele simple
    pi0 -> 2 gamma + ISR continuum (analogue DM annihilation)
    pic centre log-normal a 1.5 GeV avec sigma_log ~ 0.5 dex
    plus une queue power-law continuum E^-2 entre 0.1 et 10 GeV.

Output :
  * Spectre V5.20 vs Fermi-LAT EDGRB sur 0.1-820 GeV
  * Comparaison avec GCE
  * Verdict : compatible / sur-prediction / sous-prediction

Run :
    PYTHONPATH=. python3 scripts/V5_Fermi_LAT/v5p20_gamma_diffuse_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (
    M_PEARL_TYPICAL_MSUN,
    T_HUBBLE_S,
    V_HUBBLE_M3,
    R_HUBBLE_M,
    C_LIGHT,
    n_events_over_universe_age,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_fermi_lat_test.png"


# ---------------------------------------------------------------------------
# Constantes V5.20 cascade
# ---------------------------------------------------------------------------

E_CASCADE_GEV = 1.5            # energie typique cascade par event (P15 link n=2)
N_PHOTONS_PER_CASCADE = 10     # multiplicite photons GeV-scale (frag. hadronique)
SIGMA_LOG_PEAK = 0.5           # largeur log-normale du pic principal (dex)
F_CONT = 0.5                   # fraction d'energie dans le continuum E^-2
F_PEAK = 1.0 - F_CONT          # fraction dans le pic ~1.5 GeV
E_MIN_GEV = 0.1                # bas du continuum E^-2
E_MAX_GEV = 10.0               # haut du continuum E^-2

# Convert from "per pearl per second" to flux at Earth
# Conversion units
GEV_TO_MEV = 1.0e3
CM2_PER_M2 = 1.0e4
SR_FULL = 4.0 * math.pi


# ---------------------------------------------------------------------------
# 1. Spectre V5.20 par cascade (normalise)
# ---------------------------------------------------------------------------

def cascade_spectrum_dN_dE(E_GeV: np.ndarray) -> np.ndarray:
    """Spectre normalise photons par cascade : (1/N) dN/dE en GeV^-1.

    Mixture :
      * Pic log-normal a 1.5 GeV (fraction F_PEAK des photons)
      * Continuum E^-2 entre 0.1 et 10 GeV (fraction F_CONT)

    Normalisation : int dN/dE dE = 1
    """
    E = np.asarray(E_GeV, dtype=float)

    # Pic log-normal
    ln_mu = math.log(E_CASCADE_GEV)
    sig = SIGMA_LOG_PEAK * math.log(10)   # convert dex -> ln
    peak_kernel = np.where(E > 0,
                           np.exp(-0.5 * ((np.log(np.maximum(E, 1e-300)) - ln_mu) / sig) ** 2)
                           / (E * sig * math.sqrt(2 * math.pi)),
                           0.0)
    # already normalized to 1 by construction (log-normal)

    # Continuum E^-2 entre E_MIN et E_MAX
    # Normalisation : int_E_MIN^E_MAX E^-2 dE = 1/E_MIN - 1/E_MAX
    norm_cont = 1.0 / E_MIN_GEV - 1.0 / E_MAX_GEV
    cont = np.where((E >= E_MIN_GEV) & (E <= E_MAX_GEV),
                    1.0 / (E ** 2) / norm_cont,
                    0.0)

    return F_PEAK * peak_kernel + F_CONT * cont


def cascade_photons_total(E_min_GeV: float = 1e-3, E_max_GeV: float = 1e3,
                          n_pts: int = 5000) -> float:
    """Verification : integrale du spectre normalise (doit valoir 1)."""
    Eg = np.logspace(math.log10(E_min_GeV), math.log10(E_max_GeV), n_pts)
    dNdE = cascade_spectrum_dN_dE(Eg)
    return float(np.trapezoid(dNdE, Eg))


# ---------------------------------------------------------------------------
# 2. Intensite diffuse V5.20 (sky-isotrope, contribution cosmologique)
# ---------------------------------------------------------------------------

def n_events_per_volume_per_time(delta_c: int = 10) -> float:
    """Taux d'enchevetrement par metre cube et par seconde (events m^-3 s^-1).

    n_events_rate = N_events_total_universe / (V_Hubble * t_Hubble)
    """
    res = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, 1.55, delta_c=delta_c)
    return res["N_events_total_universe"] / (V_HUBBLE_M3 * T_HUBBLE_S)


def intensity_diffuse_v5p20(E_GeV: np.ndarray, delta_c: int = 10,
                            N_photons_per_cascade: float = N_PHOTONS_PER_CASCADE
                            ) -> np.ndarray:
    """Intensite diffuse V5.20 dI/dE (photons cm^-2 s^-1 sr^-1 GeV^-1).

    Modele : volume entier de l'univers emet de maniere isotrope.

    Taux de production de photons par unite de volume :
        Q(E) = n_events_rate * N_photons * f(E)
               [photons m^-3 s^-1 GeV^-1]

    Pour une source statique 3D cosmologique, l'intensite diffuse a Earth
    (en negligeant absorption EBL, redshift -> approximation) vaut :
        dI/dE = c * Q(E) * t_H / (4 pi)
              ~ c * Q(E) * d_H / (4 pi)   avec d_H = c * t_H = R_Hubble
        => dI/dE = R_Hubble * Q(E) / (4 pi)

    Cette forme est l'analogue de l'integrale de ligne d'une emissivite
    cosmologique uniforme jusqu'a l'horizon. Unite finale :
        [m] * [photons m^-3 s^-1 GeV^-1] / sr = [photons m^-2 s^-1 sr^-1 GeV^-1]
        -> on convertit en cm^-2.
    """
    E = np.asarray(E_GeV, dtype=float)
    Q_per_m3_per_s_per_GeV = (n_events_per_volume_per_time(delta_c)
                              * N_photons_per_cascade
                              * cascade_spectrum_dN_dE(E))
    # Ligne de vue ~ R_Hubble (approximation simple)
    dI_dE_m2 = R_HUBBLE_M * Q_per_m3_per_s_per_GeV / SR_FULL
    # m^-2 -> cm^-2 : divise par 1e4
    return dI_dE_m2 / CM2_PER_M2


# ---------------------------------------------------------------------------
# 3. EDGRB Fermi-LAT Ackermann+ 2015
# ---------------------------------------------------------------------------

# Fit single power-law (Ackermann+ 2015, ApJ 799, 86, Table 3 ligne PL)
EDGRB_I0_per_MeV = 1.12e-7   # cm^-2 s^-1 sr^-1 MeV^-1 a 100 MeV
EDGRB_ALPHA = 2.32
EDGRB_E_CUT_GeV = 279.0       # cutoff exponentiel (model B fit)


def edgrb_fermi(E_GeV: np.ndarray, with_cutoff: bool = True) -> np.ndarray:
    """EDGRB Fermi-LAT Ackermann+ 2015 (cm^-2 s^-1 sr^-1 GeV^-1).

    I(E) = I_0 (E/100 MeV)^{-alpha} * (exp(-E/E_cut) si with_cutoff)
    converti en GeV^-1.
    """
    E_MeV = np.asarray(E_GeV, dtype=float) * GEV_TO_MEV
    flux_per_MeV = EDGRB_I0_per_MeV * (E_MeV / 100.0) ** (-EDGRB_ALPHA)
    if with_cutoff:
        flux_per_MeV = flux_per_MeV * np.exp(
            -np.asarray(E_GeV) / EDGRB_E_CUT_GeV)
    # convert MeV^-1 -> GeV^-1 (multiply by 1000)
    return flux_per_MeV * GEV_TO_MEV


def gce_hooper(E_GeV: np.ndarray) -> np.ndarray:
    """GCE Hooper-Goodenough 2011 -- spectre stylise (cm^-2 s^-1 sr^-1 GeV^-1).

    Pic ~ 2 GeV, ROI |b|<20deg, |l|<20deg ~ 0.485 sr
    Flux integrated ~ 5e-7 cm^-2 s^-1 sr^-1 dans 1-10 GeV
    Modele : log-normal centre a 2 GeV, sigma 0.4 dex
    """
    E = np.asarray(E_GeV, dtype=float)
    ln_mu = math.log(2.0)
    sig = 0.4 * math.log(10)
    norm = 5.0e-7  # cm^-2 s^-1 sr^-1 (integrated 1-10 GeV)
    shape = np.where(E > 0,
                     np.exp(-0.5 * ((np.log(np.maximum(E, 1e-300)) - ln_mu) / sig) ** 2)
                     / (E * sig * math.sqrt(2 * math.pi)),
                     0.0)
    return norm * shape   # dN/dE [photons cm^-2 s^-1 sr^-1 GeV^-1]


# ---------------------------------------------------------------------------
# 4. Comparaison V5.20 vs Fermi
# ---------------------------------------------------------------------------

def ratio_v5_fermi(E_GeV: np.ndarray, delta_c: int = 10) -> np.ndarray:
    """Ratio I_V5.20(E) / I_Fermi(E)."""
    return intensity_diffuse_v5p20(E_GeV, delta_c) / edgrb_fermi(E_GeV)


def verdict_fermi_test(delta_c: int = 10) -> dict:
    """Verdict global V5.20 vs Fermi-LAT EDGRB.

    Calcule :
      * I_V5.20 et I_Fermi a 1 GeV
      * ratio max et integral V5/Fermi dans 1-100 GeV
      * Delta_c minimal pour passer sous I_Fermi (exclusion bound)
    """
    # I a 1 GeV
    E_test = np.array([1.0])
    I_V5_1GeV = float(intensity_diffuse_v5p20(E_test, delta_c)[0])
    I_F_1GeV = float(edgrb_fermi(E_test)[0])
    ratio_1GeV = I_V5_1GeV / I_F_1GeV

    # Integration 1-100 GeV
    E_int = np.logspace(0, 2, 200)
    F_V5 = np.trapezoid(intensity_diffuse_v5p20(E_int, delta_c), E_int)
    F_F = np.trapezoid(edgrb_fermi(E_int), E_int)
    ratio_int = F_V5 / F_F

    # Cherche Delta_c minimal compatible (ratio_int < 1)
    delta_c_excluded = None
    for dc in range(delta_c, 70):
        F_dc = np.trapezoid(intensity_diffuse_v5p20(E_int, dc), E_int)
        if F_dc / F_F < 1.0:
            delta_c_excluded = dc
            break

    # Verdict status
    if ratio_int > 1e3:
        status = "STRICTLY EXCLUDED"
        comment = (
            f"V5.20 (Delta_c={delta_c}) sur-predit EDGRB par {ratio_int:.2e}x. "
            f"Delta_c >= {delta_c_excluded} requis pour compatibilite."
        )
    elif ratio_int > 1.0:
        status = "EXCLUDED"
        comment = (
            f"V5.20 (Delta_c={delta_c}) sur-predit EDGRB d'un facteur "
            f"{ratio_int:.1f}. Delta_c >= {delta_c_excluded} requis."
        )
    elif ratio_int > 1e-3:
        status = "MARGINAL"
        comment = (
            f"V5.20 (Delta_c={delta_c}) contribue ~{ratio_int*100:.1f}% "
            "de l'EDGRB. Excitant si confirme."
        )
    else:
        status = "SUB-DOMINANT"
        comment = (
            f"V5.20 (Delta_c={delta_c}) sous-predit EDGRB par "
            f"{1.0/ratio_int:.2e}x. Contribution negligeable."
        )

    return {
        "delta_c": delta_c,
        "I_V5_1GeV": I_V5_1GeV,
        "I_Fermi_1GeV": I_F_1GeV,
        "ratio_1GeV": ratio_1GeV,
        "F_V5_1to100GeV": F_V5,
        "F_Fermi_1to100GeV": F_F,
        "ratio_integrated": ratio_int,
        "delta_c_excluded_above": delta_c_excluded,
        "status": status,
        "comment": comment,
    }


def gce_comparison(delta_c: int = 10) -> dict:
    """V5.20 explique-t-il le GCE ?

    GCE est galactique (pas cosmologique) -- on compare en supposant
    la meme intensite isotrope V5.20 dans la ROI |b|<20deg, |l|<20deg.

    On compare l'integral 1-10 GeV.
    """
    E = np.logspace(0, 1, 200)
    F_V5 = np.trapezoid(intensity_diffuse_v5p20(E, delta_c), E)
    F_GCE = np.trapezoid(gce_hooper(E), E)
    ratio = F_V5 / F_GCE
    if ratio > 10.0:
        verdict = "V5.20 sur-predit GCE"
    elif ratio > 0.1:
        verdict = "V5.20 peut contribuer au GCE"
    else:
        verdict = "V5.20 sous-domine GCE (autre origine)"
    return {
        "F_V5_1to10": F_V5,
        "F_GCE_1to10": F_GCE,
        "ratio": ratio,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 5. Plot 4 panels
# ---------------------------------------------------------------------------

def make_plot(delta_c_values=(10, 30, 50, 54), out_path: Path = OUTPUT_PNG):
    E = np.logspace(-1, math.log10(820.0), 400)  # 0.1 - 820 GeV
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ----- Panel (a) : spectre V5.20 pour Delta_c varies vs EDGRB -----
    ax = axes[0, 0]
    EFlux_F = E ** 2 * edgrb_fermi(E)   # E^2 dN/dE
    ax.plot(E, EFlux_F, 'k-', lw=2.0, label='Fermi-LAT EDGRB (Ackermann+ 2015)')
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(delta_c_values)))
    for c, dc in zip(colors, delta_c_values):
        EFlux_V5 = E ** 2 * intensity_diffuse_v5p20(E, dc)
        ax.plot(E, EFlux_V5, color=c, lw=1.5,
                label=f'V5.20 $\\Delta c$={dc}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$E^2 dN/dE$ (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    ax.set_title('(a) V5.20 cascade gamma vs EDGRB Fermi-LAT')
    ax.set_ylim(1e-15, 1e15)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

    # ----- Panel (b) : ratio V5/Fermi -----
    ax = axes[0, 1]
    for c, dc in zip(colors, delta_c_values):
        ratio = ratio_v5_fermi(E, dc)
        ax.plot(E, ratio, color=c, lw=1.5, label=f'$\\Delta c$={dc}')
    ax.axhline(1.0, color='r', ls='--', lw=1.0, label='compatibilite')
    ax.axhline(0.1, color='orange', ls=':', lw=1.0, label='10% Fermi')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$I_{V5.20}(E) / I_{Fermi}(E)$')
    ax.set_title('(b) Ratio V5.20 / Fermi-LAT')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # ----- Panel (c) : comparison avec GCE -----
    ax = axes[1, 0]
    E_gce = np.logspace(-0.5, 1.5, 200)
    EFlux_GCE = E_gce ** 2 * gce_hooper(E_gce)
    ax.plot(E_gce, EFlux_GCE, 'g-', lw=2.0, label='GCE (Hooper+Goodenough 2011)')
    for c, dc in zip(colors, delta_c_values):
        EFlux_V5 = E_gce ** 2 * intensity_diffuse_v5p20(E_gce, dc)
        ax.plot(E_gce, EFlux_V5, color=c, lw=1.5,
                label=f'V5.20 $\\Delta c$={dc}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E (GeV)')
    ax.set_ylabel(r'$E^2 dN/dE$ (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    ax.set_title('(c) V5.20 vs Galactic Center Excess')
    ax.set_ylim(1e-15, 1e10)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

    # ----- Panel (d) : Delta_c critique (exclusion bound) -----
    ax = axes[1, 1]
    dc_grid = np.arange(5, 60)
    E_int = np.logspace(0, 2, 200)
    F_F = np.trapezoid(edgrb_fermi(E_int), E_int)
    ratios = []
    for dc in dc_grid:
        F_V5 = np.trapezoid(intensity_diffuse_v5p20(E_int, dc), E_int)
        ratios.append(F_V5 / F_F)
    ratios = np.array(ratios)
    ax.semilogy(dc_grid, ratios, 'b-', lw=1.5)
    ax.axhline(1.0, color='r', ls='--', label='Exclusion 1x EDGRB')
    ax.axhline(0.1, color='orange', ls=':', label='Sub-dominant')
    # dc critique
    above = np.where(ratios > 1.0)[0]
    if len(above) > 0 and above[-1] + 1 < len(dc_grid):
        dc_crit = dc_grid[above[-1] + 1]
        ax.axvline(dc_crit, color='g', ls='-.', lw=1.2,
                   label=f'$\\Delta c_{{crit}}$ = {dc_crit}')
    ax.set_xlabel(r'$\Delta c$ (crossings Vassiliev)')
    ax.set_ylabel(r'$F_{V5.20}/F_{Fermi}$ (integ. 1-100 GeV)')
    ax.set_title('(d) Bound exclusion sur $\\Delta c$')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        'V5.20 cascade gamma vs Fermi-LAT EDGRB '
        '(M$_{pearl}$=2e-3 M$_\\odot$, m$_\\chi$=1.55 GeV)',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("V5.20 -- Test gamma diffus vs Fermi-LAT EDGRB")
    print("=" * 78)

    norm = cascade_photons_total()
    print(f"\nSpectre cascade normalise (integ.) = {norm:.3f} (doit ~ 1)")

    print("\n--- Intensite a 1 GeV ---")
    E_test = np.array([1.0])
    I_F = float(edgrb_fermi(E_test)[0])
    print(f"  I_Fermi(1 GeV) = {I_F:.3e} cm^-2 s^-1 sr^-1 GeV^-1")
    for dc in [10, 30, 50, 54]:
        I_V5 = float(intensity_diffuse_v5p20(E_test, dc)[0])
        print(f"  I_V5.20(1 GeV) [Dc={dc:2d}] = {I_V5:.3e}  "
              f"(ratio V5/F = {I_V5/I_F:.3e})")

    print("\n--- Verdict global ---")
    for dc in [10, 30, 50, 54]:
        v = verdict_fermi_test(dc)
        print(f"\nDelta_c = {dc} : {v['status']}")
        print(f"  ratio_int (1-100 GeV) = {v['ratio_integrated']:.3e}")
        print(f"  Delta_c_excluded above = {v['delta_c_excluded_above']}")
        print(f"  {v['comment']}")

    print("\n--- Comparaison GCE ---")
    for dc in [10, 30, 50, 54]:
        g = gce_comparison(dc)
        print(f"  Dc={dc:2d} : F_V5/F_GCE = {g['ratio']:.3e}  -> {g['verdict']}")

    print(f"\n--- Plot 4-panel ---")
    p = make_plot()
    print(f"  ecrit : {p}")


__all__ = [
    "E_CASCADE_GEV", "N_PHOTONS_PER_CASCADE",
    "cascade_spectrum_dN_dE", "cascade_photons_total",
    "n_events_per_volume_per_time",
    "intensity_diffuse_v5p20",
    "edgrb_fermi", "gce_hooper",
    "ratio_v5_fermi", "verdict_fermi_test", "gce_comparison",
    "make_plot",
]
