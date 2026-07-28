#!/usr/bin/env python3
"""V5.20 -- Analyse complète de la physique interne d'une perle DM.

Script principal V5.20 produisant le panel 4-fenêtres
``output/v5p20_pearl_internal_dynamics.png`` :

  (a) N(M_pearl, m_knot) : nombre de nœuds vs masse de la perle
       pour les 6 STRICT amphichiraux.
  (b) Profil radial rho(r), P(r) d'une perle typique (M=2e-3 M_sun,
       m_knot=1.55 GeV).
  (c) Taux d'enchevêtrement Gamma_total(delta_c) : suppression Vassiliev
       en fonction du nombre de crossings introduits.
  (d) N_events sur l'univers observable et statut.

Run :
    PYTHONPATH=. python3 scripts/V5_pearl_internal/dm_pearl_full_analysis.py
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
    DM_STRICT_AMPHI_GEV,
    M_PEARL_TYPICAL_MSUN,
    M_SUN_KG,
    ALPHA_EM,
    SIGMA_QED,
    T_HUBBLE_S,
    n_knots,
    tov_pearl,
    chandrasekhar_topological_pearl,
    chandrasekhar_white_dwarf,
    chandrasekhar_ratio,
    sigma_entanglement,
    entanglement_rate,
    n_events_over_universe_age,
    verdict_v5p20,
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_pearl_internal_dynamics.png"


# ---------------------------------------------------------------------------
# Panel (a) : N(M_pearl, m_knot)
# ---------------------------------------------------------------------------

def panel_a_n_knots(ax):
    M_grid = np.logspace(-6, 0, 100)  # M_sun
    for m_knot in DM_STRICT_AMPHI_GEV:
        N = np.array([n_knots(M, m_knot) for M in M_grid])
        ax.plot(M_grid, N, label=f"$m_\\chi$ = {m_knot:.2f} GeV", lw=1.5)
    ax.axvline(M_PEARL_TYPICAL_MSUN, color="k", ls="--", alpha=0.5,
               label=f"$M_{{\\rm pearl,typ}}$ = {M_PEARL_TYPICAL_MSUN}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm pearl}$ ($M_\odot$)")
    ax.set_ylabel(r"$N_{\rm knots}$")
    ax.set_title(r"(a) Nombre de n\oe{}uds DM par perle")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Panel (b) : profil rho(r), P(r)
# ---------------------------------------------------------------------------

def panel_b_profile(ax):
    res = tov_pearl(M_PEARL_TYPICAL_MSUN, 1.55)
    r_km = res["r"] / 1000.0
    # Normalise rho_c et P_c pour superposition
    rho_norm = res["rho"] / res["rho"][0]
    P_norm = res["P"] / res["P"][0]
    ax.plot(r_km, rho_norm, "b-", lw=2, label=r"$\rho(r)/\rho_c$")
    ax.plot(r_km, P_norm, "r--", lw=2, label=r"$P(r)/P_c$")
    R_km = res["R"] / 1000.0
    ax.axvline(R_km, color="g", ls=":", lw=1.5,
               label=f"$R_{{\\rm pearl}}$ = {R_km:.2f} km")
    ax.set_xlabel("r (km)")
    ax.set_ylabel("Profil normalis\\'e")
    ax.set_title(
        r"(b) Profil TOV pearl ($M=2\!\times\!10^{-3}\,M_\odot$,"
        r" $m_\chi=1.55$ GeV)" + "\n"
        rf"$\rho_c = {res['rho_c']:.2e}$ kg/m$^3$"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Panel (c) : Gamma_total(delta_c) -- suppression Vassiliev
# ---------------------------------------------------------------------------

def panel_c_gamma(ax):
    delta_c_grid = np.arange(2, 65, 2)
    sigmas = np.array([sigma_entanglement(int(dc)) for dc in delta_c_grid])
    sigmas_cm2 = sigmas * 1e4
    Gammas = []
    for dc in delta_c_grid:
        r = entanglement_rate(M_PEARL_TYPICAL_MSUN, 1.55, int(dc))
        Gammas.append(r["Gamma_total_pearl_Hz"])
    Gammas = np.array(Gammas)
    ax.plot(delta_c_grid, sigmas_cm2, "b-", lw=2, label=r"$\sigma$ (cm$^2$)")
    ax2 = ax.twinx()
    ax2.plot(delta_c_grid, Gammas, "r-", lw=2,
             label=r"$\Gamma_{\rm pearl}$ (Hz)")
    ax2.set_yscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta c$ (Vassiliev suppression)")
    ax.set_ylabel(r"$\sigma$ (cm$^2$)", color="b")
    ax2.set_ylabel(r"$\Gamma_{\rm pearl}$ (Hz)", color="r")
    ax.set_title(r"(c) Cross-section et taux par perle vs $\Delta c$")
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Panel (d) : N_events sur univers observable
# ---------------------------------------------------------------------------

def panel_d_events(ax):
    delta_c_grid = np.arange(2, 65, 2)
    N_events = []
    for dc in delta_c_grid:
        v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, int(dc))
        N_events.append(v["N_events_total_universe"])
    N_events = np.array(N_events)
    ax.plot(delta_c_grid, N_events, "k-", lw=2,
            label=r"$N_{\rm events,univ}$")
    ax.axhline(1.0, color="g", ls="--", lw=1.5, label=r"1 (seuil rare)")
    ax.axhline(1e-3, color="r", ls=":", lw=1.5, label=r"$10^{-3}$ (excl.)")
    # Annoter le delta_c critique pour N_events ~ 1
    # Trouver intersection
    idx_below = np.where(N_events < 1.0)[0]
    if len(idx_below) > 0:
        dc_crit = delta_c_grid[idx_below[0]]
        ax.axvline(dc_crit, color="b", ls=":", alpha=0.5,
                   label=f"$\\Delta c_{{\\rm crit}}\\approx{dc_crit}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta c$ (Vassiliev suppression)")
    ax.set_ylabel(r"$N_{\rm events}$ univers observable")
    ax.set_title(r"(d) Verdict V5.20 : enchev\^etrement vs $\Delta c$")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Assemble full figure
# ---------------------------------------------------------------------------

def make_plot(path: Path = OUTPUT_PNG) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panel_a_n_knots(axes[0, 0])
    panel_b_profile(axes[0, 1])
    panel_c_gamma(axes[1, 0])
    panel_d_events(axes[1, 1])
    fig.suptitle(
        "V5.20 -- Physique interne d'une perle DM "
        "(Class A topologique)",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary for CLI / tests
# ---------------------------------------------------------------------------

def summary() -> dict:
    res = tov_pearl(M_PEARL_TYPICAL_MSUN, 1.55)
    N = n_knots(M_PEARL_TYPICAL_MSUN, 1.55)
    M_Ch = chandrasekhar_topological_pearl(1.55)
    out = {
        "M_pearl_Msun": M_PEARL_TYPICAL_MSUN,
        "m_knot_GeV": 1.55,
        "N_knots": N,
        "R_pearl_km": res["R"] / 1000.0,
        "rho_central_kg_m3": float(res["rho_c"]),
        "n_central_m3": float(res["n"][0]),
        "M_Ch_pearl_Msun": M_Ch / M_SUN_KG,
        "M_Ch_WD_Msun": chandrasekhar_white_dwarf() / M_SUN_KG,
        "Chandrasekhar_ratio_pearl_over_WD": chandrasekhar_ratio(1.55),
    }
    # Verdict at delta_c=10 (typical Vassiliev for STRICT amphichirals)
    for dc in (5, 10, 15, 20, 30, 40, 50, 60):
        v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, dc)
        out[f"N_events_total_universe_dc{dc}"] = v["N_events_total_universe"]
        out[f"status_dc{dc}"] = v["status"]
    # delta_c critique pour N_events ~ 1
    for dc in range(2, 100):
        v = verdict_v5p20(M_PEARL_TYPICAL_MSUN, 1.55, dc)
        if v["N_events_total_universe"] < 1.0:
            out["delta_c_critical"] = dc
            break
    return out


if __name__ == "__main__":
    print("V5.20 -- Generating analysis plot...")
    p = make_plot()
    print(f"  -> {p}")
    s = summary()
    print("\nSummary:")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")
