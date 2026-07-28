#!/usr/bin/env python3
"""
V5.20 -- Test du flux diffus de neutrinos sub-PeV vs IceCube 10-yr.

Question physique
-----------------
V5.20 (theory/dm_pearl_internal.py + dm_pearls_paper) predit, par perle DM
Class A, un evenement d'enchevetrement topologique spontane rare mais
non strictement nul. Sur l'ensemble du volume de Hubble et integre sur
t_Hubble, le nombre d'evenements est de l'ordre ~10^92 (delta_c=10) pour
les perles 6_3+6_3 (m_knot = 1,55 GeV).

Chaque evenement libere ~ m_link c^2 ~ 1,5 GeV en cascade hadronic. Les
mesons pi^+/pi^-/pi^0 se desintegrent en photons (gamma test FermiLAT) ET
en neutrinos via pi^+ -> mu nu, mu -> e nu nu. On estime la fraction
neutrinique de la cascade a ~ 15 % de l'energie totale (branching cascade
hadronic standard).

Le spectre des neutrinos suit une loi de puissance soft (E^{-2,5})
typique de cascade hadronic, avec un pic autour de E_nu ~ 0,1-0,3 GeV
(via le pic du muon a ~ 50 MeV) et un cutoff effectif a E_max ~ 0,75 GeV
(moitie de l'energie de cascade dans le referentiel comobile).

Donnees observationnelles
-------------------------
* IceCube HESE 7-yr (Aartsen+ 2018) : phi(E) = 6,7e-18 (E/100 TeV)^{-2,5}
  GeV^{-1} cm^{-2} s^{-1} sr^{-1} per flavor
* IceCube 10-yr astrophysical (Abbasi+ 2022, ApJ 928, 50) :
  gamma = 2,87 +/- 0,2, normalisation comparable, range 16 TeV - 2,6 PeV

Verdict attendu
---------------
Le pic V5.20 nu (autour de 0,1-1 GeV) est largement sous le seuil
astrophysique d'IceCube (> 10 TeV). Aux energies du seuil, le flux V5.20
est essentiellement nul (tail E^{-2,5} extrapole au-dela du cutoff).
Donc : V5.20 nu sub-dominant, compatible avec IceCube.

Produit
-------
* output/v5p20_icecube_test.png : 2 panneaux
    (a) E^2 dN/dE vs E pour V5.20 (rouge), IceCube astro (noir), nu
        atmospherique (bleu pointille).
    (b) ratio phi_V5 / phi_IceCube en fonction de E.

Lancement
---------
    PYTHONPATH=. python3 scripts/V5_IceCube/v5p20_neutrino_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (
    C_LIGHT,
    GEV_KG,
    M_SUN_KG,
    T_HUBBLE_S,
    V_HUBBLE_M3,
    n_events_over_universe_age,
)


# ---------------------------------------------------------------------------
# Constantes du test V5.20 -- IceCube
# ---------------------------------------------------------------------------

# Energie liberee par evenement d'enchevetrement (cascade hadronic)
# m_link ~ 2 x m_knot = 2 x 1,55 GeV -> 3,1 GeV deposes dans une cascade
# qui produit gamma + nu. On retient E_cascade ~ 1,5 GeV en moyenne par
# voie (la moitie va aux gammas, moitie aux nu et e^+/-).
E_CASCADE_GEV = 1.5

# Fraction de l'energie de cascade qui finit dans les neutrinos
# (cascade hadronic GeV : ~ 15 % typique)
F_NU_BRANCHING = 0.15

# Multiplicite : ~3-5 nu par evenement (3 saveurs x 1-2 nu/saveur)
N_NU_PER_EVENT = 4.0

# Indice spectral cascade hadronic (typique E^{-2,5})
GAMMA_CASCADE = 2.5

# Energie de coupure superieure (moitie de l'energie de cascade)
E_CUTOFF_GEV = 0.75

# Energie minimale (~ masse muon * facteur Lorentz minimal)
E_MIN_GEV = 0.01

# Parametres pearl Class A (V5.20)
M_PEARL_TYPICAL_MSUN = 2.0e-3
M_KNOT_TYPICAL_GEV = 1.55
DELTA_C_DEFAULT = 10


# ---------------------------------------------------------------------------
# Donnees IceCube
# ---------------------------------------------------------------------------

# IceCube HESE 7-yr per-flavor astrophysical normalization
# phi(E) = phi_0 * (E / E_0)^{-gamma_obs}
# Aartsen+ 2018, ApJ 833 3 (HESE 7yr)
ICECUBE_PHI0 = 6.7e-18           # GeV^-1 cm^-2 s^-1 sr^-1 per flavor
ICECUBE_E0_GEV = 1.0e5           # 100 TeV
ICECUBE_GAMMA = 2.5              # 7-yr value, compatible 10-yr 2.87 +/- 0.2
ICECUBE_GAMMA_10YR = 2.87        # Abbasi+ 2022, ApJ 928, 50
ICECUBE_GAMMA_10YR_SIGMA = 0.20

# Range de sensibilite IceCube (HESE + tracks 10-yr)
ICECUBE_E_MIN_GEV = 1.6e4        # ~16 TeV
ICECUBE_E_MAX_GEV = 2.6e6        # ~2,6 PeV


def icecube_diffuse_flux(E_GeV: np.ndarray,
                          phi0: float = ICECUBE_PHI0,
                          gamma: float = ICECUBE_GAMMA) -> np.ndarray:
    """Flux diffus astrophysique IceCube, per flavor (GeV^-1 cm^-2 s^-1 sr^-1).

    phi(E) = phi0 * (E / 100 TeV)^{-gamma}
    """
    return phi0 * (np.asarray(E_GeV) / ICECUBE_E0_GEV) ** (-gamma)


def atmospheric_neutrino_flux(E_GeV: np.ndarray) -> np.ndarray:
    """Flux atmospherique conventionnel approxime (Honda+ 2007 type).

    Loi de puissance brisee : phi ~ E^{-2.7} dans 1-1000 GeV, plus dur
    au-dessus du TeV (~ E^{-3.7} apres pic).

    Forme approximative ici (per flavor, dominant nu_mu) :
        phi_atm(E) = 1.5e-1 * E^{-2.7} (E < 100 GeV)
        phi_atm(E) = 1.5e-1 * (E/100)^{-3.7} * 100^{-2.7} (E > 100 GeV)

    Unites : GeV^-1 cm^-2 s^-1 sr^-1.
    """
    E = np.asarray(E_GeV, dtype=float)
    out = np.zeros_like(E)
    low = E <= 100.0
    high = E > 100.0
    out[low] = 1.5e-1 * E[low] ** (-2.7)
    out[high] = 1.5e-1 * (100.0) ** (-2.7) * (E[high] / 100.0) ** (-3.7)
    return out


# ---------------------------------------------------------------------------
# Spectre neutrino V5.20
# ---------------------------------------------------------------------------

def v5p20_neutrino_spectrum_per_event(E_GeV: np.ndarray,
                                       gamma: float = GAMMA_CASCADE,
                                       E_cut_GeV: float = E_CUTOFF_GEV,
                                       E_min_GeV: float = E_MIN_GEV,
                                       n_nu: float = N_NU_PER_EVENT
                                       ) -> np.ndarray:
    """dN_nu/dE par evenement d'enchevetrement (GeV^-1).

    Spectre soft normalise tel que :
        integral_{E_min}^{E_cut} dN/dE dE = n_nu

    Forme : dN/dE = A * E^{-gamma}  pour E_min <= E <= E_cut, 0 sinon.

    Pour E > E_cut, le flux est cutoff exponentiellement (cascade epuisee).
    """
    E = np.asarray(E_GeV, dtype=float)
    out = np.zeros_like(E)

    # Normalisation A telle que integral E^{-gamma} dE = n_nu
    # int_{Emin}^{Ecut} E^{-gamma} dE = (E^{1-gamma}/(1-gamma)) evaluated
    if abs(gamma - 1.0) < 1e-6:
        norm = math.log(E_cut_GeV / E_min_GeV)
    else:
        norm = (E_cut_GeV ** (1.0 - gamma) - E_min_GeV ** (1.0 - gamma)) \
               / (1.0 - gamma)
    A = n_nu / norm

    # Power-law + exponential cutoff au-dessus de E_cut
    mask = E >= E_min_GeV
    out[mask] = A * E[mask] ** (-gamma)
    above = E > E_cut_GeV
    out[above] = A * E_cut_GeV ** (-gamma) \
                 * np.exp(-(E[above] - E_cut_GeV) / E_cut_GeV)
    return out


def v5p20_event_density() -> float:
    """Densite volumique d'evenements d'enchevetrement (m^-3 s^-1).

    n_event_dot = N_events_total_universe / (V_Hubble * t_Hubble)

    Avec ~10^92 evenements/Hubble/Hubble_volume (delta_c=10) :
        n_event_dot ~ 10^92 / (4e80 m^3 * 4,35e17 s) ~ 6e-7 m^-3 s^-1
    """
    res = n_events_over_universe_age(
        M_PEARL_TYPICAL_MSUN, M_KNOT_TYPICAL_GEV, DELTA_C_DEFAULT
    )
    N_total = res["N_events_total_universe"]
    return N_total / (V_HUBBLE_M3 * T_HUBBLE_S)


def v5p20_diffuse_flux(E_GeV: np.ndarray,
                        gamma: float = GAMMA_CASCADE,
                        E_cut_GeV: float = E_CUTOFF_GEV,
                        E_min_GeV: float = E_MIN_GEV
                        ) -> np.ndarray:
    """Flux diffus de neutrinos V5.20 (GeV^-1 cm^-2 s^-1 sr^-1) per flavor.

    Calcul :
        phi(E) = (1 / 4 pi) * n_event_dot * dN_nu/dE * c * t_H

    n_event_dot : taux par unite de volume comobile (m^-3 s^-1)
    dN_nu/dE   : spectre par evenement (GeV^-1)
    c * t_H    : longueur de Hubble (horizon de propagation)

    Le facteur 1 / (3 saveurs) -> normalisation par saveur (oscillations
    mixent en 1:1:1 sur distance cosmologique).
    """
    n_event_dot = v5p20_event_density()
    dN_dE = v5p20_neutrino_spectrum_per_event(
        E_GeV, gamma=gamma, E_cut_GeV=E_cut_GeV, E_min_GeV=E_min_GeV
    )

    # Profondeur d'horizon (c * t_H en cm)
    L_horizon_cm = C_LIGHT * T_HUBBLE_S * 100.0  # 100 -> m to cm

    # phi(E) per flavor per sr = n_dot * dN/dE * L_horizon / (4 pi * 3)
    # n_event_dot doit etre converti m^-3 -> cm^-3
    n_dot_cm3 = n_event_dot * 1.0e-6

    phi_total = n_dot_cm3 * dN_dE * L_horizon_cm / (4.0 * math.pi)
    phi_per_flavor = phi_total / 3.0  # 3 saveurs apres oscillation
    return phi_per_flavor


# ---------------------------------------------------------------------------
# Comparaison et verdict
# ---------------------------------------------------------------------------

def compare_v5_vs_icecube(E_GeV: np.ndarray) -> Dict[str, np.ndarray]:
    """Compare phi_V5 et phi_IceCube par bin d'energie.

    Renvoie un dict :
      'E_GeV', 'phi_V5', 'phi_IceCube', 'phi_atm', 'ratio'
    """
    phi_v5 = v5p20_diffuse_flux(E_GeV)
    phi_ic = icecube_diffuse_flux(E_GeV)
    phi_atm = atmospheric_neutrino_flux(E_GeV)
    ratio = np.where(phi_ic > 0, phi_v5 / phi_ic, np.nan)
    return {
        "E_GeV": E_GeV,
        "phi_V5": phi_v5,
        "phi_IceCube": phi_ic,
        "phi_atm": phi_atm,
        "ratio": ratio,
    }


def verdict(E_GeV: np.ndarray) -> Dict[str, object]:
    """Verdict global V5.20 vs IceCube.

    Note importante : la cascade V5.20 a un cutoff dur a E ~ 0.75 GeV
    (energie de cascade par evenement). Le flux astrophysique d'IceCube
    est mesure dans la bande 16 TeV - 2.6 PeV. Comme E_cut << 1 TeV,
    le spectre V5.20 est exponentiellement supprime dans toute la bande
    de sensibilite IceCube. Le verdict en bande est donc essentiellement
    "indiscernable de zero", c.a.d. STRICTLY SUB-DOMINANT.

    On evalue :
      - ratio_max_in_IceCube_band : phi_V5/phi_IceCube max dans [16 TeV, 2.6 PeV]
      - ratio_1TeV                : ratio a 1 TeV (juste sous la bande)
      - ratio_atm_GeV_peak        : ratio phi_V5/phi_atm au pic GeV (control)

    Statut :
      - STRICTLY SUB-DOMINANT (compatible)  si ratio_max < 1e-6
      - SUB-DOMINANT (compatible)           si ratio_max < 1
      - COMPARABLE (tension)                si 1 <= ratio_max < 3
      - OVERPREDICTED (V5.20 disfavored)    si ratio_max >= 3
    """
    cmp = compare_v5_vs_icecube(E_GeV)
    in_band = (cmp["E_GeV"] >= ICECUBE_E_MIN_GEV) & \
              (cmp["E_GeV"] <= ICECUBE_E_MAX_GEV)
    if not np.any(in_band):
        ratio_max = np.nan
    else:
        # filter NaN out (np.where(0/0) gives NaN)
        sub = cmp["ratio"][in_band]
        finite = np.isfinite(sub)
        ratio_max = float(np.nanmax(sub[finite])) if np.any(finite) else 0.0

    # Ratio specifique a 1 TeV (juste sous la bande IceCube)
    phi_v5_1TeV = float(v5p20_diffuse_flux(np.array([1.0e3]))[0])
    phi_ic_1TeV = float(icecube_diffuse_flux(np.array([1.0e3]))[0])
    ratio_1TeV = phi_v5_1TeV / phi_ic_1TeV if phi_ic_1TeV > 0 else np.nan

    # Pic GeV : verification que V5 ne sur-predit pas le fond atmospherique
    phi_v5_atm = float(v5p20_diffuse_flux(np.array([1.0]))[0])
    phi_atm_1GeV = float(atmospheric_neutrino_flux(np.array([1.0]))[0])
    ratio_atm_GeV = phi_v5_atm / phi_atm_1GeV if phi_atm_1GeV > 0 else np.nan

    if not np.isfinite(ratio_max) or ratio_max < 1e-6:
        status = "STRICTLY SUB-DOMINANT (compatible)"
    elif ratio_max < 1.0:
        status = "SUB-DOMINANT (compatible)"
    elif ratio_max < 3.0:
        status = "COMPARABLE (tension)"
    else:
        status = "OVERPREDICTED (V5.20 disfavored)"

    return {
        "phi_V5_1TeV": phi_v5_1TeV,
        "phi_IceCube_1TeV": phi_ic_1TeV,
        "ratio_1TeV": ratio_1TeV,
        "ratio_max_in_IceCube_band": ratio_max,
        "phi_V5_1GeV": phi_v5_atm,
        "phi_atm_1GeV": phi_atm_1GeV,
        "ratio_V5_atm_1GeV": ratio_atm_GeV,
        "status": status,
        "n_event_density_m3_s": v5p20_event_density(),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_v5_vs_icecube(out_path: Path) -> None:
    E = np.logspace(-2, 8, 400)  # 0.01 GeV - 100 PeV
    cmp = compare_v5_vs_icecube(E)
    verd = verdict(E)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) E^2 dN/dE
    ax = axes[0]
    E2 = E ** 2
    ax.loglog(E, E2 * cmp["phi_V5"], color="red", lw=2,
              label=r"V5.20 pearl cascade ($\nu$)")
    ax.loglog(E, E2 * cmp["phi_IceCube"], color="black", lw=2,
              label=r"IceCube astrophysical (Aartsen+ 2018, 7yr)")
    ax.loglog(E, E2 * cmp["phi_atm"], color="blue", ls="--", lw=1.5,
              label=r"Atmospheric $\nu$ (approx.)")
    ax.axvspan(ICECUBE_E_MIN_GEV, ICECUBE_E_MAX_GEV,
               color="gray", alpha=0.15, label="IceCube band")
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$E^2 \phi_\nu$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$] (per flavor)")
    ax.set_title("V5.20 vs IceCube 10-yr diffuse flux")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(1e-2, 1e8)
    ax.set_ylim(1e-25, 1e-2)
    ax.grid(True, which="both", alpha=0.3)

    # (b) ratio
    ax = axes[1]
    ax.loglog(E, cmp["ratio"], color="purple", lw=2)
    ax.axhline(1.0, color="red", ls="--", alpha=0.6,
               label="V5 = IceCube")
    ax.axhline(1e-3, color="orange", ls=":", alpha=0.6,
               label="V5 = 0.1% IceCube")
    ax.axvspan(ICECUBE_E_MIN_GEV, ICECUBE_E_MAX_GEV,
               color="gray", alpha=0.15, label="IceCube band")
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$\phi_{V5} / \phi_{IceCube}$")
    ax.set_title(f"Ratio (max in band = {verd['ratio_max_in_IceCube_band']:.2e})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(1e-2, 1e8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"V5.20 pearl cascade $\\nu$ vs IceCube  --  status: {verd['status']}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print(" V5.20 -- Sub-PeV neutrino flux vs IceCube 10-yr")
    print("=" * 78)

    n_dot = v5p20_event_density()
    print(f"\n  Densite d'evenements V5.20      : {n_dot:.3e} m^-3 s^-1")
    print(f"  Cascade energy par evenement    : {E_CASCADE_GEV:.2f} GeV")
    print(f"  Fraction nu branching           : {F_NU_BRANCHING:.2f}")
    print(f"  Multiplicite nu/evenement        : {N_NU_PER_EVENT:.1f}")
    print(f"  Cutoff energy                    : {E_CUTOFF_GEV:.2f} GeV")

    E_probe = np.array([1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
    print("\n  Phi(E) per flavor (GeV^-1 cm^-2 s^-1 sr^-1) :")
    print("  " + "-" * 64)
    print(f"  {'E [GeV]':>10}  {'phi_V5':>12}  {'phi_IceCube':>12}  {'ratio':>10}")
    for E in E_probe:
        phi_v5 = float(v5p20_diffuse_flux(np.array([E]))[0])
        phi_ic = float(icecube_diffuse_flux(np.array([E]))[0])
        r = phi_v5 / phi_ic if phi_ic > 0 else float("nan")
        print(f"  {E:10.1e}  {phi_v5:12.3e}  {phi_ic:12.3e}  {r:10.3e}")

    E_all = np.logspace(-2, 8, 400)
    verd = verdict(E_all)
    print("\n  --- Verdict V5.20 ---")
    print(f"  phi_V5(1 TeV)         = {verd['phi_V5_1TeV']:.3e}")
    print(f"  phi_IceCube(1 TeV)    = {verd['phi_IceCube_1TeV']:.3e}")
    print(f"  ratio V5/IceCube(1TeV)= {verd['ratio_1TeV']:.3e}")
    print(f"  max ratio en bande IC = {verd['ratio_max_in_IceCube_band']:.3e}")
    print(f"  phi_V5(1 GeV)         = {verd['phi_V5_1GeV']:.3e}")
    print(f"  phi_atm(1 GeV)        = {verd['phi_atm_1GeV']:.3e}")
    print(f"  ratio V5/atm(1 GeV)   = {verd['ratio_V5_atm_1GeV']:.3e}")
    print(f"  status                = {verd['status']}")

    out_path = ROOT / "output" / "v5p20_icecube_test.png"
    plot_v5_vs_icecube(out_path)
    print(f"\n  Plot ecrit : {out_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "E_CASCADE_GEV", "F_NU_BRANCHING", "N_NU_PER_EVENT",
    "GAMMA_CASCADE", "E_CUTOFF_GEV", "E_MIN_GEV",
    "ICECUBE_PHI0", "ICECUBE_E0_GEV", "ICECUBE_GAMMA",
    "ICECUBE_E_MIN_GEV", "ICECUBE_E_MAX_GEV",
    "icecube_diffuse_flux", "atmospheric_neutrino_flux",
    "v5p20_neutrino_spectrum_per_event",
    "v5p20_event_density", "v5p20_diffuse_flux",
    "compare_v5_vs_icecube", "verdict", "plot_v5_vs_icecube",
]
