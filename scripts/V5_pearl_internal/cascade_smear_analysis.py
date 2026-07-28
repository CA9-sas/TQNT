#!/usr/bin/env python3
"""V5.20 Class E -- Cascade smear analysis vs Fermi-LAT line search criterion.

CADRE
=====
V5.20 Class E (commit 0208e72) prédit 6 raies γ aux 6 masses des
amphichiraux STRICT (1.55, 4.70, 16.01, 17.24, 26.48, 51.78 GeV).
Le test Fermi-LAT line-search (commit 06ad81f) montre que le Δc_min
requis pour passer sous Ackermann+ 2015 vaut ≥ 30, alors que le candidat
de sélection V5.21 donne Δc = 22 (insuffisant).

QUESTION TESTÉE
===============
Si la désintégration d'un link n=2 ne produit PAS une raie δ étroite mais
un SPECTRE CONTINU via cascade hadronique (analogue WIMP DM annihilation
→ pi^0 → γγ + bremsstrahlung + Compton inverse), alors les line searches
Fermi (qui recherchent strictement ΔE/E ≲ 10%) ne s'appliquent PAS.

PHYSIQUE DE LA CASCADE (link n=2, P15)
======================================
Énergie totale E_tot = m_link (1.55, ..., 51.78 GeV) → cascade vers ~10-20
particules secondaires (γ, e±, μ±, ν, π^0, π±). Modèle PYTHIA-like
(Cirelli+ 2011 -- "PPPC4DMID" -- ; Bhattacherjee+ 2018).

Spectre γ par link désintégré (modèle minimal physiquement motivé) :

  dN/dx ≈ (a/x) * exp(-b*x) + c * x^p * (1-x)^q          (x = 2E/m_link)

avec a, b, c, p, q calibrés sur la forme générique d'un canal hadronique
"bb-like" à m_DM ~ GeV (Cirelli+ 2011 Fig.4). On a typiquement :
  * ⟨x⟩ ~ 0.1-0.2  (énergie moyenne par photon ~ m/10 → m/5)
  * multiplicité M_γ ~ 10-15
  * pic dN/dE à E_peak ~ 0.02-0.05 × m_link
  * spectre s'étend de E_min ~ 0.001 m_link à E_max ~ 0.7 m_link

LIVRABLE PRINCIPAL
==================
Comparer FWHM_cascade(m_link) à ΔE_Fermi(E) = 0.12 × E :
  * Si FWHM_cascade >> ΔE_Fermi(E_peak) → line search NE S'APPLIQUE PAS
  * Re-tester V5.20 Class E continuum (cascade smeared) vs Fermi EDGRB
    (Ackermann+ 2015) -- borne diffuse continuum, plus laxe que line

Run :
    PYTHONPATH=. python3 scripts/V5_pearl_internal/cascade_smear_analysis.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "V5_pearl_internal"))
sys.path.insert(0, str(ROOT / "scripts" / "V5_Fermi_LAT"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dm_pearl_internal import (
    ALPHA_EM,
    DM_STRICT_AMPHI_GEV,
    M_PEARL_TYPICAL_MSUN,
    R_HUBBLE_M,
    T_HUBBLE_S,
    V_HUBBLE_M3,
    n_events_over_universe_age,
)

import class_E_entanglement_test as CE
import class_E_six_lines_predictions as SL


OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "v5p20_cascade_smear.png"


# ---------------------------------------------------------------------------
# Constantes Fermi
# ---------------------------------------------------------------------------

# Fermi-LAT line search: typical resolution ΔE/E ≈ 10-15% in 1-100 GeV.
# Ackermann+ 2015 PRD 91 122002 explicitly quotes ΔE/E ~ 9% (P7REP_CLEAN_V15)
# rising to ~14% above 10 GeV (PSF + energy reco degradation). We take 12%
# as the canonical Fermi line-search width (same constant as in
# class_E_six_lines_predictions).
FERMI_DE_OVER_E = 0.12

# Line search criterion: a putative line is rejected as smooth continuum
# if its observed FWHM > k * ΔE_inst, with k ~ 1.5 (Ackermann+ 2015
# Sec.III, "line search significance"). Empirically: FWHM_apparent /
# ΔE_inst > 1.5 → not classified as a line.
LINE_VS_CONTINUUM_K = 1.5

GEV_TO_MEV = 1.0e3
SR_FULL = 4.0 * math.pi

# Cosmology (same as v5p20_gamma_diffuse_test.py)
H0_SI = 67.4e3 / 3.0857e22
G_NEWTON = 6.67430e-11
RHO_CRIT = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_NEWTON)
OMEGA_DM = 0.265

# EDGRB Ackermann+ 2015 (PL+cutoff fit) — same as v5p20_gamma_diffuse_test
EDGRB_I0_per_MeV = 1.12e-7
EDGRB_ALPHA = 2.32
EDGRB_E_CUT_GeV = 279.0


# ---------------------------------------------------------------------------
# 1. Modèle de cascade par link n=2
# ---------------------------------------------------------------------------
#
# On utilise un modèle minimal physiquement motivé. Pour un link n=2 de
# masse m_link qui se désintègre, on suppose un mélange hadronique :
#
#   (i)  fragmentation type "bb-like" PYTHIA  -> spectre soft :
#        dN/dx = A * x^(-1.5) * exp(-7.5 x)        (Cirelli+ 2011 fit)
#        où x = 2 E_gamma / m_link.
#   (ii) photons "primaires" durs (1-photon ISR + π^0 boost) :
#        dN/dx_hard = B * (1-x)^2 / sqrt(x)        normalisé
#        autour de x ~ 0.3.
#   (iii) fraction relative : f_soft = 0.85, f_hard = 0.15 (typique
#        canal hadronique à 1-50 GeV).
#
# Cette forme reproduit qualitativement les spectres tabulés Cirelli+
# 2011 pour les canaux b\bar b, q\bar q, gg, W^+W^-, tau^+tau^- dans la
# fenêtre 1-50 GeV ; la précision est suffisante pour évaluer FWHM
# (largeur ~ ordre de grandeur de m_link).
# ---------------------------------------------------------------------------

def cascade_spectrum_per_link(E_GeV: np.ndarray, m_link_GeV: float
                              ) -> np.ndarray:
    """dN/dE_γ (GeV^-1) pour 1 link n=2 de masse m_link désintégré.

    Mélange soft (fragmentation) + hard (π^0/ISR). Normalisé pour donner
    M_γ ~ 10 photons par link (multiplicité hadronique typique).
    """
    E = np.asarray(E_GeV, dtype=float)
    x = 2.0 * E / m_link_GeV

    # Soft component (PYTHIA-like, Cirelli+ 2011 fit b\bar b)
    # dN/dx = a * x^(-1.5) * exp(-7.5 x) for 1e-4 < x < 1
    # Normalisation : integral_{1e-4}^{1} a x^{-1.5} e^{-7.5 x} dx ≈ a * 178.6
    x_safe = np.clip(x, 1e-4, 1.0)
    soft = np.where((x >= 1e-4) & (x <= 1.0),
                    x_safe ** (-1.5) * np.exp(-7.5 * x_safe),
                    0.0)
    # Normalize so int dN_soft/dx dx = 1 (per channel; multiplicity attached after)
    norm_soft = 1.0 / 178.6
    soft = soft * norm_soft

    # Hard component (π^0 box-like, ISR primary photons)
    # dN/dx = b * (1-x)^2 / sqrt(x), x in [1e-3, 0.95]
    # int 0..1 (1-x)^2 / sqrt(x) dx = 2*(1 - 2*2/3 + 1/3) actually = 16/15
    hard = np.where((x >= 1e-3) & (x <= 0.95),
                    (1.0 - np.clip(x, 0, 0.95)) ** 2 / np.sqrt(x_safe),
                    0.0)
    norm_hard = 15.0 / 16.0
    hard = hard * norm_hard

    f_soft, f_hard = 0.85, 0.15
    M_gamma = 10.0     # multiplicité photons (par link)

    # Combine: dN/dx then convert to dN/dE (dx/dE = 2/m_link)
    dN_dx = M_gamma * (f_soft * soft + f_hard * hard)
    dN_dE = dN_dx * (2.0 / m_link_GeV)
    return dN_dE


def cascade_multiplicity(m_link_GeV: float, n_pts: int = 4000) -> float:
    """Multiplicité γ par link : M_γ = ∫ dN/dE dE."""
    E = np.logspace(-5, math.log10(0.95 * m_link_GeV / 2.0), n_pts) * m_link_GeV / 2.0
    # simpler grid in E
    E = np.logspace(math.log10(1e-4 * m_link_GeV), math.log10(0.6 * m_link_GeV), n_pts)
    dNdE = cascade_spectrum_per_link(E, m_link_GeV)
    return float(np.trapezoid(dNdE, E))


def cascade_mean_energy(m_link_GeV: float, n_pts: int = 4000) -> float:
    """Énergie moyenne ⟨E_γ⟩ = (∫ E dN/dE dE) / M_γ."""
    E = np.logspace(math.log10(1e-4 * m_link_GeV), math.log10(0.6 * m_link_GeV), n_pts)
    dNdE = cascade_spectrum_per_link(E, m_link_GeV)
    M = np.trapezoid(dNdE, E)
    if M <= 0:
        return 0.0
    return float(np.trapezoid(E * dNdE, E) / M)


def cascade_peak_energy(m_link_GeV: float, n_pts: int = 4000) -> float:
    """Énergie au pic E^2 dN/dE (SED peak)."""
    E = np.logspace(math.log10(1e-4 * m_link_GeV), math.log10(0.6 * m_link_GeV), n_pts)
    dNdE = cascade_spectrum_per_link(E, m_link_GeV)
    SED = E ** 2 * dNdE
    idx = int(np.argmax(SED))
    return float(E[idx])


def cascade_FWHM(m_link_GeV: float, mode: str = "dNdE", n_pts: int = 8000
                 ) -> Tuple[float, float, float]:
    """FWHM du spectre cascade (en énergie absolue, GeV).

    mode = "dNdE" : FWHM du dN/dE
    mode = "ESED" : FWHM du E^2 dN/dE (SED, plus pertinent pour line search)

    Retourne (E_low_half, E_high_half, FWHM_GeV).
    """
    E = np.logspace(math.log10(1e-4 * m_link_GeV), math.log10(0.6 * m_link_GeV), n_pts)
    spec = cascade_spectrum_per_link(E, m_link_GeV)
    if mode == "ESED":
        curve = E ** 2 * spec
    else:
        curve = spec
    if not np.any(curve > 0):
        return 0.0, 0.0, 0.0
    peak = float(np.max(curve))
    half = peak / 2.0
    # Find leftmost and rightmost crossings of half
    above = curve >= half
    idx = np.where(above)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0
    i_lo, i_hi = idx[0], idx[-1]
    # Linear interp for sub-bin precision
    def interp_cross(i, direction):
        if direction == "lo":
            if i == 0:
                return E[0]
            x0, x1 = E[i - 1], E[i]
            y0, y1 = curve[i - 1], curve[i]
        else:
            if i == len(E) - 1:
                return E[-1]
            x0, x1 = E[i], E[i + 1]
            y0, y1 = curve[i], curve[i + 1]
        if y1 == y0:
            return x0
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    E_lo = interp_cross(i_lo, "lo")
    E_hi = interp_cross(i_hi, "hi")
    return float(E_lo), float(E_hi), float(E_hi - E_lo)


# ---------------------------------------------------------------------------
# 2. Comparaison FWHM cascade vs résolution Fermi line search
# ---------------------------------------------------------------------------

def fermi_FWHM_at_E(E_GeV: float) -> float:
    """Résolution instrumentale Fermi-LAT à l'énergie E (FWHM, GeV)."""
    return FERMI_DE_OVER_E * E_GeV


def line_vs_continuum_verdict(m_link_GeV: float) -> Dict:
    """Pour un link n=2 de masse m_link, juge si cascade smear masque la raie.

    Critère :
      * Calculer FWHM_cascade à partir du SED (E^2 dN/dE)
      * Calculer FWHM_Fermi(E_peak_SED) = 0.12 × E_peak
      * Si FWHM_cascade > k × FWHM_Fermi (k=1.5) → CONTINUUM, line search ne s'applique pas
      * Sinon → reste assimilable à une raie (line search exclu)
    """
    E_peak = cascade_peak_energy(m_link_GeV)
    E_lo, E_hi, FWHM_casc = cascade_FWHM(m_link_GeV, mode="ESED")
    FWHM_fermi = fermi_FWHM_at_E(E_peak) if E_peak > 0 else 0.0
    if FWHM_fermi <= 0:
        ratio = 0.0
    else:
        ratio = FWHM_casc / FWHM_fermi
    is_smeared = ratio > LINE_VS_CONTINUUM_K
    return {
        "m_link_GeV": m_link_GeV,
        "E_peak_SED_GeV": E_peak,
        "FWHM_cascade_GeV": FWHM_casc,
        "FWHM_Fermi_at_peak_GeV": FWHM_fermi,
        "ratio_FWHM_casc_over_Fermi": ratio,
        "is_smeared_continuum": bool(is_smeared),
        "M_gamma": cascade_multiplicity(m_link_GeV),
        "mean_E_gamma_GeV": cascade_mean_energy(m_link_GeV),
    }


def class_E_cascade_smear_table() -> List[Dict]:
    """Boucle sur les 6 amphichiraux Class E."""
    return [line_vs_continuum_verdict(m) for m in DM_STRICT_AMPHI_GEV]


# ---------------------------------------------------------------------------
# 3. Intensité diffuse Class E avec cascade smear
# ---------------------------------------------------------------------------

def n_events_per_volume_per_time_classE(m_link_GeV: float, delta_c: int
                                        ) -> float:
    """Taux events m^-3 s^-1 pour 1 amphichiral Class E.

    Utilise la même formule que dm_pearl_internal mais avec l'abondance
    Class E (Ω_E/6 par amphichiral, soit Ω_per_amphi = 0.10/6 × Ω_DM).
    """
    nev = n_events_over_universe_age(M_PEARL_TYPICAL_MSUN, m_link_GeV,
                                     delta_c=delta_c)
    # n_events_over_universe_age utilise OMEGA_A_FRACTION = 0.05 en interne.
    # Rescale : OMEGA_PER_AMPHI / OMEGA_A_FRACTION.
    omega_per_amphi = (0.10 / 6.0) * OMEGA_DM
    omega_A_frac = 0.05 * OMEGA_DM
    rescale = omega_per_amphi / omega_A_frac
    return (nev["N_events_total_universe"] * rescale
            / (V_HUBBLE_M3 * T_HUBBLE_S))


def intensity_class_E_cascade(E_GeV: np.ndarray, delta_c: int = 22,
                              amphi_masses=DM_STRICT_AMPHI_GEV
                              ) -> np.ndarray:
    """Intensité diffuse Class E SOMMÉE sur les 6 amphichiraux (cascade smeared).

    Pour chaque amphichiral i :
        Q_i(E) = n_events_rate_i * dN/dE_cascade(E, m_i)
    Intensité totale :
        dI/dE = R_Hubble * Σ_i Q_i(E) / (4π)  [cm^-2 s^-1 sr^-1 GeV^-1]
    """
    E = np.asarray(E_GeV, dtype=float)
    total = np.zeros_like(E)
    for m_i in amphi_masses:
        rate_i = n_events_per_volume_per_time_classE(m_i, delta_c)
        Q_i_per_m3_s_GeV = rate_i * cascade_spectrum_per_link(E, m_i)
        # m^-2 s^-1 sr^-1 GeV^-1
        dI_dE_m2 = R_HUBBLE_M * Q_i_per_m3_s_GeV / SR_FULL
        total = total + dI_dE_m2
    # m^-2 -> cm^-2
    return total / 1.0e4


def edgrb_fermi(E_GeV: np.ndarray, with_cutoff: bool = True) -> np.ndarray:
    """EDGRB Fermi-LAT Ackermann+ 2015 (cm^-2 s^-1 sr^-1 GeV^-1)."""
    E_MeV = np.asarray(E_GeV, dtype=float) * GEV_TO_MEV
    flux_per_MeV = EDGRB_I0_per_MeV * (E_MeV / 100.0) ** (-EDGRB_ALPHA)
    if with_cutoff:
        flux_per_MeV = flux_per_MeV * np.exp(
            -np.asarray(E_GeV) / EDGRB_E_CUT_GeV)
    return flux_per_MeV * GEV_TO_MEV


def verdict_class_E_cascade_vs_EDGRB(delta_c: int = 22) -> Dict:
    """V5.20 Class E (cascade smeared) vs EDGRB Fermi-LAT.

    Mode : line search NE S'APPLIQUE PAS (cascade smear domine la
    résolution instrumentale -> à valider par
    class_E_cascade_smear_table). On compare au continuum EDGRB.
    """
    E_int = np.logspace(0, 2, 400)   # 1-100 GeV
    I_V5 = intensity_class_E_cascade(E_int, delta_c=delta_c)
    I_F = edgrb_fermi(E_int)
    # Ratio peak et intégrale
    ratio_pt = float(np.max(I_V5 / I_F))
    F_V5 = float(np.trapezoid(I_V5, E_int))
    F_F = float(np.trapezoid(I_F, E_int))
    ratio_int = F_V5 / F_F if F_F > 0 else float('inf')

    # Δc_min pour passer sous EDGRB
    dc_min = None
    for dc in range(delta_c, 70):
        I_dc = intensity_class_E_cascade(E_int, delta_c=dc)
        if np.max(I_dc / I_F) < 1.0:
            dc_min = dc
            break

    if ratio_int < 1.0 and ratio_pt < 1.0:
        status = "COMPATIBLE EDGRB"
        comment = (
            f"Cascade smear: Class E (Δc={delta_c}) sous EDGRB partout "
            f"(ratio_max={ratio_pt:.2e})."
        )
    elif ratio_pt < 1.0 and ratio_int > 1.0:
        status = "MARGINAL EDGRB"
        comment = (
            f"Ratio peak {ratio_pt:.2e} < 1 mais intégrale {ratio_int:.2e} "
            "excède: tension partielle."
        )
    else:
        status = "EXCLUDED EDGRB"
        comment = (
            f"Cascade smear: Class E (Δc={delta_c}) excède EDGRB par "
            f"{ratio_pt:.2e}x au peak. Δc ≥ {dc_min} requis."
        )
    return {
        "delta_c": delta_c,
        "ratio_peak_V5_over_EDGRB": ratio_pt,
        "ratio_integrated_1to100GeV": ratio_int,
        "F_V5_1to100": F_V5,
        "F_EDGRB_1to100": F_F,
        "delta_c_min_for_compatibility": dc_min,
        "status": status,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# 4. Synthèse globale
# ---------------------------------------------------------------------------

def global_summary(delta_c: int = 22) -> Dict:
    """Compile tout :
    - 6 amphichiraux : FWHM_cascade, FWHM_Fermi, smeared status
    - Verdict line search applicability (combien de 6 sont smeared)
    - Verdict V5 Class E continuum (cascade smeared) vs EDGRB
    """
    smear_table = class_E_cascade_smear_table()
    n_smeared = sum(1 for r in smear_table if r["is_smeared_continuum"])
    n_total = len(smear_table)

    edgrb_verdict = verdict_class_E_cascade_vs_EDGRB(delta_c=delta_c)

    if n_smeared == n_total:
        line_verdict = "LINE SEARCH NE S'APPLIQUE PAS"
        comment_line = (
            f"Toutes les {n_total} raies amphichirales sont SMEARED en "
            f"continuum (FWHM_cascade > {LINE_VS_CONTINUUM_K}×FWHM_Fermi)."
        )
    elif n_smeared > 0:
        line_verdict = "LINE SEARCH PARTIELLEMENT APPLICABLE"
        comment_line = (
            f"{n_smeared}/{n_total} raies smeared en continuum; "
            f"{n_total - n_smeared}/{n_total} restent assimilables à des raies."
        )
    else:
        line_verdict = "LINE SEARCH S'APPLIQUE (raies discrètes)"
        comment_line = (
            f"Aucune raie smeared : Fermi line search (Ackermann+ 2015) "
            "exclu V5.20 Class E."
        )

    if (n_smeared == n_total
            and edgrb_verdict["status"] == "COMPATIBLE EDGRB"):
        global_status = "V5.20 CLASS E REDEEMED (cascade smear + continuum compat.)"
    elif n_smeared == n_total:
        global_status = f"PARTIAL: smeared mais {edgrb_verdict['status']}"
    else:
        global_status = "V5.20 CLASS E TOUJOURS RÉFUTÉ (raies non smeared)"

    return {
        "n_smeared": n_smeared,
        "n_total": n_total,
        "line_search_verdict": line_verdict,
        "line_search_comment": comment_line,
        "smear_table": smear_table,
        "edgrb_verdict": edgrb_verdict,
        "global_status": global_status,
    }


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

def make_plot(path: Path = OUTPUT_PNG, delta_c: int = 22) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Panel (a) : spectres cascade par amphichiral ----
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DM_STRICT_AMPHI_GEV)))
    for m, col in zip(DM_STRICT_AMPHI_GEV, colors):
        E = np.logspace(math.log10(1e-4 * m), math.log10(0.6 * m), 400)
        dNdE = cascade_spectrum_per_link(E, m)
        ax.plot(E, E ** 2 * dNdE, color=col, lw=1.8,
                label=f"m={m:.2f} GeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\gamma$ (GeV)")
    ax.set_ylabel(r"$E^2 dN/dE$ par link (GeV)")
    ax.set_title("(a) Spectre cascade $E^2 dN/dE$ par link n=2 décomposé")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel (b) : FWHM cascade vs FWHM Fermi (au peak SED) ----
    ax = axes[0, 1]
    table = class_E_cascade_smear_table()
    masses = [t["m_link_GeV"] for t in table]
    fwhm_casc = [t["FWHM_cascade_GeV"] for t in table]
    fwhm_fermi = [t["FWHM_Fermi_at_peak_GeV"] for t in table]
    x = np.arange(len(masses))
    w = 0.35
    ax.bar(x - w/2, fwhm_casc, w, color="C0", alpha=0.85,
           label="FWHM_cascade (SED)")
    ax.bar(x + w/2, fwhm_fermi, w, color="C3", alpha=0.85,
           label=f"FWHM_Fermi @ E_peak ({FERMI_DE_OVER_E*100:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:.2f}" for m in masses], fontsize=8)
    ax.set_xlabel(r"$m_{\rm link}$ (GeV)")
    ax.set_ylabel("FWHM (GeV)")
    ax.set_yscale("log")
    ax.set_title("(b) FWHM cascade vs résolution instrumentale Fermi")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # ---- Panel (c) : ratio FWHM_casc / FWHM_Fermi ----
    ax = axes[1, 0]
    ratios = [t["ratio_FWHM_casc_over_Fermi"] for t in table]
    bar_colors = ["green" if r > LINE_VS_CONTINUUM_K else "red" for r in ratios]
    ax.bar(x, ratios, color=bar_colors, alpha=0.85)
    ax.axhline(LINE_VS_CONTINUUM_K, color="k", ls="--",
               label=f"Seuil continuum (k={LINE_VS_CONTINUUM_K})")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:.2f}" for m in masses], fontsize=8)
    ax.set_xlabel(r"$m_{\rm link}$ (GeV)")
    ax.set_ylabel("FWHM_cascade / FWHM_Fermi")
    ax.set_yscale("log")
    ax.set_title("(c) Smear ratio : > k → continuum, ≤ k → raie")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # ---- Panel (d) : intensité diffuse V5 Class E (smeared) vs EDGRB ----
    ax = axes[1, 1]
    E_grid = np.logspace(-1, 2.5, 400)
    I_V5_22 = intensity_class_E_cascade(E_grid, delta_c=22)
    I_V5_30 = intensity_class_E_cascade(E_grid, delta_c=30)
    I_F = edgrb_fermi(E_grid)
    ax.plot(E_grid, I_F, "k-", lw=1.8,
            label="EDGRB Fermi (Ackermann+ 2015)")
    ax.plot(E_grid, I_V5_22, "C0--", lw=1.6,
            label=r"V5.20 Class E smeared, $\Delta c=22$")
    ax.plot(E_grid, I_V5_30, "C2:", lw=1.6,
            label=r"V5.20 Class E smeared, $\Delta c=30$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E (GeV)")
    ax.set_ylabel(r"$dI/dE$ (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$)")
    ax.set_title("(d) Cascade smeared : V5.20 Class E vs EDGRB diffuse")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(1e-30, 1e0)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "V5.20 Class E -- Cascade smear : largeur raies vs Fermi line search",
        fontsize=13, weight="bold",
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
    print("V5.20 Class E -- Cascade smear vs Fermi line search criterion")
    print("=" * 78)

    print("\n--- Spectre cascade par amphichiral ---")
    print(f"{'m_link (GeV)':>14s} {'E_peak SED':>11s} {'FWHM casc':>11s} "
          f"{'FWHM Fermi':>11s} {'ratio':>10s} {'M_gamma':>9s} {'smeared?':>10s}")
    for r in class_E_cascade_smear_table():
        print(f"{r['m_link_GeV']:>14.2f} "
              f"{r['E_peak_SED_GeV']:>11.3f} "
              f"{r['FWHM_cascade_GeV']:>11.3f} "
              f"{r['FWHM_Fermi_at_peak_GeV']:>11.3f} "
              f"{r['ratio_FWHM_casc_over_Fermi']:>10.2f} "
              f"{r['M_gamma']:>9.2f} "
              f"{('OUI' if r['is_smeared_continuum'] else 'NON'):>10s}")

    print("\n--- Verdict cascade smeared vs EDGRB ---")
    for dc in (22, 26, 30):
        v = verdict_class_E_cascade_vs_EDGRB(delta_c=dc)
        print(f"  Δc={dc:>2d}: ratio_peak={v['ratio_peak_V5_over_EDGRB']:.3e}  "
              f"ratio_int(1-100GeV)={v['ratio_integrated_1to100GeV']:.3e}  "
              f"dc_min_compat={v['delta_c_min_for_compatibility']}  "
              f"-> {v['status']}")

    print("\n--- Synthèse globale (Δc=22, V5.21 fiducial) ---")
    s = global_summary(delta_c=22)
    print(f"  Smeared : {s['n_smeared']}/{s['n_total']}  ({s['line_search_verdict']})")
    print(f"  {s['line_search_comment']}")
    print(f"  EDGRB    : {s['edgrb_verdict']['status']}  ({s['edgrb_verdict']['comment']})")
    print(f"  GLOBAL   : {s['global_status']}")

    p = make_plot(delta_c=22)
    print(f"\nPlot -> {p}")
