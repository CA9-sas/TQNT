#!/usr/bin/env python3
"""
Analyse complète ATLAS beta 4-leptons : H -> ZZ* -> 4l + recherche b' (Gen 4)
=============================================================================
Charge les 15 fichiers ROOT beta_4lep, applique la sélection standard ATLAS,
calcule la masse invariante 4-leptons, et produit une figure 6 panneaux
avec ajustement gaussien du pic de Higgs et recherche de b' a 109 GeV.

Utilise des opérations vectorisées (awkward-array + numpy) pour la performance.

Auteur : L. A. Cariou
Date   : 2026-03-31
"""

import glob
import os
import sys
import time

import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV"
FILE_PATTERN = os.path.join(DATA_DIR, "beta_4lep_*.root")
TREE_NAME = "analysis"
OUTPUT_DIR = "/home/euk9/projets/physique_nouvelle/docs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "analyse_beta_4lep_higgs.png")

M_HIGGS = 125.0
M_BPRIME = 109.0
M_ZPRIME = 159.0
M_Z = 91.1876

ELECTRON = 11
MUON = 13


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def gaussian(x, A, mu, sigma):
    """Gaussienne pour l'ajustement."""
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def gaussian_plus_bg(x, A, mu, sigma, a0, a1):
    """Gaussienne + fond lineaire."""
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + a0 + a1 * (x - 125.0)


def compute_m4l_vectorized(pt, eta, phi, e):
    """
    Calcule la masse invariante des 4 premiers leptons (vectorise sur les evenements).
    Entrees : tableaux numpy 2D (n_events, 4).
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    E_tot = np.sum(e, axis=1)
    px_tot = np.sum(px, axis=1)
    py_tot = np.sum(py, axis=1)
    pz_tot = np.sum(pz, axis=1)

    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2, 0.0))


def compute_pair_mass(pt, eta, phi, e, i, j):
    """
    Masse invariante de la paire (lepton i, lepton j) pour tous les evenements.
    pt, eta, phi, e : tableaux (n_events, 4)
    i, j : indices de leptons (0-3)
    """
    px_i = pt[:, i] * np.cos(phi[:, i])
    py_i = pt[:, i] * np.sin(phi[:, i])
    pz_i = pt[:, i] * np.sinh(eta[:, i])

    px_j = pt[:, j] * np.cos(phi[:, j])
    py_j = pt[:, j] * np.sin(phi[:, j])
    pz_j = pt[:, j] * np.sinh(eta[:, j])

    m2 = ((e[:, i] + e[:, j])**2
           - (px_i + px_j)**2
           - (py_i + py_j)**2
           - (pz_i + pz_j)**2)
    return np.sqrt(np.maximum(m2, 0.0))


def apply_zz_pairing_vectorized(pt, eta, phi, e, types, charges):
    """
    Pour chaque evenement, trouve le meilleur appariement ZZ*.
    Les 3 appariements possibles de 4 leptons en 2 paires :
      (01,23), (02,13), (03,12)
    Retourne : m_Z1, m_Z2, mask_valid (bool)
    """
    n = len(pt)

    # Les 3 facons de partitionner {0,1,2,3} en 2 paires
    pairings = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]

    best_chi2 = np.full(n, 1e30)
    best_mZ1 = np.zeros(n)
    best_mZ2 = np.zeros(n)
    best_valid = np.zeros(n, dtype=bool)

    for i1, j1, i2, j2 in pairings:
        # Verifier SFOS pour les deux paires
        sfos1 = (types[:, i1] == types[:, j1]) & (charges[:, i1] + charges[:, j1] == 0)
        sfos2 = (types[:, i2] == types[:, j2]) & (charges[:, i2] + charges[:, j2] == 0)
        both_sfos = sfos1 & sfos2

        if not np.any(both_sfos):
            continue

        m1 = compute_pair_mass(pt, eta, phi, e, i1, j1)
        m2 = compute_pair_mass(pt, eta, phi, e, i2, j2)

        # Z1 = paire la plus proche de m_Z
        swap = np.abs(m2 - M_Z) < np.abs(m1 - M_Z)
        mZ1 = np.where(swap, m2, m1)
        mZ2 = np.where(swap, m1, m2)

        chi2 = (mZ1 - M_Z)**2 + (mZ2 - M_Z)**2

        better = both_sfos & (chi2 < best_chi2)
        best_chi2[better] = chi2[better]
        best_mZ1[better] = mZ1[better]
        best_mZ2[better] = mZ2[better]
        best_valid[better] = True

    return best_mZ1, best_mZ2, best_valid


# ---------------------------------------------------------------------------
# Chargement et selection vectorises
# ---------------------------------------------------------------------------
def load_all_data():
    """Charge tous les fichiers beta_4lep et retourne les masses 4-leptons."""
    files = sorted(glob.glob(FILE_PATTERN))
    print(f"Fichiers trouves : {len(files)}", flush=True)

    if len(files) == 0:
        print("ERREUR : aucun fichier trouve !")
        sys.exit(1)

    branches = ["lep_pt", "lep_eta", "lep_phi", "lep_e",
                 "lep_type", "lep_charge", "lep_n",
                 "lep_d0sig",
                 "lep_ptvarcone30", "lep_topoetcone20"]

    all_m4l = []
    total_events = 0
    total_selected = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        t0 = time.time()
        tree = uproot.open(fpath)[TREE_NAME]
        n_events = tree.num_entries
        total_events += n_events

        # Lecture par lots
        batch_size = 100000
        file_selected = 0

        for batch_start in range(0, n_events, batch_size):
            batch_stop = min(batch_start + batch_size, n_events)
            data = tree.arrays(branches,
                               entry_start=batch_start,
                               entry_stop=batch_stop,
                               library="ak")

            # --- Etape 1 : garder les 4 premiers leptons ---
            lep_pt = data["lep_pt"][:, :4]
            lep_eta = data["lep_eta"][:, :4]
            lep_phi = data["lep_phi"][:, :4]
            lep_e = data["lep_e"][:, :4]
            lep_type = data["lep_type"][:, :4]
            lep_charge = data["lep_charge"][:, :4]
            lep_d0sig = data["lep_d0sig"][:, :4]
            lep_ptcone = data["lep_ptvarcone30"][:, :4]
            lep_etcone = data["lep_topoetcone20"][:, :4]

            # Nombre de leptons par evenement (apres troncature a 4)
            nlep = ak.num(lep_pt)

            # --- Etape 2 : coupure >= 4 leptons ---
            mask_4lep = (nlep >= 4)

            # --- Etape 3 : types valides (electrons ou muons) ---
            types_valid = (lep_type == ELECTRON) | (lep_type == MUON)
            mask_types = ak.all(types_valid, axis=1) & mask_4lep

            # Appliquer les coupures sequentiellement pour reduire le volume
            mask = mask_types

            if ak.sum(mask) == 0:
                continue

            # Convertir en numpy pour les operations vectorisees
            # (pad to length 4 first, then convert)
            sel_pt = ak.to_numpy(ak.fill_none(ak.pad_none(lep_pt[mask], 4), 0.0))
            sel_eta = ak.to_numpy(ak.fill_none(ak.pad_none(lep_eta[mask], 4), 0.0))
            sel_phi = ak.to_numpy(ak.fill_none(ak.pad_none(lep_phi[mask], 4), 0.0))
            sel_e = ak.to_numpy(ak.fill_none(ak.pad_none(lep_e[mask], 4), 0.0))
            sel_type = ak.to_numpy(ak.fill_none(ak.pad_none(lep_type[mask], 4), 0))
            sel_charge = ak.to_numpy(ak.fill_none(ak.pad_none(lep_charge[mask], 4), 0))
            sel_d0sig = ak.to_numpy(ak.fill_none(ak.pad_none(lep_d0sig[mask], 4), 999.0))
            sel_ptcone = ak.to_numpy(ak.fill_none(ak.pad_none(lep_ptcone[mask], 4), 999.0))
            sel_etcone = ak.to_numpy(ak.fill_none(ak.pad_none(lep_etcone[mask], 4), 999.0))

            n_sel = len(sel_pt)
            if n_sel == 0:
                continue

            # --- Etape 4 : coupures en pt ---
            # pt1 > 20, pt2 > 15, pt3 > 10 (les leptons sont deja tries par pt)
            pt_sorted = np.sort(sel_pt, axis=1)[:, ::-1]  # tri decroissant
            mask_pt = (pt_sorted[:, 0] > 20.0) & (pt_sorted[:, 1] > 15.0) & (pt_sorted[:, 2] > 10.0)

            # --- Etape 5 : isolation ---
            # ptvarcone30/pt < 0.3 et topoetcone20/pt < 0.3
            safe_pt = np.maximum(sel_pt, 1e-6)
            iso_pt = np.abs(sel_ptcone) / safe_pt
            iso_et = np.abs(sel_etcone) / safe_pt
            mask_iso = np.all(iso_pt < 0.3, axis=1) & np.all(iso_et < 0.3, axis=1)

            # --- Etape 6 : parametre d'impact ---
            # |d0sig| < 5 (electrons), < 3 (muons)
            d0_limit = np.where(sel_type == ELECTRON, 5.0, 3.0)
            mask_d0 = np.all(np.abs(sel_d0sig) < d0_limit, axis=1)

            # Combiner toutes les coupures
            mask_all = mask_pt & mask_iso & mask_d0

            if not np.any(mask_all):
                continue

            # Appliquer
            pt_cut = sel_pt[mask_all]
            eta_cut = sel_eta[mask_all]
            phi_cut = sel_phi[mask_all]
            e_cut = sel_e[mask_all]
            type_cut = sel_type[mask_all]
            charge_cut = sel_charge[mask_all]

            # --- Etape 7 : appariement ZZ* ---
            mZ1, mZ2, valid_zz = apply_zz_pairing_vectorized(
                pt_cut, eta_cut, phi_cut, e_cut, type_cut, charge_cut)

            # Pour les evenements avec appariement ZZ* valide :
            # coupures sur les masses des paires
            mask_z1 = (mZ1 > 50.0) & (mZ1 < 106.0)
            mask_z2 = (mZ2 > 12.0) & (mZ2 < 115.0)
            mask_zz_good = valid_zz & mask_z1 & mask_z2

            # Pour les evenements sans appariement ZZ* : accepter si charge totale = 0
            charge_total = np.sum(charge_cut, axis=1)
            mask_no_zz = (~valid_zz) & (charge_total == 0)

            mask_final = mask_zz_good | mask_no_zz

            if not np.any(mask_final):
                continue

            # Calculer m_4l
            m4l = compute_m4l_vectorized(
                pt_cut[mask_final], eta_cut[mask_final],
                phi_cut[mask_final], e_cut[mask_final])

            # Garder seulement m4l > 0
            m4l = m4l[m4l > 0]
            all_m4l.append(m4l)
            file_selected += len(m4l)

        total_selected += file_selected
        dt = time.time() - t0
        print(f"  {fname:50s} {n_events:>8d} evt -> {file_selected:>6d} sel  ({dt:.1f}s)",
              flush=True)

    print(f"\nTotal : {total_events:,d} evenements, {total_selected:,d} selectionnes",
          flush=True)

    if len(all_m4l) == 0:
        return np.array([])
    return np.concatenate(all_m4l)


# ---------------------------------------------------------------------------
# Calcul de significativite locale
# ---------------------------------------------------------------------------
def local_significance(m4l, m_target, window=5.0, bg_range=30.0):
    """
    Calcul de significativite locale autour de m_target.
    Signal : evenements dans [m_target - window, m_target + window]
    Fond : estime par interpolation lineaire des bandes laterales.
    """
    mask_sig = (m4l >= m_target - window) & (m4l <= m_target + window)
    n_sig_total = np.sum(mask_sig)

    mask_left = (m4l >= m_target - bg_range) & (m4l < m_target - window)
    mask_right = (m4l > m_target + window) & (m4l <= m_target + bg_range)

    n_left = np.sum(mask_left)
    n_right = np.sum(mask_right)
    width_sideband = 2 * (bg_range - window)
    width_sig = 2 * window

    if width_sideband > 0:
        bg_density = (n_left + n_right) / width_sideband
        n_bg = bg_density * width_sig
    else:
        n_bg = 0

    n_excess = n_sig_total - n_bg

    if n_bg > 0:
        significance = n_excess / np.sqrt(n_bg)
    else:
        significance = 0.0

    return n_sig_total, n_bg, n_excess, significance


# ---------------------------------------------------------------------------
# Figure 6 panneaux
# ---------------------------------------------------------------------------
def make_figure(m4l):
    """Produit la figure 6 panneaux."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        r"ATLAS $\beta$ 4-leptons — $H\to ZZ^*\to 4\ell$ + recherche $b'$ (Gen 4)"
        f"\n{len(m4l):,d} evenements selectionnes / 1 488 347 totaux",
        fontsize=15, fontweight="bold", y=0.98)

    COL_HIST = "#2c7bb6"
    COL_HIGGS = "#d7191c"
    COL_BP = "#fdae61"
    COL_FIT = "#d7191c"

    # =====================================================================
    # Panel 1 : spectre complet 60-300 GeV
    # =====================================================================
    ax = axes[0, 0]
    bins_full = np.linspace(60, 300, 121)  # 2 GeV bins
    ax.hist(m4l, bins=bins_full, color=COL_HIST,
            edgecolor="black", linewidth=0.3,
            label="Donnees ATLAS 13 TeV")
    ax.axvline(M_HIGGS, color=COL_HIGGS, linestyle="--", linewidth=1.5,
               label=f"$m_H$ = {M_HIGGS:.0f} GeV")
    ax.axvline(M_BPRIME, color=COL_BP, linestyle="--", linewidth=1.5,
               label=f"$m_{{b'}}$ = {M_BPRIME:.0f} GeV")
    ax.axvline(M_ZPRIME, color="green", linestyle=":", linewidth=1.2,
               label=f"159 GeV")
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel("Evenements / 2 GeV", fontsize=12)
    ax.set_title("(a) Spectre complet", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(60, 300)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # =====================================================================
    # Panel 2 : zoom 80-160 GeV
    # =====================================================================
    ax = axes[0, 1]
    bins_zoom1 = np.linspace(80, 160, 81)  # 1 GeV bins
    ax.hist(m4l, bins=bins_zoom1, color=COL_HIST,
            edgecolor="black", linewidth=0.3)
    ax.axvline(M_HIGGS, color=COL_HIGGS, linestyle="--", linewidth=1.5,
               label=f"$m_H$ = {M_HIGGS:.0f} GeV")
    ax.axvline(M_BPRIME, color=COL_BP, linestyle="--", linewidth=1.5,
               label=f"$m_{{b'}}$ = {M_BPRIME:.0f} GeV")
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel("Evenements / 1 GeV", fontsize=12)
    ax.set_title(r"(b) Zoom pic $H$ (80-160 GeV)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(80, 160)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # =====================================================================
    # Panel 3 : zoom detaille 100-140 GeV
    # =====================================================================
    ax = axes[0, 2]
    bins_zoom2 = np.linspace(100, 140, 41)  # 1 GeV bins
    ax.hist(m4l, bins=bins_zoom2, color=COL_HIST,
            edgecolor="black", linewidth=0.3)
    ax.axvline(M_HIGGS, color=COL_HIGGS, linestyle="--", linewidth=1.5,
               label=f"$m_H$ = {M_HIGGS:.0f} GeV")
    ax.axvline(M_BPRIME, color=COL_BP, linestyle="--", linewidth=1.5,
               label=f"$m_{{b'}}$ = {M_BPRIME:.0f} GeV")
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel("Evenements / 1 GeV", fontsize=12)
    ax.set_title(r"(c) Detail $H$ et $b'$ (100-140 GeV)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(100, 140)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # =====================================================================
    # Panel 4 : ajustement gaussien du pic de Higgs (115-135 GeV)
    # =====================================================================
    ax = axes[1, 0]
    fit_lo, fit_hi = 115.0, 135.0
    bins_fit = np.linspace(fit_lo, fit_hi, 41)  # 0.5 GeV bins
    counts_fit, edges_fit, _ = ax.hist(m4l, bins=bins_fit, color=COL_HIST,
                                        edgecolor="black", linewidth=0.3,
                                        label="Donnees")

    centers = 0.5 * (edges_fit[:-1] + edges_fit[1:])
    bin_width = edges_fit[1] - edges_fit[0]

    # Estimation initiale
    idx_max = np.argmax(counts_fit)
    A0 = max(counts_fit[idx_max], 1.0)
    mu0 = centers[idx_max]
    sigma0 = 2.0
    a0_init = np.median(counts_fit)
    a1_init = 0.0

    fit_success = False
    popt = None
    try:
        errors = np.sqrt(np.maximum(counts_fit, 1))

        popt, pcov = curve_fit(
            gaussian_plus_bg, centers, counts_fit,
            p0=[A0, mu0, sigma0, a0_init, a1_init],
            sigma=errors,
            absolute_sigma=True,
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))

        A_fit, mu_fit, sigma_fit, a0_fit, a1_fit = popt
        A_err, mu_err, sigma_err, a0_err, a1_err = perr

        x_fine = np.linspace(fit_lo, fit_hi, 500)
        y_fit = gaussian_plus_bg(x_fine, *popt)
        y_bg = a0_fit + a1_fit * (x_fine - 125.0)

        ax.plot(x_fine, y_fit, color=COL_FIT, linewidth=2,
                label="Gauss + fond lin.")
        ax.plot(x_fine, y_bg, color="gray", linestyle="--", linewidth=1.2,
                label="Fond")
        ax.fill_between(x_fine, y_bg, y_fit, alpha=0.2, color=COL_FIT)

        fit_text = (
            f"$\\mu$ = {mu_fit:.1f} $\\pm$ {mu_err:.1f} GeV\n"
            f"$\\sigma$ = {abs(sigma_fit):.1f} $\\pm$ {sigma_err:.1f} GeV\n"
            f"$A$ = {A_fit:.1f} $\\pm$ {A_err:.1f}"
        )
        ax.text(0.03, 0.97, fit_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        fit_success = True
    except Exception as exc:
        print(f"ATTENTION : ajustement echoue : {exc}")

    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel(f"Evenements / {bin_width:.1f} GeV", fontsize=12)
    ax.set_title(r"(d) Ajustement gaussien pic $H$", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(fit_lo, fit_hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # =====================================================================
    # Panel 5 : distribution des residus (pull)
    # =====================================================================
    ax = axes[1, 1]
    if fit_success:
        y_pred = gaussian_plus_bg(centers, *popt)
        errors_pred = np.sqrt(np.maximum(y_pred, 1))
        pull = (counts_fit - y_pred) / errors_pred

        ax.bar(centers, pull, width=bin_width * 0.9, color=COL_HIST,
               edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(2, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(-2, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylim(-4, 4)

        ndf = len(pull) - 5
        pull_text = (
            f"$\\mu_{{pull}}$ = {np.mean(pull):.2f}\n"
            f"$\\sigma_{{pull}}$ = {np.std(pull):.2f}\n"
            f"$\\chi^2$/ndf = {np.sum(pull**2):.1f}/{ndf}"
        )
        ax.text(0.03, 0.97, pull_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Ajustement echoue", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")

    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel("Pull (residus normalises)", fontsize=12)
    ax.set_title("(e) Distribution des residus", fontsize=13)
    ax.set_xlim(fit_lo, fit_hi)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # =====================================================================
    # Panel 6 : resume et significativites
    # =====================================================================
    ax = axes[1, 2]
    ax.axis("off")

    sig_109 = local_significance(m4l, M_BPRIME, window=5.0, bg_range=25.0)
    sig_125 = local_significance(m4l, M_HIGGS, window=5.0, bg_range=25.0)
    sig_159 = local_significance(m4l, M_ZPRIME, window=5.0, bg_range=25.0)

    n_80_160 = int(np.sum((m4l >= 80) & (m4l <= 160)))
    n_115_135 = int(np.sum((m4l >= 115) & (m4l <= 135)))
    ref_2020 = 832

    summary_lines = [
        r"$\bf{Resume\ de\ l'analyse}$",
        "",
        f"Fichiers charges : 15 (2015 + 2016)",
        f"Evenements totaux : 1 488 347",
        f"Selection 4-leptons : {len(m4l):,d}",
        f"Region 80-160 GeV : {n_80_160:,d}",
        f"Region 115-135 GeV : {n_115_135:,d}",
        "",
        r"$\bf{Significativites\ locales}$"
        r" (fenetre $\pm$5 GeV) :",
        "",
        f"  $m_{{b'}}$ = 109 GeV :",
        f"     signal = {sig_109[0]:.0f}, fond = {sig_109[1]:.1f},"
        f" exces = {sig_109[2]:.1f}",
        f"     significativite = {sig_109[3]:.2f} $\\sigma$",
        "",
        f"  $m_H$ = 125 GeV :",
        f"     signal = {sig_125[0]:.0f}, fond = {sig_125[1]:.1f},"
        f" exces = {sig_125[2]:.1f}",
        f"     significativite = {sig_125[3]:.2f} $\\sigma$",
        "",
        f"  159 GeV :",
        f"     signal = {sig_159[0]:.0f}, fond = {sig_159[1]:.1f},"
        f" exces = {sig_159[2]:.1f}",
        f"     significativite = {sig_159[3]:.2f} $\\sigma$",
        "",
        r"$\bf{Comparaison\ donnees\ 2020}$"
        f" ({ref_2020} evt) :",
        f"  Facteur d'enrichissement : x{len(m4l)/ref_2020:.1f}" if ref_2020 > 0 else "",
    ]

    if fit_success:
        summary_lines += [
            "",
            r"$\bf{Ajustement\ pic\ Higgs}$" " :",
            f"  $\\mu$ = {mu_fit:.2f} $\\pm$ {mu_err:.2f} GeV",
            f"  $\\sigma$ = {abs(sigma_fit):.2f} $\\pm$ {sigma_err:.2f} GeV",
        ]

    text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="gray", alpha=0.95))
    ax.set_title("(f) Resume et significativites", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nFigure sauvegardee : {OUTPUT_FILE}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ATLAS beta 4-leptons : H -> ZZ* -> 4l + recherche b' (Gen 4)")
    print("=" * 70, flush=True)

    t_start = time.time()
    m4l = load_all_data()
    t_load = time.time() - t_start
    print(f"\nChargement termine en {t_load:.1f}s")

    if len(m4l) == 0:
        print("ERREUR : aucun evenement selectionne !")
        sys.exit(1)

    print(f"\nStatistiques m_4l :")
    print(f"  Min    = {m4l.min():.1f} GeV")
    print(f"  Max    = {m4l.max():.1f} GeV")
    print(f"  Median = {np.median(m4l):.1f} GeV")
    print(f"  Moyenne = {m4l.mean():.1f} GeV")

    make_figure(m4l)

    t_total = time.time() - t_start
    print(f"\nTemps total : {t_total:.1f}s")
    print("Termine.")
