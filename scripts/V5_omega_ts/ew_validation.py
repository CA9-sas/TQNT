"""
Validation TQNT-V5 du secteur électrofaible sur ATLAS open data (13 TeV).

Compare les prédictions Ω_ts (V5, chap. 14) aux pics observés :
  - Higgs   : m_H = 124.63 GeV (6_3+8_2+8_3)      via di-photon, fenêtre [105, 150] GeV
  - Z       : m_Z =  90.56 GeV (7_2+8_3+9_1)      via di-lepton OS, fenêtre [60, 120] GeV
  - W       : m_W =  80.68 GeV (7_1+7_2+8_3)      (optionnel, mass transverse)

Méthode :
  - chargement vectorisé de toutes les périodes data_A/B/C/D
  - sélections minimales (qualité TightID, photon E_T > 25/35 GeV, lepton p_T > 25 GeV)
  - construction vectorielle de la masse invariante
  - fit Crystal Ball (signal) + polynôme (background) via scipy.optimize.curve_fit
  - extraction de m_pic et σ_m, comparaison à la prédiction V5

Inspiré de scripts/V5_heavy_protons/lhc_search_v2_fit_subtract.py
"""

import sys
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/euk9/projets/physique_nouvelle")
from theory.omega_ts import predict_mass_V5

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
M_E = 0.000511  # GeV
M_MU = 0.105658

PDG = {
    "W": 80.377,
    "Z": 91.1876,
    "H": 125.25,
}

V5_CONFIG = {
    "W": "7_1+7_2+8_3",
    "Z": "7_2+8_3+9_1",
    "H": "6_3+8_2+8_3",
    "top": "6_2+7_4+7_5",
}

# V5 predictions (MeV -> GeV)
V5 = {k: predict_mass_V5(cfg) / 1000.0 for k, cfg in V5_CONFIG.items()}

DATA_DIR = Path("/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV")
OUT = Path("/home/euk9/projets/physique_nouvelle/output")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Kinematics (vectorisé, pas de loop Python par event)
# ---------------------------------------------------------------------------
def inv_mass_two_massless(pt1, eta1, phi1, pt2, eta2, phi2):
    """Masse invariante pour deux objets sans masse (photons), vectorisé."""
    px1 = pt1 * np.cos(phi1); py1 = pt1 * np.sin(phi1); pz1 = pt1 * np.sinh(eta1)
    px2 = pt2 * np.cos(phi2); py2 = pt2 * np.sin(phi2); pz2 = pt2 * np.sinh(eta2)
    E1 = np.sqrt(px1 * px1 + py1 * py1 + pz1 * pz1)
    E2 = np.sqrt(px2 * px2 + py2 * py2 + pz2 * pz2)
    E = E1 + E2
    px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
    m2 = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2, 0.0))


def inv_mass_two_with_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1 = pt1 * np.cos(phi1); py1 = pt1 * np.sin(phi1); pz1 = pt1 * np.sinh(eta1)
    px2 = pt2 * np.cos(phi2); py2 = pt2 * np.sin(phi2); pz2 = pt2 * np.sinh(eta2)
    E1 = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
    E2 = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
    E = E1 + E2
    px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
    m2_inv = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(np.maximum(m2_inv, 0.0))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_diphoton(files):
    """Charge la masse invariante m(γγ) à partir des deux photons leading
    qualité TightID (E_T1 > 35 GeV, E_T2 > 25 GeV, |η| < 2.37 hors crack).
    Retourne le tableau numpy des masses en GeV."""
    masses = []
    n_events_total = 0
    n_pairs = 0
    for path in files:
        try:
            f = uproot.open(path)
            tree = f["mini"]
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        n = tree.num_entries
        n_events_total += n
        print(f"  loading {path.name}  n_events={n}")
        arrs = tree.arrays(
            ["photon_n", "photon_pt", "photon_eta", "photon_phi",
             "photon_isTightID", "trigP"],
            entry_stop=n,
        )
        # Pré-sélection : trigger photon + au moins 2 photons
        mask = (arrs["photon_n"] >= 2) & (arrs["trigP"] == True)
        sel = arrs[mask]
        if len(sel) == 0:
            continue

        # Garde les deux photons leading (ATLAS open data : photons déjà triés par p_T)
        pt = sel["photon_pt"][:, :2] / 1000.0          # MeV -> GeV
        eta = sel["photon_eta"][:, :2]
        phi = sel["photon_phi"][:, :2]
        tight = sel["photon_isTightID"][:, :2]

        pt1 = np.asarray(pt[:, 0]); pt2 = np.asarray(pt[:, 1])
        eta1 = np.asarray(eta[:, 0]); eta2 = np.asarray(eta[:, 1])
        phi1 = np.asarray(phi[:, 0]); phi2 = np.asarray(phi[:, 1])
        t1 = np.asarray(tight[:, 0]).astype(bool)
        t2 = np.asarray(tight[:, 1]).astype(bool)

        # Coupures qualité
        eta_ok = lambda e: (np.abs(e) < 2.37) & ~((np.abs(e) > 1.37) & (np.abs(e) < 1.52))
        good = (
            t1 & t2
            & (pt1 > 35.0) & (pt2 > 25.0)
            & eta_ok(eta1) & eta_ok(eta2)
        )
        if not good.any():
            continue
        m = inv_mass_two_massless(pt1[good], eta1[good], phi1[good],
                                   pt2[good], eta2[good], phi2[good])
        masses.append(m)
        n_pairs += int(good.sum())

    if not masses:
        return np.array([]), n_events_total, 0
    return np.concatenate(masses), n_events_total, n_pairs


def load_dilepton(files):
    """Charge la masse invariante m(l+l-) à partir des deux leptons leading
    opposite-sign même saveur (ee ou μμ), qualité TightID, p_T > 25 GeV.
    Retourne (masses_GeV, n_events_total, n_pairs_OS_SF)."""
    masses = []
    n_events_total = 0
    n_pairs = 0
    for path in files:
        try:
            f = uproot.open(path)
            tree = f["mini"]
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        n = tree.num_entries
        n_events_total += n
        print(f"  loading {path.name}  n_events={n}")
        arrs = tree.arrays(
            ["lep_n", "lep_pt", "lep_eta", "lep_phi",
             "lep_charge", "lep_type", "lep_isTightID",
             "trigE", "trigM"],
            entry_stop=n,
        )
        # Au moins 2 leptons, trigger e OU mu
        mask = (arrs["lep_n"] >= 2) & ((arrs["trigE"] == True) | (arrs["trigM"] == True))
        sel = arrs[mask]
        if len(sel) == 0:
            continue

        pt = sel["lep_pt"][:, :2] / 1000.0
        eta = sel["lep_eta"][:, :2]
        phi = sel["lep_phi"][:, :2]
        ch = sel["lep_charge"][:, :2]
        typ = sel["lep_type"][:, :2]
        tight = sel["lep_isTightID"][:, :2]

        pt1 = np.asarray(pt[:, 0]); pt2 = np.asarray(pt[:, 1])
        eta1 = np.asarray(eta[:, 0]); eta2 = np.asarray(eta[:, 1])
        phi1 = np.asarray(phi[:, 0]); phi2 = np.asarray(phi[:, 1])
        ch1 = np.asarray(ch[:, 0]); ch2 = np.asarray(ch[:, 1])
        ty1 = np.asarray(typ[:, 0]); ty2 = np.asarray(typ[:, 1])
        t1 = np.asarray(tight[:, 0]).astype(bool)
        t2 = np.asarray(tight[:, 1]).astype(bool)

        # OS, same flavour (11=e, 13=μ), p_T > 25 GeV, tight
        good = (
            t1 & t2
            & (pt1 > 25.0) & (pt2 > 25.0)
            & (ch1 * ch2 < 0)
            & (ty1 == ty2)
            & ((ty1 == 11) | (ty1 == 13))
            & (np.abs(eta1) < 2.5) & (np.abs(eta2) < 2.5)
        )
        if not good.any():
            continue
        # Masse lepton : e ~ 0, μ = 0.105658
        m_lep = np.where(ty1[good] == 13, M_MU, M_E)
        m = inv_mass_two_with_mass(pt1[good], eta1[good], phi1[good], m_lep,
                                    pt2[good], eta2[good], phi2[good], m_lep)
        masses.append(m)
        n_pairs += int(good.sum())

    if not masses:
        return np.array([]), n_events_total, 0
    return np.concatenate(masses), n_events_total, n_pairs


# ---------------------------------------------------------------------------
# Crystal Ball + polynôme (modèle générique)
# ---------------------------------------------------------------------------
def crystal_ball(x, m, sigma, alpha, n):
    """Crystal Ball PDF unnormalisée (forme standard)."""
    t = (x - m) / sigma
    abs_a = abs(alpha)
    a_term = (n / abs_a) ** n * np.exp(-0.5 * alpha * alpha)
    b_term = n / abs_a - abs_a
    tail = a_term * np.maximum(b_term - t, 1e-30) ** (-n)
    core = np.exp(-0.5 * t * t)
    return np.where(t > -alpha, core, tail)


def model_higgs(x, m, sigma, alpha, n, N, a0, a1, a2, a3):
    """CB signal + polynôme cubique pour le continuum γγ (Drell-Yan, jets faking γ)."""
    sig = N * crystal_ball(x, m, sigma, alpha, n)
    bkg = a0 + a1 * x + a2 * x * x + a3 * x * x * x
    return sig + np.maximum(bkg, 0)


def model_z(x, m, sigma, alpha, n, N, a0, a1, a2):
    """CB signal + polynôme quadratique pour le pic Z."""
    sig = N * crystal_ball(x, m, sigma, alpha, n)
    bkg = a0 + a1 * x + a2 * x * x
    return sig + np.maximum(bkg, 0)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_higgs(masses, fit_range=(105.0, 150.0), nbins=90):
    mask = (masses >= fit_range[0]) & (masses <= fit_range[1])
    m_sel = masses[mask]
    n_data, edges = np.histogram(m_sel, bins=nbins, range=fit_range)
    centers = (edges[:-1] + edges[1:]) / 2
    sigma_data = np.sqrt(n_data + 1)

    n_med = float(np.median(n_data))
    # estimation amplitude signal : excès au-dessus du médian autour de 125
    near = (centers > 123) & (centers < 128)
    N_guess = max(1.0, float(n_data[near].max() - n_med))

    p0 = [125.0, 1.7, 1.4, 5.0, N_guess,
          n_med, 0.0, 0.0, 0.0]
    bounds_lo = [120.0, 0.8, 0.3, 0.8, 0.0,
                 -1e7, -1e7, -1e7, -1e7]
    bounds_hi = [130.0, 4.0, 8.0, 30.0, float(n_data.max()) * 10,
                 1e7, 1e7, 1e7, 1e7]
    try:
        popt, pcov = curve_fit(model_higgs, centers, n_data, p0=p0,
                               sigma=sigma_data, absolute_sigma=True,
                               bounds=(bounds_lo, bounds_hi), maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        print(f"  Fit Higgs non convergé : {e}")
        return centers, n_data, None, None, None
    n_fit = model_higgs(centers, *popt)
    return centers, n_data, n_fit, popt, perr


def fit_z(masses, fit_range=(60.0, 120.0), nbins=120):
    mask = (masses >= fit_range[0]) & (masses <= fit_range[1])
    m_sel = masses[mask]
    n_data, edges = np.histogram(m_sel, bins=nbins, range=fit_range)
    centers = (edges[:-1] + edges[1:]) / 2
    sigma_data = np.sqrt(n_data + 1)

    n_med = float(np.median(n_data))
    p0 = [91.2, 2.5, 1.2, 3.0, float(n_data.max()) * 0.8,
          n_med, 0.0, 0.0]
    bounds_lo = [85.0, 1.0, 0.3, 0.8, 0.0, -1e8, -1e8, -1e8]
    bounds_hi = [97.0, 6.0, 8.0, 30.0, float(n_data.max()) * 10,
                 1e8, 1e8, 1e8]
    try:
        popt, pcov = curve_fit(model_z, centers, n_data, p0=p0,
                               sigma=sigma_data, absolute_sigma=True,
                               bounds=(bounds_lo, bounds_hi), maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        print(f"  Fit Z non convergé : {e}")
        return centers, n_data, None, None, None
    n_fit = model_z(centers, *popt)
    return centers, n_data, n_fit, popt, perr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_higgs(centers, n_data, n_fit, popt, perr, n_pairs, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    sigma_data = np.sqrt(n_data + 1)
    ax1.errorbar(centers, n_data, yerr=sigma_data,
                 fmt='o', markersize=3, color='black',
                 label=f"ATLAS m(γγ)  (n={int(n_data.sum())})")
    if popt is not None:
        ax1.plot(centers, n_fit, 'b-', linewidth=2,
                 label="Fit : Crystal Ball + cubic")
        bkg = popt[5] + popt[6]*centers + popt[7]*centers**2 + popt[8]*centers**3
        bkg = np.maximum(bkg, 0)
        sig = popt[4] * crystal_ball(centers, popt[0], popt[1], popt[2], popt[3])
        ax1.plot(centers, bkg, 'g--', alpha=0.6, label="continuum (cubic)")
        ax1.plot(centers, sig + bkg, 'r:', alpha=0.0)
        ax1.plot(centers, sig, 'r-', alpha=0.7,
                 label=f"signal CB  m={popt[0]:.2f}±{perr[0]:.2f} GeV  σ={popt[1]:.2f} GeV")
    ax1.axvline(V5["H"], color='magenta', ls='--', lw=1.8,
                label=f"V5 prédit  m_H = {V5['H']:.2f} GeV")
    ax1.axvline(PDG["H"], color='cyan', ls=':', lw=1.8,
                label=f"PDG  m_H = {PDG['H']:.2f} GeV")
    ax1.set_ylabel("Events / bin")
    ax1.set_title(f"ATLAS Higgs → γγ — {n_pairs} paires sélectionnées")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)

    if popt is not None:
        bkg_full = popt[5] + popt[6]*centers + popt[7]*centers**2 + popt[8]*centers**3
        bkg_full = np.maximum(bkg_full, 0)
        residual = n_data - bkg_full
        ax2.errorbar(centers, residual, yerr=sigma_data,
                     fmt='o', markersize=3, color='purple',
                     label="Data − background")
        sig_curve = popt[4] * crystal_ball(centers, popt[0], popt[1], popt[2], popt[3])
        ax2.plot(centers, sig_curve, 'r-', lw=2, label="signal CB")
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(V5["H"], color='magenta', ls='--', lw=1.5)
    ax2.axvline(PDG["H"], color='cyan', ls=':', lw=1.5)
    ax2.set_xlabel("m(γγ) [GeV]")
    ax2.set_ylabel("Résidu")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nFigure : {out_path}")
    plt.close(fig)


def plot_z(centers, n_data, n_fit, popt, perr, n_pairs, out_path):
    fig, ax1 = plt.subplots(figsize=(11, 7))
    sigma_data = np.sqrt(n_data + 1)
    ax1.errorbar(centers, n_data, yerr=sigma_data,
                 fmt='o', markersize=3, color='black',
                 label=f"ATLAS m(l⁺l⁻)  (n={int(n_data.sum())})")
    if popt is not None:
        ax1.plot(centers, n_fit, 'b-', linewidth=2,
                 label="Fit : Crystal Ball + quadratique")
        bkg = popt[5] + popt[6]*centers + popt[7]*centers**2
        bkg = np.maximum(bkg, 0)
        sig = popt[4] * crystal_ball(centers, popt[0], popt[1], popt[2], popt[3])
        ax1.plot(centers, bkg, 'g--', alpha=0.6, label="continuum (Drell-Yan)")
        ax1.plot(centers, sig, 'r-', alpha=0.7,
                 label=f"signal CB  m={popt[0]:.3f}±{perr[0]:.3f} GeV  σ={popt[1]:.2f} GeV")
    ax1.axvline(V5["Z"], color='magenta', ls='--', lw=1.8,
                label=f"V5 prédit  m_Z = {V5['Z']:.2f} GeV")
    ax1.axvline(PDG["Z"], color='cyan', ls=':', lw=1.8,
                label=f"PDG  m_Z = {PDG['Z']:.2f} GeV")
    ax1.set_xlabel("m(l⁺l⁻) [GeV]")
    ax1.set_ylabel("Events / bin")
    ax1.set_title(f"ATLAS Z → l⁺l⁻ — {n_pairs} paires OS-SF sélectionnées")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nFigure : {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("V5 ÉLECTROFAIBLE — prédictions Ω_ts vs ATLAS open data 13 TeV")
    print("=" * 78)
    print(f"  W : {V5_CONFIG['W']:>15s}  m_V5 = {V5['W']:.3f} GeV   PDG {PDG['W']:.3f}")
    print(f"  Z : {V5_CONFIG['Z']:>15s}  m_V5 = {V5['Z']:.3f} GeV   PDG {PDG['Z']:.3f}")
    print(f"  H : {V5_CONFIG['H']:>15s}  m_V5 = {V5['H']:.3f} GeV   PDG {PDG['H']:.3f}")
    print()

    # --- Higgs γγ ---
    print("=" * 78)
    print("HIGGS → γγ — fenêtre [105, 150] GeV")
    print("=" * 78)
    gam_files = sorted(DATA_DIR.glob("data_*.GamGam.root"))
    print(f"  {len(gam_files)} fichiers GamGam")
    masses_gg, n_evt_gg, n_pairs_gg = load_diphoton(gam_files)
    print(f"\n  events scannés : {n_evt_gg}")
    print(f"  paires γγ sélectionnées : {n_pairs_gg}")
    in_window = ((masses_gg >= 105) & (masses_gg <= 150)).sum()
    print(f"  paires dans [105, 150] : {in_window}")

    higgs_ok = False
    m_H_fit = None
    if n_pairs_gg > 100 and in_window > 100:
        c, d, fitc, popt_h, perr_h = fit_higgs(masses_gg)
        if popt_h is not None:
            chi2 = ((d - fitc)**2 / (d + 1)).sum()
            ndof = len(d) - 9
            print(f"  Fit Higgs : χ² = {chi2:.1f} / {ndof} ndof  χ²/ndof = {chi2/ndof:.2f}")
            print(f"  m_H_fit   = {popt_h[0]:.3f} ± {perr_h[0]:.3f} GeV")
            print(f"  σ_H_fit   = {popt_h[1]:.3f} ± {perr_h[1]:.3f} GeV")
            print(f"  N_signal  = {popt_h[4]:.1f} ± {perr_h[4]:.1f}")
            m_H_fit = (popt_h[0], perr_h[0])
            higgs_ok = True
            plot_higgs(c, d, fitc, popt_h, perr_h, n_pairs_gg,
                       OUT / "ew_validation_higgs_gg.png")
    else:
        print("  Stat insuffisante — skip fit Higgs")

    # --- Z dilepton ---
    print()
    print("=" * 78)
    print("Z → l⁺l⁻ — fenêtre [60, 120] GeV")
    print("=" * 78)
    lep_files = sorted(DATA_DIR.glob("data_*.exactly2lep.root"))
    print(f"  {len(lep_files)} fichiers exactly2lep")
    masses_ll, n_evt_ll, n_pairs_ll = load_dilepton(lep_files)
    print(f"\n  events scannés : {n_evt_ll}")
    print(f"  paires OS-SF sélectionnées : {n_pairs_ll}")
    in_window_z = ((masses_ll >= 60) & (masses_ll <= 120)).sum()
    print(f"  paires dans [60, 120] : {in_window_z}")

    z_ok = False
    m_Z_fit = None
    if n_pairs_ll > 100 and in_window_z > 100:
        c, d, fitc, popt_z, perr_z = fit_z(masses_ll)
        if popt_z is not None:
            chi2 = ((d - fitc)**2 / (d + 1)).sum()
            ndof = len(d) - 8
            print(f"  Fit Z : χ² = {chi2:.1f} / {ndof} ndof  χ²/ndof = {chi2/ndof:.2f}")
            print(f"  m_Z_fit  = {popt_z[0]:.4f} ± {perr_z[0]:.4f} GeV")
            print(f"  σ_Z_fit  = {popt_z[1]:.3f} ± {perr_z[1]:.3f} GeV")
            print(f"  N_signal = {popt_z[4]:.1f} ± {perr_z[4]:.1f}")
            m_Z_fit = (popt_z[0], perr_z[0])
            z_ok = True
            plot_z(c, d, fitc, popt_z, perr_z, n_pairs_ll,
                   OUT / "ew_validation_z_ll.png")
    else:
        print("  Stat insuffisante — skip fit Z")

    # --- Tableau récapitulatif ---
    print()
    print("=" * 78)
    print("RÉCAPITULATIF — Validation V5 secteur EW")
    print("=" * 78)
    print(f"  {'particule':<10s} {'m_V5 [GeV]':>12s} {'m_fitted [GeV]':>22s} "
          f"{'Δ(fit−V5)/V5':>14s} {'Δ(fit−PDG)/PDG':>16s}")
    print("  " + "-" * 76)
    if m_H_fit is not None:
        m, em = m_H_fit
        d_v5 = (m - V5["H"]) / V5["H"] * 100
        d_pdg = (m - PDG["H"]) / PDG["H"] * 100
        print(f"  {'H (γγ)':<10s} {V5['H']:>12.3f} "
              f"{m:>13.3f} ± {em:.3f} {d_v5:>+13.2f}% {d_pdg:>+15.2f}%")
    else:
        print(f"  {'H (γγ)':<10s} {V5['H']:>12.3f}  fit non disponible")
    if m_Z_fit is not None:
        m, em = m_Z_fit
        d_v5 = (m - V5["Z"]) / V5["Z"] * 100
        d_pdg = (m - PDG["Z"]) / PDG["Z"] * 100
        print(f"  {'Z (ll)':<10s} {V5['Z']:>12.3f} "
              f"{m:>13.4f} ± {em:.4f} {d_v5:>+13.2f}% {d_pdg:>+15.2f}%")
    else:
        print(f"  {'Z (ll)':<10s} {V5['Z']:>12.3f}  fit non disponible")

    print()
    print("Note : les écarts m_fitted vs PDG mesurent la qualité du fit (résolution")
    print("de détecteur + bruit de fond) ; m_fitted vs V5 teste la prédiction Ω_ts.")


if __name__ == "__main__":
    main()
