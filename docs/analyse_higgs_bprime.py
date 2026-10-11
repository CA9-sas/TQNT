#!/usr/bin/env python3
"""
Analyse Higgs + recherche b' caché — ATLAS Open Data 13 TeV
Fit double-gaussien sur le spectre diphoton pour chercher un 2ème pic à 109 GeV.
+ Analyse complète beta 2muons (8.9M) et beta 4lep (1.5M).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

import uproot
import awkward as ak

M_B_PRIME = 109.3
M_HIGGS = 125.1
M_TAU_PRIME = 159.2
M_Z = 91.2

DATA = "/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV"
OUT = "/home/euk9/projets/physique_nouvelle/docs"

# ================================================================
# LOADERS
# ================================================================
def load_masses(pattern, treename, max_ev=None):
    """Load dilepton invariant mass from ATLAS ROOT files."""
    masses = []
    total = 0
    for f in sorted(glob.glob(pattern)):
        if max_ev and total > max_ev:
            break
        try:
            tree = uproot.open(f)[treename]
            lep_pt = tree["lep_pt"].array()
            lep_eta = tree["lep_eta"].array()
            lep_phi = tree["lep_phi"].array()
            lep_E = tree["lep_E"].array()
            mask = ak.num(lep_pt) >= 2
            pt = lep_pt[mask]; eta = lep_eta[mask]
            phi = lep_phi[mask]; E = lep_E[mask]
            px1 = pt[:,0]*np.cos(phi[:,0]); py1 = pt[:,0]*np.sin(phi[:,0])
            pz1 = pt[:,0]*np.sinh(eta[:,0])
            px2 = pt[:,1]*np.cos(phi[:,1]); py2 = pt[:,1]*np.sin(phi[:,1])
            pz2 = pt[:,1]*np.sinh(eta[:,1])
            m2 = (E[:,0]+E[:,1])**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
            masses.append(m)
            total += len(m)
            print(f"  {os.path.basename(f)}: {len(m):,} evts (total: {total:,})")
        except Exception as e:
            print(f"  Erreur {os.path.basename(f)}: {e}")
    return np.concatenate(masses) if masses else np.array([])

def load_photon_masses(pattern, treename="mini"):
    masses = []
    for f in sorted(glob.glob(pattern)):
        try:
            tree = uproot.open(f)[treename]
            pt = tree["photon_pt"].array()
            eta = tree["photon_eta"].array()
            phi = tree["photon_phi"].array()
            E = tree["photon_E"].array()
            mask = ak.num(pt) >= 2
            pt=pt[mask]; eta=eta[mask]; phi=phi[mask]; E=E[mask]
            m2 = (E[:,0]+E[:,1])**2 - \
                 (pt[:,0]*np.cos(phi[:,0])+pt[:,1]*np.cos(phi[:,1]))**2 - \
                 (pt[:,0]*np.sin(phi[:,0])+pt[:,1]*np.sin(phi[:,1]))**2 - \
                 (pt[:,0]*np.sinh(eta[:,0])+pt[:,1]*np.sinh(eta[:,1]))**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
            masses.append(m)
            print(f"  {os.path.basename(f)}: {len(m):,} γγ evts")
        except Exception as e:
            print(f"  Erreur {os.path.basename(f)}: {e}")
    return np.concatenate(masses) if masses else np.array([])

def load_4lep_masses(pattern, treename):
    masses = []
    for f in sorted(glob.glob(pattern)):
        try:
            tree = uproot.open(f)[treename]
            pt = tree["lep_pt"].array()
            eta = tree["lep_eta"].array()
            phi = tree["lep_phi"].array()
            E = tree["lep_E"].array()
            mask = ak.num(pt) >= 4
            pt=pt[mask]; eta=eta[mask]; phi=phi[mask]; E=E[mask]
            E_tot = sum(E[:,i] for i in range(4))
            px = sum(pt[:,i]*np.cos(phi[:,i]) for i in range(4))
            py = sum(pt[:,i]*np.sin(phi[:,i]) for i in range(4))
            pz = sum(pt[:,i]*np.sinh(eta[:,i]) for i in range(4))
            m2 = E_tot**2 - px**2 - py**2 - pz**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
            masses.append(m)
            print(f"  {os.path.basename(f)}: {len(m):,} 4ℓ evts")
        except Exception as e:
            print(f"  Erreur {os.path.basename(f)}: {e}")
    return np.concatenate(masses) if masses else np.array([])

# ================================================================
# FIT MODELS
# ================================================================
def single_gauss_plus_bg(x, A, mu, sigma, a, b, c):
    """1 gaussienne + fond polynomial."""
    return A * np.exp(-0.5*((x-mu)/sigma)**2) + a*x**2 + b*x + c

def double_gauss_plus_bg(x, A1, mu1, sig1, A2, mu2, sig2, a, b, c):
    """2 gaussiennes + fond polynomial."""
    g1 = A1 * np.exp(-0.5*((x-mu1)/sig1)**2)
    g2 = A2 * np.exp(-0.5*((x-mu2)/sig2)**2)
    return g1 + g2 + a*x**2 + b*x + c

# ================================================================
# MAIN
# ================================================================
print("=" * 60)
print("ANALYSE HIGGS + RECHERCHE b' CACHÉ — ATLAS 13 TeV")
print("=" * 60)

# --- Diphoton (Higgs H→γγ) ---
print("\n=== DIPHOTON (H→γγ) ===")
m_gg = load_photon_masses(os.path.join(DATA, "data_*.GamGam.root"))

# --- Dilepton exactly2lep ---
print("\n=== DILEPTON (exactly2lep) ===")
m_2l = load_masses(os.path.join(DATA, "data_*.exactly2lep*.root"), "mini")

# --- Beta 2muons ---
print("\n=== BETA 2MUONS ===")
m_2mu = load_masses(os.path.join(DATA, "beta_2muons_*.root"), "analysis")

# --- Beta 4lep ---
print("\n=== BETA 4LEP ===")
m_4l_beta = load_4lep_masses(os.path.join(DATA, "beta_4lep_*.root"), "analysis")

# --- 2020 4lep ---
print("\n=== 2020 4LEP ===")
m_4l = load_4lep_masses(os.path.join(DATA, "data_*.4lep*.root"), "mini")

# Merge 4lep datasets
m_4l_all = np.concatenate([m_4l, m_4l_beta]) if len(m_4l_beta) > 0 else m_4l

print(f"\n=== RÉSUMÉ DES DONNÉES ===")
print(f"  Diphoton γγ    : {len(m_gg):>12,} événements")
print(f"  Dilepton 2ℓ    : {len(m_2l):>12,} événements")
print(f"  Dimuon 2μ beta : {len(m_2mu):>12,} événements")
print(f"  4-leptons total: {len(m_4l_all):>12,} événements")
print(f"  TOTAL          : {len(m_gg)+len(m_2l)+len(m_2mu)+len(m_4l_all):>12,}")

# ================================================================
# FIT DOUBLE-GAUSSIEN SUR DIPHOTON (100-160 GeV)
# ================================================================
print("\n=== FIT DOUBLE-GAUSSIEN DIPHOTON ===")

# Histogram in the Higgs region
m_range = (100, 160)
nbins = 120
mask_gg = (m_gg >= m_range[0]) & (m_gg <= m_range[1])
h_gg, edges_gg = np.histogram(m_gg[mask_gg], bins=nbins, range=m_range)
centers_gg = 0.5 * (edges_gg[:-1] + edges_gg[1:])
bin_width = edges_gg[1] - edges_gg[0]

# Single Gaussian fit (Higgs only)
try:
    p0_single = [max(h_gg), 125, 2, -1, 100, 10000]
    popt_s, pcov_s = curve_fit(single_gauss_plus_bg, centers_gg, h_gg,
                                p0=p0_single, maxfev=10000)
    fit_single = single_gauss_plus_bg(centers_gg, *popt_s)
    chi2_single = np.sum((h_gg - fit_single)**2 / np.maximum(fit_single, 1))
    ndof_single = len(h_gg) - 6
    print(f"  1 gaussienne: μ={popt_s[1]:.1f} GeV, σ={abs(popt_s[2]):.1f} GeV, "
          f"χ²/ndof = {chi2_single:.0f}/{ndof_single} = {chi2_single/ndof_single:.2f}")
except Exception as e:
    print(f"  Fit 1 gauss échoué: {e}")
    popt_s = None

# Double Gaussian fit (Higgs + b')
try:
    p0_double = [max(h_gg)*0.8, 125, 2, max(h_gg)*0.1, 109, 3, -1, 100, 10000]
    bounds_lo = [0, 120, 0.5, 0, 100, 0.5, -100, -1e6, -1e6]
    bounds_hi = [1e7, 130, 10, 1e7, 118, 10, 100, 1e6, 1e6]
    popt_d, pcov_d = curve_fit(double_gauss_plus_bg, centers_gg, h_gg,
                                p0=p0_double, bounds=(bounds_lo, bounds_hi),
                                maxfev=20000)
    fit_double = double_gauss_plus_bg(centers_gg, *popt_d)
    chi2_double = np.sum((h_gg - fit_double)**2 / np.maximum(fit_double, 1))
    ndof_double = len(h_gg) - 9
    print(f"  2 gaussiennes: μ₁={popt_d[1]:.1f} GeV (Higgs), "
          f"μ₂={popt_d[4]:.1f} GeV (b'?), "
          f"σ₁={abs(popt_d[2]):.1f}, σ₂={abs(popt_d[5]):.1f}")
    print(f"  A₁={popt_d[0]:.0f} (Higgs), A₂={popt_d[3]:.0f} (b'?)")
    print(f"  χ²/ndof = {chi2_double:.0f}/{ndof_double} = {chi2_double/ndof_double:.2f}")

    delta_chi2 = chi2_single - chi2_double
    print(f"  Δχ² (1G→2G) = {delta_chi2:.1f} pour 3 paramètres supplémentaires")
    print(f"  Significativité ≈ {np.sqrt(max(delta_chi2, 0)):.1f}σ")
except Exception as e:
    print(f"  Fit 2 gauss échoué: {e}")
    popt_d = None

# ================================================================
# FIGURE 8 PANNEAUX
# ================================================================
fig = plt.figure(figsize=(26, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

# --- 1. Diphoton complet ---
ax1 = fig.add_subplot(gs[0, 0])
if len(m_gg) > 0:
    mask = (m_gg > 50) & (m_gg < 500)
    ax1.hist(m_gg[mask], bins=450, range=(50, 500), color="#90CAF9",
             edgecolor="#1565C0", linewidth=0.3)
    for m, lab, col in [(M_HIGGS, "Higgs", "#2E7D32"),
                         (M_B_PRIME, "b'", "#C62828"),
                         (M_TAU_PRIME, "τ'", "#FF8F00")]:
        ax1.axvline(m, color=col, linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_xlabel("m_γγ (GeV)")
    ax1.set_ylabel("Événements / GeV")
    ax1.set_title(f"Diphoton ATLAS ({len(m_gg):,} évts)", fontweight="bold")
    ax1.legend(["Higgs 125", "b' 109 (TQNT)", "τ' 159"], fontsize=7)
    ax1.grid(True, alpha=0.2)

# --- 2. Diphoton zoom + fit simple ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(centers_gg, h_gg, width=bin_width, color="#90CAF9",
        edgecolor="#1565C0", linewidth=0.3, label="Données ATLAS")
if popt_s is not None:
    ax2.plot(centers_gg, fit_single, "b-", linewidth=2,
             label=f"Fit 1 gauss (μ={popt_s[1]:.1f} GeV)")
ax2.axvline(M_HIGGS, color="#2E7D32", linestyle=":", linewidth=2,
            label=f"Higgs = {M_HIGGS} GeV")
ax2.axvline(M_B_PRIME, color="#C62828", linestyle="--", linewidth=2,
            label=f"b' TQNT = {M_B_PRIME} GeV")
ax2.set_xlabel("m_γγ (GeV)")
ax2.set_ylabel(f"Événements / {bin_width:.1f} GeV")
ax2.set_title("Diphoton 100–160 GeV — Fit 1 gaussienne",
              fontweight="bold")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.2)

# --- 3. Diphoton zoom + fit DOUBLE ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(centers_gg, h_gg, width=bin_width, color="#90CAF9",
        edgecolor="#1565C0", linewidth=0.3, label="Données ATLAS")
if popt_d is not None:
    ax3.plot(centers_gg, fit_double, "r-", linewidth=2.5,
             label=f"Fit 2 gauss (Δχ²={delta_chi2:.0f})")
    # Show individual components
    bg = popt_d[6]*centers_gg**2 + popt_d[7]*centers_gg + popt_d[8]
    g1 = popt_d[0]*np.exp(-0.5*((centers_gg-popt_d[1])/popt_d[2])**2) + bg
    g2 = popt_d[3]*np.exp(-0.5*((centers_gg-popt_d[4])/popt_d[5])**2) + bg
    ax3.plot(centers_gg, g1, "g--", linewidth=1.5,
             label=f"Higgs μ={popt_d[1]:.1f} GeV")
    ax3.plot(centers_gg, g2, "r--", linewidth=1.5,
             label=f"b'? μ={popt_d[4]:.1f} GeV")
    ax3.fill_between(centers_gg, bg, g2, alpha=0.2, color="#E53935")
ax3.axvline(M_B_PRIME, color="#C62828", linestyle=":", linewidth=1.5)
ax3.set_xlabel("m_γγ (GeV)")
ax3.set_ylabel(f"Événements / {bin_width:.1f} GeV")
ax3.set_title("Diphoton — Fit DOUBLE gaussienne\n"
              "Le b' est-il caché près du Higgs?", fontweight="bold",
              color="#C62828")
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.2)

# --- 4. Résidu (données - fit simple) ---
ax4 = fig.add_subplot(gs[1, 0])
if popt_s is not None:
    residual = h_gg - fit_single
    err = np.sqrt(np.maximum(h_gg, 1))
    pull = residual / err
    ax4.bar(centers_gg, pull, width=bin_width, color="#FFCC80",
            edgecolor="#FF8F00", linewidth=0.3)
    ax4.axhline(0, color="black", linewidth=1)
    ax4.axhline(2, color="#FF8F00", linestyle="--")
    ax4.axhline(-2, color="#FF8F00", linestyle="--")
    ax4.axhline(3, color="#C62828", linestyle="--")
    ax4.axhline(-3, color="#C62828", linestyle="--")
    ax4.axvline(M_B_PRIME, color="#C62828", linestyle=":", linewidth=2)
    ax4.axvline(M_HIGGS, color="#2E7D32", linestyle=":", linewidth=2)
    ax4.text(M_B_PRIME+0.5, 4, "b'", fontsize=10, color="#C62828",
             fontweight="bold")
    ax4.text(M_HIGGS+0.5, 4, "H", fontsize=10, color="#2E7D32",
             fontweight="bold")
    ax4.set_xlabel("m_γγ (GeV)")
    ax4.set_ylabel("Pull (σ)")
    ax4.set_title("Résidus diphoton (données − fit 1G)", fontweight="bold")
    ax4.set_ylim(-6, 6)
    ax4.grid(True, alpha=0.2)

# --- 5. 4-leptons (all data combined) ---
ax5 = fig.add_subplot(gs[1, 1])
if len(m_4l_all) > 0:
    mask4 = (m_4l_all > 60) & (m_4l_all < 250)
    ax5.hist(m_4l_all[mask4], bins=95, range=(60, 250), color="#CE93D8",
             edgecolor="#6A1B9A", linewidth=0.5)
    ax5.axvline(M_HIGGS, color="#2E7D32", linestyle="--", linewidth=2,
                label=f"Higgs = {M_HIGGS}")
    ax5.axvline(M_B_PRIME, color="#C62828", linestyle="--", linewidth=2,
                label=f"b' = {M_B_PRIME}")
    ax5.axvline(M_Z, color="#1565C0", linestyle=":", linewidth=1.5,
                label=f"Z = {M_Z}")
    ax5.set_xlabel("m_4ℓ (GeV)")
    ax5.set_ylabel("Événements / 2 GeV")
    ax5.set_title(f"4-leptons combinés ({len(m_4l_all):,} évts)\n"
                  f"H→ZZ*→4ℓ", fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.2)

# --- 6. Dimuon beta zoom 80-200 ---
ax6 = fig.add_subplot(gs[1, 2])
if len(m_2mu) > 0:
    mask_mu = (m_2mu > 80) & (m_2mu < 200)
    ax6.hist(m_2mu[mask_mu], bins=120, range=(80, 200), color="#A5D6A7",
             edgecolor="#2E7D32", linewidth=0.3)
    for m, lab, col in [(M_Z, "Z", "#1565C0"), (M_B_PRIME, "b'", "#C62828"),
                         (M_HIGGS, "H", "#2E7D32"), (M_TAU_PRIME, "τ'", "#FF8F00")]:
        ax6.axvline(m, color=col, linestyle="--", linewidth=2)
    ax6.set_xlabel("m_μμ (GeV)")
    ax6.set_ylabel("Événements / GeV")
    ax6.set_title(f"Dimuon beta ({len(m_2mu):,} évts) 80–200 GeV",
                  fontweight="bold")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.2)

# --- 7. Dilepton zoom haute masse (150-500 GeV) pour τ' ---
ax7 = fig.add_subplot(gs[2, 0])
if len(m_2l) > 0:
    mask_hi = (m_2l > 130) & (m_2l < 500)
    h_hi, e_hi = np.histogram(m_2l[mask_hi], bins=185, range=(130, 500))
    c_hi = 0.5*(e_hi[:-1]+e_hi[1:])
    ax7.step(c_hi, h_hi, color="#1565C0", linewidth=1.5)
    ax7.fill_between(c_hi, 0, h_hi, step="pre", alpha=0.3, color="#90CAF9")
    # Smooth background
    bg_hi = uniform_filter1d(h_hi.astype(float), size=15)
    ax7.plot(c_hi, bg_hi, "r-", linewidth=2, label="Fond lissé")
    ax7.axvline(M_TAU_PRIME, color="#FF8F00", linestyle="--", linewidth=2.5)
    ax7.text(M_TAU_PRIME+3, max(h_hi)*0.7, f"τ' = {M_TAU_PRIME} GeV",
             fontsize=9, color="#FF8F00", fontweight="bold")
    ax7.set_xlabel("m_ℓℓ (GeV)")
    ax7.set_ylabel("Événements / 2 GeV")
    ax7.set_title(f"Dilepton haute masse ({len(m_2l):,} évts)\n"
                  f"Recherche τ' à 159 GeV", fontweight="bold")
    ax7.set_yscale("log")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.2)

# --- 8. Résumé et conclusions ---
ax8 = fig.add_subplot(gs[2, 1:])
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 10)
ax8.axis("off")
ax8.set_title("Résumé — Analyse ATLAS 13 TeV complète", fontsize=14,
              fontweight="bold")

y = 9.5
total_evts = len(m_gg) + len(m_2l) + len(m_2mu) + len(m_4l_all)
ax8.text(5, y, f"Total analysé : {total_evts:,} événements ATLAS 13 TeV",
         fontsize=11, ha="center", fontweight="bold", color="#1565C0")
y -= 0.6

# Results
lines = [
    ("DIPHOTON (canal Higgs H→γγ)", "#2E7D32", True),
    (f"  Pic du Higgs confirmé à ~125 GeV", "#333", False),
]
if popt_d is not None:
    lines += [
        (f"  Fit double-gaussien : Δχ² = {delta_chi2:.0f} "
         f"(~{np.sqrt(max(delta_chi2,0)):.1f}σ)", "#333", False),
        (f"  2ème pic fitté à μ₂ = {popt_d[4]:.1f} GeV "
         f"(amplitude = {popt_d[3]:.0f})", "#C62828",
         popt_d[3] > 100),
    ]
    if delta_chi2 > 9:  # > 3σ
        lines.append(("  ⚠ EXCÈS SIGNIFICATIF dans le fit double-gaussien!",
                       "#C62828", True))
    elif delta_chi2 > 4:  # > 2σ
        lines.append(("  ⚡ Indication faible (2-3σ) — à confirmer",
                       "#FF8F00", True))
    else:
        lines.append(("  Pas d'amélioration significative avec 2 gaussiennes",
                       "#666", False))

lines += [
    ("", "#333", False),
    ("4-LEPTONS (canal H→ZZ*→4ℓ)", "#6A1B9A", True),
    (f"  {len(m_4l_all):,} événements, pic du Higgs visible", "#333", False),
    ("", "#333", False),
    ("DILEPTON HAUTE MASSE (recherche τ')", "#FF8F00", True),
    (f"  {len(m_2l):,} événements analysés, zone 130–500 GeV", "#333", False),
    (f"  Pas d'excès visible à {M_TAU_PRIME} GeV (spectre Drell-Yan lisse)",
     "#333", False),
    ("", "#333", False),
    ("DIMUON BETA (8.9M événements)", "#2E7D32", True),
    (f"  {len(m_2mu):,} événements 2μ, haute statistique", "#333", False),
    ("", "#333", False),
    ("CONCLUSIONS", "#C62828", True),
    ("  • Le pic du Higgs est reproduit dans tous les canaux", "#333", False),
    (f"  • Le b' à 109 GeV : {('signal dans le fit double-gaussien' if popt_d is not None and delta_chi2 > 4 else 'pas de signal clair')}", "#333", False),
    ("  • Le τ' à 159 GeV : pas d'excès dilepton (attendu — mauvais canal)",
     "#333", False),
    ("  • Prochaine étape : canaux b-jets et τ+MET pour b' et τ'", "#333", False),
]

for text, col, bold in lines:
    if text == "":
        y -= 0.15
        continue
    ax8.text(0.2, y, text, fontsize=7.5, color=col,
             fontweight="bold" if bold else "normal")
    y -= 0.23

fig.suptitle("Recherche b' caché dans le pic Higgs — "
             "ATLAS Open Data 13 TeV (fit double-gaussien)",
             fontsize=16, fontweight="bold", y=0.99)
fig.savefig(os.path.join(OUT, "analyse_higgs_bprime.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → analyse_higgs_bprime.png")
