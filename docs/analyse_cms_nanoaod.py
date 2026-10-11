#!/usr/bin/env python3
"""
Analyse du fichier CMS DoubleMuon NanoAOD 13 TeV — Cross-check avec ATLAS.

Lit le fichier NanoAOD, reconstruit le spectre de masse invariante dimuon,
et produit une figure 4 panneaux montrant les résonances J/ψ, Υ et Z.
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── 1. Ouvrir le fichier et lire l'arbre Events ─────────────────────────
ROOT_FILE = "/home/euk9/projets/physique_nouvelle/data/CMS_13TeV/DoubleMuon_NanoAOD_small.root"
TREE_NAME = "Events"

print(f"Ouverture de {ROOT_FILE} ...")
f = uproot.open(ROOT_FILE)
tree = f[TREE_NAME]

# ── 2. Lister les branches contenant "Muon" ─────────────────────────────
all_branches = tree.keys()
muon_branches = sorted([b for b in all_branches if "Muon" in b or "muon" in b])
print(f"\nBranches contenant 'Muon' ({len(muon_branches)}) :")
for b in muon_branches:
    print(f"  {b}")

# ── 3. Lire les données nécessaires ─────────────────────────────────────
branches_to_read = ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]
print(f"\nLecture des branches : {branches_to_read} ...")
data = tree.arrays(branches_to_read, library="ak")

import awkward as ak

n_total = len(data)
print(f"Nombre total d'événements : {n_total}")

# ── 4. Sélection : au moins 2 muons ─────────────────────────────────────
mask_2mu = data["nMuon"] >= 2
data_2mu = data[mask_2mu]
n_2mu = len(data_2mu)
print(f"Événements avec nMuon >= 2 : {n_2mu}")

# Extraire les 2 premiers muons
pt1  = ak.to_numpy(data_2mu["Muon_pt"][:, 0])
pt2  = ak.to_numpy(data_2mu["Muon_pt"][:, 1])
eta1 = ak.to_numpy(data_2mu["Muon_eta"][:, 0])
eta2 = ak.to_numpy(data_2mu["Muon_eta"][:, 1])
phi1 = ak.to_numpy(data_2mu["Muon_phi"][:, 0])
phi2 = ak.to_numpy(data_2mu["Muon_phi"][:, 1])
m1   = ak.to_numpy(data_2mu["Muon_mass"][:, 0])
m2   = ak.to_numpy(data_2mu["Muon_mass"][:, 1])
q1   = ak.to_numpy(data_2mu["Muon_charge"][:, 0])
q2   = ak.to_numpy(data_2mu["Muon_charge"][:, 1])

# ── 5. Calcul de la masse invariante dimuon ──────────────────────────────
# px = pt * cos(phi), py = pt * sin(phi), pz = pt * sinh(eta)
# E  = sqrt(pt^2 * cosh^2(eta) + mass^2)
px1 = pt1 * np.cos(phi1)
py1 = pt1 * np.sin(phi1)
pz1 = pt1 * np.sinh(eta1)
E1  = np.sqrt(pt1**2 * np.cosh(eta1)**2 + m1**2)

px2 = pt2 * np.cos(phi2)
py2 = pt2 * np.sin(phi2)
pz2 = pt2 * np.sinh(eta2)
E2  = np.sqrt(pt2**2 * np.cosh(eta2)**2 + m2**2)

m2_inv = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
# Protéger contre les valeurs négatives (précision numérique)
m_inv = np.sqrt(np.maximum(m2_inv, 0.0))

print(f"Masse invariante calculée pour {len(m_inv)} paires")
print(f"  min = {m_inv.min():.3f} GeV, max = {m_inv.max():.1f} GeV")

# ── 6. Sélection de signe opposé ────────────────────────────────────────
os_mask = (q1 * q2) < 0
m_os = m_inv[os_mask]
print(f"Paires de signe opposé : {len(m_os)} ({100*len(m_os)/len(m_inv):.1f}%)")

# ── 7. Figure 4 panneaux ────────────────────────────────────────────────

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss_plus_poly(x, A, mu, sigma, a, b):
    return gauss(x, A, mu, sigma) + a * x + b

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CMS DoubleMuon NanoAOD 13 TeV — Cross-check",
             fontsize=15, fontweight="bold", y=0.98)

# ── Panneau 1 : spectre complet ─────────────────────────────────────────
ax1 = axes[0, 0]
bins_full = np.linspace(0.5, 200, 600)
counts_full, edges_full, _ = ax1.hist(m_os, bins=bins_full, histtype="stepfilled",
                                       color="steelblue", alpha=0.7, linewidth=0.5,
                                       edgecolor="navy", label="Paires μ⁺μ⁻")
ax1.set_yscale("log")
ax1.set_xlabel("$m_{\\mu\\mu}$ [GeV]", fontsize=11)
ax1.set_ylabel("Événements / bin", fontsize=11)
ax1.set_title("Spectre de masse dimuon complet", fontsize=12)

# Marquer les résonances
resonances = {"$J/\\psi$": 3.097, "$\\Upsilon$": 9.46, "$Z$": 91.2}
for name, mass in resonances.items():
    ax1.axvline(mass, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(mass, ax1.get_ylim()[1] * 0.3, name, fontsize=9, color="red",
             ha="center", va="top", rotation=90)

ax1.set_xlim(0.5, 200)
ax1.legend(fontsize=9)

# ── Panneau 2 : zoom J/ψ ────────────────────────────────────────────────
ax2 = axes[0, 1]
jpsi_range = (2.8, 3.4)
bins_jpsi = np.linspace(*jpsi_range, 80)
counts_jpsi, edges_jpsi, _ = ax2.hist(m_os, bins=bins_jpsi, histtype="stepfilled",
                                       color="darkorange", alpha=0.7, edgecolor="saddlebrown",
                                       linewidth=0.5)
ax2.set_xlabel("$m_{\\mu\\mu}$ [GeV]", fontsize=11)
ax2.set_ylabel("Événements / bin", fontsize=11)
ax2.set_title("Région $J/\\psi$ (2.8–3.4 GeV)", fontsize=12)

# Ajustement gaussien + fond linéaire
centers_jpsi = 0.5 * (edges_jpsi[:-1] + edges_jpsi[1:])
mask_fit = counts_jpsi > 0
try:
    p0_jpsi = [counts_jpsi.max(), 3.097, 0.05, -1, 10]
    popt_jpsi, pcov_jpsi = curve_fit(gauss_plus_poly, centers_jpsi[mask_fit],
                                      counts_jpsi[mask_fit], p0=p0_jpsi,
                                      sigma=np.sqrt(np.maximum(counts_jpsi[mask_fit], 1)),
                                      maxfev=10000)
    x_fit = np.linspace(*jpsi_range, 300)
    ax2.plot(x_fit, gauss_plus_poly(x_fit, *popt_jpsi), "r-", linewidth=2,
             label=f"Ajustement : $\\mu$ = {popt_jpsi[1]:.3f} GeV\n"
                   f"$\\sigma$ = {abs(popt_jpsi[2])*1000:.0f} MeV")
    ax2.legend(fontsize=9)
    print(f"\nAjustement J/ψ : μ = {popt_jpsi[1]:.4f} GeV, σ = {abs(popt_jpsi[2])*1000:.1f} MeV")
except Exception as e:
    print(f"\nAjustement J/ψ échoué : {e}")
    ax2.text(0.5, 0.85, "Ajustement non convergé", transform=ax2.transAxes,
             fontsize=9, ha="center", color="red")

# ── Panneau 3 : zoom Υ ──────────────────────────────────────────────────
ax3 = axes[1, 0]
ups_range = (8.5, 11.0)
bins_ups = np.linspace(*ups_range, 80)
counts_ups, edges_ups, _ = ax3.hist(m_os, bins=bins_ups, histtype="stepfilled",
                                     color="mediumpurple", alpha=0.7, edgecolor="indigo",
                                     linewidth=0.5)
ax3.set_xlabel("$m_{\\mu\\mu}$ [GeV]", fontsize=11)
ax3.set_ylabel("Événements / bin", fontsize=11)
ax3.set_title("Région $\\Upsilon$ (8.5–11.0 GeV)", fontsize=12)

# Marquer les 3 états Υ
upsilon_states = {"$\\Upsilon(1S)$": 9.460, "$\\Upsilon(2S)$": 10.023, "$\\Upsilon(3S)$": 10.355}
for name, mass in upsilon_states.items():
    ax3.axvline(mass, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
    ymax = ax3.get_ylim()[1]
    ax3.text(mass + 0.02, ymax * 0.85, name, fontsize=8, color="red",
             ha="left", va="top", rotation=90)

# ── Panneau 4 : région haute masse ──────────────────────────────────────
ax4 = axes[1, 1]
high_range = (60, 200)
bins_high = np.linspace(*high_range, 120)
counts_high, edges_high, _ = ax4.hist(m_os, bins=bins_high, histtype="stepfilled",
                                       color="seagreen", alpha=0.7, edgecolor="darkgreen",
                                       linewidth=0.5)
ax4.set_xlabel("$m_{\\mu\\mu}$ [GeV]", fontsize=11)
ax4.set_ylabel("Événements / bin", fontsize=11)
ax4.set_title("Région haute masse (60–200 GeV)", fontsize=12)

# Marquer Z, b'=109, Higgs=125, τ'=159
high_marks = {"$Z$": 91.2, "$b'$ = 109": 109, "$H$ = 125": 125, "$\\tau'$ = 159": 159}
colors_hm = {"$Z$": "red", "$b'$ = 109": "blue", "$H$ = 125": "magenta", "$\\tau'$ = 159": "darkorange"}
for name, mass in high_marks.items():
    c = colors_hm[name]
    ax4.axvline(mass, color=c, linestyle="--", linewidth=1.0, alpha=0.8)
    ax4.text(mass + 1, ax4.get_ylim()[1] * 0.85, name, fontsize=8.5, color=c,
             ha="left", va="top")

# ── Finalisation ─────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0, 1, 0.96])
outpath = "/home/euk9/projets/physique_nouvelle/docs/analyse_cms_nanoaod_crosscheck.png"
fig.savefig(outpath, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nFigure sauvegardée : {outpath}")
plt.close(fig)

# ── Résumé statistique ──────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("  Résumé de l'analyse CMS DoubleMuon")
print("══════════════════════════════════════════")
print(f"  Événements totaux        : {n_total}")
print(f"  Événements >= 2 muons    : {n_2mu}")
print(f"  Paires signe opposé      : {len(m_os)}")

# Comptage dans les régions de résonance
n_jpsi = np.sum((m_os > 2.9) & (m_os < 3.3))
n_ups  = np.sum((m_os > 9.0) & (m_os < 10.8))
n_z    = np.sum((m_os > 80) & (m_os < 100))
print(f"  Candidats J/ψ (2.9-3.3)  : {n_jpsi}")
print(f"  Candidats Υ   (9.0-10.8) : {n_ups}")
print(f"  Candidats Z   (80-100)   : {n_z}")
print("══════════════════════════════════════════")
