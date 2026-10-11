#!/usr/bin/env python3
"""
Analyse du canal ATLAS 1lep+1tau — Recherche du τ' (Gen 4) prédit par la TQNT à 159 GeV.

Données : ATLAS Open Data 13 TeV, canal 1lep1tau (~820k événements).
L. A. Cariou — 2026
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Constantes physiques (masses cibles en GeV) ──────────────────────────
M_TPRIME = 159.2   # τ' prédit par la TQNT (génération 4)
M_BPRIME = 109.3   # b' prédit par la TQNT (génération 4)
M_HIGGS  = 125.1   # Boson de Higgs

# ── Fichiers de données ──────────────────────────────────────────────────
DATA_DIR = Path("/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV")
FILES = sorted(DATA_DIR.glob("data_*.1lep1tau*.root"))
print(f"Fichiers trouvés : {len(FILES)}")
for f in FILES:
    print(f"  {f.name}")

# ── Branches à lire ─────────────────────────────────────────────────────
BRANCHES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_type", "lep_n",
    "tau_pt", "tau_eta", "tau_phi", "tau_E", "tau_n",
    "met_et", "met_phi",
]

# ── Chargement des données ───────────────────────────────────────────────
print("\nChargement des données...")
all_data = {b: [] for b in BRANCHES}

total_events = 0
for fpath in FILES:
    f = uproot.open(fpath)
    tree = f["mini"]
    n = tree.num_entries
    total_events += n
    print(f"  {fpath.name}: {n:,} événements")

    # Lire en blocs pour limiter la mémoire
    arrays = tree.arrays(BRANCHES, library="ak")
    for b in BRANCHES:
        all_data[b].append(arrays[b])

import awkward as ak

for b in BRANCHES:
    all_data[b] = ak.concatenate(all_data[b])

print(f"\nTotal : {total_events:,} événements chargés")

# ── Sélection : exactement 1 lepton + 1 tau ─────────────────────────────
lep_n = all_data["lep_n"]
tau_n = all_data["tau_n"]

mask_1lep1tau = (lep_n >= 1) & (tau_n >= 1)
print(f"Événements avec ≥1 lepton et ≥1 tau : {ak.sum(mask_1lep1tau):,}")

# Extraire le premier lepton et le premier tau
lep_pt  = ak.to_numpy(all_data["lep_pt"][mask_1lep1tau][:, 0]) / 1000.0   # MeV → GeV
lep_eta = ak.to_numpy(all_data["lep_eta"][mask_1lep1tau][:, 0])
lep_phi = ak.to_numpy(all_data["lep_phi"][mask_1lep1tau][:, 0])
lep_E   = ak.to_numpy(all_data["lep_E"][mask_1lep1tau][:, 0]) / 1000.0

tau_pt  = ak.to_numpy(all_data["tau_pt"][mask_1lep1tau][:, 0]) / 1000.0
tau_eta = ak.to_numpy(all_data["tau_eta"][mask_1lep1tau][:, 0])
tau_phi = ak.to_numpy(all_data["tau_phi"][mask_1lep1tau][:, 0])
tau_E   = ak.to_numpy(all_data["tau_E"][mask_1lep1tau][:, 0]) / 1000.0

met_et  = ak.to_numpy(all_data["met_et"][mask_1lep1tau]) / 1000.0
met_phi = ak.to_numpy(all_data["met_phi"][mask_1lep1tau])

n_selected = len(lep_pt)
print(f"Événements sélectionnés (1ère paire lep+tau) : {n_selected:,}")

# ── Calcul des 4-vecteurs ───────────────────────────────────────────────
# Lepton
lep_px = lep_pt * np.cos(lep_phi)
lep_py = lep_pt * np.sin(lep_phi)
lep_pz = lep_pt * np.sinh(lep_eta)

# Tau
tau_px = tau_pt * np.cos(tau_phi)
tau_py = tau_pt * np.sin(tau_phi)
tau_pz = tau_pt * np.sinh(tau_eta)

# ── 1. Masse visible m_vis ──────────────────────────────────────────────
E_tot   = lep_E + tau_E
px_tot  = lep_px + tau_px
py_tot  = lep_py + tau_py
pz_tot  = lep_pz + tau_pz
m2_vis  = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
m_vis   = np.sqrt(np.maximum(m2_vis, 0))

print(f"\nm_vis : moyenne = {np.mean(m_vis):.1f} GeV, médiane = {np.median(m_vis):.1f} GeV")

# ── 2. Masse transverse m_T (lepton + MET) ──────────────────────────────
dphi_lep_met = lep_phi - met_phi
# Normaliser dans [-π, π]
dphi_lep_met = np.arctan2(np.sin(dphi_lep_met), np.cos(dphi_lep_met))
m_T = np.sqrt(2 * lep_pt * met_et * (1 - np.cos(dphi_lep_met)))

print(f"m_T   : moyenne = {np.mean(m_T):.1f} GeV, médiane = {np.median(m_T):.1f} GeV")

# ── 3. Masse collinéaire m_col ──────────────────────────────────────────
# Approximation collinéaire : le neutrino va dans la direction du tau
# x_tau = pT_tau / (pT_tau + MET_along_tau)
# m_col = m_vis / sqrt(x_tau)

# Projection du MET sur la direction du tau
met_px = met_et * np.cos(met_phi)
met_py = met_et * np.sin(met_phi)

# Direction du tau dans le plan transverse
tau_pt_safe = np.maximum(tau_pt, 1e-6)
tau_ux = tau_px / tau_pt_safe
tau_uy = tau_py / tau_pt_safe

# MET projeté le long du tau
met_along_tau = met_px * tau_ux + met_py * tau_uy

# Fraction d'impulsion visible du tau
x_tau = tau_pt / (tau_pt + np.maximum(met_along_tau, 0.1))
# Couper les valeurs non-physiques
x_tau = np.clip(x_tau, 0.01, 1.0)

m_col = m_vis / np.sqrt(x_tau)

# Couper les valeurs extrêmes
m_col_cut = m_col[m_col < 500]
print(f"m_col : moyenne = {np.mean(m_col_cut):.1f} GeV, médiane = {np.median(m_col_cut):.1f} GeV")

# ── 4. Méthode de significativité (sidebands) ───────────────────────────
def sideband_significance(mass_array, target_mass, window=10, sideband_width=30):
    """
    Calcule la significativité par méthode des bandes latérales.

    Fenêtre signal : [target - window, target + window]
    Bandes latérales : [target - sideband - window, target - window]
                       + [target + window, target + window + sideband]

    Retourne : (n_obs, n_bg_est, significance, excess)
    """
    # Fenêtre signal
    sig_lo = target_mass - window
    sig_hi = target_mass + window
    n_obs = np.sum((mass_array >= sig_lo) & (mass_array < sig_hi))

    # Bandes latérales
    sb_lo1 = target_mass - sideband_width - window
    sb_hi1 = target_mass - window
    sb_lo2 = target_mass + window
    sb_hi2 = target_mass + window + sideband_width

    n_sb_lo = np.sum((mass_array >= sb_lo1) & (mass_array < sb_hi1))
    n_sb_hi = np.sum((mass_array >= sb_lo2) & (mass_array < sb_hi2))

    # Largeur des bandes
    w_signal = sig_hi - sig_lo        # 2 × window
    w_sideband = 2 * sideband_width   # total des deux bandes

    # Estimation du bruit de fond dans la fenêtre signal
    n_sb_total = n_sb_lo + n_sb_hi
    n_bg_est = n_sb_total * (w_signal / w_sideband) if w_sideband > 0 else 0

    # Excès
    excess = n_obs - n_bg_est

    # Significativité (approximation gaussienne)
    if n_bg_est > 0:
        significance = excess / np.sqrt(n_bg_est)
    else:
        significance = 0.0

    return n_obs, n_bg_est, significance, excess


# Calcul pour les 3 masses cibles, sur m_vis
print("\n" + "=" * 70)
print("SIGNIFICATIVITÉ (m_vis, fenêtre ±10 GeV, bandes latérales ±30 GeV)")
print("=" * 70)

targets = [
    ("b' (Gen 4)", M_BPRIME),
    ("Higgs",      M_HIGGS),
    ("τ' (Gen 4)", M_TPRIME),
]

results_vis = {}
for name, mass in targets:
    n_obs, n_bg, sig, exc = sideband_significance(m_vis, mass)
    results_vis[name] = (n_obs, n_bg, sig, exc, mass)
    print(f"  {name:15s} ({mass:.1f} GeV) : N_obs={n_obs:6d}, N_bg={n_bg:.0f}, "
          f"excès={exc:+.0f}, signif={sig:+.2f}σ")

# Idem sur m_col
print("\nSIGNIFICATIVITÉ (m_col, fenêtre ±10 GeV, bandes latérales ±30 GeV)")
print("-" * 70)

results_col = {}
for name, mass in targets:
    n_obs, n_bg, sig, exc = sideband_significance(m_col, mass)
    results_col[name] = (n_obs, n_bg, sig, exc, mass)
    print(f"  {name:15s} ({mass:.1f} GeV) : N_obs={n_obs:6d}, N_bg={n_bg:.0f}, "
          f"excès={exc:+.0f}, signif={sig:+.2f}σ")

# ── 5. Pull distribution autour de τ' ───────────────────────────────────
def pull_distribution(mass_array, center, half_range=50, bin_width=2):
    """
    Calcule la distribution de pull : (données - fond lissé) / sqrt(fond)
    dans la région [center - half_range, center + half_range].
    """
    bins = np.arange(center - half_range, center + half_range + bin_width, bin_width)
    counts, edges = np.histogram(mass_array, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Fond lissé : polynôme de degré 3 (excluant la fenêtre signal)
    mask_sb = (centers < center - 10) | (centers > center + 10)
    if np.sum(mask_sb) < 5:
        return centers, np.zeros_like(centers, dtype=float), counts, counts.astype(float)

    coeffs = np.polyfit(centers[mask_sb], counts[mask_sb], 3)
    bg_smooth = np.polyval(coeffs, centers)
    bg_smooth = np.maximum(bg_smooth, 1)  # éviter division par 0

    pull = (counts - bg_smooth) / np.sqrt(bg_smooth)
    return centers, pull, counts, bg_smooth


# ── 6. FIGURE 6 panneaux ────────────────────────────────────────────────
print("\nGénération de la figure...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("ATLAS 1lep+1tau — Recherche τ' (Gen 4) à 159 GeV\n"
             f"ATLAS Open Data 13 TeV — {n_selected:,} événements sélectionnés",
             fontsize=14, fontweight="bold", y=0.98)

gs = GridSpec(2, 3, hspace=0.35, wspace=0.3, top=0.91, bottom=0.08, left=0.06, right=0.97)

# Couleurs
COL_DATA = "black"
COL_TPRIME = "#d62728"   # rouge
COL_HIGGS  = "#2ca02c"   # vert
COL_BPRIME = "#1f77b4"   # bleu
COL_FILL   = "#bdd7ee"

# ── Panneau 1 : m_vis (0-300 GeV) ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bins_vis = np.linspace(0, 300, 151)
ax1.hist(m_vis, bins=bins_vis, histtype="stepfilled", color=COL_FILL,
         edgecolor="steelblue", linewidth=0.5, label="Données")
ax1.axvline(M_TPRIME, color=COL_TPRIME, ls="--", lw=2, label=f"τ' = {M_TPRIME} GeV")
ax1.axvline(M_HIGGS,  color=COL_HIGGS,  ls="--", lw=1.5, label=f"Higgs = {M_HIGGS} GeV")
ax1.axvline(M_BPRIME, color=COL_BPRIME, ls="--", lw=1.5, label=f"b' = {M_BPRIME} GeV")
ax1.set_xlabel("$m_{vis}$ (lepton + tau) [GeV]", fontsize=11)
ax1.set_ylabel("Événements / 2 GeV", fontsize=11)
ax1.set_title("Masse visible", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(0, 300)

# ── Panneau 2 : m_T (0-300 GeV) ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bins_mt = np.linspace(0, 300, 151)
ax2.hist(m_T, bins=bins_mt, histtype="stepfilled", color="#f5d5a0",
         edgecolor="#c49030", linewidth=0.5, label="Données")
ax2.axvline(M_TPRIME, color=COL_TPRIME, ls="--", lw=2, label=f"τ' = {M_TPRIME} GeV")
ax2.axvline(M_HIGGS,  color=COL_HIGGS,  ls="--", lw=1.5, label=f"Higgs = {M_HIGGS} GeV")
ax2.set_xlabel("$m_T$ (lepton + MET) [GeV]", fontsize=11)
ax2.set_ylabel("Événements / 2 GeV", fontsize=11)
ax2.set_title("Masse transverse", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlim(0, 300)

# ── Panneau 3 : Zoom m_vis 120-200 GeV ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bins_zoom = np.linspace(120, 200, 41)
counts_zoom, edges_zoom, _ = ax3.hist(m_vis, bins=bins_zoom, histtype="stepfilled",
                                       color=COL_FILL, edgecolor="steelblue", linewidth=0.8)
# Fond lissé (polynôme degré 2 sur les bandes latérales)
centers_zoom = 0.5 * (edges_zoom[:-1] + edges_zoom[1:])
mask_sb_zoom = ((centers_zoom >= 120) & (centers_zoom < 145)) | \
               ((centers_zoom > 170) & (centers_zoom <= 200))
if np.sum(mask_sb_zoom) > 3:
    coeffs_zoom = np.polyfit(centers_zoom[mask_sb_zoom], counts_zoom[mask_sb_zoom], 2)
    bg_zoom = np.polyval(coeffs_zoom, centers_zoom)
    ax3.plot(centers_zoom, bg_zoom, "k-", lw=1.5, alpha=0.6, label="Fond estimé")

ax3.axvline(M_TPRIME, color=COL_TPRIME, ls="--", lw=2, label=f"τ' = {M_TPRIME} GeV")
ax3.axvline(M_HIGGS,  color=COL_HIGGS,  ls="--", lw=2, label=f"Higgs = {M_HIGGS} GeV")

# Zone signal τ' (±10 GeV)
ax3.axvspan(M_TPRIME - 10, M_TPRIME + 10, alpha=0.12, color=COL_TPRIME, label="Fenêtre τ' ±10")

ax3.set_xlabel("$m_{vis}$ [GeV]", fontsize=11)
ax3.set_ylabel("Événements / 2 GeV", fontsize=11)
ax3.set_title("Zoom : 120–200 GeV", fontsize=12, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right")
ax3.set_xlim(120, 200)

# ── Panneau 4 : Distribution MET ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
bins_met = np.linspace(0, 200, 101)
ax4.hist(met_et, bins=bins_met, histtype="stepfilled", color="#d5e8d4",
         edgecolor="#82b366", linewidth=0.5)
ax4.set_xlabel("$E_T^{miss}$ [GeV]", fontsize=11)
ax4.set_ylabel("Événements / 2 GeV", fontsize=11)
ax4.set_title("Énergie transverse manquante", fontsize=12, fontweight="bold")
ax4.set_xlim(0, 200)
ax4.set_yscale("log")

# ── Panneau 5 : Pull distribution (m_vis autour de τ') ──────────────────
ax5 = fig.add_subplot(gs[1, 1])
centers_pull, pull_vals, data_pull, bg_pull = pull_distribution(m_vis, M_TPRIME, 50, 2)

colors_pull = np.where(pull_vals > 0, COL_TPRIME, "#4477aa")
ax5.bar(centers_pull, pull_vals, width=1.8, color=colors_pull, alpha=0.7, edgecolor="gray", linewidth=0.3)
ax5.axhline(0, color="black", lw=0.8)
ax5.axhline(2, color="gray", ls=":", lw=0.8, alpha=0.5)
ax5.axhline(-2, color="gray", ls=":", lw=0.8, alpha=0.5)
ax5.axvline(M_TPRIME, color=COL_TPRIME, ls="--", lw=2, alpha=0.7)
ax5.axvspan(M_TPRIME - 10, M_TPRIME + 10, alpha=0.08, color=COL_TPRIME)
ax5.set_xlabel("$m_{vis}$ [GeV]", fontsize=11)
ax5.set_ylabel("Pull (données − fond) / √fond", fontsize=11)
ax5.set_title(f"Pull autour de τ' ({M_TPRIME} GeV)", fontsize=12, fontweight="bold")
ax5.set_xlim(M_TPRIME - 50, M_TPRIME + 50)
ax5.set_ylim(-4, 4)

# ── Panneau 6 : Tableau récapitulatif ───────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

# Construire le tableau
col_labels = ["Particule", "Masse\n[GeV]", "N_obs\n(±10 GeV)", "N_fond\nestiméé", "Excès", "Signif.\n[σ]"]
row_data = []
for name, mass in targets:
    r = results_vis[name]
    n_obs, n_bg, sig, exc, m = r
    row_data.append([name, f"{m:.1f}", f"{n_obs:,}", f"{n_bg:.0f}", f"{exc:+.0f}", f"{sig:+.2f}"])

# Ajouter résultats m_col
row_data.append(["", "", "", "", "", ""])
row_data.append(["(m_col)", "", "", "", "", ""])
for name, mass in targets:
    r = results_col[name]
    n_obs, n_bg, sig, exc, m = r
    row_data.append([name, f"{m:.1f}", f"{n_obs:,}", f"{n_bg:.0f}", f"{exc:+.0f}", f"{sig:+.2f}"])

table = ax6.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.22, 0.12, 0.15, 0.15, 0.15, 0.12],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Colorer les en-têtes
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#4472c4")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorer les lignes τ'
for i, (name, _) in enumerate(targets):
    row_idx = i + 1
    if "τ'" in name:
        for j in range(len(col_labels)):
            table[row_idx, j].set_facecolor("#fce4e4")
    elif "Higgs" in name:
        for j in range(len(col_labels)):
            table[row_idx, j].set_facecolor("#e8f5e9")
    elif "b'" in name:
        for j in range(len(col_labels)):
            table[row_idx, j].set_facecolor("#e3f2fd")

# Colorer les lignes m_col τ'
for i, (name, _) in enumerate(targets):
    row_idx = i + 6  # après l'espace et le label
    if row_idx < len(row_data) + 1:
        if "τ'" in name:
            for j in range(len(col_labels)):
                table[row_idx, j].set_facecolor("#fce4e4")

ax6.set_title("Résumé : significativités\n(m_vis puis m_col)", fontsize=12,
              fontweight="bold", pad=20)

# ── Sauvegarde ───────────────────────────────────────────────────────────
outpath = Path("/home/euk9/projets/physique_nouvelle/docs/analyse_1lep1tau_tprime.png")
fig.savefig(outpath, dpi=180, bbox_inches="tight")
print(f"\nFigure sauvegardée : {outpath}")
print("Terminé.")
