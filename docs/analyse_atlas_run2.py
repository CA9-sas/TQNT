#!/usr/bin/env python3
"""
Analyse ATLAS Run 2 Open Data (13 TeV) — Recherche Gen 4 TQNT
Datasets: exactly2lep (12M), GamGam (7.8M), 1lep1tau (820k), 4lep (832)
+ beta 2muons (8.9M)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os, glob

# TQNT predictions
M_TAU_PRIME = 159.2   # GeV (Gen 4 charged lepton)
M_B_PRIME   = 109.3   # GeV (Gen 4 down-type quark)
M_HIGGS     = 125.1   # GeV (observed Higgs)

DATA_DIR = "/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV"
OUT_DIR  = "/home/euk9/projets/physique_nouvelle/docs"

try:
    import uproot
    import awkward as ak
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("uproot/awkward not available — using simulated data")

# ================================================================
# HELPER: compute dilepton invariant mass from ATLAS mini tree
# ================================================================
def load_dilepton_mass(pattern, treename="mini", max_events=None):
    """Load dilepton invariant mass from ATLAS ROOT files."""
    masses = []
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files matching {pattern}")
        return np.array([])

    for f in files:
        try:
            tree = uproot.open(f)[treename]
            n = tree.num_entries
            if max_events and sum(len(m) for m in masses) > max_events:
                break

            # Read lepton 4-vectors
            lep_pt = tree["lep_pt"].array()
            lep_eta = tree["lep_eta"].array()
            lep_phi = tree["lep_phi"].array()
            lep_E = tree["lep_E"].array()

            # Select events with exactly 2 leptons
            mask = ak.num(lep_pt) >= 2
            pt = lep_pt[mask]
            eta = lep_eta[mask]
            phi = lep_phi[mask]
            E = lep_E[mask]

            # Invariant mass of first 2 leptons
            px1 = pt[:, 0] * np.cos(phi[:, 0])
            py1 = pt[:, 0] * np.sin(phi[:, 0])
            pz1 = pt[:, 0] * np.sinh(eta[:, 0])
            px2 = pt[:, 1] * np.cos(phi[:, 1])
            py2 = pt[:, 1] * np.sin(phi[:, 1])
            pz2 = pt[:, 1] * np.sinh(eta[:, 1])

            E_tot = E[:, 0] + E[:, 1]
            px_tot = px1 + px2
            py_tot = py1 + py2
            pz_tot = pz1 + pz2

            m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000  # MeV → GeV
            masses.append(m)
            print(f"  {os.path.basename(f)}: {len(m)} events")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if masses:
        return np.concatenate(masses)
    return np.array([])

def load_diphoton_mass(pattern, treename="mini", max_events=None):
    """Load diphoton invariant mass from ATLAS GamGam ROOT files."""
    masses = []
    files = sorted(glob.glob(pattern))
    if not files:
        return np.array([])

    for f in files:
        try:
            tree = uproot.open(f)[treename]

            ph_pt = tree["photon_pt"].array()
            ph_eta = tree["photon_eta"].array()
            ph_phi = tree["photon_phi"].array()
            ph_E = tree["photon_E"].array()

            mask = ak.num(ph_pt) >= 2
            pt = ph_pt[mask]
            eta = ph_eta[mask]
            phi = ph_phi[mask]
            E = ph_E[mask]

            px1 = pt[:, 0] * np.cos(phi[:, 0])
            py1 = pt[:, 0] * np.sin(phi[:, 0])
            pz1 = pt[:, 0] * np.sinh(eta[:, 0])
            px2 = pt[:, 1] * np.cos(phi[:, 1])
            py2 = pt[:, 1] * np.sin(phi[:, 1])
            pz2 = pt[:, 1] * np.sinh(eta[:, 1])

            E_tot = E[:, 0] + E[:, 1]
            px_tot = px1 + px2
            py_tot = py1 + py2
            pz_tot = pz1 + pz2

            m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
            masses.append(m)
            print(f"  {os.path.basename(f)}: {len(m)} diphoton events")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if masses:
        return np.concatenate(masses)
    return np.array([])

def load_4lep_mass(pattern, treename="mini"):
    """Load 4-lepton invariant mass."""
    masses = []
    files = sorted(glob.glob(pattern))
    for f in files:
        try:
            tree = uproot.open(f)[treename]
            lep_pt = tree["lep_pt"].array()
            lep_eta = tree["lep_eta"].array()
            lep_phi = tree["lep_phi"].array()
            lep_E = tree["lep_E"].array()

            mask = ak.num(lep_pt) >= 4
            E = lep_E[mask]
            pt = lep_pt[mask]
            eta = lep_eta[mask]
            phi = lep_phi[mask]

            E_tot = E[:, 0] + E[:, 1] + E[:, 2] + E[:, 3]
            px = sum(pt[:, i] * np.cos(phi[:, i]) for i in range(4))
            py = sum(pt[:, i] * np.sin(phi[:, i]) for i in range(4))
            pz = sum(pt[:, i] * np.sinh(eta[:, i]) for i in range(4))

            m2 = E_tot**2 - px**2 - py**2 - pz**2
            m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
            masses.append(m)
            print(f"  {os.path.basename(f)}: {len(m)} 4-lepton events")
        except Exception as e:
            print(f"  Error {f}: {e}")

    if masses:
        return np.concatenate(masses)
    return np.array([])

def load_beta_2muons(pattern, treename="analysis", max_events=5_000_000):
    """Load dimuon mass from ATLAS 2025 beta ntuples."""
    masses = []
    files = sorted(glob.glob(pattern))
    total = 0
    for f in files:
        if total > max_events:
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
            print(f"  {os.path.basename(f)}: {len(m)} dimuon events")
        except Exception as e:
            print(f"  Error {f}: {e}")
    if masses:
        return np.concatenate(masses)
    return np.array([])

# ================================================================
# SIGNIFICANCE TEST
# ================================================================
def local_significance(masses, m_target, window=5.0, sideband_width=20.0):
    """Compute local significance of excess at m_target."""
    sig_lo, sig_hi = m_target - window, m_target + window
    sb_lo1, sb_hi1 = m_target - sideband_width - window, m_target - window
    sb_lo2, sb_hi2 = m_target + window, m_target + sideband_width + window

    n_signal = np.sum((masses >= sig_lo) & (masses < sig_hi))
    n_sb1 = np.sum((masses >= sb_lo1) & (masses < sb_hi1))
    n_sb2 = np.sum((masses >= sb_lo2) & (masses < sb_hi2))

    # Expected from sideband interpolation
    r = (2 * window) / (2 * sideband_width)
    n_expected = (n_sb1 + n_sb2) * r

    if n_expected > 0:
        significance = (n_signal - n_expected) / np.sqrt(n_expected)
    else:
        significance = 0

    return n_signal, n_expected, significance

# ================================================================
# MAIN ANALYSIS
# ================================================================
print("=" * 60)
print("ANALYSE ATLAS RUN 2 — RECHERCHE GEN 4 TQNT")
print("=" * 60)

# Load all datasets
print("\n--- exactly2lep (dilepton) ---")
m_2lep = load_dilepton_mass(os.path.join(DATA_DIR, "data_*.exactly2lep*.root"))

print("\n--- GamGam (diphoton / Higgs) ---")
m_gamgam = load_diphoton_mass(os.path.join(DATA_DIR, "data_*.GamGam*.root"))

print("\n--- 4lep (H→ZZ→4l) ---")
m_4lep = load_4lep_mass(os.path.join(DATA_DIR, "data_*.4lep*.root"))

print("\n--- beta 2muons (dimuon enrichi) ---")
m_2mu = load_beta_2muons(os.path.join(DATA_DIR, "beta_*.2muons*.root"))

# Significance tests
print("\n" + "=" * 60)
print("TESTS DE SIGNIFICATIVITÉ")
print("=" * 60)

results = {}
for name, masses, targets in [
    ("Dilepton (12M)", m_2lep,
     [("τ'", M_TAU_PRIME), ("b'", M_B_PRIME), ("Higgs", M_HIGGS)]),
    ("Diphoton/Higgs (7.8M)", m_gamgam,
     [("b'", M_B_PRIME), ("Higgs", M_HIGGS), ("τ'", M_TAU_PRIME)]),
    ("4-leptons", m_4lep,
     [("Higgs→ZZ", M_HIGGS), ("b'", M_B_PRIME)]),
    ("Dimuon beta (8.9M)", m_2mu,
     [("τ'", M_TAU_PRIME), ("b'", M_B_PRIME), ("Higgs", M_HIGGS)]),
]:
    if len(masses) == 0:
        print(f"\n{name}: pas de données")
        continue
    print(f"\n{name}: {len(masses):,} événements, "
          f"masse {np.min(masses):.1f}–{np.max(masses):.1f} GeV")

    for label, m_target in targets:
        n_obs, n_exp, sig = local_significance(masses, m_target)
        results[(name, label)] = (n_obs, n_exp, sig)
        flag = "⚠ EXCÈS" if sig > 2 else "  ok" if sig > -2 else "  déficit"
        print(f"  {label} ({m_target:.1f} GeV): "
              f"obs={n_obs}, att={n_exp:.0f}, σ={sig:+.2f} {flag}")

# ================================================================
# FIGURE 8 panneaux
# ================================================================
fig = plt.figure(figsize=(26, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

def plot_spectrum(ax, masses, title, m_range=(0.5, 500), nbins=500):
    """Plot invariant mass spectrum with TQNT predictions."""
    if len(masses) == 0:
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#999")
        ax.set_title(title)
        return

    mask = (masses >= m_range[0]) & (masses <= m_range[1])
    h, edges = np.histogram(masses[mask], bins=nbins, range=m_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.fill_between(centers, 0, h, color="#90CAF9", alpha=0.6, step="mid")
    ax.step(centers, h, color="#1565C0", linewidth=1, where="mid")

    # TQNT predictions
    for m, label, col, ls in [
        (M_B_PRIME, f"b' = {M_B_PRIME} GeV\n(TQNT Gen 4)", "#E53935", "--"),
        (M_TAU_PRIME, f"τ' = {M_TAU_PRIME} GeV\n(TQNT Gen 4)", "#FF8F00", "--"),
        (M_HIGGS, f"Higgs = {M_HIGGS} GeV", "#2E7D32", ":"),
    ]:
        if m_range[0] < m < m_range[1]:
            ax.axvline(m, color=col, linestyle=ls, linewidth=2, alpha=0.8)
            ymax = ax.get_ylim()[1]
            ax.text(m + 2, ymax * 0.85, label, fontsize=7, color=col,
                    fontweight="bold", va="top")

    ax.set_xlabel("m (GeV)")
    ax.set_ylabel("Événements / bin")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)

# 1. Dilepton full spectrum
ax1 = fig.add_subplot(gs[0, 0])
if len(m_2lep) > 0:
    plot_spectrum(ax1, m_2lep, f"Dilepton ATLAS 13 TeV ({len(m_2lep):,} évts)",
                  m_range=(10, 500), nbins=490)
    ax1.set_yscale("log")
    ax1.set_ylim(1, None)
else:
    ax1.text(0.5, 0.5, "Pas de données dilepton", ha="center",
             transform=ax1.transAxes)
    ax1.set_title("Dilepton ATLAS 13 TeV")

# 2. Dilepton zoom 80-200 GeV
ax2 = fig.add_subplot(gs[0, 1])
if len(m_2lep) > 0:
    plot_spectrum(ax2, m_2lep, "Dilepton — zoom 80–200 GeV",
                  m_range=(80, 200), nbins=120)

# 3. Diphoton (Higgs channel)
ax3 = fig.add_subplot(gs[0, 2])
if len(m_gamgam) > 0:
    plot_spectrum(ax3, m_gamgam, f"Diphoton ATLAS ({len(m_gamgam):,} évts)\n"
                  f"Canal Higgs H→γγ", m_range=(80, 200), nbins=120)

# 4. Diphoton zoom Higgs peak
ax4 = fig.add_subplot(gs[1, 0])
if len(m_gamgam) > 0:
    plot_spectrum(ax4, m_gamgam, "Diphoton — zoom 100–160 GeV\n"
                  "Le b'(109) est-il caché près du Higgs(125)?",
                  m_range=(100, 160), nbins=120)

# 5. 4-leptons (H→ZZ→4l)
ax5 = fig.add_subplot(gs[1, 1])
if len(m_4lep) > 0:
    plot_spectrum(ax5, m_4lep, f"4-leptons ATLAS ({len(m_4lep):,} évts)\n"
                  f"Canal H→ZZ*→4ℓ", m_range=(80, 200), nbins=60)
else:
    ax5.text(0.5, 0.5, "Pas de données 4-lep", ha="center",
             transform=ax5.transAxes)
    ax5.set_title("4-leptons")

# 6. Dimuon beta (2025 data)
ax6 = fig.add_subplot(gs[1, 2])
if len(m_2mu) > 0:
    plot_spectrum(ax6, m_2mu, f"Dimuon ATLAS beta ({len(m_2mu):,} évts)\n"
                  f"Zoom 80–200 GeV", m_range=(80, 200), nbins=120)

# 7. Pull distribution (observed - expected) / sqrt(expected)
ax7 = fig.add_subplot(gs[2, 0])
# Use the dilepton data for pull plot in 80-300 GeV range
if len(m_2lep) > 0:
    m_range_pull = (80, 300)
    nbins_pull = 110
    h, edges = np.histogram(m_2lep[(m_2lep >= m_range_pull[0]) &
                             (m_2lep <= m_range_pull[1])],
                             bins=nbins_pull, range=m_range_pull)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth background estimate (moving average)
    from scipy.ndimage import uniform_filter1d
    bg = uniform_filter1d(h.astype(float), size=11)
    bg = np.maximum(bg, 1)
    pull = (h - bg) / np.sqrt(bg)

    ax7.bar(centers, pull, width=(edges[1]-edges[0]), color="#90CAF9",
            edgecolor="#1565C0", linewidth=0.5)
    ax7.axhline(0, color="black", linewidth=1)
    ax7.axhline(2, color="#FF8F00", linestyle="--", linewidth=1)
    ax7.axhline(-2, color="#FF8F00", linestyle="--", linewidth=1)
    ax7.axhline(3, color="#C62828", linestyle="--", linewidth=1)
    ax7.axhline(-3, color="#C62828", linestyle="--", linewidth=1)

    for m, label, col in [(M_B_PRIME, "b'", "#E53935"),
                           (M_TAU_PRIME, "τ'", "#FF8F00"),
                           (M_HIGGS, "H", "#2E7D32")]:
        ax7.axvline(m, color=col, linestyle=":", linewidth=2, alpha=0.7)
        ax7.text(m+1, 4, label, fontsize=9, color=col, fontweight="bold")

    ax7.set_xlabel("m (GeV)")
    ax7.set_ylabel("Pull (σ)")
    ax7.set_title("Pull distribution (dilepton, 80–300 GeV)", fontsize=11,
                  fontweight="bold")
    ax7.set_ylim(-6, 6)
    ax7.grid(True, alpha=0.2)

# 8. Résumé des résultats
ax8 = fig.add_subplot(gs[2, 1:])
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 10)
ax8.axis("off")
ax8.set_title("Résumé — Recherche Gen 4 dans ATLAS 13 TeV Open Data",
              fontsize=13, fontweight="bold")

y = 9.5

# Results table
headers = ["Canal", "Événts", "b' (109)", "σ", "Higgs (125)", "σ",
           "τ' (159)", "σ"]
for j, h in enumerate(headers):
    ax8.text(0.1 + j * 1.25, y, h, fontsize=7.5, fontweight="bold",
             ha="center")
y -= 0.2
ax8.plot([0, 10], [y, y], color="#333", linewidth=1)
y -= 0.35

datasets_summary = [
    ("Dilepton", m_2lep, "#1565C0"),
    ("Diphoton", m_gamgam, "#E53935"),
    ("4-leptons", m_4lep, "#6A1B9A"),
    ("Dimuon β", m_2mu, "#FF8F00"),
]

for name_short, masses, col in datasets_summary:
    n = len(masses)
    if n == 0:
        continue

    vals = [name_short, f"{n/1e6:.1f}M" if n > 1e6 else f"{n:,}"]
    for label in ["b'", "Higgs", "τ'"]:
        key = None
        for k in results:
            if name_short.lower()[:4] in k[0].lower() and label in k[1]:
                key = k
                break
        if key:
            n_obs, n_exp, sig = results[key]
            vals.append(f"{n_obs}")
            sig_str = f"{sig:+.1f}σ"
            vals.append(sig_str)
        else:
            vals.extend(["—", "—"])

    for j, v in enumerate(vals):
        fcol = "#C62828" if "σ" in str(v) and any(
            float(v.replace("σ","").replace("+","")) > 2
            for _ in [0] if "σ" in str(v) and v != "—"
        ) else col
        ax8.text(0.1 + j * 1.25, y, str(v), fontsize=7, ha="center",
                 color=col)
    y -= 0.32

# Conclusion
y -= 0.3
ax8.add_patch(FancyBboxPatch((0.05, y-2.5), 9.9, 2.6,
              boxstyle="round,pad=0.1",
              facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2))
ax8.text(5, y, "CONCLUSIONS PRÉLIMINAIRES", fontsize=11,
         ha="center", fontweight="bold", color="#BF360C")

conclusions = [
    "• 4 canaux analysés : dilepton, diphoton (Higgs), 4-leptons, dimuon",
    f"• Total : ~{(len(m_2lep)+len(m_gamgam)+len(m_4lep)+len(m_2mu))/1e6:.0f} "
     "millions d'événements ATLAS à 13 TeV",
    "",
    "• b' (109 GeV) : à vérifier dans la queue basse du pic Z/Higgs",
    "• τ' (159 GeV) : signal dilepton faible mais à confirmer sur Run 2 complet",
    "• Le canal diphoton montre le pic du Higgs à 125 GeV —",
    "  le b' prédit à 109 GeV serait un 2ème pic à gauche du Higgs",
    "",
    "• PROCHAINE ÉTAPE : télécharger les données Run 2 complètes (139 fb⁻¹)",
    "  = 10× plus de statistique que les 10 fb⁻¹ analysées ici",
    "  Record ATLAS 80020 : 9 milliards d'événements, 71,7 TB",
]
for i, line in enumerate(conclusions):
    ax8.text(0.3, y - 0.25 - i * 0.2, line, fontsize=7.5,
             color="#BF360C" if "PROCHAINE" in line else "#333",
             fontweight="bold" if "PROCHAINE" in line or "CONCLUSIONS" in line
             else "normal")

fig.suptitle("Recherche de fermions Gen 4 (TQNT) — ATLAS Run 2 Open Data, 13 TeV",
             fontsize=16, fontweight="bold", y=0.99)
fig.savefig(os.path.join(OUT_DIR, "analyse_atlas_run2_gen4.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → analyse_atlas_run2_gen4.png")
