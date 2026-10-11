#!/usr/bin/env python3
"""
Analyse LHCb B→hhh — recherche Ωccc et baryons lourds
5.1M événements, masse invariante 3-corps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.optimize import curve_fit

import uproot

DATA = "/home/euk9/projets/physique_nouvelle/data/LHCb/B2HHH_MagnetDown.root"
OUT = "/home/euk9/projets/physique_nouvelle/docs"

# Masses particules (MeV)
M_PI = 139.57; M_K = 493.68; M_P = 938.27

# Cibles TQNT
M_BPLUS = 5279.3   # B+ (pic principal attendu)
M_OMCCC = 5359     # Ωccc (TQNT)
M_XICC = 3621      # Ξcc++ (découvert 2017)
M_OMCC = 4182      # Ωcc+ (TQNT)
M_XIBC = 7001      # Ξbc (TQNT)

print("=== ANALYSE LHCb B→hhh (5.1M événements) ===")
tree = uproot.open(DATA)["DecayTree"]
n = tree.num_entries
print(f"Événements: {n:,}")

# Lire toutes les impulsions
h1_px = np.array(tree["H1_PX"].array())
h1_py = np.array(tree["H1_PY"].array())
h1_pz = np.array(tree["H1_PZ"].array())
h2_px = np.array(tree["H2_PX"].array())
h2_py = np.array(tree["H2_PY"].array())
h2_pz = np.array(tree["H2_PZ"].array())
h3_px = np.array(tree["H3_PX"].array())
h3_py = np.array(tree["H3_PY"].array())
h3_pz = np.array(tree["H3_PZ"].array())

# PID
h1_probK = np.array(tree["H1_ProbK"].array())
h2_probK = np.array(tree["H2_ProbK"].array())
h3_probK = np.array(tree["H3_ProbK"].array())
h1_probPi = np.array(tree["H1_ProbPi"].array())
h1_charge = np.array(tree["H1_Charge"].array())
h2_charge = np.array(tree["H2_Charge"].array())
h3_charge = np.array(tree["H3_Charge"].array())
b_vtx_chi2 = np.array(tree["B_VertexChi2"].array())
b_fd = np.array(tree["B_FlightDistance"].array())

def invariant_mass_3body(px1,py1,pz1,m1, px2,py2,pz2,m2, px3,py3,pz3,m3):
    E1 = np.sqrt(px1**2+py1**2+pz1**2+m1**2)
    E2 = np.sqrt(px2**2+py2**2+pz2**2+m2**2)
    E3 = np.sqrt(px3**2+py3**2+pz3**2+m3**2)
    Etot = E1+E2+E3
    Pxtot = px1+px2+px3; Pytot = py1+py2+py3; Pztot = pz1+pz2+pz3
    m2_inv = Etot**2 - Pxtot**2 - Pytot**2 - Pztot**2
    return np.sqrt(np.maximum(m2_inv, 0))

# Hypothèses de masse pour les 3 hadrons
print("\nCalcul des masses invariantes sous différentes hypothèses...")

# 1. K+π+π- (canal B+ standard)
m_Kpp = invariant_mass_3body(h1_px,h1_py,h1_pz,M_K,
                              h2_px,h2_py,h2_pz,M_PI,
                              h3_px,h3_py,h3_pz,M_PI)
print(f"  K+π+π-: range {m_Kpp.min():.0f}-{m_Kpp.max():.0f} MeV, "
      f"mean {m_Kpp.mean():.0f}")

# 2. K+K+K- (triple-strange ou triple-charm si mauvais PID)
m_KKK = invariant_mass_3body(h1_px,h1_py,h1_pz,M_K,
                              h2_px,h2_py,h2_pz,M_K,
                              h3_px,h3_py,h3_pz,M_K)
print(f"  K+K+K-: range {m_KKK.min():.0f}-{m_KKK.max():.0f} MeV, "
      f"mean {m_KKK.mean():.0f}")

# 3. p+K+π- (baryon Λc, Ξc territory)
m_pKp = invariant_mass_3body(h1_px,h1_py,h1_pz,M_P,
                              h2_px,h2_py,h2_pz,M_K,
                              h3_px,h3_py,h3_pz,M_PI)
print(f"  p+K+π-: range {m_pKp.min():.0f}-{m_pKp.max():.0f} MeV, "
      f"mean {m_pKp.mean():.0f}")

# 4. p+p+π- (triple-baryon, highest mass hypothesis)
m_ppP = invariant_mass_3body(h1_px,h1_py,h1_pz,M_P,
                              h2_px,h2_py,h2_pz,M_P,
                              h3_px,h3_py,h3_pz,M_PI)
print(f"  p+p+π-: range {m_ppP.min():.0f}-{m_ppP.max():.0f} MeV, "
      f"mean {m_ppP.mean():.0f}")

# Quality cuts
good = (b_vtx_chi2 < 15) & (b_fd > 0)
print(f"\nCoupure qualité (χ²<15, FD>0): {np.sum(good):,} / {n:,} "
      f"({np.sum(good)/n*100:.0f}%)")

# Significance test
def local_sig(masses, m_target, window=30, sb_width=100):
    n_sig = np.sum((masses > m_target-window) & (masses < m_target+window))
    n_sb1 = np.sum((masses > m_target-sb_width-window) & (masses < m_target-window))
    n_sb2 = np.sum((masses > m_target+window) & (masses < m_target+sb_width+window))
    n_exp = (n_sb1+n_sb2) * (2*window) / (2*sb_width)
    sig = (n_sig-n_exp)/np.sqrt(max(n_exp,1))
    return n_sig, n_exp, sig

print("\n=== RECHERCHE DE SIGNAUX ===")
for hyp_name, masses in [("K+π+π-", m_Kpp), ("K+K+K-", m_KKK),
                           ("p+K+π-", m_pKp)]:
    print(f"\nHypothèse {hyp_name}:")
    m_cut = masses[good]
    for label, m_target in [("B+", M_BPLUS), ("Ωccc", M_OMCCC),
                             ("Ξcc++", M_XICC), ("Ωcc+", M_OMCC)]:
        n_obs, n_exp, sig = local_sig(m_cut, m_target)
        flag = "⚠ EXCÈS" if sig > 3 else "? indice" if sig > 2 else ""
        print(f"  {label} ({m_target:.0f} MeV): obs={n_obs:,}, "
              f"att={n_exp:.0f}, σ={sig:+.1f} {flag}")

# ================================================================
# FIGURE
# ================================================================
fig = plt.figure(figsize=(26, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

# 1. Spectre K+π+π- complet
ax1 = fig.add_subplot(gs[0, 0])
m_cut = m_Kpp[good]
ax1.hist(m_cut, bins=500, range=(4800, 5800), color="#90CAF9",
         edgecolor="#1565C0", linewidth=0.3)
ax1.axvline(M_BPLUS, color="#2E7D32", linestyle="--", linewidth=2,
            label=f"B+ = {M_BPLUS:.0f}")
ax1.axvline(M_OMCCC, color="#C62828", linestyle="--", linewidth=2,
            label=f"Ωccc TQNT = {M_OMCCC:.0f}")
ax1.set_xlabel("m(K+π+π-) (MeV)")
ax1.set_ylabel("Événements / 2 MeV")
ax1.set_title(f"LHCb B→K+π+π- ({np.sum(good):,} évts)\n"
              f"Zone B+ et Ωccc", fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

# 2. Zoom B+ peak
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(m_cut, bins=200, range=(5150, 5450), color="#A5D6A7",
         edgecolor="#2E7D32", linewidth=0.3)
ax2.axvline(M_BPLUS, color="#2E7D32", linestyle="--", linewidth=2)
ax2.axvline(M_OMCCC, color="#C62828", linestyle="--", linewidth=2)
ax2.set_xlabel("m(K+π+π-) (MeV)")
ax2.set_title("Zoom pic B+ (5279 MeV)\net zone Ωccc (5359 MeV)", fontweight="bold")
ax2.grid(True, alpha=0.2)

# 3. Zoom Ωccc zone
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(m_cut, bins=100, range=(5300, 5500), color="#FFCC80",
         edgecolor="#FF8F00", linewidth=0.5)
ax3.axvline(M_OMCCC, color="#C62828", linestyle="--", linewidth=2.5,
            label=f"Ωccc TQNT = {M_OMCCC}")
n_zone = np.sum((m_cut > 5300) & (m_cut < 5420))
ax3.set_xlabel("m(K+π+π-) (MeV)")
ax3.set_title(f"Zone Ωccc (5300-5500 MeV)\n{n_zone:,} événements",
              fontweight="bold", color="#C62828")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

# 4. K+K+K- (triple kaon)
ax4 = fig.add_subplot(gs[1, 0])
m_KKK_cut = m_KKK[good]
ax4.hist(m_KKK_cut, bins=400, range=(4800, 6800), color="#CE93D8",
         edgecolor="#6A1B9A", linewidth=0.3)
ax4.axvline(M_BPLUS + (M_K-M_PI)*2, color="#6A1B9A", linestyle=":",
            linewidth=1.5)
ax4.axvline(M_OMCCC + (M_K-M_PI)*2, color="#C62828", linestyle="--",
            linewidth=2, label="Ωccc (KKK hyp.)")
ax4.set_xlabel("m(K+K+K-) (MeV)")
ax4.set_title("Hypothèse K+K+K-\n(triple-strange/charm)", fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.2)

# 5. p+K+π- (baryon hypothesis)
ax5 = fig.add_subplot(gs[1, 1])
m_pKp_cut = m_pKp[good]
ax5.hist(m_pKp_cut, bins=400, range=(4800, 7500), color="#FFAB91",
         edgecolor="#E53935", linewidth=0.3)
ax5.axvline(M_OMCCC, color="#C62828", linestyle="--", linewidth=2)
ax5.axvline(M_XIBC, color="#6A1B9A", linestyle="--", linewidth=2,
            label=f"Ξbc TQNT = {M_XIBC}")
ax5.set_xlabel("m(p+K+π-) (MeV)")
ax5.set_title("Hypothèse p+K+π-\n(baryons lourds)", fontweight="bold")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.2)

# 6. Spectre basse masse (pour Ξcc à 3621)
ax6 = fig.add_subplot(gs[1, 2])
m_low = m_Kpp[good & (m_Kpp > 3000) & (m_Kpp < 4500)]
ax6.hist(m_low, bins=300, range=(3000, 4500), color="#B3E5FC",
         edgecolor="#0288D1", linewidth=0.3)
ax6.axvline(M_XICC, color="#E53935", linestyle="--", linewidth=2,
            label=f"Ξcc++ = {M_XICC}")
ax6.axvline(M_OMCC, color="#FF8F00", linestyle="--", linewidth=2,
            label=f"Ωcc+ TQNT = {M_OMCC}")
ax6.set_xlabel("m(K+π+π-) (MeV)")
ax6.set_title("Basse masse (3-4.5 GeV)\nΞcc++ et Ωcc+", fontweight="bold")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.2)

# 7. Pull distribution autour de Ωccc
ax7 = fig.add_subplot(gs[2, 0])
h_zoom, edges_zoom = np.histogram(m_cut, bins=200, range=(5100, 5600))
c_zoom = 0.5*(edges_zoom[:-1]+edges_zoom[1:])
from scipy.ndimage import uniform_filter1d
bg_zoom = uniform_filter1d(h_zoom.astype(float), size=21)
bg_zoom = np.maximum(bg_zoom, 1)
pull = (h_zoom - bg_zoom) / np.sqrt(bg_zoom)

ax7.bar(c_zoom, pull, width=c_zoom[1]-c_zoom[0], color="#FFCC80",
        edgecolor="#FF8F00", linewidth=0.3)
ax7.axhline(0, color="black", linewidth=1)
ax7.axhline(3, color="#C62828", linestyle="--", linewidth=1.5)
ax7.axhline(-3, color="#C62828", linestyle="--", linewidth=1.5)
ax7.axvline(M_BPLUS, color="#2E7D32", linestyle=":", linewidth=2)
ax7.axvline(M_OMCCC, color="#C62828", linestyle=":", linewidth=2)
ax7.text(M_BPLUS+5, 5, "B+", fontsize=9, color="#2E7D32", fontweight="bold")
ax7.text(M_OMCCC+5, 5, "Ωccc?", fontsize=9, color="#C62828", fontweight="bold")
ax7.set_xlabel("m(K+π+π-) (MeV)")
ax7.set_ylabel("Pull (σ)")
ax7.set_title("Pull distribution (5100-5600 MeV)\nRecherche Ωccc",
              fontweight="bold")
ax7.set_ylim(-8, 8)
ax7.grid(True, alpha=0.2)

# 8. Résumé
ax8 = fig.add_subplot(gs[2, 1:])
ax8.set_xlim(0, 10); ax8.set_ylim(0, 10); ax8.axis("off")
ax8.set_title("Résumé — Recherche baryons lourds LHCb", fontsize=13,
              fontweight="bold")

y = 9.3
ax8.text(5, y, f"LHCb B→hhh : {np.sum(good):,} événements (7 TeV, MagDown)",
         fontsize=10, ha="center", fontweight="bold", color="#1565C0")

y -= 0.6
headers = ["Cible", "Masse TQNT", "Hypothèse", "Obs", "Att", "σ", "Verdict"]
for j, h in enumerate(headers):
    ax8.text(0.1+j*1.4, y, h, fontsize=7.5, fontweight="bold", ha="center")
y -= 0.2
ax8.plot([0, 10], [y, y], color="#333", linewidth=1)
y -= 0.3

for hyp, masses_arr in [("Kππ", m_Kpp[good]), ("KKK", m_KKK[good]),
                          ("pKπ", m_pKp[good])]:
    for label, mt in [("B+", M_BPLUS), ("Ωccc", M_OMCCC),
                       ("Ξcc++", M_XICC), ("Ωcc+", M_OMCC)]:
        n_o, n_e, s = local_sig(masses_arr, mt)
        if abs(s) > 2 or label in ["Ωccc", "B+"]:
            verdict = "✓ VU" if s > 5 else "⚠ EXCÈS" if s > 3 else \
                      "? indice" if s > 2 else "—"
            col = "#2E7D32" if s > 5 else "#C62828" if s > 3 else \
                  "#FF8F00" if s > 2 else "#666"
            vals = [label, f"{mt:.0f}", hyp, f"{n_o:,}", f"{n_e:.0f}",
                    f"{s:+.1f}", verdict]
            for j, v in enumerate(vals):
                ax8.text(0.1+j*1.4, y, v, fontsize=6.5, ha="center",
                         color=col, fontweight="bold" if s>3 else "normal")
            y -= 0.25

y -= 0.3
ax8.add_patch(FancyBboxPatch((0.05, y-1.8), 9.9, 1.9,
              boxstyle="round,pad=0.1",
              facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2))
ax8.text(5, y, "CONCLUSIONS", fontsize=11, ha="center",
         fontweight="bold", color="#BF360C")
conclusions = [
    f"• {np.sum(good):,} événements LHCb analysés (B→K+π+π-, 7 TeV)",
    f"• Pic B+ à 5279 MeV : VISIBLE (validation de l'analyse)",
    f"• Zone Ωccc (5359 MeV) : examinée sous 3 hypothèses de masse",
    "• Ce dataset est optimisé pour B+, PAS pour les baryons charmés",
    "  → Les désintégrations Ωccc ne produisent pas K+π+π-",
    "• Pour une vraie recherche Ωccc : utiliser le service de ntupling LHCb",
    "  → Demander le canal Ωccc → Ξcc+ π+ → (Λc+ K- π+) π+",
    "  → opendata-lhcb-ntupling-service.app.cern.ch",
]
for i, line in enumerate(conclusions):
    ax8.text(0.3, y-0.25-i*0.19, line, fontsize=7,
             color="#BF360C" if "ntupling" in line else "#333",
             fontweight="bold" if "ntupling" in line else "normal")

fig.suptitle("LHCb Open Data — Recherche de baryons lourds Gen 2/Gen 3 (Ωccc, Ξbc)\n"
             f"5,1 millions d'événements B→hhh, √s = 7 TeV",
             fontsize=15, fontweight="bold", y=0.99)
fig.savefig(f"{OUT}/analyse_lhcb_baryons.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → analyse_lhcb_baryons.png")
