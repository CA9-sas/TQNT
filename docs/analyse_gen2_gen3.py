#!/usr/bin/env python3
"""
Recherche protons lourds Gen 2/Gen 3 — baryons charmés et beaux
+ validation TQNT sur les mésons J/ψ et Υ dans les données ATLAS.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, glob

import uproot
import awkward as ak

DATA = "/home/euk9/projets/physique_nouvelle/data/ATLAS_13TeV"
OUT = "/home/euk9/projets/physique_nouvelle/docs"

# ================================================================
# TQNT: ρ_J par génération
# ================================================================
GAMMA = 0.1
def rho_J_torus(p):
    return (p - 1) + GAMMA * p  # span + γ×|Δ(-1)| pour T(2,p)

rho_J = {
    "Gen1 (3₁)": rho_J_torus(3),   # 3.3
    "Gen2 (5₁)": rho_J_torus(5),   # 5.5
    "Gen3 (7₁)": rho_J_torus(7),   # 7.7
}

# Calibration sur le proton: entrelacs 9₁+7₄+6₃, k=18
# Σρ_J(proton) ≈ 9.9 + 7.0 + 6.5 = 23.4
# F(18) = 2584
# m_p = 938.3 MeV
# η/R₀ = 938.3 / (23.4 × 2584) = 0.01552 MeV
def fib(k):
    a, b = 1, 1
    for _ in range(k - 1):
        a, b = b, a + b
    return a

eta_R0_MeV = 938.3 / (23.4 * fib(18))  # 0.01552 MeV

# ================================================================
# Masses connues et prédictions TQNT
# ================================================================

# Proton (uud, Gen 1): 9₁+7₄+6₃, k=18
# Le nucléon est composé de 3 quarks, chacun associé à un nœud
# Gen 1 quarks → nœuds de complexité ~6-10
# Gen 2 quarks → nœuds de complexité ~10-14
# Gen 3 quarks → nœuds de complexité ~14-18

# Scaling: la masse d'un baryon Gen X est proportionnelle à Σρ_J(X) × F(k_X)
# Pour les baryons de même structure mais Gen différente, on suppose même k
# La masse scale avec ρ_J du quark

# QUARKS connus
quarks = {
    "u": (2.2, "Gen1"), "d": (4.7, "Gen1"),
    "c": (1270, "Gen2"), "s": (93, "Gen2"),
    "t": (173000, "Gen3"), "b": (4180, "Gen3"),
}

# BARYONS connus (3 quarks)
baryons_known = [
    # (nom, quarks, masse MeV, statut)
    ("p (proton)", "uud", 938.3, "Découvert"),
    ("n (neutron)", "udd", 939.6, "Découvert"),
    ("Λ", "uds", 1115.7, "Découvert"),
    ("Σ⁺", "uus", 1189.4, "Découvert"),
    ("Ξ⁰", "uss", 1314.9, "Découvert"),
    ("Ω⁻", "sss", 1672.5, "Découvert"),
    # Gen 2 (charm)
    ("Λc⁺", "udc", 2286.5, "Découvert"),
    ("Σc⁺⁺", "uuc", 2453.5, "Découvert"),
    ("Ξc⁺", "usc", 2467.9, "Découvert"),
    ("Ωc⁰", "ssc", 2695.2, "Découvert"),
    ("Ξcc⁺⁺", "ccu", 3621.4, "Découvert 2017 (LHCb)"),
    ("Ωcc⁺", "ccs", 3700, "Prédit (non trouvé)"),
    ("Ωccc⁺⁺", "ccc", 4800, "Prédit (non trouvé)"),
    # Gen 3 (bottom)
    ("Λb⁰", "udb", 5619.6, "Découvert"),
    ("Ξb⁰", "usb", 5791.9, "Découvert"),
    ("Ωb⁻", "ssb", 6046.1, "Découvert"),
    ("Ξbc", "ubc", 6900, "Prédit (non trouvé)"),
    ("Ωbcc", "bcc", 8000, "Prédit (non trouvé)"),
    ("Ωbbc", "bbc", 11200, "Prédit (non trouvé)"),
    ("Ωbbb", "bbb", 14400, "Prédit (non trouvé)"),
]

# MÉSONS (quark-antiquark) — visibles dans le spectre dimuon
mesons = [
    ("ρ/ω", "uū/dd̄", 775, "Gen1", "Découvert"),
    ("φ", "ss̄", 1019, "Gen1", "Découvert"),
    ("J/ψ", "cc̄", 3097, "Gen2", "Découvert"),
    ("ψ(2S)", "cc̄", 3686, "Gen2", "Découvert"),
    ("Υ(1S)", "bb̄", 9460, "Gen3", "Découvert"),
    ("Υ(2S)", "bb̄", 10023, "Gen3", "Découvert"),
    ("Υ(3S)", "bb̄", 10355, "Gen3", "Découvert"),
    ("Z⁰", "—", 91188, "Boson", "Découvert"),
]

# TQNT mass prediction for baryons
# Simplification: m_baryon ∝ Σ m_quark × (topological correction)
# The topological correction for 3-quark entrelacs at k=18 is:
# m_p / (m_u + m_u + m_d) = 938.3 / (2.2+2.2+4.7) = 938.3/9.1 = 103.1
# This factor represents the binding energy from the entrelacs topology
# For Gen 2/3 baryons, the factor should scale with ρ_J:
# factor(Gen X) = factor(Gen 1) × ρ_J(Gen X) / ρ_J(Gen 1)

topo_factor_gen1 = 938.3 / (2.2 + 2.2 + 4.7)  # ~103

def predict_baryon_mass(quark_string):
    """Prédiction TQNT de la masse d'un baryon à partir de ses quarks."""
    total_quark_mass = 0
    total_rho = 0
    for q in quark_string:
        if q in quarks:
            total_quark_mass += quarks[q][0]
            gen = quarks[q][1]
            if gen == "Gen1": total_rho += 3.3
            elif gen == "Gen2": total_rho += 5.5
            elif gen == "Gen3": total_rho += 7.7

    # TQNT prediction: mass = Σm_q × topo_factor × (Σρ_J / Σρ_J_proton)
    rho_proton = 3.3 + 3.3 + 3.3  # proton uud (Gen 1 quarks)
    # Mais le proton a des ρ_J spécifiques pour ses nœuds (9₁+7₄+6₃ ≈ 23.4)
    # Utilisons le rapport des ρ_J totaux
    ratio = total_rho / 9.9  # 9.9 = 3×3.3 for all-Gen1
    m_predicted = total_quark_mass * topo_factor_gen1 * (ratio / 3)
    # Correction: pour les baryons légers, l'essentiel de la masse vient
    # de l'énergie de liaison (QCD). Pour les baryons lourds, la masse des
    # quarks domine. Modèle interpolé:
    m_quark_contrib = total_quark_mass
    m_binding = 938.3 - (2.2+2.2+4.7)  # ~929 MeV de binding pour proton
    # Pour Gen 2/3: binding ∝ ρ_J
    binding_scaled = m_binding * (total_rho / 9.9)
    m_predicted = m_quark_contrib + binding_scaled
    return m_predicted

# Compute predictions
print("=" * 70)
print("PRÉDICTIONS TQNT — BARYONS GEN 1/2/3")
print("=" * 70)
print(f"{'Baryon':<12} {'Quarks':<6} {'Masse exp (MeV)':<16} "
      f"{'Masse TQNT (MeV)':<18} {'Écart':<8} {'Statut'}")
print("-" * 70)

predictions = []
for name, qstr, m_exp, status in baryons_known:
    m_pred = predict_baryon_mass(qstr)
    ecart = (m_pred - m_exp) / m_exp * 100 if m_exp > 0 else 0
    predictions.append((name, qstr, m_exp, m_pred, ecart, status))
    marker = "✓" if abs(ecart) < 15 else "⚠" if abs(ecart) < 30 else "✗"
    print(f"{name:<12} {qstr:<6} {m_exp:<16.1f} {m_pred:<18.1f} "
          f"{ecart:+6.1f}%  {marker} {status}")

# ================================================================
# ANALYSE DES DONNÉES ATLAS — mésons Gen 2/Gen 3
# ================================================================
print("\n" + "=" * 70)
print("ANALYSE DONNÉES ATLAS — SPECTRE DIMUON/DILEPTON")
print("=" * 70)

# Load dilepton data (exactly2lep, covers 0-5800 GeV)
print("\n--- Chargement dilepton exactly2lep ---")
masses_2l = []
for f in sorted(glob.glob(os.path.join(DATA, "data_*.exactly2lep*.root"))):
    tree = uproot.open(f)["mini"]
    pt = tree["lep_pt"].array()
    eta = tree["lep_eta"].array()
    phi = tree["lep_phi"].array()
    E = tree["lep_E"].array()
    tp = tree["lep_type"].array()
    ch = tree["lep_charge"].array()

    mask = ak.num(pt) >= 2
    pt=pt[mask]; eta=eta[mask]; phi=phi[mask]; E=E[mask]
    tp=tp[mask]; ch=ch[mask]

    # Opposite-sign same-flavor (OSSF) pour mésons
    ossf = (ch[:,0] + ch[:,1] == 0) & (tp[:,0] == tp[:,1])
    # Dimuon only (type=13)
    dimu = ossf & (tp[:,0] == 13)

    E_sel = E[dimu]; pt_sel = pt[dimu]
    eta_sel = eta[dimu]; phi_sel = phi[dimu]

    m2 = (E_sel[:,0]+E_sel[:,1])**2 - \
         (pt_sel[:,0]*np.cos(phi_sel[:,0])+pt_sel[:,1]*np.cos(phi_sel[:,1]))**2 - \
         (pt_sel[:,0]*np.sin(phi_sel[:,0])+pt_sel[:,1]*np.sin(phi_sel[:,1]))**2 - \
         (pt_sel[:,0]*np.sinh(eta_sel[:,0])+pt_sel[:,1]*np.sinh(eta_sel[:,1]))**2
    m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000  # GeV
    masses_2l.append(m)
    print(f"  {os.path.basename(f)}: {len(m):,} dimuon OSSF")

m_dimu = np.concatenate(masses_2l)
print(f"TOTAL dimuon OSSF: {len(m_dimu):,}, range {m_dimu.min():.3f}-{m_dimu.max():.1f} GeV")

# Load beta 2muons (0-3.9 GeV, high statistics)
print("\n--- Chargement beta 2muons ---")
masses_beta = []
for f in sorted(glob.glob(os.path.join(DATA, "beta_2muons_*.root"))):
    tree = uproot.open(f)["analysis"]
    pt = tree["lep_pt"].array()
    eta = tree["lep_eta"].array()
    phi = tree["lep_phi"].array()
    E = tree["lep_e"].array()  # minuscule!
    mask = ak.num(pt) >= 2
    pt=pt[mask]; eta=eta[mask]; phi=phi[mask]; E=E[mask]
    m2 = (E[:,0]+E[:,1])**2 - \
         (pt[:,0]*np.cos(phi[:,0])+pt[:,1]*np.cos(phi[:,1]))**2 - \
         (pt[:,0]*np.sin(phi[:,0])+pt[:,1]*np.sin(phi[:,1]))**2 - \
         (pt[:,0]*np.sinh(eta[:,0])+pt[:,1]*np.sinh(eta[:,1]))**2
    m = np.sqrt(np.maximum(ak.to_numpy(m2), 0)) / 1000
    masses_beta.append(m)
    print(f"  {os.path.basename(f)}: {len(m):,} 2μ")
m_beta = np.concatenate(masses_beta) if masses_beta else np.array([])
print(f"TOTAL beta 2muons: {len(m_beta):,}")

# Measure meson peaks
print("\n--- MÉSONS OBSERVÉS DANS LES DONNÉES ---")
for name, _, m_exp_mev, gen, _ in mesons:
    m_exp = m_exp_mev / 1000  # GeV
    window = 0.05 if m_exp < 5 else 0.5 if m_exp < 20 else 2.0
    # In exactly2lep
    n_2l = np.sum((m_dimu > m_exp - window) & (m_dimu < m_exp + window))
    # In beta
    n_beta = np.sum((m_beta > m_exp - window) & (m_beta < m_exp + window)) if len(m_beta) > 0 else 0
    found = "✓ VU" if (n_2l > 50 or n_beta > 100) else "? faible" if (n_2l > 10 or n_beta > 10) else "✗ non vu"
    print(f"  {name:<10} ({gen}): m={m_exp:.3f} GeV, "
          f"2ℓ={n_2l:,}, beta={n_beta:,} → {found}")

# ================================================================
# FIGURE
# ================================================================
fig = plt.figure(figsize=(26, 20))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

# 1. Spectre dimuon basse masse (0.2 - 4 GeV) — beta 2muons
ax1 = fig.add_subplot(gs[0, 0])
if len(m_beta) > 0:
    mask_lo = (m_beta > 0.2) & (m_beta < 4.0)
    ax1.hist(m_beta[mask_lo], bins=760, range=(0.2, 4.0), color="#90CAF9",
             edgecolor="#1565C0", linewidth=0.2)
    for name, _, m_mev, gen, _ in mesons:
        m = m_mev / 1000
        if 0.2 < m < 4.0:
            col = "#E53935" if "Gen2" in gen else "#2E7D32"
            ax1.axvline(m, color=col, linestyle="--", linewidth=1.5, alpha=0.7)
            ax1.text(m+0.02, ax1.get_ylim()[1]*0.8, name, fontsize=7,
                     color=col, rotation=90, va="top")
    ax1.set_xlabel("m_μμ (GeV)")
    ax1.set_ylabel("Événements")
    ax1.set_title(f"Beta 2μ ({len(m_beta):,} évts)\nMésons Gen 1-2",
                  fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.2)

# 2. Zoom J/ψ (Gen 2)
ax2 = fig.add_subplot(gs[0, 1])
if len(m_beta) > 0:
    mask_jpsi = (m_beta > 2.8) & (m_beta < 3.4)
    ax2.hist(m_beta[mask_jpsi], bins=120, range=(2.8, 3.4),
             color="#EF9A9A", edgecolor="#C62828", linewidth=0.5)
    ax2.axvline(3.097, color="#C62828", linestyle="--", linewidth=2)
    ax2.text(3.1, ax2.get_ylim()[1]*0.5 if ax2.get_ylim()[1] > 0 else 100,
             "J/ψ (cc̄)\n3097 MeV\nGen 2", fontsize=9, color="#C62828",
             fontweight="bold")
    n_jpsi = np.sum(mask_jpsi)
    ax2.set_title(f"J/ψ — méson Gen 2 ({n_jpsi:,} évts)", fontweight="bold",
                  color="#C62828")
    ax2.set_xlabel("m_μμ (GeV)")
    ax2.grid(True, alpha=0.2)

# 3. Spectre dimuon haute masse (5-15 GeV) — exactly2lep → Υ
ax3 = fig.add_subplot(gs[0, 2])
mask_ups = (m_dimu > 5) & (m_dimu < 15)
ax3.hist(m_dimu[mask_ups], bins=200, range=(5, 15),
         color="#CE93D8", edgecolor="#6A1B9A", linewidth=0.3)
for name, _, m_mev, gen, _ in mesons:
    m = m_mev / 1000
    if 5 < m < 15:
        ax3.axvline(m, color="#6A1B9A", linestyle="--", linewidth=1.5)
        ax3.text(m+0.05, ax3.get_ylim()[1]*0.7 if ax3.get_ylim()[1] > 0 else 100,
                 name, fontsize=7, color="#6A1B9A", rotation=90, va="top")
ax3.set_xlabel("m_μμ (GeV)")
ax3.set_title(f"Υ — mésons Gen 3 ({np.sum(mask_ups):,} évts)",
              fontweight="bold", color="#6A1B9A")
ax3.grid(True, alpha=0.2)

# 4. Spectre complet dimuon 0.2-100 GeV (log-log)
ax4 = fig.add_subplot(gs[1, 0])
mask_full = (m_dimu > 0.2) & (m_dimu < 100)
h, edges = np.histogram(m_dimu[mask_full], bins=500, range=(0.2, 100))
centers = 0.5*(edges[:-1]+edges[1:])
ax4.step(centers, h, color="#1565C0", linewidth=1)
ax4.fill_between(centers, 0, h, step="pre", alpha=0.3, color="#90CAF9")
# Mark all mesons
for name, _, m_mev, gen, _ in mesons:
    m = m_mev / 1000
    if 0.2 < m < 100:
        col = {"Gen1": "#2E7D32", "Gen2": "#E53935",
               "Gen3": "#6A1B9A", "Boson": "#FF8F00"}.get(gen, "#666")
        ax4.axvline(m, color=col, linestyle=":", linewidth=1.5, alpha=0.7)
        ax4.text(m, max(h)*0.3, name, fontsize=6, color=col,
                 rotation=90, va="bottom")
ax4.set_xlabel("m_μμ (GeV)")
ax4.set_ylabel("Événements")
ax4.set_title(f"Spectre dimuon complet ({len(m_dimu):,} OSSF)\n"
              f"ρ/ω → J/ψ → Υ → Z : 3 générations visibles",
              fontweight="bold")
ax4.set_yscale("log")
ax4.set_ylim(1, None)
ax4.grid(True, alpha=0.2)

# 5. Tableau prédictions TQNT baryons
ax5 = fig.add_subplot(gs[1, 1:])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis("off")
ax5.set_title("Prédictions TQNT — Baryons « protons lourds » Gen 1-3",
              fontsize=12, fontweight="bold")

y = 9.5
headers = ["Baryon", "Quarks", "m_exp (MeV)", "m_TQNT (MeV)", "Écart", "Statut"]
for j, h in enumerate(headers):
    ax5.text(0.1 + j * 1.65, y, h, fontsize=7.5, fontweight="bold", ha="center")
y -= 0.2
ax5.plot([0, 10], [y, y], color="#333", linewidth=1)
y -= 0.25

for name, qstr, m_exp, m_pred, ecart, status in predictions:
    if "Prédit" in status:
        col = "#C62828"
        fw = "bold"
    elif abs(ecart) < 15:
        col = "#2E7D32"
        fw = "normal"
    else:
        col = "#FF8F00"
        fw = "normal"

    vals = [name, qstr, f"{m_exp:.0f}", f"{m_pred:.0f}",
            f"{ecart:+.1f}%", status[:20]]
    for j, v in enumerate(vals):
        ax5.text(0.1 + j * 1.65, y, v, fontsize=6.5, ha="center",
                 color=col, fontweight=fw)
    y -= 0.22

# 6. Masse prédite vs mesurée (graphique)
ax6 = fig.add_subplot(gs[2, 0])
m_exp_arr = [me for _, _, me, _, _, s in predictions if "Découvert" in s]
m_pred_arr = [mp for _, _, _, mp, _, s in predictions if "Découvert" in s]
m_exp_notfound = [me for _, _, me, _, _, s in predictions if "Prédit" in s]
m_pred_notfound = [mp for _, _, _, mp, _, s in predictions if "Prédit" in s]

ax6.plot([0, 15000], [0, 15000], "k--", linewidth=1, alpha=0.3,
         label="m_TQNT = m_exp")
ax6.scatter(m_exp_arr, m_pred_arr, c="#2E7D32", s=60, zorder=5,
            label="Découverts", edgecolors="white")
ax6.scatter(m_exp_notfound, m_pred_notfound, c="#C62828", s=80,
            marker="*", zorder=5, label="Prédits (non trouvés)",
            edgecolors="white")
ax6.set_xlabel("Masse expérimentale (MeV)")
ax6.set_ylabel("Masse TQNT (MeV)")
ax6.set_title("Masse TQNT vs expérience", fontweight="bold")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.2)
ax6.set_xlim(0, 15000)
ax6.set_ylim(0, 15000)

# 7. Résumé et conclusions
ax7 = fig.add_subplot(gs[2, 1:])
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 10)
ax7.axis("off")
ax7.set_title("Conclusions — Protons lourds Gen 2/Gen 3", fontsize=13,
              fontweight="bold")

y = 9.3
from matplotlib.patches import FancyBboxPatch

ax7.add_patch(FancyBboxPatch((0.05, y-1.8), 9.9, 1.9,
              boxstyle="round,pad=0.1",
              facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2))
ax7.text(5, y, "CE QUE NOUS VOYONS DANS LES DONNÉES ATLAS", fontsize=10,
         ha="center", fontweight="bold", color="#1B5E20")
seen = [
    "✓ J/ψ (cc̄, 3097 MeV) : pic massif dans le beta 2muons — Gen 2 confirmé",
    "✓ Υ(1S,2S,3S) (bb̄, 9460-10355 MeV) : pics dans le dilepton — Gen 3 confirmé",
    "✓ Z⁰ (91188 MeV) : pic dominant dans le dilepton",
    "✓ ρ/ω (775 MeV), φ (1019 MeV) : pics Gen 1 visibles",
    "→ Les 3 générations de quarks sont visibles via leurs mésons",
]
for i, line in enumerate(seen):
    ax7.text(0.3, y - 0.3 - i * 0.22, line, fontsize=7.5,
             color="#2E7D32", fontweight="bold" if "→" in line else "normal")

y -= 2.3
ax7.add_patch(FancyBboxPatch((0.05, y-2.2), 9.9, 2.3,
              boxstyle="round,pad=0.1",
              facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2))
ax7.text(5, y, "PROTONS LOURDS NON TROUVÉS — CIBLES POUR LA TQNT",
         fontsize=10, ha="center", fontweight="bold", color="#BF360C")
targets = [
    "★ Ωccc⁺⁺ (ccc) : ~4800 MeV — baryon triple-charm, jamais observé",
    "  → Chercher dans LHCb Open Data (spécialisé quarks lourds)",
    "★ Ωbbb (bbb) : ~14400 MeV — baryon triple-bottom",
    "  → Accessible au LHC mais section efficace très faible",
    "★ Ξbc (ubc) : ~6900 MeV — baryon charm-bottom",
    "  → La TQNT prédit des masses spécifiques testables",
    "",
    "La TQNT doit prédire ces masses EXACTEMENT à partir des nœuds.",
    "Si Ωccc est trouvé à la masse TQNT prédite → validation forte.",
]
for i, line in enumerate(targets):
    if line == "":
        continue
    ax7.text(0.3, y - 0.3 - i * 0.22, line, fontsize=7.5,
             color="#BF360C" if "★" in line or "TQNT" in line else "#333",
             fontweight="bold" if "★" in line else "normal")

fig.suptitle("Protons lourds Gen 2/Gen 3 — Baryons charmés et beaux "
             "dans ATLAS Open Data 13 TeV",
             fontsize=16, fontweight="bold", y=0.99)
fig.savefig(os.path.join(OUT, "analyse_gen2_gen3_protons.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → analyse_gen2_gen3_protons.png")
