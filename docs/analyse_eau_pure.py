#!/usr/bin/env python3
"""
Analyse eau pure — IceCube neutrinos + HITRAN spectre + facteurs de forme proton
Recherche de signatures TQNT aux masses Gen 2/3/4.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import json, os, glob, tarfile, io

DATA = "/home/euk9/projets/physique_nouvelle/data/water"
OUT = "/home/euk9/projets/physique_nouvelle/docs"

# Masses TQNT protons lourds (GeV)
M_P = 0.938; M_GEN2 = 1.95; M_GEN3 = 2.98; M_GEN4 = 4.44

fig = plt.figure(figsize=(26, 20))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

# ================================================================
# 1. IceCube HESE — neutrinos haute énergie dans la glace
# ================================================================
ax1 = fig.add_subplot(gs[0, 0])
print("=== IceCube HESE 7.5 ans ===")
with open(os.path.join(DATA, "icecube_hese7yr_events.json")) as f:
    hese = json.load(f)
energies = np.array(hese["recoDepositedEnergy"]) / 1000  # MeV → GeV
morpho = np.array(hese["recoMorphology"])  # 0=cascade, 1=track, 2=double
print(f"  {len(energies)} événements HESE, E = {energies.min():.0f}-{energies.max():.0f} GeV")

ax1.hist(energies, bins=50, range=(10, 2500), color="#90CAF9",
         edgecolor="#1565C0", linewidth=0.5, label=f"{len(energies)} événements")
# Marquer les masses TQNT × facteur (elles sont en GeV, pas en TeV)
# Les neutrinos HESE sont > 10 TeV, les masses protons sont en GeV
# → pas directement comparable, mais cherchons des structures
ax1.set_xlabel("Énergie déposée (GeV)")
ax1.set_ylabel("Événements")
ax1.set_title(f"IceCube HESE — {len(energies)} neutrinos haute énergie\n"
              f"dans la glace (eau)", fontweight="bold")
ax1.set_yscale("log"); ax1.grid(True, alpha=0.2)
ax1.legend(fontsize=8)

# 2. IceCube all-sky — spectre d'énergie des neutrinos
ax2 = fig.add_subplot(gs[0, 1])
print("\n=== IceCube All-Sky IC86-VII ===")
allsky_file = os.path.join(DATA, "icecube_allsky_IC86_VII_events.tab")
try:
    data_allsky = np.loadtxt(allsky_file, comments="#", dtype=str)
    # Colonnes: MJD, log10(E/GeV), AngErr, RA, Dec, Az, Zen
    log_e = data_allsky[:, 1].astype(float)
    energies_allsky = 10**log_e  # GeV
    print(f"  {len(log_e):,} événements, E = {energies_allsky.min():.1f}-{energies_allsky.max():.0f} GeV")

    ax2.hist(log_e, bins=100, range=(1, 7), color="#A5D6A7",
             edgecolor="#2E7D32", linewidth=0.3)
    # Marquer les masses protons Gen 2/3/4 en log10
    for label, m, col in [("p Gen2", M_GEN2, "#E53935"),
                           ("p Gen3", M_GEN3, "#FF8F00"),
                           ("p Gen4", M_GEN4, "#6A1B9A")]:
        log_m = np.log10(m)
        ax2.axvline(log_m, color=col, linestyle="--", linewidth=2, alpha=0.7)
        ax2.text(log_m+0.05, ax2.get_ylim()[1]*0.5 if ax2.get_ylim()[1]>0 else 100,
                 f"{label}\n{m} GeV", fontsize=7, color=col, fontweight="bold",
                 rotation=90, va="center")
    ax2.set_xlabel("log₁₀(E/GeV)")
    ax2.set_ylabel("Événements")
    ax2.set_title(f"IceCube All-Sky ({len(log_e):,} νs)\nMasses protons Gen 2/3/4 marquées",
                  fontweight="bold")
    ax2.grid(True, alpha=0.2)
except Exception as e:
    print(f"  Erreur: {e}")
    ax2.text(0.5, 0.5, f"Erreur: {e}", transform=ax2.transAxes)

# 3. IceCube sterile neutrino — Δm² vs sin²(2θ)
ax3 = fig.add_subplot(gs[0, 2])
print("\n=== IceCube neutrino stérile (Δm² vs sin²2θ) ===")
try:
    sterile = np.loadtxt(os.path.join(DATA, "icecube_sterile_8yr_LLH.txt"),
                          comments="#")
    sin2 = sterile[:, 0]
    dm2 = sterile[:, 1]
    llh = sterile[:, 2]

    # Reshape into 2D grid
    sin2_unique = np.unique(sin2)
    dm2_unique = np.unique(dm2)
    n_sin = len(sin2_unique)
    n_dm = len(dm2_unique)
    print(f"  Grille: {n_sin}×{n_dm}, Δm² = {dm2_unique.min():.4f}-{dm2_unique.max():.1f} eV²")

    if n_sin * n_dm == len(llh):
        LLH = llh.reshape(n_dm, n_sin)
        ax3.contourf(np.log10(sin2_unique), np.log10(dm2_unique), LLH,
                     levels=20, cmap="RdYlBu_r")
        ax3.set_xlabel("log₁₀(sin²2θ₂₄)")
        ax3.set_ylabel("log₁₀(Δm²₄₁ / eV²)")
        ax3.set_title("IceCube stérile 8 ans\nΔm² vs sin²2θ₂₄",
                      fontweight="bold")

        # TQNT prediction: ν₄ mass ~1.3 MeV → Δm²₄₁ ≈ (1.3e-3)² = 1.7e-6 eV²
        # Trop petit pour cette grille (min = 0.028 eV²)
        ax3.text(0.05, 0.05, "ν₄ TQNT (1,3 MeV)\nΔm²≈10⁻⁶ eV²\n(hors grille)",
                 transform=ax3.transAxes, fontsize=7, color="white",
                 fontweight="bold",
                 bbox=dict(boxstyle="round", fc="#C62828", alpha=0.8))
    else:
        ax3.text(0.5, 0.5, "Grille non rectangulaire", transform=ax3.transAxes)
except Exception as e:
    print(f"  Erreur: {e}")
    ax3.text(0.5, 0.5, str(e)[:50], transform=ax3.transAxes)

# 4. HITRAN — spectre d'eau (mode bending ν₂)
ax4 = fig.add_subplot(gs[1, 0])
print("\n=== HITRAN H₂O — mode de pliage ν₂ ===")
try:
    # HITRAN .par format: colonnes fixes
    # Pos 3-15: wavenumber (cm⁻¹), 15-25: intensity (cm/molecule)
    wavenumbers = []
    intensities = []
    for parfile in ["hitran_h2o_1400_1900.par", "hitran_h2o_3400_3900.par"]:
        path = os.path.join(DATA, parfile)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if len(line) > 25:
                        try:
                            wn = float(line[3:15])
                            si = float(line[15:25])
                            wavenumbers.append(wn)
                            intensities.append(si)
                        except:
                            pass
    wn = np.array(wavenumbers)
    si = np.array(intensities)
    print(f"  {len(wn):,} raies, ν = {wn.min():.1f}-{wn.max():.1f} cm⁻¹")

    ax4.semilogy(wn, si, '.', color="#1565C0", markersize=1, alpha=0.3)
    # Modes vibrationnels de H₂O
    # ν₂ (bending) ≈ 1595 cm⁻¹, ν₁ (sym stretch) ≈ 3657 cm⁻¹, ν₃ (asym) ≈ 3756 cm⁻¹
    for mode, freq, label in [("ν₂", 1595, "Pliage O-H"),
                                ("ν₁", 3657, "Étirement sym"),
                                ("ν₃", 3756, "Étirement asym")]:
        ax4.axvline(freq, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
        ax4.text(freq+10, 1e-20, f"{mode}\n{freq} cm⁻¹\n{label}", fontsize=6,
                 color="#E53935", rotation=90, va="bottom")
    ax4.set_xlabel("Nombre d'onde (cm⁻¹)")
    ax4.set_ylabel("Intensité (cm/molécule)")
    ax4.set_title(f"HITRAN H₂O — {len(wn):,} raies spectrales\n"
                  f"Vibrations de la liaison O-H", fontweight="bold")
    ax4.grid(True, alpha=0.2)
except Exception as e:
    print(f"  Erreur: {e}")

# 5. HITRAN — conversion fréquences en énergie et comparaison TQNT
ax5 = fig.add_subplot(gs[1, 1])
if len(wn) > 0:
    # Énergie des transitions en eV: E = h×c×ν = 1.2398e-4 × ν(cm⁻¹) eV
    E_eV = 1.2398e-4 * wn
    E_MeV = E_eV * 1e-3

    # Les transitions HITRAN sont à ~0.17-0.47 eV (IR)
    # Les masses protons sont en GeV — ordres de grandeur très différents
    # Mais la TQNT prédit des résonances topologiques
    # Le rapport m_proton / E_transition pourrait révéler des ratios TQNT

    # Ratio m_p / E_transition typique
    E_nu2 = 1.2398e-4 * 1595  # eV (mode bending)
    E_nu1 = 1.2398e-4 * 3657  # eV (mode stretching)
    ratio_p_nu2 = M_P * 1e9 / (E_nu2 * 1e6)  # sans unité

    ax5.hist(E_eV, bins=200, range=(0.15, 0.50), color="#FFCC80",
             edgecolor="#FF8F00", linewidth=0.3)
    ax5.set_xlabel("Énergie de transition (eV)")
    ax5.set_ylabel("Nombre de raies")
    ax5.set_title("Distribution des énergies de transition H₂O\n"
                  f"m_p/E(ν₂) = {ratio_p_nu2:.0f}", fontweight="bold")
    ax5.grid(True, alpha=0.2)

    # Annoter les rapports avec les masses TQNT
    info = (f"m_p = 938 MeV, E(ν₂) = {E_nu2*1000:.1f} meV\n"
            f"Rapport m_p/E(ν₂) = {ratio_p_nu2:.0f}\n"
            f"Rapport m_Gen2/E(ν₂) = {M_GEN2*1e9/(E_nu2*1e6):.0f}\n"
            f"Rapport m_Gen3/E(ν₂) = {M_GEN3*1e9/(E_nu2*1e6):.0f}\n"
            f"Rapport m_Gen4/E(ν₂) = {M_GEN4*1e9/(E_nu2*1e6):.0f}")
    ax5.text(0.95, 0.95, info, transform=ax5.transAxes, fontsize=7,
             va="top", ha="right",
             bbox=dict(boxstyle="round", fc="white", ec="#FF8F00"))

# 6. Facteurs de forme du proton
ax6 = fig.add_subplot(gs[1, 2])
print("\n=== Facteurs de forme du proton ===")
try:
    # Extraire les données de HepData
    tar_path = os.path.join(DATA, "hepdata_proton_ff_ins850_all.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getnames()
            print(f"  Fichiers dans l'archive: {members}")
            # Chercher des fichiers CSV/YAML
            for m in members:
                if m.endswith('.csv') or m.endswith('.yaml'):
                    f = tar.extractfile(m)
                    if f:
                        content = f.read().decode('utf-8')
                        print(f"  {m}: {len(content)} bytes")
                        print(f"    Début: {content[:200]}")

    # Lire les CSV directs
    ff_data = []
    for csvf in sorted(glob.glob(os.path.join(DATA, "hepdata_proton_ff_*.csv"))):
        try:
            lines = open(csvf).readlines()
            data_lines = [l for l in lines if not l.startswith('#') and l.strip() and ',' in l]
            if data_lines:
                header = None
                for l in lines:
                    if l.startswith('#:') and 'cmenergies' in l.lower():
                        header = l
                for dl in data_lines:
                    parts = dl.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            ff_data.append([float(parts[0]), float(parts[1])])
                        except:
                            pass
            print(f"  {os.path.basename(csvf)}: {len(data_lines)} points")
        except Exception as e:
            print(f"  Erreur {csvf}: {e}")

    if ff_data:
        ff = np.array(ff_data)
        ax6.scatter(ff[:,0], ff[:,1], c="#1565C0", s=20, zorder=5)
        ax6.set_xlabel("Q² ou |t| (GeV²)")
        ax6.set_ylabel("dσ/dΩ ou G_E")
        ax6.set_title("Facteur de forme du proton\ne⁻p → e⁻p (Borkowski 1975)",
                      fontweight="bold")
        ax6.grid(True, alpha=0.2)
    else:
        ax6.text(0.5, 0.5, "Pas de données numériques exploitables",
                 transform=ax6.transAxes, ha="center")
        ax6.set_title("Facteur de forme du proton", fontweight="bold")
except Exception as e:
    print(f"  Erreur: {e}")
    ax6.text(0.5, 0.5, str(e)[:80], transform=ax6.transAxes, ha="center", fontsize=8)

# 7. Spectre IR NIST (brut)
ax7 = fig.add_subplot(gs[2, 0])
print("\n=== NIST IR H₂O ===")
try:
    ir_file = os.path.join(DATA, "nist_h2o_ir.jdx")
    wn_ir, abs_ir = [], []
    in_data = False
    with open(ir_file) as f:
        for line in f:
            if "##XYDATA" in line or "##XYPOINTS" in line:
                in_data = True
                continue
            if in_data and line.startswith("##"):
                in_data = False
                continue
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        wn_ir.append(float(parts[0]))
                        abs_ir.append(float(parts[1]))
                    except:
                        pass
    if wn_ir:
        ax7.plot(wn_ir, abs_ir, color="#2E7D32", linewidth=0.8)
        ax7.set_xlabel("Nombre d'onde (cm⁻¹)")
        ax7.set_ylabel("Absorbance")
        ax7.set_title(f"NIST — Spectre IR H₂O (liquide)\n{len(wn_ir)} points",
                      fontweight="bold")
        ax7.grid(True, alpha=0.2)
        print(f"  {len(wn_ir)} points, ν = {min(wn_ir):.0f}-{max(wn_ir):.0f} cm⁻¹")
    else:
        ax7.text(0.5, 0.5, "Pas de données spectrales", transform=ax7.transAxes)
except Exception as e:
    print(f"  Erreur: {e}")

# 8. NIST masse spectrum H₂O
ax8 = fig.add_subplot(gs[2, 1])
print("\n=== NIST masse spectrum H₂O ===")
try:
    ms_file = os.path.join(DATA, "nist_h2o_ms.jdx")
    mz, intensity_ms = [], []
    in_data = False
    with open(ms_file) as f:
        for line in f:
            if "##PEAK" in line or "##XYDATA" in line:
                in_data = True
                continue
            if in_data and line.startswith("##"):
                break
            if in_data:
                parts = line.strip().split()
                for i in range(0, len(parts)-1, 2):
                    try:
                        mz.append(int(parts[i]))
                        intensity_ms.append(int(parts[i+1]))
                    except:
                        pass
    if mz:
        ax8.bar(mz, intensity_ms, color="#CE93D8", edgecolor="#6A1B9A", linewidth=1)
        for m, i in zip(mz, intensity_ms):
            species = {16: "O", 17: "OH", 18: "H₂O", 19: "H₃O", 20: "H₂¹⁸O"}.get(m, "")
            ax8.text(m, i+200, f"m/z={m}\n{species}", fontsize=8, ha="center",
                     fontweight="bold")
        ax8.set_xlabel("m/z")
        ax8.set_ylabel("Intensité relative")
        ax8.set_title("Spectre de masse H₂O (NIST)\nIonisation par électrons",
                      fontweight="bold")
        ax8.grid(True, alpha=0.2)
        print(f"  {len(mz)} pics: {list(zip(mz, intensity_ms))}")
    else:
        ax8.text(0.5, 0.5, "Pas de données", transform=ax8.transAxes)
except Exception as e:
    print(f"  Erreur: {e}")

# 9. Résumé
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_xlim(0, 10); ax9.set_ylim(0, 10); ax9.axis("off")
ax9.set_title("Résumé — Analyse eau pure + TQNT", fontsize=13,
              fontweight="bold")

y = 9.3
ax9.add_patch(FancyBboxPatch((0.05, y-4.5), 9.9, 4.6,
              boxstyle="round,pad=0.1",
              facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2))
ax9.text(5, y, "DONNÉES EAU PURE ANALYSÉES", fontsize=10,
         ha="center", fontweight="bold", color="#1B5E20")

results = [
    ("IceCube HESE", "102 νs haute énergie dans la glace",
     "Spectre d'énergie 10 GeV - 2 PeV"),
    ("IceCube All-Sky", "145 751 νs (IC86-VII)",
     "Spectre complet, masses Gen 2/3/4 marquées"),
    ("IceCube stérile", "Δm² vs sin²2θ₂₄ (8 ans)",
     "ν₄ TQNT (1,3 MeV) hors grille mesurée"),
    ("HITRAN H₂O", "14 450 raies spectrales",
     "Modes ν₁, ν₂, ν₃ de la liaison O-H"),
    ("NIST IR", "Spectre infrarouge complet",
     "Bandes d'absorption 650-4000 cm⁻¹"),
    ("NIST MS", "Spectre de masse H₂O",
     "m/z = 16(O), 17(OH), 18(H₂O)"),
    ("HepData", "Facteurs de forme du proton",
     "Structure interne p via e⁻p → e⁻p"),
]

for i, (source, desc, detail) in enumerate(results):
    ax9.text(0.3, y - 0.4 - i*0.5, f"• {source}", fontsize=8,
             color="#1B5E20", fontweight="bold")
    ax9.text(0.3, y - 0.58 - i*0.5, f"  {desc} — {detail}", fontsize=7,
             color="#333")

y -= 4.0
ax9.text(5, y, "CONNEXION TQNT", fontsize=10, ha="center",
         fontweight="bold", color="#C62828")
conns = [
    "• L'eau contient des protons (H) → cibles pour Gen 2/3/4",
    "• IceCube : neutrinos interagissent avec les protons de H₂O",
    "• HITRAN : les vibrations O-H sondent le potentiel protonique",
    "• Facteurs de forme : structure interne du proton = topologie 3₁",
]
for i, c in enumerate(conns):
    ax9.text(0.3, y - 0.3 - i*0.22, c, fontsize=7.5, color="#C62828")

fig.suptitle("Analyse eau pure — IceCube, HITRAN, NIST, HepData\n"
             "Connexions avec les protons lourds TQNT",
             fontsize=15, fontweight="bold", y=0.99)
fig.savefig(os.path.join(OUT, "analyse_eau_pure.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nOK → analyse_eau_pure.png")
