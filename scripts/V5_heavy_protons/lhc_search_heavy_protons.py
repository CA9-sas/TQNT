"""
Recherche TQNT-V5 des "protons lourds" Gen 5+ dans les données LHC ouvertes
(version vectorisée awkward/numpy, pas de Python loop par event).

DATASETS UTILISÉS :
  - ATLAS exactly2lep    : 664k events 2 leptons (data/ATLAS_13TeV/data_A.exactly2lep.root)
  - LHCb B2HHH MagnetDown : 5.1M events B → 3 hadrons (data/LHCb/B2HHH_MagnetDown.root)

PRÉDICTIONS V5 (papier V5p4) :
  #262 : 6_1+6_1+7_3 m=2.919 GeV  Q=+1  J=1/2  τ=3.05 s
  #269 : 6_3+9_1+9_1 m=3.037 GeV  Q=+1  J=1/2  τ=2.50 s
  #283 : 4_1+6_1+9_1 m=3.232 GeV  Q=+1  J=1/2  τ=1.83 s

LIMITES STRUCTURALES (ouvert data sans dE/dx ni TOF) :
  - Proxy : invariant mass dilepton ou dihadron
  - J/ψ à 3.097 GeV domine la fenêtre des prédictions V5
  - Recherche ≃ "borne supérieure effective" en production standard
"""

import math
import sys
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_PI = 0.139570
M_K  = 0.493677
M_MU = 0.105658
M_PROTON = 0.938272

V5_PREDICTIONS = [
    {"id": 262, "config": "6_1+6_1+7_3", "mass": 2.919},
    {"id": 269, "config": "6_3+9_1+9_1", "mass": 3.037},
    {"id": 283, "config": "4_1+6_1+9_1", "mass": 3.232},
]

SM_RESONANCES = [
    {"name": "J/ψ",    "mass": 3.0969},
    {"name": "ψ(2S)",  "mass": 3.6861},
    {"name": "Λ_c⁺",   "mass": 2.2865},
    {"name": "D⁰",     "mass": 1.8648},
    {"name": "φ",      "mass": 1.0195},
    {"name": "B+/0",   "mass": 5.2796},
]

OUT = Path("/home/euk9/projets/physique_nouvelle/output")
OUT.mkdir(parents=True, exist_ok=True)


def inv_mass_pairs(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    """Masse invariante (vectorisée) pour 2 particules."""
    px1 = pt1 * np.cos(phi1); py1 = pt1 * np.sin(phi1); pz1 = pt1 * np.sinh(eta1)
    px2 = pt2 * np.cos(phi2); py2 = pt2 * np.sin(phi2); pz2 = pt2 * np.sinh(eta2)
    E1 = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
    E2 = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
    E = E1 + E2
    px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
    m2_inv = E*E - (px*px + py*py + pz*pz)
    return np.sqrt(np.maximum(m2_inv, 0))


def search_atlas_dilepton(file_path: Path):
    print("=" * 78)
    print(f"ATLAS exactly2lep — {file_path.name}")
    print("=" * 78)
    f = uproot.open(file_path)
    tree = f["mini"]
    n_total = tree.num_entries
    print(f"  Total events : {n_total}")
    arrs = tree.arrays(
        ["lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_charge"],
        entry_stop=n_total
    )
    mask = arrs["lep_n"] >= 2
    sel = arrs[mask]
    pt = sel["lep_pt"][:, :2] / 1000.0   # MeV -> GeV
    eta = sel["lep_eta"][:, :2]
    phi = sel["lep_phi"][:, :2]
    ch = sel["lep_charge"][:, :2]
    pt1 = np.asarray(pt[:, 0]); pt2 = np.asarray(pt[:, 1])
    eta1 = np.asarray(eta[:, 0]); eta2 = np.asarray(eta[:, 1])
    phi1 = np.asarray(phi[:, 0]); phi2 = np.asarray(phi[:, 1])
    ch1 = np.asarray(ch[:, 0]); ch2 = np.asarray(ch[:, 1])
    m = inv_mass_pairs(pt1, eta1, phi1, M_MU, pt2, eta2, phi2, M_MU)
    OS = ch1 * ch2 < 0
    SS = ch1 * ch2 > 0
    print(f"  Events 2+ leptons : {len(m)}")
    print(f"  Opposite-sign     : {OS.sum()}")
    print(f"  Same-sign         : {SS.sum()}")
    return m[OS], m[SS]


def search_lhcb_dihadron(file_path: Path, n_max: int = 500_000):
    print("=" * 78)
    print(f"LHCb B2HHH — {file_path.name}")
    print("=" * 78)
    f = uproot.open(file_path)
    tree = f["DecayTree"]
    print(f"  Total events : {tree.num_entries}")
    print(f"  Limite       : {n_max}")
    arrs = tree.arrays(
        ["H1_PX", "H1_PY", "H1_PZ", "H1_Charge", "H1_ProbK", "H1_ProbPi",
         "H2_PX", "H2_PY", "H2_PZ", "H2_Charge", "H2_ProbK", "H2_ProbPi",
         "H3_PX", "H3_PY", "H3_PZ", "H3_Charge", "H3_ProbK", "H3_ProbPi"],
        entry_stop=n_max
    )
    masses = {}
    for label, (i, j) in [("H12", (1, 2)), ("H13", (1, 3)), ("H23", (2, 3))]:
        px1 = np.asarray(arrs[f"H{i}_PX"]) / 1000.0
        py1 = np.asarray(arrs[f"H{i}_PY"]) / 1000.0
        pz1 = np.asarray(arrs[f"H{i}_PZ"]) / 1000.0
        ch1 = np.asarray(arrs[f"H{i}_Charge"])
        m1 = np.where(np.asarray(arrs[f"H{i}_ProbK"]) >
                      np.asarray(arrs[f"H{i}_ProbPi"]), M_K, M_PI)
        E1 = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
        px2 = np.asarray(arrs[f"H{j}_PX"]) / 1000.0
        py2 = np.asarray(arrs[f"H{j}_PY"]) / 1000.0
        pz2 = np.asarray(arrs[f"H{j}_PZ"]) / 1000.0
        ch2 = np.asarray(arrs[f"H{j}_Charge"])
        m2 = np.where(np.asarray(arrs[f"H{j}_ProbK"]) >
                      np.asarray(arrs[f"H{j}_ProbPi"]), M_K, M_PI)
        E2 = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
        E = E1 + E2
        px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
        m2_inv = E*E - (px*px + py*py + pz*pz)
        m_pair = np.sqrt(np.maximum(m2_inv, 0))
        # Garder OS pairs uniquement
        os_mask = (ch1 * ch2) < 0
        masses[label] = m_pair[os_mask]
        print(f"  {label} pairs OS : {len(masses[label])}")
    return masses


def analyze_peak(masses: np.ndarray, m_pred: float, window: float = 0.10):
    in_sig = (masses >= m_pred - window) & (masses <= m_pred + window)
    in_l = (masses >= m_pred - 3*window) & (masses < m_pred - window)
    in_r = (masses > m_pred + window) & (masses <= m_pred + 3*window)
    n_sig = int(in_sig.sum())
    n_bg = (int(in_l.sum()) + int(in_r.sum())) / 2.0
    excess = n_sig - n_bg
    sigma = excess / math.sqrt(n_bg) if n_bg > 0 else float("inf")
    return n_sig, n_bg, excess, sigma


def plot_results(atlas_OS, lhcb_h12):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax = axes[0]
    bins = np.linspace(1.5, 4.5, 250)
    ax.hist(atlas_OS, bins=bins, alpha=0.7, color="steelblue",
            label=f"ATLAS m(lep+ lep-) OS, n={len(atlas_OS)}")
    for p in V5_PREDICTIONS:
        ax.axvline(p["mass"], color="red", linestyle="--", alpha=0.8,
                   linewidth=2, label=f"V5 #{p['id']} {p['mass']:.2f} GeV")
    for r in SM_RESONANCES:
        if 1.5 <= r["mass"] <= 4.5:
            ax.axvline(r["mass"], color="green", linestyle=":", alpha=0.6,
                       label=f"{r['name']} {r['mass']:.2f}")
    ax.set_xlabel("m(lep+ lep-) [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title("ATLAS exactly2lep — invariant mass dilepton OS")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0.5, 5.5, 300)
    ax.hist(lhcb_h12, bins=bins, alpha=0.7, color="darkorange",
            label=f"LHCb m(H1 H2) OS, n={len(lhcb_h12)}")
    for p in V5_PREDICTIONS:
        ax.axvline(p["mass"], color="red", linestyle="--", alpha=0.8,
                   linewidth=2, label=f"V5 #{p['id']} {p['mass']:.2f}")
    for r in SM_RESONANCES:
        if 0.5 <= r["mass"] <= 5.5:
            ax.axvline(r["mass"], color="green", linestyle=":", alpha=0.6,
                       label=f"{r['name']} {r['mass']:.2f}")
    ax.set_xlabel("m(H1, H2) sous hyp. K/π [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title("LHCb B2HHH — invariant mass 2 hadrons OS")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT / "lhc_v5_heavy_protons_search.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nFigure : {out_path}")


def main():
    # ATLAS : data_A.exactly2lep (664k events)
    atlas_path = Path("data/ATLAS_13TeV/data_A.exactly2lep.root")
    atlas_OS, atlas_SS = search_atlas_dilepton(atlas_path)
    print()

    # LHCb : 500k events sur 5.1M
    lhcb_path = Path("data/LHCb/B2HHH_MagnetDown.root")
    lhcb_masses = search_lhcb_dihadron(lhcb_path, n_max=500_000)
    print()

    # Analyses des excès
    print("=" * 78)
    print("EXCÈS aux masses V5 prédites (fenêtre ±0.10 GeV)")
    print("=" * 78)
    print("\n[ATLAS m(l+l-) OS]")
    print(f"  {'cible':>15s}  {'m_pred':>8s}  {'n_sig':>7s}  {'n_bg':>9s}  {'excess':>8s}  {'σ':>6s}")
    for p in V5_PREDICTIONS:
        n, b, e, s = analyze_peak(atlas_OS, p["mass"])
        print(f"  V5 #{p['id']:>4d}        {p['mass']:>8.3f}  {n:>7d}  {b:>9.1f}  {e:>+8.1f}  {s:>+6.1f}")
    # Contrôle J/ψ
    n, b, e, s = analyze_peak(atlas_OS, 3.0969)
    print(f"  {'J/ψ (ctrl)':>15s}  {3.0969:>8.3f}  {n:>7d}  {b:>9.1f}  {e:>+8.1f}  {s:>+6.1f}")

    print("\n[LHCb m(H1,H2) OS]")
    print(f"  {'cible':>15s}  {'m_pred':>8s}  {'n_sig':>7s}  {'n_bg':>9s}  {'excess':>8s}  {'σ':>6s}")
    h12 = lhcb_masses["H12"]
    for p in V5_PREDICTIONS:
        n, b, e, s = analyze_peak(h12, p["mass"])
        print(f"  V5 #{p['id']:>4d}        {p['mass']:>8.3f}  {n:>7d}  {b:>9.1f}  {e:>+8.1f}  {s:>+6.1f}")
    n, b, e, s = analyze_peak(h12, 1.8648)
    print(f"  {'D⁰ (ctrl)':>15s}  {1.8648:>8.3f}  {n:>7d}  {b:>9.1f}  {e:>+8.1f}  {s:>+6.1f}")

    plot_results(atlas_OS, h12)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""
  Limites structurales :
    1. ATLAS/LHCb open data sans dE/dx ni TOF — pas d'identification
       par MASSE des tracks individuels.
    2. J/ψ à 3.0969 GeV domine la fenêtre [2.9, 3.2] des prédictions V5.
    3. Si V5 #262 existe, il apparaîtrait comme PISTE STABLE ISOLÉE
       (τ=3s >> géométrie détecteur), pas comme résonance dileptonique.

  Cette recherche est un PROXY. Tout excès >3σ aux masses V5 mérite
  réanalyse HSCP dédiée avec accès aux variables dE/dx ATLAS/CMS.
  Silence statistique = borne supérieure effective sur σ(pp → #262) via
  cinématique 2-lepton ou 2-hadron.
""")


if __name__ == "__main__":
    main()
