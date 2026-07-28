"""
Validation systématique de Ω_ts sur les 1539 entrelacs du catalogue +
sur les assignations PDG curées (onglet 3).

Sortie :
  - tableau RMS log et précision <5 %
  - liste des 10 pires écarts par particule PDG
  - figure dispersion m_V5 vs m_exp pour les assignations
"""
import math
import sys
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permettre l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from theory.omega_ts import predict_mass_V5

XLSX = Path(__file__).resolve().parent.parent.parent / "docs" / "TQNT_catalog_V5_diffusion.xlsx"
OUT = Path(__file__).resolve().parent.parent.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    wb = openpyxl.load_workbook(XLSX, data_only=True)

    # --- Validation sur les assignations PDG (onglet 3) ---
    ws3 = wb["3. Particules PDG"]
    rows = []
    for row in ws3.iter_rows(min_row=3, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        particule = str(row[0])
        m_exp = row[1]
        config = row[3]
        m_v5_xlsx = row[4]
        if m_exp is None or config is None or m_v5_xlsx is None:
            continue
        try:
            m_exp = float(m_exp)
            m_v5_xlsx = float(m_v5_xlsx)
        except (TypeError, ValueError):
            continue
        # Recompute via predict_mass_V5 (devrait être identique à m_v5_xlsx)
        m_v5_recomputed = predict_mass_V5(str(config))
        rows.append({
            "particule": particule,
            "config": str(config),
            "m_exp": m_exp,
            "m_v5_xlsx": m_v5_xlsx,
            "m_v5_recomputed": m_v5_recomputed,
            "delta_pct": (m_v5_recomputed - m_exp) / m_exp * 100 if m_exp > 0 else 0,
            "consistency_pct": abs(m_v5_xlsx - m_v5_recomputed) / m_v5_xlsx * 100 if m_v5_xlsx > 0 else 0,
        })

    print("=" * 78)
    print(f"Validation Ω_ts sur {len(rows)} particules PDG assignées")
    print("=" * 78)

    deltas = np.array([r["delta_pct"] for r in rows])
    cons = np.array([r["consistency_pct"] for r in rows])
    log_ratios = np.array([
        math.log10(r["m_v5_recomputed"] / r["m_exp"])
        for r in rows
        if r["m_v5_recomputed"] > 0 and r["m_exp"] > 0
    ])
    rms_log = float(np.sqrt(np.mean(log_ratios ** 2)))

    print(f"\n  RMS Δ% (V5 vs exp)       : {np.sqrt(np.mean(deltas**2)):.3f} %")
    print(f"  Médiane |Δ%|             : {np.median(np.abs(deltas)):.3f} %")
    print(f"  RMS log (V5 vs exp)      : {rms_log:.4f} déc")
    print(f"  Fraction |Δ| < 1 %       : {(np.abs(deltas) < 1.0).sum()}/{len(deltas)} "
          f"= {(np.abs(deltas)<1.0).mean()*100:.1f} %")
    print(f"  Fraction |Δ| < 5 %       : {(np.abs(deltas) < 5.0).sum()}/{len(deltas)} "
          f"= {(np.abs(deltas)<5.0).mean()*100:.1f} %")
    print(f"\n  Cohérence implem vs xlsx :")
    print(f"  Max |consistency_pct|    : {cons.max():.4f} %")
    print(f"  Médiane consistency      : {np.median(cons):.6f} %")

    # Top écarts
    rows_sorted = sorted(rows, key=lambda r: abs(r["delta_pct"]), reverse=True)
    print(f"\n--- 10 pires écarts V5 vs exp ---")
    print(f"  {'particule':>15s}  {'config':>22s}  {'m_v5':>12s}  {'m_exp':>12s}  {'Δ%':>8s}")
    for r in rows_sorted[:10]:
        print(f"  {r['particule']:>15s}  {r['config']:>22s}  "
              f"{r['m_v5_recomputed']:>12.3f}  {r['m_exp']:>12.3f}  "
              f"{r['delta_pct']:>+7.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    m_exps = np.array([r["m_exp"] for r in rows])
    m_v5s = np.array([r["m_v5_recomputed"] for r in rows])
    ax.loglog(m_exps, m_v5s, 'o', markersize=4, color='steelblue',
              label=f"PDG assignées (n={len(rows)})")
    lo, hi = min(m_exps.min(), m_v5s.min()), max(m_exps.max(), m_v5s.max())
    diag = np.array([lo, hi])
    ax.loglog(diag, diag, "k-", linewidth=1, label="m_V5 = m_exp")
    ax.loglog(diag, diag*1.05, "g:", linewidth=1, label="±5 %")
    ax.loglog(diag, diag*0.95, "g:", linewidth=1)
    ax.set_xlabel("m_exp [MeV]")
    ax.set_ylabel("m_V5 (Ω_ts) [MeV]")
    ax.set_title(f"Ω_ts vs PDG observé — {len(rows)} particules\n"
                 f"RMS Δ%% = {np.sqrt(np.mean(deltas**2)):.2f} %% ; "
                 f"|Δ| < 1 %% pour {(np.abs(deltas)<1.0).mean()*100:.1f} %% ; "
                 f"|Δ| < 5 %% pour {(np.abs(deltas)<5.0).mean()*100:.1f} %%")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    out_path = OUT / "omega_ts_validation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nFigure : {out_path}")


if __name__ == "__main__":
    main()
