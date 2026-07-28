"""
Anomaly screening sur le wiDB (Waterisotopes Database, Bowen Lab Utah)
pour identifier des stations candidate à une analyse FT-ICR ciblée
de super-H₂O TQNT-V5.

CONTEXTE :
Le wiDB ne peut PAS détecter directement de la super-H₂O (les δ²H et
δ¹⁸O standards sortent du canal IRMS m/z=2 vs 3, et la super-H sortirait
au m/z=4 non scanné). Mais la base de 42 963 records de précipitation
globales (2010-2020) permet une **anomaly screening** : repérer les
stations dont les ratios isotopiques s'écartent significativement de
la Global Meteoric Water Line (GMWL) de Craig 1961 :
  δ²H = 8 · δ¹⁸O + 10

Hypothèse de travail : si la super-H₂O est produite préférentiellement
à haute altitude (flux cosmique majoré) ou dans des contextes
géophysiques particuliers, les outliers GMWL devraient se concentrer
géographiquement de manière non aléatoire. Le script :

  1. charge le CSV wiDB
  2. calcule la régression GMWL pondérée
  3. identifie les outliers à |residual| > 3σ
  4. produit une carte des stations outliers
  5. tabule les candidats à analyser en FT-ICR

LIMITE EXPLICITE : ce screening n'EST PAS une preuve de super-H₂O. Il
identifie seulement des candidats où une analyse FT-ICR à R ≥ 1.5×10⁴
serait justifiée.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/mnt/data/data_recherche/water_isotopes/wiDB")
OUT = Path("/home/euk9/projets/physique_nouvelle/output")
OUT.mkdir(parents=True, exist_ok=True)

# GMWL Craig 1961 : δ²H = 8 · δ¹⁸O + 10
GMWL_SLOPE = 8.0
GMWL_INTERCEPT = 10.0


def find_csv() -> Path:
    candidates = sorted(DATA.glob("*-data.csv"))
    if not candidates:
        sys.exit(f"Pas de fichier wiDB *-data.csv dans {DATA}")
    return candidates[-1]


def load_widb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Garder seulement les colonnes utiles
    cols = ["Site_ID", "Site_Name", "Latitude", "Longitude", "Elevation",
            "Sample_ID", "Collection_Date", "Phase", "d2H", "d18O",
            "d2H_Analytical_SD", "d18O_Analytical_SD",
            "WI_Analysis_Source", "Project_ID"]
    df = df[[c for c in cols if c in df.columns]].copy()
    # Filtrer les valeurs valides
    for c in ("d2H", "d18O"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["d2H", "d18O"])
    # Filtrer les valeurs sentinelles 9999 / -9999 (embargo)
    df = df[(df["d2H"].abs() < 1000) & (df["d18O"].abs() < 100)].copy()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Elevation"] = pd.to_numeric(df["Elevation"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df.reset_index(drop=True)


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Résidu de chaque point par rapport à la GMWL."""
    expected_d2H = GMWL_SLOPE * df["d18O"] + GMWL_INTERCEPT
    df["residual_d2H"] = df["d2H"] - expected_d2H
    return df


def find_outliers(df: pd.DataFrame, sigma_threshold: float = 3.0
                  ) -> tuple[pd.DataFrame, float]:
    sigma = df["residual_d2H"].std()
    mu = df["residual_d2H"].mean()
    threshold = sigma_threshold * sigma
    outliers = df[(df["residual_d2H"] - mu).abs() > threshold].copy()
    return outliers, sigma


def plot_gmwl(df: pd.DataFrame, outliers: pd.DataFrame, sigma: float,
              out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1 : scatter δ²H vs δ¹⁸O avec GMWL ---
    ax = axes[0]
    sample = df.sample(min(len(df), 5000), random_state=42)
    ax.scatter(sample["d18O"], sample["d2H"], s=3, alpha=0.3, color="steelblue",
               label=f"wiDB précipitations (n={len(df)}, sample n={len(sample)})")
    ax.scatter(outliers["d18O"], outliers["d2H"], s=20, color="red",
               edgecolor="black", linewidth=0.5,
               label=f"Outliers > 3σ ({len(outliers)})", zorder=5)
    # Tracer la GMWL
    x = np.linspace(df["d18O"].min(), df["d18O"].max(), 100)
    y_gmwl = GMWL_SLOPE * x + GMWL_INTERCEPT
    ax.plot(x, y_gmwl, "k--", linewidth=2,
            label=f"GMWL : δ²H = 8δ¹⁸O + 10 (Craig 1961)")
    ax.plot(x, y_gmwl + 3*sigma, "r:", linewidth=1, alpha=0.5,
            label=f"GMWL ± 3σ (σ={sigma:.1f})")
    ax.plot(x, y_gmwl - 3*sigma, "r:", linewidth=1, alpha=0.5)
    ax.set_xlabel("δ¹⁸O (‰)")
    ax.set_ylabel("δ²H (‰)")
    ax.set_title("Régression GMWL et outliers anomalies isotopiques")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 2 : carte mondiale des outliers ---
    ax = axes[1]
    # Background : tout le dataset, gris
    ax.scatter(df["Longitude"], df["Latitude"], s=2, alpha=0.2, color="gray",
               label=f"wiDB stations (n={df['Site_ID'].nunique()} sites)")
    # Outliers : rouge, taille proportionnelle au résidu
    out_size = (outliers["residual_d2H"] - df["residual_d2H"].mean()).abs() / sigma * 4
    ax.scatter(outliers["Longitude"], outliers["Latitude"], s=out_size,
               color="red", alpha=0.6, edgecolor="black", linewidth=0.3,
               label=f"Outliers (n={len(outliers)})", zorder=5)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Distribution géographique des outliers GMWL")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Figure : {out_path}")


def analyze_elevation_correlation(df: pd.DataFrame, outliers: pd.DataFrame):
    """Teste si les outliers se concentrent à haute altitude
    (flux cosmique majoré → hypothèse super-H₂O cosmogenique)."""
    df_elev = df.dropna(subset=["Elevation"])
    out_elev = outliers.dropna(subset=["Elevation"])
    if len(df_elev) == 0 or len(out_elev) == 0:
        return
    print("\n--- Test corrélation altitude / outlier rate ---")
    bins = [0, 500, 1000, 2000, 3000, 5000]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df_elev["Elevation"] >= lo) & (df_elev["Elevation"] < hi)
        mask_out = (out_elev["Elevation"] >= lo) & (out_elev["Elevation"] < hi)
        n_tot = mask.sum()
        n_out = mask_out.sum()
        if n_tot == 0:
            continue
        frac = n_out / n_tot * 100
        print(f"  {lo:>5d} – {hi:>5d} m : {n_out:>4d} / {n_tot:>5d} "
              f"({frac:>5.2f} % outliers)")


def main():
    print("=" * 78)
    print("Anomaly screening — wiDB precipitation 2010-2020")
    print("=" * 78)

    csv_path = find_csv()
    print(f"Source : {csv_path.name}")
    df = load_widb(csv_path)
    print(f"  Records valides : {len(df)}")
    print(f"  Sites uniques   : {df['Site_ID'].nunique()}")
    print(f"  Pays uniques    : {df['Project_ID'].nunique()} projets")
    print(f"  Plage δ²H       : {df['d2H'].min():.1f} → {df['d2H'].max():.1f} ‰")
    print(f"  Plage δ¹⁸O      : {df['d18O'].min():.1f} → {df['d18O'].max():.1f} ‰")
    print()

    df = compute_residuals(df)
    outliers, sigma = find_outliers(df, sigma_threshold=3.0)
    print(f"GMWL régression (Craig 1961) : δ²H = 8·δ¹⁸O + 10")
    print(f"  σ résiduel global : {sigma:.2f} ‰")
    print(f"  Outliers >3σ      : {len(outliers)} / {len(df)} = "
          f"{len(outliers)/len(df)*100:.2f} %")
    print()

    # Top outliers
    print("--- Top 10 outliers par |résidu| ---")
    top = outliers.reindex(
        outliers["residual_d2H"].abs().sort_values(ascending=False).index
    ).head(10)
    for _, r in top.iterrows():
        date = str(r.get("Collection_Date", "?"))[:10]
        print(f"  {r['Site_ID']:>12s}  ({r['Latitude']:+7.2f}, {r['Longitude']:+8.2f})  "
              f"elev={r['Elevation'] if pd.notna(r['Elevation']) else '?':>5}  "
              f"{date}  d2H={r['d2H']:>6.1f}  d18O={r['d18O']:>5.1f}  "
              f"resid={r['residual_d2H']:+6.1f} ({(r['residual_d2H']/sigma):+.1f}σ)")

    analyze_elevation_correlation(df, outliers)

    # Sauvegardes
    outliers.to_csv(OUT / "widb_outliers_3sigma.csv", index=False)
    print(f"\nCSV outliers : {OUT / 'widb_outliers_3sigma.csv'}")
    plot_gmwl(df, outliers, sigma, OUT / "widb_gmwl_anomaly_map.png")

    print()
    print("=" * 78)
    print("VERDICT méthodologique")
    print("=" * 78)
    print(f"""
  - {len(outliers)} stations identifiées avec δ²H significativement
    hors de la GMWL globale (>3σ, σ_global = {sigma:.1f} ‰).
  - Ces outliers NE PROUVENT PAS la présence de super-H₂O TQNT-V5.
    Ils signalent simplement des régimes hydrologiques particuliers :
    évaporation extrême, mélange de masses d'eau, contamination
    isotopique géologique, ou erreur de mesure.
  - Les outliers à HAUTE ALTITUDE (>2000 m) sont les candidats prioritaires
    pour une analyse FT-ICR (R≥1.5×10⁴) ciblant le m/z=20.131 :
    le flux cosmique y est plus élevé, hypothétiquement plus de production
    super-H si la rare possibilité existe.
  - La limite physique reste : seul un FT-ICR ou Orbitrap peut DISCRIMINER
    H̄H¹⁶O (20.131) de H₂¹⁸O (20.005). Le wiDB indique OÙ chercher,
    pas QUOI trouver.
""")


if __name__ == "__main__":
    main()
