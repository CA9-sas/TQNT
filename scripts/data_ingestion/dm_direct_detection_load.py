#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DM direct-detection: loader & TQNT-V5 comparison plot.

Charge les courbes d'exclusion 90 % CL spin-indépendante des trois grandes
expériences xénon liquide de la dernière génération :

  * LZ (LUX-ZEPLIN), 4.2 t·yr, arXiv:2410.17036  -> PRL 135, 011802 (2025)
  * XENONnT,         1.1 t·yr, arXiv:2303.14729  -> PRL 131, 041002 (2023)
  * PandaX-4T,       1.54 t·yr,arXiv:2408.00664  -> PRL 134, 011805 (2025)

Les fichiers sources sont entreposés dans
    /mnt/data/data_recherche/dm_direct_detection/<expt>/curve.csv
La provenance (HEPData officiel / ancillaire arXiv / digitalisation) est
documentée dans le README.md du même dossier.

Le script génère ensuite le spectre TQNT-V5 :

        σ_DM-N(c) ≈ σ_EM · α_EM^(c - 18)        (V5 §13.6.4)

avec α_EM = 1/137, c le nombre de croisements du noeud de matière noire, et
σ_EM calibré pour que σ(c=19) ≈ 1 × 10⁻²⁷ cm² (point d'entrée TQNT-V5).

Le spectre est tracé sur la même figure (échelle log–log) que les trois
courbes expérimentales. Le résumé en sortie identifie :
  * le seuil c_min pour lequel σ_TQNT(c) passe sous LZ,
  * la fenêtre c ≥ 29 (prédiction TQNT-V5 du papier),
  * la sous-classe Q_phys = 0 (parité miroir bipartite), σ ≈ 10⁻⁵⁵ cm².

Backend matplotlib forcé en Agg ; la figure est sauvegardée dans
    /home/euk9/projets/physique_nouvelle/output/dm_direct_detection_comparison.png

Pas de venv : utilise /usr/bin/python3 + numpy + pandas + matplotlib (Agg).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless obligatoire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/data/data_recherche/dm_direct_detection")
OUTPUT_PNG = Path(
    "/home/euk9/projets/physique_nouvelle/output/dm_direct_detection_comparison.png"
)

DETECTORS = {
    "LZ_2024":        {"label": "LZ 4.2 t·yr (2024, arXiv:2410.17036)",        "color": "#1f77b4"},
    "XENONnT_2023":   {"label": "XENONnT 1.1 t·yr (2023, arXiv:2303.14729)",   "color": "#d62728"},
    "PandaX_4T_2024": {"label": "PandaX-4T 1.54 t·yr (2024, arXiv:2408.00664)", "color": "#2ca02c"},
}

# ----------------------------------------------------------------------------
# Constantes TQNT-V5
# ----------------------------------------------------------------------------
ALPHA_EM = 1.0 / 137.0          # constante de structure fine
SIGMA_EM_AT_C19 = 1.0e-27       # ancrage TQNT-V5 (CLAUDE.md, V5 §13.6.4)
# σ(c=19) = σ_EM · α_EM^(19-18) = σ_EM · α_EM   →  σ_EM = σ(c=19) / α_EM
SIGMA_EM = SIGMA_EM_AT_C19 / ALPHA_EM

C_RANGE = np.arange(19, 33)      # c ∈ [19, 32]
C_THRESHOLD_INVISIBLE = 29       # seuil "TQNT prédit σ < LZ" (CLAUDE.md)
SIGMA_QPHYS0 = 1.0e-55           # canal sous-dominant pour la classe Q_phys = 0


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def _load_hepdata_multisection(path: Path) -> pd.DataFrame:
    """Parse un CSV HEPData multi-section et renvoie la première section
    (par convention HEPData : `mass,limit` = 90% CL observed upper limit)."""
    rows: list[tuple[float, float]] = []
    in_target_section = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("mass,"):
                # Une nouvelle sous-table commence. On garde la première
                # (qui est `mass,limit` = observed limit) et on stoppe à la
                # deuxième.
                if not in_target_section and line.lower() == "mass,limit":
                    in_target_section = True
                    continue
                if in_target_section:
                    break  # fin de la sous-table observed
                continue
            if in_target_section:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    return pd.DataFrame(rows, columns=["mass_GeV", "sigma_SI_cm2"])


def load_curve(detector: str) -> pd.DataFrame:
    """Charge la courbe d'exclusion (mass_GeV, sigma_SI_cm2) du détecteur.

    Préférence (du plus autoritaire au moins autoritaire) :
      1. `curve_hepdata.csv`     — HEPData multi-section (LZ 2024)
      2. `curve_pandax_digit.csv` — extraction vectorielle PDF (PandaX-4T 2024)
      3. `curve.csv`             — digitalisation ou ancillaire arXiv (fallback)
    """
    folder = DATA_ROOT / detector

    hepdata_path = folder / "curve_hepdata.csv"
    if hepdata_path.exists():
        return _load_hepdata_multisection(hepdata_path).reset_index(drop=True)

    for candidate in ("curve_pandax_digit.csv", "curve.csv"):
        path = folder / candidate
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            comment="#",
            sep=None,
            engine="python",  # auto-détection du séparateur (',' ou tab)
            header=None,
            usecols=[0, 1],
            names=["mass_GeV", "sigma_SI_cm2"],
            skiprows=0,
        )
        # Si la première ligne valide était un header texte ("mass_GeV"), elle
        # a été parsée mais sera non-numérique : on la retire.
        df = df[pd.to_numeric(df["mass_GeV"], errors="coerce").notna()].copy()
        df["mass_GeV"] = df["mass_GeV"].astype(float)
        df["sigma_SI_cm2"] = df["sigma_SI_cm2"].astype(float)
        return df.reset_index(drop=True)

    raise FileNotFoundError(f"Aucune courbe trouvée pour {detector} dans {folder}")


def load_all() -> dict[str, pd.DataFrame]:
    return {key: load_curve(key) for key in DETECTORS}


# ----------------------------------------------------------------------------
# Spectre TQNT-V5
# ----------------------------------------------------------------------------
def tqnt_v5_spectrum(c_values: np.ndarray = C_RANGE) -> pd.DataFrame:
    """Renvoie le spectre σ_DM-N(c) = σ_EM · α_EM^(c-18) pour c entier."""
    sigma = SIGMA_EM * (ALPHA_EM ** (c_values - 18))
    return pd.DataFrame({"crossings_c": c_values, "sigma_cm2": sigma})


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def make_comparison_figure(
    detectors: dict[str, pd.DataFrame],
    tqnt: pd.DataFrame,
    out_path: Path = OUTPUT_PNG,
) -> Path:
    """Trace LZ / XENONnT / PandaX + spectre TQNT-V5 sur un même axe log-log.

    On choisit un axe horizontal commun (masse en GeV) en représentant les
    candidats TQNT-V5 à la masse nucléon m_N ≈ 0.94 GeV — ce sont des points
    discrets puisque l'observable TQNT est la *section efficace* à c donné,
    indépendante de la masse cinématique de la DM. On les empile donc en
    colonne sur la droite du graphique pour la lisibilité.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for key, meta in DETECTORS.items():
        df = detectors[key]
        ax.plot(
            df["mass_GeV"],
            df["sigma_SI_cm2"],
            label=meta["label"],
            color=meta["color"],
            lw=2.0,
        )

    # Plage de masse couverte par les expériences (axe x)
    mass_min = min(df["mass_GeV"].min() for df in detectors.values())
    mass_max = max(df["mass_GeV"].max() for df in detectors.values())
    ax.set_xlim(mass_min * 0.8, mass_max * 1.2)

    # ------------------------------------------------------------------
    # Spectre TQNT-V5 : points discrets (c entier) tracés à la même masse
    # ------------------------------------------------------------------
    # On choisit m = 100 GeV comme masse de référence (échelle nucléon × 100
    # est un compromis lisible — la TQNT prédit la *section efficace*, pas la
    # masse cinématique). On peut aussi étaler horizontalement avec un léger
    # offset par c pour la lisibilité.
    mass_anchor = 100.0  # GeV
    x_tqnt = mass_anchor * (1.0 + 0.05 * (tqnt["crossings_c"] - 25))  # léger spread

    ax.scatter(
        x_tqnt,
        tqnt["sigma_cm2"],
        marker="D",
        s=55,
        edgecolor="black",
        facecolor="#ff7f0e",
        zorder=5,
        label="TQNT-V5 spectrum  σ(c) = σ_EM · α_EM^(c-18)",
    )
    for c, x, s in zip(tqnt["crossings_c"], x_tqnt, tqnt["sigma_cm2"]):
        ax.annotate(
            f"c={c}",
            xy=(x, s),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="#cc5500",
            va="center",
        )

    # Floor Q_phys = 0 (sous-dominant, isovector miroir)
    ax.axhline(
        SIGMA_QPHYS0,
        color="purple",
        ls=":",
        lw=1.6,
        label=f"TQNT-V5 Q_phys=0 floor σ ≈ {SIGMA_QPHYS0:.0e} cm²",
    )

    # Plage prédite par TQNT-V5 comme "invisible LZ"  (c >= 29)
    sigma_c29 = SIGMA_EM * (ALPHA_EM ** (C_THRESHOLD_INVISIBLE - 18))
    ax.axhline(
        sigma_c29,
        color="orange",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"σ(c={C_THRESHOLD_INVISIBLE}) ≈ {sigma_c29:.1e} cm² (TQNT seuil invisible)",
    )

    ax.set_xlabel(r"WIMP / DM mass [GeV/$c^2$]")
    ax.set_ylabel(r"$\sigma_{\rm SI}^{\rm DM-N}$  [cm$^2$]   (90 % CL)")
    ax.set_title(
        "Direct detection 90 % CL exclusions  vs.  TQNT-V5 σ_DM-N(c) spectrum"
    )
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # On étend l'axe y pour englober le sol TQNT-V5 Q_phys=0
    ax.set_ylim(5e-57, 5e-43)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Résumé textuel
# ----------------------------------------------------------------------------
def summarize(detectors: dict[str, pd.DataFrame], tqnt: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== DM direct detection: data summary ===")
    for key in DETECTORS:
        df = detectors[key]
        i_min = df["sigma_SI_cm2"].idxmin()
        lines.append(
            f"  {key:16s}  n={len(df):3d} points  min σ = "
            f"{df.loc[i_min, 'sigma_SI_cm2']:.3e} cm²  @ m = {df.loc[i_min, 'mass_GeV']:.1f} GeV"
        )

    # Sensibilité LZ minimum, utilisée pour évaluer le seuil c d'invisibilité
    lz_min = detectors["LZ_2024"]["sigma_SI_cm2"].min()
    # σ(c) = σ_EM · α^(c-18) ;  on trouve c_min t.q. σ(c) < LZ_min
    # σ(c)<lz_min  ⇔  α^(c-18) < lz_min/σ_EM  ⇔  c > 18 + log(lz_min/σ_EM)/log(α)
    import math

    c_threshold_real = 18 + math.log(lz_min / SIGMA_EM) / math.log(ALPHA_EM)
    c_min_int = int(math.ceil(c_threshold_real))

    lines.append("")
    lines.append("=== TQNT-V5 prediction (σ_DM-N(c) = σ_EM · α_EM^(c-18)) ===")
    lines.append(f"  σ_EM (calibrated to σ(c=19)={SIGMA_EM_AT_C19:.0e} cm²) = {SIGMA_EM:.3e} cm²")
    for _, row in tqnt.iterrows():
        c = int(row["crossings_c"])
        s = row["sigma_cm2"]
        lines.append(f"    c={c:2d}  ->  σ = {s:.3e} cm²")
    lines.append("")
    lines.append(
        f"  LZ minimum exclusion: σ_LZ^min = {lz_min:.3e} cm²"
    )
    lines.append(
        f"  TQNT-V5 σ(c) < σ_LZ^min  for  c >= {c_min_int}  (continuous threshold c = {c_threshold_real:.2f})"
    )
    lines.append(
        "  TQNT-V5 prédit σ < LZ pour c >= 29 ; classe Q_phys=0 invisible σ ~ 10^-55 cm²"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Entrée
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _ = argv  # placeholder
    detectors = load_all()
    tqnt = tqnt_v5_spectrum()
    out_path = make_comparison_figure(detectors, tqnt)
    print(summarize(detectors, tqnt))
    print()
    print(f"Figure written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
