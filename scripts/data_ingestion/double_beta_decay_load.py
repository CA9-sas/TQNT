#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
double_beta_decay_load.py — Comparaison TQNT-V5 vs limites 0νββ publiées
============================================================================

PRÉDICTION TQNT-V5 (CRUCIALE) :

En TQNT-V5, les neutrinos sont des FRAGMENTS D'ONDE LIBRE auto-rotatifs
longitudinaux du champ χ, PAS des nœuds topologiques. Ce sont des fermions
de Dirac (non Majorana). Conséquence empirique directe et NON négociable :

        τ(0νββ) = ∞      EXACTEMENT
        m_ββ    = 0      EXACTEMENT

Le mécanisme 0νββ requiert un terme de masse de Majorana qui n'existe pas
dans V5. Le taux est structurellement interdit, pas seulement supprimé.

CONSÉQUENCES POUR L'EXPÉRIMENT :

1. TOUTE DÉTECTION POSITIVE de 0νββ à T_{1/2} fini falsifierait V5.
   C'est un test « one-shot » : un seul événement non-bruit suffirait.

2. Les limites actuelles (KamLAND-Zen 3.8×10²⁶ ans) sont COMPATIBLES avec
   V5 puisqu'elles sont des bornes INFÉRIEURES sur T_{1/2} et que V5
   prédit T_{1/2} = ∞. Aucune limite, aussi serrée soit-elle, ne pourra
   jamais EXCLURE V5 — seule une détection le pourra.

3. Les limites actuelles ne distinguent PAS V5 du Modèle Standard avec
   neutrinos Majorana à m_ββ inférieur à la sensibilité de l'expérience.
   Il faut une DÉTECTION (pas une limite) pour discriminer les deux
   hypothèses (Dirac/V5 vs Majorana/SM).

Voir : docs/TQNT_detaillee_V5_FR.tex, chap. neutrinos.
       memory/neutrinos-are-free-chi-waves.md
============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interactif obligatoire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
DATA_CSV = Path(
    "/mnt/data/data_recherche/double_beta_decay/kamland_zen_2024.csv"
)
OUTPUT_PNG = Path(
    "/home/euk9/projets/physique_nouvelle/output/double_beta_decay_limits_v5.png"
)


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_data(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Charge le CSV des limites 0νββ publiées en DataFrame pandas."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}. Vérifier l'ingestion préalable."
        )
    df = pd.read_csv(csv_path)
    df["half_life_lower_limit_years"] = df["half_life_lower_limit_years"].astype(float)
    return df


# ----------------------------------------------------------------------------
# Prédiction V5
# ----------------------------------------------------------------------------
def tqnt_v5_prediction() -> dict:
    """
    Prédiction TQNT-V5 pour le taux de double désintégration bêta sans neutrino.

    Returns
    -------
    dict
        Dictionnaire contenant les valeurs prédites par V5 et la justification
        physique (neutrinos = fragments d'onde libre Dirac, pas Majorana).
    """
    return {
        "tau_0nbb_years": np.inf,
        "m_beta_beta_meV": 0.0,
        "reason": (
            "V5 neutrinos are Dirac, free chi-wave fragments (no topological "
            "knot), so no Majorana mass term exists. The 0nubb amplitude is "
            "structurally zero, not merely suppressed."
        ),
        "falsifiable_by": (
            "Any positive 0nubb detection at finite T_{1/2} would falsify V5."
        ),
    }


# ----------------------------------------------------------------------------
# Comparaison
# ----------------------------------------------------------------------------
def compare_v5_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare la prédiction V5 (τ = ∞) à chaque limite expérimentale.

    Pour chaque ligne, calcule :
      - v5_compatible : True si la limite est cohérente avec τ = ∞.
        (Toujours True pour une borne inférieure : ∞ > toute borne finie.)
      - falsification_threshold : la limite expérimentale au-delà de laquelle
        une détection serait nécessaire pour falsifier V5. Ici on renvoie la
        limite inférieure courante de T_{1/2} comme « seuil minimal qu'il
        faudrait dépasser via une détection positive » — V5 reste compatible
        tant qu'aucune désintégration n'est observée.
      - sensitivity_decades_to_falsify : nombre de décades de T_{1/2}
        supplémentaires que l'expérience peut encore sonder sans falsifier V5.
        On prend comme borne supérieure pragmatique 10⁴⁰ ans (largement au-delà
        de toute sensibilité réaliste — V5 reste compatible jusque-là et
        au-delà, ad infinitum).

    Notes
    -----
    Une limite inférieure sur T_{1/2} ne pourra JAMAIS exclure V5 :
    seule une DÉTECTION positive le pourrait. La colonne « decades to
    falsify » illustre que toute la fenêtre expérimentale future reste
    compatible avec V5.
    """
    v5 = tqnt_v5_prediction()
    out = df.copy()

    # Toute limite inférieure sur T_{1/2} est compatible avec τ_V5 = ∞.
    out["v5_compatible"] = out["half_life_lower_limit_years"].apply(
        lambda t: v5["tau_0nbb_years"] > t  # ∞ > t pour tout t fini : True
    )

    # Le « seuil de falsification » n'est pas une valeur de T_{1/2} : c'est
    # le fait d'observer un événement de signal au-dessus de la limite
    # actuelle. On rapporte la limite actuelle comme « plancher » à dépasser
    # par une détection.
    out["falsification_threshold"] = out["half_life_lower_limit_years"]

    # Décades restant avant que la sensibilité atteigne 10^40 ans (en deçà,
    # V5 reste structurellement compatible ; au-delà, V5 reste compatible
    # ad infinitum tant qu'il n'y a pas de détection).
    horizon_years = 1e40
    out["sensitivity_decades_to_falsify"] = np.log10(
        horizon_years / out["half_life_lower_limit_years"]
    )

    return out


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_limits(df: pd.DataFrame, output_path: Path = OUTPUT_PNG) -> Path:
    """Bar chart horizontal des limites T_{1/2} avec annotation V5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("half_life_lower_limit_years").reset_index(drop=True)
    labels = [
        f"{row.experiment}\n({row.isotope}, {int(row.year)})"
        for row in df_sorted.itertuples()
    ]
    half_lives = df_sorted["half_life_lower_limit_years"].values

    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.barh(
        labels,
        half_lives,
        color=colors[: len(half_lives)],
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$T_{1/2}(0\nu\beta\beta)$ lower limit  [years, 90% CL]", fontsize=11)
    ax.set_title(
        r"Limites publiées sur $0\nu\beta\beta$ vs prédiction TQNT-V5",
        fontsize=12,
    )

    # Annoter chaque barre avec la valeur
    for bar, t12 in zip(bars, half_lives):
        ax.text(
            t12 * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{t12:.1e} yr",
            va="center",
            fontsize=9,
        )

    # Annotation V5 — flèche pointant vers la droite (vers l'infini)
    xmin, xmax = 1e25, 1e30
    ax.set_xlim(xmin, xmax)
    ax.axvline(
        xmax * 0.95,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.annotate(
        r"V5 : $\tau(0\nu\beta\beta) = \infty$" + "\n(Dirac, free $\\chi$-waves)",
        xy=(xmax * 0.95, len(labels) - 0.5),
        xytext=(xmax * 0.18, len(labels) - 0.3),
        fontsize=11,
        color="red",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="red"),
    )

    ax.grid(True, which="both", axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    df = load_data()
    v5 = tqnt_v5_prediction()
    cmp = compare_v5_to_data(df)

    print("=" * 78)
    print(" TQNT-V5 vs Neutrinoless Double Beta Decay (0nubb) experimental limits")
    print("=" * 78)
    print()
    print("V5 prediction :")
    print(f"  tau_0nbb        = {v5['tau_0nbb_years']}  years")
    print(f"  m_beta_beta     = {v5['m_beta_beta_meV']}  meV")
    print(f"  reason          : {v5['reason']}")
    print(f"  falsifiable by  : {v5['falsifiable_by']}")
    print()

    cols = [
        "experiment",
        "isotope",
        "half_life_lower_limit_years",
        "m_beta_beta_lower_meV",
        "m_beta_beta_upper_meV",
        "CL",
        "year",
        "v5_compatible",
        "sensitivity_decades_to_falsify",
    ]
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", lambda x: f"{x:.3g}",
    ):
        print(cmp[cols].to_string(index=False))
    print()

    n_compat = int(cmp["v5_compatible"].sum())
    n_total = len(cmp)
    print(
        f"  -> {n_compat}/{n_total} experiments compatible with V5 "
        f"(all current lower limits are bounded by tau = infinity)."
    )
    print(
        "  -> V5 falsification requires a POSITIVE detection of 0nubb, "
        "not a tighter lower limit."
    )
    print()

    png_path = plot_limits(cmp)
    print(f"Plot saved to : {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
