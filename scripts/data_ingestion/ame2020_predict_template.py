"""
ame2020_predict_template.py — Squelette de prédiction TQNT-V5 des énergies
de liaison nucléaire, à confronter à l'AME 2020.

Contexte TQNT-V5
----------------
Le nucléon V5 est l'entrelacs 3₁ + 7₂ + 7₂ (m_p = 938,54 MeV, m_n = 939,83 MeV)
— cf. CLAUDE.md et docs/TQNT_detaillee_V5_FR.tex chap. 6. Pour un NOYAU à
A = N + Z nucléons, la TQNT-V5 prédit une énergie de liaison émergeant de :

  * l'enlacement collectif (linking number Lk) entre les sous-entrelacs
    nucléoniques, contribuant via le potentiel harmonique
        E_tension = β · (Lk - Lk_opt)² · ⟨√(c_J(i)·c_J(j))⟩  (β = 0,210 MeV) ;
  * la métrique conformément émergente (P5) g_ij = (1 + κ|Ψ|²) δ_ij
    qui structure le bassin nucléaire et capture les effets de saturation ;
  * la nucléation aux frontières (P22) — sous-dominante à l'échelle nucléaire
    mais potentiellement pertinente pour les noyaux limites de stabilité.

L'opérateur canonique m(L) = Ω_ts(L) (V5 chap. 14) appliqué au compositum
de A nucléons doit reproduire :
    BE(A,Z) = Z·m_p + N·m_n - m_noyau(A,Z)

avec en particulier le pic ⁵⁶Fe à 8,794 MeV/nucléon (cf. courbe AME 2020
générée par ame2020_load.py).

Ce module fournit le SQUELETTE numérique (loader + χ² + sortie) prêt à
recevoir une implémentation de predict_binding_V5.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ame2020_load import (
    binding_energy_curve,
    load_ame2020,
    predict_mass_V5,
)


def predict_binding_V5(Z: int, N: int) -> float:
    """Prédiction TQNT-V5 de l'énergie de liaison TOTALE BE(Z,N) en keV.

    À implémenter à partir de :
      - décomposition du noyau en A = Z+N sous-entrelacs nucléoniques
        (3₁+7₂+7₂ par nucléon, charge par proton via Lk),
      - calcul du linking number collectif Lk_tot,
      - application de l'opérateur axiomatique Ω_ts sur l'entrelacs total,
      - soustraction de Σ m_nucléons libres pour obtenir BE.

    Tant que Ω_ts n'est pas implémenté, retourne np.nan.
    """
    # NOTE : la masse libre des nucléons (3₁+7₂+7₂) est connue ; ce qui manque
    # est la masse de l'entrelacs collectif des A nucléons. Voir
    # docs/TQNT_detaillee_V5_FR.tex ch. 14 + ch. 6 (tension β).
    _ = predict_mass_V5(Z, N)  # déclenche le même warning, cohérence
    return np.nan


def compute_chi2(
    df: pd.DataFrame,
    prediction_fn=predict_binding_V5,
    use_unc_floor_keV: float = 10.0,
    require_experimental: bool = True,
) -> dict:
    """Compare une prédiction TQNT à l'AME 2020.

    Pour chaque noyau du DataFrame, appelle prediction_fn(Z, N) (BE totale en keV)
    et calcule χ² = Σ ((BE_pred - BE_AME) / σ)² où σ = max(σ_AME, use_unc_floor_keV).

    Paramètres
    ----------
    df : DataFrame issu de load_ame2020(...)
    prediction_fn : callable(Z:int, N:int) -> float  (BE totale en keV)
    use_unc_floor_keV : plancher d'incertitude pour éviter σ→0 (déf. 10 keV)
    require_experimental : si True, exclut les noyaux extrapolés (déf. True)

    Retour
    ------
    dict :
        n_nuclei            : nombre de noyaux comparés
        n_valid_predictions : nombre de prédictions non-NaN
        chi2                : χ² total
        chi2_per_dof        : χ²/n_valid_predictions
        rms_residual_keV    : RMS des résidus (BE_pred − BE_AME)
        max_residual_keV    : pire écart (signé)
        max_residual_nuclide: tuple (A, Z, element) du pire écart
    """
    sub = df.dropna(subset=["Z", "N", "binding_E_total_keV"]).copy()
    if require_experimental and "extrapolated" in sub.columns:
        sub = sub.loc[~sub["extrapolated"]]

    preds = np.array(
        [prediction_fn(int(z), int(n)) for z, n in zip(sub["Z"], sub["N"])],
        dtype=float,
    )
    sub["BE_pred_keV"] = preds
    sub["residual_keV"] = sub["BE_pred_keV"] - sub["binding_E_total_keV"]

    sigma = sub["binding_E_per_A_unc_keV"] * sub["A"]
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, use_unc_floor_keV)
    sigma = np.maximum(sigma, use_unc_floor_keV)

    mask = np.isfinite(sub["residual_keV"].to_numpy())
    n_valid = int(mask.sum())

    if n_valid == 0:
        return {
            "n_nuclei": len(sub),
            "n_valid_predictions": 0,
            "chi2": np.nan,
            "chi2_per_dof": np.nan,
            "rms_residual_keV": np.nan,
            "max_residual_keV": np.nan,
            "max_residual_nuclide": None,
            "message": "Aucune prédiction valide — implémenter predict_binding_V5.",
        }

    r = sub.loc[mask, "residual_keV"].to_numpy()
    s = sigma[mask]
    chi2 = float(np.sum((r / s) ** 2))
    rms = float(np.sqrt(np.mean(r * r)))
    i_worst = int(np.argmax(np.abs(r)))
    worst_row = sub.loc[mask].iloc[i_worst]

    return {
        "n_nuclei": len(sub),
        "n_valid_predictions": n_valid,
        "chi2": chi2,
        "chi2_per_dof": chi2 / n_valid,
        "rms_residual_keV": rms,
        "max_residual_keV": float(r[i_worst]),
        "max_residual_nuclide": (
            int(worst_row["A"]),
            int(worst_row["Z"]),
            str(worst_row["element"]),
        ),
    }


def _main() -> None:
    print("=" * 64)
    print("Template prédiction TQNT-V5 vs AME 2020")
    print("=" * 64)
    df = load_ame2020(keep_extrapolated=False)
    A_arr, BE_per_A = binding_energy_curve(df)
    print(f"AME 2020 chargé : {len(df)} noyaux expérimentaux.")
    print(f"Courbe BE/A vs A : {len(A_arr)} points ; pic à A={A_arr[np.argmax(BE_per_A)]}"
          f" ({BE_per_A.max()/1000:.3f} MeV/A).")

    # On capture le warning émis par le stub pour ne pas spammer la sortie
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = compute_chi2(df, predict_binding_V5)

    print("-" * 64)
    print("Statistiques de comparaison (stub) :")
    for k, v in stats.items():
        print(f"  {k:24s} : {v}")
    print("-" * 64)
    print("predict_binding_V5(Z, N) n'est PAS encore implémenté.")
    print("Pour activer la comparaison TQNT-V5 ↔ AME 2020 :")
    print("  1. Implémenter Ω_ts dans theory/ (cf. TQNT_detaillee_V5_FR.tex ch. 14)")
    print("  2. Décomposer chaque noyau (A,Z) en sous-entrelacs nucléoniques")
    print("     (3₁ + 7₂ + 7₂ par nucléon, Q via Lk = +3 proton / +2 neutron)")
    print("  3. Remplacer le stub predict_binding_V5 ci-dessus par")
    print("     BE = Z*m_p + N*m_n - Ω_ts(entrelacs collectif)")
    print("  4. Re-exécuter ce script ; compute_chi2 renverra χ²/dof.")


if __name__ == "__main__":
    _main()
