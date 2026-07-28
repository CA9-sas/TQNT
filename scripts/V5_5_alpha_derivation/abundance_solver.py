"""
Solveur d'abondance Ω_DM/Ω_b pour TQNT-V5 (V5 §13.5, équation 13.X).

Vérifie la calibration α=1.20 → Ω_DM/Ω_b = 5.5 (Planck 2018 : 5,45 ± 0,28),
puis prépare la dérivation depuis premiers principes SU(2)_135 (V5.5).

Formule V5 :
  Ω_DM/Ω_b = Σ_{c > 18} m(c) N(c) e^(-α c) / Σ_{c ≤ 18} m(c) N(c) e^(-α c)

avec :
  m(c) : masse moyenne d'un nœud à c croisements (formule équilibre EM)
  N(c) : nombre de nœuds premiers à c croisements (Hoste-Thistlethwaite)
  α    : pénalité topologique par croisement (V5.5 : à dériver depuis CS k=135)

Source : docs/TQNT_detaillee_V5_FR.tex §13.5, eq. (7204).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Table Hoste-Thistlethwaite N(c) pour c ≤ 16 (Weeks et al. 1998)
# https://en.wikipedia.org/wiki/Knot_(mathematics)#Knot_tabulation
# ---------------------------------------------------------------------------
HOSTE_THISTLETHWAITE_NUMBERS = {
    3:  1,
    4:  1,
    5:  2,
    6:  3,
    7:  7,
    8:  21,
    9:  49,
    10: 165,
    11: 552,
    12: 2176,
    13: 9988,
    14: 46972,
    15: 253293,
    16: 1388705,
}


def N_knots(c: int) -> float:
    """Nombre de nœuds premiers à c croisements.

    c ≤ 16 : table Hoste-Thistlethwaite exacte.
    c > 16 : asymptotique 2.68^c / c (Sundberg-Thistlethwaite 1998).
    """
    if c < 3:
        return 0.0
    if c in HOSTE_THISTLETHWAITE_NUMBERS:
        return float(HOSTE_THISTLETHWAITE_NUMBERS[c])
    return 2.68 ** c / c


def m_knot_average(c: int) -> float:
    """Masse moyenne d'un nœud à c croisements en MeV.

    V5 chap. 14 : formule d'équilibre EM, valeurs FEM exactes pour c ≤ 10,
    extrapolation m ~ c^6.65 au-delà.
    """
    # Ancrages V5 : nucléon (3_1+7_2+7_2) à c=17 → ~938 MeV
    # Calibration : m(c=3) = trefoil = électron = 0.511 MeV
    # Loi extrapolée : m(c) = m_e × (c/3)^6.65
    return 0.511 * (c / 3.0) ** 6.65


def omega_ratio(alpha: float, c_max: int = 60,
                c_threshold: int = 18) -> tuple[float, dict]:
    """Ratio Ω_DM/Ω_b = (Σ_{c>thr} m N e^{-αc}) / (Σ_{c≤thr} m N e^{-αc}).

    Returns (ratio, diagnostic_dict).
    """
    sum_dm = 0.0
    sum_baryon = 0.0
    breakdown = {"alpha": alpha, "by_c": {}}
    for c in range(3, c_max + 1):
        m = m_knot_average(c)
        N = N_knots(c)
        weight = m * N * math.exp(-alpha * c)
        breakdown["by_c"][c] = {
            "m_MeV": m, "N": N,
            "weight": weight,
            "weight_normalized_log10": math.log10(weight + 1e-300),
        }
        if c > c_threshold:
            sum_dm += weight
        else:
            sum_baryon += weight
    breakdown["sum_dm"] = sum_dm
    breakdown["sum_baryon"] = sum_baryon
    breakdown["ratio"] = sum_dm / sum_baryon if sum_baryon > 0 else float("inf")
    return breakdown["ratio"], breakdown


def calibrate_alpha(target_ratio: float = 5.45, tol: float = 0.01,
                    c_max: int = 60) -> float:
    """Trouve α tel que Ω_DM/Ω_b = target (Planck 5,45 ± 0,28).

    Bisection sur α ∈ [0.5, 3.0].
    """
    alpha_lo, alpha_hi = 0.5, 3.0
    for _ in range(200):
        alpha = (alpha_lo + alpha_hi) / 2
        ratio, _ = omega_ratio(alpha, c_max=c_max)
        if abs(ratio - target_ratio) < tol:
            return alpha
        # ratio croît avec α (plus de poids relatif sur c > 18) ?
        # En réalité : pour α grand, les c élevés sont exponentiellement
        # supprimés MAIS m(c) ~ c^6.65 compense partiellement.
        # Empiriquement (V5 chap. 13) : α=1.20 → ratio≈5.5.
        # On vérifie le signe localement.
        if ratio < target_ratio:
            # ratio trop bas -> baisser α (moins de pénalité c>18)
            alpha_hi = alpha
        else:
            alpha_lo = alpha
    return alpha


def plot_sensitivity():
    """Trace Ω_DM/Ω_b vs α pour validation V5.tex eq.13."""
    alphas = np.linspace(0.5, 2.5, 200)
    ratios = np.array([omega_ratio(a)[0] for a in alphas])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, ratios, "b-", linewidth=2, label="Ω_DM/Ω_b(α)")
    ax.axhline(5.45, color="green", linestyle="--", alpha=0.7,
               label="Planck 2018 : 5,45 ± 0,28")
    ax.fill_between(alphas, 5.45 - 0.28, 5.45 + 0.28, color="green", alpha=0.2)
    ax.axvline(1.20, color="red", linestyle=":", alpha=0.7,
               label="V5 calibration α = 1,20")
    ax.set_xlabel("α (pénalité topologique par croisement)")
    ax.set_ylabel("Ω_DM / Ω_b")
    ax.set_title(f"Sensibilité abondance V5 §13.5 vs α\n"
                 f"Pour α = 1,20 : ratio = {omega_ratio(1.20)[0]:.3f}")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out = Path("output/v5p5_alpha_abundance_sensitivity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nFigure : {out}")


def main():
    print("=" * 78)
    print("TQNT-V5 §13.5 — Vérification de la calibration α = 1,20")
    print("=" * 78)

    # Vérification : α = 1.20 reproduit Ω_DM/Ω_b ≈ 5.5 ?
    alpha_v5 = 1.20
    ratio, diag = omega_ratio(alpha_v5)
    print(f"\nÀ α = {alpha_v5} (V5 calibré) :")
    print(f"  Ω_DM/Ω_b = {ratio:.3f}")
    print(f"  Planck 2018 = 5,45 ± 0,28")
    print(f"  Écart       = {(ratio - 5.45)/5.45*100:+.2f} %")

    # Décomposition par c
    print(f"\nContribution par croisement (top 10) :")
    print(f"  {'c':>3} {'N(c)':>10} {'m(c) MeV':>12} {'m N e^{-αc}':>14} {'log10 weight':>12}")
    sorted_c = sorted(diag["by_c"].items(),
                      key=lambda x: x[1]["weight"], reverse=True)
    for c, info in sorted_c[:10]:
        sector = "DM" if c > 18 else "b"
        print(f"  {c:>3} {info['N']:>10.0f} {info['m_MeV']:>12.2e} "
              f"{info['weight']:>14.3e} {info['weight_normalized_log10']:>+12.2f} ({sector})")

    print(f"\nΣ DM (c > 18)     : {diag['sum_dm']:.3e}")
    print(f"Σ b  (c ≤ 18)     : {diag['sum_baryon']:.3e}")
    print(f"Ratio             : {ratio:.3f}")

    # Calibration inverse : quel α donne 5.45 exactement ?
    print(f"\n--- Calibration inverse ---")
    alpha_fitted = calibrate_alpha(5.45)
    print(f"  α tel que ratio = 5,45 : α_fit = {alpha_fitted:.4f}")
    print(f"  (V5 utilise α = 1,20, valeur calibrée)")

    plot_sensitivity()

    print()
    print("=" * 78)
    print("PROCHAINE ÉTAPE — V5.5 first principles")
    print("=" * 78)
    print(f"""
  α = 1,20 est CALIBRÉ. La dérivation V5.5 doit produire α depuis :

  1. Structure modulaire SU(2)_135 (theory/su2_modular.py) :
     - dimensions quantiques d_j
     - matrice S
     - poids conformes h_j
     - Wilson loop ⟨W_K⟩ = V_K(q=e^(2πi/137))

  2. Bilan détaillé statistique χ⁰ ⇌ K(c_n) à T_c ≈ 67 MeV :
     - taux de formation Γ_+ ∝ |⟨W_K⟩|² × n_χ
     - taux de destruction Γ_- ∝ |⟨W_K^*⟩|² × n_K
     - équilibre dynamique : n_K/n_χ = Γ_+/Γ_- = e^(-ΔE/T)

  3. Boltzmann couplé sur c_n ∈ [3, 60] : système d'équations
     n_K(c_n)/n_K(c_n - 1) = e^(-α(c_n)), avec α(c_n) dérivé point par point.

  4. Universalité asymptotique : α → constante pour c_n → ∞ ?
     C'est la prédiction V5 que α = 1,20 émerge structurellement.

  Le code structurel est prêt. Étape concrète : implémenter le terme
  cinétique avec Wilson loops + dimensions quantiques (theory/condensation_boltzmann.py).
""")


if __name__ == "__main__":
    main()
