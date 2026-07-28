r"""
REDÉFINITION DE LA MASSE : tube de flux χ EN ROTATION (trajectoires de Regge).

Contexte : la formule géométrique Ω_ts (entrée U_κ/U_τ) s'est révélée
phénoménologique (§Re-mesure de brouillon_Ukappa_Utau_geometrie.tex : la vraie
géométrie casse le spectre à 75 % RMS). Il faut redéfinir la masse.

Idée physique (utilisateur) : la masse intègre l'inertie des boucles, le spin,
la charge et le MOMENT CINÉTIQUE — « plus le moment est important, plus le champ
χ prend du poids ». C'est exactement un tube de flux χ EN ROTATION : le tube
tournant avec moment cinétique J stocke de l'énergie χ, d'où

        M²(config, J) = M₀²(inertie des boucles / topologie)      [tête de bande]
                      + 2π σ_χ · J                                 [rotation χ]
                      + ΔE_EM(Q, Lk)                               [charge, petit]

Le terme rotationnel EST une trajectoire de Regge de pente 1/α'_χ = 2π σ_χ, où
σ_χ ≈ 0,18 GeV² est la tension du tube χ (déjà dans la TQNT via Cornell V5.37b).
La pente est donc PRÉDITE, non ajustée.

Résultat : sur 5 trajectoires réelles, M² est linéaire en J (r² ≥ 0,997) de pente
moyenne 1,096 GeV²/J, contre 2π·0,18 = 1,131 prédit → écart 3,1 %.

Statut honnête :
  - la DÉPENDANCE EN J (bande rotationnelle) est bien définie et prédite (3 %) ;
  - la TÊTE DE BANDE M₀ (état J-minimal) reste l'inertie topologique — la partie
    dure, non résolue (comme l'ancien intercept) ;
  - σ_χ = 0,18 GeV² est UNE constante universelle calibrée (Cornell), pas par
    particule — donc pas de retour à la calibration per-nœud.
"""
from __future__ import annotations
import numpy as np

SIGMA_CHI = 0.18  # GeV² — tension du tube de flux χ (Cornell V5.37b)


def regge_slope_pred(sigma=SIGMA_CHI):
    """Pente de Regge prédite 1/α'_χ = 2π σ_χ (GeV²/J) et α'_χ (GeV⁻²)."""
    return 2 * np.pi * sigma, 1.0 / (2 * np.pi * sigma)


# trajectoires de Regge réelles : (nom, M en GeV, J)
TRAJECTORIES = {
    "ρ/a (I=1)":  [("ρ", 0.7753, 1), ("a₂", 1.3183, 2), ("ρ₃", 1.689, 3), ("a₄", 1.967, 4)],
    "ω/f (I=0)":  [("ω", 0.7827, 1), ("f₂", 1.2755, 2), ("ω₃", 1.667, 3), ("f₄", 2.018, 4)],
    "N (nucléon)": [("N", 0.9396, 0.5), ("N1680", 1.685, 2.5), ("N2220", 2.25, 4.5)],
    "Δ (décuplet)": [("Δ", 1.232, 1.5), ("Δ1950", 1.930, 3.5), ("Δ2420", 2.40, 5.5)],
    "K* (étrange)": [("K*", 0.8917, 1), ("K*2", 1.4273, 2), ("K*3", 1.776, 3), ("K*4", 2.045, 4)],
}


def fit_trajectory(pts):
    """Ajuste M² = slope·J + inter. Retourne (slope, M0, r2)."""
    J = np.array([p[2] for p in pts])
    M2 = np.array([p[1] ** 2 for p in pts])
    A = np.vstack([J, np.ones_like(J)]).T
    (slope, inter), *_ = np.linalg.lstsq(A, M2, rcond=None)
    r2 = 1 - np.sum((M2 - (slope * J + inter)) ** 2) / np.sum((M2 - M2.mean()) ** 2)
    return slope, np.sqrt(max(inter, 0.0)), r2


LAMBDA_CHI = 0.216  # GeV — échelle topologique χ (Witten-Veneziano, masse η')

ROPELENGTH = {  # ropelength idéale (mass_ropelength_algebra.py)
    "3_1": 16.372, "4_1": 20.99, "5_1": 23.55, "5_2": 26.13, "6_1": 28.31,
    "6_2": 28.13, "6_3": 27.21, "7_1": 30.69, "7_2": 32.96, "7_4": 34.97,
    "7_5": 35.16, "7_6": 32.66, "8_1": 37.49, "9_1": 38.62, "10_1": 42.33,
}


def sigma_from_lambda_chi():
    """σ_χ se ramène-t-il à Λ_χ ? Renvoie (√σ, 2Λ_χ, écart %)."""
    import numpy as np
    sqrt_s = np.sqrt(SIGMA_CHI)
    return sqrt_s, 2 * LAMBDA_CHI, abs(sqrt_s - 2 * LAMBDA_CHI) / (2 * LAMBDA_CHI) * 100


def test_cgq_selfenergy():
    """M₀ = auto-énergie du puits χ GRAVITATIONNEL (CGQ) ? Renvoie le ratio
    U_CGQ / m_p c². Attendu ~10⁻³⁹ (39 décades trop faible)."""
    G, c = 6.674e-11, 2.998e8
    mp, rp = 1.6726e-27, 0.841e-15
    U_cgq = G * mp ** 2 / rp          # ~ G m²/r
    return U_cgq / (mp * c ** 2)


def test_bandhead_ropelength():
    """M₀ ∝ ropelength ? Test à J=1 (vecteurs). Renvoie la table + le contre-ex."""
    vectors = {
        "ρ⁰": (0.7753, ["6_2", "6_2"]), "φ": (1.0195, ["7_5", "7_5"]),
        "J/ψ": (3.0969, ["7_4", "7_4"]), "Υ": (9.460, ["7_6", "7_6", "4_1"]),
    }
    rows = [(nm, m, sum(ROPELENGTH[k] for k in ks)) for nm, (m, ks) in vectors.items()]
    return rows


def main():
    pente, alpha = regge_slope_pred()
    print(f"Prédiction χ-tube : pente 1/α'_χ = 2π σ_χ = {pente:.3f} GeV²/J "
          f"(α'_χ = {alpha:.3f} GeV⁻²)\n")
    print(f"{'trajectoire':16s} {'pente (GeV²/J)':16s} {'M₀ (GeV)':10s} {'r²':8s}")
    slopes = []
    for name, pts in TRAJECTORIES.items():
        s, M0, r2 = fit_trajectory(pts)
        slopes.append(s)
        print(f"{name:16s} {s:14.3f}   {M0:8.3f}   {r2:.4f}")
    mean = float(np.mean(slopes))
    print(f"\nPente moyenne mesurée : {mean:.3f} GeV²/J")
    print(f"Écart à la prédiction : {abs(mean - pente) / pente * 100:.1f} %")
    print("\nLa dépendance en J (moment cinétique) est bien définie et PRÉDITE ;")
    print("la tête de bande M₀ reste l'inertie topologique (partie dure, non résolue).")

    print("\n--- Tâche 1 : σ_χ depuis Λ_χ ---")
    sqrt_s, two_lam, ecart = sigma_from_lambda_chi()
    print(f"  √σ_χ = {sqrt_s*1000:.1f} MeV ≈ 2Λ_χ = {two_lam*1000:.0f} MeV "
          f"(écart {ecart:.1f} %) → σ_χ ≈ 4Λ_χ². Facteur 4 heuristique.")

    print("\n--- Tâche 2 : M₀ ∝ ropelength ? (vecteurs J=1) ---")
    for nm, m, r in test_bandhead_ropelength():
        print(f"  {nm:4s} Σrope={r:6.1f}  masse={m*1000:7.1f} MeV")
    print("  φ (70.3) ≈ J/ψ (69.9) en ropelength mais masses ×3 → M₀ N'EST PAS")
    print("  géométrique : l'échelle saveur vit dans l'assignation, pas la ropelength.")

    print("\n--- M₀ = auto-énergie du puits χ gravitationnel (CGQ) ? ---")
    ratio = test_cgq_selfenergy()
    print(f"  U_CGQ / m_p c² = {ratio:.1e} ({np.log10(ratio):.0f} décades) → 39 décades")
    print("  trop faible. CGQ (gravitationnel) RÉFUTÉ pour M₀. M₀ = confinement χ FORT")
    print("  (spectral, Jones-dépendant), pas le puits gravitationnel faible.")


if __name__ == "__main__":
    main()
