r"""
Argument d'échelle pour le principe σ = D²Λ² (tension du tube χ).

Le tube de flux χ vit dans l'ordre d'Ising (gap Λ_χ, longueur de corrélation
ξ=1/Λ_χ). Sa tension = densité d'énergie × section du cœur :
    σ = ρ_E · A_core.
  - A_core ~ ξ² = 1/Λ_χ²   (le cœur a la taille de la longueur de corrélation).
  - ρ_E ~ D²·Λ_χ⁴          (D² = dimension quantique TOTALE = 'taille' de
    l'espace de Hilbert anyonique réorganisé quand le tube perce le condensat).
  => σ ~ (O(1)) · D² · Λ_χ².

STATUT :
  - EXACT et structurel : le facteur D² = 4 (dimension totale d'Ising SU(2)_2).
  - DÉRIVÉ (échelle) : σ ∝ D²Λ² (analyse dimensionnelle + comptage des canaux).
  - NON dérivé : le coefficient O(1) du cœur (géométrie exacte du tube). Les
    données préfèrent c≈1 (cœur ≈ 1/Λ², 4 %) ; c=π donnerait 226 %. C'est le
    problème GÉNÉRIQUE de la tension de corde (coefficient sans dimension non
    dérivable de 1er principe, même en QCD).

PRÉDICTION (extrapolation) : σ = D²Λ² fixe une tension SPÉCIFIQUE par ordre
topologique confinant (Ising 4, Fibonacci 2,62, toric code Z_2 4, SU(2)_3 5,24).
Dans la TQNT, seul le secteur Ising confine → une seule mesure, non testable
ailleurs.
"""
from __future__ import annotations
import numpy as np

LAMBDA_CHI = 0.216  # GeV
SIGMA_OBS = 0.18    # GeV²
D2_ISING = 4        # dimension quantique totale au carré (1 + 2 + 1)


def sigma_scaling(core_factor=1.0, D2=D2_ISING, Lam=LAMBDA_CHI):
    """σ = core_factor · D² · Λ²."""
    return core_factor * D2 * Lam ** 2


def main():
    print("σ = ρ_E · A_core ;  A_core ~ 1/Λ² ;  ρ_E ~ D²·Λ⁴  =>  σ ~ (O(1))·D²·Λ²\n")
    print("Dimension : [ρ_E]=E⁴, [A]=E⁻², [σ]=E² ✓\n")
    print("Sensibilité au facteur de cœur O(1) :")
    for c, lab in [(1.0, "1  (cœur = 1/Λ²)"), (np.pi, "π  (cœur = πξ²)"),
                   (1 / np.pi, "1/π")]:
        s = sigma_scaling(c)
        print(f"  c={lab:16s} σ={s:.4f} GeV² (obs {SIGMA_OBS}, écart {abs(s-SIGMA_OBS)/SIGMA_OBS*100:.0f} %)")
    print(f"\n  Les données préfèrent c≈1 → σ ≈ D²Λ² = {sigma_scaling():.4f} (4 %).")

    print("\nStatut : D²=4 EXACT ; σ∝D²Λ² dérivé (échelle) ; coeff O(1) NON dérivé")
    print("         (problème générique de la tension de corde).")

    print("\nPrédiction σ=D²Λ² pour d'autres ordres (non testable dans la TQNT) :")
    orders = {"Ising SU(2)_2": 4.0, "Fibonacci": (2 * np.cos(np.pi / 5)) ** 2 + 1,
              "toric code Z_2": 4.0, "SU(2)_3": 2 + (2 * np.cos(np.pi / 5)) ** 2}
    for nm, D2 in orders.items():
        print(f"  {nm:16s} D²={D2:.3f} → σ/Λ² = {D2:.2f}")


if __name__ == "__main__":
    main()
