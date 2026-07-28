r"""
Statut honnête du facteur D²=4 — CONVENTION dans le proton, SUGGESTIF ailleurs.

Demande utilisateur : « justifie topologiquement le facteur D²=4 déjà utilisé ».
Réponse rigoureuse : D²=4 n'est PAS rigoureusement dérivé (rétractation d'un
'D²=4 dérivé' laissé glisser dans la série masse absolue).

(1) DANS LE PROTON, D²=4 S'ANNULE (c'est une convention).
    m_p/Λ_χ = √(2π D² (J−α_0)) avec Λ_χ = √(σ_χ/D²)
            = √(2π D² (J−α_0)) · √(σ_χ/D²) = √(2π σ_χ (J−α_0)).
    → m_p = √(2π σ_χ (J−α_0)) : REGGE STANDARD (σ_χ tension, α_0 intercept, tous
    deux physiques/mesurés). Le proton ne TESTE pas D²=4. Le '4' vient de la
    DÉFINITION Λ_χ = √(σ_χ/4). Le proton est le fondamental de la trajectoire de
    Regge du nucléon — comme en QCD, pas une dérivation ab initio de D²=4.

(2) D²=4 EST une affirmation SEULEMENT si Λ_χ est indépendant (gap de masse /
    condensat χ). Alors σ_χ = D²Λ_χ² relie tension et gap ; D=√(d_0²+d_σ²+d_ψ²)=
    √(1+2+1)=2 (dim. quantique totale d'Ising), D²=4. Mesuré : σ_χ/Λ_χ² ≈ 3,86
    vs 4 → 3,5% (SUGGESTIF).

(3) PAS RIGOUREUSEMENT DÉRIVÉ (flag antérieur, brouillon_masse_redefinition) :
    (i) coefficient O(1) dans σ ~ (O(1))D²Λ² non dérivable (générique corde) ;
    (ii) D² n'est PAS l'invariant universel : l'universel est D via S_topo=−log D
    (Kitaev-Preskill), LOGARITHMIQUE ; σ~Λ²log D faux d'un facteur 5,6. Les
    données veulent D² (linéaire) pas −log D → CFT ne tranche pas.

VERDICT : D²=4 SUGGESTIF (Ising, cohérent avec √2 de Koide) mais NON dérivé.
Convention dans le proton (s'annule) ; suggestif comme σ_χ=D²Λ_χ² (3,86≈4). La
vraie base du proton = Regge (σ_χ, α_0 physiques). RÉTRACTATION : ne plus écrire
'D²=4 dérivé'. Impact : dans la synthèse masse absolue, remplacer 'D²=4 dérivé'
par 'σ_χ tension de Regge (mesurée)' ; le proton est standard-Regge, pas une
dérivation du facteur 4. Cohérent [[epistemic-assessment-V6]].
"""
from __future__ import annotations
import numpy as np

SIGMA = 1.096 / (2 * np.pi)     # σ_χ, GeV² (pente de Regge / 2π)
D2 = 4.0
LAM = np.sqrt(SIGMA / D2)
MP, J = 0.938272, 0.5


def main():
    a0 = J - MP ** 2 / (2 * np.pi * SIGMA)
    mp_regge = np.sqrt(2 * np.pi * SIGMA * (J - a0))
    print("(1) Proton : m_p = √(2π σ_χ (J−α_0)) [Regge standard] — D² S'ANNULE.")
    print(f"    √(2π σ_χ (J−α_0)) = {mp_regge:.4f} GeV = m_p ; α_0={a0:.3f} (intercept mesuré)")
    print(f"    le '4' = convention Λ_χ=√(σ_χ/4), pas une dérivation.\n")
    print("(2) σ_χ = D² Λ_χ² (si Λ_χ indépendant) : D(Ising)=√(1+2+1)=2, D²=4.")
    print(f"    mesuré σ_χ/Λ_χ² = {SIGMA / LAM ** 2:.2f} vs 4 → SUGGESTIF (3,5%).\n")
    print("(3) Non dérivé : coefficient O(1) non calculable ; D² vs log D (universel) ambigu.")
    print("\nVERDICT : D²=4 suggestif (Ising), PAS dérivé. Rétractation du 'D²=4 dérivé'.")
    print("  Le proton = Regge standard (σ_χ, α_0 physiques) ; le 4 est convention/suggestif.")


if __name__ == "__main__":
    main()
