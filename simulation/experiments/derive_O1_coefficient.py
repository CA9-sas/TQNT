r"""
Le coefficient O(1) de σ = c·D²Λ² est-il dérivable ? — NON, et cela révèle une
limite plus profonde (bedrock).

Fait numérique : σ_χ/Λ_χ² = 0,18/0,216² = 3,86 ≈ 4 = D²(Ising) ; c = 0,965 ≈ 1.

Deux obstacles :
  1. GÉNÉRIQUE — le coefficient sans dimension d'une tension de corde n'est pas
     dérivable de 1er principe (même en QCD, √σ/Λ_QCD est un nombre du réseau).
  2. PLUS PROFOND — D² n'est PAS l'invariant topologique UNIVERSEL. L'invariant
     universel est D, via l'entropie d'intrication topologique S_topo = −log D
     (Kitaev-Preskill), donc une dépendance LOGARITHMIQUE. Une relation
     rigoureuse donnerait σ ~ Λ²·f(log D) ~ Λ²·log2 = 0,032 GeV², FAUX d'un
     facteur ~5,6. Les données préfèrent D²=4 (linéaire), pas −log D.

CONCLUSION : σ_χ ≈ 4Λ_χ² est une relation NUMÉRIQUE propre (3,7 %) qui coïncide
avec D²(Ising)=4 et est cohérente avec le √2 de Koide (même secteur Ising). Mais
la « dérivation » par D² est SUGGESTIVE, non rigoureuse : ni le coefficient O(1)
ni la puissance D² (vs l'universel log D) ne se dérivent. C'est le bedrock : le
facteur 4 est une coïncidence structurellement séduisante, pas un théorème.
"""
from __future__ import annotations
import numpy as np

LAMBDA_CHI, SIGMA, D = 0.216, 0.18, 2


def main():
    r = SIGMA / LAMBDA_CHI ** 2
    print(f"σ/Λ² = {r:.3f} ; D²(Ising) = {D**2} ; c = {r/D**2:.3f} ≈ 1\n")

    print("Obstacle 1 (générique) : coefficient de corde non dérivable (cf. QCD réseau).\n")

    print("Obstacle 2 (profond) : l'invariant universel est D (log), pas D² (linéaire).")
    logD = np.log(D)
    sig_log = logD * LAMBDA_CHI ** 2
    print(f"  S_topo = -log D = {-logD:+.3f} (universel, Kitaev-Preskill)")
    print(f"  relation rigoureuse σ ~ Λ²·log D = {sig_log:.3f} GeV² vs obs {SIGMA}"
          f" → faux d'un facteur {SIGMA/sig_log:.1f}")
    print(f"  les données veulent D²={D**2} (linéaire), PAS -log D.\n")

    print("BEDROCK : σ_χ≈4Λ_χ² = coïncidence numérique (3,7 %) cohérente avec D²(Ising)")
    print("et le √2 de Koide, MAIS ni le O(1) ni la puissance D² ne se dérivent.")
    print("Le facteur 4 est structurellement séduisant, pas un théorème.")


if __name__ == "__main__":
    main()
