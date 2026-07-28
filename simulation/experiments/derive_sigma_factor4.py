r"""
Dérivation du facteur 4 de σ_χ = 4Λ_χ² (tension du tube χ vs échelle χ).

Le tube de flux χ vit dans le secteur χ FORT = ordre d'Ising SU(2)_2 (programme
gluon-Ising). La tension d'un tube dans un ordre topologique est fixée par sa
DIMENSION QUANTIQUE TOTALE D : σ = D²·Λ². Pour Ising, D² = 4.

Dimensions quantiques des primaires SU(2)_k : d_j = sin((2j+1)π/(k+2))/sin(π/(k+2)).
À k=2 (Ising), les trois anyons j=0,1/2,1 :
    d_0 = 1 (vide) ,  d_{1/2} = √2 (σ, spin) ,  d_1 = 1 (ψ, fermion).
    D² = Σ_j d_j² = 1 + 2 + 1 = 4 ,   D = 2.

D'où  σ_χ = D²·Λ_χ² = 4Λ_χ²   et   √σ_χ = D·Λ_χ = 2Λ_χ.
Vérif : 4·(0,216)² = 0,187 GeV² vs σ_χ=0,18 (3,7 %) ; ou Λ_χ=√σ/2=212 MeV vs 216 (1,8 %).

COHÉRENCE INTERNE (même secteur Ising k=2) :
  - Koide   : amplitude √2 = d_{1/2} = d_σ (l'anyon de spin).
  - Tension : facteur 4 = D² = d_1²+d_σ²+d_ψ² (dimension totale).
Le facteur 4 n'est donc PLUS heuristique : c'est D²(Ising), structurel, exact,
et cohérent avec le √2 de Koide.

Réserve : le principe « tension = D²·Λ² » (la tension du tube réorganise tous les
canaux d'anyons, comptés par D²) est motivé mais sa dérivation de théorie des
champs rigoureuse reste à faire. L'IDENTIFICATION du 4 à D²=4 est exacte.
"""
from __future__ import annotations
import sympy as sp


def quantum_dim(j, k):
    """Dimension quantique de spin j au niveau SU(2)_k."""
    return sp.simplify(sp.sin((2 * j + 1) * sp.pi / (k + 2)) / sp.sin(sp.pi / (k + 2)))


def total_quantum_dim_sq(k):
    """D² = Σ_j d_j², somme sur les primaires j=0,1/2,...,k/2."""
    js = [sp.Rational(n, 2) for n in range(k + 1)]
    return sp.simplify(sum(quantum_dim(j, k) ** 2 for j in js)), js


def main():
    k = 2
    print(f"Secteur χ fort = Ising = SU(2)_{k} (programme gluon)\n")
    D2, js = total_quantum_dim_sq(k)
    for j in js:
        print(f"  d_{j} = {sp.nsimplify(quantum_dim(j, k))}")
    print(f"\n  D² = Σ d_j² = {D2} ,  D = {sp.sqrt(D2)}")
    print(f"\n  => σ_χ = D²·Λ_χ² = {D2}·Λ_χ²  ,  √σ_χ = D·Λ_χ = 2Λ_χ")

    Lam, sig = 0.216, 0.18
    print(f"\n  Vérif : 4·(0.216)² = {4*Lam**2:.4f} GeV² vs {sig} "
          f"(écart {abs(4*Lam**2-sig)/sig*100:.1f} %)")
    print(f"          Λ_χ = √σ/2 = {(sig**0.5)/2*1000:.0f} MeV vs 216 "
          f"(écart {abs((sig**0.5)/2*1000-216)/216*100:.1f} %)")

    print("\n  Cohérence Ising k=2 : Koide √2 = d_σ ; tension 4 = D². Même secteur.")
    assert D2 == 4


if __name__ == "__main__":
    main()
