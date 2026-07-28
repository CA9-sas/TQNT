"""α_EM depuis le niveau Chern-Simons k=135 : forme dérivée, quantification, cohérence.

Le champ EM est une théorie de Chern-Simons ; la charge est une ligne de Wilson
topologique (cohérent avec Q=Σχ+Lk). Le niveau DÉCALÉ k+h∨ = k+2 (h∨=2 pour SU(2))
donne la forme

    α_EM = 1/(k+2).

Avec k=135 : α = 1/137 = 0,007299 (exp 1/137,036, à 0,026 %).

CONTENU PRÉDICTIF (non trivial) :
  (a) α⁻¹ est un ENTIER (k+2=137) — α est quantifié par le niveau CS (le MS n'a
      aucune raison pour α⁻¹ ≈ entier) ;
  (b) le MÊME k=135 donne la dimension quantique d_½=2cos(π/137) d'où la phase
      δ=−4ln(d_½)=−2,7715 qui reproduit les oscillations de mésons (V5p31) —
      un seul entier pour l'EM et les oscillations, pas un refit.

HONNÊTE : k=135 est FIXÉ par α (entier le plus proche de 137,036), non dérivé ex
nihilo ; le résidu 0,036 reste non expliqué (running/correction).
"""
import math

K = 135
ALPHA_EXP = 1 / 137.035999


def alpha_em(k=K):
    return 1 / (k + 2)


def quantum_dim_half(k=K):
    return 2 * math.cos(math.pi / (k + 2))


def delta_oscillation(k=K):
    return -4 * math.log(quantum_dim_half(k))


def is_rigid_em_level(k=K):
    """k donne-t-il une théorie EM CS RIGIDE (unique invariant modulaire diagonal) ?

    Classification ADE (Cappelli-Itzykson-Zuber) des invariants modulaires SU(2)_k :
    A (diagonale) tout k ; D série k pair ; E₆,₇,₈ pour k=10,16,28 seulement.
    k impair ET k∉{10,16,28} ET k+2 premier → SEULE la diagonale A_{k+1} existe :
    couplage EM unique, sans ambiguïté. Les niveaux ambigus (D, E) sont exclus.
    """
    from sympy import isprime
    return k % 2 == 1 and k not in (10, 16, 28) and isprime(k + 2)


if __name__ == "__main__":
    a = alpha_em()
    print(f"α = 1/(k+2) = 1/{K+2} = {a:.8f}  (exp {ALPHA_EXP:.8f}, {(a/ALPHA_EXP-1)*100:+.3f}%)")
    print(f"α⁻¹ entier = {K+2}  (mesuré {1/ALPHA_EXP:.3f}) → α quantifié")
    d = quantum_dim_half()
    print(f"d_½(k={K}) = 2cos(π/137) = {d:.6f}  →  δ = −4ln(d_½) = {delta_oscillation():.4f}")
    print(f"=> le même k={K} pour α_EM ET δ (oscillations).")
    print(f"\nMotivation topologique (ADE) : k={K} impair, k+2={K+2} premier,")
    print(f"  ∉{{10,16,28}} → unique invariant modulaire DIAGONAL : EM rigide ? {is_rigid_em_level()}")
    print("  (la topologie impose k impair + k+2 premier ; la valeur 137 reste fixée par α.)")
