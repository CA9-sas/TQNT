r"""
DÉRIVATION RIGOUREUSE du facteur 2/3 (valeur de Koide) — le facteur Z₃ de la saveur.

Le facteur 2/3 apparaît (i) dans la relation de Koide leptonique (Q=2/3, exact) et
(ii) dans l'énergie de mode Z₃ hadronique (E_Z₃ ≈ (2/3)Λ_χ). On le dérive.

ÉTAPE 1 — depuis la paramétrisation Z₃. La relation de Koide s'écrit
    √m_k = M(1 + A cos(2πk/3 + δ)),  k = 0,1,2   (trois modes Z₃, amplitude A).
Les trois angles sont équi-répartis (pas 2π/3) ⇒ Σcos(θ_k)=0 et Σcos²(θ_k)=3/2.
Donc, INDÉPENDAMMENT de δ :
    Σ√m = 3M ,   Σm = 3M²(A²+2)/2 ,
    Q = Σm/(Σ√m)² = (A² + 2)/6 = (1 + A²/2)/3.

ÉTAPE 2 — d'où vient A = √2. La dimension quantique du spin-½ au niveau de
Chern-Simons k est d_{1/2}(k) = 2cos(π/(k+2)). Au niveau du secteur χ FORT
(vortex d'Ising SU(2)_2, k=2, programme gluon) :
    d_{1/2}(2) = 2cos(π/4) = √2.
Donc A = √2 n'est PAS un postulat : c'est la dimension quantique spin-½ à k=2.

ÉTAPE 3 — assemblage.
    Q = (1 + d_{1/2}(2)²/2)/N_gen = (1 + 2/2)/3 = 2/3.
avec N_gen = 3 (Z₃ du trèfle : 3 croisements / symétrie 3-fold / 3 générations)
et d_{1/2}(2)² = 2. Le 2/3 est DÉRIVÉ ; le postulat r=√2 est éliminé.

RÉSERVE : (a) k=2 pour le secteur χ fort repose sur le programme gluon-Ising
(SU(2)_2). (b) Le MÊME 2/3 dans l'énergie hadronique E_Z₃=(2/3)Λ_χ (147 vs 144,
1,8 %) reflète la même structure Z₃/k=2, MAIS l'incrément hadronique est LINÉAIRE
(espacement égal Gell-Mann-Okubo), pas de forme Koide — le lien « fraction
d'énergie = Q » reste à établir. Rigoureux : le Q leptonique. Suggestif : l'énergie.
"""
from __future__ import annotations
import sympy as sp


def koide_Q_symbolic():
    """Q en fonction de l'amplitude A (paramétrisation Z₃, N=3). δ-indépendant."""
    M, A, d = sp.symbols("M A delta", positive=True, real=True)
    sm = [M * (1 + A * sp.cos(2 * sp.pi * k / 3 + d)) for k in range(3)]
    S1 = sp.simplify(sum(sm))
    S2 = sp.simplify(sum(s ** 2 for s in sm))
    return sp.simplify(S2 / S1 ** 2), sp.simplify(S1), sp.simplify(S2)


def quantum_dim_half(k):
    """Dimension quantique du spin-½ au niveau CS k : d_{1/2}(k) = 2cos(π/(k+2))."""
    return 2 * sp.cos(sp.pi / (k + 2))


def main():
    print("ÉTAPE 1 — Q depuis la paramétrisation Z₃ (N=3) :")
    Q, S1, S2 = koide_Q_symbolic()
    print(f"  Σ√m = {S1} ;  Σm = {S2}")
    print(f"  Q = (A²+2)/6 = {Q}   [indépendant de δ]")

    print("\nÉTAPE 2 — A = √2 = dimension quantique spin-½ à k=2 :")
    for k in (2, 135):
        v = quantum_dim_half(k)
        print(f"  d_1/2({k}) = 2cos(π/{k+2}) = {sp.nsimplify(v)} = {float(v):.4f}")
    print("  secteur χ FORT (Ising SU(2)_2) : k=2 ⇒ A = √2, A² = 2")

    print("\nÉTAPE 3 — assemblage :")
    A_sq = 2  # d_1/2(2)² = (√2)²
    Asym = next(iter(Q.free_symbols))              # le symbole A dans Q
    Qfin = sp.nsimplify(Q.subs(Asym, sp.sqrt(A_sq)))
    print(f"  Q = (1 + d_1/2(2)²/2)/3 = (1 + {A_sq}/2)/3 = {Qfin}")
    assert Qfin == sp.Rational(2, 3)
    print("  ✓ 2/3 DÉRIVÉ — plus de postulat r=√2 (c'est d_1/2(k=2)).")
    print("\n  Ingrédients : N_gen=3 (Z₃ trèfle) + d_1/2(k=2)=√2 (spin-½ à k=2).")


if __name__ == "__main__":
    main()
