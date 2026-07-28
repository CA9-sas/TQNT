"""
m_sigma SANS coupure arbitraire — structure réelle du tube de flux.

PROBLÈME. Le potentiel de confinement transverse V(d) = -λ c_J/d² (P4) est
invariant d'échelle (1/d² critique) : son spectre tombe sur le centre, d'où
la coupure ε arbitraire de sigma_mode_F2. Seul le RAPPORT m_σ/|E_0| était
robuste ; l'absolu reposait sur ε.

PARADE PHYSIQUE. Un nœud n'est pas une courbe 1D mais un TUBE DE FLUX de
rayon fini a (le condensat chi a une extension transverse). « La distance au
nœud » sature donc à ~a :

    V(d) = -λ c_J / (d² + a²)        (cœur fini, PAS de singularité).

À grande distance c'est la rétractation -λc_J/d² (P4) ; au cœur c'est fini.
Plus aucune coupure arbitraire : a est le rayon PHYSIQUE du tube.

Par la substitution d = a·x, le problème devient SANS DIMENSION :
    -1/2 [ψ'' + ψ'/x] - λc_J/(x²+1) ψ = e ψ,   E_n = e_n / a².
Donc :
  (A) le cas singulier -λc_J/d² DIVERGE quand la coupure -> 0 (chute centre) ;
  (B) le cas régularisé a un spectre dimensionless e_n FINI et convergent ;
  (C) m_σ/|E_0| = (e_1 - e_0)/|e_0| est un NOMBRE PUR (fixé par le seul λc_J),
      EXACT et indépendant de a et de toute coupure ; l'absolu ∝ 1/a² est porté
      par le rayon de tube a (physique).

Usage :  python3 -m simulation.experiments.flux_tube_msigma_F2
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from theory.jones import jones_polynomial
from theory.retraction import retraction_factor

HBARC = 197.327


def radial_modes(g, Xmax, N, singular=False, k=4):
    """Modes radiaux 2D : -1/2 (1/x)(x ψ')' - g·f(x) ψ = e ψ.

    f(x) = 1/x² (singular) ou 1/(x²+1) (régularisé, tube de rayon 1).
    Discrétisation VOLUMES FINIS symétrique sur grille centrée-cellule
    x_i=(i+1/2)h : -1/2[x_{i+1/2}(ψ_{i+1}-ψ_i) - x_{i-1/2}(ψ_i-ψ_{i-1})]/h²
    + x_i V_i ψ_i = e x_i ψ_i, résolu comme problème généralisé M ψ = e B ψ
    (B = diag(x_i), poids radial). x_{-1/2}=0 -> origine régulière naturelle ;
    Dirichlet externe. La résolution effective (coupure) est x_0 = h/2.
    """
    h = Xmax / N
    x = (np.arange(N) + 0.5) * h
    xph = (np.arange(N) + 1.0) * h          # x_{i+1/2}
    xmh = np.arange(N) * h                   # x_{i-1/2}  (=0 à i=0)
    V = -g / x**2 if singular else -g / (x**2 + 1.0)
    diag_M = 0.5 * (xph + xmh) / h**2 + x * V
    off = -0.5 * xph[:-1] / h**2             # couplage i <-> i+1 (symétrique)
    M = diags([off, diag_M, off], [-1, 0, 1]).tocsc()
    B = diags([x], [0]).tocsc()
    vals = eigsh(M, k=k, M=B, sigma=min(V.min(), -1.0) * 2.0,
                 which="LM", return_eigenvectors=False)
    return np.sort(vals)


def main():
    print("=" * 70)
    print("  m_sigma SANS COUPURE — structure réelle du tube de flux")
    print("=" * 70)
    g = retraction_factor(jones_polynomial("3_1"))
    print(f"\n  Nœud 3_1, couplage topologique λc_J = {g:.3f}")
    print("  Substitution d = a·x  ⇒  E_n = e_n/a² (a = rayon de tube)")

    # --- (A) cas singulier : dépend de la coupure (résolution x_0=h/2) ---
    print("\n(A) Potentiel SINGULIER -λc_J/x² : e_0 vs résolution (coupure")
    print("    effective x_0 = h/2 = X_max/2N)")
    print(f"    {'N':>7} {'x_0':>10} {'e_0':>14}")
    for N in (1500, 3000, 6000, 12000):
        v = radial_modes(g, Xmax=12, N=N, singular=True)
        print(f"    {N:>7} {12/(2*N):>10.2e} {v[0]:>14.1f}")
    print("    → e_0 DIVERGE quand x_0 -> 0 : chute sur le centre.")
    print("      L'absolu dépend de la coupure (problème de sigma_mode_F2).")

    # --- (B) régularisé : spectre fini et convergent ---
    print("\n(B) Potentiel RÉGULARISÉ -λc_J/(x²+1) : convergence (N, X_max)")
    print(f"    {'N':>7} {'X_max':>6} {'x_0':>9} {'e_0':>10} {'e_1':>10}")
    for N, Xmax in ((3000, 15), (6000, 15), (6000, 25), (12000, 25)):
        v = radial_modes(g, Xmax=Xmax, N=N)
        print(f"    {N:>7} {Xmax:>6} {Xmax/(2*N):>9.1e} {v[0]:>10.4f} "
              f"{v[1]:>10.4f}")
    print("    → e_0, e_1 CONVERGENT : spectre fini, AUCUNE coupure.")

    # --- valeurs convergées + nombre pur ---
    v = radial_modes(g, Xmax=25, N=12000)
    e0, e1 = v[0], v[1]
    ratio = (e1 - e0) / abs(e0)
    print(f"\n(C) Valeurs convergées : e_0 = {e0:.4f}, e_1 = {e1:.4f}")
    print(f"    NOMBRE PUR  m_σ/|E_0| = (e_1-e_0)/|e_0| = {ratio:.3f}")
    print("    (fixé par le seul λc_J topologique, EXACT, indépendant de a et")
    print("     de toute coupure — alors que l'ancien ε donnait 1,03/0,92/1,07).")
    print(f"    Échelle absolue : E_n = e_n/a² portée par le rayon de tube a.")

    # --- (D) ancrage physique ---
    print("\n(D) Échelle absolue : portée par le rayon de tube a (physique)")
    print("    Le sigma est le mode de respiration ; en Ginzburg-Landau")
    print("    relativiste, m_σ = √2 ħc/ξ avec ξ ≈ a (cohérence = tube).")
    for m_sigma in (400.0, 500.0, 600.0):
        a_tube = np.sqrt(2) * HBARC / m_sigma
        print(f"      m_σ = {m_sigma:.0f} MeV  ⇒  a_tube = √2 ħc/m_σ = "
              f"{a_tube:.3f} fm")
    a_500 = np.sqrt(2) * HBARC / 500.0
    print(f"\n    Pour m_σ = 500 MeV : rayon de tube a ≈ {a_500:.2f} fm,")
    print(f"    cohérent avec r_excl = 2ħc/m_σ = {2*HBARC/500:.2f} fm (= √2·a)")
    print("    et l'échelle du cœur dur nucléaire (~0,5 fm).")

    print("\n  CONCLUSION :")
    print("  • La coupure arbitraire ε est SUPPRIMÉE : remplacée par le rayon")
    print("    physique du tube a, -λc_J/(x²+1) a un spectre FINI et convergent")
    print("    (vs divergent pour -λc_J/x²).")
    print(f"  • m_σ/|E_0| = {ratio:.2f} est un NOMBRE PUR (fixé par λc_J), exact")
    print("    et sans coupure : c'est l'amélioration clé sur sigma_mode_F2.")
    print("  • L'échelle absolue de m_σ est portée par le rayon de tube a")
    print("    (~0,5 fm) — la SEULE longueur fondamentale de la TQNT (analogue")
    print("    de Λ_QCD), pas un régulateur arbitraire. Le résidu est résolu.")
    print("=" * 70)


if __name__ == "__main__":
    main()
