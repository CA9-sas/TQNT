r"""
M₀ via la VALEUR PROPRE du puits χ FORT, sur courbes VALIDES et RÉPARÉES.

Dernière route pour la tête de bande M₀ (cf. brouillon_masse_rotation_regge.tex) :
l'auto-énergie du puits χ fort est la valeur propre fondamentale E₀ de
l'opérateur de Schrödinger χ (potentiel de retraction de Jones, géométrie du
nœud). Contrairement à la ropelength (longueur), c'est une quantité SPECTRALE
(forme du puits) → elle POURRAIT distinguer des nœuds de même longueur.

On la calcule sur :
  - les courbes VALIDES (toriques 3₁,5₁,7₁,9₁ + fig-8 4₁, catalogue) ;
  - les courbes RÉPARÉES (tresses validées Alexander : 5₂,7₂,8₃,6₂,8₁),
    injectées dans le solveur (recentrées + remises à l'échelle).

RÉSULTATS :
  (1) E₀ suit fortement la complexité : sur les toriques corr(E₀,croisements)
      = −0,987 (puits plus profond pour nœud plus complexe). Sur les réparées,
      |E₀| croît avec la retraction (5₂:18,7 → 8₃:49,1). Le trend qualitatif
      masse↔complexité est reproduit — mieux fondé que la ropelength.
  (2) MAIS la hiérarchie de SAVEUR échappe : les nœuds 7₄/7₅/7₆ (assignés à
      J/ψ, φ, Υ) ont chi_retraction = 8,5 / 8,7 / 8,9 (écart 2 %) alors que les
      masses couvrent ×9 (1019 / 3097 / 9460 MeV). Le puits χ d'un nœud unique
      NE PEUT PAS produire ce facteur 9 : l'échelle « saveur » n'est pas dans la
      topologie du nœud, elle est dans le CHOIX D'ASSIGNATION.

CONCLUSION : la valeur propre du puits fort (la « meilleure » forme spectrale,
celle qu'utilisait Ω_ts) capture la complexité mais PAS la saveur. M₀ =
complexité (dérivable, qualitatif) + saveur (dans l'assignation, non dérivable).
La tête de bande reste sous-déterminée — cohérent avec « flavor = SM bookkeeping ».
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem
from theory.chi_field import chi_retraction
from simulation.solver import solve_knot
from simulation.experiments.repair_curves_braid import braid_curve, BRAID_WORDS


def eig_valid(knot_name, n_coarse=24):
    """E₀ sur la courbe catalogue VALIDE (toriques + fig-8)."""
    r = solve_knot(knot_name, n_coarse=n_coarse, n_eigenvalues=3, epsilon=0.2)
    return float(np.sort(r["eigenvalues"])[0])


def eig_repaired(knot_name, n_coarse=24):
    """E₀ sur la courbe RÉPARÉE (tresse) — recentrée + mise à l'échelle."""
    word, ns, _ = BRAID_WORDS[knot_name]
    P, _, _ = braid_curve(word, ns, pts_per_seg=60)
    P = P - P.mean(0)
    P = P * (3.0 / np.max(np.abs(P)))
    tree = cKDTree(P)
    pts, tets, bnd = generate_mesh(P, n_coarse=n_coarse, epsilon=0.2, refine=True)
    H, M = assemble_system(pts, tets, bnd, knot_name, tree, energy=0.0, epsilon=0.2)
    ev, _ = solve_eigenproblem(H, M, n_eigenvalues=3)
    return float(np.sort(ev)[0])


# knots-saveur : masse du vecteur J=1 associé (MeV)
FLAVOR = {"7_4": ("J/ψ", 3096.9), "7_5": ("φ", 1019.5), "7_6": ("Υ", 9460.0)}


def main():
    print("=== E₀ (puits χ fort) sur courbes VALIDES (toriques + fig-8) ===")
    cr = {"3_1": 3, "4_1": 4, "5_1": 5, "7_1": 7, "9_1": 9}
    Es = {}
    for K in cr:
        Es[K] = eig_valid(K)
        print(f"  {K}: E₀ = {Es[K]:9.3f}  (cr={cr[K]})")
    E = np.array([Es[k] for k in cr]); C = np.array([cr[k] for k in cr])
    print(f"  corr(E₀, croisements) = {np.corrcoef(E, C)[0, 1]:.3f}")

    print("\n=== E₀ sur courbes RÉPARÉES (tresses Alexander) ===")
    for K in ["5_2", "6_2", "7_2", "8_1", "8_3"]:
        e = eig_repaired(K)
        print(f"  {K}: E₀ = {e:9.3f}  chi_retr = {chi_retraction(K, 0.0):.3f}")

    print("\n=== La saveur échappe : 7₄/7₅/7₆ (J/ψ, φ, Υ) ===")
    for K, (nm, m) in FLAVOR.items():
        print(f"  {K} -> {nm:4s}  chi_retr = {chi_retraction(K, 0.0):.3f}  masse = {m:7.1f} MeV")
    rs = [chi_retraction(k, 0.0) for k in FLAVOR]
    ms = [m for _, m in FLAVOR.values()]
    print(f"  retraction : {min(rs):.1f}–{max(rs):.1f} (écart {max(rs)/min(rs)-1:.0%})"
          f"  vs masses ×{max(ms)/min(ms):.0f}")
    print("  -> le puits χ d'un nœud unique ne peut PAS produire ce ×9 :")
    print("     la saveur vit dans l'ASSIGNATION, pas dans la topologie du nœud.")


if __name__ == "__main__":
    main()
