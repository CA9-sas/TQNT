r"""
Chantier 1 — qualifier les boucles superposées par rapport aux axes.

Objectif (insight utilisateur) : atteindre 0 % d'écart sur p/n. Ce
nécessite une qualification GÉOMÉTRIQUE FINE de la manière dont les
composantes d'un entrelacs sont disposées et superposées par rapport
à leurs axes principaux.

Méthode :
1. Pour chaque nœud du catalogue, calculer son ELLIPSOÏDE D'INERTIE
   via PCA des points 3D :
     - λ₁ ≥ λ₂ ≥ λ₃ = valeurs propres
     - axes principaux e₁, e₂, e₃
     - sphéricité S = 3λ₃ / (λ₁+λ₂+λ₃)
     - planarité P = (λ₂-λ₃)/(λ₁+λ₂+λ₃)
     - linearité L = (λ₁-λ₂)/(λ₁+λ₂+λ₃)
2. Caractériser chaque nœud comme :
     - "axial" : très linéaire (un axe dominant — torus, ring)
     - "planar" : plus plat que linéaire (figure-eight)
     - "sphérique" : isotrope (complex tangle)
3. Pour les composites du spectre, postuler l'arrangement axial OPTIMAL
   (composantes co-axiales si possible) et estimer le facteur de
   superposition géométrique.
4. Tester si cette « algèbre d'axes » peut expliquer l'écart résiduel
   après la correction linking pour p/n.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K


def inertia_analysis(pts: np.ndarray) -> dict:
    """Analyse PCA d'une courbe 3D : valeurs propres et axes principaux
    du tenseur d'inertie (matrice de covariance centrée)."""
    pts = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    cov = (pts_c.T @ pts_c) / len(pts_c)
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending order
    # tri descendant
    idx = np.argsort(eigvals)[::-1]
    lam = eigvals[idx]
    vecs = eigvecs[:, idx]
    total = lam.sum()
    return {
        "lam1": float(lam[0]), "lam2": float(lam[1]), "lam3": float(lam[2]),
        "axis1": vecs[:, 0], "axis2": vecs[:, 1], "axis3": vecs[:, 2],
        "sphericity": float(3 * lam[2] / total) if total > 0 else 0.0,
        "linearity":  float((lam[0] - lam[1]) / total) if total > 0 else 0.0,
        "planarity":  float((lam[1] - lam[2]) / total) if total > 0 else 0.0,
        "anisotropy": float((lam[0] - lam[2]) / lam[0]) if lam[0] > 0 else 0.0,
        "centroid": centroid,
        "size": float(np.sqrt(total)),
    }


def classify(props: dict) -> str:
    """Classification simple basée sur (linearity, planarity, sphericity)."""
    L = props["linearity"]; P = props["planarity"]; S = props["sphericity"]
    if L > 0.4:
        return "axial"
    if P > 0.3:
        return "planar"
    if S > 0.6:
        return "sphérique"
    return "intermédiaire"


def main():
    print("=" * 90)
    print("  QUALIFICATION DES BOUCLES PAR LEURS AXES (PCA / inertie)")
    print("=" * 90)

    n_points = 3000

    # ---------- (1) PCA de chaque nœud du catalogue ---------------------
    print(f"\n  (1) Ellipsoïdes d'inertie (PCA) des 19 catalogues :")
    print(f"  {'K':>5} {'λ₁':>7} {'λ₂':>7} {'λ₃':>7} {'linéarité':>10} "
          f"{'planarité':>10} {'sphéricité':>11} {'classe':>14}")
    print("  " + "-" * 80)
    props_all = {}
    for K in REFERENCE_N50K:
        pts = knot_curve(K, n_points)
        p = inertia_analysis(pts)
        cls = classify(p)
        props_all[K] = (p, cls)
        print(f"  {K:>5} {p['lam1']:>7.3f} {p['lam2']:>7.3f} {p['lam3']:>7.3f} "
              f"{p['linearity']:>10.3f} {p['planarity']:>10.3f} "
              f"{p['sphericity']:>11.3f} {cls:>14}")

    # ---------- (2) Composantes du nucléon : 3_1, 7_2, 7_2 -------------
    print(f"\n  (2) Composantes du NUCLÉON (3_1 + 7_2 + 7_2) :")
    for K in ("3_1", "7_2"):
        p, cls = props_all[K]
        print(f"\n    {K} :")
        print(f"      taille = {p['size']:.3f}")
        print(f"      axe principal (e₁) = {p['axis1']}")
        print(f"      linéarité = {p['linearity']:.3f}")
        print(f"      planarité = {p['planarity']:.3f}")
        print(f"      sphéricité = {p['sphericity']:.3f}")
        print(f"      classe : {cls}")

    # ---------- (3) Test d'alignement potentiel ------------------------
    p31, _ = props_all["3_1"]
    p72, _ = props_all["7_2"]
    cos_align = abs(float(p31["axis1"] @ p72["axis1"]))
    print(f"\n  (3) Alignement potentiel des axes 3_1 / 7_2 (cos angle) :")
    print(f"      |cos(e₁(3_1), e₁(7_2))| = {cos_align:.3f}")
    print(f"      (1 = parfaitement co-axial, 0 = orthogonal)")

    # ---------- (4) Pour TOUS les nœuds, classes ----------------------
    print(f"\n  (4) Répartition des classes sur le catalogue :")
    from collections import Counter
    classes = Counter(cls for _, cls in props_all.values())
    for c, n in classes.most_common():
        print(f"    {c:>14} : {n} nœuds")

    # ---------- (5) Synthèse pour le proton ----------------------------
    print(f"\n  (5) Implication pour la masse du proton :")
    p_3_1 = props_all["3_1"][0]
    p_7_2 = props_all["7_2"][0]
    print(f"    3_1 et 7_2 sont {props_all['3_1'][1]} / {props_all['7_2'][1]}.")
    print(f"    En arrangement co-axial OPTIMAL, les 3 composantes s'enroulent")
    print(f"    autour d'un axe commun. Le 'volume occupé' est alors :")
    print(f"      V ≈ π · r_max² · (somme des longueurs axiales)")
    print(f"    au lieu de la somme V_i = Σ π · r_i² · L_i.")
    print(f"")
    print(f"    Le facteur de superposition axiale dépend de :")
    print(f"      - la linéarité de chaque composante (axe bien défini ou pas) ;")
    print(f"      - leur taille relative (les 'petites' s'insèrent dans les 'grandes') ;")
    print(f"      - la phase d'enroulement (Lk = nombre de tours co-axial entiers).")

    # ---------- (6) Honnêteté ------------------------------------------
    print(f"\n  HONNÊTETÉ :")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Pour atteindre 0 % d'écart sur p, n (précision PDG ~10⁻⁸), il")
    print(f"  faudrait :")
    print(f"  ")
    print(f"  (a) Spécifier l'arrangement 3D EXACT des 3 composantes du nucléon")
    print(f"      (positions, orientations, phases d'enroulement) ;")
    print(f"  (b) Calculer la rope-length VRAIE de cet entrelacs (Pieranski sur")
    print(f"      le LIEN, pas la somme des composantes isolées) ;")
    print(f"  (c) Compter l'effet de superposition axiale (PCA jointe des 3) ;")
    print(f"  (d) Inclure les modes d'oscillation autour de l'équilibre.")
    print(f"  ")
    print(f"  Le catalogue actuel theory/knot_catalog.py ne contient que des")
    print(f"  paramétrisations INDIVIDUELLES — l'arrangement multi-composantes")
    print(f"  du nucléon n'y est PAS spécifié. C'est une donnée manquante :")
    print(f"  V4 contourne en sommant U_EM, c_J indépendamment, ce qui fitte")
    print(f"  empiriquement la bonne valeur via α₀ et les f(n).")
    print(f"  ")
    print(f"  Pour atteindre 0 %, il faudrait CONSTRUIRE explicitement la")
    print(f"  courbe 3D du nucléon (et de chaque composite) et la traiter")
    print(f"  comme un objet géométrique unique.")
    print(f"  ")
    print(f"  ⇒ C'est le chantier 1 jalon J6 (composites) : reste ouvert.")
    print("=" * 90)


if __name__ == "__main__":
    main()
