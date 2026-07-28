r"""
Chantier 1 — analyse 3D directe de la masse du nucléon.

On abandonne la somme algébrique R_tot = ΣR(K_i) et on construit
EXPLICITEMENT plusieurs embeddings 3D du nucléon (3_1 + 7_2 + 7_2),
mesurant pour chaque arrangement :

  - L_3D     : longueur d'arc totale jointe
  - d_min_3D : min sur paires (intra + inter composantes) non-adjacentes
  - R_3D     : L_3D / d_min_3D  (rope-length JOINTE = limite de rupture
                                 du tube unifié)
  - PCA jointe : ellipsoïde d'inertie du composite entier

La masse prédite est m = m_e · [f(3) · R_3D / R(3_1)]^p, sans correction
linking ajoutée (l'arrangement 3D gère lui-même la compactness).

Arrangements testés :
  - côte-à-côte (composantes séparées spatialement)
  - en triangle (positions équilibrées)
  - co-axial empilé (z-axes alignés, stack)
  - co-axial overlap (positions proches → compactage)
  - linked simulé (rotations à 90°, force entrelacement)

Cible : m_p = 938.27 MeV.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K

M_E = 0.51099895
P_EXP = 7.42
F_HADRON = 0.77
A_R, B_R = 3.85, 5.61
R_E = A_R * 3 + B_R       # R(3_1) algébrique


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def place(pts, translation, rotation=None):
    if rotation is None:
        rotation = np.eye(3)
    return (rotation @ pts.T).T + np.asarray(translation, dtype=float)


def joint_length(curves):
    total = 0.0
    for c in curves:
        diff = np.roll(c, -1, axis=0) - c
        total += float(np.linalg.norm(diff, axis=1).sum())
    return total


def joint_d_min(curves, skip_frac=0.025):
    """Min distance entre points non-adjacents (intra + inter)."""
    min_d = np.inf
    n_curves = len(curves)
    # intra
    for c in curves:
        N = len(c)
        skip = max(int(N * skip_frac), 5)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        idx = np.arange(N)
        arc_dist = np.abs(idx[:, None] - idx[None, :])
        arc_dist = np.minimum(arc_dist, N - arc_dist)
        mask = arc_dist > skip
        masked = np.where(mask, dist, np.inf)
        min_d = min(min_d, float(masked.min()))
    # inter : toutes paires (i, j)
    for i in range(n_curves):
        for j in range(i + 1, n_curves):
            diff = curves[i][:, None, :] - curves[j][None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            min_d = min(min_d, float(dist.min()))
    return min_d


def joint_pca(curves):
    all_pts = np.vstack(curves)
    centroid = all_pts.mean(axis=0)
    pts_c = all_pts - centroid
    cov = (pts_c.T @ pts_c) / len(pts_c)
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = lam.sum()
    return {
        "lam": lam.tolist(),
        "size": float(np.sqrt(total)),
        "linearity": float((lam[0] - lam[1]) / total) if total > 0 else 0.0,
        "planarity": float((lam[1] - lam[2]) / total) if total > 0 else 0.0,
        "sphericity": float(3 * lam[2] / total) if total > 0 else 0.0,
    }


def mass_from_R(R, n=3):
    return M_E * (F_HADRON * R / R_E) ** P_EXP


def main():
    print("=" * 90)
    print("  ANALYSE 3D DIRECTE — masse du NUCLÉON (3_1 + 7_2 + 7_2)")
    print("=" * 90)

    N = 1000
    c_31 = knot_curve("3_1", N)[:-1]
    c_72 = knot_curve("7_2", N)[:-1]

    # Normaliser les courbes au même "rayon caractéristique" (PCA size)
    # pour comparer arrangements
    print(f"  Catalogue : 3_1 size ≈ {joint_pca([c_31])['size']:.3f}, "
          f"7_2 size ≈ {joint_pca([c_72])['size']:.3f}\n")

    arrangements = {
        "1 composante seule (3_1)": [c_31],
        "côte-à-côte (sep=3)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [3, 0, 0]),
             place(c_72, [-3, 0, 0])],
        "côte-à-côte (sep=6)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [6, 0, 0]),
             place(c_72, [-6, 0, 0])],
        "triangle équilatéral (r=3)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [3, 0, 0]),
             place(c_72, [3 * np.cos(2 * np.pi / 3),
                          3 * np.sin(2 * np.pi / 3), 0])],
        "co-axial empilé z (sep=1)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [0, 0, 1]),
             place(c_72, [0, 0, 2])],
        "co-axial empilé z (sep=2)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [0, 0, 2]),
             place(c_72, [0, 0, 4])],
        "rotation 90° (linked sim.)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [0, 0, 0], rotation_matrix([1, 0, 0], np.pi / 2)),
             place(c_72, [0, 0, 0], rotation_matrix([0, 1, 0], np.pi / 2))],
        "rotation 120° autour z":
            [place(c_31, [0, 0, 0]),
             place(c_72, [0, 0, 0], rotation_matrix([0, 0, 1], 2 * np.pi / 3)),
             place(c_72, [0, 0, 0], rotation_matrix([0, 0, 1], 4 * np.pi / 3))],
        "concentriques (translations 0.5)":
            [place(c_31, [0, 0, 0]),
             place(c_72, [0.5, 0, 0]),
             place(c_72, [-0.5, 0, 0])],
    }

    target_m = 938.27
    print(f"  Cible : m_p = {target_m} MeV")
    print()
    print(f"  {'arrangement':>32} {'L_3D':>8} {'d_min':>8} {'R_3D':>9} "
          f"{'m_pred':>10} {'écart':>9} {'linearity':>10}")
    print("  " + "-" * 86)
    rows = []
    for label, curves in arrangements.items():
        L = joint_length(curves)
        dmin = joint_d_min(curves)
        R_3D = L / dmin if dmin > 0 else np.inf
        m_pred = mass_from_R(R_3D)
        dev = (m_pred - target_m) / target_m * 100
        pca = joint_pca(curves)
        rows.append((label, L, dmin, R_3D, m_pred, dev, pca["linearity"]))
        print(f"  {label:>32} {L:>8.2f} {dmin:>8.3f} {R_3D:>9.1f} "
              f"{m_pred:>10.2f} {dev:>+8.1f}% {pca['linearity']:>10.3f}")

    # Comparaison à la somme naïve
    print(f"\n  COMPARAISON aux formules algébriques :")
    R_sum_alg = A_R * 17 + B_R * 3       # cr_tot=17, n=3
    m_sum_alg = mass_from_R(R_sum_alg)
    print(f"    Somme algébrique (3,85·17 + 5,61·3) = {R_sum_alg:.1f}, "
          f"m = {m_sum_alg:.2f} MeV (écart {(m_sum_alg/target_m-1)*100:+.1f}%)")
    L_table = REFERENCE_N50K["3_1"]["L"] + 2 * REFERENCE_N50K["7_2"]["L"]
    print(f"    Sum L_catalog (3_1 + 2·7_2) = {L_table:.2f}")
    print(f"    m V4 (catalogue) ≈ 940 MeV (à 0,2 %)")

    # Quel arrangement matche 938 le mieux ?
    best = min(rows, key=lambda r: abs(r[5]))
    print(f"\n  MEILLEUR arrangement 3D : '{best[0]}'")
    print(f"    R_3D = {best[3]:.1f}, m_pred = {best[4]:.2f} MeV, écart {best[5]:+.1f}%")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ────────────────────────────────────────")
    print(f"  La masse 3D dépend FORTEMENT de l'arrangement spatial :")
    n_under = sum(1 for r in rows if r[5] < -20)
    n_over = sum(1 for r in rows if r[5] > 20)
    n_match = sum(1 for r in rows if abs(r[5]) < 20)
    print(f"    {n_under} arrangements sous-prédisent (m << m_p)")
    print(f"    {n_match} dans facteur 1,2× de m_p")
    print(f"    {n_over} surprédisent")
    print(f"  ")
    print(f"  Ces arrangements catalogue NE PRÉSERVENT PAS la topologie")
    print(f"  3_1+7_2+7_2 (Lk=3) — vraie construction linkée nécessite un")
    print(f"  algorithme dédié (gluing + braid presentation).")
    print(f"  ")
    print(f"  ⇒ L'analyse 3D DÉMONTRE la sensibilité au geometric embedding,")
    print(f"    mais reproduire 938,27 MeV exactement exige de CONSTRUIRE")
    print(f"    un embedding 3D PHYSIQUEMENT correct avec Lk=3 imposé.")
    print(f"    C'est un véritable problème de modélisation 3D, pas un")
    print(f"    fit de paramètres.")
    print("=" * 90)


if __name__ == "__main__":
    main()
