r"""
Chantier 1 — Pieranski 2 sur le proton : embedding 3D physique de l'entrelacs
3_1 + 7_2 + 7_2 avec contrainte d'enlacement Lk=3.

Méthode :
  1. Partir des courbes catalogue de 3_1 et 7_2 (knot_curve), pré-relaxer
     chacune individuellement par Pieranski 1 (smoothing + auto-évitement).
  2. Placer les 3 composantes en arrangement "Hopf-linked triplet" :
     chaque paire interpénètre l'autre avec Lk(i,j) = 1, somme = 3.
  3. Joint Pieranski 2 :
       - smoothing Laplacien intra-composante
       - self-avoidance intra (D_intra)
       - inter-avoidance entre composantes (D_inter)
       - reparametrisation uniforme régulière
  4. Vérifier Lk total numériquement (formule de Gauss).
  5. Calculer X_geom sur l'UNION et appliquer GJ2.
  6. Comparer m_GJ2(embedding physique) vs m_GJ2(embedding triangulaire)
     vs m_V4 vs m_exp.

Question : un embedding physiquement correct corrige-t-il la pathologie
de GJ2 sur composites (×19,5 sur le proton avec arrangement triangulaire) ?
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import mass, u_em, c_j
from simulation.experiments.mass_distribution_factor import (
    pca_sphericity, octant_entropy, radial_uniformity, angular_coverage,
)
from simulation.experiments.mass_gj2_composites import (
    fit_gj2_on_singles, m_tight_composite,
)


def gauss_linking(c1: np.ndarray, c2: np.ndarray) -> float:
    """Nombre d'enlacement de Gauss entre deux polylignes fermées."""
    N1, N2 = len(c1), len(c2)
    e1 = np.roll(c1, -1, axis=0) - c1     # (N1,3) segments
    e2 = np.roll(c2, -1, axis=0) - c2     # (N2,3)
    # Points medians de chaque segment
    m1 = c1 + 0.5 * e1
    m2 = c2 + 0.5 * e2
    # diff[i,j] = m1[i] - m2[j]
    diff = m1[:, None, :] - m2[None, :, :]    # (N1,N2,3)
    r3 = np.linalg.norm(diff, axis=-1) ** 3 + 1e-30
    # cross[i,j] = e1[i] × e2[j]
    cross = np.cross(e1[:, None, :], e2[None, :, :])    # (N1,N2,3)
    # numerator = (diff · cross)
    num = (diff * cross).sum(axis=-1)
    Lk = num.sum() / (4 * np.pi * r3.shape[0])
    # Renormalisation : la somme donne un produit moyen, il faut intégrer
    # ds1 ds2 → multiplier par N1·N2/total ; mais comme on a moyenné par r3.shape,
    # corrigeons :
    Lk = (num / r3).sum() / (4 * np.pi)
    return float(Lk)


def smooth_and_avoid(curves, alpha=0.15, beta=0.5, D_intra=1.0, D_inter=1.5,
                     skip=5):
    """Une itération Pieranski 2 sur un ensemble de composantes."""
    new_curves = []
    for ci, c in enumerate(curves):
        N = len(c)
        # Smoothing Laplacien intra
        p_next = np.roll(c, -1, axis=0)
        p_prev = np.roll(c, 1, axis=0)
        lap = p_next + p_prev - 2 * c
        c2 = c + alpha * lap

        # Self-avoidance intra
        diff = c2[:, None, :] - c2[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        idx = np.arange(N)
        arc = np.abs(idx[:, None] - idx[None, :])
        arc = np.minimum(arc, N - arc)
        far = arc > skip
        too_close = (dist < D_intra) & far
        safe = np.where(dist > 1e-30, dist, 1e-30)
        push_factor = np.where(too_close, beta * (D_intra - dist) / safe, 0.0)
        push = (push_factor[:, :, None] * diff).sum(axis=1)
        c2 = c2 + push

        # Inter-avoidance : depuis toutes les autres composantes
        for cj, other in enumerate(curves):
            if cj == ci:
                continue
            d_inter = c2[:, None, :] - other[None, :, :]
            dist_i = np.linalg.norm(d_inter, axis=-1)
            too_close_i = dist_i < D_inter
            safe_i = np.where(dist_i > 1e-30, dist_i, 1e-30)
            pf = np.where(too_close_i, beta * (D_inter - dist_i) / safe_i, 0.0)
            push_i = (pf[:, :, None] * d_inter).sum(axis=1)
            c2 = c2 + push_i

        new_curves.append(c2)
    return new_curves


def reparametrize(pts, n_target):
    diff = np.roll(pts, -1, axis=0) - pts
    seg = np.linalg.norm(diff, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    L = s[-1] + seg[-1]
    s_new = np.linspace(0, L, n_target, endpoint=False)
    s_full = np.concatenate([s, [L]])
    pts_full = np.vstack([pts, pts[:1]])
    new_pts = np.zeros((n_target, 3))
    for i in range(3):
        new_pts[:, i] = np.interp(s_new, s_full, pts_full[:, i])
    return new_pts


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def construct_hopf_triplet(c_31, c_72a, c_72b, sep=2.5):
    """Place 3 composantes en triplet mutuellement Hopf-linké.

    Stratégie : trois plans orthogonaux, centres décalés pour que chaque
    paire s'interpénètre. Lk(i,j) ≈ 1 pour chaque paire, total ≈ 3.
    """
    # Normalisation : taille caractéristique = 1
    def normalize(c):
        radius = float(np.sqrt(np.mean(np.var(c, axis=0))))
        return (c - c.mean(axis=0)) / radius

    c1 = normalize(c_31)
    c2 = normalize(c_72a)
    c3 = normalize(c_72b)

    # Composante 1 : dans plan XY, centrée à origine
    A = c1
    # Composante 2 : rotée pour passer dans plan XZ, translatée le long de Y
    Ry = rotation_matrix([1, 0, 0], np.pi / 2)
    B = (Ry @ c2.T).T + np.array([sep * 0.7, 0, 0])
    # Composante 3 : rotée pour plan YZ, translatée le long de Z
    Rz = rotation_matrix([0, 1, 0], np.pi / 2)
    C = (Rz @ c3.T).T + np.array([0, sep * 0.7, sep * 0.4])

    return [A, B, C]


def main():
    print("=" * 95)
    print("  PIERANSKI 2 — embedding 3D physique du proton (3_1 + 7_2 + 7_2)")
    print("=" * 95)

    N = 150
    c_31 = knot_curve("3_1", N + 1)[:-1]
    c_72a = knot_curve("7_2", N + 1)[:-1]
    c_72b = knot_curve("7_2", N + 1)[:-1]

    print(f"\n  Construction du triplet Hopf-linké initial …")
    curves = construct_hopf_triplet(c_31, c_72a, c_72b, sep=2.5)

    # Mesure initiale du linking
    Lk12 = gauss_linking(curves[0], curves[1])
    Lk13 = gauss_linking(curves[0], curves[2])
    Lk23 = gauss_linking(curves[1], curves[2])
    print(f"  Lk(3_1, 7_2_a) = {Lk12:+.3f}")
    print(f"  Lk(3_1, 7_2_b) = {Lk13:+.3f}")
    print(f"  Lk(7_2_a, 7_2_b) = {Lk23:+.3f}")
    Lk_tot = Lk12 + Lk13 + Lk23
    print(f"  Lk_total (somme) = {Lk_tot:+.3f}")
    print(f"  Target : Lk_total = +3 (proton)")

    # Joint Pieranski 2 — iter
    print(f"\n  Joint Pieranski 2 : 800 itérations …")
    D_intra = 0.40
    D_inter = 0.60
    for it in range(800):
        curves = smooth_and_avoid(
            curves, alpha=0.10, beta=0.30,
            D_intra=D_intra, D_inter=D_inter, skip=4)
        if (it + 1) % 200 == 0:
            curves = [reparametrize(c, N) for c in curves]
            # Mesure
            Lk12 = gauss_linking(curves[0], curves[1])
            Lk13 = gauss_linking(curves[0], curves[2])
            Lk23 = gauss_linking(curves[1], curves[2])
            Lk_t = Lk12 + Lk13 + Lk23
            print(f"    iter {it+1}: Lk_total = {Lk_t:+.3f} "
                  f"(Lk12,Lk13,Lk23 = {Lk12:+.2f}, {Lk13:+.2f}, {Lk23:+.2f})")

    # État final
    print(f"\n  Embedding final :")
    Lk12 = gauss_linking(curves[0], curves[1])
    Lk13 = gauss_linking(curves[0], curves[2])
    Lk23 = gauss_linking(curves[1], curves[2])
    Lk_tot = Lk12 + Lk13 + Lk23
    print(f"    Lk_total = {Lk_tot:+.3f}")
    L_tot = sum(float(np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1).sum())
                for c in curves)
    print(f"    L_tot (longueur jointe) = {L_tot:.2f}")

    # X_geom sur l'union
    union = np.vstack(curves)
    g_pieranski = {
        "sph": pca_sphericity(union),
        "oct": octant_entropy(union),
        "rad": radial_uniformity(union),
        "ang": angular_coverage(union),
    }
    print(f"\n  X_geom (embedding Pieranski 2) :")
    for k, v in g_pieranski.items():
        print(f"     {k:>4} = {v:.4f}")

    # Comparaison avec embedding triangulaire (mass_gj2_composites)
    from simulation.experiments.mass_gj2_composites import composite_geom_features
    g_triangle = composite_geom_features(["3_1", "7_2", "7_2"])
    print(f"\n  Comparaison avec embedding triangulaire (arrangement naïf) :")
    print(f"     {'feature':>4} {'triangle':>10} {'Pieranski 2':>12}  Δ")
    for k in g_pieranski:
        delta = g_pieranski[k] - g_triangle[k]
        print(f"     {k:>4} {g_triangle[k]:>10.4f} {g_pieranski[k]:>12.4f}  "
              f"{delta:+.4f}")

    # Prédiction GJ2 avec l'embedding Pieranski 2
    beta = fit_gj2_on_singles()
    components = ["3_1", "7_2", "7_2"]
    c_J_tot = sum(c_j(K) for K in components)
    U_EM_tot = sum(u_em(K) for K in components)
    X = np.array([g_pieranski["sph"], g_pieranski["oct"],
                  g_pieranski["rad"], g_pieranski["ang"],
                  np.log(c_J_tot), np.log(U_EM_tot)])
    log_corr = beta[0] + beta[1:] @ X
    m_t = m_tight_composite(components)
    m_pieranski = m_t * np.exp(log_corr)
    m_v4 = mass(components, "hadron")

    print(f"\n  RÉSULTATS :")
    print(f"    m_exp (proton)            = 938.27 MeV")
    print(f"    m_V4 (avec f(3)=0.77)     = {m_v4:.2f} MeV   ({(m_v4/938.27-1)*100:+.2f}%)")
    print(f"    m_GJ2 (embedding triangle) = 18 327 MeV  (×19.5)")
    print(f"    m_GJ2 (embedding Pieranski2) = {m_pieranski:.2f} MeV  "
          f"(×{m_pieranski/m_v4:.2f})")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────")
    ratio = m_pieranski / m_v4
    if 0.5 < ratio < 2.0:
        print(f"  ✓ L'embedding physique CORRIGE la pathologie de GJ2 :")
        print(f"    ratio {ratio:.2f} dans l'ordre de V4. La géométrie 3D")
        print(f"    physique capture donc bien la masse des baryons.")
    elif ratio < 5:
        print(f"  ~ Amélioration partielle (ratio {ratio:.2f} vs ×19,5).")
        print(f"    L'embedding influe, mais d'autres ingrédients manquent")
        print(f"    (probablement le f(n) topologique).")
    else:
        print(f"  ✗ L'embedding physique n'aide presque pas (ratio {ratio:.1f}).")
        print(f"    L'écrantage hadronique f(n) est probablement un effet")
        print(f"    purement topologique (lié à n et au confinement) qu'aucune")
        print(f"    géométrie 3D ne peut reconstruire.")
    print("=" * 95)


if __name__ == "__main__":
    main()
