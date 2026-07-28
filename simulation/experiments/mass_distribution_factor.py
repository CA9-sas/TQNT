r"""
Chantier 1 — facteur de distribution des brins dans la sphère.

Insight utilisateur : « plus les brins/noeuds sont bien distribués dans
la sphère de la particule, plus la masse est moins pesante pour
l'espace ». On code plusieurs métriques d'homogénéité spatiale 3D et
on teste si elles capturent l'écart résiduel de la formule tightness.

Métriques implémentées :

  1. SPHÉRICITÉ PCA : S = 3λ₃/(λ₁+λ₂+λ₃) ∈ [0,1]
     1 = sphère uniforme, 0 = ligne (concentré sur un axe)

  2. ENTROPIE OCTANTAIRE : Shannon entropy des occupations des 8 octants
     1 = pts dans tous les octants également, 0 = pts dans 1 octant

  3. UNIFORMITÉ RADIALE : variance des occupations de coquilles
     concentriques de volume égal
     1 = remplissage uniforme du volume, 0 = concentré sur 1 coquille

  4. UNIFORMITÉ ANGULAIRE : couverture des directions (θ,φ) depuis
     le centroid, mesurée par la fraction de directions « voyant » du
     brin dans une grille sphérique.

Hypothèse : la masse a un facteur correctif

   m_eff = m_naive · (1 + α·(1 - H_combined))^p

où H_combined est une combinaison des 4 métriques. Plus l'embedding est
homogène (H grand), plus m_eff est petit.

Cf. docs/brouillon_hamiltonien_multipart.tex (insight utilisateur).
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import c_j, mass

KNOTS = list(REFERENCE_N50K.keys())


def bounding_sphere(pts: np.ndarray):
    """Centroid + rayon max."""
    c = pts.mean(axis=0)
    return c, float(np.linalg.norm(pts - c, axis=1).max())


def pca_sphericity(pts: np.ndarray) -> float:
    c = pts.mean(axis=0)
    cov = ((pts - c).T @ (pts - c)) / len(pts)
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if lam.sum() <= 0:
        return 0.0
    return float(3 * lam[2] / lam.sum())


def octant_entropy(pts: np.ndarray) -> float:
    """Entropie Shannon normalisée de l'occupation des 8 octants."""
    c = pts.mean(axis=0)
    p_c = pts - c
    oct_id = ((p_c[:, 0] > 0).astype(int) * 4 +
              (p_c[:, 1] > 0).astype(int) * 2 +
              (p_c[:, 2] > 0).astype(int))
    counts = np.bincount(oct_id, minlength=8) / len(pts)
    counts = counts[counts > 0]
    S = -np.sum(counts * np.log(counts))
    return float(S / np.log(8))


def radial_uniformity(pts: np.ndarray, n_shells: int = 8) -> float:
    """Uniformité de l'occupation de coquilles concentriques de volume égal.
    1 = uniforme, 0 = concentré sur une seule coquille."""
    c, r_max = bounding_sphere(pts)
    r = np.linalg.norm(pts - c, axis=1)
    # Coquilles à volume égal : r ∝ shell^(1/3)
    edges = r_max * np.linspace(0, 1, n_shells + 1) ** (1.0 / 3.0)
    counts, _ = np.histogram(r, bins=edges)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    # Distance L1 à la distribution uniforme
    uniform = 1.0 / n_shells
    deviation = np.abs(p - uniform).sum() / 2.0  # ∈ [0,1]
    return float(1.0 - deviation)


def angular_coverage(pts: np.ndarray, n_theta: int = 8, n_phi: int = 16) -> float:
    """Fraction des directions (θ,φ) de la sphère unité qui contiennent au
    moins un point du brin (en projection radiale)."""
    c = pts.mean(axis=0)
    p_c = pts - c
    r = np.linalg.norm(p_c, axis=1)
    valid = r > 0
    p_c = p_c[valid]
    r = r[valid]
    theta = np.arccos(p_c[:, 2] / r)             # [0, π]
    phi = np.arctan2(p_c[:, 1], p_c[:, 0])        # [-π, π]
    # Binner
    theta_bin = (theta / np.pi * n_theta).astype(int).clip(0, n_theta - 1)
    phi_bin = ((phi + np.pi) / (2 * np.pi) * n_phi).astype(int).clip(0, n_phi - 1)
    grid = np.zeros((n_theta, n_phi), dtype=bool)
    grid[theta_bin, phi_bin] = True
    return float(grid.sum() / (n_theta * n_phi))


def combined_H(pts: np.ndarray) -> dict:
    """Toutes les métriques + moyenne pondérée."""
    sph = pca_sphericity(pts)
    oct_S = octant_entropy(pts)
    rad = radial_uniformity(pts)
    ang = angular_coverage(pts)
    H_mean = (sph + oct_S + rad + ang) / 4
    return {"sphericity": sph, "oct_entropy": oct_S,
            "radial_unif": rad, "angular_cov": ang,
            "H_mean": H_mean}


# ============================================================
# Composites : placement n-gone + distribution boucles
# ============================================================
def composite_curves(config_str: str, radius: float = 4.0, N: int = 1500):
    """Pour 'K1+K2+...' : place les composantes aux sommets d'un n-gone
    dans le plan xy, chacune centrée et orientée z."""
    knots = config_str.split("+")
    n = len(knots)
    if n == 1:
        return [knot_curve(knots[0], N)[:-1]]
    curves = []
    for i, K in enumerate(knots):
        angle = 2 * np.pi * i / n
        translation = np.array([radius * np.cos(angle),
                                radius * np.sin(angle), 0.0])
        c = knot_curve(K, N)[:-1]
        c = c - c.mean(axis=0)         # centrer
        curves.append(c + translation)
    return curves


def H_loops_distribution(loop_centers: np.ndarray) -> float:
    """Uniformité de la distribution des CENTRES de boucles dans l'espace.
    1 = centres équidistants (n-gone parfait), 0 = colinéaires/coïncidents."""
    n = len(loop_centers)
    if n < 2:
        return 1.0
    # CV des distances inter-centres
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(np.linalg.norm(loop_centers[i] - loop_centers[j]))
    d = np.array(d)
    if d.mean() < 1e-10:
        return 0.0
    cv = d.std() / d.mean()
    return float(np.exp(-cv))    # 1 si cv=0 (uniforme), → 0 si cv grand


def composite_homogeneity(config_str: str) -> dict:
    """Calcule H_brins (sur l'union des points) + H_boucles (centres)."""
    curves = composite_curves(config_str)
    all_pts = np.vstack(curves)
    H_strands = combined_H(all_pts)
    centers = np.array([c.mean(axis=0) for c in curves])
    H_loops = H_loops_distribution(centers)
    return {
        "H_strands_mean": H_strands["H_mean"],
        "H_strands_sphericity": H_strands["sphericity"],
        "H_loops": H_loops,
        "H_combined": (H_strands["H_mean"] + H_loops) / 2,
        "n_components": len(curves),
    }


def main():
    print("=" * 90)
    print("  FACTEUR DE DISTRIBUTION : BRINS + BOUCLES dans la sphère")
    print("=" * 90)

    N = 2000

    # ---------- (1) Calcul des métriques par nœud ----------------------
    print(f"\n  (1) Métriques d'homogénéité pour les 19 nœuds (N={N}) :")
    print(f"  {'K':>5} {'rayon':>7} {'sphericity':>11} {'oct_S':>7} "
          f"{'rad_U':>7} {'ang_cov':>8} {'H_mean':>8}")
    print("  " + "-" * 65)
    data = {}
    for K in KNOTS:
        pts = knot_curve(K, N)[:-1]
        _, r = bounding_sphere(pts)
        H = combined_H(pts)
        data[K] = (r, H)
        print(f"  {K:>5} {r:>7.2f} {H['sphericity']:>11.3f} "
              f"{H['oct_entropy']:>7.3f} {H['radial_unif']:>7.3f} "
              f"{H['angular_cov']:>8.3f} {H['H_mean']:>8.3f}")

    # ---------- (2) Corrélations avec masses V4 -----------------------
    print(f"\n  (2) Corrélations avec m_V4 (single-knot) :")
    H_arr = {k: np.array([data[K][1][k] for K in KNOTS])
             for k in ("sphericity", "oct_entropy", "radial_unif",
                       "angular_cov", "H_mean")}
    m_V4 = np.array([mass(K, "single") for K in KNOTS])
    log_m = np.log(m_V4)
    for key, vals in H_arr.items():
        r_corr = np.corrcoef(vals, log_m)[0, 1]
        print(f"    log(m_V4) vs {key:>15} : r = {r_corr:+.3f}")

    # ---------- (3) Corrélation avec écart tightness vs V4 ------------
    print(f"\n  (3) Corrélation avec l'écart log(m_tightness / m_V4) :")
    # Formule tightness à recompiler
    A_R, B_R, P_EXP = 3.85, 5.61, 7.42
    R_E = A_R * 3 + B_R
    M_E = 0.51099895
    m_tight = np.array([M_E * (1.0 * (A_R * int(K.split("_")[0]) + B_R) / R_E) ** P_EXP
                        for K in KNOTS])
    delta = np.log(m_tight / m_V4)
    print(f"    écart tightness/V4 : moyenne {delta.mean():+.3f}, "
          f"std {delta.std():.3f}")
    for key, vals in H_arr.items():
        r_corr = np.corrcoef(vals, delta)[0, 1]
        print(f"    delta vs {key:>15} : r = {r_corr:+.3f}")

    # ---------- (4) Test correctif ------------------------------------
    print(f"\n  (4) Tester correctif m_eff = m_tight · (1 + α·(1-H_mean))^q :")
    print(f"  {'α':>6} {'q':>6} {'RMS log vs V4':>16}")
    H_mean_arr = H_arr["H_mean"]
    best = (np.inf, None, None)
    for alpha in np.linspace(-1.0, 2.0, 31):
        for q in np.linspace(-3, 3, 13):
            try:
                correction = (1 + alpha * (1 - H_mean_arr)) ** q
                m_corr = m_tight * correction
                ratio = m_corr / m_V4
                if np.any(ratio <= 0) or np.any(np.isnan(ratio)):
                    continue
                rms = np.sqrt(np.mean(np.log(ratio) ** 2))
                if rms < best[0]:
                    best = (rms, alpha, q)
            except Exception:
                continue
    if best[1] is not None:
        print(f"    BEST : α = {best[1]:.2f}, q = {best[2]:.1f}, RMS = {best[0]:.3f}")
    rms_no_corr = np.sqrt(np.mean(np.log(m_tight / m_V4) ** 2))
    print(f"    Sans correctif : RMS = {rms_no_corr:.3f}")
    if best[1] is not None and best[0] < rms_no_corr:
        gain = (rms_no_corr - best[0]) / rms_no_corr * 100
        print(f"    Gain : {gain:.1f} %")

    # ---------- (5) Test sur composites (brins + boucles) -----------
    print(f"\n  (5) Composites : brins + boucles (arrangement n-gone) :")
    composites = [
        ("mu-",     105.66, "3_1+6_1"),
        ("pi0",     134.98, "8_1"),
        ("eta",     547.86, "3_1+4_1+5_1"),
        ("Lambda",  1115.7, "4_1+7_4"),
        ("Sigma+",  1189.4, "6_2+7_2"),
        ("p",       938.27, "3_1+7_2+7_2"),
        ("n",       939.57, "3_1+7_2+7_2"),
        ("Omega-",  1672.,  "5_2+5_2+7_2"),
        ("Delta",   1232.,  "3_1+5_2+6_2"),
        ("W",       80377., "4_1+7_3+8_3"),
        ("H",       125250.,"6_1+7_4+8_3"),
    ]
    print(f"  {'particule':>10} {'config':>16} {'n':>2} {'H_brins':>9} "
          f"{'H_boucles':>11} {'H_combined':>11} {'m_exp':>10}")
    for label, m_exp, cfg in composites:
        H = composite_homogeneity(cfg)
        print(f"  {label:>10} {cfg:>16} {H['n_components']:>2} "
              f"{H['H_strands_mean']:>9.3f} {H['H_loops']:>11.3f} "
              f"{H['H_combined']:>11.3f} {m_exp:>10.2f}")

    # ---------- Lecture ------------------------------------------
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────────────────")
    sph_arr = H_arr["sphericity"]
    print(f"  Sphéricité catalogue : moyenne {sph_arr.mean():.3f}, "
          f"min {sph_arr.min():.3f}, max {sph_arr.max():.3f}.")
    print(f"  La plupart des courbes sont 'planaires' (sphéricité ~ 0,3) ;")
    print(f"  6_3 et 8_3 sont les plus 'axiaux' (sphéricité ~ 0,5).")
    print(f"  ")
    correlations = {k: np.corrcoef(H_arr[k], delta)[0, 1] for k in H_arr}
    best_corr = max(correlations, key=lambda k: abs(correlations[k]))
    print(f"  Meilleure corrélation distribution vs écart : {best_corr}")
    print(f"  (r = {correlations[best_corr]:+.3f})")
    if abs(correlations[best_corr]) > 0.5:
        print(f"  ⇒ corrélation modérée à forte : l'idée de l'utilisateur a un")
        print(f"     soutien numérique. À raffiner.")
    elif abs(correlations[best_corr]) > 0.3:
        print(f"  ⇒ corrélation faible mais non nulle : effet présent mais")
        print(f"     pas dominant.")
    else:
        print(f"  ⇒ corrélation faible : la distribution dans la sphère n'est")
        print(f"     pas la cause principale de l'écart résiduel.")
    print("=" * 90)


if __name__ == "__main__":
    main()
