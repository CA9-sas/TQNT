r"""
Chantier 1 — application de GJ2 aux composites (proton, neutron, π±, etc.).

GJ2 (cf. mass_combined_geom_jones.py) bat V4 strictement de 12 % en
leave-one-out sur les 19 nœuds individuels. On teste maintenant si cette
amélioration se transfère aux entrelacs physiques.

Protocole :
  1. Refit GJ2 sur les 19 singles → coefficients (β₀, β_sph, β_oct,
     β_rad, β_ang, β_logcJ, β_logUEM).
  2. Pour chaque composite, construire un embedding 3D (triangle
     équilatéral, écart 3·R_3D_carac), calculer X_geom sur l'UNION
     des courbes, et X_jones = (log c_J_tot, log U_EM_tot).
  3. Calculer m_GJ2 = m_tight_composite · exp(β·X_composite).
  4. Comparer à m_V4 et m_expérimental.

Composites testés :
  - p (3_1+7_2+7_2, hadron)
  - n (idem, autre Lk)
  - π+ (5_2+7_2, hadron, mésons à 2 composantes)
  - Higgs (6_1+7_4+8_3, EW)
  - W± (4_1+7_3+8_3, EW)

Note : la formule GJ2 ne contient pas le facteur f(n) d'écrantage
hadronique de V4 ; cette information n'est pas dans les singles. La
prédiction GJ2 va donc surestimer les baryons d'un facteur ~(1/0,77)^p.
Le but est de SÉPARER la contribution f(n) (correction d'écrantage)
de la contribution géométrique pure.

Cf. mass_combined_geom_jones.py.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import mass, u_em, c_j
from theory.jones import jones_polynomial
from simulation.experiments.mass_distribution_factor import (
    pca_sphericity, octant_entropy, radial_uniformity, angular_coverage,
)

KNOTS = list(REFERENCE_N50K.keys())
M_E = 0.51099895
A_R, B_R, P_EXP = 3.85, 5.61, 7.42
R_E = A_R * 3 + B_R


def cr(K): return int(K.split("_")[0])


def m_tight_single(K):
    R = A_R * cr(K) + B_R
    return M_E * (R / R_E) ** P_EXP


def m_tight_composite(components):
    """m_tight pour un composite : R_tot = ΣR(K_i)."""
    R_tot = sum(A_R * cr(K) + B_R for K in components)
    return M_E * (R_tot / R_E) ** P_EXP


def composite_geom_features(components, sep_factor=3.0):
    """Place les composantes en arrangement triangulaire ; X_geom sur l'union."""
    n = len(components)
    curves = []
    for i, K in enumerate(components):
        pts = knot_curve(K, 1500)[:-1]
        if n == 1:
            offset = np.array([0., 0., 0.])
        else:
            # placement en n-gone régulier dans le plan xy
            theta = 2 * np.pi * i / n
            # rayon en fonction de la "taille" PCA moyenne
            size = float(np.sqrt(np.mean(np.var(pts, axis=0))))
            offset = sep_factor * size * np.array(
                [np.cos(theta), np.sin(theta), 0.])
        curves.append(pts + offset)
    union = np.vstack(curves)
    return {
        "sph": pca_sphericity(union),
        "oct": octant_entropy(union),
        "rad": radial_uniformity(union),
        "ang": angular_coverage(union),
    }


def fit_gj2_on_singles():
    """Refit du modèle GJ2 sur les 19 singles → coefficients."""
    X_list, y_list = [], []
    for K in KNOTS:
        pts = knot_curve(K, 2000)[:-1]
        feats = [
            pca_sphericity(pts),
            octant_entropy(pts),
            radial_uniformity(pts),
            angular_coverage(pts),
            np.log(c_j(K)),
            np.log(u_em(K)),
        ]
        X_list.append(feats)
        y_list.append(np.log(mass(K, "single") / m_tight_single(K)))
    X = np.array(X_list)
    y = np.array(y_list)
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta  # [β₀, β_sph, β_oct, β_rad, β_ang, β_logcJ, β_logUEM]


def predict_gj2_composite(components, beta, sep_factor=3.0):
    """Prédit m via GJ2 pour un composite."""
    g = composite_geom_features(components, sep_factor=sep_factor)
    # c_J_tot et U_EM_tot : somme des composantes
    c_J_tot = sum(c_j(K) for K in components)
    U_EM_tot = sum(u_em(K) for K in components)
    X = np.array([g["sph"], g["oct"], g["rad"], g["ang"],
                  np.log(c_J_tot), np.log(U_EM_tot)])
    log_correction = beta[0] + beta[1:] @ X
    m_t = m_tight_composite(components)
    return m_t * np.exp(log_correction), g, c_J_tot, U_EM_tot


def main():
    print("=" * 95)
    print("  GJ2 SUR COMPOSITES — test du correctif géométrique sur entrelacs réels")
    print("=" * 95)

    beta = fit_gj2_on_singles()
    print(f"\n  Coefficients GJ2 (fit sur 19 singles) :")
    names = ["β₀", "β_sph", "β_oct", "β_rad", "β_ang", "β_logcJ", "β_logUEM"]
    for n, b in zip(names, beta):
        print(f"     {n:>10} = {b:+.4f}")

    # Composites à tester
    composites = [
        ("p (3_1+7_2+7_2)",      ["3_1", "7_2", "7_2"], "hadron",  938.27),
        ("n (3_1+7_2+7_2)",      ["3_1", "7_2", "7_2"], "hadron",  939.57),
        ("π+ (5_2+7_2)",         ["5_2", "7_2"],        "hadron",  139.57),
        ("Higgs (6_1+7_4+8_3)",  ["6_1", "7_4", "8_3"], "ew",      125100),
        ("W± (4_1+7_3+8_3)",     ["4_1", "7_3", "8_3"], "ew",       80377),
        ("Z⁰ (5_1+7_2+8_3)",     ["5_1", "7_2", "8_3"], "ew",       91188),
        ("top (7_4+7_6+8_3)",    ["7_4", "7_6", "8_3"], "ew",      172690),
    ]

    print(f"\n  {'composite':<24} {'sect.':>7} {'m_exp':>11} "
          f"{'m_V4':>11} {'err V4':>7} {'m_GJ2':>11} {'err GJ2':>8} "
          f"{'ratio GJ2/V4':>12}")
    print("  " + "-" * 96)

    for label, comp, sector, m_exp in composites:
        m_v4 = mass(comp, sector)
        m_pred, g, c_J_tot, U_EM_tot = predict_gj2_composite(comp, beta)
        err_v4 = (m_v4 - m_exp) / m_exp * 100
        err_gj2 = (m_pred - m_exp) / m_exp * 100
        ratio = m_pred / m_v4
        print(f"  {label:<24} {sector:>7} {m_exp:>11.2f} {m_v4:>11.2f} "
              f"{err_v4:>+6.2f}% {m_pred:>11.2f} {err_gj2:>+7.1f}% "
              f"{ratio:>12.3f}")

    # Analyse : décomposer la prédiction GJ2 pour le proton
    print(f"\n  ─── DÉCOMPOSITION pour le proton :")
    comp = ["3_1", "7_2", "7_2"]
    g = composite_geom_features(comp)
    c_J_tot = sum(c_j(K) for K in comp)
    U_EM_tot = sum(u_em(K) for K in comp)
    m_t = m_tight_composite(comp)
    print(f"     m_tight (R_tot = {A_R}·17 + {B_R}·3 = {A_R*17+B_R*3:.1f}) = {m_t:.2f} MeV")
    print(f"     X_geom (composite triangle) :")
    for k, v in g.items():
        print(f"        {k:>4} = {v:.4f}")
    print(f"     X_jones :")
    print(f"        log c_J_tot = log({c_J_tot:.2f}) = {np.log(c_J_tot):.3f}")
    print(f"        log U_EM_tot = log({U_EM_tot:.2f}) = {np.log(U_EM_tot):.3f}")

    contribs = [
        ("intercept β₀", beta[0]),
        ("sph",  beta[1] * g["sph"]),
        ("oct",  beta[2] * g["oct"]),
        ("rad",  beta[3] * g["rad"]),
        ("ang",  beta[4] * g["ang"]),
        ("log c_J",  beta[5] * np.log(c_J_tot)),
        ("log U_EM", beta[6] * np.log(U_EM_tot)),
    ]
    print(f"     Contributions à log(correction) :")
    total = 0
    for n, c in contribs:
        total += c
        print(f"        {n:>10} : {c:+.4f}")
    print(f"        {'TOTAL':>10} : {total:+.4f}  → correction = {np.exp(total):.3f}")
    print(f"     m_GJ2 = m_tight × correction = {m_t:.1f} × {np.exp(total):.3f} = "
          f"{m_t*np.exp(total):.1f} MeV")
    print(f"     m_V4 (avec f(3)=0,77) = {mass(comp,'hadron'):.1f} MeV")
    print(f"     m_exp = 938,27 MeV")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────")
    print(f"  GJ2 (fit sur singles) ne contient PAS le facteur d'écrantage")
    print(f"  hadronique f(n). Sur les baryons, on attend donc une")
    print(f"  surestimation d'un facteur (1/0,77)^p ≈ 2,5 si la géométrie")
    print(f"  composite ne compense pas.")
    print(f"  ")
    print(f"  Lecture du ratio GJ2/V4 :")
    print(f"    ratio ≈ 2,5 : la géométrie composite n'apporte aucune")
    print(f"                  correction, GJ2 ignore juste f(n) → écrantage")
    print(f"                  est un effet TOPOLOGIQUE, pas géométrique.")
    print(f"    ratio < 1   : la géométrie composite (PCA non-sphérique,")
    print(f"                  oct_entropy spécifique) compense partiellement.")
    print(f"    ratio > 5   : la géométrie composite AMPLIFIE l'erreur,")
    print(f"                  les features géom. ne sont pas extrapolables.")
    print(f"  ")
    print(f"  Pour les EW (top, Z, W, H), V4 atteint déjà <0,5 % : GJ2 sans")
    print(f"  réajustement va dégrader (correction non-nulle vers le mauvais")
    print(f"  côté).")
    print("=" * 95)


if __name__ == "__main__":
    main()
