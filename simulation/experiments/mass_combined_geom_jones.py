r"""
Chantier 1 — formule unifiée : distribution géométrique + topologie de Jones.

On combine les 4 métriques de distribution spatiale (sphericity,
oct_entropy, radial_unif, angular_cov) — qui apportent +55 % sur m_tight
sans information topologique — avec les invariants tirés de V_K(t)
(c_J, U_EM, span, det, braid_index) pour tester si on peut converger
vers la précision V4 (~5 % RMS log).

Question honnête (le piège à éviter)
-------------------------------------
V4 s'écrit par CONSTRUCTION : log m_V4 = log α₀ + log U_EM + p·log c_J.
Donc une régression de log(m_V4/m_tight) sur [log U_EM, log c_J] va
trivialement recouvrer V4 (R² ≈ 1) — c'est tautologique.

Ce qui est pertinent est plutôt :
  1) avec UNE SEULE info de Jones (la plus simple) + 4 features géom.,
     atteint-on V4 ?
  2) quelle info topologique minimale est nécessaire ?

Modèles testés (sur log(m_V4/m_tight), 19 nœuds) :
  M0  : référence (rien)                                     0 params
  G   : 4 features géom. (= M2 précédent)                    5 params
  J1  : log(c_J) seul                                        2 params
  J2  : log(c_J) + log(U_EM)  (= V4 par construction)        3 params
  GJ1 : G + log(c_J)                                         6 params
  GJ2 : G + log(c_J) + log(U_EM)                             7 params
  GJ3 : G + log(c_J) + braid_index                           7 params
  GJ4 : G + log(c_J) + span                                  7 params

Cf. mass_distribution_multi.py et mass_distribution_factor.py.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import mass, u_em, c_j
from theory.jones import (
    jones_polynomial, jones_span, jones_determinant, braid_index,
)
from simulation.experiments.mass_distribution_factor import (
    pca_sphericity, octant_entropy, radial_uniformity, angular_coverage,
)

KNOTS = list(REFERENCE_N50K.keys())
M_E = 0.51099895
A_R, B_R, P_EXP = 3.85, 5.61, 7.42
R_E = A_R * 3 + B_R


def cr(K): return int(K.split("_")[0])


def m_tight(K):
    R = A_R * cr(K) + B_R
    return M_E * (R / R_E) ** P_EXP


def fit_linear(X, y):
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    res = y - y_pred
    rms = float(np.sqrt(np.mean(res ** 2)))
    R2 = float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    return beta, y_pred, rms, R2


def loo_cv_rms(X, y):
    """Leave-one-out cross-validation RMS (test prediction error)."""
    n = len(y)
    res = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[i:i+1], y[i]
        X_tr_aug = np.hstack([np.ones((len(X_tr), 1)), X_tr])
        beta, *_ = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
        X_te_aug = np.hstack([[[1.0]], X_te])
        res.append(float((X_te_aug @ beta).item() - y_te))
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def main():
    print("=" * 92)
    print("  FORMULE UNIFIÉE : distribution géométrique + topologie de Jones")
    print("=" * 92)

    # --- Calcul des features ---
    geom_feats = {}
    jones_feats = {}
    m_V4 = np.zeros(len(KNOTS))
    m_T = np.zeros(len(KNOTS))
    for i, K in enumerate(KNOTS):
        pts = knot_curve(K, 2000)[:-1]
        geom_feats[K] = {
            "sph": pca_sphericity(pts),
            "oct": octant_entropy(pts),
            "rad": radial_uniformity(pts),
            "ang": angular_coverage(pts),
        }
        poly = jones_polynomial(K)
        jones_feats[K] = {
            "log_cJ":   np.log(c_j(K)),
            "log_UEM":  np.log(u_em(K)),
            "span":     float(jones_span(poly)),
            "det":      float(jones_determinant(poly)),
            "b":        float(braid_index(K)),
            "log_det":  np.log(jones_determinant(poly) + 1e-12),
        }
        m_V4[i] = mass(K, "single")
        m_T[i] = m_tight(K)

    y = np.log(m_V4 / m_T)
    rms0 = float(np.sqrt(np.mean(y ** 2)))
    print(f"\n  Cible : log(m_V4/m_tight), N={len(KNOTS)}, RMS₀ = {rms0:.3f}\n")

    def build(geom_names, jones_names):
        cols = list(geom_names) + list(jones_names)
        X = np.array([[
            *(geom_feats[K][n] for n in geom_names),
            *(jones_feats[K][n] for n in jones_names),
        ] for K in KNOTS])
        return X, cols

    geom_all = ["sph", "oct", "rad", "ang"]
    models = [
        ("G    (4 géom)",                geom_all, []),
        ("J1   (log c_J)",               [], ["log_cJ"]),
        ("J2   (log c_J + log U_EM)",    [], ["log_cJ", "log_UEM"]),
        ("GJ1  (G + log c_J)",           geom_all, ["log_cJ"]),
        ("GJ2  (G + log c_J + log U_EM)", geom_all, ["log_cJ", "log_UEM"]),
        ("GJ3  (G + log c_J + b)",       geom_all, ["log_cJ", "b"]),
        ("GJ4  (G + log c_J + span)",    geom_all, ["log_cJ", "span"]),
        ("GJ5  (G + log c_J + log det)", geom_all, ["log_cJ", "log_det"]),
    ]

    print(f"  {'modèle':<33} {'params':>7} {'RMS fit':>9} {'R²':>7} "
          f"{'gain vs M0':>11} {'RMS LOO':>9}")
    print("  " + "-" * 84)

    # M0 baseline
    print(f"  {'M0   (référence)':<33} {0:>7} {rms0:>9.3f} {0.0:>7.3f} "
          f"{0.0:>+10.1f}% {rms0:>9.3f}")

    results = {}
    for name, gnames, jnames in models:
        X, cols = build(gnames, jnames)
        beta, ypred, rms, R2 = fit_linear(X, y)
        loo = loo_cv_rms(X, y)
        gain = (rms0 - rms) / rms0 * 100
        results[name] = (beta, cols, rms, R2, loo)
        print(f"  {name:<33} {len(cols)+1:>7} {rms:>9.3f} {R2:>7.3f} "
              f"{gain:>+10.1f}% {loo:>9.3f}")

    # Détail des coefficients pour les modèles intéressants
    print(f"\n  ─── Coefficients du modèle GJ1 (G + log c_J) :")
    beta, cols, _, _, _ = results["GJ1  (G + log c_J)"]
    print(f"     β₀ = {beta[0]:+.3f}")
    for i, c in enumerate(cols):
        print(f"     β({c:>10}) = {beta[i+1]:+.3f}")

    print(f"\n  ─── Coefficients du modèle GJ2 (G + log c_J + log U_EM) :")
    beta, cols, _, _, _ = results["GJ2  (G + log c_J + log U_EM)"]
    print(f"     β₀ = {beta[0]:+.3f}")
    for i, c in enumerate(cols):
        print(f"     β({c:>10}) = {beta[i+1]:+.3f}")

    print(f"\n  ─── Coefficients du modèle J2 (V4 pur) :")
    beta, cols, _, _, _ = results["J2   (log c_J + log U_EM)"]
    print(f"     β₀ = {beta[0]:+.3f}")
    for i, c in enumerate(cols):
        print(f"     β({c:>10}) = {beta[i+1]:+.3f}")
    print(f"  (Vérification : J2 doit ≈ recouvrir V4, donc RMS très petit.)")

    # Prédictions GJ1 par nœud
    print(f"\n  Prédictions GJ1 (G + log c_J) par nœud :")
    print(f"  {'K':>5} {'m_V4':>11} {'m_tight':>11} {'m_GJ1':>11} "
          f"{'log Δ':>9}")
    beta, cols, _, _, _ = results["GJ1  (G + log c_J)"]
    X, _ = build(geom_all, ["log_cJ"])
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    ypred = X_aug @ beta
    m_GJ1 = m_T * np.exp(ypred)
    for i, K in enumerate(KNOTS):
        log_d = float(np.log(m_GJ1[i] / m_V4[i]))
        flag = " " if abs(log_d) < 0.1 else "·"
        print(f"  {K:>5} {m_V4[i]:>11.3f} {m_T[i]:>11.3f} "
              f"{m_GJ1[i]:>11.3f} {log_d:>+9.3f} {flag}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ──────────────────────────────────────")
    rms_G = results["G    (4 géom)"][2]
    rms_J1 = results["J1   (log c_J)"][2]
    rms_J2 = results["J2   (log c_J + log U_EM)"][2]
    rms_GJ1 = results["GJ1  (G + log c_J)"][2]
    rms_GJ2 = results["GJ2  (G + log c_J + log U_EM)"][2]
    loo_GJ1 = results["GJ1  (G + log c_J)"][4]
    print(f"  • Géom. seule       (5 params) → RMS {rms_G:.3f}")
    print(f"  • Jones seul (c_J)  (2 params) → RMS {rms_J1:.3f}")
    print(f"  • V4 pur (J2)       (3 params) → RMS {rms_J2:.3f}  ← référence cible")
    print(f"  • GJ1 = G + log c_J (6 params) → RMS {rms_GJ1:.3f}  LOO {loo_GJ1:.3f}")
    print(f"  • GJ2 = G + V4 pur  (7 params) → RMS {rms_GJ2:.3f}")
    print(f"  ")
    if rms_GJ1 < rms_J2 * 1.3:
        print(f"  ✓ GJ1 (1 seule info Jones + géom.) approche V4 (J2) à ±30 %.")
    else:
        print(f"  ✗ GJ1 reste loin de V4 — la géométrie ne remplace pas U_EM.")
    print(f"  ")
    print(f"  La colonne 'RMS LOO' est la cross-validation leave-one-out :")
    print(f"  si LOO >> RMS_fit, le modèle overfit.")
    print("=" * 92)


if __name__ == "__main__":
    main()
