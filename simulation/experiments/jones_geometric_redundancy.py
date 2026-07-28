r"""
Chantier 1 — analyse théorique : pourquoi log(c_J) est redondant avec
oct_entropy dans GJ2 ?

Observation surprenante (mass_combined_geom_jones.py) :
  - J1 seul (log c_J) : RMS 0.824, R² = 0.12
  - G seul (4 géom)   : RMS 0.463, R² = 0.72
  - GJ1 = G + log c_J : RMS 0.455 ≈ G seul, et β(log c_J) tombe de
    -1.85 (J2 pur) à +0.31 (presque nul).

C'est-à-dire : sachant les 4 features géométriques, log c_J n'apporte
PAS d'information sur log(m_V4/m_tight). Pourquoi ?

Hypothèses à tester :
  H1. log c_J corrèle directement avec oct_entropy (ou un autre feature
      géom.) sur le catalogue.
  H2. log c_J corrèle avec une combinaison linéaire des 4 features.
  H3. C'est un artefact du petit échantillon (19 nœuds).

On calcule :
  - corrélations Pearson(log c_J, X_i) pour i ∈ {sph, oct, rad, ang}
  - R² de régression log c_J ~ {X_i, ..., X_4}
  - corrélation Pearson(log U_EM, X_i) — pour comparaison
  - log c_J vs autres invariants topologiques (cr, span, det, b)
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import u_em, c_j
from theory.jones import jones_polynomial, jones_span, jones_determinant, braid_index
from simulation.experiments.mass_distribution_factor import (
    pca_sphericity, octant_entropy, radial_uniformity, angular_coverage,
)

KNOTS = list(REFERENCE_N50K.keys())


def cr(K): return int(K.split("_")[0])


def fit_R2(X, y):
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    print("=" * 88)
    print("  REDONDANCE log(c_J) ↔ oct_entropy — analyse théorique")
    print("=" * 88)

    # Calcule tous les indicateurs sur les 19 nœuds
    rows = []
    for K in KNOTS:
        pts = knot_curve(K, 2000)[:-1]
        poly = jones_polynomial(K)
        rows.append({
            "K":          K,
            "cr":         cr(K),
            "sph":        pca_sphericity(pts),
            "oct":        octant_entropy(pts),
            "rad":        radial_uniformity(pts),
            "ang":        angular_coverage(pts),
            "log_cJ":     np.log(c_j(K)),
            "log_UEM":    np.log(u_em(K)),
            "span":       jones_span(poly),
            "det":        jones_determinant(poly),
            "b":          braid_index(K),
        })

    keys = list(rows[0].keys())[1:]
    M = np.array([[r[k] for k in keys] for r in rows])
    idx = {k: i for i, k in enumerate(keys)}

    log_cJ = M[:, idx["log_cJ"]]
    log_UEM = M[:, idx["log_UEM"]]
    geom = ["sph", "oct", "rad", "ang"]

    # --- 1. Corrélations Pearson individuelles ---
    print(f"\n  ── Corrélations Pearson(log c_J, X) sur 19 nœuds ──")
    print(f"  {'X':>12} {'r(log_cJ, X)':>16} {'r(log_UEM, X)':>16} "
          f"{'r(cr, X)':>11}")
    print("  " + "-" * 60)
    for k in geom + ["cr", "span", "det", "b"]:
        x = M[:, idx[k]]
        r_cJ  = np.corrcoef(log_cJ, x)[0, 1]
        r_UEM = np.corrcoef(log_UEM, x)[0, 1]
        r_cr  = np.corrcoef(M[:, idx["cr"]], x)[0, 1]
        print(f"  {k:>12} {r_cJ:>+16.3f} {r_UEM:>+16.3f} {r_cr:>+11.3f}")

    # --- 2. R² de régression : log c_J ~ {features géométriques} ---
    print(f"\n  ── Régression log c_J ~ géométrie ──")
    X_geom = np.array([[r[k] for k in geom] for r in rows])
    R2_full = fit_R2(X_geom, log_cJ)
    print(f"     R² (4 features géom.)            = {R2_full:.3f}")
    for k in geom:
        X1 = M[:, idx[k]:idx[k]+1]
        print(f"     R² ({k:>4} seul)                  = {fit_R2(X1, log_cJ):.3f}")
    # avec cr ajouté
    X_geom_cr = np.hstack([X_geom, M[:, [idx["cr"]]]])
    print(f"     R² (4 géom. + cr)                = "
          f"{fit_R2(X_geom_cr, log_cJ):.3f}")

    # --- 3. Idem pour log U_EM (qui n'est PAS redondant) ---
    print(f"\n  ── Régression log U_EM ~ géométrie (comparaison) ──")
    R2_UEM = fit_R2(X_geom, log_UEM)
    print(f"     R² (4 features géom.)            = {R2_UEM:.3f}")
    for k in geom:
        X1 = M[:, idx[k]:idx[k]+1]
        print(f"     R² ({k:>4} seul)                  = {fit_R2(X1, log_UEM):.3f}")
    R2_UEM_cr = fit_R2(X_geom_cr, log_UEM)
    print(f"     R² (4 géom. + cr)                = {R2_UEM_cr:.3f}")

    # --- 4. log c_J vs cr (réponse théorique potentielle) ---
    cr_arr = M[:, idx["cr"]]
    r_lc_cr = np.corrcoef(log_cJ, cr_arr)[0, 1]
    print(f"\n  ── log c_J vs cr (théorie ?) ──")
    print(f"     Pearson(log c_J, cr)             = {r_lc_cr:+.3f}")
    # fit log c_J ~ a + b·cr
    X1 = cr_arr.reshape(-1, 1)
    X_aug = np.hstack([np.ones((len(X1), 1)), X1])
    beta, *_ = np.linalg.lstsq(X_aug, log_cJ, rcond=None)
    pred = X_aug @ beta
    resid = log_cJ - pred
    print(f"     log c_J ≈ {beta[0]:+.3f} {beta[1]:+.3f}·cr  "
          f"(R² = {fit_R2(X1, log_cJ):.3f})")
    print(f"     RMS résiduel = {np.std(resid):.3f}")
    print(f"     Théorie : pour les nœuds alternés, span(V_K) = cr,")
    print(f"     donc c_J ≈ cr + γ·det → log c_J ≈ log(cr) + petite correction.")

    # --- 5. oct_entropy vs cr ---
    oct_arr = M[:, idx["oct"]]
    r_oct_cr = np.corrcoef(oct_arr, cr_arr)[0, 1]
    print(f"\n  ── oct_entropy vs cr ──")
    print(f"     Pearson(oct, cr)                 = {r_oct_cr:+.3f}")
    X1 = cr_arr.reshape(-1, 1)
    print(f"     R² (oct ~ cr)                    = {fit_R2(X1, oct_arr):.3f}")

    # --- 6. Conclusion empirique ---
    r_oct_lc = np.corrcoef(log_cJ, oct_arr)[0, 1]
    print(f"\n  ── DIAGNOSTIC FINAL ──")
    print(f"     Pearson(log c_J, oct_entropy)    = {r_oct_lc:+.3f}")
    if abs(r_oct_lc) > 0.7:
        print(f"     → CORRÉLATION FORTE : oct_entropy capture déjà l'info de c_J")
    elif R2_full > 0.6:
        print(f"     → COMBINAISON DES 4 features géom. capture log c_J (R²={R2_full:.2f})")
    else:
        print(f"     → La redondance dans GJ1 n'est PAS due à une corrélation")
        print(f"       simple : c'est un effet de partage de variance dans la")
        print(f"       régression linéaire à 19 points.")

    # --- 7. Listing des nœuds ---
    print(f"\n  ── Données brutes (19 nœuds) ──")
    print(f"  {'K':>5} {'cr':>3} {'log_cJ':>8} {'oct':>7} {'sph':>7} "
          f"{'rad':>7} {'ang':>7} {'log_UEM':>9}")
    for r in rows:
        print(f"  {r['K']:>5} {r['cr']:>3} {r['log_cJ']:>8.3f} "
              f"{r['oct']:>7.3f} {r['sph']:>7.3f} "
              f"{r['rad']:>7.3f} {r['ang']:>7.3f} {r['log_UEM']:>9.3f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
