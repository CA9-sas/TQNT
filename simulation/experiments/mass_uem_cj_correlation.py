r"""
Chantier 1 — corrélation U_EM ↔ c_J et décomposition de V4.

Hypothèse : si U_EM(K) ≈ K_const · c_J(K)^q (loi de puissance), alors la
formule V4 m = α₀·U_EM·c_J^3,55 peut se réécrire

    m ≈ α₀·K_const · c_J^(q+3,55)  =  α₀_eff · c_J^p_total

avec p_total = q + 3,55. L'exposant « 3,55 » apparent dans V4 serait alors
le RÉSIDU d'une factorisation arbitraire ; l'exposant FONDAMENTAL serait
p_total. Si p_total a une expression analytique simple (3+1/2 + corrections,
ou liée à la charge centrale CS), ça éclaire l'origine du fit V4.

On regarde aussi :
  - les outliers (en particulier 8_3 qui a U_EM=1492, démesurément grand) ;
  - le refit en excluant les outliers ;
  - si U_EM/c_J^q corrèle avec d'autres invariants (L, braid index).

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, mass, ALPHA0, P_EXP
from theory.abraham import REFERENCE_N50K
from theory.jones import braid_index

KNOTS = list(REFERENCE_N50K.keys())
P_V4 = P_EXP   # = 3,55


def fit_loglog(x_log, y_log):
    """Retourne (pente, intercept, R²)."""
    p, log_K = np.polyfit(x_log, y_log, 1)
    yp = p * x_log + log_K
    R2 = 1 - np.sum((y_log - yp) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
    return float(p), float(log_K), float(R2)


def main():
    cJ_arr = np.array([c_j(K) for K in KNOTS])
    UEM_arr = np.array([u_em(K) for K in KNOTS])
    L_arr = np.array([REFERENCE_N50K[K]["L"] for K in KNOTS])
    bi_arr = np.array([braid_index(K) for K in KNOTS])
    n_cross = np.array([int(K.split("_")[0]) for K in KNOTS])
    log_cJ = np.log(cJ_arr)
    log_UEM = np.log(UEM_arr)

    print("=" * 76)
    print("  CORRÉLATION U_EM ↔ c_J (chantier 1)")
    print("=" * 76)

    # ----- (1) Fit global U_EM ≈ K · c_J^q ------------------------------
    q, log_K, R2 = fit_loglog(log_cJ, log_UEM)
    K_const = np.exp(log_K)
    print(f"\n  Fit log U_EM = q · log c_J + log K :")
    print(f"    q       = {q:.3f}")
    print(f"    K       = {K_const:.3f}")
    print(f"    R²      = {R2:.4f}    (1 = parfait)")
    print(f"    Donc U_EM ≈ {K_const:.3f} · c_J^{q:.3f}")

    # ----- (2) Décomposition de V4 --------------------------------------
    p_total = q + P_V4
    print(f"\n  Décomposition de V4 si U_EM ≈ K·c_J^q :")
    print(f"    m = α₀·U_EM·c_J^{P_V4} ≈ α₀·K·c_J^(q+{P_V4})")
    print(f"        = α₀_eff · c_J^{p_total:.3f}")
    print(f"    où α₀_eff = α₀·K = {ALPHA0:.2e}·{K_const:.3f} = "
          f"{ALPHA0*K_const:.2e} MeV")
    print(f"    p_total apparent = {p_total:.3f}  (à comparer à la pente brute "
          f"log m vs log c_J = 6,363 calculée précédemment)")

    # ----- (3) Tableau des résidus --------------------------------------
    print(f"\n  Résidus par nœud (U_EM observé vs fit) :")
    UEM_fit = K_const * cJ_arr ** q
    log_res = log_UEM - np.log(UEM_fit)
    print(f"  {'K':>5} {'c_J':>5} {'U_EM':>9} {'U_EM fit':>9} {'ratio':>7} "
          f"{'log écart':>10}")
    print("  " + "-" * 50)
    for i, K in enumerate(KNOTS):
        ratio = UEM_arr[i] / UEM_fit[i]
        flag = "★" if abs(log_res[i]) > 0.5 else " "
        print(f"  {K:>5} {cJ_arr[i]:>5.2f} {UEM_arr[i]:>9.2f} {UEM_fit[i]:>9.2f} "
              f"{ratio:>7.2f}x {log_res[i]:>+10.3f}  {flag}")

    # ----- (4) Top outliers ----------------------------------------------
    print(f"\n  Top outliers (|log écart| > 0,5) :")
    idx_outliers = np.where(np.abs(log_res) > 0.5)[0]
    for idx in idx_outliers[np.argsort(-np.abs(log_res[idx_outliers]))]:
        K = KNOTS[idx]
        print(f"    {K} : U_EM = {UEM_arr[idx]:.1f}, fit = {UEM_fit[idx]:.1f}, "
              f"ratio = {UEM_arr[idx]/UEM_fit[idx]:.2f}x")

    # ----- (5) Refit sans 8_3 -------------------------------------------
    mask = np.array([K != "8_3" for K in KNOTS])
    q_no, log_K_no, R2_no = fit_loglog(log_cJ[mask], log_UEM[mask])
    print(f"\n  Refit sans 8_3 (gros outlier — U_EM={u_em('8_3'):.0f} vs fit) :")
    print(f"    q (sans 8_3) = {q_no:.3f}   (vs {q:.3f} avec)")
    print(f"    R² = {R2_no:.4f}")
    print(f"    p_total = {q_no + P_V4:.3f}")

    # ----- (6) Corrélations résiduelles ---------------------------------
    print(f"\n  Le résidu log(U_EM/U_EM_fit) corrèle-t-il avec autre chose ?")
    for label, x in [("L (longueur)", L_arr),
                     ("braid index", bi_arr),
                     ("# crossings", n_cross)]:
        r = np.corrcoef(x, log_res)[0, 1]
        print(f"    log_résidu vs {label:>15} : r = {r:+.3f}")

    # ----- (7) Test direct : m vs c_J seul ------------------------------
    print(f"\n  Test direct : m_V4 (= α₀·U·c_J^3,55) suit-il une loi en c_J^p ?")
    m_V4 = np.array([mass(K, "single") for K in KNOTS])
    log_m = np.log(m_V4)
    p_direct, log_C, R2_direct = fit_loglog(log_cJ, log_m)
    print(f"    log m_V4 = {p_direct:.3f}·log c_J + {log_C:.3f}")
    print(f"    R² = {R2_direct:.4f}")
    print(f"    Si q+3,55 = {p_total:.3f} ≈ {p_direct:.3f}, "
          f"la décomposition U·c_J^3,55 EST redondante.")

    # ----- (8) Conclusion ------------------------------------------------
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────────────────────")
    if R2 > 0.7:
        print(f"  Forte corrélation U_EM ↔ c_J^{q:.1f} (R² = {R2:.2f}) : la formule")
        print(f"  V4 a un facteur U_EM essentiellement REDONDANT avec c_J. La vraie")
        print(f"  loi est m ∝ c_J^{p_total:.2f}, l'exposant FONDAMENTAL est ~{p_total:.0f}.")
    elif R2 > 0.4:
        print(f"  Corrélation modérée (R² = {R2:.2f}) : U_EM et c_J sont liés mais")
        print(f"  U_EM porte de l'info propre. La formule V4 reste utile en factorisant")
        print(f"  les deux dépendances.")
    else:
        print(f"  Faible corrélation (R² = {R2:.2f}) : U_EM et c_J sont essentiellement")
        print(f"  indépendants. La formule V4 utilise les deux comme dimensions vraiment")
        print(f"  indépendantes.")

    # Si la plus grande partie de la variance est dans c_J seul, on peut
    # comparer la dispersion des résidus de m vs c_J^p_total à celle de
    # m vs U_EM·c_J^3.55.
    log_residus_direct = log_m - (p_direct * log_cJ + log_C)
    rms_direct = np.sqrt(np.mean(log_residus_direct**2))
    print(f"\n  RMS log m vs c_J^{p_direct:.2f} seul     : {rms_direct:.4f}")
    print(f"  (vs V4 qui a RMS log = 0 par construction)")
    print("=" * 76)


if __name__ == "__main__":
    main()
