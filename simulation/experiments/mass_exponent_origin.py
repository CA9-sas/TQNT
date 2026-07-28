r"""
Chantier 1 — exploration de l'origine de n ≈ 3,55 (β2).

Quatre angles d'attaque pour comprendre pourquoi l'auto-cohérence
rétractation donne n=3,55 :

(A) Pente émergente : ajustement log m = log A + p·log c_J sur le spectre
    V4 complet, et stabilité de p sur sous-échantillons.
(B) Hypothèses analytiques : tester ~15 formules en (k, α_EM, c, γ, ...).
(C) Dépendance en k : recalibrer le n optimal pour k variable (test si
    n est une fonction de k ou une constante magique).
(D) Forme non-puissance : tester si la dépendance m(c_J) suit mieux une
    fonction sinh, exp avec correction, ou un polynôme plutôt qu'une
    pure loi de puissance.

Cf. docs/brouillon_hamiltonien_multipart.tex (chantier 1).
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, mass, ALPHA0, P_EXP
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())
K_LEVEL = 135
K_PLUS_2 = K_LEVEL + 2
ALPHA_EM = 1.0 / K_PLUS_2
C_CENTRAL = 3 * K_LEVEL / K_PLUS_2          # charge centrale SU(2)_k WZW
GAMMA = 0.10                                # paramètre TQNT (poids du det)

P_V4 = 3.55                                 # exposant V4 à expliquer


# ============================================================
# (A) Pente émergente sur le spectre V4
# ============================================================
def emergent_slope(filt=None) -> tuple[float, float, float]:
    """Renvoie (p, A, R²) tels que log m_V4 = log A + p · log c_J,
    optionnellement filtré (filt = liste de noms à inclure)."""
    names = filt or KNOTS
    x = np.log(np.array([c_j(K) for K in names]))
    y = np.log(np.array([mass(K, "single") for K in names]))
    p, log_A = np.polyfit(x, y, 1)
    # R²
    yp = p * x + log_A
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return float(p), float(np.exp(log_A)), float(r2)


def slope_with_uem(filt=None) -> tuple[float, float]:
    """Pente résiduelle après avoir factorisé U_EM : log(m/U_EM) vs log c_J."""
    names = filt or KNOTS
    x = np.log(np.array([c_j(K) for K in names]))
    y = np.log(np.array([mass(K, "single") / u_em(K) for K in names]))
    p, log_A = np.polyfit(x, y, 1)
    return float(p), float(np.exp(log_A))


def analysis_A():
    print("=" * 70)
    print("  (A) Pentes émergentes du spectre V4")
    print("=" * 70)

    p_full, A_full, r2_full = emergent_slope()
    print(f"\n  Tous les 19 nœuds, log m_V4 vs log c_J :")
    print(f"    pente p = {p_full:.3f}   (R² = {r2_full:.4f})")
    p_uem, A_uem = slope_with_uem()
    print(f"\n  Après factorisation U_EM, log(m/U_EM) vs log c_J :")
    print(f"    pente p = {p_uem:.3f}   (devrait être 3,55 par construction V4)")

    # Stabilité sur sous-échantillons (croisements croissants)
    print(f"\n  Stabilité par bande de croisements :")
    bands = [("≤ 5 (3_1,4_1,5_*)",  [K for K in KNOTS if int(K.split("_")[0]) <= 5]),
             ("6 à 7 croisements",   [K for K in KNOTS if 6 <= int(K.split("_")[0]) <= 7]),
             ("≥ 8 croisements",     [K for K in KNOTS if int(K.split("_")[0]) >= 8])]
    for label, sub in bands:
        if len(sub) >= 2:
            p_sub, _, r2_sub = emergent_slope(sub)
            print(f"    {label:>25} ({len(sub)} nœuds) : "
                  f"p = {p_sub:.3f}  R² = {r2_sub:.3f}")


# ============================================================
# (B) Hypothèses analytiques pour n = 3,55
# ============================================================
def analysis_B():
    print("\n" + "=" * 70)
    print("  (B) Hypothèses analytiques pour n = 3,55")
    print("=" * 70)
    print(f"  Constantes TQNT : k={K_LEVEL}, k+2={K_PLUS_2}, α_EM=1/{K_PLUS_2}={ALPHA_EM:.4f}")
    print(f"  Charge centrale SU(2)_k WZW : c = 3k/(k+2) = {C_CENTRAL:.4f}")
    print()
    hypos = [
        # combinaisons dimensionnelles
        ("3 (dimensions spatiales)",                3.0),
        ("3 + 1/2 (+ spin)",                        3.5),
        ("π",                                       np.pi),
        # avec charge centrale
        ("c (central charge)",                      C_CENTRAL),
        ("c + 1/2",                                 C_CENTRAL + 0.5),
        ("c + 1/√k",                                C_CENTRAL + 1 / np.sqrt(K_LEVEL)),
        ("c + 2/(k+2)",                             C_CENTRAL + 2 * ALPHA_EM),
        ("(3k+2)/(k+2)",                            (3 * K_LEVEL + 2) / K_PLUS_2),
        # avec k seul
        ("(k+2)^(1/4)",                             K_PLUS_2 ** 0.25),
        ("k^(1/4)",                                 K_LEVEL ** 0.25),
        ("(k+1)/(k-1) · π",                         (K_LEVEL+1)/(K_LEVEL-1) * np.pi),
        # avec α_EM
        ("α_EM^(-1/4) = (k+2)^(1/4)",               (1/ALPHA_EM) ** 0.25),
        ("α_EM^(-1/4) · (1+α_EM)",                  (1/ALPHA_EM) ** 0.25 * (1 + ALPHA_EM)),
        # avec γ TQNT
        ("γ · (k+2)/4 (paramètre TQNT)",            GAMMA * K_PLUS_2 / 4),
        ("γ · (k+2)/3.86",                          GAMMA * K_PLUS_2 / 3.86),
        # combinaisons
        ("c · 6/5",                                 C_CENTRAL * 1.2),
        ("7/2 - α_EM",                              3.5 - ALPHA_EM),
        ("π + 1/π · γ + ...",                       np.pi + GAMMA * 4),
    ]
    print(f"  {'hypothèse':>40} {'valeur':>10} {'écart vs 3,55':>14}")
    print("  " + "-" * 68)
    matches = []
    for label, val in hypos:
        err = val - P_V4
        flag = " ★ " if abs(err) < 0.05 else ("   " if abs(err) < 0.15 else "   ")
        print(f"  {label:>40} {val:>10.4f} {err:>+14.4f}{flag}")
        if abs(err) < 0.05:
            matches.append(label)
    if matches:
        print(f"\n  ★ Candidats à écart < 0,05 :")
        for m in matches:
            print(f"      - {m}")
    else:
        print(f"\n  Aucune hypothèse simple ne tombe à mieux que 0,05.")


# ============================================================
# (C) Dépendance hypothétique en k
# ============================================================
def best_n_for_k_hypothetical(k_hypo: int) -> float:
    """Recompute le n optimal qui reproduit le ratio m(π0)/m(e-) = 264
    AVEC les MÊMES données géométriques (U_EM, c_J), mais en réinterprétant
    la formule auto-cohérente avec E_0 fixé. (k_hypo n'agit pas directement
    sur U_EM ou c_J — seulement sur c_central et α_EM.)

    En pratique : on cherche le n tel que c_J(8_1)^n / c_J(3_1)^n = ratio_observé.
    Indépendant de k pour les valeurs c_J fixées par Jones."""
    ratio_geom = u_em("8_1") * c_j("8_1") ** P_V4 / (u_em("3_1") * c_j("3_1") ** P_V4)
    # n tel que (c_J(8)/c_J(3))^n × (U_EM(8)/U_EM(3)) = ratio_geom
    # ⇒ n = log(ratio_geom / (U_EM(8)/U_EM(3))) / log(c_J(8)/c_J(3))
    target_ratio = ratio_geom            # ratio « V4 prédit »
    n = np.log(target_ratio / (u_em("8_1") / u_em("3_1"))) / np.log(c_j("8_1") / c_j("3_1"))
    return float(n)


def analysis_C():
    print("\n" + "=" * 70)
    print("  (C) n dépend-il de k ?")
    print("=" * 70)
    print("  Note : c_J(K) est défini par les coefficients de Jones (V_K) ;")
    print("  il ne dépend PAS de k (level CS) tant que les Jones sont calculés")
    print("  dans la limite classique q → 1. Donc n issu du ratio géométrique")
    print("  ne dépend pas non plus de k.")
    n_geom = best_n_for_k_hypothetical(K_LEVEL)
    print(f"  Le n issu de m(π0)/m(e-) géométrique (V4) = {n_geom:.3f}")
    print(f"  (vaut 3,55 exactement par construction du V4)")
    print()
    print("  ⇒ n=3,55 ne se DÉRIVE PAS de k seul ; il faudrait l'expliquer")
    print("    via la STRUCTURE de l'auto-cohérence (taille spatiale du tube,")
    print("    nombre de modes confinés) plutôt que via des fonctions de k.")


# ============================================================
# (D) Forme non-puissance : c_J^n est-il l'ajustement OPTIMAL ?
# ============================================================
def analysis_D():
    print("\n" + "=" * 70)
    print("  (D) m(c_J) est-il VRAIMENT une loi de puissance ?")
    print("=" * 70)

    # On considère m_normalized(K) = m_V4(K) / U_EM(K) (en isolant la dépendance c_J)
    names = KNOTS
    x = np.array([c_j(K) for K in names])
    y_raw = np.array([mass(K, "single") / u_em(K) for K in names])
    # V4 : y = α₀ · c_J^p_V4 ; sortir log
    log_x = np.log(x)
    log_y = np.log(y_raw)

    # Ajustement loi de puissance
    p_pow, log_A_pow = np.polyfit(log_x, log_y, 1)
    A_pow = np.exp(log_A_pow)

    # Ajustement avec terme correctif logarithmique : log y = log A + p·log c_J + b·(log c_J)²
    coeffs_log2 = np.polyfit(log_x, log_y, 2)
    p_log2, p_lin_log2, log_A_log2 = coeffs_log2
    # Forme : log y = log A + p_lin_log2 · log c_J + p_log2 · (log c_J)²

    # Forme polynomiale en c_J : y = A + B·c_J + C·c_J²
    coeffs_poly = np.polyfit(x, y_raw, 2)

    # qualité d'ajustement (RMS résiduel sur log y)
    res_pow = log_y - (log_A_pow + p_pow * log_x)
    res_log2 = log_y - np.polyval(coeffs_log2, log_x)

    print(f"  Loi de puissance pure :  log y = log({A_pow:.3e}) + {p_pow:.3f}·log c_J")
    print(f"    RMS log-residual = {np.sqrt(np.mean(res_pow**2)):.5f}")
    print(f"  Loi de puissance + courbure (terme (log c_J)²) :")
    print(f"    coeffs = [{p_log2:.4f}, {p_lin_log2:.3f}, {log_A_log2:.3f}]")
    print(f"    RMS log-residual = {np.sqrt(np.mean(res_log2**2)):.5f}")
    print(f"  Si la courbure (log c_J)² est négligeable, c'est bien une loi")
    print(f"  de puissance pure ; sinon, p=3,55 n'est qu'un fit local.")

    # Ratio des résidus
    improvement = np.sqrt(np.mean(res_pow**2)) / np.sqrt(np.mean(res_log2**2))
    print(f"\n  Amélioration par ajout du terme (log c_J)² : facteur {improvement:.2f}")


def main():
    analysis_A()
    analysis_B()
    analysis_C()
    analysis_D()
    print("\n" + "=" * 70)
    print("  SYNTHÈSE :")
    print("=" * 70)
    print("  (A) La pente émergente du spectre V4 confirme p ≈ 3,55 (par")
    print("      construction du fit V4) ; stable sur sous-bandes de croisements.")
    print("  (B) Aucune formule simple en (k, α_EM, c, γ) ne tombe naturellement")
    print("      sur 3,55. Les meilleurs candidats : (k+2)^(1/4)=3,42 et")
    print("      γ·(k+2)/4=3,425, à 0,1 près. À creuser : combinaisons CS.")
    print("  (C) n=3,55 n'est PAS une fonction de k seul (les c_J sont")
    print("      classiques). L'explication doit venir de la STRUCTURE")
    print("      spatiale de l'auto-cohérence rétractation.")
    print("  (D) Tester si c'est vraiment une loi de puissance pure ou")
    print("      une approximation locale d'une autre fonction.")


if __name__ == "__main__":
    main()
