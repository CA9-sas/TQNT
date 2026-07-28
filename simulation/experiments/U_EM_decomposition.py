r"""
Chantier 1 → D1-quater — décomposition U_EM = topologique + paramétrique.

Hypothèse exploratoire : si U_EM_V4 contient à la fois un signal
topologique réel (capturé partiellement par |V(0.5)|² et autres invariants)
et un résidu paramétrique (artefacts Fourier), on devrait pouvoir
écrire :
   U_EM_V4(K) = α · I_topo(K) + β · X_param(K) + ε

avec :
   - I_topo : combinaison linéaire d'invariants topologiques
   - X_param : quantité géométrique paramétrique (rope-length, longueur
               de la courbe Fourier, etc.)

On utilise aussi U_EM_Pieranski comme PROXY DIRECT de la partie
topologique \"propre\" (la valeur de ∮(κ²+τ²)ds sur la courbe relaxée,
sans artefact haute-fréquence).

Quatre tests :

  T1. Régression multi-feature : U_EM_V4 ~ {invariants topologiques + cr}
  T2. Décomposition explicite : U_EM_V4 = U_EM_Pieranski + residual
       (l'idée : U_EM_Pieranski est la partie \"vraie\", le résidu est
       l'artefact catalogue)
  T3. Régression U_EM_Pieranski ~ invariants : a-t-elle un meilleur
       contenu topologique que U_EM_V4 ?
  T4. Application au spectre R6 : la prédiction multi-feature reproduit-
       elle les masses ?

Cf. find_8_3_invariant.py (commit eed1b00) et catalog_rebuild_pieranski.py
(commit b30df12).
"""
from __future__ import annotations
import numpy as np

from theory.jones import (
    jones_polynomial, evaluate_jones, jones_span, jones_determinant,
    braid_index, is_amphicheiral,
)
from theory.abraham import REFERENCE_N50K
from theory.mass_formula import c_j, EW_MARKER
from theory.knot_catalog import knot_curve
from simulation.experiments.catalog_rebuild_pieranski import (
    U_EM_from_points, pieranski_relax,
)

KNOTS = list(REFERENCE_N50K.keys())


def cr(K): return int(K.split("_")[0])


def U_EM_V4(K):
    return float(REFERENCE_N50K[K]["sqrt_U"]) ** 2


def features(K):
    """Vecteur de features topologiques pour le nœud K."""
    p = jones_polynomial(K)
    return {
        "cr": float(cr(K)),
        "span": float(jones_span(p)),
        "det": float(jones_determinant(p)),
        "V05_sq": float(abs(evaluate_jones(p, 0.5)) ** 2),
        "V2_sq": float(abs(evaluate_jones(p, 2.0)) ** 2),
        "V3_sq": float(abs(evaluate_jones(p, 3.0)) ** 2),
        "braid": float(braid_index(K)),
        "amph": 1.0 if is_amphicheiral(p) else 0.0,
        "sum_c": float(sum(abs(c) for c in p.values())),
    }


def fit_linear(X, y):
    """OLS avec intercept. Retourne (beta, y_pred, R²)."""
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return beta, y_pred, R2


def main():
    print("=" * 100)
    print("  DÉCOMPOSITION U_EM_V4 = TOPOLOGIQUE + PARAMÉTRIQUE")
    print("=" * 100)

    # Calcul des features et des U_EM (V4 + Pieranski)
    print(f"\n  Calcul U_EM_V4 et U_EM_Pieranski sur {len(KNOTS)} nœuds…")
    feats = {K: features(K) for K in KNOTS}
    U_V4_arr = np.array([U_EM_V4(K) for K in KNOTS])
    pieranski_pts = {K: pieranski_relax(K, n_iter=500) for K in KNOTS}
    U_PI_arr = np.array([U_EM_from_points(pieranski_pts[K]) for K in KNOTS])

    feat_names = list(feats[KNOTS[0]].keys())
    F = np.array([[feats[K][n] for n in feat_names] for K in KNOTS])
    log_F = np.log(np.where(F > 0, F, 1e-12))
    log_U_V4 = np.log(U_V4_arr)
    log_U_PI = np.log(np.where(U_PI_arr > 0, U_PI_arr, 1e-12))

    # ─── T1 : régression U_EM_V4 ~ features ──────────────────────────────
    print(f"\n  ─── T1 : Régression U_EM_V4 ~ features topologiques ───")
    # Single-feature R²
    print(f"  {'feature':<10} {'R² log-log':>12} {'R² linéaire':>13}")
    print("  " + "-" * 38)
    for i, n in enumerate(feat_names):
        # log-log
        if (F[:, i] > 0).all():
            _, _, R2_log = fit_linear(log_F[:, i:i+1], log_U_V4)
        else:
            R2_log = float("nan")
        _, _, R2_lin = fit_linear(F[:, i:i+1], U_V4_arr)
        print(f"  {n:<10} {R2_log:>12.3f} {R2_lin:>13.3f}")

    # Régression multivariée (toutes features, log-log)
    valid_feat = [i for i, n in enumerate(feat_names) if (F[:, i] > 0).all()]
    Xm = log_F[:, valid_feat]
    _, ypred, R2_multi = fit_linear(Xm, log_U_V4)
    print(f"\n  Régression log U_EM_V4 ~ TOUTES log(features) :")
    print(f"     R² = {R2_multi:.3f}  ({len(valid_feat)} features, 19 points)")

    # ─── T2 : décomposition U_EM_V4 = U_EM_PI + résidu ───────────────────
    print(f"\n  ─── T2 : U_EM_V4 vs U_EM_Pieranski (proxy topologique \"propre\") ───")
    print(f"  {'K':>5} {'cr':>3} {'U_V4':>10} {'U_PI':>10} {'ratio':>10} "
          f"{'résidu':>12}")
    print("  " + "-" * 56)
    for i, K in enumerate(KNOTS):
        ratio = U_V4_arr[i] / U_PI_arr[i] if U_PI_arr[i] > 1e-9 else float("inf")
        residu = U_V4_arr[i] - U_PI_arr[i]
        print(f"  {K:>5} {cr(K):>3} {U_V4_arr[i]:>10.2f} "
              f"{U_PI_arr[i]:>10.2f} {ratio:>10.2f}× {residu:>+12.2f}")

    # Statistique du ratio
    valid_ratios = np.array([U_V4_arr[i] / U_PI_arr[i]
                              for i in range(len(KNOTS)) if U_PI_arr[i] > 1e-9])
    print(f"\n  Ratio U_V4 / U_PI : médiane = {np.median(valid_ratios):.2f}×, "
          f"max = {np.max(valid_ratios):.0f}×")

    # ─── T3 : U_EM_Pieranski a-t-il plus de contenu topologique ? ────────
    print(f"\n  ─── T3 : Régression U_EM_Pieranski ~ features ───")
    print(f"  Si U_PI est la partie 'topologique vraie', sa régression sur")
    print(f"  les invariants devrait avoir R² plus haut que celle de U_V4.")
    print(f"  {'feature':<10} {'R² V4 log':>12} {'R² Pier log':>14} {'gain':>7}")
    print("  " + "-" * 47)
    for i, n in enumerate(feat_names):
        if (F[:, i] > 0).all() and (U_PI_arr > 0).all():
            _, _, R2_v4 = fit_linear(log_F[:, i:i+1], log_U_V4)
            _, _, R2_pi = fit_linear(log_F[:, i:i+1], log_U_PI)
            gain = R2_pi - R2_v4
            flag = "★" if gain > 0.2 else " "
            print(f"  {n:<10} {R2_v4:>12.3f} {R2_pi:>14.3f} "
                  f"{gain:>+7.3f} {flag}")

    if (U_PI_arr > 0).all():
        _, _, R2_PI_multi = fit_linear(log_F[:, valid_feat], log_U_PI)
        print(f"\n  Régression log U_PI ~ TOUTES log(features) :")
        print(f"     R² = {R2_PI_multi:.3f}  (vs {R2_multi:.3f} pour U_V4)")
        if R2_PI_multi > R2_multi + 0.1:
            print(f"  ★ U_PI a +{R2_PI_multi-R2_multi:.2f} de contenu topologique en plus.")
            print(f"    Confirmé : la partie Fourier de U_V4 est du bruit non-topologique.")
        else:
            print(f"  ~ Pas de gain significatif : Pieranski 1 n'extrait pas mieux")
            print(f"    le signal topologique que le catalogue Fourier (algo imparfait).")

    # ─── T4 : test mass formula avec prédiction multi-feature ────────────
    print(f"\n  ─── T4 : Formule R6 avec prédicteur multi-feature ───")
    print(f"  On remplace U_EM_V4(K) par U_pred(K) = exp(régression log) et")
    print(f"  on applique la formule R6 sur 7 canoniques.")

    # Faire la régression multi-feature
    beta_mu, _, R2_mu = fit_linear(log_F[:, valid_feat], log_U_V4)
    selected_names = [feat_names[i] for i in valid_feat]

    def U_pred(K):
        x = np.array([feats[K][n] for n in selected_names])
        log_x = np.log(np.where(x > 0, x, 1e-12))
        log_u = beta_mu[0] + beta_mu[1:] @ log_x
        return float(np.exp(log_u))

    print(f"\n     Coefficients régression :")
    print(f"        intercept = {beta_mu[0]:+.3f}")
    for j, n in enumerate(selected_names):
        print(f"        log({n:<8}) → β = {beta_mu[j+1]:+.3f}")

    # Test des prédictions
    print(f"\n     U_EM prédit vs actuel :")
    print(f"     {'K':>5} {'U_V4':>10} {'U_pred':>10} {'ratio':>8}")
    for K in KNOTS:
        u_pred = U_pred(K)
        ratio = u_pred / U_EM_V4(K) if U_EM_V4(K) > 0 else float("inf")
        print(f"     {K:>5} {U_EM_V4(K):>10.2f} {u_pred:>10.2f} {ratio:>8.2f}×")

    # Formule R6 avec U_pred
    SPECTRUM = [
        ("e-",  ["3_1"],               0.51099895),
        ("p",   ["3_1", "7_2", "7_2"], 938.272),
        ("n",   ["3_1", "7_2", "7_2"], 939.565),
        ("W",   ["4_1", "7_3", "8_3"], 80377.0),
        ("Z",   ["5_1", "7_2", "8_3"], 91188.0),
        ("H",   ["6_1", "7_4", "8_3"], 125250.0),
        ("t",   ["7_4", "7_6", "8_3"], 173100.0),
    ]

    def f_screening(comps):
        n = len(comps)
        if n == 1 or EW_MARKER in comps:
            return 1.0
        return ((n + 1) / 2) ** (-3 / 8)

    def mass_pred(comps, alpha0, U_func):
        I_tot = sum(U_func(K) for K in comps)
        cJ_tot = sum(c_j(K) for K in comps)
        f = f_screening(comps)
        return alpha0 * I_tot * (f * cJ_tot) ** (32 / 9)

    def calib(U_func):
        log_alphas = [np.log(me / mass_pred(comps, 1.0, U_func))
                       for nm, comps, me in SPECTRUM]
        return float(np.exp(np.mean(log_alphas)))

    a0_pred = calib(U_pred)
    print(f"\n     Test R6 avec U_pred (multi-feature) :")
    print(f"     α₀ calibré = {a0_pred:.4e} MeV")
    print(f"     {'part.':>6} {'m_exp':>11} {'m_pred':>11} {'Δ %':>8}")
    residuals = []
    for nm, comps, me in SPECTRUM:
        m = mass_pred(comps, a0_pred, U_pred)
        e = (m - me) / me * 100
        residuals.append(np.log(m / me))
        flag = "★" if abs(e) < 1 else (" " if abs(e) < 10 else "?")
        print(f"     {nm:>6} {me:>11.2f} {m:>11.2f} {e:>+7.2f} {flag}")
    rms_pred = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    print(f"     RMS log = {rms_pred:.4f}")

    # Comparaison
    print(f"\n  ─── BILAN DES RMS log SUR 7 CANONIQUES ───")
    print(f"     R6 référence (U_EM_V4)            : 0.005")
    print(f"     R6 + |V(0.5)|² seul              : 0.945")
    print(f"     R6 + U_pred (régression multi)   : {rms_pred:.3f}")
    print(f"     R6 + U_EM_Pieranski (catalogue propre) : 1.760")

    # ─── Conclusion ──────────────────────────────────────────────────────
    print(f"\n  ─── DIAGNOSTIC ───")
    print(f"  Régression log U_EM_V4 ~ 8 features : R² = {R2_multi:.3f}")
    if R2_multi > 0.95:
        print(f"  ★ U_EM_V4 est PRESQUE entièrement déterminé par les invariants")
        print(f"    topologiques. Le 'paramétrique' est marginal.")
    elif R2_multi > 0.7:
        print(f"  ✓ U_EM_V4 a une composante topologique substantielle ({R2_multi:.0%})")
        print(f"    mais ~{(1-R2_multi)*100:.0f}% de la variance reste inexpliquée.")
        print(f"    Cette résiduelle est probablement la part paramétrique.")
    else:
        print(f"  ✗ U_EM_V4 a une composante topologique faible ({R2_multi:.0%})")
        print(f"    L'essentiel de sa variance n'est pas dérivable des invariants")
        print(f"    standards. Confirmation que c'est largement paramétrique.")
    print("=" * 100)


if __name__ == "__main__":
    main()
