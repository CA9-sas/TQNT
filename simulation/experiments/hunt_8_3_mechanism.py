r"""
Chantier 1 → D1-cinq — chasse exhaustive d'un mécanisme topologique
expliquant U_EM(8_3) = 1492 (commit 9a9c10b).

Le nœud 8_3 a été identifié comme l'unique grand outlier paramétrique
du catalogue : sa valeur U_EM(8_3) = 1492 est 25× plus grande que celle
de ses voisins cr=8 (60 et 117), et 80 % de cette valeur n'est PAS
explicable par les 8 invariants standards testés.

Cette chasse explore des invariants plus exotiques :

  A. Volume hyperbolique du complément (Snappy / table publiée)
  B. Évaluations colorées du Jones J_n(K, e^{2iπ/(k+2)}) pour n,k variés
  C. Combinaisons non-linéaires (Vol × cr², genre × det², ...)
  D. Quantités sensibles à l'amphichiralité (signature Tristram-Levine,
     déterminant pour t^a)
  E. Invariants de "complexité" : longueur d'arc euclidienne minimale
     dans une présentation de tresse, nombre d'écritures braid possibles
  F. PROFIL DES ÉCARTS : pour chaque invariant testé, calculer le ratio
     I(8_3) / median(I(autres cr=8)) — chercher tout invariant donnant
     un ratio ≥ 15 (cible : 1492 / 88 = 17).

L'hypothèse de travail : il existe un invariant standard donnant
naturellement 8_3 ≈ 20× ses voisins. Si oui, on a la justification
topologique de U_EM(8_3) = 1492. Si non, on accepte définitivement
le 1492 comme paramètre phénoménologique du modèle.
"""
from __future__ import annotations
import numpy as np

from theory.jones import (
    jones_polynomial, evaluate_jones, jones_span, jones_determinant,
    braid_index, bridge_number, is_amphicheiral,
)
from theory.chern_simons import (
    cs_q, quantum_integer, colored_jones,
)
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())

# ============================================================
# Données de référence : volumes hyperboliques (KnotInfo)
# Source : http://www.indiana.edu/~knotinfo
# Knot torus : volume = 0 (pas hyperbolique). On laisse 0.0 ici.
# ============================================================
HYPERBOLIC_VOLUME = {
    "3_1":   0.0,            # torus
    "4_1":   2.029883,
    "5_1":   0.0,            # torus
    "5_2":   2.828122,
    "6_1":   3.163963,
    "6_2":   4.400833,
    "6_3":   5.693021,
    "7_1":   0.0,            # torus
    "7_2":   3.331744,
    "7_3":   4.591885,
    "7_4":   5.074708,
    "7_5":   6.443537,
    "7_6":   7.084327,
    "7_7":   7.643375,
    "8_1":   3.427057,
    "8_2":   4.934876,
    "8_3":   6.551745,
    "9_1":   0.0,            # torus
    "10_1":  7.705763,
}

# ============================================================
# Genre minimal (Seifert genus, KnotInfo)
# ============================================================
SEIFERT_GENUS = {
    "3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1, "6_1": 1, "6_2": 2,
    "6_3": 2, "7_1": 3, "7_2": 1, "7_3": 2, "7_4": 1, "7_5": 2,
    "7_6": 2, "7_7": 2, "8_1": 1, "8_2": 3, "8_3": 1, "9_1": 4,
    "10_1": 1,
}

# ============================================================
# Signature classique σ(K) (Tristram-Levine à -1, ≡ Trotter)
# ============================================================
SIGNATURE = {
    "3_1": -2, "4_1": 0, "5_1": -4, "5_2": -2, "6_1": 0, "6_2": -2,
    "6_3": 0, "7_1": -6, "7_2": -2, "7_3": -4, "7_4": -2, "7_5": -4,
    "7_6": -2, "7_7": 0, "8_1": 0, "8_2": -4, "8_3": 0, "9_1": -8,
    "10_1": 0,
}


def cr(K): return int(K.split("_")[0])
def U_EM_V4(K): return float(REFERENCE_N50K[K]["sqrt_U"]) ** 2


def report_invariant(name, vals, U_arr, knots):
    """Affiche l'invariant et calcule des statistiques de discrimination."""
    if "8_3" not in knots:
        return None
    i83 = vals[knots.index("8_3")]
    # Médian des cr=8 autres
    cr8_others = [v for k, v in zip(knots, vals) if cr(k) == 8 and k != "8_3"]
    med_cr8 = float(np.median(cr8_others)) if cr8_others else 0.0
    ratio83 = i83 / med_cr8 if abs(med_cr8) > 1e-9 else float("inf")
    # Z-score global
    z = (i83 - np.mean(vals)) / (np.std(vals) + 1e-9)
    # Corrélation avec U_EM
    try:
        r = np.corrcoef(vals, U_arr)[0, 1]
    except Exception:
        r = float("nan")
    flag = "★" if (abs(ratio83) > 15 and abs(r) > 0.7) else \
           ("✓" if (abs(ratio83) > 5 or abs(r) > 0.85) else " ")
    return {
        "name": name, "i83": i83, "med_cr8": med_cr8,
        "ratio": ratio83, "z": z, "r": r, "flag": flag,
    }


def main():
    print("=" * 100)
    print("  CHASSE D'UN MÉCANISME TOPOLOGIQUE EXPLIQUANT U_EM(8_3) = 1492")
    print("=" * 100)
    print(f"  Cible : I(8_3) / median(I(autres cr=8)) ≥ 15  ET  Pearson(I, U_EM) > 0.7")
    print(f"  Valeurs observées : U_EM(8_1)=60, U_EM(8_2)=117, U_EM(8_3)=1492")
    print(f"  Ratio cible : 1492 / median(60, 117) = 1492 / 88 = 17×")

    U_arr = np.array([U_EM_V4(K) for K in KNOTS])
    results = []

    # ─── A. Volume hyperbolique ──────────────────────────────────────────
    print(f"\n  ─── A. VOLUME HYPERBOLIQUE ───")
    vols = np.array([HYPERBOLIC_VOLUME[K] for K in KNOTS])
    r = report_invariant("vol", vols, U_arr, KNOTS)
    print(f"    vol(8_1) = {HYPERBOLIC_VOLUME['8_1']:.3f}")
    print(f"    vol(8_2) = {HYPERBOLIC_VOLUME['8_2']:.3f}")
    print(f"    vol(8_3) = {HYPERBOLIC_VOLUME['8_3']:.3f}  ← outlier ratio "
          f"{r['ratio']:.2f}×")
    print(f"    Pearson(vol, U_EM) = {r['r']:+.3f}")
    results.append(r)

    # Combinaisons avec volume
    print(f"\n  ─── A bis. Combinaisons avec volume ───")
    combos_vol = [
        ("vol²",        vols ** 2),
        ("vol³",        vols ** 3),
        ("exp(vol)",    np.exp(vols)),
        ("vol · cr",    vols * np.array([cr(K) for K in KNOTS])),
        ("vol · det",   vols * np.array([jones_determinant(jones_polynomial(K)) for K in KNOTS])),
        ("vol² · cr",   vols ** 2 * np.array([cr(K) for K in KNOTS])),
        ("vol³ · cr²",  vols ** 3 * np.array([cr(K) for K in KNOTS]) ** 2),
        ("exp(vol/π)",  np.exp(vols / np.pi)),
    ]
    print(f"  {'invariant':<18} {'I(8_3)':>10} {'med cr=8':>10} "
          f"{'ratio':>8} {'r(U_EM)':>9} {'flag':>5}")
    for name, vals in combos_vol:
        r = report_invariant(name, vals, U_arr, KNOTS)
        results.append(r)
        print(f"  {name:<18} {r['i83']:>10.3f} {r['med_cr8']:>10.3f} "
              f"{r['ratio']:>7.2f}× {r['r']:>+8.3f} {r['flag']:>5}")

    # ─── B. Jones colorés J_n(K, q) à différents q ──────────────────────
    print(f"\n  ─── B. Évaluations colorées Jones ───")
    print(f"  Seulement les nœuds connus pour J_n (torus + 4_1).")
    # On utilise des évaluations directes du polynôme non-coloré à des points exotiques
    points_exotic = [
        ("V(t=e^{iπ/10})",  np.exp(1j * np.pi / 10)),
        ("V(t=e^{iπ/12})",  np.exp(1j * np.pi / 12)),
        ("V(t=e^{iπ/8})",   np.exp(1j * np.pi / 8)),
        ("V(t=e^{2iπ/15})", np.exp(2j * np.pi / 15)),
        ("V(t=-2)",         -2.0 + 0j),
        ("V(t=-0.5)",       -0.5 + 0j),
        ("V(t=0.25)",       0.25 + 0j),
        ("V(t=4)",          4.0 + 0j),
        ("V(t=1.5)",        1.5 + 0j),
    ]
    for name, t in points_exotic:
        vals = np.array([float(abs(evaluate_jones(jones_polynomial(K), t)) ** 2)
                          for K in KNOTS])
        r = report_invariant(name, vals, U_arr, KNOTS)
        results.append(r)
        print(f"  {name:<18} {r['i83']:>10.3f} {r['med_cr8']:>10.3f} "
              f"{r['ratio']:>7.2f}× {r['r']:>+8.3f} {r['flag']:>5}")

    # ─── C. Genre et signature ───────────────────────────────────────────
    print(f"\n  ─── C. Genre, signature, et combinaisons ───")
    genus = np.array([SEIFERT_GENUS[K] for K in KNOTS])
    sig = np.array([SIGNATURE[K] for K in KNOTS])
    combos_g = [
        ("genus",                genus),
        ("|signature|",          np.abs(sig)),
        ("genus · cr",           genus * np.array([cr(K) for K in KNOTS])),
        ("genus² · cr",          genus ** 2 * np.array([cr(K) for K in KNOTS])),
        ("(2g+1) · det",         (2 * genus + 1) * np.array([jones_determinant(jones_polynomial(K)) for K in KNOTS])),
        ("amph · vol²",          np.array([1.0 if is_amphicheiral(jones_polynomial(K)) else 0.0 for K in KNOTS]) * vols ** 2),
    ]
    print(f"  {'invariant':<18} {'I(8_3)':>10} {'med cr=8':>10} "
          f"{'ratio':>8} {'r(U_EM)':>9} {'flag':>5}")
    for name, vals in combos_g:
        r = report_invariant(name, vals, U_arr, KNOTS)
        results.append(r)
        print(f"  {name:<18} {r['i83']:>10.3f} {r['med_cr8']:>10.3f} "
              f"{r['ratio']:>7.2f}× {r['r']:>+8.3f} {r['flag']:>5}")

    # ─── D. Combinaisons amphichirales spécifiques ──────────────────────
    print(f"\n  ─── D. Tests spécifiques amphichiralité ───")
    print(f"  L'amphichiralité de 8_3 est-elle suffisante pour le distinguer ?")
    amph_arr = np.array([1.0 if is_amphicheiral(jones_polynomial(K)) else 0.0 for K in KNOTS])
    amphi_knots = [K for K in KNOTS if amph_arr[KNOTS.index(K)] == 1.0]
    print(f"  Nœuds amphichiraux dans le catalogue : {amphi_knots}")
    for K in amphi_knots:
        print(f"    U_EM({K}) = {U_EM_V4(K):.2f}")

    # ─── E. Évaluations Jones aux roots of unity Chern-Simons ────────────
    print(f"\n  ─── E. Jones aux points Chern-Simons q = exp(2iπ/(k+2)) ───")
    print(f"  Pour k = 1, 2, ..., 20.")
    for k_cs in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        q = cs_q(k_cs)
        vals = np.array([float(abs(evaluate_jones(jones_polynomial(K), q)) ** 2)
                          for K in KNOTS])
        r = report_invariant(f"|V(q_k={k_cs})|²", vals, U_arr, KNOTS)
        results.append(r)
        if abs(r['ratio']) > 3 or abs(r['r']) > 0.8:
            print(f"  k={k_cs:>3}: I(8_3)={r['i83']:>9.3f}, med_cr8={r['med_cr8']:>9.3f}, "
                  f"ratio={r['ratio']:>6.2f}×, r={r['r']:+.3f} {r['flag']}")

    # ─── F. Meilleurs candidats globaux ─────────────────────────────────
    print(f"\n  {'═' * 80}")
    print(f"  ▶ TOP 10 INVARIANTS PAR SCORE COMBINÉ (ratio × |r(U_EM)|)")
    print(f"  {'═' * 80}")
    # Score = ratio × |r| (haute ratio ET forte corrélation = idéal)
    valid = [r for r in results if r and not np.isnan(r['ratio']) and not np.isinf(r['ratio'])]
    for r in valid:
        r['score'] = abs(r['ratio']) * abs(r['r'])
    valid.sort(key=lambda r: -r['score'])
    print(f"  {'invariant':<22} {'I(8_3)':>10} {'med_cr8':>10} "
          f"{'ratio':>8} {'r(U_EM)':>9} {'score':>7} {'flag':>5}")
    print("  " + "-" * 75)
    for r in valid[:12]:
        print(f"  {r['name']:<22} {r['i83']:>10.3f} {r['med_cr8']:>10.3f} "
              f"{r['ratio']:>7.2f}× {r['r']:>+8.3f} {r['score']:>7.2f} {r['flag']:>5}")

    # ─── G. TEST SUBSTITUTION : R6 avec |V(0.25)|² au lieu de U_EM ──────
    print(f"\n  {'═' * 80}")
    print(f"  ▶ TEST CRITIQUE : substitution |V(0.25)|² dans la formule R6")
    print(f"  {'═' * 80}")

    from theory.mass_formula import c_j, EW_MARKER
    from scipy.optimize import minimize

    SPECTRUM = [
        ("e-",  ["3_1"],               0.51099895),
        ("p",   ["3_1", "7_2", "7_2"], 938.272),
        ("n",   ["3_1", "7_2", "7_2"], 939.565),
        ("W",   ["4_1", "7_3", "8_3"], 80377.0),
        ("Z",   ["5_1", "7_2", "8_3"], 91188.0),
        ("H",   ["6_1", "7_4", "8_3"], 125250.0),
        ("t",   ["7_4", "7_6", "8_3"], 173100.0),
    ]

    def V025_sq(K):
        return float(abs(evaluate_jones(jones_polynomial(K), 0.25)) ** 2)

    def f_screening(comps):
        n = len(comps)
        if n == 1 or EW_MARKER in comps:
            return 1.0
        return ((n + 1) / 2) ** (-3 / 8)

    def mass_pred(comps, alpha0, q, I_func):
        I_tot = sum(I_func(K) for K in comps)
        cJ_tot = sum(c_j(K) for K in comps)
        f = f_screening(comps)
        return alpha0 * I_tot * (f * cJ_tot) ** q

    def loss(x, I_func):
        log_a, q = x
        a0 = float(np.exp(log_a))
        residuals = [np.log(mass_pred(c, a0, q, I_func) / me)
                     for nm, c, me in SPECTRUM]
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    # Optim avec |V(0.25)|² comme substitut U_EM
    res = minimize(lambda x: loss(x, V025_sq), x0=[np.log(1e-6), 3.55],
                   method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9})
    a0_v25, q_v25 = float(np.exp(res.x[0])), res.x[1]
    rms_v25 = float(res.fun)
    print(f"\n  Substitution U_EM → |V(0.25)|² (q libre) :")
    print(f"    α₀ optim = {a0_v25:.4e} MeV, q optim = {q_v25:.4f}")
    print(f"    RMS log = {rms_v25:.4f}")
    print(f"    {'part.':>6} {'m_exp':>11} {'m_pred':>11} {'Δ %':>9}")
    for nm, comps, me in SPECTRUM:
        m = mass_pred(comps, a0_v25, q_v25, V025_sq)
        e = (m - me) / me * 100
        flag = "★" if abs(e) < 1 else (" " if abs(e) < 10 else "?")
        print(f"    {nm:>6} {me:>11.2f} {m:>11.2f} {e:>+8.2f} {flag}")

    # Test avec q = 32/9 fixé
    res_qf = minimize(lambda x: loss([x[0], 32 / 9], V025_sq), x0=[np.log(1e-6)],
                      method="Nelder-Mead")
    rms_v25_qf = float(res_qf.fun)
    print(f"\n  Substitution U_EM → |V(0.25)|² avec q = 32/9 FIXÉ :")
    print(f"    α₀ optim = {float(np.exp(res_qf.x[0])):.4e} MeV")
    print(f"    RMS log = {rms_v25_qf:.4f}")

    # Comparaison
    print(f"\n  ─── BILAN DES SUBSTITUTIONS ───")
    print(f"    R6 + U_EM_V4 (référence)         : RMS = 0.005 ★")
    print(f"    R6 + |V(0.5)|²                   : RMS = 0.945")
    print(f"    R6 + U_pred multi-feature        : RMS = 0.795")
    print(f"    R6 + |V(0.25)|² (q libre)        : RMS = {rms_v25:.3f}")
    print(f"    R6 + |V(0.25)|² (q = 32/9 fixé)  : RMS = {rms_v25_qf:.3f}")
    print(f"    R6 + U_EM_Pieranski              : RMS = 1.760")

    # Analyse de pourquoi |V(0.25)|² ne marche pas malgré les bons indicateurs
    print(f"\n  ─── DIAGNOSTIC ÉCHEC SUBSTITUTION |V(0.25)|² ───")
    print(f"  Bien que |V(0.25)|²(8_3) soit grand (45 951), il l'est AUSSI")
    print(f"  pour les petits nœuds par symétrie t^(-4) :")
    for K in ["3_1", "4_1", "5_1", "7_2", "8_3"]:
        u = U_EM_V4(K)
        v = V025_sq(K)
        ratio = u / v if v > 0 else float("inf")
        print(f"     {K}: U_EM={u:>7.1f}, |V(0.25)|²={v:>10.1f}, "
              f"ratio = {ratio:>.5f}")
    print(f"  Le ratio U_EM / |V(0.25)|² varie de 2.5e-4 (3_1) à 3.2e-2 (8_3),")
    print(f"  soit 130× de variation — pas constant.")
    print(f"  Donc |V(0.25)|² capture l'OUTLIER 8_3 (Z, ratio), mais ne")
    print(f"  reproduit pas le PROFIL fonctionnel de U_EM(K) sur les 19 nœuds.")

    # ─── Verdict ────────────────────────────────────────────────────────
    print(f"\n  ─── VERDICT ───")
    has_match = any(abs(r['ratio']) > 15 and abs(r['r']) > 0.7 for r in valid)
    if has_match:
        print(f"  ★ TROUVÉ : invariant satisfaisant ratio≥15× ET |r|>0.7")
        best = max(valid, key=lambda r: r['score'])
        print(f"    Meilleur : {best['name']}")
        print(f"    I(8_3)/med_cr=8 = {best['ratio']:.2f}×  (cible ≥15)")
        print(f"    Pearson(I, U_EM) = {best['r']:+.3f}  (cible ≥0.7)")
        print(f"    ⇒ Origine topologique candidate identifiée !")
    else:
        print(f"  ✗ AUCUN invariant testé n'atteint simultanément :")
        print(f"      ratio I(8_3)/median(autres cr=8) ≥ 15")
        print(f"      ET Pearson(I, U_EM_V4) ≥ 0.7")
        max_ratio = max(r['ratio'] for r in valid)
        max_corr = max(abs(r['r']) for r in valid)
        print(f"    Meilleur ratio = {max_ratio:.1f}×")
        print(f"    Meilleur |r|   = {max_corr:.3f}")
        print(f"  ")
        print(f"  ⇒ U_EM(8_3) = 1492 n'est pas dérivable des invariants")
        print(f"    topologiques standards (Jones polynomial, volume, genre,")
        print(f"    signature, et combinaisons). C'est un paramètre")
        print(f"    phénoménologique du modèle TQNT V4.")
    print("=" * 100)


if __name__ == "__main__":
    main()
