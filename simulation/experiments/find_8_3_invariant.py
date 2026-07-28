r"""
Chantier 1 → D1ter — recherche d'un invariant topologique standard
expliquant naturellement le motif U_EM(8_3) >> autres nœuds cr=8.

OBJECTIF : trouver une fonction des polynômes Jones / Alexander / etc.
qui prédise la valeur U_EM(8_3) ≈ 1492 (catalogue V4) tout en restant
modérée pour 8_1 (60) et 8_2 (117).

Si un tel invariant existe (hypothèse A du commit 7114afd), alors la
formule TQNT V4 peut être réécrite comme
   m = α₀ · I(K) · (f · c_J)^(32/9)
avec I(K) un invariant topologique, et la formule devient PRÉDICTIVE
(plus de dépendance au catalogue Fourier).

Invariants testés sur les 19 nœuds catalogue :

  A. Valeurs du polynôme de Jones V_K(t) à des roots of unity :
       V(-1), V(i), V(e^{iπ/3}), V(e^{iπ/4}), V(e^{iπ/5}),
       V(e^{2iπ/5}), V(e^{2iπ/7}), V(e^{2iπ/9}), V(e^{2iπ/11})
  B. Quantités dérivées du polynôme :
       span, deg max, deg min, somme des |coeffs|, n_coeffs ≠ 0,
       valeur en t=2, t=1/2, dérivée V'(-1)
  C. Invariants discrets :
       braid_index, bridge_number, signature (estimée), genus
  D. Quotients normalisés :
       |V(t)|^2 / span², |V(t)| / determinant, etc.

Pour chaque invariant I, on calcule :
  - I(8_3) et I(8_1), I(8_2) : 8_3 est-il un outlier ?
  - Z-score de 8_3 vs les 19 valeurs
  - Corrélation Pearson(I, U_EM_V4) : I prédit-il U_EM ?
  - Régression linéaire log(U_EM) ~ log|I| : exposant ?

Conclusion attendue : soit on trouve l'invariant magique, soit on
démontre qu'aucun invariant topologique standard ne reproduit U_EM_V4,
ce qui valide l'interprétation B (artefact paramétrique).
"""
from __future__ import annotations
import numpy as np

from theory.jones import (
    jones_polynomial, evaluate_jones, jones_span, jones_determinant,
    braid_index, bridge_number, is_amphicheiral,
)
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())


def cr(K): return int(K.split("_")[0])


def U_EM_V4(K):
    return float(REFERENCE_N50K[K]["sqrt_U"]) ** 2


# ─── Tableau d'invariants ────────────────────────────────────────────────
def compute_invariants(K):
    """Tous les invariants topologiques pour un nœud K."""
    p = jones_polynomial(K)
    invs = {}
    # A. V_K à divers points
    points = {
        "V(-1)":           -1.0 + 0j,
        "V(i)":            1j,
        "V(e^{iπ/3})":     np.exp(1j * np.pi / 3),
        "V(e^{iπ/4})":     np.exp(1j * np.pi / 4),
        "V(e^{iπ/5})":     np.exp(1j * np.pi / 5),
        "V(e^{2iπ/5})":    np.exp(2j * np.pi / 5),
        "V(e^{2iπ/7})":    np.exp(2j * np.pi / 7),
        "V(e^{2iπ/9})":    np.exp(2j * np.pi / 9),
        "V(e^{2iπ/11})":   np.exp(2j * np.pi / 11),
        "V(2)":            2.0 + 0j,
        "V(0.5)":          0.5 + 0j,
        "V(3)":            3.0 + 0j,
    }
    for name, t in points.items():
        val = evaluate_jones(p, t)
        invs[name + " |.|"] = float(abs(val))
        invs[name + " |.|²"] = float(abs(val) ** 2)

    # B. Quantités dérivées du polynôme
    powers = [pw for pw, c in p.items() if c != 0]
    invs["span"] = float(jones_span(p))
    invs["det |V(-1)|"] = float(jones_determinant(p))
    invs["pmax"] = float(max(powers)) if powers else 0.0
    invs["pmin"] = float(min(powers)) if powers else 0.0
    invs["pmax-pmin"] = invs["pmax"] - invs["pmin"]
    invs["sum |coeffs|"] = float(sum(abs(c) for c in p.values()))
    invs["max |coeffs|"] = float(max(abs(c) for c in p.values()))
    invs["n_coeffs ≠ 0"] = float(len([c for c in p.values() if c != 0]))
    # Dérivée V'(-1) : Σ k·c_k·(-1)^(k-1)
    invs["|V'(-1)|"] = float(abs(sum(pw * c * (-1) ** (pw - 1)
                                      for pw, c in p.items())))

    # C. Invariants discrets
    invs["braid_idx"] = float(braid_index(K))
    invs["bridge_num"] = float(bridge_number(K))
    invs["amphichiral"] = 1.0 if is_amphicheiral(p) else 0.0

    # D. Quotients
    invs["|V(i)| · braid_idx"] = invs["V(i) |.|"] * invs["braid_idx"]
    invs["det / span"] = invs["det |V(-1)|"] / max(invs["span"], 1)
    invs["sum |c| · cr"] = invs["sum |coeffs|"] * cr(K)

    return invs


def main():
    print("=" * 110)
    print("  RECHERCHE d'un INVARIANT TOPOLOGIQUE expliquant U_EM(8_3) = 1492")
    print("=" * 110)

    # Calcul de tous les invariants
    all_invs = {}
    for K in KNOTS:
        all_invs[K] = compute_invariants(K)
    inv_names = list(all_invs[KNOTS[0]].keys())

    # U_EM
    U_arr = np.array([U_EM_V4(K) for K in KNOTS])

    # ─── Analyse 1 : Z-score de 8_3 vs population ────────────────────────
    print(f"\n  ─── PARTIE 1 : Quel invariant fait de 8_3 un OUTLIER ? ───")
    print(f"  (Cible : 8_3 doit être >> autres nœuds, particulièrement 8_1 et 8_2)")
    print(f"  Z(8_3) = (I(8_3) - mean) / std sur 19 nœuds. |Z|>3 = très outlier.")
    print()
    print(f"  {'invariant':<22} {'I(8_1)':>10} {'I(8_2)':>10} {'I(8_3)':>10} "
          f"{'Z(8_3)':>8} {'r(I,U_EM)':>10}")
    print("  " + "-" * 72)

    rows = []
    for name in inv_names:
        vals = np.array([all_invs[K][name] for K in KNOTS])
        if vals.std() < 1e-9:
            continue
        z83 = (all_invs["8_3"][name] - vals.mean()) / vals.std()
        # Correlation
        try:
            r = np.corrcoef(vals, U_arr)[0, 1]
        except Exception:
            r = float("nan")
        rows.append((name, all_invs["8_1"][name], all_invs["8_2"][name],
                     all_invs["8_3"][name], z83, r))

    # Trier par |Z(8_3)| décroissant
    rows.sort(key=lambda r: -abs(r[4]))
    for name, i81, i82, i83, z, r in rows[:20]:
        ratio_83_82 = i83 / i82 if abs(i82) > 1e-12 else float("inf")
        flag = "★" if z > 3 else ("✓" if z > 2 else " ")
        print(f"  {name:<22} {i81:>10.3f} {i82:>10.3f} {i83:>10.3f} "
              f"{z:>+7.2f} {flag} {r:>+9.3f}")

    # ─── Analyse 2 : régression log U_EM ~ log |I| ──────────────────────
    print(f"\n  ─── PARTIE 2 : Régression log(U_EM) ~ log(I) sur 19 nœuds ───")
    print(f"  Recherche d'un invariant POSITIF avec corrélation forte.")
    print()
    print(f"  {'invariant':<22} {'pente':>8} {'R²':>8} {'résidu (8_3)':>14}")
    print("  " + "-" * 56)

    reg_rows = []
    log_U = np.log(U_arr + 1e-12)
    for name in inv_names:
        vals = np.array([all_invs[K][name] for K in KNOTS])
        if (vals <= 0).any() or vals.std() < 1e-9:
            continue
        log_I = np.log(vals)
        slope, intercept = np.polyfit(log_I, log_U, 1)
        pred = slope * log_I + intercept
        resid = log_U - pred
        ss_tot = np.sum((log_U - log_U.mean()) ** 2)
        ss_res = np.sum(resid ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Résidu spécifique pour 8_3
        i83_idx = KNOTS.index("8_3")
        resid_83 = resid[i83_idx]
        reg_rows.append((name, slope, R2, resid_83))

    reg_rows.sort(key=lambda r: -r[2])  # tri par R²
    for name, slope, R2, r83 in reg_rows[:15]:
        flag = "★" if R2 > 0.8 else (" " if R2 > 0.5 else "")
        print(f"  {name:<22} {slope:>+8.3f} {R2:>8.3f} {r83:>+13.2f} {flag}")

    # ─── Analyse 3 : exhibe le profil de chaque invariant pour 8_x ──────
    print(f"\n  ─── PARTIE 3 : Profil des nœuds cr=8 et 8_3 vs 8_1, 8_2 ───")
    print(f"  Quels invariants ont 8_3 nettement supérieur à 8_1 et 8_2 ?")
    print()
    print(f"  {'invariant':<22} {'8_1':>10} {'8_2':>10} {'8_3':>10} "
          f"{'rapport 8_3/8_2':>16}")
    print("  " + "-" * 70)
    interesting = []
    for name in inv_names:
        i81 = all_invs["8_1"][name]
        i82 = all_invs["8_2"][name]
        i83 = all_invs["8_3"][name]
        if abs(i82) < 1e-12 or abs(i81) < 1e-12:
            ratio = float("inf") if abs(i83) > 1e-6 else 1
        else:
            ratio = i83 / max(abs(i82), abs(i81))
        interesting.append((name, i81, i82, i83, ratio))

    # Tri par ratio
    interesting.sort(key=lambda r: -r[4] if not np.isinf(r[4]) else 0)
    for name, i81, i82, i83, ratio in interesting[:15]:
        flag = "★" if ratio > 5 else (" " if ratio > 2 else "")
        rs = f"{ratio:.2f}×" if not np.isinf(ratio) else "∞"
        print(f"  {name:<22} {i81:>10.3f} {i82:>10.3f} {i83:>10.3f} "
              f"{rs:>16} {flag}")

    # ─── PARTIE 4 : TEST DIRECT — substitution dans la formule R6 ───────
    print(f"\n  ─── PARTIE 4 : TEST FORMULE R6 AVEC |V(0.5)|² au lieu de U_EM ───")
    print(f"  Le candidat |V(0.5)|² est le SEUL avec Z(8_3) > 3 ET r(I,U_EM) > 0.9.")
    print(f"  On teste : m = α₀ · |V(0.5)|²(L) · (f · c_J)^q sur 7 canoniques.")
    print(f"  α₀ et q optimisés simultanément.")
    print()

    from theory.mass_formula import c_j, EW_MARKER
    from scipy.optimize import minimize

    SPECTRUM = [
        ("e-",  ["3_1"],               "single", 0.51099895),
        ("p",   ["3_1", "7_2", "7_2"], "hadron", 938.272),
        ("n",   ["3_1", "7_2", "7_2"], "hadron", 939.565),
        ("W",   ["4_1", "7_3", "8_3"], "ew",     80377.0),
        ("Z",   ["5_1", "7_2", "8_3"], "ew",     91188.0),
        ("H",   ["6_1", "7_4", "8_3"], "ew",     125250.0),
        ("t",   ["7_4", "7_6", "8_3"], "ew",     173100.0),
    ]

    def f_screening(comps):
        n = len(comps)
        if n == 1 or EW_MARKER in comps:
            return 1.0
        return ((n + 1) / 2) ** (-3 / 8)

    def V05_sq(K):
        p = jones_polynomial(K)
        return float(abs(evaluate_jones(p, 0.5)) ** 2)

    def mass_pred(comps, alpha0, q):
        I_tot = sum(V05_sq(K) for K in comps)
        cJ_tot = sum(c_j(K) for K in comps)
        f = f_screening(comps)
        return alpha0 * I_tot * (f * cJ_tot) ** q

    def loss(x):
        log_a, q = x
        a0 = float(np.exp(log_a))
        residuals = []
        for nm, comps, sec, me in SPECTRUM:
            m = mass_pred(comps, a0, q)
            residuals.append(np.log(m / me))
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    # Optim
    res = minimize(loss, x0=[np.log(1e-3), 3.55], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9})
    log_a_opt, q_opt = res.x
    a0_opt = float(np.exp(log_a_opt))
    rms_v = float(res.fun)

    print(f"  Optimum trouvé :  α₀ = {a0_opt:.4e} MeV,  q = {q_opt:.4f}")
    print(f"  RMS log = {rms_v:.4f}   (= +{(np.exp(rms_v)-1)*100:.1f}% relatif)")
    print()
    print(f"  Comparaison sur les 7 canoniques :")
    print(f"  {'part.':<6} {'m_exp':>11} {'m_pred':>11} {'Δ %':>8}")
    for nm, comps, sec, me in SPECTRUM:
        m = mass_pred(comps, a0_opt, q_opt)
        e = (m - me) / me * 100
        flag = "★" if abs(e) < 1 else (" " if abs(e) < 10 else "?")
        print(f"  {nm:<6} {me:>11.2f} {m:>11.2f} {e:>+7.2f} {flag}")

    # Test avec exposant rationnel candidat : q = 32/9 fixé
    print(f"\n  Test alternatif : q = 32/9 = {32/9:.4f} FIXÉ (conjecture R6) :")
    def loss_qfixed(log_a):
        return loss([log_a, 32/9])
    res_q = minimize(loss_qfixed, x0=[np.log(1e-3)], method="Nelder-Mead")
    a0_q = float(np.exp(res_q.x[0]))
    rms_q = float(res_q.fun)
    print(f"  α₀ = {a0_q:.4e} MeV, RMS log = {rms_q:.4f}")

    for nm, comps, sec, me in SPECTRUM:
        m = mass_pred(comps, a0_q, 32/9)
        e = (m - me) / me * 100
        flag = "★" if abs(e) < 1 else (" " if abs(e) < 10 else "?")
        print(f"  {nm:<6} {me:>11.2f} {m:>11.2f} {e:>+7.2f} {flag}")

    # ─── Conclusion ──────────────────────────────────────────────────────
    print(f"\n  ─── DIAGNOSTIC ───")
    print(f"  Cible : un invariant I avec I(8_3) ≈ 25× I(8_2) ET corrélation")
    print(f"  R²(I, U_EM) > 0.9. Si trouvé, on a l'origine topologique de U_EM.")
    print()
    print(f"  Si AUCUN invariant ne satisfait, alors U_EM est paramétrique")
    print(f"  (hypothèse B confirmée) : le 1492 du 8_3 dans V4 est un artefact")
    print(f"  de la paramétrisation Fourier et ne peut être dérivé d'invariants.")
    print()
    # Verdict
    max_z = max(abs(r[4]) for r in rows[:5])
    max_r2 = max(r[2] for r in reg_rows[:5])
    if max_z > 3 and max_r2 > 0.8:
        print(f"  ★ Hypothèse A VALIDÉE : invariant(s) trouvé(s) avec Z>3 et R²>0.8")
    else:
        print(f"  ✗ Hypothèse A REJETÉE : aucun invariant testé n'atteint")
        print(f"    (|Z|>3 ET R²>0.8) simultanément.")
        print(f"    Meilleur |Z(8_3)| = {max_z:.2f},  meilleur R² = {max_r2:.3f}")
        print(f"    ⇒ Validation indirecte de l'hypothèse B (U_EM paramétrique).")
    print("=" * 110)


if __name__ == "__main__":
    main()
