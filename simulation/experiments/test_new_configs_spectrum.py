r"""
TEST DU SPECTRE COMPLET avec les NOUVELLES CONFIGURATIONS optimales.

On compare deux scénarios :
  A. Paramètres opérateur FIXÉS (a, p, q, x, ρ_0, β du commit précédent)
     → on observe l'amélioration mécanique du seul changement de configs
  B. Paramètres opérateur RE-OPTIMISÉS sur le nouveau spectre
     → on obtient le vrai pic de précision avec configs + paramètres
        cohérents

Le but : confirmer que les nouvelles configs (commit 896649a) donnent
un spectre beaucoup plus précis.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from collections import Counter

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3
M_L = M_E * (1 - F_KAPPA) / 3

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

print("Calcul Frenet…")
U_KT = {}
TOPO = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
    pj = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(pj),
    }
P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}

# Configurations originales
SPECTRUM_OLD = [
    ("e⁻",    ["3_1"],                       0.511),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    ("K±",    ["4_1", "4_1", "5_2"],         493.68),
    ("η",     ["3_1", "4_1", "5_1"],         547.86),       # outlier
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "4_1", "6_2"],         1019.46),      # outlier
    ("J/ψ",   ["4_1", "5_2", "9_1"],         3096.9),       # outlier
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("p",     ["3_1", "7_2", "7_2"],         938.272),
    ("n",     ["3_1", "7_2", "7_2"],         939.565),
    ("Λ",     ["4_1", "7_4"],                1115.7),       # outlier
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),       # outlier
    ("Ω⁻",    ["5_2", "5_2", "7_2"],         1672.0),       # outlier
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.6),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91188.0),      # outlier
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),     # outlier
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),     # outlier
]

# Configurations NOUVELLES (résultat du commit 896649a)
SPECTRUM_NEW = [
    ("e⁻",    ["3_1"],                       0.511),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    ("K±",    ["4_1", "4_1", "5_2"],         493.68),
    ("η",     ["5_2", "5_2", "6_1"],         547.86),       # NEW
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "7_1", "7_1"],         1019.46),      # NEW
    ("J/ψ",   ["5_2", "6_1", "6_2"],         3096.9),       # NEW
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("p",     ["3_1", "7_2", "7_2"],         938.272),
    ("n",     ["3_1", "7_2", "7_2"],         939.565),
    ("Λ",     ["5_1", "7_7"],                1115.7),       # NEW
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Δ",     ["4_1", "7_2", "7_2"],         1232.0),       # NEW
    ("Ω⁻",    ["3_1", "6_1", "7_1"],         1672.0),       # NEW
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.6),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["7_4", "9_1", "8_3"],         91188.0),      # NEW
    ("H",     ["7_7", "8_2", "8_3"],         125250.0),     # NEW
    ("t",     ["3_1", "7_5", "8_3"],         173100.0),     # NEW
]


def operator(comps, params):
    a, p, q, x, rho0, beta = params
    n = len(comps)
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** p)
    if abs(a) < 1e-8:
        skeleton = float(np.prod(ws) ** (1 / len(ws)))
    else:
        skeleton = float(sum(w ** a for w in ws) ** (1 / a))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-x) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = rho0 * TOPO[K]["cr"] ** (-beta)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** q


def loss_for(spectrum):
    def f(params):
        res = []
        for nm, comps, me in spectrum:
            try:
                m_pred = operator(comps, params)
            except (OverflowError, ValueError, ZeroDivisionError):
                return 1e6
            if m_pred > 0 and np.isfinite(m_pred):
                res.append(np.log(m_pred / me))
            else:
                res.append(20)
        return float(np.sqrt(np.mean(np.array(res) ** 2)))
    return f


def evaluate(spectrum, params):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    for nm, comps, me in spectrum:
        m_pred = operator(comps, params)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, comps, me, m_pred, err, flag))
    res_log = [np.log(r[3]/r[2]) for r in rows]
    rms = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    return rows, flags, rms


def main():
    print("\n" + "=" * 100)
    print("  TEST DU SPECTRE AVEC NOUVELLES CONFIGURATIONS")
    print("=" * 100)

    PARAMS_OLD = (0.414, 0.773, 0.739, -1.473, 14.159, 1.752)

    # Scénario A : paramètres anciens, nouvelles configs
    print(f"\n  ─── A. Paramètres FIXÉS, nouvelles configs ───")
    rows_A, flags_A, rms_A = evaluate(SPECTRUM_NEW, PARAMS_OLD)
    n5_A = flags_A["★★"] + flags_A["★"]
    n20_A = n5_A + flags_A["✓"]
    print(f"  RMS log = {rms_A:.4f}, <5% : {n5_A}/25, <20% : {n20_A}/25")

    # Scénario B : re-optimisation des paramètres
    print(f"\n  ─── B. Paramètres RE-OPTIMISÉS sur nouveau spectre ───")
    best = None
    for a0 in [0.3, 0.4]:
        for p0 in [0.6, 0.8]:
            for q0 in [0.6, 0.8]:
                for x0 in [-1.5, 0, 0.4]:
                    for rho0 in [10, 14, 20]:
                        for beta in [1.5, 1.8, 2.1]:
                            x_init = [a0, p0, q0, x0, rho0, beta]
                            res = minimize(loss_for(SPECTRUM_NEW), x_init,
                                           method="Nelder-Mead",
                                           options={"maxiter": 3000, "xatol": 1e-7})
                            if best is None or res.fun < best.fun:
                                best = res
    PARAMS_NEW = tuple(best.x)
    rows_B, flags_B, rms_B = evaluate(SPECTRUM_NEW, PARAMS_NEW)
    n5_B = flags_B["★★"] + flags_B["★"]
    n20_B = n5_B + flags_B["✓"]
    print(f"  Paramètres optimisés :")
    print(f"    a={PARAMS_NEW[0]:.4f}, p={PARAMS_NEW[1]:.4f}, q={PARAMS_NEW[2]:.4f}")
    print(f"    x={PARAMS_NEW[3]:.4f}, ρ_0={PARAMS_NEW[4]:.4f}, β={PARAMS_NEW[5]:.4f}")
    print(f"  RMS log = {rms_B:.4f}, <5% : {n5_B}/25, <20% : {n20_B}/25")

    # Tableau final B
    print(f"\n  ─── PRÉDICTIONS COMPLÈTES (B : re-optimisé) ───")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 70)
    for nm, comps, me, m_pred, err, flag in rows_B:
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} {m_pred:>11.2f} "
              f"{err:>+8.2f} {flag:>5}")

    # ─── COMPARAISON FINALE ─────────────────────────────────────────
    rows_old, flags_old, rms_old = evaluate(SPECTRUM_OLD, PARAMS_OLD)
    n5_old = flags_old["★★"] + flags_old["★"]
    n20_old = n5_old + flags_old["✓"]

    print(f"\n  ═══ COMPARAISON FINALE ═══")
    print(f"  {'scénario':<45} {'RMS log':>9} {'★+★★':>6} {'<20%':>6}")
    print("  " + "-" * 70)
    print(f"  {'(0) configs originales + params V4':<45} {rms_old:>9.4f} "
          f"{n5_old:>6} {n20_old:>6}")
    print(f"  {'(A) nouvelles configs + params V4 FIXES':<45} {rms_A:>9.4f} "
          f"{n5_A:>6} {n20_A:>6}")
    print(f"  {'(B) nouvelles configs + params RE-OPTIMISÉS':<45} {rms_B:>9.4f} "
          f"{n5_B:>6} {n20_B:>6}")

    print(f"\n  ─── BILAN ───")
    print(f"  Progression de (0) à (B) :")
    print(f"    RMS log :  {rms_old:.3f} → {rms_B:.3f}  ({(rms_B-rms_old)/rms_old*100:+.0f}%)")
    print(f"    ★+★★ :    {n5_old}/25 → {n5_B}/25")
    print(f"    <20% :    {n20_old}/25 → {n20_B}/25")
    print("=" * 100)


if __name__ == "__main__":
    main()
