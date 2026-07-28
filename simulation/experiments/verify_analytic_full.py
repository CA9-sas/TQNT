r"""
VÉRIFICATION FINALE des valeurs analytiques sur le spectre.

Hypothèse analytique :
   a = √(h_{j=5/2}(SU(2)_8) / π) = √(7/(8π))   ≈ 0.52775
   p = q = 2·h_{j=3/2}(SU(2)_8) = 3/4           = 0.75
   x = -√(2·dim SU(2)) = -√6                    ≈ -2.44949
   β = c(SU(2)_8) - 2·h_{j=3/2} = 33/20         = 1.65

ρ_0 et k_b sont déterminés par les ancrages exacts proton et neutron.

On évalue sur les 45 particules canoniques et on compare au RMS de
l'optimum numérique (0.0595).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from collections import Counter

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
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

# ─── VALEURS ANALYTIQUES (SU(2)_8) ───
A_ANALYTIC = np.sqrt(7 / (8 * np.pi))   # √(h_{5/2}/π) = √(7/(8π))
P_ANALYTIC = 3 / 4                      # 2·h_{3/2}
Q_ANALYTIC = 3 / 4                      # 2·h_{3/2}
X_ANALYTIC = -np.sqrt(6)                # -√(2·dim SU(2))
B_ANALYTIC = 33 / 20                    # c - 2·h_{3/2}

# Valeurs optimisées pour comparaison
A_OPT = 0.52825
P_OPT = 0.77955
Q_OPT = 0.72177
X_OPT = -2.45599
B_OPT = 1.65095

print("Calcul Frenet (50000 points)…")
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


def operator(comps, a, p, q, x, rho0, beta, kb=1.0, neutron_mode=False):
    n = len(comps)
    ws = []
    for K_idx, K in enumerate(comps):
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        w = M_NAIVE[K] * T ** p
        if neutron_mode and K == "7_2":
            counted = sum(1 for k in comps[:K_idx] if k == "7_2")
            if counted == 1:
                w *= kb
        ws.append(w)
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


def solve_rho0(a, p, q, x, beta):
    def f(log_rho0):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, np.exp(log_rho0), beta) - M_P
    return float(np.exp(brentq(f, -10, 10, xtol=1e-10)))


def solve_kb(a, p, q, x, rho0, beta):
    def f(log_kb):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, rho0, beta,
                        kb=np.exp(log_kb), neutron_mode=True) - M_N
    return float(np.exp(brentq(f, -5, 5, xtol=1e-10)))


SPECTRUM = [
    ("e⁻",    ["3_1"],                       0.511),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    ("K±",    ["4_1", "4_1", "5_2"],         493.68),
    ("K⁰_S",  ["3_1", "6_1", "6_1"],         497.611),
    ("η",     ["5_2", "5_2", "6_1"],         547.86),
    ("η'",    ["5_1", "5_2", "5_2"],         957.78),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "7_1", "7_1"],         1019.46),
    ("J/ψ",   ["5_2", "6_1", "6_2"],         3096.9),
    ("ψ(2S)", ["5_1", "7_7", "7_7"],         3686.10),
    ("η_c",   ["3_1", "7_1", "9_1"],         2983.9),
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("D±",    ["7_7", "9_1"],                1869.66),
    ("D_s",   ["6_3", "9_1"],                1968.34),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("B⁰",    ["7_6", "7_7"],                5279.65),
    ("B_s",   ["7_6", "7_7"],                5366.92),
    ("B_c",   ["5_1", "10_1"],               6274.9),
    ("p",     ["3_1", "7_2", "7_2"],         M_P),
    ("n",     ["3_1", "7_2", "7_2"],         M_N),
    ("Λ",     ["5_1", "7_7"],                1115.7),
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Σ⁰",    ["5_1", "6_3"],                1192.642),
    ("Σ⁻",    ["5_1", "6_3"],                1197.449),
    ("Ξ⁰",    ["5_1", "5_1", "5_2"],         1314.86),
    ("Ξ⁻",    ["4_1", "9_1", "9_1"],         1321.71),
    ("Δ",     ["4_1", "7_2", "7_2"],         1232.0),
    ("Σ*",    ["3_1", "5_2", "7_1"],         1383.7),
    ("Ξ*",    ["3_1", "6_1", "7_2"],         1531.8),
    ("Ω⁻",    ["3_1", "6_1", "7_1"],         1672.0),
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Σc",    ["5_2", "5_2", "8_1"],         2453.97),
    ("Ξc",    ["5_1", "5_2", "7_2"],         2467.94),
    ("Ωc",    ["6_1", "6_1", "8_1"],         2695.2),
    ("Λb",    ["5_1", "6_1", "6_3"],         5619.6),
    ("Σb",    ["4_1", "7_7", "9_1"],         5811.0),
    ("Ξb",    ["6_1", "6_3", "6_3"],         5797.0),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["7_4", "9_1", "8_3"],         91188.0),
    ("H",     ["7_7", "8_2", "8_3"],         125250.0),
    ("t",     ["3_1", "7_5", "8_3"],         173100.0),
]


def evaluate(a, p, q, x, beta, rho0, kb):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    res_log = []
    for nm, comps, me in SPECTRUM:
        if nm == "n":
            m_pred = operator(comps, a, p, q, x, rho0, beta, kb=kb, neutron_mode=True)
        else:
            m_pred = operator(comps, a, p, q, x, rho0, beta)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, comps, me, m_pred, err, flag))
        if me > 0 and nm not in ("e⁻", "p", "n"):
            res_log.append(np.log(m_pred / me))
    rms = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    return rows, flags, rms


def main():
    print("=" * 100)
    print("  VÉRIFICATION DES VALEURS ANALYTIQUES SU(2)_8 sur le SPECTRE 45")
    print("=" * 100)

    print(f"\n  Valeurs analytiques :")
    print(f"    a = √(h_{{5/2}}/π) = √(7/(8π)) = {A_ANALYTIC:.6f}")
    print(f"    p = q = 2·h_{{3/2}} = 3/4      = {P_ANALYTIC:.6f}")
    print(f"    x = -√(2·dim SU(2)) = -√6     = {X_ANALYTIC:.6f}")
    print(f"    β = c - 2·h_{{3/2}} = 33/20    = {B_ANALYTIC:.6f}")

    print(f"\n  Calcul de ρ_0 et k_b par les ancrages…")
    rho0 = solve_rho0(A_ANALYTIC, P_ANALYTIC, Q_ANALYTIC, X_ANALYTIC, B_ANALYTIC)
    kb = solve_kb(A_ANALYTIC, P_ANALYTIC, Q_ANALYTIC, X_ANALYTIC, rho0, B_ANALYTIC)
    print(f"    ρ_0 = {rho0:.5f}")
    print(f"    k_b = {kb:.6f}")

    rows, flags, rms = evaluate(
        A_ANALYTIC, P_ANALYTIC, Q_ANALYTIC, X_ANALYTIC, B_ANALYTIC, rho0, kb)

    # Vérification ancrages
    print(f"\n  ─── ANCRAGES ───")
    for nm in ("e⁻", "p", "n"):
        for r in rows:
            if r[0] == nm:
                print(f"    {nm}: Δ = {r[4]:+.6f}%")
                break

    print(f"\n  ─── PRÉDICTIONS COMPLÈTES ───")
    print(f"  {'particule':>6} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 65)
    for nm, comps, me, m_pred, err, flag in rows:
        print(f"  {nm:>6} {'+'.join(comps):>17} {me:>11.3f} {m_pred:>11.3f} "
              f"{err:>+8.3f} {flag:>5}")

    n5 = flags["★★"] + flags["★"]
    n20 = n5 + flags["✓"]
    total = len(SPECTRUM)
    print(f"\n  ─── BILAN ANALYTIQUE ───")
    print(f"    ★★ (<0.5%) : {flags['★★']}/{total}")
    print(f"    ★  (<5%)   : {flags['★']}/{total}")
    print(f"    ✓  (<20%)  : {flags['✓']}/{total}")
    print(f"    ordinaire  : {flags[' ']}/{total}")
    print(f"    outlier    : {flags['?']}/{total}")
    print(f"  ")
    print(f"    Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"    Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")
    print(f"    RMS log (hors ancrés) = {rms:.4f}")

    print(f"\n  ─── COMPARAISON ANALYTIQUE vs NUMÉRIQUE ───")
    print(f"  {'critère':<25} {'analytique':>12} {'numérique':>12} {'ratio':>8}")
    print("  " + "-" * 60)
    print(f"  {'a':<25} {A_ANALYTIC:>12.5f} {A_OPT:>12.5f} {A_ANALYTIC/A_OPT:>8.4f}")
    print(f"  {'p':<25} {P_ANALYTIC:>12.5f} {P_OPT:>12.5f} {P_ANALYTIC/P_OPT:>8.4f}")
    print(f"  {'q':<25} {Q_ANALYTIC:>12.5f} {Q_OPT:>12.5f} {Q_ANALYTIC/Q_OPT:>8.4f}")
    print(f"  {'x':<25} {X_ANALYTIC:>12.5f} {X_OPT:>12.5f} {X_ANALYTIC/X_OPT:>8.4f}")
    print(f"  {'β':<25} {B_ANALYTIC:>12.5f} {B_OPT:>12.5f} {B_ANALYTIC/B_OPT:>8.4f}")
    print(f"  {'RMS log':<25} {rms:>12.5f} {0.0478:>12.5f} {rms/0.0478:>8.4f}")
    print(f"  {'★+★★':<25} {n5:>12d} {15:>12d}")
    print(f"  {'<20%':<25} {n20:>12d} {45:>12d}")

    print(f"\n  ─── VERDICT ───")
    if rms < 0.15 and n20 >= 40:
        print(f"  ★★ VALIDATION : les valeurs analytiques reproduisent le spectre")
        print(f"     à RMS log = {rms:.3f}, {n20}/{total} à <20%.")
        print(f"     L'opérateur SU(2)_8 est viable comme expression CLOSE.")
    elif rms < 0.3:
        print(f"  ✓ Acceptable mais moins précis que l'optimum numérique.")
    else:
        print(f"  ✗ Trop de divergence. Une perturbation manque.")
    print("=" * 100)


if __name__ == "__main__":
    main()
