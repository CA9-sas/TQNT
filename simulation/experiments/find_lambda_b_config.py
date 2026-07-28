r"""
RECHERCHE DE LA MEILLEURE CONFIGURATION pour Λb.

Λb (= 5619.6 MeV) est le seul outlier restant après les corrections
du commit 896649a. Sa config actuelle 5_2+5_2+8_2 donne +24% d'erreur
avec les paramètres ré-optimisés.

On cherche parmi toutes les combinaisons de 3 nœuds (sans 8_3) celle
qui prédit le mieux Λb.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement

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

# Paramètres optimisés (commit e771a40)
PARAMS = (0.5051, 0.7896, 0.7138, -2.3696, 8.5297, 1.5410)

M_LAMBDA_B = 5619.6


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


def operator(comps):
    a, p, q, x, rho0, beta = PARAMS
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


def main():
    print("=" * 100)
    print("  RECHERCHE DE LA MEILLEURE CONFIG POUR Λb (= 5619.6 MeV)")
    print("=" * 100)

    # Config actuelle
    current = ["5_2", "5_2", "8_2"]
    m_curr = operator(current)
    err_curr = (m_curr - M_LAMBDA_B) / M_LAMBDA_B * 100
    print(f"\n  Config actuelle : {'+'.join(current):>17}  → "
          f"m_pred = {m_curr:>10.2f}  Δ = {err_curr:+.2f}%")

    # Recherche exhaustive 3-component (sans 8_3 puisque Λb est un baryon b)
    pool = [K for K in KNOTS if K != "8_3"]
    print(f"\n  Recherche exhaustive sur {len(pool)} nœuds (sans 8_3),")
    print(f"  toutes combinaisons à 3 composantes…")

    candidates = []
    for combo in combinations_with_replacement(pool, 3):
        m_pred = operator(list(combo))
        err = (m_pred - M_LAMBDA_B) / M_LAMBDA_B * 100
        candidates.append((list(combo), m_pred, err))

    candidates.sort(key=lambda r: abs(r[2]))

    print(f"\n  Top 10 alternatives (par erreur absolue) :")
    print(f"  {'rang':>4} {'config':>17} {'m_pred':>12} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 55)
    for i, (cfg, mp, err) in enumerate(candidates[:10]):
        flag = "★★" if abs(err) < 0.5 else ("★" if abs(err) < 5 else "✓")
        print(f"  {i+1:>4} {'+'.join(cfg):>17} {mp:>12.2f} {err:>+8.3f}% {flag:>5}")

    # Pour Λb, recherche également avec 4 composantes (b-baryon, peut être plus lourd)
    print(f"\n  Recherche avec 4 composantes (au cas où) :")
    candidates_4 = []
    for combo in combinations_with_replacement(pool, 4):
        m_pred = operator(list(combo))
        err = (m_pred - M_LAMBDA_B) / M_LAMBDA_B * 100
        candidates_4.append((list(combo), m_pred, err))
    candidates_4.sort(key=lambda r: abs(r[2]))
    print(f"  Top 5 :")
    for i, (cfg, mp, err) in enumerate(candidates_4[:5]):
        flag = "★★" if abs(err) < 0.5 else ("★" if abs(err) < 5 else "✓")
        print(f"  {i+1:>4} {'+'.join(cfg):>21} {mp:>12.2f} {err:>+8.3f}% {flag:>5}")

    print(f"\n  ─── ANALYSE ───")
    best = candidates[0]
    if abs(best[2]) < 5:
        print(f"  ✓ Meilleure config 3-comp : {'+'.join(best[0])} à Δ={best[2]:+.2f}%")
    else:
        print(f"  Pas de bonne config 3-comp ; meilleure = {'+'.join(best[0])} "
              f"à {best[2]:+.2f}%")
    best_4 = candidates_4[0]
    if abs(best_4[2]) < abs(best[2]):
        print(f"  ✓ Config 4-comp meilleure : {'+'.join(best_4[0])} à Δ={best_4[2]:+.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
