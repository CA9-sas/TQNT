r"""
VÉRIFICATION des 32 particules exotiques avec la formule ANALYTIQUE SU(2)_8.

On utilise les valeurs analytiques :
   a = √(7/(8π))
   p = q = 3/4
   x = -√6
   β = 33/20
   ρ_0 = 10 - 3/(8π)
   k_b = (2(m_n^a - m_e^a)/(m_p^a - m_e^a) - 1)^(1/a)

Pour les 32 particules exotiques (mega_spectrum.py), on RE-SEARCHE les
configurations optimales avec ces paramètres fixés (les anciennes configs
ont été trouvées avec les paramètres numériques).
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement

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

# ═══════ PARAMÈTRES ANALYTIQUES SU(2)_8 ═══════
K_LEVEL = 8
H_COX = 2

A = np.sqrt(7 / (8 * np.pi))        # √(h_{5/2}/π)
P = 3 / 4                            # 2·h_{3/2}
Q = 3 / 4
X = -np.sqrt(6)                      # -√(2·dim SU(2))
B = 33 / 20                          # c - 2·h_{3/2}
RHO_0 = (K_LEVEL + H_COX) - 3 / (8 * np.pi)  # (k+h^∨) - 3a²/7

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


# k_b analytique (Hölder pur)
K_B = ((2 * (M_N**A - M_E**A) / (M_P**A - M_E**A) - 1) ** (1/A))


def operator(comps):
    n = len(comps)
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** P)
    skeleton = float(sum(w ** A for w in ws) ** (1 / A))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-X) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = RHO_0 * TOPO[K]["cr"] ** (-B)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** Q


def search_config(m_exp, n_comp):
    pool = [K for K in KNOTS if K != "8_3"]
    best = (None, float("inf"), float("inf"))
    for combo in combinations_with_replacement(pool, n_comp):
        cfg = list(combo)
        mp = operator(cfg)
        err = abs((mp - m_exp) / m_exp * 100)
        if err < best[2]:
            best = (cfg, mp, err)
    return best


# 32 particules exotiques (depuis mega_spectrum.py)
EXOTIC = [
    # Charmonium excitations
    ("χc0",      3414.71,  3),
    ("χc1",      3510.67,  3),
    ("χc2",      3556.17,  3),
    ("ψ(3770)",  3773.7,   3),
    ("ψ(4040)",  4039.0,   3),
    ("η_c(2S)",  3637.5,   3),
    # Bottomonium excitations
    ("χb0",      9859.44,  3),
    ("χb1",      9892.78,  3),
    ("χb2",      9912.21,  3),
    ("Υ(2S)",    10023.26, 3),
    ("Υ(3S)",    10355.2,  3),
    ("η_b",      9398.7,   3),
    # Doublement-charmés
    ("Ξcc",      3621.6,   3),
    ("Ωcc",      3700.0,   3),
    # N* résonances
    ("N(1440)",  1440.0,   3),
    ("N(1535)",  1535.0,   3),
    ("N(1650)",  1655.0,   3),
    ("N(1710)",  1710.0,   3),
    # Delta résonances
    ("Δ(1600)",  1600.0,   3),
    ("Δ(1620)",  1620.0,   3),
    ("Δ(1700)",  1700.0,   3),
    ("Δ(1905)",  1880.0,   3),
    # K* mésons
    ("K*(892)",  891.66,   2),
    ("K*(1430)", 1425.6,   2),
    ("K1(1270)", 1272.0,   2),
    # f0 résonances
    ("f0(500)",  500.0,    2),
    ("f0(980)",  990.0,    2),
    # Exotiques
    ("X(3872)",  3871.69,  3),
    ("Pc(4312)", 4311.9,   3),
    ("Pc(4380)", 4380.0,   3),
    ("Pc(4450)", 4450.0,   3),
    ("Tcc",      3875.0,   3),
]


def main():
    print("\n" + "=" * 100)
    print("  VÉRIFICATION DES 32 PARTICULES EXOTIQUES — formule ANALYTIQUE SU(2)_8")
    print("=" * 100)
    print(f"\n  Paramètres analytiques :")
    print(f"    a   = √(7/(8π))           = {A:.6f}")
    print(f"    p,q = 3/4                  = {P:.6f}")
    print(f"    x   = -√6                  = {X:.6f}")
    print(f"    β   = 33/20                = {B:.6f}")
    print(f"    ρ_0 = (k+h^∨) - 3/(8π)     = {RHO_0:.6f}")
    print(f"    k_b = (Hölder pur)          = {K_B:.6f}")
    print(f"\n  Recherche automatique de configurations pour 32 exotiques…")

    print(f"\n  {'particule':>10} {'config trouvée':>22} {'m_exp':>10} "
          f"{'m_pred':>10} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 80)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    for nm, me, n in EXOTIC:
        cfg, mp, _ = search_config(me, n)
        err = (mp - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, cfg, me, mp, err, flag))
        print(f"  {nm:>10} {'+'.join(cfg):>22} {me:>10.2f} "
              f"{mp:>10.2f} {err:>+8.2f} {flag:>5}")

    total = len(EXOTIC)
    n5 = flags["★★"] + flags["★"]
    n20 = n5 + flags["✓"]
    print(f"\n  ─── BILAN sur 32 EXOTIQUES ─── ")
    print(f"    ★★ (<0.5%) : {flags['★★']}/{total}")
    print(f"    ★  (<5%)   : {flags['★']}/{total}")
    print(f"    ✓  (<20%)  : {flags['✓']}/{total}")
    print(f"    ordinaire  : {flags[' ']}/{total}")
    print(f"    outlier    : {flags['?']}/{total}")
    print(f"  ")
    print(f"    Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"    Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")

    # Comparaison aux résultats numériques (mega_spectrum)
    print(f"\n  ─── COMPARAISON ───")
    print(f"  Mega spectrum (numérique) sur 32 exotiques :")
    print(f"    27 ★★, 4 ★, 1 ✓ (31/32 à <5%, 32/32 à <20%)")
    print(f"  Cette vérification (analytique SU(2)_8) :")
    print(f"    {flags['★★']} ★★, {flags['★']} ★, {flags['✓']} ✓ "
          f"({n5}/32 à <5%, {n20}/32 à <20%)")

    print("=" * 100)


if __name__ == "__main__":
    main()
