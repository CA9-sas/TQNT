r"""
MEGA SPECTRE TQNT — extension à 70+ particules.

Ajoute aux 45 particules validées :
   - Excitations charmonium : χc0/1/2, ψ(3770), ψ(4040), η_c(2S)
   - Excitations bottomonium : χb0/1/2, Υ(2S), Υ(3S), η_b
   - Baryons doublement-charmés : Ξcc, Ωcc
   - Doublement-beaux : Ξbb (théorique)
   - N* résonances : N(1440), N(1535), N(1650), N(1710)
   - Δ résonances : Δ(1620), Δ(1700), Δ(1905)
   - K* mésons : K*(892), K*(1430)
   - Vectoriels : f0(500), f0(980)
   - Pentaquark Pc(4380), Pc(4450)
   - Tetraquark : Tcc(3875)
   - X(3872)

Pour chaque, recherche automatique de config 2 ou 3 composantes
avec opérateur Ωₜₛ' (paramètres fixés du commit e771a40).
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
PARAMS = (0.5051, 0.7896, 0.7138, -2.3696, 8.5297, 1.5410)


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


# 45 particules validées (configs et masses)
VALIDATED = [
    ("e⁻",       ["3_1"],                       0.511),
    ("μ⁻",       ["3_1", "6_1"],                105.66),
    ("τ⁻",       ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",       ["8_1"],                       134.98),
    ("π±",       ["8_1"],                       139.57),
    ("K±",       ["4_1", "4_1", "5_2"],         493.68),
    ("K⁰_S",     ["3_1", "6_1", "6_1"],         497.611),
    ("η",        ["5_2", "5_2", "6_1"],         547.86),
    ("η'",       ["5_1", "5_2", "5_2"],         957.78),
    ("ρ/ω",      ["5_2", "7_7"],                775.0),
    ("φ",        ["3_1", "7_1", "7_1"],         1019.46),
    ("J/ψ",      ["5_2", "6_1", "6_2"],         3096.9),
    ("ψ(2S)",    ["5_1", "7_7", "7_7"],         3686.10),
    ("η_c",      ["3_1", "7_1", "9_1"],         2983.9),
    ("Υ",        ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",       ["6_2", "7_7"],                1864.84),
    ("D±",       ["7_7", "9_1"],                1869.66),
    ("D_s",      ["6_3", "9_1"],                1968.34),
    ("B",        ["4_1", "10_1"],               5279.4),
    ("B⁰",       ["7_6", "7_7"],                5279.65),
    ("B_s",      ["7_6", "7_7"],                5366.92),
    ("B_c",      ["5_1", "10_1"],               6274.9),
    ("p",        ["3_1", "7_2", "7_2"],         938.272),
    ("n",        ["3_1", "7_2", "7_2"],         939.565),
    ("Λ",        ["5_1", "7_7"],                1115.7),
    ("Σ⁺",       ["6_2", "7_2"],                1189.4),
    ("Σ⁰",       ["5_1", "6_3"],                1192.642),
    ("Σ⁻",       ["5_1", "6_3"],                1197.449),
    ("Ξ⁰",       ["5_1", "5_1", "5_2"],         1314.86),
    ("Ξ⁻",       ["4_1", "9_1", "9_1"],         1321.71),
    ("Δ",        ["4_1", "7_2", "7_2"],         1232.0),
    ("Σ*",       ["3_1", "5_2", "7_1"],         1383.7),
    ("Ξ*",       ["3_1", "6_1", "7_2"],         1531.8),
    ("Ω⁻",       ["3_1", "6_1", "7_1"],         1672.0),
    ("Λc",       ["6_3", "8_1"],                2286.46),
    ("Σc",       ["5_2", "5_2", "8_1"],         2453.97),
    ("Ξc",       ["5_1", "5_2", "7_2"],         2467.94),
    ("Ωc",       ["6_1", "6_1", "8_1"],         2695.2),
    ("Λb",       ["5_1", "6_1", "6_3"],         5619.6),
    ("Σb",       ["4_1", "7_7", "9_1"],         5811.0),
    ("Ξb",       ["6_1", "6_3", "6_3"],         5797.0),
    ("W±",       ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",       ["7_4", "9_1", "8_3"],         91188.0),
    ("H",        ["7_7", "8_2", "8_3"],         125250.0),
    ("t",        ["3_1", "7_5", "8_3"],         173100.0),
]


# Nouvelles particules à ajouter (nom, m_exp, n_comp)
TO_ADD = [
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
    ("Ωcc",      3700.0,   3),  # théorique mais estimé
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
    # États exotiques (multiquarks)
    ("X(3872)",  3871.69,  3),
    ("Pc(4312)", 4311.9,   3),  # pentaquark candidate
    ("Pc(4380)", 4380.0,   3),
    ("Pc(4450)", 4450.0,   3),
    ("Tcc",      3875.0,   3),  # tetraquark candidate (~doublement-charmé)
]


def search_config(m_exp, n):
    pool = [K for K in KNOTS if K != "8_3"]
    best = (None, float("inf"), float("inf"))
    for combo in combinations_with_replacement(pool, n):
        config = list(combo)
        m_pred = operator(config)
        err = abs((m_pred - m_exp) / m_exp * 100)
        if err < best[2]:
            best = (config, m_pred, err)
    return best


def main():
    print("\n" + "=" * 105)
    print("  MEGA SPECTRE TQNT — extension à ~75 particules")
    print("=" * 105)

    print(f"\n  ─── PARTIE 1 : 45 particules validées (rappel rapide) ───")
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps, me in VALIDATED:
        m_pred = operator(comps)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
    n5_p1 = flags["★★"] + flags["★"]
    n20_p1 = n5_p1 + flags["✓"]
    print(f"    ★★={flags['★★']}, ★={flags['★']}, ✓={flags['✓']}, "
          f"<5%={n5_p1}/45, <20%={n20_p1}/45")

    print(f"\n  ─── PARTIE 2 : 32 nouvelles particules ───")
    print(f"  {'particule':>10} {'config trouvée':>22} {'m_exp':>10} "
          f"{'m_pred':>10} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 80)
    flags_new = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    new_rows = []
    for nm, me, n in TO_ADD:
        cfg, mp, _ = search_config(me, n)
        err = (mp - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags_new[flag] += 1
        new_rows.append((nm, cfg, me, mp, err, flag))
        print(f"  {nm:>10} {'+'.join(cfg):>22} {me:>10.2f} "
              f"{mp:>10.2f} {err:>+8.2f} {flag:>5}")

    n5_p2 = flags_new["★★"] + flags_new["★"]
    n20_p2 = n5_p2 + flags_new["✓"]
    print(f"\n    Bilan nouvelles : ★★={flags_new['★★']}, "
          f"★={flags_new['★']}, ✓={flags_new['✓']}, "
          f"ordinaire={flags_new[' ']}, ?={flags_new['?']}")
    print(f"    Précision <5%={n5_p2}/{len(TO_ADD)}, <20%={n20_p2}/{len(TO_ADD)}")

    # Bilan total
    flags_all = {k: flags[k] + flags_new[k] for k in flags}
    total = sum(flags_all.values())
    n5 = flags_all["★★"] + flags_all["★"]
    n20 = n5 + flags_all["✓"]
    print(f"\n  ═══ BILAN TOTAL ({total} particules) ═══")
    print(f"    ★★ (<0.5%) : {flags_all['★★']}/{total} "
          f"({flags_all['★★']/total*100:.0f}%)")
    print(f"    ★  (<5%)   : {flags_all['★']}/{total} "
          f"({flags_all['★']/total*100:.0f}%)")
    print(f"    ✓  (<20%)  : {flags_all['✓']}/{total} "
          f"({flags_all['✓']/total*100:.0f}%)")
    print(f"    ordinaire  : {flags_all[' ']}/{total}")
    print(f"    outlier    : {flags_all['?']}/{total}")
    print(f"  ")
    print(f"    Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"    Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")
    print("=" * 105)


if __name__ == "__main__":
    main()
