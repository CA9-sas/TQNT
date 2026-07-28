r"""
SPECTRE ÉTENDU TQNT — 45 particules.

On garde les 25 particules validées et on ajoute :
   - Σ⁰, Σ⁻ (octet)
   - Ξ⁰, Ξ⁻ (octet)
   - Σ*, Ξ* (décuplet)
   - Σc, Ξc, Ωc (charmés)
   - Σb, Ξb (beaux)
   - K⁰_S, D±, D±_s, D⁰_s (mésons)
   - B⁰, B_s, B_c (mésons beaux)
   - η'

Pour chaque nouvelle particule, on cherche la meilleure configuration
3-comp (sans 8_3 sauf si EW) avec les paramètres Ωₜₛ' fixés (commit e771a40).
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


# Configs VALIDÉES du spectre 25 (commit e647fd4)
VALIDATED = {
    "e⁻":    ["3_1"],
    "μ⁻":    ["3_1", "6_1"],
    "τ⁻":    ["3_1", "5_1", "7_1"],
    "π⁰":    ["8_1"],
    "π±":    ["8_1"],
    "K±":    ["4_1", "4_1", "5_2"],
    "η":     ["5_2", "5_2", "6_1"],
    "ρ/ω":   ["5_2", "7_7"],
    "φ":     ["3_1", "7_1", "7_1"],
    "J/ψ":   ["5_2", "6_1", "6_2"],
    "Υ":     ["3_1", "7_4", "9_1"],
    "D⁰":    ["6_2", "7_7"],
    "B":     ["4_1", "10_1"],
    "p":     ["3_1", "7_2", "7_2"],
    "n":     ["3_1", "7_2", "7_2"],
    "Λ":     ["5_1", "7_7"],
    "Σ⁺":    ["6_2", "7_2"],
    "Δ":     ["4_1", "7_2", "7_2"],
    "Ω⁻":    ["3_1", "6_1", "7_1"],
    "Λc":    ["6_3", "8_1"],
    "Λb":    ["5_1", "6_1", "6_3"],
    "W±":    ["4_1", "7_3", "8_3"],
    "Z⁰":    ["7_4", "9_1", "8_3"],
    "H":     ["7_7", "8_2", "8_3"],
    "t":     ["3_1", "7_5", "8_3"],
}

# Particules à AJOUTER (nom, m_exp_MeV, n_components, allow_8_3)
TO_ADD = [
    # Octet baryons
    ("Σ⁰",       1192.642, 2, False),
    ("Σ⁻",       1197.449, 2, False),
    ("Ξ⁰",       1314.86,  3, False),
    ("Ξ⁻",       1321.71,  3, False),
    # Décuplet
    ("Σ*",       1383.7,   3, False),
    ("Ξ*",       1531.8,   3, False),
    # Charmés
    ("Σc",       2453.97,  3, False),
    ("Ξc",       2467.94,  3, False),
    ("Ωc",       2695.2,   3, False),
    # Beaux
    ("Σb",       5811.0,   3, False),
    ("Ξb",       5797.0,   3, False),
    # Mésons exotiques
    ("K⁰_S",     497.611,  3, False),
    ("η'",       957.78,   3, False),
    ("D±",       1869.66,  2, False),
    ("D_s",      1968.34,  2, False),
    ("B⁰",       5279.65,  2, False),
    ("B_s",      5366.92,  2, False),
    ("B_c",      6274.9,   2, False),
    # Quarkonium
    ("ψ(2S)",    3686.097, 3, False),
    ("η_c",      2983.9,   3, False),
]


def search_config(m_exp, n, allow_8_3):
    """Cherche la meilleure config n-composantes pour m_exp."""
    if allow_8_3:
        pool_other = [K for K in KNOTS if K != "8_3"]
        best = (None, float("inf"), float("inf"))
        for combo in combinations_with_replacement(pool_other, n - 1):
            config = list(combo) + ["8_3"]
            m_pred = operator(config)
            err = abs((m_pred - m_exp) / m_exp * 100)
            if err < best[2]:
                best = (config, m_pred, err)
    else:
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
    print("\n" + "=" * 100)
    print("  SPECTRE ÉTENDU TQNT — Validation + recherche pour 20 nouvelles particules")
    print("=" * 100)

    # Masses expérimentales pour les particules déjà validées
    m_exp_validated = {
        "e⁻": 0.511, "μ⁻": 105.66, "τ⁻": 1776.86,
        "π⁰": 134.98, "π±": 139.57, "K±": 493.68,
        "η": 547.86, "ρ/ω": 775.0, "φ": 1019.46,
        "J/ψ": 3096.9, "Υ": 9460.3,
        "D⁰": 1864.84, "B": 5279.4,
        "p": 938.272, "n": 939.565,
        "Λ": 1115.7, "Σ⁺": 1189.4,
        "Δ": 1232.0, "Ω⁻": 1672.0,
        "Λc": 2286.46, "Λb": 5619.6,
        "W±": 80369.0, "Z⁰": 91188.0,
        "H": 125250.0, "t": 173100.0,
    }

    print(f"\n  ─── PARTIE 1 : 25 particules validées (rappel) ───")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 70)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps in VALIDATED.items():
        me = m_exp_validated[nm]
        m_pred = operator(comps)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} {m_pred:>11.2f} "
              f"{err:>+8.2f} {flag:>5}")
    print(f"\n  Bilan partie 1 : ★★={flags['★★']}, ★={flags['★']}, "
          f"✓={flags['✓']}")

    print(f"\n  ─── PARTIE 2 : 20 nouvelles particules — recherche automatique ───")
    print(f"  {'particule':>10} {'config trouvée':>22} {'m_exp':>11} {'m_pred':>11} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 80)
    new_rows = []
    flags_new = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, me, n, allow83 in TO_ADD:
        cfg, m_pred, err_abs = search_config(me, n, allow83)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags_new[flag] += 1
        new_rows.append((nm, cfg, me, m_pred, err, flag))
        print(f"  {nm:>10} {'+'.join(cfg):>22} {me:>11.2f} {m_pred:>11.2f} "
              f"{err:>+8.2f} {flag:>5}")

    print(f"\n  Bilan partie 2 (nouvelles) : ★★={flags_new['★★']}, "
          f"★={flags_new['★']}, ✓={flags_new['✓']}")

    # Bilan global
    flags_all = {k: flags[k] + flags_new[k] for k in flags}
    total = sum(flags_all.values())
    n5 = flags_all["★★"] + flags_all["★"]
    n20 = n5 + flags_all["✓"]
    print(f"\n  ═══ BILAN GLOBAL (45 particules) ═══")
    print(f"    ★★ (<0.5%) : {flags_all['★★']}/{total}")
    print(f"    ★  (<5%)   : {flags_all['★']}/{total}")
    print(f"    ✓  (<20%)  : {flags_all['✓']}/{total}")
    print(f"    Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"    Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")
    print("=" * 100)


if __name__ == "__main__":
    main()
