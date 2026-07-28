r"""
OPÉRATEUR HYBRIDE Ωₜₛ — avec ÉCRANTAGE (mass-reducing combinations)

Extension du Ωₜ hybride en ajoutant un FACTEUR D'ÉCRANTAGE qui REDUIT
la masse pour certaines combinaisons de nœuds :

   m(P) = Φ(P) · [Σ_i (m_naive·T^p)^a]^(1/a) · G_composite^q

avec deux écrantages distincts :
   1. Φ_n(P) = ((n+1)/2)^(-x)         (écrantage gluonique global,
                                         lemme R6 = -3/8)
   2. Φ_pair(P) = ρ^(N_paires)          (réduction par paire de nœuds
                                         identiques, comme 7_2+7_2 dans p)

Total :  Φ(P) = Φ_n(P) · Φ_pair(P)

Les paires de NŒUDS IDENTIQUES créent une liaison qui réduit l'énergie
totale (effet diquark, condensat de paires). Multiple paires multiplient.

Pour la TQNT :
   Φ_n(1) = 1 (single, pas d'écrantage)
   Φ_n(2) = (3/2)^(-x)  (mésons)
   Φ_n(3) = 2^(-x)       (baryons + EW)

Les paramètres x et ρ sont déterminés par l'optimisation.
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


print("Calcul Frenet (50000 points) pour les 19 nœuds…")
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
    p = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(p),
    }

P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}


SPECTRUM = [
    ("e⁻",    ["3_1"],                       0.511),
    ("μ⁻",    ["3_1", "6_1"],                105.66),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.98),
    ("π±",    ["8_1"],                       139.57),
    ("K±",    ["4_1", "4_1", "5_2"],         493.68),
    ("η",     ["3_1", "4_1", "5_1"],         547.86),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "4_1", "6_2"],         1019.46),
    ("J/ψ",   ["4_1", "5_2", "9_1"],         3096.9),
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("p",     ["3_1", "7_2", "7_2"],         938.272),
    ("n",     ["3_1", "7_2", "7_2"],         939.565),
    ("Λ",     ["4_1", "7_4"],                1115.7),
    ("Σ⁺",    ["6_2", "7_2"],                1189.4),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),
    ("Ω⁻",    ["5_2", "5_2", "7_2"],         1672.0),
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.6),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91188.0),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def n_pairs(comps):
    """Nombre de paires de nœuds identiques (somme sur K de (count_K - 1)
       pour count_K >= 2)."""
    counter = Counter(comps)
    return sum(c - 1 for c in counter.values() if c >= 2)


def hybrid_screened(comps, params):
    """Opérateur hybride Ωₜₛ avec écrantage :

       m(P) = ((n+1)/2)^(-x) · ρ^N_paires
              · [Σ_i (m_naive_i · T_i^p)^a]^(1/a) · G_composite^q
    """
    a, p, q, x, rho = params
    n = len(comps)
    # Squelette
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** p)
    if abs(a) < 1e-8:
        skeleton = float(np.prod(ws) ** (1 / len(ws)))
    else:
        skeleton = float(sum(w ** a for w in ws) ** (1 / a))
    # Discriminant géométrique
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    # Écrantages
    Phi_n = ((n + 1) / 2) ** (-x) if n > 0 else 1.0
    Phi_pair = rho ** n_pairs(comps)
    return Phi_n * Phi_pair * skeleton * G_comp ** q


def loss(params):
    res = []
    for nm, comps, me in SPECTRUM:
        try:
            m_pred = hybrid_screened(comps, params)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e6
        if m_pred > 0 and np.isfinite(m_pred):
            res.append(np.log(m_pred / me))
        else:
            res.append(20)
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def main():
    print("\n" + "=" * 100)
    print("  OPÉRATEUR HYBRIDE Ωₜₛ AVEC ÉCRANTAGE (mass-reducing)")
    print("=" * 100)
    print(f"\n  m(P) = ((n+1)/2)^(-x) · ρ^(N_paires) · "
          f"[Σ(m_naive·T^p)^a]^(1/a) · G^q")
    print(f"  où :")
    print(f"    x = exposant d'écrantage global (R6 prédit 3/8 = 0.375)")
    print(f"    ρ = facteur de réduction par paire identique (<1 attendu)")

    print(f"\n  Particules avec paires :")
    for nm, comps, me in SPECTRUM:
        np_ = n_pairs(comps)
        if np_ > 0:
            print(f"    {nm:>5} ({'+'.join(comps)}) : {np_} paire(s)")

    print(f"\n  Optimisation 5 paramètres (a, p, q, x, ρ)…")
    best = None
    for a0 in [0.3, 0.5, 1.0]:
        for p0 in [0.5, 0.7, 1.0]:
            for q0 in [0.5, 0.8, 1.0]:
                for x0 in [0.0, 0.4, 1.0]:
                    for rho0 in [0.5, 0.8, 1.0]:
                        res = minimize(loss, [a0, p0, q0, x0, rho0],
                                       method="Nelder-Mead",
                                       options={"maxiter": 5000, "xatol": 1e-7})
                        if best is None or res.fun < best.fun:
                            best = res

    a, p, q, x, rho = best.x
    rms = best.fun
    print(f"\n  Paramètres optimaux :")
    print(f"    a   = {a:.5f}   (Hölder squelette)")
    print(f"    p   = {p:.5f}   (puissance topo)")
    print(f"    q   = {q:.5f}   (puissance géom)")
    print(f"    x   = {x:.5f}   (écrantage global ; R6 prédit 0.375)")
    print(f"    ρ   = {rho:.5f}  (réduction par paire ; <1 = réduction)")
    print(f"  RMS log = {rms:.4f}  (= +{(np.exp(rms)-1)*100:.0f}% relatif)")

    # Évaluation
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    for nm, comps, me in SPECTRUM:
        m_pred = hybrid_screened(comps, [a, p, q, x, rho])
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, comps, me, m_pred, err, flag))

    print(f"\n  Prédictions :")
    print(f"  {'particule':>5} {'config':>17} {'pairs':>5} {'m_exp':>11} "
          f"{'m_pred':>11} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 75)
    for nm, comps, me, m_pred, err, flag in rows:
        np_ = n_pairs(comps)
        print(f"  {nm:>5} {'+'.join(comps):>17} {np_:>5} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")

    total = len(SPECTRUM)
    n5 = flags["★★"] + flags["★"]
    n20 = n5 + flags["✓"]
    print(f"\n  Bilan : ★★={flags['★★']}, ★={flags['★']}, ✓={flags['✓']}, "
          f"ordinaire={flags[' ']}, ?={flags['?']} sur {total}")
    print(f"  Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"  Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")

    print(f"\n  ─── COMPARAISON ───")
    print(f"  Ωₜ hybride     : RMS 0.34, 4/25 à <5%, 11/25 à <20%")
    print(f"  Ωₜₛ + écrantage : RMS {rms:.2f}, {n5}/25 à <5%, {n20}/25 à <20%")
    print("=" * 100)


if __name__ == "__main__":
    main()
