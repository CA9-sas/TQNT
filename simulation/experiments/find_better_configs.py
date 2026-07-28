r"""
RECHERCHE de meilleures CONFIGURATIONS pour les particules outliers.

Avec l'opérateur Ωₜₛ' optimisé sur le spectre actuel, on prend les
paramètres FIXÉS (a, p, q, x, ρ_0, β) et on cherche, pour chaque
particule mal prédite, la combinaison de nœuds qui DONNE LA MEILLEURE
PRÉDICTION.

On respecte les contraintes :
   - Nombre de composantes n inchangé
   - Pas de 8_3 introduit pour les particules NON-EW
   - 8_3 préservé pour W, Z, H, t (EW)

Outliers ciblés : Λ, φ, Δ, η, J/ψ, Ω⁻, Z⁰, H, top
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

# Paramètres OPTIMAUX FIXÉS de Ωₜₛ' (V4)
PARAMS = (0.414, 0.773, 0.739, -1.473, 14.159, 1.752)


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


def operator(comps, params=PARAMS):
    """Opérateur Ωₜₛ' avec ρ(K) = ρ_0 · cr^(-β)."""
    a, p, q, x, rho0, beta = params
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
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-x) if n > 0 else 1.0
    # Écrantage par paires : ρ(K) = ρ_0 · cr(K)^(-β)
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = rho0 * TOPO[K]["cr"] ** (-beta)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** q


# Particules outliers : (nom, n_components, m_exp, autorise_8_3)
OUTLIERS = [
    ("Λ",     2,  1115.7,   False),
    ("φ",     3,  1019.46,  False),
    ("Δ",     3,  1232.0,   False),
    ("η",     3,  547.86,   False),
    ("J/ψ",   3,  3096.9,   False),
    ("Ω⁻",    3,  1672.0,   False),
    ("Z⁰",    3,  91188.0,  True),
    ("H",     3,  125250.0, True),
    ("t",     3,  173100.0, True),
]

CURRENT_CONFIGS = {
    "Λ":   ["4_1", "7_4"],
    "φ":   ["3_1", "4_1", "6_2"],
    "Δ":   ["3_1", "5_2", "6_2"],
    "η":   ["3_1", "4_1", "5_1"],
    "J/ψ": ["4_1", "5_2", "9_1"],
    "Ω⁻":  ["5_2", "5_2", "7_2"],
    "Z⁰":  ["5_1", "7_2", "8_3"],
    "H":   ["6_1", "7_4", "8_3"],
    "t":   ["7_4", "7_6", "8_3"],
}


def search_best_config(name, n, m_exp, allow_8_3):
    """Cherche la combinaison de n nœuds donnant m_pred le plus proche de m_exp."""
    pool = list(KNOTS)
    if not allow_8_3:
        pool = [K for K in pool if K != "8_3"]
    # Pour EW (8_3 obligatoire) : fixe 8_3 et cherche les n-1 autres
    if allow_8_3:
        # 8_3 fixé, on cherche les n-1 autres
        pool_other = [K for K in KNOTS if K != "8_3"]
        candidates = []
        for combo in combinations_with_replacement(pool_other, n - 1):
            config = list(combo) + ["8_3"]
            m_pred = operator(config)
            err = (m_pred - m_exp) / m_exp * 100
            candidates.append((config, m_pred, err))
    else:
        candidates = []
        for combo in combinations_with_replacement(pool, n):
            config = list(combo)
            m_pred = operator(config)
            err = (m_pred - m_exp) / m_exp * 100
            candidates.append((config, m_pred, err))
    # Tri par erreur absolue
    candidates.sort(key=lambda r: abs(r[2]))
    return candidates[:5]   # top 5


def main():
    print("\n" + "=" * 100)
    print("  RECHERCHE DE MEILLEURES CONFIGURATIONS")
    print("  Opérateur Ωₜₛ' fixé (paramètres V4), recherche sur 19 nœuds")
    print("=" * 100)

    for name, n, m_exp, allow_83 in OUTLIERS:
        current = CURRENT_CONFIGS[name]
        m_pred_curr = operator(current)
        err_curr = (m_pred_curr - m_exp) / m_exp * 100
        print(f"\n  ─── {name} (m_exp = {m_exp} MeV, n = {n}) ───")
        print(f"  Config actuelle : {'+'.join(current):>17}  → "
              f"m_pred = {m_pred_curr:>10.2f}  Δ = {err_curr:+.2f}%")

        best = search_best_config(name, n, m_exp, allow_83)
        print(f"  Top 5 alternatives :")
        for i, (cfg, mp, err) in enumerate(best):
            flag = "★" if abs(err) < 5 else (" " if abs(err) < 20 else "?")
            print(f"    {i+1}. {'+'.join(cfg):>17}  → "
                  f"m_pred = {mp:>10.2f}  Δ = {err:+7.2f}% {flag}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
