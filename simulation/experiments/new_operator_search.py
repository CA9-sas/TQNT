r"""
RECHERCHE D'UN NOUVEL OPÉRATEUR sur les mêmes principes que Hölder,
intégrant les découvertes :
   - m_iso(K) = m_naive · P(K)/P(3_1) avec P = U_κ · U_τ
   - k_s = 2 pour boucle "serrée" (apparaît 1 fois dans config)
   - k_s = 1 pour libre/appariée (apparaît ≥2 fois)
   - couplage géométrique inter-boucles (Σ U_κ · U_τ du composite)

On teste 5 opérateurs candidats :

   H₀  Hölder pur     :  m(P) = (Σ m_iso(K)^a)^(1/a)
   H_s Hölder + état  :  m(P) = (Σ (m_iso·ν_i)^a)^(1/a), ν=2 si serrée
   H_F Hölder + Frenet:  m(P) = (Σ m_iso^a)^(1/a) · F_couplage
   H_W Hölder pondéré :  m(P) = (Σ ω_i m_iso^a)^(1/a), ω = U_κ_i·U_τ_i normalisé
   H_C Combinaison    :  H_s · F_couplage

Pour chaque opérateur, on optimise les paramètres libres et on évalue
RMS log sur le spectre observable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from theory.abraham import frenet_geometry

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


def cr(K):
    return int(K.split("_")[0])


# Calcul des intégrales Frenet une seule fois
print("Calcul Frenet (50000 points) pour les 19 nœuds…")
U_KT = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]

# Poids isolé pour chaque nœud
M_ISO = {}
for K in KNOTS:
    U_k, U_t = U_KT[K]
    m_naive = cr(K) * M_C + N_LOBES[K] * M_L
    M_ISO[K] = m_naive * U_k * U_t / P_31

print(f"Calcul terminé. P(3_1) = {P_31:.4f}")


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
    ("W±",    ["4_1", "7_3", "8_3"],         80377.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91188.0),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def state_factor(K, comps, k_s):
    """ν_K = k_s si K apparaît 1 fois (serrée), 1 sinon."""
    if comps.count(K) == 1 and len(comps) > 1:
        return k_s
    return 1.0


# ───────────────────────────────────────────────────────────────────
# Opérateurs candidats
# ───────────────────────────────────────────────────────────────────
def H0_hölder(comps, params):
    """Hölder pur sur m_iso."""
    a = params[0]
    ws = [M_ISO[K] for K in comps]
    if abs(a) < 1e-8:
        return float(np.prod(ws) ** (1 / len(ws)))
    return float(sum(w ** a for w in ws) ** (1 / a))


def Hs_state(comps, params):
    """Hölder avec facteur d'état (serrée vs libre)."""
    a, k_s = params
    ws = [M_ISO[K] * state_factor(K, comps, k_s) for K in comps]
    if abs(a) < 1e-8:
        return float(np.prod(ws) ** (1 / len(ws)))
    return float(sum(w ** a for w in ws) ** (1 / a))


def HF_couplage(comps, params):
    """Hölder × facteur de couplage Frenet."""
    a, b = params
    ws = [M_ISO[K] for K in comps]
    base = sum(w ** a for w in ws) ** (1 / a) if abs(a) > 1e-8 \
           else float(np.prod(ws) ** (1 / len(ws)))
    # Couplage : produit des P(K_i) normalisé
    coupling = np.prod([U_KT[K][0] * U_KT[K][1] / P_31 for K in comps])
    return float(base * coupling ** b)


def HW_weighted(comps, params):
    """Hölder pondéré par P(K_i)."""
    a, b = params
    P_ks = [U_KT[K][0] * U_KT[K][1] / P_31 for K in comps]
    P_sum = sum(P_ks)
    ws = [(P_k / P_sum) ** b * M_ISO[K] ** a for K, P_k in zip(comps, P_ks)]
    if abs(a) < 1e-8:
        return float(np.prod([M_ISO[K] for K in comps]) ** (1 / len(comps)))
    return float(sum(ws) ** (1 / a))


def HC_combined(comps, params):
    """Hs + couplage Frenet : combinaison serrée + couplage."""
    a, k_s, b = params
    ws = [M_ISO[K] * state_factor(K, comps, k_s) for K in comps]
    base = sum(w ** a for w in ws) ** (1 / a) if abs(a) > 1e-8 \
           else float(np.prod(ws) ** (1 / len(ws)))
    coupling = np.prod([U_KT[K][0] * U_KT[K][1] / P_31 for K in comps])
    return float(base * coupling ** b)


# ───────────────────────────────────────────────────────────────────
# Optimisation
# ───────────────────────────────────────────────────────────────────
def loss(op, params):
    res = []
    for nm, comps, me in SPECTRUM:
        try:
            m_pred = op(comps, params)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e6
        if m_pred > 0 and np.isfinite(m_pred):
            res.append(np.log(m_pred / me))
        else:
            res.append(20)
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def fit(op, n_params, n_starts=5):
    best = None
    starts_a = [0.15, 0.25, 0.35, 0.5, 1.0]
    for i in range(min(n_starts, len(starts_a))):
        x0 = [starts_a[i]] + [1.0] * (n_params - 1)
        res = minimize(lambda x: loss(op, x), x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def evaluate(op, params):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res = []
    rows = []
    for nm, comps, me in SPECTRUM:
        m_pred = op(comps, params)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        res.append(np.log(m_pred / me))
        rows.append((nm, comps, me, m_pred, err, flag))
    rms = float(np.sqrt(np.mean(np.array(res) ** 2)))
    return rows, flags, rms


def main():
    print("\n" + "=" * 100)
    print("  RECHERCHE D'OPÉRATEUR : Hölder + ingrédients géométriques")
    print("=" * 100)

    candidates = [
        ("H₀  Hölder pur",                       H0_hölder,    1),
        ("H_s Hölder + état serrée",             Hs_state,     2),
        ("H_F Hölder × couplage Frenet",         HF_couplage,  2),
        ("H_W Hölder pondéré par P(K)",          HW_weighted,  2),
        ("H_C Hölder + état + couplage",         HC_combined,  3),
    ]

    print(f"\n  {'opérateur':<35} {'params opt.':<35} {'RMS log':>9} "
          f"{'★+★★':>6} {'<20%':>6}")
    print("  " + "-" * 96)

    all_results = []
    for name, op, n_p in candidates:
        params, rms = fit(op, n_p)
        rows, flags, _ = evaluate(op, params)
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        param_str = ", ".join(f"{p:.4f}" for p in params)
        all_results.append((name, op, params, rms, n5, n20, rows, flags))
        print(f"  {name:<35} {param_str:<35} {rms:>9.4f} {n5:>6} {n20:>6}")

    # Détails du meilleur
    best = min(all_results, key=lambda r: r[3])
    print(f"\n  ─── MEILLEUR : {best[0]} ───")
    print(f"  Paramètres optimaux : {best[2]}")
    print(f"  RMS log = {best[3]:.4f} ; ★+★★ = {best[4]}/{len(SPECTRUM)} ; "
          f"<20% = {best[5]}/{len(SPECTRUM)}")
    print(f"\n  Prédictions détaillées :")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 70)
    for nm, comps, me, m_pred, err, flag in best[6]:
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")

    print("=" * 100)


if __name__ == "__main__":
    main()
