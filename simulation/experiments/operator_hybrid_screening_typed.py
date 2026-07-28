r"""
OPÉRATEUR Ωₜₛ' — écrantage par paires DIFFÉRENCIÉ selon le type de nœud.

Le ρ uniforme du Ωₜₛ marche pour certains pairs mais pas tous.
Diagnostic :
   K± (4_1+4_1+5_2) : Δ = -39% → pair 4_1 sur-réduit
   p  (3_1+7_2+7_2) : Δ = +52% → pair 7_2 sous-réduit
   Ω⁻ (5_2+5_2+7_2) : Δ = -40% → pair 5_2 sur-réduit
   Λb (5_2+5_2+8_2) : Δ = +18% → pair 5_2 légèrement sous-réduit

Il semble que pour les paires de nœuds LOURDS (cr ≥ 6), la réduction
doit être plus forte ; pour les LÉGERS (cr ≤ 5), moins forte.

On introduit deux paramètres :
   ρ_light : facteur pour paire avec cr ≤ 5
   ρ_heavy : facteur pour paire avec cr ≥ 6

Variantes testées :
   V1 : ρ uniforme (référence Ωₜₛ)
   V2 : ρ_light + ρ_heavy (split par cr)
   V3 : ρ(K) = exp(-α · cr(K)) (continu)
   V4 : ρ(K) = ρ_0 · (cr/cr_ref)^β (loi de puissance)
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


def get_pairs_by_type(comps):
    """Retourne dict {K: nombre de paires} pour les nœuds répétés."""
    c = Counter(comps)
    return {K: count - 1 for K, count in c.items() if count >= 2}


def hybrid_full(comps, params, variant="V2"):
    """Opérateur hybride avec écrantage typé.

    V1 : ρ uniforme
    V2 : ρ_light (cr≤5) et ρ_heavy (cr≥6)
    V3 : ρ(K) = ρ_0 · exp(-α · cr(K)) (continu)
    V4 : ρ(K) = ρ_0 · (cr(K))^(-β) (loi de puissance)
    """
    a, p, q, x = params[:4]
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

    # Écrantage par paires (varie selon variante)
    pairs = get_pairs_by_type(comps)
    Phi_pair = 1.0
    if variant == "V1":
        rho = params[4]
        Phi_pair = rho ** sum(pairs.values())
    elif variant == "V2":
        rho_l, rho_h = params[4], params[5]
        for K, npair in pairs.items():
            cr_K = TOPO[K]["cr"]
            r = rho_l if cr_K <= 5 else rho_h
            Phi_pair *= r ** npair
    elif variant == "V3":
        rho0, alpha = params[4], params[5]
        for K, npair in pairs.items():
            r = rho0 * np.exp(-alpha * TOPO[K]["cr"])
            Phi_pair *= r ** npair
    elif variant == "V4":
        rho0, beta = params[4], params[5]
        for K, npair in pairs.items():
            r = rho0 * TOPO[K]["cr"] ** (-beta)
            Phi_pair *= r ** npair

    return Phi_n * Phi_pair * skeleton * G_comp ** q


def make_loss(variant, n_pair_params):
    def loss(params):
        res = []
        for nm, comps, me in SPECTRUM:
            try:
                m_pred = hybrid_full(comps, params, variant)
            except (OverflowError, ValueError, ZeroDivisionError):
                return 1e6
            if m_pred > 0 and np.isfinite(m_pred):
                res.append(np.log(m_pred / me))
            else:
                res.append(20)
        return float(np.sqrt(np.mean(np.array(res) ** 2)))
    return loss


def fit_variant(variant, n_pair_params):
    """Optimise pour une variante donnée."""
    loss_fn = make_loss(variant, n_pair_params)
    best = None
    for a0 in [0.3, 0.5]:
        for p0 in [0.6, 0.8]:
            for q0 in [0.5, 0.8]:
                for x0 in [-0.5, 0, 0.4]:
                    inits = []
                    if n_pair_params == 1:
                        inits = [[0.5], [0.7], [0.9]]
                    elif n_pair_params == 2:
                        inits = [[0.9, 0.6], [0.7, 0.5], [0.8, 0.7],
                                 [1.0, 0.5], [0.95, 0.6]]
                    for pair_init in inits:
                        x0_full = [a0, p0, q0, x0] + pair_init
                        res = minimize(loss_fn, x0_full, method="Nelder-Mead",
                                       options={"maxiter": 5000, "xatol": 1e-7})
                        if best is None or res.fun < best.fun:
                            best = res
    return best.x, best.fun


def evaluate(variant, params):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    for nm, comps, me in SPECTRUM:
        m_pred = hybrid_full(comps, params, variant)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, comps, me, m_pred, err, flag))
    return rows, flags


def main():
    print("\n" + "=" * 100)
    print("  OPÉRATEUR Ωₜₛ' — écrantage par paire DIFFÉRENCIÉ par type de nœud")
    print("=" * 100)

    variants = [
        ("V1 ρ uniforme",            "V1", 1),
        ("V2 ρ_light + ρ_heavy",     "V2", 2),
        ("V3 ρ_0·exp(-α·cr) continu", "V3", 2),
        ("V4 ρ_0·cr^(-β) puissance", "V4", 2),
    ]

    results = []
    for name, var, n_pp in variants:
        print(f"\n  {name}…")
        params, rms = fit_variant(var, n_pp)
        rows, flags = evaluate(var, params)
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        results.append((name, var, params, rms, n5, n20, rows, flags))

    print(f"\n  ─── COMPARAISON ───")
    print(f"  {'variante':<32} {'RMS log':>9} {'<5%':>6} {'<20%':>6}")
    print("  " + "-" * 60)
    for name, var, params, rms, n5, n20, _, _ in results:
        ps = ", ".join(f"{p:.3f}" for p in params)
        print(f"  {name:<32} {rms:>9.4f} {n5:>6} {n20:>6}  [{ps}]")

    # Meilleur
    best = min(results, key=lambda r: r[3])
    print(f"\n  ─── MEILLEUR : {best[0]} ───")
    print(f"  Paramètres : a={best[2][0]:.3f}, p={best[2][1]:.3f}, "
          f"q={best[2][2]:.3f}, x={best[2][3]:.3f}")
    if best[1] in ("V2", "V3", "V4"):
        if best[1] == "V2":
            print(f"             ρ_light={best[2][4]:.3f}, ρ_heavy={best[2][5]:.3f}")
        else:
            print(f"             param₁={best[2][4]:.3f}, param₂={best[2][5]:.3f}")
    print(f"  RMS = {best[3]:.4f}, <5% : {best[4]}/25, <20% : {best[5]}/25")

    print(f"\n  Prédictions :")
    print(f"  {'particule':>5} {'config':>17} {'pairs':>5} {'m_exp':>11} "
          f"{'m_pred':>11} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 75)
    for nm, comps, me, m_pred, err, flag in best[6]:
        np_ = sum(get_pairs_by_type(comps).values())
        print(f"  {nm:>5} {'+'.join(comps):>17} {np_:>5} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")
    print("=" * 100)


if __name__ == "__main__":
    main()
