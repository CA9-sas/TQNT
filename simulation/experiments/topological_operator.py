r"""
OPÉRATEUR TOPOLOGIQUE — basé uniquement sur les comptes :
cr (croisements de brins) et n_lobes (boucles).

Aucune intégrale géométrique. Seulement les invariants topologiques.

On invente 7 opérateurs candidats τ_1 à τ_7, tous bâtis sur :
   - S(P) = Σ_i cr(K_i)        (brins total)
   - L(P) = Σ_i n_lobes(K_i)   (boucles total)
   - {cr_i}, {n_lobes_i}        (distribution par constituant)

Calibration : m(3_1) = m_e (par construction).

Pour 7_2 isolé : on cherche m ≈ 33 MeV (déduit du proton dans Hölder)
ou alternativement m ≈ 66 MeV si k_s est intégré dans l'opérateur.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

M_E = 0.51099895

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


# ───────────────────────────────────────────────────────────────────
# OPÉRATEURS TOPOLOGIQUES INVENTÉS
# ───────────────────────────────────────────────────────────────────
def t1_only_strands(comps, params):
    """τ_1 : seulement les brins. m = m_e · (S/3)^p"""
    p = params[0]
    S = sum(cr(K) for K in comps)
    return M_E * (S / 3) ** p


def t2_strands_and_loops(comps, params):
    """τ_2 : brins ET boucles séparés. m = m_e · (S/3)^p · (L/3)^q"""
    p, q = params
    S = sum(cr(K) for K in comps)
    L = sum(N_LOBES[K] for K in comps)
    return M_E * (S / 3) ** p * (L / 3) ** q


def t3_product_per_knot(comps, params):
    """τ_3 : produit cr·n_lobes par nœud, sommé. m = m_e · (Σ(cr·n_l)/9)^p"""
    p = params[0]
    prod_sum = sum(cr(K) * N_LOBES[K] for K in comps)
    return M_E * (prod_sum / 9) ** p


def t4_holder_topo(comps, params):
    """τ_4 : Hölder-like sur (cr_i · n_lobes_i / 9)^p
       m = m_e · [Σ_i (cr_i · n_l_i / 9)^a]^(1/a · p)
    """
    a, p = params
    if abs(a) < 1e-8:
        x = float(np.prod([cr(K) * N_LOBES[K] / 9 for K in comps]) ** (1 / len(comps)))
    else:
        x = sum((cr(K) * N_LOBES[K] / 9) ** a for K in comps) ** (1 / a)
    return M_E * x ** p


def t5_geometric_mean_topo(comps, params):
    """τ_5 : moyenne géométrique des produits cr·n_lobes / 9
       m = m_e · [Π_i (cr_i · n_l_i / 9)]^(p/n)
    """
    p = params[0]
    n = len(comps)
    prod = float(np.prod([cr(K) * N_LOBES[K] / 9 for K in comps]))
    return M_E * prod ** (p / n)


def t6_mixed(comps, params):
    """τ_6 : combinaison brins × boucles avec couplage croisé inter-nœud.
       m = m_e · (S·L/9)^p · n^q  où n = nb composants
    """
    p, q = params
    S = sum(cr(K) for K in comps)
    L = sum(N_LOBES[K] for K in comps)
    n = len(comps)
    return M_E * (S * L / 9) ** p * n ** q


def t7_double_holder(comps, params):
    """τ_7 : Hölder séparés sur S et L.
       m = m_e · [Σ cr_i^a]^(p/a) · [Σ n_l_i^b]^(q/b)
    """
    a, b, p, q = params
    if abs(a) < 1e-8 or abs(b) < 1e-8:
        return 0
    Sx = sum(cr(K) ** a for K in comps) ** (p / a)
    Lx = sum(N_LOBES[K] ** b for K in comps) ** (q / b)
    return M_E * Sx * Lx / (3 ** p) / (3 ** q)


OPERATORS = [
    ("τ_1 brins seul   m=m_e·(S/3)^p",                  t1_only_strands,     1),
    ("τ_2 brins+boucles m=m_e·(S/3)^p·(L/3)^q",         t2_strands_and_loops, 2),
    ("τ_3 produit cr·n_l m=m_e·(Σcr·n_l/9)^p",          t3_product_per_knot,  1),
    ("τ_4 Hölder topo  m_e·[Σ(cr_i·n_l_i/9)^a]^(p/a)",  t4_holder_topo,      2),
    ("τ_5 moy géom     m_e·[Π(cr_i·n_l_i/9)]^(p/n)",    t5_geometric_mean_topo, 1),
    ("τ_6 mixte S·L+n  m_e·(S·L/9)^p·n^q",              t6_mixed,             2),
    ("τ_7 double Hölder Σcr^a/3^a × Σnl^b/3^b",         t7_double_holder,     4),
]


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


def fit(op, n_p, n_starts=5):
    best = None
    starts = [0.5, 1.0, 1.5, 2.0, 3.0][:n_starts]
    for s in starts:
        x0 = [s] * n_p
        res = minimize(lambda x: loss(op, x), x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def evaluate(op, params):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
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
        rows.append((nm, comps, me, m_pred, err, flag))
    return rows, flags


def main():
    print("=" * 100)
    print("  OPÉRATEURS TOPOLOGIQUES — basés sur cr (brins) et n_lobes (boucles)")
    print("=" * 100)

    print(f"\n  {'opérateur':<48} {'params':<20} {'RMS':>9} {'★+★★':>6} "
          f"{'<20%':>6}")
    print("  " + "-" * 95)

    results = []
    for name, op, np_ in OPERATORS:
        params, rms = fit(op, np_)
        rows, flags = evaluate(op, params)
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        results.append((name, op, params, rms, n5, n20, rows, flags))
        param_str = ", ".join(f"{p:.3f}" for p in params)
        print(f"  {name:<48} {param_str:<20} {rms:>9.4f} {n5:>6} {n20:>6}")

    # Détails du meilleur
    best = min(results, key=lambda r: r[3])
    print(f"\n  ─── MEILLEUR : {best[0]} ───")
    print(f"  Paramètres : {best[2]}")
    print(f"  RMS log = {best[3]:.4f}")
    print(f"  ★+★★ = {best[4]}/25, <20% = {best[5]}/25")

    print(f"\n  Prédictions :")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 67)
    for nm, comps, me, m_pred, err, flag in best[6]:
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")
    print("=" * 100)


if __name__ == "__main__":
    main()
