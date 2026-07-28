r"""
OPÉRATEUR TOPO-AMPLIFIÉ — la topologie sert à AMPLIFIER les coefficients
de la formule de base, pas à la remplacer.

Structure :
   m(P) = squelette_combinatoire(P) · ampli_topologique(P) · couplage_géom(P)

Pour chaque nœud K, on dispose des invariants topologiques :
   cr(K)       = nombre de croisements (brins)
   n_lobes(K)  = nombre de lobes (boucles)
   b(K)        = indice de tresse
   det(K)      = déterminant de Jones |V_K(-1)|
   σ(K)        = signature
   bridge(K)   = indice de pontage

Les coefficients d'amplification :
   A_brins(K)   = b(K) · det(K)        (force par croisement amplifiée par tresse·det)
   A_boucles(K) = n_lobes(K) · b(K)    (force par boucle amplifiée par tresse)

Opérateur inventé :
   m(P) = Σ_i [cr_i · m_c · A_brins(K_i)^p_s + n_l_i · m_l · A_boucles(K_i)^p_l]
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from theory.jones import jones_polynomial, jones_determinant, braid_index

M_E = 0.51099895
F_KAPPA = 0.8682
M_C = M_E * F_KAPPA / 3        # = 0.147883
M_L = M_E * (1 - F_KAPPA) / 3  # = 0.022450

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_LOBES = {
    "3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2,
    "6_2": 3, "6_3": 3, "7_1": 7, "7_2": 2, "7_3": 3,
    "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3, "8_1": 2,
    "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2,
}

# Pré-calcul des invariants
TOPO = {}
for K in KNOTS:
    p = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(p),
    }


def cr(K): return TOPO[K]["cr"]
def n_l(K): return TOPO[K]["n_l"]
def b(K): return TOPO[K]["b"]
def det(K): return TOPO[K]["det"]


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


# Invariants 3_1 pour normalisation
A_brins_31 = b("3_1") * det("3_1")       # 2 · 3 = 6
A_boucles_31 = n_l("3_1") * b("3_1")     # 3 · 2 = 6


# ───────────────────────────────────────────────────────────────────
# OPÉRATEUR "BRINS-BOUCLES TOPO-AMPLIFIÉ"
# ───────────────────────────────────────────────────────────────────
def brins_boucles_amp(comps, params):
    """m(P) = Σ_i [cr_i·m_c·A_brins(K_i)^p_s + n_l_i·m_l·A_boucles(K_i)^p_l]

    A_brins(K) = (b(K)·det(K)) / (b(3_1)·det(3_1))  (amplificateur brins)
    A_boucles(K) = (n_l(K)·b(K)) / (n_l(3_1)·b(3_1))  (amplificateur boucles)
    """
    p_s, p_l = params
    total = 0.0
    for K in comps:
        A_s = b(K) * det(K) / A_brins_31
        A_l = n_l(K) * b(K) / A_boucles_31
        m_s = cr(K) * M_C * A_s ** p_s
        m_l_k = n_l(K) * M_L * A_l ** p_l
        total += m_s + m_l_k
    return total


def brins_boucles_amp_holder(comps, params):
    """Variante Hölder : m(P) = (Σ_i [m_naive_i · A_topo_i]^a)^(1/a)

    m_naive_i = cr_i·m_c + n_l_i·m_l
    A_topo_i = (b_i · det_i · n_l_i)^p   (amplificateur composite)
    """
    a, p = params
    A_31_topo = b("3_1") * det("3_1") * n_l("3_1")
    ws = []
    for K in comps:
        m_n = cr(K) * M_C + n_l(K) * M_L
        A = (b(K) * det(K) * n_l(K) / A_31_topo) ** p
        ws.append(m_n * A)
    if abs(a) < 1e-8:
        return float(np.prod(ws) ** (1 / len(ws)))
    return float(sum(w ** a for w in ws) ** (1 / a))


def topo_amp_product(comps, params):
    """m(P) = (Σ m_naive_i^a)^(1/a) · Π_i (A_topo_i)^b

    Le squelette est Hölder de m_naive, et l'amplification est le PRODUIT
    de tous les amplificateurs topologiques (effet multiplicatif).
    """
    a, q = params
    A_31_topo = b("3_1") * det("3_1") * n_l("3_1")
    m_naive = [cr(K) * M_C + n_l(K) * M_L for K in comps]
    if abs(a) < 1e-8:
        skeleton = float(np.prod(m_naive) ** (1 / len(m_naive)))
    else:
        skeleton = float(sum(m ** a for m in m_naive) ** (1 / a))
    amp = float(np.prod([b(K) * det(K) * n_l(K) / A_31_topo for K in comps]) ** q)
    return skeleton * amp


def topo_holder_amp(comps, params):
    """Hölder sur m_naive × amplificateur topologique distinct par nœud.

    m(P) = [Σ_i (m_naive · A_topo)^a]^(1/a) · (somme A_topo)^b
    """
    a, b_exp = params
    A_31 = b("3_1") * det("3_1") * n_l("3_1")
    A_i = [b(K) * det(K) * n_l(K) / A_31 for K in comps]
    m_naive_i = [cr(K) * M_C + n_l(K) * M_L for K in comps]
    if abs(a) < 1e-8:
        skel = float(np.prod([mn * A for mn, A in zip(m_naive_i, A_i)]) ** (1 / len(comps)))
    else:
        skel = float(sum((mn * A) ** a for mn, A in zip(m_naive_i, A_i)) ** (1 / a))
    amp_global = sum(A_i) ** b_exp
    return skel * amp_global


OPERATORS = [
    ("brins-boucles amp séparée",      brins_boucles_amp,        2),
    ("Hölder ampli×m_naive",            brins_boucles_amp_holder, 2),
    ("Hölder + produit topo amp",       topo_amp_product,         2),
    ("Hölder ampli + somme topo",       topo_holder_amp,          2),
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


def fit(op, np_):
    best = None
    for s1 in [0.3, 0.5, 1.0, 2.0]:
        for s2 in [0.5, 1.0, 1.5, 2.0]:
            x0 = [s1, s2][:np_]
            if len(x0) < np_: x0 = x0 + [1.0] * (np_ - len(x0))
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
    print("  OPÉRATEUR TOPO-AMPLIFIÉ — topologie pour amplifier les coefficients")
    print("=" * 100)

    print(f"\n  Invariants topologiques utilisés :")
    print(f"  {'K':>5} {'cr':>3} {'n_l':>4} {'b':>3} {'det':>4}")
    for K in KNOTS:
        d = TOPO[K]
        print(f"  {K:>5} {d['cr']:>3} {d['n_l']:>4} {d['b']:>3} {d['det']:>4}")

    print(f"\n  {'opérateur':<35} {'params':<22} {'RMS':>9} {'★+★★':>6} {'<20%':>6}")
    print("  " + "-" * 85)

    results = []
    for name, op, np_ in OPERATORS:
        params, rms = fit(op, np_)
        rows, flags = evaluate(op, params)
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        results.append((name, op, params, rms, n5, n20, rows, flags))
        ps = ", ".join(f"{p:.3f}" for p in params)
        print(f"  {name:<35} {ps:<22} {rms:>9.4f} {n5:>6} {n20:>6}")

    best = min(results, key=lambda r: r[3])
    print(f"\n  ─── MEILLEUR : {best[0]} ───")
    print(f"  RMS log = {best[3]:.4f}, ★+★★={best[4]}/25, <20%={best[5]}/25")
    print(f"\n  Prédictions :")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} {'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 67)
    for nm, comps, me, m_pred, err, flag in best[6]:
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} {m_pred:>11.2f} {err:>+8.2f} {flag:>5}")
    print("=" * 100)


if __name__ == "__main__":
    main()
