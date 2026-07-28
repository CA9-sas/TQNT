r"""
HÖLDER avec k_b (boucle bloquée, déduit de n/p) ET k_s (boucle serrée).

Définitions :
   - 7_2 LIBRE   : 7_2 apparié avec un autre 7_2 (proton)
                  → poids = m(7_2)
   - 7_2 BLOQUÉ  : 7_2 sous contrainte d'enlacement non-optimal (neutron)
                  → poids = k_b · m(7_2), k_b déduit de m(n)/m(p)
   - 7_2 SERRÉ   : 7_2 seul dans la config (Σ, Ξ, Σ*, Ξ*, Ω⁻, Z⁰)
                  → poids = k_s · m(7_2)

Recherche d'un k_s PHYSIQUEMENT MOTIVÉ (pas un fit libre).

On teste plusieurs candidats :
   k_s = 1      (pas d'effet, baseline)
   k_s = √2     (1.414, géométrique)
   k_s = φ      (1.618, nombre d'or)
   k_s = π/2    (1.571)
   k_s = e/2    (1.359)
   k_s = 2      (doublement, "2 brins/1 brin")
   k_s = e      (2.718)
   k_s libre    (référence du fit optimal)

Pour chaque candidat fixé, on optimise les 17 autres poids + a et on
compare le RMS log.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]
N_K = len(KNOTS)
M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
IDX_31 = KNOTS.index("3_1")
IDX_72 = KNOTS.index("7_2")

PARTICLES = [
    ("e⁻",    ["3_1"],                       M_E),
    ("μ⁻",    ["3_1", "6_1"],                105.6584),
    ("τ⁻",    ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",    ["8_1"],                       134.9768),
    ("π±",    ["8_1"],                       139.5704),
    ("K±",    ["4_1", "4_1", "5_2"],         493.677),
    ("K⁰_S",  ["4_1", "4_1", "5_2"],         497.611),
    ("η",     ["3_1", "4_1", "5_1"],         547.862),
    ("η'",    ["5_2", "5_2", "6_1"],         957.78),
    ("ρ/ω",   ["5_2", "7_7"],                775.0),
    ("φ",     ["3_1", "4_1", "6_2"],         1019.461),
    ("J/ψ",   ["4_1", "5_2", "9_1"],         3096.900),
    ("Υ",     ["3_1", "7_4", "9_1"],         9460.30),
    ("D⁰",    ["6_2", "7_7"],                1864.84),
    ("D±",    ["6_2", "7_7"],                1869.66),
    ("B",     ["4_1", "10_1"],               5279.4),
    ("p",     ["3_1", "7_2", "7_2"],         M_P),
    ("n",     ["3_1", "7_2", "7_2_b"],       M_N),
    ("Λ",     ["4_1", "7_4"],                1115.683),
    ("Σ⁺",    ["6_2", "7_2_s"],              1189.37),
    ("Σ⁰",    ["6_2", "7_2_s"],              1192.642),
    ("Σ⁻",    ["6_2", "7_2_s"],              1197.449),
    ("Ξ⁰",    ["4_1", "4_1", "7_2_s"],       1314.86),
    ("Ξ⁻",    ["4_1", "4_1", "7_2_s"],       1321.71),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),
    ("Σ*",    ["5_2", "6_2", "7_2_s"],       1383.7),
    ("Ξ*",    ["4_1", "5_2", "7_2_s"],       1531.8),
    ("Ω⁻",    ["5_2", "5_2", "7_2_s"],       1672.45),
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Σc",    ["4_1", "5_1", "7_1"],         2453.97),
    ("Ωc",    ["3_1", "6_2", "6_2"],         2695.2),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.60),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["5_1", "7_2_s", "8_3"],       91187.6),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def holder_mean(values, a):
    if abs(a) < 1e-8:
        return float(np.prod(values) ** (1 / len(values)))
    s = sum(v ** a for v in values)
    return float(s ** (1 / a))


def m_72_from_p(a):
    diff = M_P ** a - M_E ** a
    return (diff / 2) ** (1 / a) if diff > 0 else None


def k_b_from_n(a):
    num = 2 * (M_N ** a - M_E ** a)
    den = M_P ** a - M_E ** a
    val = num / den - 1
    return val ** (1 / a) if val > 0 else None


def get_weights(comps, w_base, k_b, k_s):
    weights = []
    for K in comps:
        if K == "7_2_b":
            weights.append(k_b * w_base[IDX_72])
        elif K == "7_2_s":
            weights.append(k_s * w_base[IDX_72])
        else:
            weights.append(w_base[KNOTS.index(K)])
    return weights


def fit_with_ks(k_s_fixed=None):
    """Fit avec k_s soit fixé soit optimisé."""
    free_idx = [i for i in range(N_K) if i not in (IDX_31, IDX_72)]
    N_free = len(free_idx)
    optimize_ks = k_s_fixed is None

    def unpack(x):
        log_w = x[:N_free]
        a = x[N_free]
        k_s = np.exp(x[N_free + 1]) if optimize_ks else k_s_fixed
        w = np.zeros(N_K)
        w[IDX_31] = M_E
        m72 = m_72_from_p(a)
        if m72 is None or not np.isfinite(m72):
            return None, None, None, None
        w[IDX_72] = m72
        kb = k_b_from_n(a)
        if kb is None or not np.isfinite(kb):
            return None, None, None, None
        for k, idx in enumerate(free_idx):
            w[idx] = np.exp(log_w[k])
        return w, a, kb, k_s

    def loss(x):
        w, a, kb, ks = unpack(x)
        if w is None:
            return 1e6
        residuals = []
        for nm, comps, me in PARTICLES:
            ws = get_weights(comps, w, kb, ks)
            try:
                m_pred = holder_mean(ws, a)
            except (OverflowError, ValueError):
                return 1e6
            if m_pred <= 0 or not np.isfinite(m_pred):
                residuals.append(20)
            else:
                residuals.append(np.log(m_pred / me))
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    init_w_log = [np.log(2 * int(KNOTS[i].split("_")[0])) for i in free_idx]
    best = None
    for a0 in [0.15, 0.25, 0.35, 0.5]:
        if optimize_ks:
            x0 = np.concatenate([init_w_log, [a0, np.log(2.0)]])
        else:
            x0 = np.concatenate([init_w_log, [a0]])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 60000, "xatol": 1e-9,
                                "fatol": 1e-11, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    w, a, kb, ks = unpack(best.x)
    return w, a, kb, ks, best.fun


def evaluate(w, a, kb, ks):
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res_log = []
    rows = []
    for nm, comps, me in PARTICLES:
        ws = get_weights(comps, w, kb, ks)
        m_pred = holder_mean(ws, a)
        err = (m_pred - me) / me * 100
        res_log.append(np.log(m_pred / me))
        rows.append((nm, comps, me, m_pred, err))
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
    rms = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    return rows, flags, rms


def main():
    print("=" * 105)
    print("  HÖLDER + k_b (bloqué, de n/p) + k_s (serré, candidats physiques)")
    print("=" * 105)

    candidates = [
        ("1.000  (baseline, pas d'effet)", 1.0),
        ("√2 = 1.414 (géométrique 2D)",     np.sqrt(2)),
        ("e/2 = 1.359",                     np.e / 2),
        ("π/2 = 1.571",                     np.pi / 2),
        ("φ = 1.618 (nombre d'or)",         (1 + np.sqrt(5)) / 2),
        ("e/√2 = 1.922",                    np.e / np.sqrt(2)),
        ("2.000 (doublement)",              2.0),
        ("e = 2.718",                       np.e),
        ("π = 3.142",                       np.pi),
        ("LIBRE (fit optimal)",             None),
    ]

    print(f"\n  {'k_s candidat':<32} {'a optimal':>10} {'k_b':>9} "
          f"{'RMS log':>9} {'<5%':>5} {'<20%':>5}")
    print("  " + "-" * 80)

    results = []
    for name, val in candidates:
        w, a, kb, ks, rms = fit_with_ks(k_s_fixed=val)
        rows, flags, rms_eval = evaluate(w, a, kb, ks)
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        results.append((name, val, w, a, kb, ks, rms, n5, n20, flags, rows))
        print(f"  {name:<32} {a:>10.5f} {kb:>9.5f} "
              f"{rms:>9.4f} {n5:>5} {n20:>5}")

    # Meilleur candidat
    best_idx = min(range(len(results)), key=lambda i: results[i][6])
    best = results[best_idx]
    print(f"\n  ─── MEILLEUR : {best[0]} (k_s = {best[5]:.5f}) ───")
    print(f"  a = {best[3]:.5f}, k_b = {best[4]:.5f}, k_s = {best[5]:.5f}")
    print(f"  m(7_2)         = {best[2][IDX_72]:.4f} MeV")
    print(f"  m(7_2)_bloqué  = {best[4] * best[2][IDX_72]:.4f} MeV")
    print(f"  m(7_2)_serré   = {best[5] * best[2][IDX_72]:.4f} MeV")

    # Détails des prédictions du meilleur
    print(f"\n  Prédictions ({best[0]}):")
    print(f"  {'particule':>5} {'configuration':>17} "
          f"{'m_exp':>11} {'m_pred':>11} {'Δ %':>8} {'flag':>5}")
    print("  " + "-" * 65)
    for nm, comps, me, m_pred, err in best[10]:
        cfg = "+".join(comps)
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        print(f"  {nm:>5} {cfg:>17} {me:>11.3f} {m_pred:>11.3f} "
              f"{err:>+7.2f} {flag:>5}")

    print(f"\n  ─── COMPARAISON RANG-PAR-RANG ───")
    print(f"  {'k_s candidat':<32} {'RMS log':>9} {'<5%/36':>9} {'<20%/36':>9}")
    print("  " + "-" * 65)
    for name, val, w, a, kb, ks, rms, n5, n20, _, _ in sorted(results, key=lambda r: r[6]):
        ks_str = f"{val:.4f}" if val is not None else f"{ks:.4f} (fit)"
        print(f"  {name:<32} {rms:>9.4f} {n5:>9} {n20:>9}")
    print("=" * 105)


if __name__ == "__main__":
    main()
