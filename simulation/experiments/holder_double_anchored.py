r"""
OPÉRATEUR DE HÖLDER À DEUX ANCRAGES : électron ET proton.

m(P) = (Σ m(K)^a)^(1/a)

Contraintes EXACTES :
  C1 : m(3_1) = m_e = 0.51099895 MeV  (électron)
  C2 : m_pred(p) = m_exp(p) = 938.27208 MeV  (proton)

La C2 fixe m(7_2) en fonction de m(3_1) et a, via la relation
  (m(3_1)^a + 2·m(7_2)^a)^(1/a) = m_p
  ⟹ m(7_2) = ((m_p^a - m(3_1)^a) / 2)^(1/a)

Donc on a 17 poids libres (les autres K) + le paramètre `a`,
soit 18 variables à optimiser, avec p et e⁻ pinés EXACTEMENT.

Le neutron (même config 3_1+7_2+7_2) sera prédit identique au proton :
   m_pred(n) = 938.27 MeV, exp = 939.57, écart = +0.14% (acceptable).
Le splitting n-p (1.3 MeV) doit venir d'un autre mécanisme
(linking tension, cf. CLAUDE.md).

On exclut TOUTES les masses de quarks (théoriques) :
   u, d, s, c, b → exclus
   → 31 particules observables conservées.
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
IDX_31 = KNOTS.index("3_1")
IDX_72 = KNOTS.index("7_2")

# 31 particules observables (quarks exclus)
PARTICLES = [
    ("e⁻",    ["3_1"],                       M_E),         # ANCRE 1
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
    ("p",     ["3_1", "7_2", "7_2"],         M_P),         # ANCRE 2
    ("n",     ["3_1", "7_2", "7_2"],         939.5654),
    ("Λ",     ["4_1", "7_4"],                1115.683),
    ("Σ⁺",    ["6_2", "7_2"],                1189.37),
    ("Σ⁰",    ["6_2", "7_2"],                1192.642),
    ("Σ⁻",    ["6_2", "7_2"],                1197.449),
    ("Ξ⁰",    ["4_1", "4_1", "7_2"],         1314.86),
    ("Ξ⁻",    ["4_1", "4_1", "7_2"],         1321.71),
    ("Δ",     ["3_1", "5_2", "6_2"],         1232.0),
    ("Σ*",    ["5_2", "6_2", "7_2"],         1383.7),
    ("Ξ*",    ["4_1", "5_2", "7_2"],         1531.8),
    ("Ω⁻",    ["5_2", "5_2", "7_2"],         1672.45),
    ("Λc",    ["6_3", "8_1"],                2286.46),
    ("Σc",    ["4_1", "5_1", "7_1"],         2453.97),
    ("Ωc",    ["3_1", "6_2", "6_2"],         2695.2),
    ("Λb",    ["5_2", "5_2", "8_2"],         5619.60),
    ("W±",    ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",    ["5_1", "7_2", "8_3"],         91187.6),
    ("H",     ["6_1", "7_4", "8_3"],         125250.0),
    ("t",     ["7_4", "7_6", "8_3"],         173100.0),
]


def holder_mean(values, a):
    if abs(a) < 1e-8:
        return float(np.prod(values) ** (1 / len(values)))
    s = sum(v ** a for v in values)
    return float(s ** (1 / a))


def m_7_2_from_proton(a):
    """Calcule m(7_2) tel que m_pred(proton) = M_P avec m(3_1) = M_E."""
    # (M_E^a + 2·m_7_2^a)^(1/a) = M_P
    # 2·m_7_2^a = M_P^a - M_E^a
    diff = M_P ** a - M_E ** a
    if diff <= 0:
        return None
    return (diff / 2) ** (1 / a)


def fit_double_anchored():
    """Optimisation : 17 poids libres + a, m(3_1) et m(7_2) contraints."""
    free_idx = [i for i in range(N_K) if i not in (IDX_31, IDX_72)]
    N_free = len(free_idx)

    def unpack(x):
        log_w = x[:N_free]
        a = x[-1]
        w = np.zeros(N_K)
        w[IDX_31] = M_E
        m_72 = m_7_2_from_proton(a)
        if m_72 is None or not np.isfinite(m_72):
            return None, None
        w[IDX_72] = m_72
        for k, idx in enumerate(free_idx):
            w[idx] = np.exp(log_w[k])
        return w, a

    def loss(x):
        w, a = unpack(x)
        if w is None:
            return 1e6
        residuals = []
        for nm, comps, me in PARTICLES:
            ws = [w[KNOTS.index(K)] for K in comps]
            try:
                m_pred = holder_mean(ws, a)
            except (OverflowError, ValueError):
                return 1e6
            if m_pred <= 0 or not np.isfinite(m_pred):
                residuals.append(20)
            else:
                residuals.append(np.log(m_pred / me))
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    init_w_log = []
    for idx in free_idx:
        cr = int(KNOTS[idx].split("_")[0])
        init_w_log.append(np.log(2 * cr))

    best = None
    for trial_a in [0.1, 0.2, 0.285, 0.4, 0.5, 0.7, 1.0]:
        x0 = np.concatenate([init_w_log, [trial_a]])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 50000, "xatol": 1e-9,
                                "fatol": 1e-11, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
            print(f"    trial a₀={trial_a}: RMS = {res.fun:.4f}, "
                  f"a_opt = {res.x[-1]:.5f}, "
                  f"m(7_2) = {m_7_2_from_proton(res.x[-1]):.3f}")

    w, a = unpack(best.x)
    return w, a, best.fun


def main():
    print("=" * 100)
    print("  OPÉRATEUR DE HÖLDER À DEUX ANCRAGES (électron + proton)")
    print("=" * 100)
    print(f"\n  Ancrage 1 : m(3_1) = m_e = {M_E} MeV  (e⁻ = [3_1])")
    print(f"  Ancrage 2 : m_pred(p) = {M_P} MeV     (p = [3_1, 7_2, 7_2])")
    print(f"             ⟹ m(7_2) déterminé par a et m(3_1)")
    print(f"\n  Particules : 31 observables (quarks théoriques exclus)")
    print(f"  Variables  : 17 poids libres + 1 paramètre a")
    print(f"\n  Optimisation (Nelder-Mead, 7 restarts) :")

    w, a, rms = fit_double_anchored()

    m_72_val = m_7_2_from_proton(a)
    print(f"\n  ─── PARAMÈTRES ───")
    print(f"  a optimal      = {a:.5f}")
    print(f"  m(3_1)         = {M_E:.5f} MeV  (ancré sur e⁻)")
    print(f"  m(7_2)         = {m_72_val:.5f} MeV  (déduit du proton)")
    if abs(a - 2/7) < 0.01:
        print(f"  Note : a ≈ 2/7 = {2/7:.5f}  (fraction simple)")
    if abs(a - 1/4) < 0.01:
        print(f"  Note : a ≈ 1/4 = {1/4:.5f}")
    if abs(a - 1/3) < 0.01:
        print(f"  Note : a ≈ 1/3 = {1/3:.5f}")

    print(f"\n  ─── TABLEAU COMPLET DES POIDS m(K) ───")
    print(f"  {'K':>5} {'m(K) MeV':>15} {'cr':>3} {'origine':<20}")
    print("  " + "-" * 50)
    for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
        cr = int(K.split("_")[0])
        if K == "3_1":
            origin = "★ ANCRÉ (e⁻)"
        elif K == "7_2":
            origin = "★ DÉDUIT (proton)"
        else:
            origin = "libre"
        print(f"  {K:>5} {mK:>15.4f} {cr:>3} {origin:<20}")

    # Prédictions
    print(f"\n  ─── PRÉDICTIONS SUR LES 31 PARTICULES OBSERVABLES ───")
    print(f"  {'particule':>5} {'n':>2} {'configuration':>16} "
          f"{'m_exp':>12} {'m_pred':>12} {'Δ %':>8} {'flag':>5}")
    print("  " + "-" * 76)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res_log = []
    rows = []
    for nm, comps, me in PARTICLES:
        ws = [w[KNOTS.index(K)] for K in comps]
        m_pred = holder_mean(ws, a)
        err = (m_pred - me) / me * 100
        res_log.append(np.log(m_pred / me))
        rows.append((nm, comps, me, m_pred, err))
        if abs(err) < 0.5:
            flag = "★★"
        elif abs(err) < 5:
            flag = "★"
        elif abs(err) < 20:
            flag = "✓"
        elif abs(err) < 50:
            flag = " "
        else:
            flag = "?"
        flags[flag] += 1
        cfg_str = "+".join(comps)
        print(f"  {nm:>5} {len(comps):>2} {cfg_str:>16} "
              f"{me:>12.4f} {m_pred:>12.4f} {err:>+7.2f} {flag:>5}")

    rms_log = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    print(f"\n  RMS log = {rms_log:.4f}  (= +{(np.exp(rms_log)-1)*100:.1f}% relatif)")

    print(f"\n  ─── BILAN ───")
    total = len(PARTICLES)
    print(f"  ★★ (<0.5%) : {flags['★★']}/{total}")
    print(f"  ★  (<5%)   : {flags['★']}/{total}")
    print(f"  ✓  (<20%)  : {flags['✓']}/{total}")
    print(f"     (<50%)  : {flags[' ']}/{total}")
    print(f"  ?  (>50%)  : {flags['?']}/{total}")

    excellent = flags["★★"] + flags["★"]
    good = excellent + flags["✓"]
    print(f"\n  Précision < 5%  : {excellent}/{total} ({excellent/total*100:.0f}%)")
    print(f"  Précision < 20% : {good}/{total} ({good/total*100:.0f}%)")
    print("=" * 100)


if __name__ == "__main__":
    main()
