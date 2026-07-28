r"""
OPÉRATEUR DE HÖLDER avec COEFFICIENT k_b déduit du rapport m(n)/m(p).

Idée physique :
   Le proton (Lk = 3 = Lk_opt) a ses deux 7_2 RELAXÉS (boucle libre).
   Le neutron (Lk = 2 ≠ Lk_opt) a une tension topologique : l'une des
   deux 7_2 est BLOQUÉE par la contrainte d'enlacement non-optimal.

   On introduit un coefficient k_b appliqué à la boucle bloquée :
       m(7_2)_bloqué = k_b · m(7_2)_libre

   k_b est déterminé exactement par la masse du neutron :
       Proton  : (m_e^a + 2·m(7_2)^a)^(1/a)              = m_p
       Neutron : (m_e^a + m(7_2)^a + (k_b·m(7_2))^a)^(1/a) = m_n

   ⟹  k_b^a = 2·(m_n^a - m_e^a)/(m_p^a - m_e^a) - 1

Aucun paramètre libre supplémentaire : k_b est entièrement contraint
par les 3 ancrages exacts (e⁻, p, n) une fois a fixé.

Variables libres : 17 poids m(K) pour K ∉ {3_1, 7_2} + paramètre a.
Ancrages : m(3_1) = m_e, m(7_2) du proton, k_b du neutron.
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
    ("p",     ["3_1", "7_2", "7_2"],         M_P),     # 2× 7_2 libres
    ("n",     ["3_1", "7_2", "7_2_b"],       M_N),     # 1 libre + 1 bloqué
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


def m_72_from_proton(a):
    diff = M_P ** a - M_E ** a
    if diff <= 0:
        return None
    return (diff / 2) ** (1 / a)


def k_b_from_neutron(a):
    """k_b^a = 2(m_n^a - m_e^a) / (m_p^a - m_e^a) - 1"""
    num = 2 * (M_N ** a - M_E ** a)
    den = M_P ** a - M_E ** a
    val = num / den - 1
    if val <= 0:
        return None
    return val ** (1 / a)


def get_weights(comps, w_base, k_b):
    """Renvoie la liste de poids pour une config, gérant '7_2_b' = bloqué."""
    weights = []
    for K in comps:
        if K == "7_2_b":
            weights.append(k_b * w_base[IDX_72])
        else:
            weights.append(w_base[KNOTS.index(K)])
    return weights


def fit():
    free_idx = [i for i in range(N_K) if i not in (IDX_31, IDX_72)]
    N_free = len(free_idx)

    def unpack(x):
        log_w = x[:N_free]
        a = x[-1]
        w = np.zeros(N_K)
        w[IDX_31] = M_E
        m72 = m_72_from_proton(a)
        if m72 is None or not np.isfinite(m72):
            return None, None, None
        w[IDX_72] = m72
        kb = k_b_from_neutron(a)
        if kb is None or not np.isfinite(kb):
            return None, None, None
        for k, idx in enumerate(free_idx):
            w[idx] = np.exp(log_w[k])
        return w, a, kb

    def loss(x):
        w, a, kb = unpack(x)
        if w is None:
            return 1e6
        residuals = []
        for nm, comps, me in PARTICLES:
            ws = get_weights(comps, w, kb)
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
    for a0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        x0 = np.concatenate([init_w_log, [a0]])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 60000, "xatol": 1e-9,
                                "fatol": 1e-11, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
            kb = k_b_from_neutron(res.x[-1])
            print(f"    trial a₀={a0}: RMS={res.fun:.4f}, "
                  f"a={res.x[-1]:.5f}, k_b={kb:.6f}")
    w, a, kb = unpack(best.x)
    return w, a, kb, best.fun


def main():
    print("=" * 105)
    print("  HÖLDER + COEFFICIENT k_b DÉDUIT DU RAPPORT m(n)/m(p)")
    print("=" * 105)
    print(f"\n  Données expérimentales :")
    print(f"    m(e⁻) = {M_E} MeV")
    print(f"    m(p)  = {M_P} MeV  (Lk=3 optimal → 2× 7_2 libres)")
    print(f"    m(n)  = {M_N} MeV  (Lk=2 ≠ opt  → 1× libre + 1× bloqué)")
    print(f"    rapport m(n)/m(p) = {M_N/M_P:.8f}")
    print(f"\n  Modèle :")
    print(f"    m(P) = (Σ m(K)^a)^(1/a)")
    print(f"    m(7_2)_libre  = ((m(p)^a − m(e)^a)/2)^(1/a)")
    print(f"    m(7_2)_bloqué = k_b · m(7_2)_libre")
    print(f"    k_b^a = 2(m(n)^a − m(e)^a)/(m(p)^a − m(e)^a) − 1")

    print(f"\n  Optimisation (Nelder-Mead, 5 restarts) :")
    w, a, kb, rms = fit()

    m72 = w[IDX_72]
    m72b = kb * m72
    print(f"\n  ─── PARAMÈTRES DÉTERMINÉS ───")
    print(f"  a              = {a:.6f}  (Hölder, libre, optimisé)")
    print(f"  m(3_1)         = {M_E:.5f} MeV  (ancré sur e⁻)")
    print(f"  m(7_2)_libre   = {m72:.5f} MeV  (déduit du proton)")
    print(f"  k_b            = {kb:.6f}        (déduit du neutron)")
    print(f"  m(7_2)_bloqué  = {m72b:.5f} MeV  (= k_b · m(7_2)_libre)")
    print(f"  m(7_2)_bloqué / m(7_2)_libre = {kb:.6f}")

    # Vérification triple ancrage
    print(f"\n  ─── VÉRIFICATION DES 3 ANCRAGES ───")
    for nm, comps, me in [PARTICLES[0], PARTICLES[16], PARTICLES[17]]:
        ws = get_weights(comps, w, kb)
        m_pred = holder_mean(ws, a)
        err = (m_pred - me) / me * 100
        print(f"    {nm:>3} ({'+'.join(comps):>17}) : "
              f"m_exp={me:.5f}, m_pred={m_pred:.5f}, Δ={err:+.5f}%")

    # Tableau des 19 poids
    print(f"\n  ─── TABLEAU DES 19 POIDS m(K) ───")
    print(f"  {'K':>5} {'m(K) MeV':>15} {'cr':>3} {'note':<30}")
    print("  " + "-" * 60)
    for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
        cr = int(K.split("_")[0])
        if K == "3_1":
            note = "★ ancré (e⁻)"
        elif K == "7_2":
            note = f"★ déduit du p ; bloqué = ×k_b"
        else:
            note = ""
        print(f"  {K:>5} {mK:>15.4f} {cr:>3} {note:<30}")

    # Prédictions complètes
    print(f"\n  ─── PRÉDICTIONS SUR 31 PARTICULES OBSERVABLES ───")
    print(f"  {'particule':>5} {'n':>2} {'configuration':>17} "
          f"{'m_exp':>12} {'m_pred':>12} {'Δ %':>8} {'flag':>5}")
    print("  " + "-" * 78)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res_log = []
    for nm, comps, me in PARTICLES:
        ws = get_weights(comps, w, kb)
        m_pred = holder_mean(ws, a)
        err = (m_pred - me) / me * 100
        res_log.append(np.log(m_pred / me))
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
        print(f"  {nm:>5} {len(comps):>2} {'+'.join(comps):>17} "
              f"{me:>12.4f} {m_pred:>12.4f} {err:>+7.3f} {flag:>5}")

    rms_log = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    print(f"\n  RMS log = {rms_log:.4f}  (= +{(np.exp(rms_log)-1)*100:.1f}% relatif)")

    total = len(PARTICLES)
    excellent = flags["★★"] + flags["★"]
    good = excellent + flags["✓"]
    print(f"\n  ─── BILAN ───")
    print(f"  ★★ (<0.5%) : {flags['★★']}/{total}")
    print(f"  ★  (<5%)   : {flags['★']}/{total}")
    print(f"  ✓  (<20%)  : {flags['✓']}/{total}")
    print(f"     (<50%)  : {flags[' ']}/{total}")
    print(f"  ?  (>50%)  : {flags['?']}/{total}")
    print(f"\n  Précision <5%  : {excellent}/{total} ({excellent/total*100:.0f}%)")
    print(f"  Précision <20% : {good}/{total} ({good/total*100:.0f}%)")
    print("=" * 105)


if __name__ == "__main__":
    main()
