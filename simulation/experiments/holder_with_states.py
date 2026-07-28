r"""
OPÉRATEUR DE HÖLDER avec ÉTATS PHYSIQUES DU 7_2.

Hypothèse physique : le même nœud topologique 7_2 peut être dans
deux états physiques différents selon le contexte :
  - "BLOQUÉ"  : 7_2 apparié avec un autre 7_2 (configuration symétrique
                stable, énergie verrouillée par appariement)
  - "SERRÉ"   : 7_2 seul dans l'entrelacs (configuration libre,
                comprimée, énergie supérieure)

Règle d'attribution :
  - si 7_2 apparaît ≥2 fois dans la config → mode BLOQUÉ
  - si 7_2 apparaît une seule fois         → mode SERRÉ

Formule :
  m(7_2)_bloqué  = m(7_2)         (référence)
  m(7_2)_serré   = k_s · m(7_2)   (avec k_s à optimiser)

Ancrages exacts :
  m(3_1)         = m_e = 0.51099895 MeV       (électron)
  m(7_2)_bloqué  = déduit de m(p) = 938.272   (proton avec 7_2 paired)

Optimisation libre :
  17 poids m(K) pour K ∉ {3_1, 7_2}
  paramètre a (Hölder)
  paramètre k_s (ratio serré/bloqué pour 7_2)

→ 19 paramètres pour 31 particules observables.
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

# 31 particules observables
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


def state_of_72(comps):
    """Retourne 'bloqué' si 7_2 apparaît ≥2 fois, 'serré' si 1 fois, None si absent."""
    n_72 = comps.count("7_2")
    if n_72 >= 2:
        return "bloqué"
    elif n_72 == 1:
        return "serré"
    else:
        return None


def holder_mean(values, a):
    if abs(a) < 1e-8:
        return float(np.prod(values) ** (1 / len(values)))
    s = sum(v ** a for v in values)
    return float(s ** (1 / a))


def get_weights_for_config(comps, w_base, k_s):
    """Renvoie les poids effectifs pour chaque K de la config,
    en distinguant 7_2 bloqué vs serré."""
    n_72 = comps.count("7_2")
    state = state_of_72(comps)
    weights = []
    for K in comps:
        if K == "7_2":
            if state == "bloqué":
                weights.append(w_base[IDX_72])           # référence
            else:  # serré
                weights.append(k_s * w_base[IDX_72])     # facteur
        else:
            weights.append(w_base[KNOTS.index(K)])
    return weights


def m_72_bloqué_from_proton(a):
    """Calcule m(7_2)_bloqué tel que m_pred(proton) = M_P."""
    diff = M_P ** a - M_E ** a
    if diff <= 0:
        return None
    return (diff / 2) ** (1 / a)


def fit_with_states():
    """17 poids libres + a + k_s, contraintes : m(3_1)=M_E, m(7_2)_b déduit du p."""
    free_idx = [i for i in range(N_K) if i not in (IDX_31, IDX_72)]
    N_free = len(free_idx)

    def unpack(x):
        log_w = x[:N_free]
        a = x[-2]
        k_s = x[-1]
        w = np.zeros(N_K)
        w[IDX_31] = M_E
        m_72b = m_72_bloqué_from_proton(a)
        if m_72b is None or not np.isfinite(m_72b):
            return None, None, None
        w[IDX_72] = m_72b
        for k, idx in enumerate(free_idx):
            w[idx] = np.exp(log_w[k])
        return w, a, k_s

    def loss(x):
        w, a, k_s = unpack(x)
        if w is None:
            return 1e6
        residuals = []
        for nm, comps, me in PARTICLES:
            ws = get_weights_for_config(comps, w, k_s)
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
    for trial in [(0.1, 1.0), (0.2, 1.5), (0.3, 2.0), (0.4, 1.0),
                  (0.5, 0.7), (0.3, 3.0), (0.3, 0.5)]:
        a0, ks0 = trial
        x0 = np.concatenate([init_w_log, [a0, ks0]])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 80000, "xatol": 1e-9,
                                "fatol": 1e-11, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
            a_opt, ks_opt = res.x[-2], res.x[-1]
            print(f"    trial (a₀={a0}, k_s₀={ks0}): RMS = {res.fun:.4f}, "
                  f"a = {a_opt:.4f}, k_s = {ks_opt:.4f}")
    w, a, k_s = unpack(best.x)
    return w, a, k_s, best.fun


def main():
    print("=" * 105)
    print("  HÖLDER À DOUBLE ANCRAGE + ÉTATS PHYSIQUES DU 7_2 (bloqué vs serré)")
    print("=" * 105)
    print(f"\n  Hypothèse : 7_2 a deux états selon contexte :")
    print(f"    - bloqué : 7_2 apparié (≥2 fois dans la config) → référence")
    print(f"    - serré  : 7_2 seul (1 fois) → masse différente")
    print(f"\n  Ancrages :")
    print(f"    m(3_1) = {M_E} MeV (e⁻)")
    print(f"    m(7_2)_bloqué = déduit de m(p) = {M_P} MeV")
    print(f"\n  Particules avec 7_2 :")
    for nm, comps, me in PARTICLES:
        if "7_2" in comps:
            state = state_of_72(comps)
            n = comps.count("7_2")
            print(f"    {nm:>5} ({n}× 7_2) → état {state}")

    print(f"\n  Optimisation (Nelder-Mead, 7 restarts) :")
    w, a, k_s, rms = fit_with_states()

    m_72b = w[IDX_72]
    m_72s = k_s * m_72b
    print(f"\n  ─── PARAMÈTRES OPTIMISÉS ───")
    print(f"  a              = {a:.5f}")
    print(f"  k_s            = {k_s:.5f}")
    print(f"  m(7_2)_bloqué  = {m_72b:.4f} MeV  (référence, déduit du p)")
    print(f"  m(7_2)_serré   = {m_72s:.4f} MeV  (= {k_s:.3f} × bloqué)")

    print(f"\n  ─── TABLEAU DES POIDS m(K) (19 nœuds, pas de doublons) ───")
    print(f"  {'K':>5} {'m(K) MeV':>15} {'cr':>3} {'note':<30}")
    print("  " + "-" * 60)
    for K, mK in sorted(zip(KNOTS, w), key=lambda kv: kv[1]):
        cr = int(K.split("_")[0])
        if K == "3_1":
            note = "★ ANCRÉ (e⁻)"
        elif K == "7_2":
            note = f"★ déduit du p ; serré = ×{k_s:.3f}"
        else:
            note = ""
        print(f"  {K:>5} {mK:>15.4f} {cr:>3} {note:<30}")

    # Prédictions
    print(f"\n  ─── PRÉDICTIONS SUR 31 PARTICULES OBSERVABLES ───")
    print(f"  {'particule':>5} {'n':>2} {'configuration':>16} {'72?':>4} "
          f"{'m_exp':>12} {'m_pred':>12} {'Δ %':>8} {'flag':>5}")
    print("  " + "-" * 82)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res_log = []
    rows = []
    for nm, comps, me in PARTICLES:
        ws = get_weights_for_config(comps, w, k_s)
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
        state = state_of_72(comps) or "-"
        state_str = "b" if state == "bloqué" else ("s" if state == "serré" else "-")
        print(f"  {nm:>5} {len(comps):>2} {'+'.join(comps):>16} {state_str:>4} "
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
    print(f"\n  Précision <5%  : {excellent}/{total} ({excellent/total*100:.0f}%)")
    print(f"  Précision <20% : {good}/{total} ({good/total*100:.0f}%)")

    print(f"\n  ─── COMPARAISON AVEC SIMPLE ANCRAGE (sans états) ───")
    print(f"  Sans états (1 valeur m(7_2)) : RMS = 0.179, 20★/31 = 56%")
    print(f"  Avec états (bloqué/serré)    : RMS = {rms_log:.3f}, "
          f"{excellent}★/{total} = {excellent/total*100:.0f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
