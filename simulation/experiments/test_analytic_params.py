r"""
TEST DES VALEURS ANALYTIQUES TROUVÉES sur le spectre.

Hypothèses analytiques candidates :
   β = √e                  (erreur 0.14%)
   |x| = √6                (erreur 0.27%)
   p + q = 3/2             (erreur 0.09%)
   p · q = 9/16            (erreur 0.05%)
   ⟹ p = q = 3/4 (algébriquement)
   a = ?  (à déterminer)

On teste plusieurs combinaisons et on vérifie si le spectre est
conservé. Pour chaque ensemble de paramètres analytiques, on
recalibre ρ_0 et k_b par les contraintes proton/neutron.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq, minimize
from collections import Counter

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

M_E = 0.51099895
M_P = 938.27208
M_N = 939.56542
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


def operator(comps, a, p, q, x, rho0, beta, kb=1.0, neutron_mode=False):
    n = len(comps)
    ws = []
    for K_idx, K in enumerate(comps):
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        w = M_NAIVE[K] * T ** p
        if neutron_mode and K == "7_2":
            # première occurrence de 7_2 = bloquée
            counted = sum(1 for k in comps[:K_idx] if k == "7_2")
            if counted == 1:  # 2e occurrence
                w *= kb
        ws.append(w)
    if abs(a) < 1e-8:
        skeleton = float(np.prod(ws) ** (1 / len(ws)))
    else:
        skeleton = float(sum(w ** a for w in ws) ** (1 / a))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    Phi_n = ((n + 1) / 2) ** (-x) if n > 0 else 1.0
    c = Counter(comps)
    Phi_pair = 1.0
    for K, count in c.items():
        if count >= 2:
            rho_K = rho0 * TOPO[K]["cr"] ** (-beta)
            Phi_pair *= rho_K ** (count - 1)
    return Phi_n * Phi_pair * skeleton * G_comp ** q


def solve_rho0(a, p, q, x, beta):
    def f(log_rho0):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, np.exp(log_rho0), beta) - M_P
    try:
        return float(np.exp(brentq(f, -10, 10, xtol=1e-10)))
    except Exception:
        return None


def solve_kb(a, p, q, x, rho0, beta):
    def f(log_kb):
        return operator(["3_1", "7_2", "7_2"], a, p, q, x, rho0, beta,
                        kb=np.exp(log_kb), neutron_mode=True) - M_N
    try:
        return float(np.exp(brentq(f, -5, 5, xtol=1e-10)))
    except Exception:
        return None


SPECTRUM = [
    ("μ⁻", ["3_1", "6_1"], 105.66),
    ("τ⁻", ["3_1", "5_1", "7_1"], 1776.86),
    ("π⁰", ["8_1"], 134.98),
    ("π±", ["8_1"], 139.57),
    ("K±", ["4_1", "4_1", "5_2"], 493.68),
    ("K⁰_S", ["3_1", "6_1", "6_1"], 497.611),
    ("η", ["5_2", "5_2", "6_1"], 547.86),
    ("η'", ["5_1", "5_2", "5_2"], 957.78),
    ("ρ/ω", ["5_2", "7_7"], 775.0),
    ("φ", ["3_1", "7_1", "7_1"], 1019.46),
    ("J/ψ", ["5_2", "6_1", "6_2"], 3096.9),
    ("Υ", ["3_1", "7_4", "9_1"], 9460.3),
    ("D⁰", ["6_2", "7_7"], 1864.84),
    ("B", ["4_1", "10_1"], 5279.4),
    ("Λ", ["5_1", "7_7"], 1115.7),
    ("Σ⁺", ["6_2", "7_2"], 1189.4),
    ("Δ", ["4_1", "7_2", "7_2"], 1232.0),
    ("Σ*", ["3_1", "5_2", "7_1"], 1383.7),
    ("Ξ*", ["3_1", "6_1", "7_2"], 1531.8),
    ("Ω⁻", ["3_1", "6_1", "7_1"], 1672.0),
    ("Λc", ["6_3", "8_1"], 2286.46),
    ("Λb", ["5_1", "6_1", "6_3"], 5619.6),
    ("W±", ["4_1", "7_3", "8_3"], 80369.0),
    ("Z⁰", ["7_4", "9_1", "8_3"], 91188.0),
    ("H", ["7_7", "8_2", "8_3"], 125250.0),
    ("t", ["3_1", "7_5", "8_3"], 173100.0),
]


def rms_log(a, p, q, x, beta, rho0, kb):
    res = []
    for nm, comps, me in SPECTRUM:
        if nm == "n":  # neutron en mode bloqué
            m_pred = operator(comps, a, p, q, x, rho0, beta, kb=kb, neutron_mode=True)
        else:
            m_pred = operator(comps, a, p, q, x, rho0, beta)
        if m_pred > 0:
            res.append(np.log(m_pred / me))
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def evaluate(params_dict, name):
    a, p, q, x, beta = params_dict["a"], params_dict["p"], params_dict["q"], params_dict["x"], params_dict["beta"]
    rho0 = solve_rho0(a, p, q, x, beta)
    if rho0 is None:
        return None, None, None
    kb = solve_kb(a, p, q, x, rho0, beta)
    if kb is None:
        return None, None, None
    rms = rms_log(a, p, q, x, beta, rho0, kb)
    # Count flags
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps, me in SPECTRUM:
        m_pred = operator(comps, a, p, q, x, rho0, beta)
        err = abs((m_pred - me) / me * 100)
        if err < 0.5: flag = "★★"
        elif err < 5: flag = "★"
        elif err < 20: flag = "✓"
        elif err < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
    return rms, flags, (rho0, kb)


CANDIDATES = [
    ("OPTIMUM numérique", {
        "a": 0.52825, "p": 0.77955, "q": 0.72177,
        "x": -2.45599, "beta": 1.65095,
    }),
    ("a=√e/π, p=q=3/4, |x|=√6, β=√e", {
        "a": np.sqrt(np.e) / np.pi, "p": 3/4, "q": 3/4,
        "x": -np.sqrt(6), "beta": np.sqrt(np.e),
    }),
    ("a=1/2, p=q=3/4, |x|=√6, β=√e", {
        "a": 0.5, "p": 3/4, "q": 3/4,
        "x": -np.sqrt(6), "beta": np.sqrt(np.e),
    }),
    ("a=17/32, p=25/32, q=23/32, |x|=√6, β=√e", {
        "a": 17/32, "p": 25/32, "q": 23/32,
        "x": -np.sqrt(6), "beta": np.sqrt(np.e),
    }),
    ("a=17/32, p=25/32, q=23/32, |x|=79/32, β=53/32", {
        "a": 17/32, "p": 25/32, "q": 23/32,
        "x": -79/32, "beta": 53/32,
    }),
    ("a=√e/π, p=π/4, q=π/4-1/16, |x|=√6, β=√e", {
        "a": np.sqrt(np.e) / np.pi, "p": np.pi/4, "q": np.pi/4 - 1/16,
        "x": -np.sqrt(6), "beta": np.sqrt(np.e),
    }),
]


def main():
    print("=" * 100)
    print("  TEST DES VALEURS ANALYTIQUES sur le spectre (avec ρ_0, k_b recalibrés)")
    print("=" * 100)
    print(f"\n  {'hypothèse':<55} {'RMS log':>9} {'★+★★':>6} {'<20%':>6} "
          f"{'ρ_0':>8} {'k_b':>10}")
    print("  " + "-" * 105)

    results = []
    for name, params in CANDIDATES:
        rms, flags, derived = evaluate(params, name)
        if rms is None:
            print(f"  {name:<55} ÉCHEC (no solution)")
            continue
        rho0, kb = derived
        n5 = flags["★★"] + flags["★"]
        n20 = n5 + flags["✓"]
        results.append((name, params, rms, n5, n20, rho0, kb))
        print(f"  {name:<55} {rms:>9.4f} {n5:>6}/26 {n20:>6}/26 "
              f"{rho0:>8.3f} {kb:>10.6f}")

    print(f"\n  ─── INTERPRÉTATION ───")
    print(f"  Si une hypothèse analytique donne un RMS ≤ 1.2× l'optimum,")
    print(f"  c'est une candidature sérieuse pour l'origine analytique des paramètres.")

    opt_rms = results[0][2]
    print(f"\n  RMS optimum numérique : {opt_rms:.4f}")
    print(f"  Hypothèses dans facteur 1.2× :")
    for name, params, rms, n5, n20, rho0, kb in results[1:]:
        if rms <= opt_rms * 1.2:
            print(f"    ★ {name} : RMS {rms:.4f}")
        elif rms <= opt_rms * 2:
            print(f"    ✓ {name} : RMS {rms:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
