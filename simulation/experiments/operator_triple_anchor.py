r"""
OPÉRATEUR Ωₜₛ TRIPLE-ANCRÉ — e⁻, proton, neutron exactement à 0%.

L'électron est déjà exact par calibration de m_c et m_l.
Pour ancrer aussi proton ET neutron :
   - Proton (3_1 + 7_2 + 7_2) : on contraint le paramètre ρ_0 pour
     que m_pred(p) = m_p_exp
   - Neutron (3_1 + 7_2 + 7_2_b) : on introduit un coefficient k_b
     appliqué à UNE des deux boucles 7_2 (l'autre reste libre),
     déterminé par m_n_exp.

   m_p = (m_e^a + 2·m_w(7_2)^a)^(1/a) · ρ_pair · G · Φ_n
   m_n = (m_e^a + m_w(7_2)^a + (k_b·m_w(7_2))^a)^(1/a) · ρ_pair' · G' · Φ_n'

Le k_b dépend du rapport m_n/m_p — exactement comme l'opérateur Hölder
double-ancré du commit précédent (holder_kb_from_np.py).

On optimise les 5 autres paramètres (a, p, q, x, β) sur le spectre
des 74 autres particules, avec ρ_0 et k_b ENTIÈREMENT DÉTERMINÉS par
les contraintes proton + neutron.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, brentq
from collections import Counter
from itertools import combinations_with_replacement

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


def operator_with_ks(comps, a, p, q, x, rho0, beta, ks_dict=None):
    """Opérateur Ωₜₛ' avec optionnellement coefficients par knot (ks_dict).

    ks_dict : {K: facteur} appliqué multiplicativement au poids du nœud K.
              Si un nœud apparaît plusieurs fois, ks s'applique à TOUTES
              ses occurrences. Pour appliquer à UNE seule occurrence,
              utiliser une clé "K_b" qui ajoute une 2e variante.
    """
    n = len(comps)
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** p)
    # Appliquer les facteurs ks
    if ks_dict:
        # Compteur d'occurrences par nœud
        c = Counter(comps)
        idx_per_knot = {K: 0 for K in c}
        new_ws = []
        for i, K in enumerate(comps):
            occ = idx_per_knot[K]
            key_specific = f"{K}_{occ}"
            if key_specific in ks_dict:
                factor = ks_dict[key_specific]
            elif K in ks_dict:
                factor = ks_dict[K]
            else:
                factor = 1.0
            new_ws.append(ws[i] * factor)
            idx_per_knot[K] = occ + 1
        ws = new_ws
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


def m_pred_e(a, p, q, x, rho0, beta):
    return operator_with_ks(["3_1"], a, p, q, x, rho0, beta)


def m_pred_p(a, p, q, x, rho0, beta):
    return operator_with_ks(["3_1", "7_2", "7_2"], a, p, q, x, rho0, beta)


def m_pred_n(a, p, q, x, rho0, beta, k_b):
    """Pour le neutron : la 2e occurrence de 7_2 est bloquée (k_b)."""
    return operator_with_ks(["3_1", "7_2", "7_2"], a, p, q, x, rho0, beta,
                            ks_dict={"7_2_1": k_b})


def solve_rho0_for_proton(a, p, q, x, beta, target=M_P):
    """Trouve ρ_0 tel que m_pred(proton) = m_p_exp."""
    def f(log_rho0):
        return m_pred_p(a, p, q, x, np.exp(log_rho0), beta) - target
    try:
        return float(np.exp(brentq(f, -10, 10, xtol=1e-9)))
    except Exception:
        return None


def solve_kb_for_neutron(a, p, q, x, rho0, beta, target=M_N):
    """Trouve k_b tel que m_pred(neutron) = m_n_exp."""
    def f(log_kb):
        return m_pred_n(a, p, q, x, rho0, beta, np.exp(log_kb)) - target
    try:
        return float(np.exp(brentq(f, -5, 5, xtol=1e-9)))
    except Exception:
        return None


# Spectre des 75 autres particules (e/p/n exclus, ancrés)
SPECTRUM_OTHER = [
    ("μ⁻",       ["3_1", "6_1"],                105.66),
    ("τ⁻",       ["3_1", "5_1", "7_1"],         1776.86),
    ("π⁰",       ["8_1"],                       134.98),
    ("π±",       ["8_1"],                       139.57),
    ("K±",       ["4_1", "4_1", "5_2"],         493.68),
    ("K⁰_S",     ["3_1", "6_1", "6_1"],         497.611),
    ("η",        ["5_2", "5_2", "6_1"],         547.86),
    ("η'",       ["5_1", "5_2", "5_2"],         957.78),
    ("ρ/ω",      ["5_2", "7_7"],                775.0),
    ("φ",        ["3_1", "7_1", "7_1"],         1019.46),
    ("J/ψ",      ["5_2", "6_1", "6_2"],         3096.9),
    ("ψ(2S)",    ["5_1", "7_7", "7_7"],         3686.10),
    ("η_c",      ["3_1", "7_1", "9_1"],         2983.9),
    ("Υ",        ["3_1", "7_4", "9_1"],         9460.3),
    ("D⁰",       ["6_2", "7_7"],                1864.84),
    ("D±",       ["7_7", "9_1"],                1869.66),
    ("D_s",      ["6_3", "9_1"],                1968.34),
    ("B",        ["4_1", "10_1"],               5279.4),
    ("B⁰",       ["7_6", "7_7"],                5279.65),
    ("B_s",      ["7_6", "7_7"],                5366.92),
    ("B_c",      ["5_1", "10_1"],               6274.9),
    ("Λ",        ["5_1", "7_7"],                1115.7),
    ("Σ⁺",       ["6_2", "7_2"],                1189.4),
    ("Σ⁰",       ["5_1", "6_3"],                1192.642),
    ("Σ⁻",       ["5_1", "6_3"],                1197.449),
    ("Ξ⁰",       ["5_1", "5_1", "5_2"],         1314.86),
    ("Ξ⁻",       ["4_1", "9_1", "9_1"],         1321.71),
    ("Δ",        ["4_1", "7_2", "7_2"],         1232.0),
    ("Σ*",       ["3_1", "5_2", "7_1"],         1383.7),
    ("Ξ*",       ["3_1", "6_1", "7_2"],         1531.8),
    ("Ω⁻",       ["3_1", "6_1", "7_1"],         1672.0),
    ("Λc",       ["6_3", "8_1"],                2286.46),
    ("Σc",       ["5_2", "5_2", "8_1"],         2453.97),
    ("Ξc",       ["5_1", "5_2", "7_2"],         2467.94),
    ("Ωc",       ["6_1", "6_1", "8_1"],         2695.2),
    ("Λb",       ["5_1", "6_1", "6_3"],         5619.6),
    ("Σb",       ["4_1", "7_7", "9_1"],         5811.0),
    ("Ξb",       ["6_1", "6_3", "6_3"],         5797.0),
    ("W±",       ["4_1", "7_3", "8_3"],         80369.0),
    ("Z⁰",       ["7_4", "9_1", "8_3"],         91188.0),
    ("H",        ["7_7", "8_2", "8_3"],         125250.0),
    ("t",        ["3_1", "7_5", "8_3"],         173100.0),
]


def loss(params):
    a, p, q, x, beta = params
    rho0 = solve_rho0_for_proton(a, p, q, x, beta)
    if rho0 is None or rho0 <= 0:
        return 1e6
    # Évaluer le reste du spectre
    res = []
    for nm, comps, me in SPECTRUM_OTHER:
        try:
            m_pred = operator_with_ks(comps, a, p, q, x, rho0, beta)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e6
        if m_pred > 0 and np.isfinite(m_pred):
            res.append(np.log(m_pred / me))
        else:
            res.append(20)
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def main():
    print("\n" + "=" * 100)
    print("  OPÉRATEUR Ωₜₛ TRIPLE-ANCRÉ — e⁻, p, n exactement à 0%")
    print("=" * 100)
    print(f"\n  Ancrages exacts :")
    print(f"    e⁻ : m_e = {M_E} MeV (par calibration M_C, M_L)")
    print(f"    p  : m_p = {M_P} MeV (ρ_0 ajusté)")
    print(f"    n  : m_n = {M_N} MeV (k_b ajusté pour 7_2 bloqué)")

    print(f"\n  Optimisation des 5 paramètres (a, p, q, x, β) sur {len(SPECTRUM_OTHER)} particules…")
    best = None
    for a0 in [0.4, 0.5]:
        for p0 in [0.7, 0.8]:
            for q0 in [0.7, 0.8]:
                for x0 in [-2.5, -1.5]:
                    for b0 in [1.3, 1.7]:
                        x_init = [a0, p0, q0, x0, b0]
                        res = minimize(loss, x_init, method="Nelder-Mead",
                                       options={"maxiter": 5000, "xatol": 1e-7})
                        if best is None or res.fun < best.fun:
                            best = res

    a, p, q, x, beta = best.x
    rho0 = solve_rho0_for_proton(a, p, q, x, beta)
    k_b = solve_kb_for_neutron(a, p, q, x, rho0, beta)

    print(f"\n  ─── PARAMÈTRES FINAUX ───")
    print(f"  a     = {a:.5f}  (Hölder)")
    print(f"  p     = {p:.5f}  (topo)")
    print(f"  q     = {q:.5f}  (géom)")
    print(f"  x     = {x:.5f}  (écrantage)")
    print(f"  β     = {beta:.5f}  (puissance paire)")
    print(f"  ρ_0   = {rho0:.5f}  (calculé pour proton exact)")
    print(f"  k_b   = {k_b:.6f}  (calculé pour neutron exact)")

    # Vérifications ancrages
    print(f"\n  ─── VÉRIFICATION ANCRAGES ───")
    me_pred = m_pred_e(a, p, q, x, rho0, beta)
    mp_pred = m_pred_p(a, p, q, x, rho0, beta)
    mn_pred = m_pred_n(a, p, q, x, rho0, beta, k_b)
    print(f"  e⁻  : prédit {me_pred:.8f}  exp {M_E:.8f}  Δ = {(me_pred-M_E)/M_E*100:+.6f}%")
    print(f"  p   : prédit {mp_pred:.6f}    exp {M_P:.6f}    Δ = {(mp_pred-M_P)/M_P*100:+.6f}%")
    print(f"  n   : prédit {mn_pred:.6f}    exp {M_N:.6f}    Δ = {(mn_pred-M_N)/M_N*100:+.6f}%")

    # Évaluer le reste
    print(f"\n  ─── PRÉDICTIONS DU RESTE DU SPECTRE ───")
    print(f"  {'particule':>10} {'config':>22} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 80)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    res_log = []
    for nm, comps, me in SPECTRUM_OTHER:
        m_pred = operator_with_ks(comps, a, p, q, x, rho0, beta)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        res_log.append(np.log(m_pred / me))
        print(f"  {nm:>10} {'+'.join(comps):>22} {me:>11.2f} {m_pred:>11.2f} "
              f"{err:>+8.2f} {flag:>5}")

    rms = float(np.sqrt(np.mean(np.array(res_log) ** 2)))
    total = len(SPECTRUM_OTHER) + 3   # +e,p,n
    n5 = flags["★★"] + flags["★"] + 3  # +3 ancrés
    n20 = n5 + flags["✓"]
    print(f"\n  ═══ BILAN GLOBAL ({total} particules) ═══")
    print(f"    e, p, n ancrés EXACTEMENT à 0%")
    print(f"    ★★ (autres) : {flags['★★']}")
    print(f"    ★  (autres) : {flags['★']}")
    print(f"    ✓  (autres) : {flags['✓']}")
    print(f"    ordinaire   : {flags[' ']}")
    print(f"    outlier     : {flags['?']}")
    print(f"  ")
    print(f"    Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"    Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")
    print(f"    RMS log (hors ancrés) = {rms:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
