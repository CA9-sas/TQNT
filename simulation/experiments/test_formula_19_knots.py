r"""
TEST DE LA FORMULE m(K) = m_naive(K) · k_state · P(K)/P(3_1)
sur les 19 nœuds du catalogue + le spectre des particules.

   m_naive(K) = cr(K) · m_per_crossing + n_lobes(K) · m_per_loop
   m_per_crossing = 0.147883 MeV  (du 3_1 électron, partition Abraham)
   m_per_loop     = 0.022450 MeV
   k_state = 2 si knot dans un composite, 1 si isolé
   P(K) = U_κ(K) · U_τ(K)
   P(3_1) = 9.348

Pour chaque nœud :
  1. Calcul des intégrales Frenet (50 000 points)
  2. Application de la formule
  3. Sortie d'un poids m(K) physique

Pour chaque particule du spectre :
  1. Application de l'opérateur de Hölder avec ces nouveaux poids
  2. Optimisation du seul paramètre a
  3. Comparaison à m_exp
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from theory.abraham import frenet_geometry

# Constantes électron
M_E = 0.51099895
F_KAPPA = 0.8682   # de la décomposition du 3_1
M_C = M_E * F_KAPPA / 3        # par croisement = 0.147883 MeV
M_L = M_E * (1 - F_KAPPA) / 3  # par lobe       = 0.022450 MeV
K_S = 2.0                       # facteur composite

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]

# Nombre de lobes pour chaque nœud (estimation par type)
# (2,n) torus : n lobes ; twist : 2 lobes ; figure-8 : 2 ;
# amphichiraux 6_3, 8_3 : 3 lobes ; autres : 2 ou 3 selon diagramme
N_LOBES = {
    "3_1": 3,   "4_1": 2,   "5_1": 5,   "5_2": 2,
    "6_1": 2,   "6_2": 3,   "6_3": 3,
    "7_1": 7,   "7_2": 2,   "7_3": 3,   "7_4": 2,
    "7_5": 3,   "7_6": 2,   "7_7": 3,
    "8_1": 2,   "8_2": 3,   "8_3": 3,
    "9_1": 9,   "10_1": 2,
}


def cr(K):
    return int(K.split("_")[0])


def compute_U_kt(K, n_points=50000):
    geo = frenet_geometry(K, n_points)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    return (float(np.sum(kappa ** 2 * ds)),
            float(np.sum(tau ** 2 * ds)))


def main():
    print("=" * 100)
    print("  TEST FORMULE m(K) = m_naive · k_state · P(K)/P(3_1)")
    print("=" * 100)
    print(f"\n  m_per_crossing = {M_C:.6f} MeV")
    print(f"  m_per_loop     = {M_L:.6f} MeV")
    print(f"  k_state (composite) = {K_S}")

    # Calcul U_κ × U_τ pour les 19 nœuds
    print(f"\n  Calcul Frenet sur les 19 nœuds…")
    U_kt = {}
    for K in KNOTS:
        U_k, U_t = compute_U_kt(K)
        U_kt[K] = (U_k, U_t)
        # progrès silencieux

    P_31 = U_kt["3_1"][0] * U_kt["3_1"][1]
    print(f"  P(3_1) = U_κ × U_τ = {P_31:.4f}")

    # ─── Tableau des poids physiques pour les 19 nœuds ─────────────────
    print(f"\n  ─── POIDS m(K) selon la nouvelle formule ───")
    print(f"  Formule isolé    : m(K) = m_naive · P(K)/P(3_1)")
    print(f"  Formule composite : m(K) = m_naive · k_s · P(K)/P(3_1)")
    print()
    print(f"  {'K':>5} {'cr':>3} {'lobes':>5} {'U_κ':>9} {'U_τ':>9} "
          f"{'P/P_31':>9} {'m_naive':>9} {'m_isolé':>11} {'m_compo':>11}")
    print("  " + "-" * 88)
    weights_iso = {}
    weights_comp = {}
    for K in KNOTS:
        U_k, U_t = U_kt[K]
        P_K = U_k * U_t
        ratio = P_K / P_31
        nl = N_LOBES[K]
        m_n = cr(K) * M_C + nl * M_L
        m_iso = m_n * ratio
        m_comp = K_S * m_iso
        weights_iso[K] = m_iso
        weights_comp[K] = m_comp
        anchor = " ★" if K == "3_1" else ""
        print(f"  {K:>5} {cr(K):>3} {nl:>5} {U_k:>9.3f} {U_t:>9.3f} "
              f"{ratio:>9.3f} {m_n:>9.4f} {m_iso:>11.4f} "
              f"{m_comp:>11.4f}{anchor}")

    # Comparaison avec poids Hölder de référence
    holder = {
        "3_1": 0.5110, "4_1": 15.367, "5_1": 70.165, "5_2": 18.970,
        "6_1": 48.966, "6_2": 187.354, "6_3": 361.719,
        "7_1": 199.934, "7_2": 66.083, "7_3": 58.063, "7_4": 484.513,
        "7_5": 0.0, "7_6": 669.395, "7_7": 199.128,
        "8_1": 137.254, "8_2": 1297.067, "8_3": 40439.145,
        "9_1": 752.635, "10_1": 2837.028,
    }

    print(f"\n  ─── COMPARAISON avec poids Hölder (référence) ───")
    print(f"  {'K':>5} {'m_isolé':>11} {'m_compo':>11} {'m_Hölder':>11} "
          f"{'iso/Hölder':>11} {'compo/Hölder':>13}")
    print("  " + "-" * 75)
    for K in KNOTS:
        miso = weights_iso[K]
        mcom = weights_comp[K]
        mH = holder[K]
        if mH > 0:
            r1 = miso / mH; r2 = mcom / mH
            f1 = "★" if 0.5 < r1 < 2 else ""
            f2 = "★" if 0.5 < r2 < 2 else ""
            print(f"  {K:>5} {miso:>11.3f} {mcom:>11.3f} {mH:>11.3f} "
                  f"{r1:>9.3f}× {f1:>2} {r2:>11.3f}× {f2}")
        else:
            print(f"  {K:>5} {miso:>11.3f} {mcom:>11.3f} {mH:>11.3f}  (jamais utilisé)")

    # ─── Application au spectre via Hölder ──────────────────────────────
    print(f"\n  ─── TEST SUR SPECTRE : Hölder avec poids m_compo ───")
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

    # Pour singles : poids isolé. Pour composites : poids composite.
    def holder_mass(comps, a):
        if len(comps) == 1:
            return weights_iso[comps[0]]
        if abs(a) < 1e-8:
            return float(np.prod([weights_comp[K] for K in comps]) ** (1/len(comps)))
        return float((sum(weights_comp[K] ** a for K in comps)) ** (1 / a))

    def loss(a):
        res = []
        for nm, comps, me in SPECTRUM:
            m_pred = holder_mass(comps, a)
            if m_pred > 0:
                res.append(np.log(m_pred / me))
        return float(np.sqrt(np.mean(np.array(res) ** 2)))

    r = minimize_scalar(loss, bounds=(0.05, 5), method="bounded",
                        options={"xatol": 1e-7})
    a_opt = r.x; rms = r.fun
    print(f"\n  a optimal = {a_opt:.5f},  RMS log = {rms:.4f}")

    print(f"\n  {'particule':>5} {'config':>16} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 65)
    flags = {"★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps, me in SPECTRUM:
        m_pred = holder_mass(comps, a_opt)
        err = (m_pred - me) / me * 100
        if abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        print(f"  {nm:>5} {'+'.join(comps):>16} {me:>11.2f} {m_pred:>11.2f} "
              f"{err:>+7.2f} {flag:>5}")

    print(f"\n  Bilan : ★={flags['★']}, ✓={flags['✓']}, "
          f"ordinaire={flags[' ']}, ?={flags['?']} sur {len(SPECTRUM)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
