r"""
OPÉRATEUR HYBRIDE — squelette TOPOLOGIQUE (cr, n_lobes, b, det)
                  + géométrie LIMITÉE au discriminant nécessaire (U_κ, U_τ
                    seulement pour résoudre la dégénérescence 8_3 et alike)

Structure inventée :

   m(P) = [Σ_i (m_naive(K_i) · T(K_i)^p)^a]^(1/a)  ·  G_discriminant(P)^q

où :
   m_naive(K) = cr(K) · m_c + n_lobes(K) · m_l    (squelette combinatoire)
   T(K) = b(K) · det(K) / (b(3_1) · det(3_1))     (amplification topologique)
   G_discriminant(P) = Σ_i (U_κ_i · U_τ_i) / P_31  (discriminant géométrique
                                                    au niveau composite)

L'IDÉE : la topologie (T) amplifie individuellement chaque constituant
selon sa complexité (braid × det), et la géométrie (G) entre seulement
au niveau du COMPOSITE comme un seul facteur global qui capture la
spécificité du 8_3.

Pour singles : G est juste P(K)/P_31 (cohérent avec formule isolée).
Pour composites : G est la SOMME des P_i (couplage faible inter-boucles).

Paramètres libres : a (Hölder), p (puissance topo), q (puissance géom).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

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


print("Calcul Frenet (50000 points) pour les 19 nœuds…")
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
    p = jones_polynomial(K)
    TOPO[K] = {
        "cr":  int(K.split("_")[0]),
        "n_l": N_LOBES[K],
        "b":   braid_index(K),
        "det": jones_determinant(p),
    }

P_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"] * TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"] * M_C + TOPO[K]["n_l"] * M_L for K in KNOTS}
print(f"P_31 (géom) = {P_31:.4f}")
print(f"T_31 (topo) = {T_31}")


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


def hybrid(comps, params):
    """Opérateur hybride : squelette topo × discriminant géom composite.

       m(P) = [Σ_i (m_naive_i · T_i^p)^a]^(1/a) · G_composite^q

       T_i = b_i · det_i / (b_31 · det_31)
       G_composite = Σ_i U_κ_i · U_τ_i / P_31
    """
    a, p, q = params
    ws = []
    for K in comps:
        T = TOPO[K]["b"] * TOPO[K]["det"] / T_31
        ws.append(M_NAIVE[K] * T ** p)
    if abs(a) < 1e-8:
        skeleton = float(np.prod(ws) ** (1 / len(ws)))
    else:
        skeleton = float(sum(w ** a for w in ws) ** (1 / a))
    G_comp = sum(U_KT[K][0] * U_KT[K][1] for K in comps) / P_31
    return skeleton * G_comp ** q


def loss(params):
    res = []
    for nm, comps, me in SPECTRUM:
        try:
            m_pred = hybrid(comps, params)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e6
        if m_pred > 0 and np.isfinite(m_pred):
            res.append(np.log(m_pred / me))
        else:
            res.append(20)
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def main():
    print("\n" + "=" * 100)
    print("  OPÉRATEUR HYBRIDE Ωₜ — squelette topo + discriminant géom")
    print("=" * 100)
    print(f"\n  m(P) = [Σ_i (m_naive_i · T(K_i)^p)^a]^(1/a) · G_composite^q")
    print(f"  T(K) = b · det / (b_31 · det_31)")
    print(f"  G_comp = Σ (U_κ · U_τ) / P_31")

    # Optimisation
    print(f"\n  Optimisation 3 paramètres (a, p, q)…")
    best = None
    for a0 in [0.3, 0.5, 1.0]:
        for p0 in [0.5, 1.0, 1.5]:
            for q0 in [0.5, 1.0, 1.5]:
                res = minimize(loss, [a0, p0, q0], method="Nelder-Mead",
                               options={"maxiter": 10000, "xatol": 1e-8})
                if best is None or res.fun < best.fun:
                    best = res

    a_opt, p_opt, q_opt = best.x
    rms = best.fun
    print(f"\n  Paramètres optimaux :")
    print(f"    a = {a_opt:.5f}  (Hölder squelette)")
    print(f"    p = {p_opt:.5f}  (puissance topo)")
    print(f"    q = {q_opt:.5f}  (puissance géom)")
    print(f"  RMS log = {rms:.4f}  (= +{(np.exp(rms)-1)*100:.0f}% relatif)")

    # Évaluation
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    rows = []
    for nm, comps, me in SPECTRUM:
        m_pred = hybrid(comps, [a_opt, p_opt, q_opt])
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        rows.append((nm, comps, me, m_pred, err, flag))

    print(f"\n  Prédictions :")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 67)
    for nm, comps, me, m_pred, err, flag in rows:
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")

    total = len(SPECTRUM)
    n5 = flags["★★"] + flags["★"]
    n20 = n5 + flags["✓"]
    print(f"\n  Bilan : ★★={flags['★★']}, ★={flags['★']}, ✓={flags['✓']}, "
          f"ordinaire={flags[' ']}, ?={flags['?']} sur {total}")
    print(f"  Précision <5%  : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"  Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")

    # Comparaison
    print(f"\n  ─── COMPARAISON ───")
    print(f"  Ω géom pur        : RMS 0.44, 2/25 à <5%, 9/25 à <20%")
    print(f"  H_W pondéré       : RMS 0.44, 3/25 à <5%, 8/25 à <20%")
    print(f"  Topologique pur   : RMS 0.98, 2/25, 3/25")
    print(f"  Topo-amplifié     : RMS 0.95, 2/25, 5/25")
    print(f"  Ωₜ HYBRIDE (ici)  : RMS {rms:.2f}, {n5}/25, {n20}/25")
    print("=" * 100)


if __name__ == "__main__":
    main()
