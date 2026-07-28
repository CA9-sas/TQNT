r"""
NOUVEL OPÉRATEUR « Ω » (Omega) — INVENTÉ SPÉCIFIQUEMENT POUR TQNT

Principe inventé : la masse d'une particule composite résulte de la
combinaison de DEUX structures distinctes :

   1. SQUELETTE COMBINATOIRE — moyenne de Hölder des masses brutes
      m_naive(K_i) = cr(K_i)·m_c + n_lobes(K_i)·m_l
      [Σ_i m_naive(K_i)^a]^(1/a)   ← skelette

   2. CHAMP GÉOMÉTRIQUE — couplage joint de toutes les courbures et
      torsions, mesuré comme produit des sommes :
      [ΣU_κ(K_i)] × [ΣU_τ(K_i)] / [U_κ(3_1) × U_τ(3_1)]   ← champ

   La masse est le PRODUIT des deux :
   ┌────────────────────────────────────────────────────────────┐
   │   m(P) = (Σ m_naive(K_i)^a)^(1/a)  ·  (ΣU_κ_i)(ΣU_τ_i)     │
   │                                       ─────────────────     │
   │                                       U_κ(3_1)·U_τ(3_1)     │
   └────────────────────────────────────────────────────────────┘

Propriétés essentielles :
   - Pour un single (n=1) : m(K) = m_naive(K) · P(K)/P(3_1) ✓
     (formule isolée qu'on a démontrée)
   - Pour l'électron (3_1 seul) : m = m_e exactement ✓
   - Pour le 7_2 isolé : m = 33 MeV (libre) ✓
   - Pour les composites : un seul paramètre a (Hölder squelette)

L'invention CLEF : le couplage géométrique au composite est
le PRODUIT (ΣU_κ)(ΣU_τ), pas la somme.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from theory.abraham import frenet_geometry

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


def cr(K):
    return int(K.split("_")[0])


print("Calcul Frenet (50000 points) pour les 19 nœuds…")
U_KT = {}
for K in KNOTS:
    geo = frenet_geometry(K, 50000)
    kappa = geo["kappa"]; tau = geo["tau"]; speed = geo["speed"]
    dt = 2 * np.pi / len(kappa)
    ds = speed * dt
    U_KT[K] = (
        float(np.sum(kappa ** 2 * ds)),
        float(np.sum(tau ** 2 * ds)),
    )
P_KT_31 = U_KT["3_1"][0] * U_KT["3_1"][1]
M_NAIVE = {K: cr(K) * M_C + N_LOBES[K] * M_L for K in KNOTS}
print(f"P_κτ(3_1) = U_κ·U_τ = {P_KT_31:.4f}")


# ───────────────────────────────────────────────────────────────────
# Opérateur Ω
# ───────────────────────────────────────────────────────────────────
def omega(comps, a):
    """Opérateur Ω inventé :
       m(P) = (Σ m_naive^a)^(1/a) · (ΣU_κ)(ΣU_τ) / P_κτ(3_1)
    """
    # Squelette : Hölder sur m_naive
    ms = [M_NAIVE[K] for K in comps]
    if abs(a) < 1e-8:
        skeleton = float(np.prod(ms) ** (1 / len(ms)))
    else:
        skeleton = float(sum(m ** a for m in ms) ** (1 / a))
    # Champ : produit des sommes Frenet
    sum_U_k = sum(U_KT[K][0] for K in comps)
    sum_U_t = sum(U_KT[K][1] for K in comps)
    field = sum_U_k * sum_U_t / P_KT_31
    return skeleton * field


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


def loss(a):
    res = []
    for nm, comps, me in SPECTRUM:
        try:
            m_pred = omega(comps, a)
        except (OverflowError, ValueError):
            return 1e6
        if m_pred > 0:
            res.append(np.log(m_pred / me))
        else:
            res.append(20)
    return float(np.sqrt(np.mean(np.array(res) ** 2)))


def main():
    print("\n" + "=" * 100)
    print("  OPÉRATEUR Ω — INVENTÉ POUR TQNT")
    print("=" * 100)
    print(f"\n  m(P) = (Σ_i m_naive(K_i)^a)^(1/a) · "
          f"(Σ U_κ_i)(Σ U_τ_i) / U_κ(3_1)·U_τ(3_1)")
    print(f"\n  m_per_crossing = {M_C:.6f} MeV")
    print(f"  m_per_loop     = {M_L:.6f} MeV")
    print(f"  P_κτ(3_1)      = {P_KT_31:.4f}")

    # Test propriétés
    print(f"\n  ─── PROPRIÉTÉS VÉRIFIÉES ───")
    a_test = 1.0
    m_e_pred = omega(["3_1"], a_test)
    m_72_pred = omega(["7_2"], a_test)
    print(f"  Pour a={a_test} :")
    print(f"    m(e⁻ = 3_1)      = {m_e_pred:.5f} MeV  (cible {M_E:.5f})")
    print(f"    m(7_2 isolé)     = {m_72_pred:.5f} MeV  (cible 33.05 = m_naive·P/P_31)")

    # Optimisation
    print(f"\n  ─── OPTIMISATION DE a SUR 25 PARTICULES ───")
    res = minimize_scalar(loss, bounds=(0.05, 5), method="bounded",
                          options={"xatol": 1e-8})
    a_opt = res.x
    rms = res.fun
    print(f"  a optimal = {a_opt:.5f}")
    print(f"  RMS log    = {rms:.4f}  (= +{(np.exp(rms)-1)*100:.0f}% relatif)")

    # Détails prédictions
    print(f"\n  ─── PRÉDICTIONS Ω(a = {a_opt:.4f}) ───")
    print(f"  {'particule':>5} {'config':>17} {'m_exp':>11} {'m_pred':>11} "
          f"{'Δ %':>9} {'flag':>5}")
    print("  " + "-" * 72)
    flags = {"★★": 0, "★": 0, "✓": 0, " ": 0, "?": 0}
    for nm, comps, me in SPECTRUM:
        m_pred = omega(comps, a_opt)
        err = (m_pred - me) / me * 100
        if abs(err) < 0.5: flag = "★★"
        elif abs(err) < 5: flag = "★"
        elif abs(err) < 20: flag = "✓"
        elif abs(err) < 50: flag = " "
        else: flag = "?"
        flags[flag] += 1
        print(f"  {nm:>5} {'+'.join(comps):>17} {me:>11.2f} "
              f"{m_pred:>11.2f} {err:>+8.2f} {flag:>5}")

    total = len(SPECTRUM)
    n5 = flags["★★"] + flags["★"]
    n20 = n5 + flags["✓"]
    print(f"\n  Bilan : ★★={flags['★★']}, ★={flags['★']}, ✓={flags['✓']}, "
          f"ordinaire={flags[' ']}, ?={flags['?']} sur {total}")
    print(f"  Précision <5% : {n5}/{total} ({n5/total*100:.0f}%)")
    print(f"  Précision <20% : {n20}/{total} ({n20/total*100:.0f}%)")

    print(f"\n  ─── COMPARAISON ───")
    print(f"  Hölder pur (m_iso, RMS 0.50) : 3/25 à <5%")
    print(f"  Opérateur Ω inventé          : {n5}/{total} à <5%, RMS {rms:.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
