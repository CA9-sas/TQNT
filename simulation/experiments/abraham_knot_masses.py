r"""
MASSE D'UN NŒUD VIA LA FORMULE D'ABRAHAM (calibrée sur l'électron)

Formule d'Abraham classique : la masse d'une particule chargée est
l'énergie EM de son champ. Pour un fil chargé refermé en boucle nouée :

   m(K) = η · U_EM(K)
   U_EM(K) = ∮_K (κ² + τ²) ds   (intégrale d'Abraham-Frenet)

η est calibré sur l'électron en posant m(3_1) = m_e exactement :

   η = m_e / U_EM(3_1)

Avec U_EM(3_1) tabulé dans theory.abraham.REFERENCE_N50K (50 000 points),
on obtient η et on déduit m(K) pour les 19 nœuds catalogue.

On compare ensuite ces poids "Abraham" aux poids du fit Hölder.
"""
from __future__ import annotations
import numpy as np

from theory.abraham import REFERENCE_N50K

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
         "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7",
         "8_1", "8_2", "8_3", "9_1", "10_1"]

M_E = 0.51099895  # MeV (électron)


def U_EM(K):
    """U_EM(K) = ∮(κ²+τ²)ds, valeur catalogue 50 000 points."""
    return float(REFERENCE_N50K[K]["sqrt_U"]) ** 2


def L(K):
    return float(REFERENCE_N50K[K]["L"])


def int_kappa(K):
    return float(REFERENCE_N50K[K]["int_kappa"])


def main():
    print("=" * 90)
    print("  MASSE DES 19 NŒUDS PAR LA FORMULE D'ABRAHAM (calibrée sur e⁻)")
    print("=" * 90)
    print(f"\n  Formule : m(K) = η · U_EM(K), U_EM = ∮(κ²+τ²)ds")
    print(f"  Calibration sur l'électron : m(3_1) = m_e = {M_E} MeV exactement")

    # Calibration
    U_31 = U_EM("3_1")
    eta = M_E / U_31
    print(f"\n  U_EM(3_1) = {U_31:.4f}  (catalogue Abraham 50 000 points)")
    print(f"  η         = m_e / U_EM(3_1) = {M_E} / {U_31:.4f} "
          f"= {eta:.6e} MeV / unité")

    # Tableau des masses Abraham
    print(f"\n  ─── TABLEAU DES MASSES ABRAHAM ───")
    print(f"  {'K':>5} {'L':>7} {'∫|κ|ds':>9} {'U_EM':>10} "
          f"{'m_Abraham (MeV)':>17}")
    print("  " + "-" * 55)
    masses = {}
    for K in KNOTS:
        u = U_EM(K)
        L_K = L(K)
        ik = int_kappa(K)
        m_K = eta * u
        masses[K] = m_K
        anchor = " ★" if K == "3_1" else ""
        print(f"  {K:>5} {L_K:>7.2f} {ik:>9.2f} {u:>10.2f} "
              f"{m_K:>16.4f}{anchor}")

    # Stats
    arr = np.array([masses[K] for K in KNOTS])
    print(f"\n  Statistiques sur les 19 masses :")
    print(f"    min    = {arr.min():>10.3f} MeV  ({KNOTS[arr.argmin()]})")
    print(f"    max    = {arr.max():>10.3f} MeV  ({KNOTS[arr.argmax()]})")
    print(f"    médian = {float(np.median(arr)):>10.3f} MeV")
    print(f"    range  = ×{arr.max()/arr.min():.1f}")

    # Comparaison avec les poids Hölder (a=0.244, k_s=2, k_b≈1)
    holder_weights = {
        "3_1": 0.511, "4_1": 5.37, "5_2": 6.02, "6_1": 26.85, "5_1": 28.29,
        "7_3": 58.06, "7_2": 25.54, "7_1": 86.09, "6_2": 77.22, "6_3": 122.64,
        "7_7": 153.10, "7_4": 316.37, "7_6": 488.46, "9_1": 329.32,
        "8_1": 137.25, "8_2": 760.75, "10_1": 2215.96, "7_5": 11882.16,
        "8_3": 40439.15,
    }
    print(f"\n  ─── COMPARAISON Abraham vs Hölder (poids inférés du spectre) ───")
    print(f"  {'K':>5} {'m_Abraham':>11} {'m_Hölder':>11} {'ratio H/A':>10}")
    print("  " + "-" * 42)
    for K in KNOTS:
        mA = masses[K]
        mH = holder_weights[K]
        ratio = mH / mA if mA > 0 else float("inf")
        print(f"  {K:>5} {mA:>11.4f} {mH:>11.4f} {ratio:>9.2f}×")

    # Tester la formule de Hölder avec poids Abraham
    print(f"\n  ─── TEST : Hölder avec POIDS ABRAHAM sur 7 particules canoniques ───")
    SPECTRUM = [
        ("e⁻",  ["3_1"],               0.511),
        ("π⁰",  ["8_1"],               134.98),
        ("p",   ["3_1", "7_2", "7_2"], 938.272),
        ("n",   ["3_1", "7_2", "7_2"], 939.565),
        ("W±",  ["4_1", "7_3", "8_3"], 80377.0),
        ("Z⁰",  ["5_1", "7_2", "8_3"], 91188.0),
        ("H",   ["6_1", "7_4", "8_3"], 125250.0),
        ("t",   ["7_4", "7_6", "8_3"], 173100.0),
    ]

    def holder(comps, w, a):
        if abs(a) < 1e-8:
            return float(np.prod([w[K] for K in comps]) ** (1 / len(comps)))
        s = sum(w[K] ** a for K in comps)
        return float(s ** (1 / a))

    # Optimiser a pour Abraham
    from scipy.optimize import minimize_scalar
    def loss(a):
        residuals = []
        for nm, comps, me in SPECTRUM:
            try:
                m_pred = holder(comps, masses, a)
                residuals.append(np.log(m_pred / me))
            except:
                return 1e6
        return float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    res = minimize_scalar(loss, bounds=(0.05, 5.0), method="bounded",
                          options={"xatol": 1e-7})
    a_opt = res.x
    rms_opt = res.fun

    print(f"\n  a optimal (Hölder + poids Abraham bruts) = {a_opt:.4f}")
    print(f"  RMS log = {rms_opt:.4f}")
    print(f"\n  {'particule':>5} {'m_exp':>11} {'m_pred Abr.':>13} {'Δ %':>9}")
    print("  " + "-" * 45)
    for nm, comps, me in SPECTRUM:
        m_pred = holder(comps, masses, a_opt)
        err = (m_pred - me) / me * 100
        flag = "★" if abs(err) < 5 else (" " if abs(err) < 20 else "?")
        print(f"  {nm:>5} {me:>11.2f} {m_pred:>13.4f} "
              f"{err:>+8.2f}% {flag}")

    print(f"\n  ─── LECTURE ───")
    print(f"  Range des masses Abraham : min={arr.min():.3f}, max={arr.max():.1f}, "
          f"facteur {arr.max()/arr.min():.0f}×")
    print(f"  Range des masses Hölder  : min=0.51, max=40 439, facteur "
          f"{40439/0.511:.0f}×")
    print(f"  ")
    if rms_opt > 0.5:
        print(f"  ✗ Les poids Abraham bruts (U_EM/cr seul) NE SUFFISENT PAS pour")
        print(f"    reproduire les masses lourdes (proton, W, Z, H, top) via Hölder.")
        print(f"    Le 8_3 catalogue Abraham (U_EM=1492 → m=84 MeV) reste bien")
        print(f"    en dessous des 40 GeV nécessaires.")
    else:
        print(f"  ✓ Précision raisonnable obtenue.")
    print("=" * 90)


if __name__ == "__main__":
    main()
