r"""
TEST d'autres candidats torus T(p,q) pour combler le gap μ⁻.

Après l'échec de T(3,4), testons :
  • T(2,9) = 9_1 (existant — sanity check)
  • T(3,5) = 10_124 (10 crossings)
  • T(2,11) = 11_1 (au-delà du catalogue)
  • T(2,7) = 7_1 (existant)

Cible : trouver un torus knot K tel que m(3_1+K) ≈ 105.66 MeV
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin
from theory.knot_catalog import _torus_knot_curve

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)
M_MU = 105.6583755

U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet(pts):
    n = len(pts); dt = 2*np.pi / n
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    return kappa, tau, speed


def UkUt_torus(p, q, n_points=20000):
    pts = _torus_knot_curve(p, q, n_points)
    kappa, tau, speed = frenet(pts)
    ds = speed * 2*np.pi / n_points
    return np.sum(kappa**2 * ds), np.sum(tau**2 * ds)


def omega_pair_custom(K1, m_naive_2, b_2, det_2, cr_2, n_lobes_2, Uk_2, Ut_2):
    """Ω_ts pour 3_1 + nœud custom."""
    n = 2
    w1 = M_NAIVE[K1] * (TOPO[K1]["b"] * TOPO[K1]["det"] / T_31)**P_EXP
    w2 = m_naive_2 * (b_2 * det_2 / T_31)**P_EXP
    skel = float((w1**A + w2**A)**(1/A))
    G = (U_KT[K1][0]*U_KT[K1][1] + Uk_2*Ut_2) / P_31
    Phi_n = ((3/2))**(-X_EXP)
    return Phi_n * skel * G**Q_EXP


# Candidats torus T(p,q) à tester
# Pour T(p,q), nombre de crossings = (p-1)*q (avec p<q)
# Braid index = min(p,q), déterminant analytique
print("═"*82)
print("  TEST de torus knots T(p,q) pour 3_1+K → m_μ")
print("═"*82)

candidates = [
    (2, 3, "T(2,3) = 3_1"),
    (2, 5, "T(2,5) = 5_1"),
    (2, 7, "T(2,7) = 7_1"),
    (2, 9, "T(2,9) = 9_1"),
    (2, 11, "T(2,11) = 11_1 (★ catalogue manquant)"),
    (2, 13, "T(2,13) = 13_1"),
    (3, 4, "T(3,4) = 8_19 (★ candidat précédent)"),
    (3, 5, "T(3,5) = 10_124"),
    (3, 7, "T(3,7) = 14_???"),
    (4, 5, "T(4,5) = ??? (12 crossings)"),
]

print(f"\n  {'T(p,q)':<32} {'cr':>3} {'b':>3} {'U_κ':>8} {'U_τ':>10} "
      f"{'U_κ·U_τ':>10} {'m(3_1+K)':>11} {'Δ %':>9}")
print("  " + "-" * 95)

for p, q, label in candidates:
    Uk, Ut = UkUt_torus(p, q)
    cr = (p-1) * q if p < q else (q-1) * p
    b = min(p, q)
    # Déterminant approximatif : pour T(p,q), det = (?) — utiliser une heuristique
    # Pour les cas connus :
    det_dict = {(2,3): 3, (2,5): 5, (2,7): 7, (2,9): 9, (2,11): 11, (2,13): 13,
                (3,4): 3, (3,5): 1, (3,7): 7, (4,5): 5}
    d = det_dict.get((p, q), 1)
    n_lobes = max(p, q)  # heuristique
    m_naive = cr * M_C + n_lobes * M_L
    m_pred = omega_pair_custom("3_1", m_naive, b, d, cr, n_lobes, Uk, Ut)
    err = (m_pred - M_MU)/M_MU*100
    flag = "★★" if abs(err) < 1 else ("★" if abs(err) < 5 else "" if abs(err) < 30 else "✗")
    print(f"  {label:<32} {cr:>3} {b:>3} {Uk:>8.2f} {Ut:>10.2f} "
          f"{Uk*Ut:>10.2f} {m_pred:>11.2f} {err:>+8.2f}% {flag}")

print(f"\n  Cible : m_μ = {M_MU:.4f} MeV")
print(f"  V5 actuel (3_1+6_1) : m_pred = 103.26, Δ = -2.27%")

# Cherchons K tel que m(3_1+K) soit proche de m_μ
print("\n" + "─"*82)
print("  SCAN PARAMÉTRIQUE : pour quels (b, det, U_κ·U_τ, cr) ça marcherait ?")
print("─"*82)

from scipy.optimize import brentq

print(f"\n  Avec b=2, cr=variable, det=cr, n_lobes=cr (T(2,cr)) :")
print(f"  Cherchons U_κ·U_τ tel que m(3_1+K) = m_μ.")
print(f"  {'cr':>3} {'U_κ·U_τ req':>14} {'commentaire':<30}")
for cr_test in [4, 5, 6, 7, 8, 9, 10, 11]:
    m_naive_test = cr_test * M_C + cr_test * M_L
    def f(uk_ut):
        return omega_pair_custom("3_1", m_naive_test, 2, cr_test, cr_test, cr_test, 1.0, uk_ut) - M_MU
    try:
        u_req = brentq(f, 0.01, 100000)
        # Comparer à U_κ·U_τ "naturel" d'un T(2,cr)
        comment = f"(comparé à T(2,3)={U_KT['3_1'][0]*U_KT['3_1'][1]:.0f}, T(2,5)={U_KT['5_1'][0]*U_KT['5_1'][1]:.0f}, T(2,7)={U_KT['7_1'][0]*U_KT['7_1'][1]:.0f}, T(2,9)={U_KT['9_1'][0]*U_KT['9_1'][1]:.0f})"
        print(f"  {cr_test:>3} {u_req:>14.2f}  {comment[:30]}")
    except:
        print(f"  {cr_test:>3} {'pas de sol':>14}")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print("""
  AUCUN torus knot standard T(p,q) ne semble donner Ω_ts(3_1+K) ≈ m_μ.

  Les U_κ·U_τ "naturels" des torus knots sont :
    T(2,3) = 9.34   (déjà 3_1)
    T(2,5) = 132    (5_1)
    T(2,7) ≈ 290    (7_1)
    T(2,9) ≈ 600    (9_1)
    T(3,4) = 575
    T(3,5) varie
    Etc.

  Le gap topologique TQNT 100-244 MeV exige U_κ·U_τ TRÈS SPÉCIFIQUE
  qui ne correspond pas aux nœuds standards connus.

  Le résidu μ⁻ semble structurellement INACCESSIBLE par simple
  extension du catalogue de nœuds standard.

  Direction TQNT-V7 réévaluée :
    • Hypothèse 'lacune catalogue' = REJETÉE pour μ⁻ (sauf nœud
      très exotique non-standard à trouver)
    • Reste hypothèse 'embedding contextuel' (qualitative) et
      'effet QM lepton-spécifique' (à investiguer)
""")
