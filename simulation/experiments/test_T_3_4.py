r"""
TEST DIRECT : T(3,4) = 8_19 résout-il μ⁻ ?

1. Générer la courbe paramétrique de T(3,4) (torus knot)
2. Calculer U_κ et U_τ par Frenet
3. Construire ses propriétés topologiques (b=3, det=3 selon table)
4. Évaluer Ω_ts(3_1 + T(3,4))
5. Comparer à m_μ = 105.66 MeV

Si succès → résout μ⁻ sans paramètre ajustable.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin
from theory.knot_catalog import _torus_knot_curve, knot_curve

# Constantes V5
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

print("Précalcul standard…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj), "s": topological_spin(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_curvature_torsion(pts):
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


def U_KT_from_curve(pts):
    """Calcule U_κ et U_τ pour une courbe discrétisée."""
    n = len(pts)
    kappa, tau, speed = frenet_curvature_torsion(pts)
    ds = speed * 2*np.pi / n
    return np.sum(kappa**2 * ds), np.sum(tau**2 * ds)


# ═══════════ ÉTAPE 1 : générer T(3,4) ═══════════
print("\n" + "═"*82)
print("  ÉTAPE 1 — Générer la courbe T(3,4)")
print("═"*82)

# Vérification : T(2,3) = trefoil 3_1
print("\n  Vérification : T(2,3) doit être le trefoil 3_1")
pts_T23 = _torus_knot_curve(2, 3, 50000)
Uk_T23, Ut_T23 = U_KT_from_curve(pts_T23)
print(f"    U_κ(T(2,3)) calculé  : {Uk_T23:.4f}")
print(f"    U_κ(3_1) référence   : {U_KT['3_1'][0]:.4f}")
print(f"    U_τ(T(2,3)) calculé  : {Ut_T23:.4f}")
print(f"    U_τ(3_1) référence   : {U_KT['3_1'][1]:.4f}")
print(f"    → Ratios κ/τ : {Uk_T23/U_KT['3_1'][0]:.4f} / {Ut_T23/U_KT['3_1'][1]:.4f}")

# Générer T(3,4) — équivalent à 8_19
print("\n  Génération de T(3,4) = 8_19 (torus knot)…")
pts_T34 = _torus_knot_curve(3, 4, 50000)
Uk_T34, Ut_T34 = U_KT_from_curve(pts_T34)
print(f"    U_κ(T(3,4)) = {Uk_T34:.4f}")
print(f"    U_τ(T(3,4)) = {Ut_T34:.4f}")
print(f"    U_κ × U_τ   = {Uk_T34 * Ut_T34:.4f}")

# Comparaison avec les nœuds 8_x existants
print(f"\n  Comparaison avec les 8_x existants :")
for K in ["8_1","8_2","8_3"]:
    uk, ut = U_KT[K]
    print(f"    {K}: U_κ={uk:.2f}, U_τ={ut:.2f}, U_κ×U_τ = {uk*ut:.2f}")
print(f"    T(3,4) : U_κ={Uk_T34:.2f}, U_τ={Ut_T34:.2f}, U_κ×U_τ = {Uk_T34*Ut_T34:.2f}")

# ═══════════ ÉTAPE 2 : propriétés topologiques de T(3,4) ═══════════
print("\n" + "═"*82)
print("  ÉTAPE 2 — Propriétés topologiques de T(3,4) = 8_19")
print("═"*82)

# Jones polynomial de T(3,4) calculée analytiquement
# V_T(p,q)(t) = t^((p-1)(q-1)/2) · (1 - t^(p+1) - t^(q+1) + t^(p+q)) / (1 - t²)
# Pour T(3,4) : (p-1)(q-1)/2 = 3
# Simplification : (1-t^4-t^5+t^7)/(1-t²) = 1 + t² - t^5
# Donc V_T(3,4)(t) = t^3 + t^5 - t^8
jones_T34 = {3: 1.0, 5: 1.0, 8: -1.0}

# Vérification : V(1) = 1, V'(1) = 0, V(-1) = -3
V_at_1 = sum(c for c in jones_T34.values())
V_prime_at_1 = sum(p*c for p, c in jones_T34.items())
V_at_minus1 = sum(c * (-1)**p for p, c in jones_T34.items())
print(f"\n  Polynôme de Jones de T(3,4) :  V(t) = t³ + t⁵ - t⁸")
print(f"    V(1)  = {V_at_1} (doit être 1) {'✓' if V_at_1==1 else '✗'}")
print(f"    V'(1) = {V_prime_at_1} (doit être 0) {'✓' if V_prime_at_1==0 else '✗'}")
print(f"    V(-1) = {V_at_minus1}  →  det = {abs(V_at_minus1)}")

det_T34 = abs(V_at_minus1)
b_T34 = 3  # min(p, q) pour torus knot
cr_T34 = 8  # nb croisements de T(p,q) = (p-1)*q = 2*4 = 8 ✓
n_lobes_T34 = 4  # période de q dans z = -sin(q·t)
m_naive_T34 = cr_T34 * M_C + n_lobes_T34 * M_L
s_T34 = (b_T34 - 1) / 2  # = 1 (boson)

print(f"\n  Propriétés topologiques (calculées) :")
print(f"    crossings   c = {cr_T34}")
print(f"    braid index b = {b_T34}")
print(f"    déterminant   = {det_T34}")
print(f"    n_lobes       = {n_lobes_T34}")
print(f"    M_naive       = {m_naive_T34:.4f} MeV")
print(f"    spin s        = {s_T34} (boson)")

# ═══════════ ÉTAPE 3 : Calculer Ω_ts(3_1 + T(3,4)) ═══════════
print("\n" + "═"*82)
print("  ÉTAPE 3 — Calculer Ω_ts(3_1 + T(3,4))")
print("═"*82)

def omega_3_1_plus_T34():
    """Ω_ts pour 3_1 + T(3,4)."""
    n = 2
    # w_3_1
    w1 = M_NAIVE["3_1"] * (TOPO["3_1"]["b"] * TOPO["3_1"]["det"] / T_31)**P_EXP
    # w_T(3,4)
    T_T34 = b_T34 * det_T34
    w2 = m_naive_T34 * (T_T34 / T_31)**P_EXP
    # Hölder mean
    skel = float((w1**A + w2**A)**(1/A))
    # G
    G = (U_KT["3_1"][0]*U_KT["3_1"][1] + Uk_T34*Ut_T34) / P_31
    # Phi_n
    Phi_n = ((n+1)/2)**(-X_EXP)
    # Phi_pair (3_1 et T(3,4) sont distincts)
    Phi_p = 1.0
    return Phi_n * Phi_p * skel * G**Q_EXP

m_pred = omega_3_1_plus_T34()
err_pct = (m_pred - M_MU)/M_MU*100

print(f"\n  m_pred(3_1 + T(3,4)) = {m_pred:.4f} MeV")
print(f"  m_μ (PDG)            = {M_MU:.4f} MeV")
print(f"  Δ                    = {m_pred - M_MU:+.4f} MeV")
print(f"  Δ %                  = {err_pct:+.3f}%")

if abs(err_pct) < 0.1:
    print(f"\n  ★★★ EXACT ! T(3,4) résout μ⁻ sans paramètre ajustable.")
elif abs(err_pct) < 1.0:
    print(f"\n  ★★ TRÈS BON : <1% — bien meilleur que V5 (3_1+6_1 : +2.27%)")
elif abs(err_pct) < 2.27:
    print(f"\n  ★ AMÉLIORATION : meilleur que V5 sur μ⁻")
elif abs(err_pct) < 5:
    print(f"\n  ⚠ MOINS BON que V5 mais plausible")
else:
    print(f"\n  ✗ ÉCART IMPORTANT — T(3,4) n'aide pas pour μ⁻")

# ═══════════ ÉTAPE 4 : test des spin et charge ═══════════
print("\n" + "═"*82)
print("  ÉTAPE 4 — Vérification spin et charge pour 3_1 + T(3,4)")
print("═"*82)

# Spin set : couplage Clebsch-Gordan
# s(3_1) = 1/2, s(T(3,4)) = (b-1)/2 = 1
# Somme : 1/2 + 1 = 3/2
# CG : J ∈ {1/2, 3/2}
s_31 = 0.5
print(f"\n  Spin de 3_1 : s = {s_31}")
print(f"  Spin de T(3,4) : s = {s_T34}")
print(f"  Somme s_max = {s_31 + s_T34}")
print(f"  J_set (P7quater) = {{0.5, 1.5}} (couplage CG)")
print(f"  → J = 1/2 ACCESSIBLE pour le muon ✓")

# Charge : Q(K) du polynôme de Jones (centre normalisé)
def jones_power_center(poly):
    nonzero = [p for p, c in poly.items() if c != 0]
    if not nonzero: return 0.0
    return (min(nonzero) + max(nonzero)) / (max(nonzero) - min(nonzero))

q_T34 = jones_power_center(jones_T34)
q_3_1 = jones_power_center(jones_polynomial("3_1"))
q_topo = q_3_1 + q_T34
print(f"\n  Q(3_1)   = {q_3_1:.3f}")
print(f"  Q(T(3,4)) = {q_T34:.3f}")
print(f"  Q_topo total = {q_topo:.3f}")
print(f"  Pour Q_μ = -1, il faut Lk = {-1 - q_topo:.3f}")
lk_round = round(-1 - q_topo)
print(f"  Lk arrondi = {lk_round} (tolérance ±1/3)")
if abs(-1 - q_topo - lk_round) < 0.34:
    print(f"  → Charge Q = -1 ACCESSIBLE ✓")
else:
    print(f"  → Charge Q = -1 PROBLÉMATIQUE (résidu {abs(-1 - q_topo - lk_round):.3f})")

# ═══════════ ÉTAPE 5 : Comparaison V5 vs V7 ═══════════
print("\n" + "═"*82)
print("  ÉTAPE 5 — Comparaison V5 vs V7 (avec T(3,4))")
print("═"*82)

# m_μ avec 3_1+6_1 (V5)
def omega_pair(K1, K2):
    n = 2
    w1 = M_NAIVE[K1] * (TOPO[K1]["b"] * TOPO[K1]["det"] / T_31)**P_EXP
    w2 = M_NAIVE[K2] * (TOPO[K2]["b"] * TOPO[K2]["det"] / T_31)**P_EXP
    skel = float((w1**A + w2**A)**(1/A))
    G = (U_KT[K1][0]*U_KT[K1][1] + U_KT[K2][0]*U_KT[K2][1]) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    return Phi_n * skel * G**Q_EXP

m_V5_mu = omega_pair("3_1", "6_1")
m_V5_phi_mu = 103.26

print(f"\n  V5 (3_1+6_1) :      m_pred = {m_V5_mu:.3f} MeV, Δ = {(m_V5_mu-M_MU)/M_MU*100:+.2f}%")
print(f"  V7 (3_1+T(3,4)) :   m_pred = {m_pred:.3f} MeV, Δ = {err_pct:+.2f}%")
gain = abs((m_V5_mu-M_MU)/M_MU*100) - abs(err_pct)
print(f"  Gain V7 vs V5 sur μ⁻ : {gain:+.2f} points de pourcentage")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  T(3,4) = 8_19 torus knot :
    • U_κ      = {Uk_T34:.2f}
    • U_τ      = {Ut_T34:.2f}
    • U_κ·U_τ  = {Uk_T34*Ut_T34:.2f}
    • b        = {b_T34}, det = {det_T34}, c = {cr_T34}

  m(3_1 + T(3,4)) = {m_pred:.3f} MeV  (cible m_μ = {M_MU:.3f})
  Écart : {err_pct:+.3f}%

  V5 actuel  (3_1+6_1)   : Δ = +2.27%
  V7 candidat (3_1+T(3,4)): Δ = {err_pct:+.2f}%

  Verdict : {'★★★ T(3,4) RÉSOUT μ⁻' if abs(err_pct) < 0.5 else
             '★ Amélioration significative' if abs(err_pct) < abs((m_V5_mu-M_MU)/M_MU*100) else
             'Pas d améliorations majeures'}
""")
