r"""
EXPLORATION des nœuds non-toriques exotiques pour combler le gap μ⁻.

Stratégie en 3 phases :

PHASE A — Caractérisation analytique de la cible
   Pour quels (b, det, U_κU_τ, n_l) du partenaire de 3_1 obtient-on
   exactement m_μ = 105.66 MeV ?

PHASE B — Scan exhaustif du catalogue (19 nœuds actuels)
   Tester m(3_1 + K) pour chaque K seul du catalogue.
   Identifier ceux dont les propriétés sont les plus proches de la cible.

PHASE C — Nouveaux candidats non-toriques implémentés
   • Pretzel knot P(p,q,r) — parametrisation explicite
   • Twist knot T_n (famille 4_1 → 5_2 → 6_1 → 7_2 → 8_1 → 9_2 → 10_1 → ...)
   • Knot 8_5 (= P(3,3,2)) à 8 croisements, non-torique

Pour chaque candidat : calcul Frenet des U_κU_τ, comparaison à la cible.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# ───────── Constantes TQNT-V5 ─────────
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

print("═"*82); print("  PHASE A — Caractérisation analytique de la cible"); print("═"*82)

# Précalcul des nœuds catalogués
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_pair_custom(K1, m_naive_2, b_2, det_2, U_κτ_2):
    """Ω_ts pour 3_1 + nœud custom à 2 composantes."""
    w1 = M_NAIVE[K1] * (TOPO[K1]["b"] * TOPO[K1]["det"] / T_31)**P_EXP
    w2 = m_naive_2 * (b_2 * det_2 / T_31)**P_EXP
    skel = float((w1**A + w2**A)**(1/A))
    G = (U_KT[K1][0]*U_KT[K1][1] + U_κτ_2) / P_31
    Phi_n = (3/2)**(-X_EXP)
    return Phi_n * skel * G**Q_EXP


# CIBLE : trouver U_κU_τ_K tel que m(3_1+K) = m_μ exact,
# pour différentes valeurs (b, det) caractéristiques
print(f"\n  Cible : m_μ = {M_MU} MeV")
print(f"  Référence V5 : 3_1+6_1 donne m_pred = 103.26 MeV (Δ = -2.27%)")
print(f"\n  Pour quels (b, det, n_l) du partenaire de 3_1 obtient-on m_μ EXACT?")
print(f"  (En faisant varier U_κU_τ comme inconnue)")
print()
print(f"  {'b':>3} {'det':>4} {'n_l':>4} {'cr':>3} {'M_naive':>9} {'U_κU_τ_req':>13} {'Δ vs 6_1':>11}")
print("  " + "-" * 60)
print(f"  {'(6_1 actuel = b=4, det=9, n_l=2, cr=6, U_κU_τ ≈ ?)':<60}")
print(f"  6_1: U_κU_τ = {U_KT['6_1'][0]*U_KT['6_1'][1]:.2f}")
print()

U_6_1_actuel = U_KT['6_1'][0]*U_KT['6_1'][1]

cibles_test = [
    (4, 9, 2, 6),   # 6_1 actuel
    (3, 9, 2, 6),   # b plus faible
    (5, 9, 2, 6),   # b plus élevé
    (4, 11, 2, 6),  # det plus élevé
    (4, 7, 2, 6),   # det plus faible
    (4, 9, 3, 6),   # n_l plus élevé
    (3, 7, 2, 5),   # 5_2 type
    (4, 13, 2, 7),  # 7_2 type
    (3, 5, 2, 4),   # 4_1 type
    (2, 5, 2, 5),   # twist knot 5_2
]

for b, d, nl, cr in cibles_test:
    m_naive = cr*M_C + nl*M_L
    def f(u):
        return omega_pair_custom("3_1", m_naive, b, d, u) - M_MU
    try:
        u_req = brentq(f, 0.1, 10000)
        delta = (u_req - U_6_1_actuel)/U_6_1_actuel*100
        print(f"  {b:>3} {d:>4} {nl:>4} {cr:>3} {m_naive:>9.3f} {u_req:>13.2f} {delta:>+10.1f}%")
    except Exception as e:
        print(f"  {b:>3} {d:>4} {nl:>4} {cr:>3} {m_naive:>9.3f} {'no_sol':>13}")


print("\n" + "═"*82); print("  PHASE B — Scan exhaustif du catalogue (19 nœuds)"); print("═"*82)
print(f"\n  m(3_1 + K) pour chaque K seul du catalogue actuel :")
print(f"  {'K':<6} {'cr':>3} {'b':>3} {'det':>4} {'U_κU_τ':>9} {'m(3_1+K)':>11} {'Δ vs m_μ':>11}")
print("  " + "-" * 70)
results_cat = []
for K in KNOTS:
    Uk, Ut = U_KT[K]
    U_κτ = Uk*Ut
    m_pred = omega_pair_custom("3_1", M_NAIVE[K], TOPO[K]["b"], TOPO[K]["det"], U_κτ)
    delta = (m_pred - M_MU)/M_MU*100
    results_cat.append((K, U_κτ, m_pred, delta))

# Trier par proximité à m_μ
results_cat.sort(key=lambda x: abs(x[3]))
for K, u, m, d in results_cat:
    flag = "★★" if abs(d) < 1 else ("★" if abs(d) < 5 else ("•" if abs(d) < 15 else ""))
    print(f"  {K:<6} {TOPO[K]['cr']:>3} {TOPO[K]['b']:>3} {TOPO[K]['det']:>4} "
          f"{u:>9.2f} {m:>11.3f} {d:>+10.2f}% {flag}")


# ───────── PHASE C : nouveaux candidats non-toriques ─────────
print("\n" + "═"*82); print("  PHASE C — Implémentation de nouveaux candidats non-toriques"); print("═"*82)


def frenet_curve(pts):
    """Calcul des invariants de Frenet à partir d'une courbe paramétrique."""
    n = len(pts); dt = 2*np.pi/n
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    return kappa, tau, speed


def UKUT(pts):
    k, t, s = frenet_curve(pts)
    ds = s * 2*np.pi/len(pts)
    return float(np.sum(k**2 * ds)), float(np.sum(t**2 * ds))


# ──── Candidat 1 : Twist knot T_n ────
# Paramétrisation : deux loops connectés par n demi-twists.
# Famille topologique : T_n = (n=1:3_1) (n=2:4_1) (n=3:5_2) (n=4:6_1)
#                     (n=5:7_2) (n=6:8_1) (n=7:9_2) (n=8:10_1) (n=9:11_2) ...
def twist_knot_curve(n, n_points=20000):
    """Twist knot avec n demi-twists. R = grand rayon, r = petit rayon."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    R, r = 3, 1
    # Forme 2-bridge : oscillation entre deux lobes connectés par n twists
    # On reprend la forme T(2,n) avec ajustement non-torique pour n pair
    # (Pour n impair → torus T(2,n), pour n pair → hyperbolique)
    # Forme générale : (R+r·cos(n·t))·cos(t), (R+r·cos(n·t))·sin(t), r·sin(n·t)
    # avec n=p/q rationnel pour les non-toriques.
    p, q = n, 2  # n croisements demandés
    x = (R + r*np.cos(p*t/q)) * np.cos(t)
    y = (R + r*np.cos(p*t/q)) * np.sin(t)
    z = r*np.sin(p*t/q)
    return np.column_stack([x, y, z])


# ──── Candidat 2 : Pretzel knot P(p,q,r) ────
# Paramétrisation simple : trois lobes oscillants à 120° chacun
# avec twists p, q, r dans chaque région
def pretzel_curve(p, q, r, n_points=20000):
    """Pretzel P(p,q,r) approximé par trois lobes twistés."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # Trois lobes à angles 0, 2π/3, 4π/3
    # Dans chaque lobe, twist count correspond à p, q, r
    twists = [p, q, r]
    x = np.zeros(n_points); y = np.zeros(n_points); z = np.zeros(n_points)
    for i, tw in enumerate(twists):
        phi0 = i * 2*np.pi/3
        # Localisation autour du lobe i : window centré sur t = phi0
        # plus continuité globale via somme de fenêtres gaussiennes
        for k, tk in enumerate(t):
            d = (tk - phi0) % (2*np.pi)
            d = min(d, 2*np.pi - d)
            w = np.exp(-d**2 * 4)
            x[k] += w * 3*np.cos(tk)
            y[k] += w * 3*np.sin(tk)
            z[k] += w * 0.7*np.sin(tw*tk + i*np.pi/3)
    return np.column_stack([x, y, z])


# ──── Candidat 3 : 8_5 (knot non-torique) ────
# 8_5 = P(3,3,-2) pretzel non-torique
def knot_8_5_curve(n_points=20000):
    """8_5 non-torique = pretzel P(3,3,-2)."""
    return pretzel_curve(3, 3, 2, n_points)  # signes ignorés (Frenet symétrique)


# ──── Candidat 4 : 9_2 (twist knot) ────
def knot_9_2_curve(n_points=20000):
    """9_2 = twist knot avec 7 demi-twists (famille 4_1/5_2/6_1/7_2/8_1/9_2/10_1)."""
    return twist_knot_curve(7, n_points)


# ──── Candidat 5 : 8_19 = T(3,4) (déjà testé pour mémoire) ────
def knot_T34_curve(n_points=20000):
    """T(3,4) = 8_19."""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    R, r = 3, 1
    p, q = 3, 4
    x = (R + r*np.cos(q*t)) * np.cos(p*t)
    y = (R + r*np.cos(q*t)) * np.sin(p*t)
    z = r*np.sin(q*t)
    return np.column_stack([x, y, z])


# Liste de candidats à tester
candidates = [
    ("twist T_3 (≈5_2)", twist_knot_curve(3), 2, 7, 2, 5),
    ("twist T_4 (≈6_1)", twist_knot_curve(4), 4, 9, 2, 6),  # comparaison
    ("twist T_5 (≈7_2)", twist_knot_curve(5), 4, 11, 2, 7),
    ("twist T_6 (≈8_1)", twist_knot_curve(6), 5, 13, 2, 8),
    ("twist T_7 (=9_2)", twist_knot_curve(7), 6, 15, 2, 9),
    ("twist T_8 (≈10_1)", twist_knot_curve(8), 7, 17, 2, 10),
    ("pretzel P(3,3,3)", pretzel_curve(3, 3, 3), 4, 13, 3, 9),
    ("pretzel P(3,3,5)", pretzel_curve(3, 3, 5), 4, 17, 3, 11),
    ("pretzel P(3,5,5)", pretzel_curve(3, 5, 5), 4, 17, 3, 13),
    ("8_5 = P(3,3,2)",  pretzel_curve(3, 3, 2), 3, 21, 3, 8),
    ("T(3,4) = 8_19",   knot_T34_curve(),       3, 3, 4, 8),
]

print(f"\n  Test : m(3_1 + K_exotique) pour chaque candidat")
print(f"  {'Candidat':<24} {'cr':>3} {'b':>3} {'det':>4} {'U_κU_τ':>9} "
      f"{'m(3_1+K)':>11} {'Δ vs m_μ':>10}")
print("  " + "-" * 80)
exotic_results = []
for name, curve, b, det, nl, cr in candidates:
    Uk, Ut = UKUT(curve)
    U_κτ = Uk * Ut
    m_naive = cr*M_C + nl*M_L
    m_pred = omega_pair_custom("3_1", m_naive, b, det, U_κτ)
    delta = (m_pred - M_MU)/M_MU*100
    flag = "★★" if abs(delta) < 1 else ("★" if abs(delta) < 5 else "")
    print(f"  {name:<24} {cr:>3} {b:>3} {det:>4} {U_κτ:>9.2f} "
          f"{m_pred:>11.3f} {delta:>+9.2f}% {flag}")
    exotic_results.append((name, b, det, U_κτ, m_pred, delta))


print("\n" + "═"*82); print("  SYNTHÈSE — qu'avons-nous trouvé ?"); print("═"*82)

# Identifier le meilleur candidat global
all_results = []
for K, u, m, d in results_cat:
    all_results.append((K, "catalogue", u, m, d, TOPO[K]["b"], TOPO[K]["det"]))
for name, b, det, u, m, d in exotic_results:
    all_results.append((name, "exotique", u, m, d, b, det))
all_results.sort(key=lambda x: abs(x[4]))

print(f"\n  Top 8 candidats les plus proches de m_μ = {M_MU:.2f} MeV :")
print(f"  {'Candidat':<24} {'origine':<12} {'b':>3} {'det':>4} {'U_κU_τ':>9} "
      f"{'m_pred':>10} {'Δ %':>8}")
print("  " + "-" * 80)
for nm, orig, u, m, d, b, dd in all_results[:8]:
    print(f"  {nm:<24} {orig:<12} {b:>3} {dd:>4} {u:>9.2f} {m:>10.3f} {d:>+7.2f}%")

print("\n  Diagnostic :")
print("    • Le scan du catalogue montre quel K seul donne m(3_1+K) le plus")
print("      proche de m_μ (actuellement 6_1 → -2.27%).")
print("    • Les candidats exotiques testent si un nœud non-catalogué")
print("      pourrait combler le gap topologique [50, 120].")
print("    • ATTENTION : pour les nouveaux candidats, le calcul de Frenet")
print("      utilise une paramétrisation approximative. Pour une validation")
print("      rigoureuse, il faudrait :")
print("        (i)  vérifier que la paramétrisation est bien un nœud")
print("             du type prétendu (calcul d'invariants de Jones)")
print("        (ii) raffiner b, det par identification topologique exacte")
print("        (iii) tester d'autres paramétrisations indépendantes")
print("    • Les valeurs (b, det) utilisées pour les candidats exotiques")
print("      proviennent de tables de Rolfsen (à vérifier).")
