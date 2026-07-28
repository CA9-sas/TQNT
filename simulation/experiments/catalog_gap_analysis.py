r"""
ANALYSE DU GAP TOPOLOGIQUE — quel nœud manque-t-il pour μ⁻ ?

Le catalogue TQNT actuel contient 19 nœuds parmi les ~165 connus à
≤10 croisements :
  3_1, 4_1, 5_1, 5_2, 6_1, 6_2, 6_3,
  7_1, 7_2, 7_3, 7_4, 7_5, 7_6, 7_7,
  8_1, 8_2, 8_3, 9_1, 10_1.

Manquent : 8_4 à 8_21 (18 nœuds), 9_2 à 9_49 (48 nœuds),
           10_2 à 10_165 (164 nœuds).

Pour résoudre μ⁻ (m=105.66 MeV) avec une config 3_1+K, on cherche K
tel que Ω_ts(3_1+K) ≈ 105.66.

Algorithme :
  1. Pour chaque K du catalogue actuel, calculer Ω_ts(3_1+K).
  2. Identifier le gap autour de m_μ.
  3. Calculer les paramètres (b, det, U_κ·U_τ) que devrait avoir un
     nœud manquant K* pour combler exactement le gap.
  4. Confronter aux propriétés tabulées des 8_4-8_21 et autres.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq, minimize_scalar
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

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

print("Précalcul…")
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


def omega_pair(K1, K2, m_naive_K2=None, b_K2=None, det_K2=None,
               n_lobes_K2=None, cr_K2=None, U_kappa_tau_K2=None):
    """Ω_ts(K1 + K2) où K2 peut être HYPOTHÉTIQUE (paramètres fournis)."""
    comps = [K1, K2]
    n = 2
    # K1 utilise les valeurs du catalogue
    w1 = M_NAIVE[K1] * (TOPO[K1]["b"] * TOPO[K1]["det"] / T_31)**P_EXP
    # K2 : si paramètres fournis, utiliser ceux-ci
    if m_naive_K2 is None: m_naive_K2 = M_NAIVE[K2]
    if b_K2 is None: b_K2 = TOPO[K2]["b"]
    if det_K2 is None: det_K2 = TOPO[K2]["det"]
    if cr_K2 is None: cr_K2 = TOPO[K2]["cr"]
    if U_kappa_tau_K2 is None: U_kappa_tau_K2 = U_KT[K2][0] * U_KT[K2][1]
    T_K2 = b_K2 * det_K2
    w2 = m_naive_K2 * (T_K2 / T_31)**P_EXP
    skel = float((w1**A + w2**A)**(1/A))
    G = (U_KT[K1][0]*U_KT[K1][1] + U_kappa_tau_K2) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    # Φ_pair : seulement si K1 == K2 (pas le cas ici si K2 hypothétique différent)
    Phi_p = 1.0
    if K1 == K2:
        Phi_p = RHO_0 * cr_K2**(-B)
    return Phi_n * Phi_p * skel * G**Q_EXP


def omega_3_1_plus(K, m_naive=None, b=None, det=None, cr=None, U_kt=None):
    """Wrapper pour 3_1 + K."""
    return omega_pair("3_1", K if K else "3_1", m_naive, b, det,
                      None, cr, U_kt)


# ───────── ÉTAPE 1 : Spectre actuel 3_1+K pour K du catalogue ─────────
print("\n" + "═"*82)
print("  ÉTAPE 1 — Spectre des entrelacs 3_1+K du catalogue actuel")
print("═"*82)
print(f"\n  Cible : m_μ = {M_MU:.3f} MeV")
print(f"  {'K':<6} {'cr':>3} {'b':>3} {'det':>5} {'U_κ·U_τ':>10} "
      f"{'m(3_1+K)':>11} {'Δ vs m_μ':>10}")
print("  " + "-" * 60)

spectrum = []
for K in KNOTS:
    if K == "3_1": continue  # 3_1+3_1 traité plus loin
    m = omega_pair("3_1", K)
    err_mev = m - M_MU
    spectrum.append((K, m, err_mev))
    flag = "★ proche" if abs(err_mev) < 50 else ""
    print(f"  {K:<6} {TOPO[K]['cr']:>3} {TOPO[K]['b']:>3} "
          f"{TOPO[K]['det']:>5} {U_KT[K][0]*U_KT[K][1]:>10.2f} "
          f"{m:>11.3f} {err_mev:>+9.2f} {flag}")

spectrum.sort(key=lambda x: x[1])
print(f"\n  ─── GAP TOPOLOGIQUE autour de m_μ ───")
below = [s for s in spectrum if s[1] < M_MU]
above = [s for s in spectrum if s[1] >= M_MU]
if below: print(f"  Juste EN DESSOUS : {below[-1][0]} → {below[-1][1]:.2f} MeV (Δ = {below[-1][2]:+.2f})")
if above: print(f"  Juste AU DESSUS  : {above[0][0]} → {above[0][1]:.2f} MeV (Δ = {above[0][2]:+.2f})")
print(f"  GAP : {above[0][1] - below[-1][1]:.2f} MeV (de {below[-1][1]:.2f} à {above[0][1]:.2f})")

# ───────── ÉTAPE 2 : quelles propriétés devrait avoir K* hypothétique ? ─────────
print("\n" + "═"*82)
print("  ÉTAPE 2 — Propriétés requises pour un nœud K* qui donnerait m_μ exact")
print("═"*82)

# On fixe 4 propriétés sur 5 et on cherche la 5e qui donne m_μ.
# On essaie différentes hypothèses crédibles.

print(f"\n  Hypothèse A : c=7, M_naive ≈ {7*M_C + 2*M_L:.4f}, n_lobes=2")
print(f"  (cherchons b et det compatibles)")
print(f"  {'b':>3} {'det':>5} {'U_κ·U_τ':>10} {'m_pred':>11} {'Δ %':>7}")
print("  " + "-" * 42)
# Pour c=7, b ∈ {2,3,4,5}, det varie
for b_test in [2, 3, 4, 5]:
    for det_test in [1, 3, 5, 7, 9, 11, 13, 15, 21, 27]:
        # U_κ · U_τ : approximer par interpolation des 7_x existants
        U_kt_typique = 130  # moyenne des 7_x
        m_pred = omega_pair("3_1", "DUMMY",
                            m_naive_K2=7*M_C + 2*M_L,
                            b_K2=b_test, det_K2=det_test,
                            cr_K2=7,
                            U_kappa_tau_K2=U_kt_typique)
        if abs(m_pred - M_MU) < 5:
            err = (m_pred - M_MU)/M_MU*100
            print(f"  {b_test:>3} {det_test:>5} {U_kt_typique:>10.2f} {m_pred:>11.2f} {err:>+7.2f}%")

# Cherchons systématiquement pour des c entre 6 et 11 quels U_κ·U_τ donnent m_μ
print(f"\n  ─── PROPRIÉTÉS REQUISES de K* pour COMBLER le gap exactement ───")
print(f"  Pour chaque (c, b, det) raisonnable, on cherche U_κ·U_τ tel que")
print(f"  Ω_ts(3_1+K*) = m_μ exact.\n")
print(f"  {'c':>3} {'b':>3} {'det':>5} {'U_κ·U_τ requis':>16} {'Validité':>12}")
print("  " + "-" * 45)

# Échantillon des paramètres possibles selon les knot tables standards :
# Source : http://katlas.org
candidates = [
    # (c, b, det, name, source)
    (8,  3, 19, "8_4",  "katlas"),
    (8,  3, 21, "8_5",  "katlas"),
    (8,  4, 23, "8_6",  "katlas"),
    (8,  4, 23, "8_7",  "katlas"),
    (8,  4, 25, "8_8",  "katlas"),
    (8,  3, 25, "8_9",  "katlas"),
    (8,  4, 27, "8_10", "katlas"),
    (8,  3, 27, "8_11", "katlas"),
    (8,  4, 29, "8_12", "katlas"),
    (8,  4, 29, "8_13", "katlas"),
    (8,  4, 31, "8_14", "katlas"),
    (8,  4, 33, "8_15", "katlas"),
    (8,  4, 35, "8_16", "katlas"),
    (8,  4, 37, "8_17", "katlas"),
    (8,  4, 45, "8_18", "katlas"),
    (8,  3, 3,  "8_19", "T(3,4) torus"),
    (8,  3, 9,  "8_20", "katlas"),
    (8,  3, 15, "8_21", "katlas"),
    (9,  2, 9,  "9_2 (T(2,9)?)", "approx"),
    (9,  3, 17, "9_3",  "katlas"),
    (9,  4, 25, "9_10", "katlas"),
    (9,  5, 39, "9_42", "katlas"),
]

print(f"\n  {'K*':<22} {'c':>3} {'b':>3} {'det':>5} {'U_κ·U_τ requis':>16}")
print("  " + "-" * 65)
for c, b, d, name, src in candidates:
    m_naive_K = c*M_C + 2*M_L  # estimation n_lobes=2 (typique)
    # Résoudre pour U_κ·U_τ
    def f_uk(u_kt):
        return omega_pair("3_1", "DUMMY",
                          m_naive_K2=m_naive_K, b_K2=b, det_K2=d,
                          cr_K2=c, U_kappa_tau_K2=u_kt) - M_MU
    try:
        # Estimer une plage raisonnable
        u_lo, u_hi = 1.0, 1000.0
        if f_uk(u_lo) * f_uk(u_hi) < 0:
            u_req = brentq(f_uk, u_lo, u_hi)
            # Comparer à U_κ·U_τ des 7_x existants
            sim_7 = [U_KT[k][0]*U_KT[k][1] for k in KNOTS if k.startswith("7_")]
            sim_8 = [U_KT[k][0]*U_KT[k][1] for k in KNOTS if k.startswith("8_")]
            print(f"  {name:<22} {c:>3} {b:>3} {d:>5} {u_req:>14.2f}")
        else:
            print(f"  {name:<22} {c:>3} {b:>3} {d:>5} (pas de solution dans [1, 1000])")
    except Exception as e:
        print(f"  {name:<22} erreur : {e}")

# Comparaison avec U_κ·U_τ typiques
print(f"\n  ─── U_κ·U_τ TYPIQUES dans le catalogue actuel ───")
for prefix in ["6_", "7_", "8_"]:
    vals = [U_KT[K][0]*U_KT[K][1] for K in KNOTS if K.startswith(prefix)]
    if vals:
        print(f"    {prefix}_x : min={min(vals):.1f}, max={max(vals):.1f}, "
              f"moyen={np.mean(vals):.1f}, médian={np.median(vals):.1f}")

# ───────── ÉTAPE 3 : combien de configs n=2 et n=3 sont accessibles ─────────
print("\n" + "═"*82)
print("  ÉTAPE 3 — Si on ajoutait 8_4 (b=3, det=19) au catalogue, gain ?")
print("═"*82)

# Plage de U_κ·U_τ "raisonnable" pour 8_4 (entre 8_1 et 8_2)
U_31_82 = (U_KT["8_1"][0]*U_KT["8_1"][1], U_KT["8_2"][0]*U_KT["8_2"][1])
print(f"\n  U_κ·U_τ(8_1) = {U_KT['8_1'][0]*U_KT['8_1'][1]:.2f}")
print(f"  U_κ·U_τ(8_2) = {U_KT['8_2'][0]*U_KT['8_2'][1]:.2f}")
print(f"  → Pour 8_4, plage plausible : [50, 1000]\n")

print(f"  Quelle U_κ·U_τ(8_4) donnerait m(3_1+8_4) = m_μ EXACT ?")
m_naive_84 = 8*M_C + 2*M_L
def f_84(u_kt):
    return omega_pair("3_1", "DUMMY",
                      m_naive_K2=m_naive_84,
                      b_K2=3, det_K2=19, cr_K2=8,
                      U_kappa_tau_K2=u_kt) - M_MU
try:
    u_required = brentq(f_84, 1.0, 1e6)
    print(f"  → U_κ·U_τ(8_4) requis : {u_required:.2f}")
    sim_8 = [U_KT[k][0]*U_KT[k][1] for k in KNOTS if k.startswith("8_")]
    print(f"  → Comparé à 8_1, 8_2, 8_3 : {sim_8}")
    if min(sim_8) < u_required < max(sim_8):
        print(f"  ★ Valeur dans la plage des 8_x existants → PLAUSIBLE")
    elif u_required < min(sim_8):
        print(f"  ⚠ Plus petit que tous les 8_x existants")
    else:
        print(f"  ⚠ Plus grand que tous les 8_x existants")
except Exception as e:
    print(f"  Erreur : {e}")

print("\n" + "═"*82)
print("  ÉTAPE 4 — Liste des nœuds candidats à AJOUTER pour TQNT-V7")
print("═"*82)
print(f"""
  Pour combler le gap topologique 100-244 MeV (essentiel pour μ⁻),
  les nœuds suivants méritent d'être étudiés et possiblement
  ajoutés au catalogue TQNT :

  PRIORITÉ HAUTE (8 crossings) :
    • 8_4 : b=3, det=19  — combler gap basé sur estimation U_κ·U_τ
    • 8_5 : b=3, det=21  — alternative
    • 8_19 = T(3,4) torus knot : b=3, det=3 — structure spéciale

  PRIORITÉ MOYENNE (9 crossings) :
    • 9_2, 9_3, 9_4, 9_5, 9_6 — nœuds 9-crossings non-toriques
    • Le catalogue actuel n'a que 9_1 (torus T(2,9))

  PRIORITÉ BASSE (10 crossings) :
    • 10_2 à 10_165 : 164 nœuds non testés

  IMPLÉMENTATION REQUISE :
    Pour chaque nœud ajouté, il faut :
      1. Construire la courbe paramétrique r(t) (algorithme : harmoniques
         de Fourier ajustées au diagramme de braid)
      2. Calculer U_κ et U_τ par Frenet
      3. Obtenir le polynôme de Jones (tables ou algorithme Kauffman)
      4. Tester sur le spectre PDG

  LE NŒUD QUI COMBLERAIT EXACTEMENT μ⁻ aurait, idéalement :
    c = 8 ou 9, b = 3 ou 4, det dans [9, 25],
    U_κ·U_τ ∈ valeur calculée ci-dessus
""")
