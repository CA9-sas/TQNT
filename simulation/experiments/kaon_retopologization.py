r"""
RÉVISION TOPOLOGIQUE DES KAONS

L'assignation actuelle K = 3_1+3_1+6_2 donne m_V5 ≈ 2255 MeV
(écart +353% par rapport à m_K_exp ≈ 494 MeV).

Hypothèse : la topologie est fondamentalement incorrecte.

Stratégie de recherche :
  1. Scan EXHAUSTIF des paires (K1, K2) et triplets (K1, K2, K3)
     du catalogue de 19 nœuds.
  2. Filtrer celles donnant m_V5 ∈ [m_K - 5%, m_K + 5%]
     pour K⁻ (493.68) et K⁰ (497.61).
  3. Identifier la (ou les) assignation(s) physiquement consistantes.

Cibles :
  K± : m = 493.68 MeV, Q=±1, J=0, S=∓1, I_3=±1/2
  K⁰ : m = 497.61 MeV, Q=0,  J=0, S=±1, I_3=∓1/2

Contraintes quantiques (V5 actuel) :
  • Q = Σ Q_topo(K_i) + Lk (à charge mod 1)
  • J : couplage des spins topologiques par P7quater
  • Étrangeté : pas explicite dans V5, à interpréter
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "spin": topological_spin(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_link(components):
    n = len(components)
    ws = []; Us = []
    for K in components:
        Us.append(U_KT[K][0]*U_KT[K][1])
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# Cibles
M_K_PLUS = 493.677
M_K_0    = 497.611
M_K_AVG  = (M_K_PLUS + M_K_0) / 2  # 495.6
TOL = 0.05  # ±5%

print(f"\n  Cibles :")
print(f"    K± : {M_K_PLUS:.3f} MeV")
print(f"    K⁰ : {M_K_0:.3f} MeV")
print(f"    Tolérance : ±{TOL*100:.1f}%  → [{M_K_AVG*(1-TOL):.1f}, {M_K_AVG*(1+TOL):.1f}] MeV")

# ───────── ÉTAPE 1 — Scan des configurations à 2 et 3 composantes ─────────
print("\n" + "═"*100)
print("  ÉTAPE 1 — Scan des paires (n=2)")
print("═"*100)

candidates_n2 = []
for K1, K2 in itertools.combinations_with_replacement(KNOTS, 2):
    cfg = [K1, K2]
    m = omega_link(cfg)
    err = abs(m - M_K_AVG)/M_K_AVG
    if err < TOL:
        candidates_n2.append((cfg, m, err))

candidates_n2.sort(key=lambda x: x[2])
print(f"  Trouvé {len(candidates_n2)} configurations à n=2 dans la tolérance ±{TOL*100:.0f}%")
print(f"  {'config':<22} {'m_pred':>10} {'Δ K_avg':>9}")
print("  " + "-" * 50)
for cfg, m, err in candidates_n2[:15]:
    delta = (m - M_K_AVG)/M_K_AVG*100
    print(f"  {'+'.join(cfg):<22} {m:>10.3f} {delta:>+8.2f}%")


print("\n" + "═"*100)
print("  ÉTAPE 2 — Scan des triplets (n=3)")
print("═"*100)

candidates_n3 = []
for K1, K2, K3 in itertools.combinations_with_replacement(KNOTS, 3):
    cfg = [K1, K2, K3]
    m = omega_link(cfg)
    err = abs(m - M_K_AVG)/M_K_AVG
    if err < TOL:
        candidates_n3.append((cfg, m, err))

candidates_n3.sort(key=lambda x: x[2])
print(f"  Trouvé {len(candidates_n3)} configurations à n=3 dans la tolérance ±{TOL*100:.0f}%")
print(f"  {'config':<22} {'m_pred':>10} {'Δ K_avg':>9} {'cr_tot':>7} "
      f"{'b_max':>6} {'det_prod':>9}")
print("  " + "-" * 80)
for cfg, m, err in candidates_n3[:30]:
    delta = (m - M_K_AVG)/M_K_AVG*100
    cr_tot = sum(TOPO[K]["cr"] for K in cfg)
    b_max = max(TOPO[K]["b"] for K in cfg)
    det_prod = np.prod([TOPO[K]["det"] for K in cfg])
    print(f"  {'+'.join(cfg):<22} {m:>10.3f} {delta:>+8.2f}% "
          f"{cr_tot:>7} {b_max:>6} {det_prod:>9}")


print("\n" + "═"*100)
print("  ÉTAPE 3 — Scan des quadruplets (n=4) — tester si nécessaire")
print("═"*100)

candidates_n4 = []
TOL_n4 = 0.07
# n=4 est trop massive, restreindre aux nœuds petits seulement
SMALL_KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2"]
for cfg_tup in itertools.combinations_with_replacement(SMALL_KNOTS, 4):
    cfg = list(cfg_tup)
    m = omega_link(cfg)
    err = abs(m - M_K_AVG)/M_K_AVG
    if err < TOL_n4:
        candidates_n4.append((cfg, m, err))

candidates_n4.sort(key=lambda x: x[2])
print(f"  Trouvé {len(candidates_n4)} configurations à n=4 dans ±{TOL_n4*100:.0f}%")
print(f"  {'config':<25} {'m_pred':>10} {'Δ K_avg':>9}")
print("  " + "-" * 55)
for cfg, m, err in candidates_n4[:10]:
    delta = (m - M_K_AVG)/M_K_AVG*100
    print(f"  {'+'.join(cfg):<25} {m:>10.3f} {delta:>+8.2f}%")


print("\n" + "═"*100)
print("  ÉTAPE 4 — Analyse physique des meilleurs candidats")
print("═"*100)

# Combiner tous les candidats triés
all_cands = []
for cfg, m, err in candidates_n2[:5]:
    all_cands.append((cfg, m, err, "n=2"))
for cfg, m, err in candidates_n3[:10]:
    all_cands.append((cfg, m, err, "n=3"))
for cfg, m, err in candidates_n4[:5]:
    all_cands.append((cfg, m, err, "n=4"))
all_cands.sort(key=lambda x: x[2])

print(f"\n  Top 15 candidats globaux :")
print(f"  {'rang':<5} {'n':<4} {'config':<22} {'m_pred':>10} {'Δ':>8} {'cr':>5} "
      f"{'b':>5} {'spin_topo':>10}")
print("  " + "-" * 85)
for rank, (cfg, m, err, n_lab) in enumerate(all_cands[:15], 1):
    delta = (m - M_K_AVG)/M_K_AVG*100
    cr_tot = sum(TOPO[K]["cr"] for K in cfg)
    b_max = max(TOPO[K]["b"] for K in cfg)
    spins = [TOPO[K]["spin"] for K in cfg]
    spin_str = ",".join(f"{s:.1f}" if isinstance(s, float) else str(s) for s in spins)
    print(f"  {rank:<5} {n_lab:<4} {'+'.join(cfg):<22} {m:>10.3f} {delta:>+7.2f}% "
          f"{cr_tot:>5} {b_max:>5} {spin_str:>10}")


# ───────── ÉTAPE 5 — Vérifier la cohérence avec autres particules ─────────
print("\n" + "═"*100)
print("  ÉTAPE 5 — Vérifier que les nouvelles assignations sont cohérentes")
print("═"*100)
print("""
  Le kaon est dans le multiplet pseudoscalaire (J=0) avec π, η.
  La progression de masse π(140) → K(495) → η(548) → η'(958)
  est due au MIXING U(1)_A et à la massiveness du quark étrange.

  Notre V5 donne :
    π⁰ = 3_1+3_1+5_2 → 144.78 MeV (≈ 140 PDG)
    η  = 3_1+3_1+6_1 → 573.10 MeV (≈ 548 PDG)
  Donc K se positionne entre π et η.

  Si on suit la logique 3_1+3_1+X où X croît :
    π : X = 5_2 (U_κU_τ = 9)    → 144
    K : X = ?  (U_κU_τ ≈ ?)     → 495
    η : X = 6_1 (U_κU_τ = 116)  → 573

  Pour K, X devrait avoir U_κU_τ ∈ [50, 90] (interpolation).
""")

# Identifier les candidats à 3 composantes avec deux 3_1
n3_with_two_31 = [c for c in candidates_n3 if Counter(c[0])["3_1"] == 2]
if n3_with_two_31:
    print(f"  Candidats 3_1+3_1+X (cohérents avec π/η) :")
    print(f"  {'config':<22} {'m_pred':>10} {'Δ':>8} {'U_κU_τ(X)':>11}")
    for cfg, m, err in n3_with_two_31[:10]:
        delta = (m - M_K_AVG)/M_K_AVG*100
        X = [K for K in cfg if K != "3_1"][0]
        U_X = U_KT[X][0]*U_KT[X][1]
        print(f"  {'+'.join(cfg):<22} {m:>10.3f} {delta:>+7.2f}% {U_X:>11.2f}")

print("\n  Si AUCUN candidat 3_1+3_1+X ne marche → catalogue insuffisant")
print("  (besoin de nœud X avec U_κU_τ dans la gamme intermédiaire).")
print("  Le pattern de masse est :")
print(f"    π : U_κU_τ(5_2) = {U_KT['5_2'][0]*U_KT['5_2'][1]:.1f}  → 144.8 MeV")
print(f"    K : ?           → 495 MeV (besoin de U_κU_τ ≈ 50-90)")
print(f"    η : U_κU_τ(6_1) = {U_KT['6_1'][0]*U_KT['6_1'][1]:.1f}  → 573.1 MeV")
print(f"\n  Saut entre 5_2 (9) et 6_1 (116) : facteur 13 — c'est un GROS GAP.")
print(f"  Combler ce gap demanderait un nœud entre 5_2 et 6_1.")
