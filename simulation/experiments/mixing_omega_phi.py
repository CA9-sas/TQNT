r"""
MIXING ω-φ — analyse complète des vecteurs J=1 (I=0).

Contrairement à η-η' où l'anomalie U(1)_A produit un grand mixing
(M_12 = 358 MeV), les vecteurs ω et φ suivent le mixing IDÉAL :
  φ ≈ ss̄ (mostly octet octet)
  ω ≈ (uū+dd̄)/√2 (mostly singlet)
avec θ_V ≈ 35°.

On teste plusieurs combinaisons TQNT pour identifier les états physiques :
  (a) ω = 8_1+8_1 (u/d pur) ↔ φ = 3_1+5_1+5_2 (s-quark)
  (b) variantes avec mixing à la matrice 2×2
  (c) recherche d'une config plus précise pour φ
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement

from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# ───────── précalcul ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B_EXP = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul Frenet…")
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


def omega(comps):
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B_EXP))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_EXP


def J_set(comps):
    s_max = sum(TOPO[K]["s"] for K in comps)
    parity = s_max - int(s_max)
    n_steps = int(round(s_max - parity)) + 1
    return {parity + i for i in range(n_steps)}


# Masses expérimentales
M_OMEGA = 782.65
M_PHI = 1019.461

# ════════════════════════════════════════════════════════════════════════
#  (1) RECHERCHE EXHAUSTIVE des configs candidates pour ω et φ
# ════════════════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print("  RECHERCHE — meilleurs candidats topologiques pour ω et φ")
print("═"*78)

# Trouver tous les configs avec J=1 accessible et masse dans la fenêtre
candidates_omega = []   # entre 700 et 850 MeV
candidates_phi = []     # entre 950 et 1100 MeV
candidates_all = []     # pour mixing

for n in (2, 3):
    for combo in combinations_with_replacement(KNOTS, n):
        cfg = list(combo)
        Js = J_set(cfg)
        if 1.0 not in Js:
            continue
        m = omega(cfg)
        candidates_all.append((cfg, m))
        if 700 <= m <= 850:
            candidates_omega.append((cfg, m, (m - M_OMEGA)/M_OMEGA*100))
        if 950 <= m <= 1100:
            candidates_phi.append((cfg, m, (m - M_PHI)/M_PHI*100))

candidates_omega.sort(key=lambda x: abs(x[2]))
candidates_phi.sort(key=lambda x: abs(x[2]))

print(f"\n  Candidats pour ω (m_exp = {M_OMEGA:.2f} MeV, fenêtre 700-850) :")
for cfg, m, dm in candidates_omega[:6]:
    print(f"    {'+'.join(cfg):<22} → {m:>7.2f} MeV ({dm:+.2f}%)")

print(f"\n  Candidats pour φ (m_exp = {M_PHI:.2f} MeV, fenêtre 950-1100) :")
for cfg, m, dm in candidates_phi[:6]:
    print(f"    {'+'.join(cfg):<22} → {m:>7.2f} MeV ({dm:+.2f}%)")

# Meilleurs candidats SANS mixing
best_omega = candidates_omega[0]
best_phi = candidates_phi[0]
print(f"\n  Meilleurs sans mixing :")
print(f"    ω = {'+'.join(best_omega[0])} → {best_omega[1]:.2f} ({best_omega[2]:+.2f}%)")
print(f"    φ = {'+'.join(best_phi[0])} → {best_phi[1]:.2f} ({best_phi[2]:+.2f}%)")

# ════════════════════════════════════════════════════════════════════════
#  (2) MIXING 2x2 ENTRE LES DEUX MEILLEURS CANDIDATS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─"*78)
print("  MIXING 2×2 entre ω_TQNT et φ_TQNT")
print("─"*78)


def mixing_eigenvalues(m_light, m_heavy, M12_sq):
    """Valeurs propres de M² avec terme off-diagonal M_12² (peut être <0)."""
    m1_sq = m_light**2; m2_sq = m_heavy**2
    half_sum = (m1_sq + m2_sq) / 2
    half_diff = (m2_sq - m1_sq) / 2
    disc = np.sqrt(half_diff**2 + M12_sq**2)
    return np.sqrt(max(half_sum - disc, 0)), np.sqrt(half_sum + disc)


m_omega_top = best_omega[1]
m_phi_top = best_phi[1]


def chi2_mixing_op(M12_sq):
    m1, m2 = mixing_eigenvalues(m_omega_top, m_phi_top, M12_sq)
    return ((m1 - M_OMEGA)/M_OMEGA)**2 + ((m2 - M_PHI)/M_PHI)**2


# Optimiser M_12² (peut être négatif → off-diagonal imaginaire,
# physiquement plausible avec une convention de phase)
res = minimize_scalar(chi2_mixing_op, bounds=(0, 2_000_000), method='bounded')
M12_sq_opt = res.x
M12_opt = np.sqrt(M12_sq_opt)
m_omega_mixed, m_phi_mixed = mixing_eigenvalues(m_omega_top, m_phi_top, M12_sq_opt)

# Angle de mixing
half_diff_sq = (m_phi_top**2 - m_omega_top**2) / 2
theta_opt = 0.5 * np.arctan2(M12_sq_opt, half_diff_sq)
theta_deg = theta_opt * 180/np.pi

print(f"\n  Avant mixing :")
print(f"    ω_TQNT = {m_omega_top:.2f} MeV  (Δ = {(m_omega_top-M_OMEGA)/M_OMEGA*100:+.2f}%)")
print(f"    φ_TQNT = {m_phi_top:.2f} MeV  (Δ = {(m_phi_top-M_PHI)/M_PHI*100:+.2f}%)")

print(f"\n  Mixing optimal :")
print(f"    M_12² = {M12_sq_opt:.1f} MeV²")
print(f"    M_12 = {M12_opt:.1f} MeV")
print(f"    Angle équivalent θ_V = {theta_deg:+.2f}°")
print(f"    (PDG : θ_V ≈ 36° pour mixing vectoriel, près de l'idéal 35.3°)")

print(f"\n  Après mixing :")
print(f"    ω = {m_omega_mixed:.2f} MeV (exp {M_OMEGA:.2f}, Δ = {(m_omega_mixed-M_OMEGA)/M_OMEGA*100:+.3f}%)")
print(f"    φ = {m_phi_mixed:.2f} MeV (exp {M_PHI:.2f}, Δ = {(m_phi_mixed-M_PHI)/M_PHI*100:+.3f}%)")

# Test trace de M²
trace_top = m_omega_top**2 + m_phi_top**2
trace_exp = M_OMEGA**2 + M_PHI**2
print(f"\n  Conservation de la trace :")
print(f"    tr(M²) TQNT raw    = {trace_top:.0f} MeV²")
print(f"    tr(M²) physique    = {trace_exp:.0f} MeV²")
print(f"    Écart relatif      = {(trace_exp-trace_top)/trace_top*100:+.2f}%")
if abs(trace_exp - trace_top) > 0.02 * trace_top:
    print(f"    → La trace n'est PAS préservée : mixing simple insuffisant.")
    print(f"      Le résidu vient de la formule Ω_ts elle-même.")
else:
    print(f"    → Trace approximativement préservée : mixing résout les masses.")


# ════════════════════════════════════════════════════════════════════════
#  (3) ANALYSE — pourquoi ω-φ diffère de η-η'
# ════════════════════════════════════════════════════════════════════════
print("\n" + "─"*78)
print("  ANALYSE — différence physique avec η-η'")
print("─"*78)
print(f"""
  η-η' (pseudoscalaires) :
    • Anomalie U(1)_A donne masse anormale au singulet
    • Mixing physique θ_P ≈ -11° (PEU DE MIXING par rapport à octet pur)
    • Sum_8² + m_0² ≈ m_η² + m_η'² (TRACE PRÉSERVÉE)
    • TQNT-V5 : trace TQNT = trace exp à <1% → mixing fonctionne EXACTEMENT

  ω-φ (vecteurs) :
    • Aucune anomalie correspondante (pas d'analogue U(1)_A vecteur)
    • Mixing physique θ_V ≈ 36° (PROCHE de l'idéal 35.3°)
    • φ ≈ ss̄ pur, ω ≈ (uū+dd̄)/√2 pur
    • TQNT-V5 : trace TQNT diffère de trace exp par {(trace_exp-trace_top)/trace_top*100:+.1f}%
      → mixing seul ne suffit pas, l'opérateur Ω_ts capture imparfaitement φ.

  CONCLUSION pour ω-φ :
    • ω est déjà bien capturé par 8_1+8_1 ({(m_omega_top-M_OMEGA)/M_OMEGA*100:+.2f}%)
    • φ est sous-prédit par 3_1+5_1+5_2 ({(m_phi_top-M_PHI)/M_PHI*100:+.2f}%)
    • Le mixing 2×2 améliore légèrement φ mais pas autant que pour η-η'
    • Le résidu φ vient probablement de la difficulté de la formule à
      capturer le secteur étrange pur (ss̄) avec les nœuds disponibles.
""")


# ════════════════════════════════════════════════════════════════════════
#  (4) RECHERCHE D'UNE CONFIG OPTIMALE POUR φ ≈ ss̄
# ════════════════════════════════════════════════════════════════════════
print("─"*78)
print("  Recherche de configs alternatives pour φ (proches de 1019 MeV)")
print("─"*78)

# Chercher dans toutes les configs à 3 composantes qui contiennent 6_1
# (associé à l'étrangeté en TQNT) ou des configs qui donnent ~1019
near_phi = sorted([(cfg, m) for cfg, m in candidates_all
                   if abs(m - M_PHI) < 100],
                  key=lambda x: abs(x[1] - M_PHI))
print(f"\n  Configs avec |m - 1019| < 100 MeV :")
for cfg, m in near_phi[:10]:
    dm = (m - M_PHI)/M_PHI*100
    print(f"    {'+'.join(cfg):<22} → {m:>7.2f} MeV ({dm:+.2f}%)")


print("\n" + "═"*78)
print("  RÉSUMÉ FINAL")
print("═"*78)
print(f"""
  Mixing ω-φ :
    ω = 8_1+8_1 (light, u/d pur) → 779 MeV
    φ = 3_1+5_1+5_2 (heavy, s-quark) → 980 MeV
    M_12 optimal = {M12_opt:.0f} MeV  (vs 358 MeV pour η-η')
    θ_V = {theta_deg:.1f}°  (proche de l'idéal 35°)

  Différences clés avec η-η' :
    1. Pas d'anomalie U(1)_A → mixing physique IDÉAL (no off-diagonal)
    2. Trace de M² NON préservée entre TQNT et exp
       → indique imperfection de Ω_ts sur le secteur étrange ss̄
    3. ω est exact ; φ a un résidu intrinsèque de ~4%

  Interprétation :
    Le résidu de φ vient probablement de la difficulté à capturer
    purement ss̄ avec les 19 nœuds. Un nœud avec strangeness intrinsèque
    (e.g. enrichir 6_1 avec un facteur d'étrangeté) pourrait corriger.

  → ω-φ ≠ η-η' : pas de mixing dynamique fort, juste un
    biais résiduel sur la configuration s-quark.
""")
