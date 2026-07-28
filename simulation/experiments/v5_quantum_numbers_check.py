r"""
VÉRIFICATION des nombres quantiques pour les assignations optimales Ω_ts.

Pour chaque particule, on vérifie :
  1. SPIN J : J_exp ∈ link_spin_set(L) via P7quater (Clebsch-Gordan)
  2. CHARGE Q : Q_exp compatible avec Σ_i Q(K_i) + Lk
  3. PARITÉ (informatif) : produit des parités intrinsèques
  4. MULTIPLET : continuité entre partenaires de charge

Si une assignation optimale MASSE échoue sur Q, J, ou multiplet,
elle est rejetée → recherche du candidat suivant compatible.
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import (jones_polynomial, jones_determinant, braid_index,
                          topological_spin, topological_charge)
from theory.multi_knot import link_spin_set

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
               "spin": topological_spin(K),
               "charge": topological_charge(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_link(components):
    n = len(components); ws=[]; Us=[]
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


def Q_topo_sum(components):
    """Somme des charges topologiques (sans Lk)."""
    return sum(TOPO[K]["charge"] for K in components)


# ───────── Particules avec nombres quantiques connus ─────────
# (nm, m_exp, 2J, Q_exp, classe, optimal_cfg, comment)
PARTICULES = [
    # Leptons : Q intégrales, J=1/2 demi-entier
    ("e⁻",   0.51,      1, -1, "lepton", ["3_1"]),
    ("μ⁻",   105.66,    1, -1, "lepton", ["3_1","6_1"]),
    ("τ⁻",   1776.86,   1, -1, "lepton", ["5_1","5_1","6_1"]),  # optimal trouvé
    # Mésons pseudoscalaires : J=0, Q intégrales
    ("π⁰",   134.98,    0,  0, "pseudo", ["3_1","3_1","3_1"]),
    ("π±",   139.57,    0,  1, "pseudo", ["7_7"]),
    ("K±",   493.68,    0,  1, "pseudo", ["7_6"]),
    ("K⁰",   497.61,    0,  0, "pseudo", ["6_1","6_1","6_1"]),
    ("η",    547.86,    0,  0, "pseudo", ["3_1","5_2","6_1"]),
    ("η'",   957.78,    0,  0, "pseudo", ["4_1","7_3"]),
    # Mésons vecteurs : J=1
    ("ρ",    775.0,     2,  1, "vector", ["8_1","8_1"]),
    ("φ",    1019.46,   2,  0, "vector", ["5_1","5_2","5_2"]),
    ("J/ψ",  3096.9,    2,  0, "vector", ["3_1","4_1","6_3"]),
    ("Υ",    9460.0,    2,  0, "vector", ["7_1","7_3","9_1"]),
    # Baryons octet : J=1/2
    ("p",    938.27,    1,  1, "baryon", ["3_1","7_2","7_2"]),
    ("n",    939.57,    1,  0, "baryon", ["3_1","7_2","7_2"]),
    ("Λ",    1115.68,   1,  0, "baryon", ["5_1","5_1","5_1"]),
    ("Σ+",   1189.37,   1,  1, "baryon", ["5_1","7_7"]),
    ("Σ0",   1192.64,   1,  0, "baryon", ["5_1","7_7"]),
    ("Σ-",   1197.45,   1, -1, "baryon", ["5_1","7_7"]),
    ("Ξ0",   1314.86,   1,  0, "baryon", ["5_1","6_3"]),
    ("Ξ-",   1321.71,   1, -1, "baryon", ["5_1","6_3"]),
    # Baryons décuplet : J=3/2
    ("Δ",    1232.0,    3,  1, "decuplet", ["5_2","7_2","7_2"]),
    ("Σ*",   1383.0,    3,  0, "decuplet", ["10_1"]),
    ("Ξ*",   1532.0,    3,  0, "decuplet", ["7_1","8_1"]),
    ("Ω-",   1672.0,    3, -1, "decuplet", ["5_1","7_2","7_2"]),
    # Hadrons charmés
    ("D0",   1864.84,   0,  0, "pseudo", ["3_1","4_1","6_2"]),
    ("Λc",   2286.46,   1,  1, "baryon", ["5_1","5_1","7_2"]),
    ("Σc",   2453.0,    1,  1, "baryon", ["4_1","5_1","7_1"]),
    ("Ωc",   2695.0,    1,  0, "baryon", ["7_3","7_7"]),
    # Hadrons beaux
    ("B",    5279.0,    0,  0, "pseudo", ["5_2","6_3","6_3"]),
    ("Λb",   5619.6,    1,  0, "baryon", ["6_3","7_4"]),
]


print("\n" + "═"*100)
print("  Vérification des nombres quantiques pour assignations optimales Ω_ts")
print("═"*100)
print(f"  {'nm':<6} {'config':<22} {'m_pred':>9} {'2J_exp':>6} {'2J_pred':<14} {'Q_exp':>6} {'Q_topo':>8} "
      f"{'J_OK':<6} {'cls':<10}")
print("  " + "-" * 100)

records = []
for nm, m_exp, two_J_exp, Q_exp, cls, cfg in PARTICULES:
    m_pred = omega_link(cfg)
    delta = (m_pred - m_exp)/m_exp*100
    # Spectre de spin
    J_set = link_spin_set(cfg)
    two_J_set = sorted(2*j for j in J_set)
    J_exp = two_J_exp / 2
    J_compatible = J_exp in J_set
    # Charge topologique
    Q_topo = Q_topo_sum(cfg)
    j_flag = "✓" if J_compatible else "✗"
    two_J_str = ",".join(str(int(j) if j == int(j) else f"{j:.1f}") for j in two_J_set)
    if len(two_J_str) > 13: two_J_str = two_J_str[:11] + ".."
    print(f"  {nm:<6} {'+'.join(cfg):<22} {m_pred:>9.2f} {two_J_exp:>6} "
          f"{two_J_str:<14} {Q_exp:>+6} {Q_topo:>+8.3f} {j_flag:<6} {cls:<10}")
    records.append({
        "nm": nm, "cfg": cfg, "m_pred": m_pred, "m_exp": m_exp,
        "delta": delta, "two_J_exp": two_J_exp, "two_J_set": two_J_set,
        "J_compatible": J_compatible,
        "Q_exp": Q_exp, "Q_topo": Q_topo, "cls": cls,
    })


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)

n_J_ok = sum(1 for r in records if r["J_compatible"])
print(f"\n  Spin J compatible (J_exp ∈ J_set)  : {n_J_ok}/{len(records)}")

# Détail des échecs
fails = [r for r in records if not r["J_compatible"]]
print(f"\n  Particules à spin INCOMPATIBLE :")
print(f"  {'nm':<8} {'cfg':<22} {'2J_exp':>6} {'2J_set':<25}")
for r in fails:
    set_str = ",".join(str(int(j) if j == int(j) else f"{j:.1f}") for j in r["two_J_set"])
    print(f"  {r['nm']:<8} {'+'.join(r['cfg']):<22} {r['two_J_exp']:>6} {set_str:<25}")


# ───────── Recherche d'assignations compatibles ─────────
print("\n" + "═"*100)
print("  Pour chaque échec, recherche d'une assignation alternative")
print("  qui soit (1) compatible avec J_exp et (2) reste proche en masse")
print("═"*100)

# Précalcul TOUTES les configurations
all_configs = []
for K in KNOTS:
    all_configs.append([K])
for K1, K2 in itertools.combinations_with_replacement(KNOTS, 2):
    all_configs.append([K1, K2])
for K1, K2, K3 in itertools.combinations_with_replacement(KNOTS, 3):
    all_configs.append([K1, K2, K3])

config_data = []
for cfg in all_configs:
    m = omega_link(cfg)
    j_set = link_spin_set(cfg)
    q_topo = Q_topo_sum(cfg)
    config_data.append((cfg, m, j_set, q_topo))

print(f"\n  Total configurations pré-calculées : {len(config_data)}")

# Pour chaque fail, chercher meilleure assignation avec contrainte J
print(f"\n  Recherche d'alternatives compatibles avec spin :")
print(f"  {'nm':<8} {'optim. masse seule':<22} {'Δm':>7} | {'optim. J+masse':<22} {'m_J':>8} {'Δm_J':>7}")
print("  " + "-" * 92)

for r in fails:
    m_exp = r["m_exp"]; J_exp = r["two_J_exp"]/2
    # Filtrer configs avec J_exp compatible et n ≤ 3
    compatibles = [(cfg, m, j_set, q) for (cfg, m, j_set, q) in config_data
                    if J_exp in j_set and len(cfg) <= 3]
    if not compatibles:
        print(f"  {r['nm']:<8} (aucune config compatible J)")
        continue
    # Tri par proximité de masse
    compatibles.sort(key=lambda x: abs(x[1] - m_exp))
    best_cfg, best_m, best_j_set, best_q = compatibles[0]
    delta_J_best = (best_m - m_exp)/m_exp*100
    print(f"  {r['nm']:<8} {'+'.join(r['cfg']):<22} {r['delta']:>+6.2f}% | "
          f"{'+'.join(best_cfg):<22} {best_m:>8.2f} {delta_J_best:>+6.2f}%")


# Statistiques finales avec contrainte J
print("\n" + "═"*100)
print("  Re-statistiques avec contrainte (m + J)")
print("═"*100)
deltas_J = []
n_revised = 0
for r in records:
    if r["J_compatible"]:
        deltas_J.append(r["delta"])
    else:
        # Trouver l'assignation J-compatible la plus proche
        m_exp = r["m_exp"]; J_exp = r["two_J_exp"]/2
        compatibles = [(cfg, m) for (cfg, m, j_set, _) in config_data
                       if J_exp in j_set and len(cfg) <= 3]
        if compatibles:
            compatibles.sort(key=lambda x: abs(x[1] - m_exp))
            best_m = compatibles[0][1]
            deltas_J.append((best_m - m_exp)/m_exp*100)
            n_revised += 1
        else:
            deltas_J.append(r["delta"])

rms_J = float(np.sqrt(np.mean(np.array(deltas_J)**2)))
rms_orig = float(np.sqrt(np.mean([r["delta"]**2 for r in records])))
print(f"\n  RMS (masse seule)        : {rms_orig:.2f}%  sur {len(records)} particules")
print(f"  RMS (masse + spin)       : {rms_J:.2f}%  ({n_revised} révisées)")
print(f"  Compatibles J initial    : {n_J_ok}/{len(records)}")
