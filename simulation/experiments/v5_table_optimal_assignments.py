r"""
RECHERCHE des meilleures assignations topologiques pour chaque particule
avec la formule Ω_ts STRICTE (zéro paramètre libre).

Pour chaque particule, on scanne :
  • Toutes les paires (K1, K2) du catalogue (19 nœuds)
  • Tous les triplets (K1, K2, K3) avec K_i identiques autorisés

Critère : minimiser |m_Ω_ts - m_exp| / m_exp

On rapporte aussi l'assignation V5 publiée pour comparaison.
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index

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
               "b": braid_index(K), "det": jones_determinant(pj)}
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


# Précalcul des m_pred pour toutes les configurations
print("Calcul des Ω_ts pour 190 paires + 1330 triplets + 19 singletons…")
all_configs = []
for K in KNOTS:
    all_configs.append([K])
for K1, K2 in itertools.combinations_with_replacement(KNOTS, 2):
    all_configs.append([K1, K2])
for K1, K2, K3 in itertools.combinations_with_replacement(KNOTS, 3):
    all_configs.append([K1, K2, K3])
print(f"  Total configurations à évaluer : {len(all_configs)}")

# Calculer m pour toutes
print("Calcul m_Ω_ts pour toutes les configurations…")
config_masses = []
for cfg in all_configs:
    m = omega_link(cfg)
    config_masses.append((cfg, m))


def find_best_assignment(m_exp, n_components_max=3, m_window=None):
    """Trouve l'assignation minimisant |Δ| pour m_exp."""
    candidates = []
    for cfg, m in config_masses:
        if len(cfg) > n_components_max: continue
        if m_window and not (m_window[0] < m < m_window[1]): continue
        err = abs(m - m_exp)/m_exp
        candidates.append((cfg, m, err))
    candidates.sort(key=lambda x: x[2])
    return candidates[:5]  # top 5


# ───────── Particules de la table V5 ─────────
PARTICULES = [
    ("e⁻",   0.51,    ["3_1"]),
    ("μ⁻",   105.66,  ["3_1","6_1"]),
    ("τ⁻",   1776.86, ["3_1","5_1","7_1"]),
    ("s",    93.0,    ["5_2","5_2"]),
    ("c",    1275.0,  ["7_3","7_7"]),
    ("b",    4180.0,  ["4_1","5_2","7_6"]),
    ("π⁰",   134.98,  ["8_1"]),
    ("π±",   139.57,  ["8_1"]),
    ("K±",   493.68,  ["4_1","4_1","5_2"]),
    ("K⁰",   497.61,  ["4_1","4_1","5_2"]),
    ("η",    547.86,  ["3_1","4_1","5_1"]),
    ("η'",   957.78,  ["5_2","5_2","6_1"]),
    ("ρ/ω",  775.0,   ["5_2","7_7"]),
    ("φ",    1019.46, ["3_1","4_1","6_2"]),
    ("J/ψ",  3096.9,  ["4_1","5_2","9_1"]),
    ("Υ",    9460.0,  ["3_1","7_4","9_1"]),
    ("p",    938.27,  ["3_1","7_2","7_2"]),
    ("n",    939.57,  ["3_1","7_2","7_2"]),
    ("Λ",    1115.68, ["4_1","7_4"]),
    ("Σ+",   1189.37, ["6_2","7_2"]),
    ("Σ0",   1192.64, ["6_2","7_2"]),
    ("Σ-",   1197.45, ["6_2","7_2"]),
    ("Ξ0",   1314.86, ["4_1","4_1","7_2"]),
    ("Ξ-",   1321.71, ["4_1","4_1","7_2"]),
    ("Δ",    1232.0,  ["3_1","5_2","6_2"]),
    ("Σ*",   1383.0,  ["5_2","6_2","7_2"]),
    ("Ξ*",   1532.0,  ["4_1","5_2","7_2"]),
    ("Ω-",   1672.0,  ["5_2","5_2","7_2"]),
    ("D0",   1864.84, ["6_2","7_7"]),
    ("D+",   1869.66, ["6_2","7_7"]),
    ("Λc",   2286.46, ["6_3","8_1"]),
    ("Σc",   2453.0,  ["4_1","5_1","7_1"]),
    ("Ωc",   2695.0,  ["3_1","6_2","6_2"]),
    ("B",    5279.0,  ["4_1","10_1"]),
    ("Λb",   5619.6,  ["5_2","5_2","8_2"]),
]


print("\n" + "═"*100)
print("  ASSIGNATIONS OPTIMALES vs V5 PUBLIÉ — Ω_ts strict")
print("═"*100)
print(f"  {'nm':<6} {'m_exp':>9} {'V5 publié':<22} {'m_V5_Ω_ts':>11} {'Δ V5':>7}"
      f"  {'Meilleure assign. (Ω_ts)':<26} {'m_best':>10} {'Δ best':>7}")
print("  " + "-" * 110)

records = []
for nm, m_exp, cfg_v5 in PARTICULES:
    m_v5_omega = omega_link(cfg_v5)
    d_v5 = (m_v5_omega - m_exp)/m_exp*100
    # Borne n_components ≤ 3
    candidates = find_best_assignment(m_exp, n_components_max=3,
                                       m_window=(m_exp*0.5, m_exp*2.0))
    if candidates:
        best_cfg, best_m, best_err = candidates[0]
        d_best = (best_m - m_exp)/m_exp*100
    else:
        # fallback: élargir
        candidates = find_best_assignment(m_exp, n_components_max=3)
        best_cfg, best_m, best_err = candidates[0]
        d_best = (best_m - m_exp)/m_exp*100
    same = (sorted(best_cfg) == sorted(cfg_v5))
    sign_same = "=" if same else "≠"
    print(f"  {nm:<6} {m_exp:>9.2f} {'+'.join(cfg_v5):<22} {m_v5_omega:>11.2f} "
          f"{d_v5:>+6.2f}%  {'+'.join(best_cfg):<26} {best_m:>10.2f} "
          f"{d_best:>+6.2f}% {sign_same}")
    records.append({
        "nm": nm, "m_exp": m_exp,
        "cfg_v5": cfg_v5, "m_v5_omega": m_v5_omega, "d_v5": d_v5,
        "cfg_best": best_cfg, "m_best": best_m, "d_best": d_best,
        "same": same,
    })


# ───────── Statistiques ─────────
print("\n" + "═"*100)
print("  STATISTIQUES")
print("═"*100)

d_v5_omega = np.array([r["d_v5"] for r in records])
d_best = np.array([r["d_best"] for r in records])
rms_v5_omega = float(np.sqrt(np.mean(d_v5_omega**2)))
rms_best = float(np.sqrt(np.mean(d_best**2)))
n_same = sum(1 for r in records if r["same"])
print(f"\n  N = {len(records)} particules")
print(f"  RMS V5 (assignations publiées avec Ω_ts) : {rms_v5_omega:.2f}%")
print(f"  RMS optimal Ω_ts                          : {rms_best:.2f}%")
print(f"  Assignations préservées (V5 = optimal)    : {n_same}/{len(records)}")
print(f"  Assignations à réviser                    : {len(records)-n_same}/{len(records)}")
