r"""
RECALCUL STRICT du tableau de masses V5 avec Ω_ts (zéro paramètre libre).

Ω_ts utilise les paramètres analytiques fixes :
  a = √(7/(8π))      (exposant A_FREN)
  p = q = 3/4         (P_EXP, Q_EXP)
  x = -√6             (X_EXP pour Φ_n)
  β = 33/20           (BB pour Φ_p)
  ρ_0 = 10 - 3/(8π)   (RHO_0)

Pas de f(2), f(3), pas de calibration par particule.
α_0 = 0,8682 × m_e/3 (seule calibration : m_e).

Pour chaque particule de la table V5 publiée, on calcule m_pred
avec ces paramètres et on compare aux valeurs publiées et
expérimentales.
"""
from __future__ import annotations
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

print("Précalcul des invariants des 19 nœuds…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
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


# ───────── Tableau V5 publié (assignations exactes) ─────────
# Format : (nom, config, m_TQNT_publié, m_exp, Δ%_publié)
TABLE_V5 = [
    # Leptons
    ("e⁻",   ["3_1"],                0.51,    0.51,   0.0),
    ("μ⁻",   ["3_1","6_1"],         102.0,    106.0,  3.2),
    ("τ⁻",   ["3_1","5_1","7_1"],  1781.0,  1777.0,   0.2),
    # Quarks (mention spéciale, embedding/entrelacs)
    ("s",    ["5_2","5_2"],          93.0,    93.0,   0.7),
    ("c",    ["7_3","7_7"],        1275.0,  1270.0,   0.4),
    ("b",    ["4_1","5_2","7_6"],  4180.0,  4180.0,   0.0),
    ("t",    ["7_4","7_6","8_3"], 174174.0, 172760.0, 0.8),
    # Mésons pseudoscalaires
    ("π⁰",   ["8_1"],               134.0,    135.0,  0.8),
    ("π±",   ["8_1"],               134.0,    140.0,  4.3),
    ("K±",   ["4_1","4_1","5_2"],   494.0,    494.0,  0.0),
    ("K⁰",   ["4_1","4_1","5_2"],   494.0,    498.0,  0.8),
    ("η",    ["3_1","4_1","5_1"],   545.0,    548.0,  0.5),
    ("η'",   ["5_2","5_2","6_1"],   955.0,    958.0,  0.3),
    # Mésons vecteurs
    ("ρ/ω",  ["5_2","7_7"],         782.0,    775.0,  0.1),
    ("φ",    ["3_1","4_1","6_2"],  1015.0,   1019.0,  0.5),
    ("J/ψ",  ["4_1","5_2","9_1"],  3097.0,   3097.0,  0.0),
    ("Υ",    ["3_1","7_4","9_1"],  9428.0,   9460.0,  0.3),
    # Baryons octet
    ("p",    ["3_1","7_2","7_2"],   940.0,    938.0,  0.2),
    ("n",    ["3_1","7_2","7_2"],   941.0,    940.0,  0.2),
    ("Λ",    ["4_1","7_4"],        1123.0,   1116.0,  0.6),
    ("Σ+",   ["6_2","7_2"],        1193.0,   1189.0,  0.0),
    ("Σ0",   ["6_2","7_2"],        1193.0,   1192.0,  0.0),
    ("Σ-",   ["6_2","7_2"],        1199.0,   1197.0,  0.1),
    ("Ξ0",   ["4_1","4_1","7_2"],  1307.0,   1315.0,  0.6),
    ("Ξ-",   ["4_1","4_1","7_2"],  1307.0,   1322.0,  1.1),
    # Baryons décuplet
    ("Δ",    ["3_1","5_2","6_2"],  1232.0,   1232.0,  0.0),
    ("Σ*",   ["5_2","6_2","7_2"],  1384.0,   1383.0,  0.1),
    ("Ξ*",   ["4_1","5_2","7_2"],  1532.0,   1532.0,  0.0),
    ("Ω-",   ["5_2","5_2","7_2"],  1672.0,   1672.0,  0.0),
    # Hadrons charmés
    ("D0",   ["6_2","7_7"],        1869.0,   1865.0,  0.0),
    ("D+",   ["6_2","7_7"],        1869.0,   1870.0,  0.0),
    ("Λc",   ["6_3","8_1"],        2285.0,   2286.0,  0.1),
    ("Σc",   ["4_1","5_1","7_1"],  2453.0,   2453.0,  0.0),
    ("Ωc",   ["3_1","6_2","6_2"],  2697.0,   2695.0,  0.1),
    # Hadrons beaux
    ("B",    ["4_1","10_1"],       5277.0,   5279.0,  0.0),
    ("Λb",   ["5_2","5_2","8_2"],  5613.0,   5620.0,  0.1),
    # Bosons EW
    ("W±",   ["4_1","7_3","8_3"], 80867.0,  80377.0,  0.6),
    ("Z0",   ["5_1","7_2","8_3"], 91489.0,  91188.0,  0.3),
    ("H",    ["6_1","7_4","8_3"], 125499.0, 125250.0, 0.2),
]


# ───────── Calcul strict de Ω_ts pour chaque assignation ─────────
print("\n" + "═"*100)
print("  RECALCUL STRICT — Ω_ts (zéro paramètre libre) sur table V5")
print("═"*100)
print(f"  {'Particule':<8} {'Config':<22} {'m_V5_publié':>11} {'m_Ω_ts':>11} "
      f"{'m_exp':>10} {'Δ_publié':>9} {'Δ_Ω_ts':>9} {'écart':>9}")
print("  " + "-" * 100)

results = []
for nm, cfg, m_published, m_exp, delta_pub in TABLE_V5:
    m_omega = omega_link(cfg)
    d_omega = (m_omega - m_exp)/m_exp*100
    ratio_omega_pub = m_omega / m_published
    consistency = abs(m_omega - m_published) / m_published * 100
    flag = "✓" if consistency < 5 else ("·" if consistency < 30 else "✗")
    print(f"  {nm:<8} {'+'.join(cfg):<22} {m_published:>11.2f} "
          f"{m_omega:>11.2f} {m_exp:>10.2f} {delta_pub:>+8.2f}% "
          f"{d_omega:>+8.2f}% {consistency:>7.1f}% {flag}")
    results.append({
        "nm": nm, "cfg": cfg, "m_published": m_published, "m_exp": m_exp,
        "m_omega": m_omega, "delta_pub": delta_pub, "delta_omega": d_omega,
        "consistency": consistency,
    })


# ───────── Statistiques globales ─────────
print("\n" + "═"*100)
print("  STATISTIQUES GLOBALES")
print("═"*100)

deltas_pub = np.array([r["delta_pub"] for r in results])
deltas_omega = np.array([r["delta_omega"] for r in results])
consistencies = np.array([r["consistency"] for r in results])

rms_pub = float(np.sqrt(np.mean(deltas_pub**2)))
rms_omega = float(np.sqrt(np.mean(deltas_omega**2)))

print(f"\n  N = {len(results)} particules")
print(f"\n  m_publié vs m_exp :")
print(f"    RMS = {rms_pub:.3f}%")
print(f"    max |Δ| = {np.max(np.abs(deltas_pub)):.2f}% sur {results[np.argmax(np.abs(deltas_pub))]['nm']}")
print(f"\n  m_Ω_ts vs m_exp :")
print(f"    RMS = {rms_omega:.3f}%")
print(f"    max |Δ| = {np.max(np.abs(deltas_omega)):.2f}% sur {results[np.argmax(np.abs(deltas_omega))]['nm']}")

print(f"\n  Cohérence m_publié vs m_Ω_ts :")
print(f"    RMS cohérence = {float(np.sqrt(np.mean(consistencies**2))):.2f}%")
print(f"    Particules cohérentes (<5%) : {sum(1 for c in consistencies if c<5)}/{len(results)}")
print(f"    Particules incohérentes (>30%) : {sum(1 for c in consistencies if c>30)}/{len(results)}")


# Particules suspectes
print(f"\n  Particules avec écart |m_Ω_ts - m_publié|/m_publié > 30% :")
for r in results:
    if r["consistency"] > 30:
        print(f"    {r['nm']:<6} {'+'.join(r['cfg']):<22} "
              f"publié={r['m_published']:.2f}  Ω_ts={r['m_omega']:.2f}")


print(f"\n  Particules avec |Δ_Ω_ts| > 5% (résidus structurels non-négligeables) :")
for r in sorted(results, key=lambda x: abs(x["delta_omega"]), reverse=True)[:15]:
    if abs(r["delta_omega"]) > 5:
        print(f"    {r['nm']:<6} {'+'.join(r['cfg']):<22} "
              f"Δ_Ω_ts = {r['delta_omega']:+7.2f}%, "
              f"m_exp = {r['m_exp']:.2f}")


# ───────── Diagnostic final ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC")
print("═"*100)

n_consistent = sum(1 for c in consistencies if c < 5)
ratio = n_consistent/len(results)*100
print(f"""
  Cohérence m_publié vs m_Ω_ts strict : {n_consistent}/{len(results)} ({ratio:.0f}%)

  Interprétation :
    Si {ratio:.0f}% sont incohérents (>5% d'écart), cela signifie que la
    table V5 publiée utilise une formule différente de Ω_ts strict :
    soit l'ancienne formule V4 (m = α_0 × U_EM × c_J^3.55 avec f(2), f(3))
    soit des paramètres calibrés par classe (corrections QM post-hoc).

  Le RMS Ω_ts strict = {rms_omega:.2f}% reflète la précision RÉELLE
  de la formule analytique fermée, sans calibration cachée.
""")
