r"""
DIAGNOSTIC SYSTÉMATIQUE des résidus Ω_ts strict + assignations optimales.

Question : quel invariant prédit |Δ Ω_ts| ?
Si on identifie cette dépendance, on a le principe manquant.

Méthode :
  1. Pour chaque particule, calculer Ω_ts strict avec optimal assignment
  2. Tabuler le résidu (m_pred - m_exp)/m_exp
  3. Calculer ~15 invariants du link
  4. Régression / corrélation systématique
  5. Identifier la (les) variable(s) qui distinguent les Δ ~ 0 des Δ ~ -3%
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


def omega_strict(components):
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


# ───────── Assignations optimales Ω_ts strict ─────────
PARTICULES = [
    # (nm, config_optimal, m_exp, 2J, classe)
    ("μ⁻",   ["3_1","6_1"],          105.66,   1, "lepton"),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86,   1, "lepton"),
    ("π̄",    ["3_1","3_1","3_1"],   137.27,    0, "pseudo"),
    ("η",    ["3_1","5_2","6_1"],   547.86,    0, "pseudo"),
    ("K̄",    ["6_1","6_1","6_1"],   495.65,    0, "pseudo"),
    ("D̄",    ["3_1","4_1","6_2"],  1867.0,     0, "pseudo"),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9,     2, "vector"),
    ("p",    ["3_1","7_2","7_2"],   938.27,    1, "baryon"),
    ("n",    ["3_1","7_2","7_2"],   939.57,    1, "baryon"),
    ("Λ_c",  ["5_1","5_1","7_2"],  2286.46,    1, "baryon"),
    ("Σ_c",  ["4_1","5_1","7_1"],  2453.0,     1, "baryon"),  # V5 match
    ("Λ",    ["5_1","5_1","5_1"],  1115.68,    1, "baryon"),
]


# ───────── Calculer résidus et invariants ─────────
def invariants(cfg):
    """Calcule plusieurs invariants pour le link."""
    n = len(cfg)
    cr_vals = np.array([TOPO[K]["cr"] for K in cfg])
    b_vals = np.array([TOPO[K]["b"] for K in cfg])
    det_vals = np.array([TOPO[K]["det"] for K in cfg])
    U_vals = np.array([U_KT[K][0]*U_KT[K][1] for K in cfg])
    Mn_vals = np.array([M_NAIVE[K] for K in cfg])

    counter = Counter(cfg)
    n_unique = len(counter)
    n_repeat = sum(c for c in counter.values() if c >= 2)
    has_3_1 = "3_1" in cfg
    n_3_1 = counter.get("3_1", 0)
    has_6_1 = "6_1" in cfg
    n_6_1 = counter.get("6_1", 0)

    return {
        "n": n,
        "n_unique": n_unique,
        "n_repeat": n_repeat,
        "is_symmetric": n_unique < n,    # au moins une paire répétée
        "is_fully_sym": n_unique == 1,    # tous identiques
        "has_3_1": int(has_3_1),
        "n_3_1": n_3_1,
        "has_6_1": int(has_6_1),
        "n_6_1": n_6_1,
        "sum_cr": float(np.sum(cr_vals)),
        "mean_cr": float(np.mean(cr_vals)),
        "min_cr": float(np.min(cr_vals)),
        "max_cr": float(np.max(cr_vals)),
        "std_cr": float(np.std(cr_vals)),
        "sum_U": float(np.sum(U_vals)),
        "mean_U": float(np.mean(U_vals)),
        "max_U": float(np.max(U_vals)),
        "min_U": float(np.min(U_vals)),
        "ratio_U": float(np.max(U_vals)/np.min(U_vals)) if np.min(U_vals) > 0 else 1,
        "sum_b": float(np.sum(b_vals)),
        "sum_det": float(np.sum(det_vals)),
        "sum_Mn": float(np.sum(Mn_vals)),
    }


print("\n" + "═"*100)
print("  Résidus Ω_ts strict + assignations optimales")
print("═"*100)
print(f"  {'nm':<6} {'config':<22} {'cls':<8} {'2J':>3} {'m_pred':>9} {'m_exp':>9} "
      f"{'Δ %':>8} {'|Δ|>1%':>8}")
print("  " + "-" * 90)

data = []
for nm, cfg, m_exp, two_J, cls in PARTICULES:
    m_pred = omega_strict(cfg)
    delta = (m_pred - m_exp)/m_exp*100
    inv = invariants(cfg)
    flag = "★" if abs(delta) > 1 else ""
    print(f"  {nm:<6} {'+'.join(cfg):<22} {cls:<8} {two_J:>3} "
          f"{m_pred:>9.2f} {m_exp:>9.2f} {delta:>+7.2f}% {flag:>8}")
    inv["nm"] = nm; inv["delta"] = delta; inv["abs_delta"] = abs(delta)
    inv["cls"] = cls; inv["two_J"] = two_J
    data.append(inv)


# ───────── Corrélations |Δ| vs invariants ─────────
print("\n" + "═"*100)
print("  Corrélations Pearson |Δ| vs invariants")
print("═"*100)

invariant_names = ["n", "n_unique", "is_symmetric", "is_fully_sym",
                    "has_3_1", "n_3_1", "has_6_1", "n_6_1",
                    "sum_cr", "mean_cr", "min_cr", "max_cr", "std_cr",
                    "sum_U", "mean_U", "max_U", "min_U", "ratio_U",
                    "sum_b", "sum_det", "sum_Mn", "two_J"]

abs_delta = np.array([d["abs_delta"] for d in data])
signed_delta = np.array([d["delta"] for d in data])

print(f"\n  {'invariant':<18} {'r vs |Δ|':>11} {'r vs Δ':>11} {'note':<20}")
print("  " + "-" * 60)
for inv_name in invariant_names:
    v = np.array([d[inv_name] for d in data], dtype=float)
    if np.std(v) > 1e-10:
        r_abs = np.corrcoef(v, abs_delta)[0, 1]
        r_sign = np.corrcoef(v, signed_delta)[0, 1]
        flag = "★★" if abs(r_abs) > 0.7 else ("★" if abs(r_abs) > 0.5 else "")
        print(f"  {inv_name:<18} {r_abs:>+10.3f}  {r_sign:>+10.3f}  {flag:<20}")


# ───────── Test d'hypothèse symétrie ─────────
print("\n" + "═"*100)
print("  Hypothèse SYMÉTRIE : la répétition de composantes réduit |Δ|")
print("═"*100)

print(f"\n  Particules SYMÉTRIQUES (au moins une paire répétée) :")
print(f"  {'nm':<6} {'config':<22} {'Δ %':>8}")
sym_deltas = []
for d in data:
    if d["is_symmetric"]:
        # Reconstruct cfg
        cfg = next(p[1] for p in PARTICULES if p[0] == d["nm"])
        print(f"  {d['nm']:<6} {'+'.join(cfg):<22} {d['delta']:>+7.2f}%")
        sym_deltas.append(d["delta"])

print(f"\n  Particules ASYMÉTRIQUES (toutes composantes différentes) :")
asym_deltas = []
for d in data:
    if not d["is_symmetric"]:
        cfg = next(p[1] for p in PARTICULES if p[0] == d["nm"])
        print(f"  {d['nm']:<6} {'+'.join(cfg):<22} {d['delta']:>+7.2f}%")
        asym_deltas.append(d["delta"])

print(f"\n  Stats :")
print(f"    symétriques (n={len(sym_deltas)}) : "
      f"⟨Δ⟩ = {np.mean(sym_deltas):+.2f}%, "
      f"⟨|Δ|⟩ = {np.mean(np.abs(sym_deltas)):.2f}%")
print(f"    asymétriques (n={len(asym_deltas)}) : "
      f"⟨Δ⟩ = {np.mean(asym_deltas):+.2f}%, "
      f"⟨|Δ|⟩ = {np.mean(np.abs(asym_deltas)):.2f}%")


# ───────── Diagnostic final ─────────
print("\n" + "═"*100)
print("  Pattern le plus fort identifié ?")
print("═"*100)

# Top 3 corrélations
correlations = []
for inv_name in invariant_names:
    v = np.array([d[inv_name] for d in data], dtype=float)
    if np.std(v) > 1e-10:
        r = abs(np.corrcoef(v, abs_delta)[0, 1])
        correlations.append((inv_name, r))
correlations.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Top 5 invariants par corrélation avec |Δ| :")
for name, r in correlations[:5]:
    print(f"    {name:<20} : r = {r:.3f}")

print(f"""
  Hypothèses à tester séparément si pattern unique se dégage :
    1. Si SYMETRIE corrèle fortement → principe : « la symétrie
       inter-composantes annule le résidu via cancellation »
    2. Si HAS_3_1 + classe corrèle → principe : « 3_1 dans contexte
       asymétrique génère un biais »
    3. Si N_UNIQUE corrèle → principe : « plus de variété topologique
       = plus de bias »
""")
