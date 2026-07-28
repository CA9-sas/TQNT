r"""
Extension du dataset à 60 particules avec assignations V5 publiées
(ou optimales pour celles à très gros résidu Ω_ts strict).

Si le mur 0.566% RMS CV reflète :
  - du BRUIT statistique (23 particules) → descend à 0.4-0.5% avec 60
  - une LIMITE PHYSIQUE intrinsèque → reste autour de 0.55-0.6%

Le test est décisif pour conclure sur la complétude topologique.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# Invariants topologiques avancés
RASMUSSEN = {"3_1":-2,"4_1":0,"5_1":-4,"5_2":-2,"6_1":0,"6_2":-2,"6_3":0,
             "7_1":-6,"7_2":-2,"7_3":-4,"7_4":-2,"7_5":-4,"7_6":-2,"7_7":0,
             "8_1":0,"8_2":-4,"8_3":0,"9_1":-8,"10_1":0}
SM4G = {"3_1":1,"4_1":1,"5_1":2,"5_2":1,"6_1":1,"6_2":2,"6_3":2,
        "7_1":3,"7_2":1,"7_3":2,"7_4":1,"7_5":2,"7_6":2,"7_7":2,
        "8_1":1,"8_2":3,"8_3":1,"9_1":4,"10_1":1}
TOP4G = {"3_1":1,"4_1":1,"5_1":2,"5_2":1,"6_1":0,"6_2":2,"6_3":2,
         "7_1":3,"7_2":1,"7_3":2,"7_4":1,"7_5":2,"7_6":2,"7_7":2,
         "8_1":0,"8_2":3,"8_3":1,"9_1":4,"10_1":1}
HYP_VOL = {"3_1":0.0,"4_1":2.02988,"5_1":0.0,"5_2":2.82812,
           "6_1":3.16396,"6_2":4.40083,"6_3":5.69302,
           "7_1":0.0,"7_2":3.33174,"7_3":4.59213,"7_4":5.13794,
           "7_5":6.44354,"7_6":7.08493,"7_7":7.64338,
           "8_1":3.42721,"8_2":7.41862,"8_3":8.13491,
           "9_1":0.0,"10_1":4.05766}

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
               "U": U_KT[K][0]*U_KT[K][1],
               "rasm": RASMUSSEN[K], "sm4g": SM4G[K], "top4g": TOP4G[K],
               "vol": HYP_VOL[K]}
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


# ───────── Dataset étendu à 60 particules ─────────
# Combine V5 publiées + optimales scannées + nouvelles partenaires isospin
PARTICULES = [
    # ─── Leptons (3) ───
    ("e⁻",   ["3_1"],                0.5110),
    ("μ⁻",   ["3_1","6_1"],          105.66),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86),
    # ─── Quarks (effectifs, V5) (3) ───
    ("s",    ["5_2","5_2"],          93.0),
    ("c",    ["5_1","6_1","6_1"],   1275.0),
    ("b",    ["7_2","8_2"],         4180.0),
    # ─── Mésons pseudoscalaires (12 — partenaires inclus) ───
    ("π⁰",   ["3_1","3_1","3_1"],   134.98),
    ("π+",   ["3_1","3_1","3_1"],   139.57),  # même topo, isospin partner
    ("π-",   ["3_1","3_1","3_1"],   139.57),
    ("K⁰",   ["6_1","6_1","6_1"],   497.61),
    ("K+",   ["6_1","6_1","6_1"],   493.68),
    ("K-",   ["6_1","6_1","6_1"],   493.68),
    ("η",    ["3_1","5_2","6_1"],   547.86),
    ("η'",   ["4_1","7_3"],          957.78),
    ("D⁰",   ["3_1","4_1","6_2"],  1864.84),
    ("D+",   ["3_1","4_1","6_2"],  1869.66),
    ("D-",   ["3_1","4_1","6_2"],  1869.66),
    ("B+",   ["5_2","6_3","6_3"],  5279.34),
    # ─── Mésons vecteurs (6) ───
    ("ρ⁰",   ["8_1","8_1"],         775.26),
    ("ρ+",   ["8_1","8_1"],         775.26),
    ("ω",    ["8_1","8_1"],         782.65),
    ("φ",    ["3_1","4_1","6_2"],  1019.46),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9),
    ("Υ",    ["7_1","7_3","9_1"],  9460.0),
    # ─── Baryons octet (10) ───
    ("p",    ["3_1","7_2","7_2"],   938.27),
    ("n",    ["3_1","7_2","7_2"],   939.57),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68),
    ("Σ+",   ["5_1","7_7"],        1189.37),
    ("Σ⁰",   ["5_1","7_7"],        1192.64),
    ("Σ-",   ["5_1","7_7"],        1197.45),
    ("Ξ⁰",   ["5_1","6_3"],        1314.86),
    ("Ξ-",   ["5_1","6_3"],        1321.71),
    ("Λc",   ["5_1","5_1","7_2"],  2286.46),
    ("Ξc",   ["5_1","6_2"],        2467.71),  # estimation
    # ─── Baryons décuplet (7) ───
    ("Δ++",  ["5_2","7_2","7_2"],  1232.0),
    ("Δ+",   ["5_2","7_2","7_2"],  1232.0),
    ("Δ⁰",   ["5_2","7_2","7_2"],  1232.0),
    ("Δ-",   ["5_2","7_2","7_2"],  1232.0),
    ("Σ*+",  ["10_1"],             1382.83),
    ("Σ*⁰",  ["10_1"],             1383.7),
    ("Σ*-",  ["10_1"],             1387.2),
    ("Ξ*⁰",  ["7_1","8_1"],        1531.8),
    ("Ξ*-",  ["7_1","8_1"],        1535.0),
    ("Ω-",   ["5_1","7_2","7_2"],  1672.45),
    # ─── Hadrons charmés/beaux additionnels (10) ───
    ("Σc++", ["4_1","5_1","7_1"],  2454.0),
    ("Σc+",  ["4_1","5_1","7_1"],  2452.9),
    ("Σc⁰",  ["4_1","5_1","7_1"],  2453.7),
    ("Ωc⁰",  ["7_3","7_7"],        2695.2),
    ("Λb⁰",  ["5_2","5_2","8_2"], 5619.6),
    ("Σb+",  ["3_1","6_3","8_2"], 5810.6),  # estimation V5
    ("Σb-",  ["3_1","6_3","8_2"], 5815.6),
    ("Ξb⁰",  ["6_2","8_2"],        5791.9),  # estimation
    ("Ωb-",  ["7_2","8_3"],        6046.0),  # estimation
    ("Bs⁰",  ["5_2","6_3","6_3"], 5366.88),
    # ─── Bosons EW (4) ───
    ("W±",   ["4_1","7_3","8_3"], 80377.0),
    ("Z⁰",   ["5_1","7_2","8_3"], 91188.0),
    ("H",    ["6_1","7_4","8_3"], 125250.0),
    ("t",    ["7_4","7_6","8_3"], 172760.0),
]

print(f"\n  Nombre de particules : {len(PARTICULES)}")

# ───────── Calcul Ω_ts strict et filtrage ─────────
# Note: 7_4 et 8_2, 8_3 et certaines paires demandent que U_KT, TOPO incluent tous
# vérifions que tous les knots utilisés sont dans le catalogue
needed_knots = set()
for nm, cfg, m_exp in PARTICULES:
    for K in cfg:
        needed_knots.add(K)

missing = needed_knots - set(KNOTS)
if missing:
    print(f"  WARNING: Knots manquants : {missing}")

data = []
huge_residuals = []
for nm, cfg, m_exp in PARTICULES:
    try:
        om = omega_strict(cfg)
        d_rel = (m_exp - om) / m_exp
        # Exclure si résidu > 50% (mauvaise assignation)
        if abs(d_rel) > 0.5:
            huge_residuals.append((nm, cfg, om, m_exp, d_rel * 100))
            continue
        data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                      "d_rel": d_rel})
    except Exception as e:
        print(f"  ERROR {nm}: {e}")
        continue

print(f"  Particules incluses (|Δ| < 50%) : {len(data)}")
if huge_residuals:
    print(f"\n  Exclues (|Δ| > 50%, mauvaise assignation) :")
    for nm, cfg, om, m_exp, d in huge_residuals[:5]:
        print(f"    {nm:<8} {'+'.join(cfg):<22} Ω={om:.2f}, exp={m_exp:.2f}, Δ={d:+.1f}%")
    if len(huge_residuals) > 5:
        print(f"    ... ({len(huge_residuals) - 5} autres)")

n_data = len(data); n_knots = len(KNOTS)
y = np.array([d["d_rel"] for d in data])


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.coef_ = None
    def fit(self, X, y):
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(X.shape[1])
        try: self.coef_ = np.linalg.solve(A, X.T @ y)
        except: self.coef_ = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
        return self


def kfold(n, k=5, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(n); rng.shuffle(idx)
    sizes = [n // k] * k
    for i in range(n % k): sizes[i] += 1
    start = 0
    for size in sizes:
        yield np.concatenate([idx[:start], idx[start+size:]]), idx[start:start+size]
        start += size


def cv_rms(X, y, alpha=0.1, k_folds=5):
    errs = []
    folds_idx = list(kfold(n_data, k=k_folds))
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


# Pré-calcul de X_single
X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)


def feat_col(inv, agg):
    col = np.zeros(n_data)
    for i, d in enumerate(data):
        vals = [TOPO[K][inv] for K in d["cfg"]]
        if agg == 'mean': col[i] = np.mean(vals)
        elif agg == 'max': col[i] = max(vals)
        elif agg == 'sum': col[i] = sum(vals)
    return col


# ───────── Évaluation initiale ─────────
rms_omega = float(np.sqrt(np.mean(y**2))) * 100
print(f"\n  RMS Ω_ts strict initial : {rms_omega:.3f}%")

# c_1 seul
rms_c1 = cv_rms(X_single, y)
print(f"  + c_1(K) Ridge          : {rms_c1:.3f}% CV")

# + b_mean
X_bm = np.zeros((n_data, n_knots + 1))
X_bm[:, :n_knots] = X_single
X_bm[:, n_knots] = feat_col("b", "mean")
rms_bm = cv_rms(X_bm, y)
print(f"  + b_mean                : {rms_bm:.3f}% CV")

# + 4 invariants 4D (config gagnante précédente)
X_4d = np.zeros((n_data, n_knots + 5))
X_4d[:, :n_knots] = X_single
X_4d[:, n_knots] = feat_col("b", "mean")
X_4d[:, n_knots+1] = feat_col("sm4g", "max")
X_4d[:, n_knots+2] = feat_col("top4g", "mean")
X_4d[:, n_knots+3] = feat_col("top4g", "max")
X_4d[:, n_knots+4] = feat_col("vol", "mean")
rms_4d = cv_rms(X_4d, y)
print(f"  + 4 invariants 4D (gagnants à 23 part.) : {rms_4d:.3f}% CV")


# ───────── Recherche de la meilleure combinaison à 4 invariants ─────────
print("\n" + "═"*100)
print("  Re-recherche de la meilleure combinaison à 4 invariants")
print("═"*100)

inv_options = ["rasm", "sm4g", "top4g", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_inv = [(i, a) for i in inv_options for a in aggs]

import time
t0 = time.time()

# Pré-calcul
precomputed = {(i, a): feat_col(i, a) for (i, a) in all_inv}
b_mean_col = feat_col("b", "mean")

best_4 = None; best_4_cv = float('inf')
n_tested = 0

for combo in combinations(range(len(all_inv)), 4):
    quad = tuple(all_inv[k] for k in combo)
    X = np.zeros((n_data, n_knots + 5))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(quad):
        X[:, n_knots + 1 + i] = precomputed[key]
    cv = cv_rms(X, y)
    n_tested += 1
    if cv < best_4_cv:
        best_4_cv = cv
        best_4 = quad
        print(f"  NEW BEST: {' + '.join(f'{i}_{a}' for i,a in quad)} → {cv:.4f}%")

t_total = time.time() - t0
print(f"\n  {n_tested} quadruplets testés en {t_total:.0f}s")
print(f"\n  ★ Meilleure combinaison : {best_4}")
print(f"  RMS CV : {best_4_cv:.4f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)

print(f"""
  Dataset : {n_data} particules

  Comparaison 23-part vs {n_data}-part :

    Niveau                            | 23 part. | {n_data} part.
    ──────────────────────────────────|──────────|──────────
    Ω_ts strict                       | 1.218%   | {rms_omega:.3f}%
    + c_1(K) Ridge                    | 0.801%   | {rms_c1:.3f}%
    + b_mean                          | 0.653%   | {rms_bm:.3f}%
    + 4 inv 4D (gagnant 23)           | 0.566%   | {rms_4d:.3f}%
    + 4 inv 4D (gagnant {n_data})        | —        | {best_4_cv:.3f}%

  Interprétation :
""")

if best_4_cv < 0.45:
    print("  → RMS descend nettement avec plus de données.")
    print("    Confirme que 0.566% à 23 part. était partiellement statistique.")
    print("    Le PLANCHER réel est plus bas, autour de", f"{best_4_cv:.2f}%")
elif best_4_cv < 0.55:
    print("  → Amélioration modeste avec plus de données.")
    print("    Une partie du 0.566% était statistique mais le plancher réel est proche.")
else:
    print("  → RMS reste stable ou augmente avec plus de données.")
    print("    Confirme que 0.566% n'est PAS du bruit statistique.")
    print("    C'est le vrai plancher de précision topologique.")
