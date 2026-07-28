r"""
Dataset à 60 particules avec scan d'assignations optimales pour chacune.

Étape 1 : scan exhaustif des configs (1-3 composantes) pour Ω_ts strict
Étape 2 : sélectionner la meilleure assignation par particule
Étape 3 : appliquer le modèle topologique (c_1 + b_mean + 4D)
Étape 4 : comparer RMS CV à celui obtenu avec 23 particules
"""
from __future__ import annotations
import numpy as np
import time
from collections import Counter
from itertools import combinations, combinations_with_replacement
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


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


# Pré-calcul de toutes les masses possibles
print("Pré-calcul de toutes les configurations 1-3 composantes…")
all_configs = []
for K in KNOTS:
    all_configs.append([K])
for K1, K2 in combinations_with_replacement(KNOTS, 2):
    all_configs.append([K1, K2])
for K1, K2, K3 in combinations_with_replacement(KNOTS, 3):
    all_configs.append([K1, K2, K3])

config_masses = []
for cfg in all_configs:
    m = omega_strict(cfg)
    config_masses.append((cfg, m))
print(f"  Total configs : {len(config_masses)}")


def find_best_config(m_target, n_max=3):
    """Trouve la config qui minimise |Δ|."""
    best = None; best_err = float('inf')
    for cfg, m in config_masses:
        if len(cfg) > n_max: continue
        err = abs(m - m_target) / m_target
        if err < best_err:
            best_err = err
            best = (cfg, m)
    return best


# ───────── Liste cible des 60 particules ─────────
TARGETS = [
    # Leptons
    ("e⁻", 0.5110),  ("μ⁻", 105.66), ("τ⁻", 1776.86),
    # Quarks
    ("s", 93.0), ("c", 1275.0), ("b", 4180.0),
    # Mésons pseudoscalaires
    ("π⁰", 134.98), ("π+", 139.57), ("π-", 139.57),
    ("K⁰", 497.61), ("K+", 493.68), ("K-", 493.68),
    ("η", 547.86), ("η'", 957.78),
    ("D⁰", 1864.84), ("D+", 1869.66), ("D-", 1869.66), ("Ds+", 1968.34),
    ("B+", 5279.34), ("B⁰", 5279.66), ("Bs⁰", 5366.88), ("Bc+", 6274.9),
    # Mésons vecteurs
    ("ρ⁰", 775.26), ("ρ+", 775.26), ("ω", 782.65),
    ("φ", 1019.46), ("K*+", 891.66), ("K*⁰", 895.55),
    ("J/ψ", 3096.9), ("ψ(2S)", 3686.10), ("Υ", 9460.0), ("Υ(2S)", 10023.26),
    # Baryons octet
    ("p", 938.27), ("n", 939.57),
    ("Λ", 1115.68),
    ("Σ+", 1189.37), ("Σ⁰", 1192.64), ("Σ-", 1197.45),
    ("Ξ⁰", 1314.86), ("Ξ-", 1321.71),
    ("Λc", 2286.46), ("Σc++", 2454.0), ("Σc+", 2452.9), ("Σc⁰", 2453.7),
    ("Ξc+", 2467.71), ("Ξc⁰", 2470.91),
    ("Ωc⁰", 2695.2),
    ("Λb⁰", 5619.6), ("Ξb⁰", 5791.9), ("Ωb-", 6046.0),
    # Baryons décuplet
    ("Δ++", 1232.0), ("Δ+", 1232.0), ("Δ⁰", 1232.0), ("Δ-", 1232.0),
    ("Σ*+", 1382.83), ("Σ*⁰", 1383.7), ("Σ*-", 1387.2),
    ("Ξ*⁰", 1531.8), ("Ξ*-", 1535.0), ("Ω-", 1672.45),
]

print(f"\n  Recherche optimale pour {len(TARGETS)} particules…")
data = []
for nm, m_exp in TARGETS:
    cfg, m = find_best_config(m_exp)
    err = (m - m_exp) / m_exp
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": m,
                  "d_rel": -err})  # convention : (m_exp - m)/m_exp

print(f"  Toutes assignations trouvées.")

n_data = len(data); n_knots = len(KNOTS)
y = np.array([d["d_rel"] for d in data])

rms_omega = float(np.sqrt(np.mean(y**2))) * 100
print(f"\n  RMS Ω_ts strict (optimal) : {rms_omega:.3f}%")

# Détails
print(f"\n  Détail (top 10 pires) :")
sorted_data = sorted(data, key=lambda d: abs(d["d_rel"]), reverse=True)[:10]
for d in sorted_data:
    print(f"    {d['nm']:<8} {'+'.join(d['cfg']):<22} "
          f"exp={d['m_exp']:.2f}, opt={d['om']:.2f}, Δ={d['d_rel']*100:+.2f}%")


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
    for tr, te in list(kfold(n_data, k=k_folds)):
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


# Pré-calcul matrices
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


rms_c1 = cv_rms(X_single, y)
print(f"\n  + c_1(K) Ridge : {rms_c1:.3f}% CV")

X_bm = np.zeros((n_data, n_knots + 1))
X_bm[:, :n_knots] = X_single
X_bm[:, n_knots] = feat_col("b", "mean")
rms_bm = cv_rms(X_bm, y)
print(f"  + b_mean       : {rms_bm:.3f}% CV")

# 4 invariants gagnants à 23 particules
X_4d = np.zeros((n_data, n_knots + 5))
X_4d[:, :n_knots] = X_single
X_4d[:, n_knots] = feat_col("b", "mean")
X_4d[:, n_knots+1] = feat_col("sm4g", "max")
X_4d[:, n_knots+2] = feat_col("top4g", "mean")
X_4d[:, n_knots+3] = feat_col("top4g", "max")
X_4d[:, n_knots+4] = feat_col("vol", "mean")
rms_4d = cv_rms(X_4d, y)
print(f"  + 4 inv 4D (gagnant 23) : {rms_4d:.3f}% CV")


# ───────── Re-recherche meilleure combinaison ─────────
inv_options = ["rasm", "sm4g", "top4g", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_inv = [(i, a) for i in inv_options for a in aggs]
precomputed = {(i, a): feat_col(i, a) for (i, a) in all_inv}
b_mean_col = feat_col("b", "mean")

print(f"\n  Recherche meilleure combinaison à 4 invariants ({n_data} particules)…")
t0 = time.time()
best_4 = None; best_4_cv = float('inf')
for combo in combinations(range(len(all_inv)), 4):
    quad = tuple(all_inv[k] for k in combo)
    X = np.zeros((n_data, n_knots + 5))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(quad):
        X[:, n_knots + 1 + i] = precomputed[key]
    cv = cv_rms(X, y)
    if cv < best_4_cv:
        best_4_cv = cv; best_4 = quad
print(f"  Tps : {time.time() - t0:.0f}s")
print(f"\n  ★ Meilleure combinaison ({n_data} part.) : {best_4}")
print(f"  RMS CV : {best_4_cv:.4f}%")


# ───────── Synthèse comparative ─────────
print("\n" + "═"*100)
print("  COMPARAISON 23 vs", n_data, "PARTICULES")
print("═"*100)
print(f"""
  Niveau                         | 23 part. | {n_data} part.
  ───────────────────────────────|──────────|──────────
  Ω_ts strict (optimal)          | 1.218%   | {rms_omega:.3f}%
  + c_1(K) Ridge                 | 0.801%   | {rms_c1:.3f}%
  + b_mean                       | 0.653%   | {rms_bm:.3f}%
  + 4 inv 4D (gagnant 23)        | 0.566%   | {rms_4d:.3f}%
  + 4 inv 4D (re-recherché)      | 0.566%   | {best_4_cv:.3f}%

  Interprétation :
""")

if best_4_cv < 0.5:
    print("  → RMS descend nettement avec plus de données.")
    print(f"    Le plancher réel est ~{best_4_cv:.2f}%, ce qui suggère")
    print("    qu'à 23 particules il restait du bruit statistique.")
elif best_4_cv < 0.6:
    print("  → RMS reste stable autour de 0.5-0.6% avec plus de données.")
    print(f"    Le plancher de précision topologique réel est ~{best_4_cv:.2f}%.")
else:
    print("  → RMS plus grand avec plus de données.")
    print("    Cela peut indiquer soit des assignations sous-optimales")
    print("    pour les nouvelles particules, soit un échantillonnage")
    print("    plus représentatif (les anciennes 23 particules étaient")
    print("    biaisées vers les meilleurs cas).")
