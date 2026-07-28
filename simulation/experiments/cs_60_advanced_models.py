r"""
B : explorer si un modèle plus sophistiqué dépasse 0.979% RMS CV
sur le dataset à 60 particules.

Approches testées :
  M1 : Ridge avec alpha optimisé (au lieu de 0.1 par défaut)
  M2 : LASSO (sparsité, sélection automatique de features)
  M3 : Features groupées (pseudo, vector, fermion, baryon, ...)
  M4 : Features non-linéaires (log, sqrt, ratios)
  M5 : 5 invariants (au lieu de 4)
  M6 : 6 invariants
  M7 : Random forest sur invariants
  M8 : Combinaison "additive" sans c_1 (juste invariants globaux)
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


# Reproduire le dataset à 60 + scan optimal
all_configs = []
for K in KNOTS:
    all_configs.append([K])
for K1, K2 in combinations_with_replacement(KNOTS, 2):
    all_configs.append([K1, K2])
for K1, K2, K3 in combinations_with_replacement(KNOTS, 3):
    all_configs.append([K1, K2, K3])

config_masses = []
for cfg in all_configs:
    config_masses.append((cfg, omega_strict(cfg)))


def find_best(m_target, n_max=3):
    best = None; best_err = float('inf')
    for cfg, m in config_masses:
        if len(cfg) > n_max: continue
        err = abs(m - m_target) / m_target
        if err < best_err: best_err = err; best = (cfg, m)
    return best


TARGETS = [
    ("e⁻", 0.5110),  ("μ⁻", 105.66), ("τ⁻", 1776.86),
    ("s", 93.0), ("c", 1275.0), ("b", 4180.0),
    ("π⁰", 134.98), ("π+", 139.57), ("π-", 139.57),
    ("K⁰", 497.61), ("K+", 493.68), ("K-", 493.68),
    ("η", 547.86), ("η'", 957.78),
    ("D⁰", 1864.84), ("D+", 1869.66), ("D-", 1869.66), ("Ds+", 1968.34),
    ("B+", 5279.34), ("B⁰", 5279.66), ("Bs⁰", 5366.88), ("Bc+", 6274.9),
    ("ρ⁰", 775.26), ("ρ+", 775.26), ("ω", 782.65),
    ("φ", 1019.46), ("K*+", 891.66), ("K*⁰", 895.55),
    ("J/ψ", 3096.9), ("ψ(2S)", 3686.10), ("Υ", 9460.0), ("Υ(2S)", 10023.26),
    ("p", 938.27), ("n", 939.57),
    ("Λ", 1115.68),
    ("Σ+", 1189.37), ("Σ⁰", 1192.64), ("Σ-", 1197.45),
    ("Ξ⁰", 1314.86), ("Ξ-", 1321.71),
    ("Λc", 2286.46), ("Σc++", 2454.0), ("Σc+", 2452.9), ("Σc⁰", 2453.7),
    ("Ξc+", 2467.71), ("Ξc⁰", 2470.91),
    ("Ωc⁰", 2695.2),
    ("Λb⁰", 5619.6), ("Ξb⁰", 5791.9), ("Ωb-", 6046.0),
    ("Δ++", 1232.0), ("Δ+", 1232.0), ("Δ⁰", 1232.0), ("Δ-", 1232.0),
    ("Σ*+", 1382.83), ("Σ*⁰", 1383.7), ("Σ*-", 1387.2),
    ("Ξ*⁰", 1531.8), ("Ξ*-", 1535.0), ("Ω-", 1672.45),
]

data = []
for nm, m_exp in TARGETS:
    cfg, m = find_best(m_exp)
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": m,
                  "d_rel": (m_exp - m) / m_exp})

n_data = len(data); n_knots = len(KNOTS)
y = np.array([d["d_rel"] for d in data])
print(f"  Dataset {n_data} particules, RMS Ω_ts strict = {np.sqrt(np.mean(y**2))*100:.3f}%")


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.coef_ = None
    def fit(self, X, y):
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(X.shape[1])
        try: self.coef_ = np.linalg.solve(A, X.T @ y)
        except: self.coef_ = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
        return self


def Lasso_iter(X, y, alpha=0.001, n_iter=200):
    """LASSO par descente coordonnée simple."""
    n, p = X.shape
    coef = np.zeros(p)
    # Standardize
    X_norm = np.linalg.norm(X, axis=0) + 1e-12
    for it in range(n_iter):
        for j in range(p):
            r_j = y - X @ coef + X[:, j] * coef[j]
            z_j = X[:, j] @ r_j
            if z_j > alpha:
                coef[j] = (z_j - alpha) / X_norm[j]**2
            elif z_j < -alpha:
                coef[j] = (z_j + alpha) / X_norm[j]**2
            else:
                coef[j] = 0
    return coef


def kfold(n, k=5, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(n); rng.shuffle(idx)
    sizes = [n // k] * k
    for i in range(n % k): sizes[i] += 1
    start = 0
    for size in sizes:
        yield np.concatenate([idx[:start], idx[start+size:]]), idx[start:start+size]
        start += size


folds_idx = list(kfold(n_data, k=5))


def cv_rms_ridge(X, alpha=0.1):
    errs = []
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


def cv_rms_lasso(X, alpha=0.001):
    errs = []
    for tr, te in folds_idx:
        coef = Lasso_iter(X[tr], y[tr], alpha=alpha)
        errs.append(np.sqrt(np.mean((X[te] @ coef - y[te])**2)))
    return np.mean(errs) * 100


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


# Pré-calcul
inv_options = ["rasm", "sm4g", "top4g", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_inv = [(i, a) for i in inv_options for a in aggs]
precomputed = {(i, a): feat_col(i, a) for (i, a) in all_inv}
b_mean_col = feat_col("b", "mean")


print("\n" + "═"*100)
print("  M1 : Ridge avec alpha optimal")
print("═"*100)

# Tester plusieurs alpha
print(f"\n  c_1 + b_mean + 4 inv (re-recherché) :")
best_combo = (('rasm', 'sum'), ('top4g', 'mean'), ('det', 'max'), ('cr', 'sum'))
X = np.zeros((n_data, n_knots + 5))
X[:, :n_knots] = X_single
X[:, n_knots] = b_mean_col
for i, key in enumerate(best_combo):
    X[:, n_knots + 1 + i] = precomputed[key]

for alpha in [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0]:
    rms = cv_rms_ridge(X, alpha=alpha)
    print(f"    alpha = {alpha:<6.3f} : RMS CV = {rms:.4f}%")


print("\n" + "═"*100)
print("  M2 : LASSO (sparsité)")
print("═"*100)
for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05]:
    rms = cv_rms_lasso(X, alpha=alpha)
    print(f"    LASSO alpha = {alpha:<7.4f} : RMS CV = {rms:.4f}%")


print("\n" + "═"*100)
print("  M5 : 5 invariants (au lieu de 4)")
print("═"*100)
t0 = time.time()
best_5 = None; best_5_cv = float('inf')
for combo in combinations(range(len(all_inv)), 5):
    quint = tuple(all_inv[k] for k in combo)
    X = np.zeros((n_data, n_knots + 6))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(quint):
        X[:, n_knots + 1 + i] = precomputed[key]
    cv = cv_rms_ridge(X)
    if cv < best_5_cv:
        best_5_cv = cv; best_5 = quint
print(f"  Tps : {time.time() - t0:.0f}s, meilleur : {best_5_cv:.4f}%")
print(f"  Combo : {best_5}")


print("\n" + "═"*100)
print("  M6 : 6 invariants (au lieu de 4)")
print("═"*100)
t0 = time.time()
best_6 = None; best_6_cv = float('inf')
n_tested = 0
for combo in combinations(range(len(all_inv)), 6):
    sext = tuple(all_inv[k] for k in combo)
    X = np.zeros((n_data, n_knots + 7))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(sext):
        X[:, n_knots + 1 + i] = precomputed[key]
    cv = cv_rms_ridge(X)
    n_tested += 1
    if cv < best_6_cv:
        best_6_cv = cv; best_6 = sext
print(f"  Tps : {time.time() - t0:.0f}s, testé {n_tested}, meilleur : {best_6_cv:.4f}%")
print(f"  Combo : {best_6}")


print("\n" + "═"*100)
print("  M8 : Modèle minimal — invariants globaux seuls (sans c_1 par knot)")
print("═"*100)

# Juste b_mean + 4 invariants 4D, sans les counts par knot
X_min = np.zeros((n_data, 5))
X_min[:, 0] = b_mean_col
X_min[:, 1] = precomputed[("rasm", "sum")]
X_min[:, 2] = precomputed[("top4g", "mean")]
X_min[:, 3] = precomputed[("det", "max")]
X_min[:, 4] = precomputed[("cr", "sum")]
for alpha in [0.01, 0.1, 1.0, 10.0]:
    rms = cv_rms_ridge(X_min, alpha=alpha)
    print(f"  Modèle minimal (5 features seulement) alpha = {alpha} : {rms:.4f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE B — recherche d'amélioration sur 60 particules")
print("═"*100)
print(f"""
  Baseline (référence)            : 0.979% RMS CV (4 inv, Ridge alpha=0.1)

  M1 (Ridge alpha optimisé)        : voir tableau ci-dessus
  M2 (LASSO)                       : voir tableau ci-dessus
  M5 (5 invariants)                : {best_5_cv:.4f}%
  M6 (6 invariants)                : {best_6_cv:.4f}%

  Si aucun ne dépasse 0.9%, le plancher topologique est confirmé.
  Si M6 < 0.85%, plus d'invariants compte vraiment.
  Si Ridge alpha > 1.0 améliore, sur-régularisation aide.
""")
