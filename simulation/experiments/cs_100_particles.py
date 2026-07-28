r"""
Extension à 100 particules : test de robustesse du modèle minimal et
de Kernel Ridge.

But : si 0.777% RMS CV à 60 particules reste stable à 100, on a un
plancher robuste. Si dégradation, il y a encore du sur-ajustement à 60.
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


all_configs = []
for K in KNOTS:
    all_configs.append([K])
for K1, K2 in combinations_with_replacement(KNOTS, 2):
    all_configs.append([K1, K2])
for K1, K2, K3 in combinations_with_replacement(KNOTS, 3):
    all_configs.append([K1, K2, K3])
config_masses = [(cfg, omega_strict(cfg)) for cfg in all_configs]


def find_best(m_target, n_max=3):
    best = None; best_err = float('inf')
    for cfg, m in config_masses:
        if len(cfg) > n_max: continue
        err = abs(m - m_target) / m_target
        if err < best_err: best_err = err; best = (cfg, m)
    return best


# ───────── 100 particules cibles avec QM ─────────
# (nm, m_exp, Q, |I3|, |S|, |C|, |B|, 2J)
TARGETS = [
    # Leptons (3)
    ("e⁻", 0.5110, -1, 0, 0, 0, 0, 1),
    ("μ⁻", 105.66, -1, 0, 0, 0, 0, 1),
    ("τ⁻", 1776.86, -1, 0, 0, 0, 0, 1),
    # Quarks (6)
    ("s", 93.0, -1/3, 0.5, 1, 0, 0, 1),
    ("c", 1275.0, 2/3, 0, 0, 1, 0, 1),
    ("b", 4180.0, -1/3, 0, 0, 0, 1, 1),
    # Pseudoscalar mesons (15)
    ("π⁰", 134.98, 0, 1, 0, 0, 0, 0),
    ("π+", 139.57, 1, 1, 0, 0, 0, 0),
    ("π-", 139.57, -1, 1, 0, 0, 0, 0),
    ("K⁰", 497.61, 0, 0.5, 1, 0, 0, 0),
    ("K+", 493.68, 1, 0.5, 1, 0, 0, 0),
    ("K-", 493.68, -1, 0.5, 1, 0, 0, 0),
    ("η", 547.86, 0, 0, 0, 0, 0, 0),
    ("η'", 957.78, 0, 0, 0, 0, 0, 0),
    ("D⁰", 1864.84, 0, 0.5, 0, 1, 0, 0),
    ("D+", 1869.66, 1, 0.5, 0, 1, 0, 0),
    ("D-", 1869.66, -1, 0.5, 0, 1, 0, 0),
    ("Ds+", 1968.34, 1, 0, 1, 1, 0, 0),
    ("B+", 5279.34, 1, 0.5, 0, 0, 1, 0),
    ("B⁰", 5279.66, 0, 0.5, 0, 0, 1, 0),
    ("Bs⁰", 5366.88, 0, 0, 1, 0, 1, 0),
    # Vector mesons (12)
    ("ρ⁰", 775.26, 0, 1, 0, 0, 0, 2),
    ("ρ+", 775.26, 1, 1, 0, 0, 0, 2),
    ("ω", 782.65, 0, 0, 0, 0, 0, 2),
    ("φ", 1019.46, 0, 0, 0, 0, 0, 2),
    ("K*+", 891.66, 1, 0.5, 1, 0, 0, 2),
    ("K*⁰", 895.55, 0, 0.5, 1, 0, 0, 2),
    ("J/ψ", 3096.9, 0, 0, 0, 0, 0, 2),
    ("ψ(2S)", 3686.10, 0, 0, 0, 0, 0, 2),
    ("ψ(3770)", 3773.7, 0, 0, 0, 0, 0, 2),
    ("Υ", 9460.30, 0, 0, 0, 0, 0, 2),
    ("Υ(2S)", 10023.26, 0, 0, 0, 0, 0, 2),
    ("Υ(3S)", 10355.20, 0, 0, 0, 0, 0, 2),
    # Tensor mesons (5)
    ("f2(1270)", 1275.4, 0, 0, 0, 0, 0, 4),
    ("a2(1320)", 1318.2, 0, 1, 0, 0, 0, 4),
    ("K2*(1430)", 1427.3, 0, 0.5, 1, 0, 0, 4),
    ("f2'(1525)", 1517.4, 0, 0, 0, 0, 0, 4),
    ("D*2(2460)", 2461.1, 0, 0.5, 0, 1, 0, 4),
    # Charmonium χ states (4)
    ("χc0", 3414.71, 0, 0, 0, 0, 0, 0),
    ("χc1", 3510.67, 0, 0, 0, 0, 0, 2),
    ("χc2", 3556.17, 0, 0, 0, 0, 0, 4),
    ("hc", 3525.38, 0, 0, 0, 0, 0, 2),
    # Bottomonium χ states (4)
    ("χb0", 9859.44, 0, 0, 0, 0, 0, 0),
    ("χb1", 9892.78, 0, 0, 0, 0, 0, 2),
    ("χb2", 9912.21, 0, 0, 0, 0, 0, 4),
    ("ηc", 2983.9, 0, 0, 0, 0, 0, 0),
    # D and Ds excited (3)
    ("D*+", 2010.26, 1, 0.5, 0, 1, 0, 2),
    ("Ds*+", 2112.2, 1, 0, 1, 1, 0, 2),
    ("Bc+", 6274.9, 1, 0, 0, 1, 1, 0),
    # Baryons octet (10)
    ("p", 938.27, 1, 0.5, 0, 0, 0, 1),
    ("n", 939.57, 0, 0.5, 0, 0, 0, 1),
    ("Λ", 1115.68, 0, 0, 1, 0, 0, 1),
    ("Σ+", 1189.37, 1, 1, 1, 0, 0, 1),
    ("Σ⁰", 1192.64, 0, 1, 1, 0, 0, 1),
    ("Σ-", 1197.45, -1, 1, 1, 0, 0, 1),
    ("Ξ⁰", 1314.86, 0, 0.5, 2, 0, 0, 1),
    ("Ξ-", 1321.71, -1, 0.5, 2, 0, 0, 1),
    ("Λ(1405)", 1405.1, 0, 0, 1, 0, 0, 1),
    ("Λ(1520)", 1518.5, 0, 0, 1, 0, 0, 3),
    # N* and Δ resonances (8)
    ("N(1440)", 1440.0, 1, 0.5, 0, 0, 0, 1),
    ("N(1520)", 1515.0, 0, 0.5, 0, 0, 0, 3),
    ("N(1535)", 1530.0, 0, 0.5, 0, 0, 0, 1),
    ("N(1650)", 1655.0, 0, 0.5, 0, 0, 0, 1),
    ("Δ++", 1232.0, 2, 1.5, 0, 0, 0, 3),
    ("Δ+", 1232.0, 1, 1.5, 0, 0, 0, 3),
    ("Δ⁰", 1232.0, 0, 1.5, 0, 0, 0, 3),
    ("Δ-", 1232.0, -1, 1.5, 0, 0, 0, 3),
    # Decuplet (6)
    ("Σ*+", 1382.83, 1, 1, 1, 0, 0, 3),
    ("Σ*⁰", 1383.7, 0, 1, 1, 0, 0, 3),
    ("Σ*-", 1387.2, -1, 1, 1, 0, 0, 3),
    ("Ξ*⁰", 1531.8, 0, 0.5, 2, 0, 0, 3),
    ("Ξ*-", 1535.0, -1, 0.5, 2, 0, 0, 3),
    ("Ω-", 1672.45, -1, 0, 3, 0, 0, 3),
    # Charm baryons (8)
    ("Λc", 2286.46, 1, 0, 0, 1, 0, 1),
    ("Σc++", 2454.0, 2, 1, 0, 1, 0, 1),
    ("Σc+", 2452.9, 1, 1, 0, 1, 0, 1),
    ("Σc⁰", 2453.7, 0, 1, 0, 1, 0, 1),
    ("Ξc+", 2467.71, 1, 0.5, 1, 1, 0, 1),
    ("Ξc⁰", 2470.91, 0, 0.5, 1, 1, 0, 1),
    ("Ωc⁰", 2695.2, 0, 0, 2, 1, 0, 1),
    ("Λc(2595)", 2592.25, 1, 0, 0, 1, 0, 1),
    # Bottom baryons (4)
    ("Λb⁰", 5619.6, 0, 0, 0, 0, 1, 1),
    ("Σb+", 5810.56, 1, 1, 0, 0, 1, 1),
    ("Ξb⁰", 5791.9, 0, 0.5, 1, 0, 1, 1),
    ("Ωb-", 6046.0, -1, 0, 2, 0, 1, 1),
    # Exotic and excited (6)
    ("X(3872)", 3871.65, 0, 0, 0, 0, 0, 2),
    ("Zc(3900)", 3887.1, 1, 1, 0, 0, 0, 2),
    ("Y(4260)", 4230.0, 0, 0, 0, 0, 0, 2),
    ("Pc(4380)", 4380.0, 1, 0.5, 0, 0, 0, 3),
    ("Pc(4450)", 4449.8, 1, 0.5, 0, 0, 0, 5),
    ("X(6900)", 6900.0, 0, 0, 0, 0, 0, 0),
    # Bosons EW (4)
    ("W±", 80377.0, 1, 0, 0, 0, 0, 2),
    ("Z⁰", 91188.0, 0, 0, 0, 0, 0, 2),
    ("H", 125250.0, 0, 0, 0, 0, 0, 0),
    ("t", 172760.0, 2/3, 0, 0, 0, 0, 1),
]

print(f"\n  Total cibles : {len(TARGETS)} particules")

# Assignations optimales
data = []
for nm, m_exp, Q, I3, S, C, B, two_J in TARGETS:
    cfg, m = find_best(m_exp)
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": m,
                  "d_rel": (m_exp - m) / m_exp,
                  "Q": Q, "I3": I3, "S": S, "C": C, "B": B, "two_J": two_J,
                  "n_phys": two_J + 1})

n_data = len(data); n_knots = len(KNOTS)
y = np.array([d["d_rel"] for d in data])
rms_om = float(np.sqrt(np.mean(y**2))) * 100
print(f"  RMS Ω_ts strict optimal : {rms_om:.3f}%")


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.coef_ = None
    def fit(self, X, y):
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(X.shape[1])
        try: self.coef_ = np.linalg.solve(A, X.T @ y)
        except: self.coef_ = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
        return self


def Lasso_iter(X, y, alpha=0.05, n_iter=300):
    n, p = X.shape; coef = np.zeros(p)
    X_norm2 = np.sum(X**2, axis=0) + 1e-12
    for _ in range(n_iter):
        for j in range(p):
            r_j = y - X @ coef + X[:, j] * coef[j]
            z_j = X[:, j] @ r_j
            if z_j > alpha: coef[j] = (z_j - alpha) / X_norm2[j]
            elif z_j < -alpha: coef[j] = (z_j + alpha) / X_norm2[j]
            else: coef[j] = 0
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


def cv_rms_ridge(X, alpha=10.0):
    errs = []
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


def cv_rms_lasso(X, alpha=0.05):
    errs = []
    for tr, te in folds_idx:
        coef = Lasso_iter(X[tr], y[tr], alpha=alpha)
        errs.append(np.sqrt(np.mean((X[te] @ coef - y[te])**2)))
    return np.mean(errs) * 100


def cv_rms_kernel(X, alpha=1.0, gamma=0.1):
    errs = []
    for tr, te in folds_idx:
        X_tr = X[tr]; X_te = X[te]; y_tr = y[tr]; y_te = y[te]
        K_tr = np.exp(-gamma * ((X_tr[:, None] - X_tr[None, :])**2).sum(axis=2))
        K_te = np.exp(-gamma * ((X_te[:, None] - X_tr[None, :])**2).sum(axis=2))
        n_tr = len(y_tr)
        a = np.linalg.solve(K_tr + alpha * np.eye(n_tr), y_tr)
        y_pred = K_te @ a
        errs.append(np.sqrt(np.mean((y_pred - y_te)**2)))
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


# Modèle minimal : 5 features
X_min = np.column_stack([
    feat_col("b", "mean"),
    feat_col("rasm", "sum"),
    feat_col("top4g", "mean"),
    feat_col("det", "max"),
    feat_col("cr", "sum"),
])
X_min_norm = (X_min - X_min.mean(axis=0)) / (X_min.std(axis=0) + 1e-12)

print(f"\n  Tests des modèles à N={n_data} :")
print(f"  {'Modèle':<40} {'RMS CV':>9}")
print("  " + "-" * 50)

# Baseline
print(f"  {'Ω_ts strict':<40} {rms_om:>8.4f}%")
print(f"  {'+ c_1(K) Ridge α=0.1':<40} {cv_rms_ridge(X_single, alpha=0.1):>8.4f}%")
print(f"  {'+ c_1(K) Ridge α=10':<40} {cv_rms_ridge(X_single, alpha=10.0):>8.4f}%")

# 5 features (modèle minimal)
print(f"  {'5 features + Ridge α=10':<40} {cv_rms_ridge(X_min, alpha=10.0):>8.4f}%")
print(f"  {'5 features + LASSO α=0.05':<40} {cv_rms_lasso(X_min, alpha=0.05):>8.4f}%")
print(f"  {'5 features + LASSO α=0.5':<40} {cv_rms_lasso(X_min, alpha=0.5):>8.4f}%")

# Kernel Ridge
print(f"\n  Kernel Ridge (γ × α) :")
for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        rms = cv_rms_kernel(X_min_norm, alpha=alpha, gamma=gamma)
        if rms < rms_om - 0.05:
            print(f"    γ={gamma:<5} α={alpha:<5} → {rms:>6.4f}%")


# Re-recherche des meilleurs 4 invariants
inv_options = ["rasm", "sm4g", "top4g", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_inv = [(i, a) for i in inv_options for a in aggs]
precomputed = {(i, a): feat_col(i, a) for (i, a) in all_inv}
b_mean_col = feat_col("b", "mean")

print(f"\n  Re-recherche meilleur quadruplet pour N={n_data}…")
t0 = time.time()
best = (None, float('inf'))
for combo in combinations(range(len(all_inv)), 4):
    quad = tuple(all_inv[k] for k in combo)
    X = np.zeros((n_data, n_knots + 5))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(quad):
        X[:, n_knots + 1 + i] = precomputed[key]
    cv = cv_rms_ridge(X, alpha=10.0)
    if cv < best[1]: best = (quad, cv)
print(f"  Tps : {time.time() - t0:.0f}s")
print(f"  ★ Meilleur quadruplet (N={n_data}, Ridge α=10) : {best[0]}")
print(f"  RMS CV : {best[1]:.4f}%")


# ───────── Synthèse comparative ─────────
print("\n" + "═"*100)
print(f"  COMPARAISON N=23, 60, {n_data}")
print("═"*100)
print(f"""
  Modèle                        | N=23     | N=60     | N={n_data}
  ─────────────────────────────|──────────|──────────|──────────
  Ω_ts strict                   | 1.218%   | 0.914%   | {rms_om:.3f}%
  Ridge α=0.1 (c_1)              | 0.801%   | 0.979%   | {cv_rms_ridge(X_single, 0.1):.3f}%
  Ridge α=10 (c_1)               | —        | 0.839%   | {cv_rms_ridge(X_single, 10.0):.3f}%
  Modèle minimal 5 features      | —        | 0.889%   | {cv_rms_ridge(X_min, 10.0):.3f}%
  LASSO 5 features α=0.05        | —        | 0.806%   | {cv_rms_lasso(X_min, 0.05):.3f}%
  Kernel Ridge γ=0.1, α=1        | —        | 0.777%   | {cv_rms_kernel(X_min_norm, 1.0, 0.1):.3f}%
""")
