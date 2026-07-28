r"""
Test F : intégration de ξ_BS (Biot-Savart) comme 6ᵉ feature au modèle
Kernel Ridge à N=95.

Question : la plasticité P23 (mécanisme physique) apporte-t-elle de
l'information additionnelle au-delà des 5 invariants topologiques
qui donnent 0.670% RMS CV ?

Construction de ξ_BS pour chaque entrelacs :
  ξ_BS(L) = 0  si n=1 (symétrie radiale)
  ξ_BS(L) = Σ_{i<j} (b_i b_j / 36) · √(U_i U_j) / (Σ U)²   sinon

Cette forme :
  • s'annule exactement pour les nœuds isolés (P23 prédiction)
  • combine indice de tresse (repr SU(2)) et énergie EM (Biot-Savart)
  • capture l'interaction inter-composantes

Le test compare :
  Modèle baseline (5 features) :        0.670% RMS CV
  Modèle baseline + ξ_BS (6 features) : ???
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


# Reproduire dataset à 95
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


TARGETS = [
    ("e⁻", 0.5110), ("μ⁻", 105.66), ("τ⁻", 1776.86),
    ("s", 93.0), ("c", 1275.0), ("b", 4180.0),
    ("π⁰", 134.98), ("π+", 139.57), ("π-", 139.57),
    ("K⁰", 497.61), ("K+", 493.68), ("K-", 493.68),
    ("η", 547.86), ("η'", 957.78),
    ("D⁰", 1864.84), ("D+", 1869.66), ("D-", 1869.66), ("Ds+", 1968.34),
    ("B+", 5279.34), ("B⁰", 5279.66), ("Bs⁰", 5366.88),
    ("ρ⁰", 775.26), ("ρ+", 775.26), ("ω", 782.65),
    ("φ", 1019.46), ("K*+", 891.66), ("K*⁰", 895.55),
    ("J/ψ", 3096.9), ("ψ(2S)", 3686.10), ("ψ(3770)", 3773.7),
    ("Υ", 9460.30), ("Υ(2S)", 10023.26), ("Υ(3S)", 10355.20),
    ("f2(1270)", 1275.4), ("a2(1320)", 1318.2), ("K2*(1430)", 1427.3),
    ("f2'(1525)", 1517.4), ("D*2(2460)", 2461.1),
    ("χc0", 3414.71), ("χc1", 3510.67), ("χc2", 3556.17), ("hc", 3525.38),
    ("χb0", 9859.44), ("χb1", 9892.78), ("χb2", 9912.21), ("ηc", 2983.9),
    ("D*+", 2010.26), ("Ds*+", 2112.2), ("Bc+", 6274.9),
    ("p", 938.27), ("n", 939.57), ("Λ", 1115.68),
    ("Σ+", 1189.37), ("Σ⁰", 1192.64), ("Σ-", 1197.45),
    ("Ξ⁰", 1314.86), ("Ξ-", 1321.71),
    ("Λ(1405)", 1405.1), ("Λ(1520)", 1518.5),
    ("N(1440)", 1440.0), ("N(1520)", 1515.0), ("N(1535)", 1530.0), ("N(1650)", 1655.0),
    ("Δ++", 1232.0), ("Δ+", 1232.0), ("Δ⁰", 1232.0), ("Δ-", 1232.0),
    ("Σ*+", 1382.83), ("Σ*⁰", 1383.7), ("Σ*-", 1387.2),
    ("Ξ*⁰", 1531.8), ("Ξ*-", 1535.0), ("Ω-", 1672.45),
    ("Λc", 2286.46), ("Σc++", 2454.0), ("Σc+", 2452.9), ("Σc⁰", 2453.7),
    ("Ξc+", 2467.71), ("Ξc⁰", 2470.91), ("Ωc⁰", 2695.2), ("Λc(2595)", 2592.25),
    ("Λb⁰", 5619.6), ("Σb+", 5810.56), ("Ξb⁰", 5791.9), ("Ωb-", 6046.0),
    ("X(3872)", 3871.65), ("Zc(3900)", 3887.1), ("Y(4260)", 4230.0),
    ("Pc(4380)", 4380.0), ("Pc(4450)", 4449.8), ("X(6900)", 6900.0),
    ("W±", 80377.0), ("Z⁰", 91188.0), ("H", 125250.0), ("t", 172760.0),
]

data = []
for nm, m_exp in TARGETS:
    cfg, m = find_best(m_exp)
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": m,
                  "d_rel": (m_exp - m) / m_exp})

n_data = len(data); n_knots = len(KNOTS)
y = np.array([d["d_rel"] for d in data])
print(f"  Dataset N={n_data}, RMS Ω_ts strict = {np.sqrt(np.mean(y**2))*100:.4f}%")


# ───────── Définition de ξ_BS Biot-Savart ─────────
def xi_BS(cfg):
    """ξ_BS basée sur Biot-Savart entre composantes.
       0 si n=1 (symétrie radiale).
       Pour n≥2 : somme sur paires de (b_i b_j / 36) · √(U_i U_j) / (Σ U)²
    """
    if len(cfg) < 2:
        return 0.0
    Us = [TOPO[K]["U"] for K in cfg]
    bs = [TOPO[K]["b"] for K in cfg]
    sum_U = sum(Us)
    total = 0.0
    for i in range(len(cfg)):
        for j in range(i+1, len(cfg)):
            total += (bs[i]*bs[j] / 36) * np.sqrt(Us[i]*Us[j]) / sum_U**2
    return total


# Autres variantes pour comparaison
def xi_BS_signed(cfg, two_J=1):
    """ξ_BS avec signe pseudoscalaire/autre."""
    sigma = +1 if two_J == 0 else -1
    return sigma * xi_BS(cfg)


def xi_BS_gamma(cfg):
    """Forme P23 originelle : σ · γ · ε."""
    if len(cfg) < 2: return 0.0
    sum_U = sum(TOPO[K]["U"] for K in cfg)
    return 1.0 / sum_U


def xi_BS_pair_braid(cfg):
    """Somme des produits de braid index sur paires."""
    if len(cfg) < 2: return 0.0
    bs = [TOPO[K]["b"] for K in cfg]
    return sum(bs[i]*bs[j] for i in range(len(bs)) for j in range(i+1, len(bs)))


def xi_BS_full(cfg):
    """Combinaison Biot-Savart complète : (b_i b_j) · (U_i + U_j) / (Σ U) sommé."""
    if len(cfg) < 2: return 0.0
    Us = [TOPO[K]["U"] for K in cfg]
    bs = [TOPO[K]["b"] for K in cfg]
    sum_U = sum(Us)
    total = 0
    for i in range(len(cfg)):
        for j in range(i+1, len(cfg)):
            total += bs[i] * bs[j] * (Us[i] + Us[j]) / (sum_U + 1e-12)
    return total


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


folds_idx = list(kfold(n_data, k=5))


def cv_rms_ridge(X, alpha=10.0):
    errs = []
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


def cv_rms_kernel(X, alpha=1.0, gamma=0.1):
    errs = []
    for tr, te in folds_idx:
        X_tr = X[tr]; X_te = X[te]; y_tr = y[tr]; y_te = y[te]
        K_tr = np.exp(-gamma * ((X_tr[:, None] - X_tr[None, :])**2).sum(axis=2))
        K_te = np.exp(-gamma * ((X_te[:, None] - X_tr[None, :])**2).sum(axis=2))
        n_tr = len(y_tr)
        a = np.linalg.solve(K_tr + alpha * np.eye(n_tr), y_tr)
        errs.append(np.sqrt(np.mean((K_te @ a - y_te)**2)))
    return np.mean(errs) * 100


def feat_col(inv, agg):
    col = np.zeros(n_data)
    for i, d in enumerate(data):
        vals = [TOPO[K][inv] for K in d["cfg"]]
        if agg == 'mean': col[i] = np.mean(vals)
        elif agg == 'max': col[i] = max(vals)
        elif agg == 'sum': col[i] = sum(vals)
    return col


# Modèle 5 features
X_5 = np.column_stack([
    feat_col("b", "mean"),
    feat_col("rasm", "sum"),
    feat_col("top4g", "mean"),
    feat_col("det", "max"),
    feat_col("cr", "sum"),
])
X_5_norm = (X_5 - X_5.mean(axis=0)) / (X_5.std(axis=0) + 1e-12)

# Construire les features ξ_BS
xi_BS_col = np.array([xi_BS(d["cfg"]) for d in data])
xi_BS_pair_col = np.array([xi_BS_pair_braid(d["cfg"]) for d in data])
xi_BS_full_col = np.array([xi_BS_full(d["cfg"]) for d in data])
xi_BS_gamma_col = np.array([xi_BS_gamma(d["cfg"]) for d in data])

print(f"\n  ξ_BS statistics:")
for name, col in [("xi_BS", xi_BS_col), ("xi_BS_pair", xi_BS_pair_col),
                   ("xi_BS_full", xi_BS_full_col), ("xi_BS_gamma", xi_BS_gamma_col)]:
    n_zero = sum(1 for v in col if abs(v) < 1e-12)
    print(f"    {name}: range [{col.min():.4f}, {col.max():.4f}], "
          f"{n_zero} zéros (composantes uniques)")


# ───────── Tests : Ridge α=10 ─────────
print("\n" + "═"*100)
print("  TEST F : ajout de ξ_BS au modèle baseline (Ridge α=10)")
print("═"*100)
print(f"\n  Baseline (5 features) : {cv_rms_ridge(X_5, 10.0):.4f}% RMS CV\n")

for name, col in [("xi_BS (b·b·√(U·U)/ΣU²)", xi_BS_col),
                   ("xi_BS_pair (b·b sum)", xi_BS_pair_col),
                   ("xi_BS_full (b·b·(U+U)/ΣU)", xi_BS_full_col),
                   ("xi_BS_gamma (1/ΣU)", xi_BS_gamma_col)]:
    X_6 = np.column_stack([X_5, col])
    rms = cv_rms_ridge(X_6, 10.0)
    print(f"  + {name:<35} : {rms:.4f}%")


# ───────── Tests : Kernel Ridge ─────────
print("\n" + "═"*100)
print("  TEST F : Kernel Ridge sur 6 features (5 topo + ξ_BS)")
print("═"*100)
print(f"\n  Baseline Kernel Ridge (5 features γ=0.1, α=1) : {cv_rms_kernel(X_5_norm, 1.0, 0.1):.4f}%\n")

for name, col in [("xi_BS", xi_BS_col),
                   ("xi_BS_pair", xi_BS_pair_col),
                   ("xi_BS_full", xi_BS_full_col),
                   ("xi_BS_gamma", xi_BS_gamma_col)]:
    # Normaliser ξ_BS
    col_norm = (col - col.mean()) / (col.std() + 1e-12)
    X_6 = np.column_stack([X_5_norm, col_norm])
    # Tester plusieurs (γ, α)
    best_cv = float('inf')
    best_ga = None
    for gamma in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            rms = cv_rms_kernel(X_6, alpha=alpha, gamma=gamma)
            if rms < best_cv:
                best_cv = rms
                best_ga = (gamma, alpha)
    print(f"  + {name:<15} → meilleur (γ={best_ga[0]}, α={best_ga[1]}) : {best_cv:.4f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE TEST F")
print("═"*100)

ridge_baseline = cv_rms_ridge(X_5, 10.0)
ridge_with_best_xi = min(
    cv_rms_ridge(np.column_stack([X_5, col]), 10.0)
    for col in [xi_BS_col, xi_BS_pair_col, xi_BS_full_col, xi_BS_gamma_col]
)

kernel_baseline = cv_rms_kernel(X_5_norm, 1.0, 0.1)

print(f"""
  RIDGE α=10 :
    Baseline (5 features) : {ridge_baseline:.4f}%
    Meilleur avec ξ_BS    : {ridge_with_best_xi:.4f}%
    Gain : {(ridge_baseline - ridge_with_best_xi)/ridge_baseline*100:+.1f}%

  KERNEL RIDGE :
    Baseline (5 features) : {kernel_baseline:.4f}%
    (voir tableau ci-dessus pour ξ_BS variants)

  CONCLUSION :
    Si ξ_BS améliore CV de >5%, la plasticité physique apporte
    de l'information additionnelle au-delà des invariants topologiques.

    Si pas d'amélioration ou dégradation, la plasticité P23 est
    déjà encodée implicitement dans les 5 invariants topologiques.
""")
