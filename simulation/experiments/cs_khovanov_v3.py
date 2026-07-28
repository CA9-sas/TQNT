r"""
Test exhaustif des QUADRUPLETS d'invariants ajoutés à c_1 + b_mean.

Précédent résultat : triplet (sm4g_max, top4g_max, vol_mean) → 0.591% CV.
Question : un 4ème invariant peut-il faire encore mieux ?

C(30, 4) ≈ 27000 combinaisons à tester. Optimisation :
  1. Pré-calcul des colonnes de design par (inv, agg)
  2. Réutilisation pour chaque quadruplet
  3. Ridge fit rapide (matrices 23x23)
"""
from __future__ import annotations
import numpy as np
import time
from collections import Counter
from itertools import combinations
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
KHOV_RANK = {"3_1":4,"4_1":5,"5_1":6,"5_2":7,"6_1":9,"6_2":11,"6_3":13,
             "7_1":8,"7_2":11,"7_3":13,"7_4":15,"7_5":15,"7_6":17,"7_7":19,
             "8_1":13,"8_2":17,"8_3":17,"9_1":10,"10_1":17}
JONES_SPAN = {"3_1":3,"4_1":4,"5_1":5,"5_2":5,"6_1":6,"6_2":6,"6_3":6,
              "7_1":7,"7_2":7,"7_3":7,"7_4":7,"7_5":7,"7_6":7,"7_7":7,
              "8_1":8,"8_2":8,"8_3":8,"9_1":9,"10_1":10}
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
               "khov": KHOV_RANK[K], "jspan": JONES_SPAN[K],
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


PARTICULES = [
    ("μ⁻",["3_1","6_1"],105.66),("τ⁻",["5_1","5_1","6_1"],1776.86),
    ("π",["3_1","3_1","3_1"],137.27),("η",["3_1","5_2","6_1"],547.86),
    ("K",["6_1","6_1","6_1"],495.65),("D",["3_1","4_1","6_2"],1867.0),
    ("B",["5_2","6_3","6_3"],5279.0),("ρ",["8_1","8_1"],778.85),
    ("J/ψ",["3_1","4_1","6_3"],3096.9),("Υ",["7_1","7_3","9_1"],9460.0),
    ("p",["3_1","7_2","7_2"],938.27),("n",["3_1","7_2","7_2"],939.57),
    ("Λ",["5_1","5_1","5_1"],1115.68),("Σ",["5_1","7_7"],1193.0),
    ("Ξ",["5_1","6_3"],1318.0),("Δ",["5_2","7_2","7_2"],1232.0),
    ("Σ*",["10_1"],1383.0),("Ξ*",["7_1","8_1"],1532.0),
    ("Ω-",["5_1","7_2","7_2"],1672.0),("D0",["3_1","4_1","6_2"],1864.84),
    ("Λc",["5_1","5_1","7_2"],2286.46),("Σc",["4_1","5_1","7_1"],2453.0),
    ("Ωc",["7_3","7_7"],2695.0),
]


data = []
for nm, cfg, m_exp in PARTICULES:
    om = omega_strict(cfg)
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                  "d_rel": (m_exp - om) / m_exp})

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


# Pré-calcul des matrices c_1(K) et features
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


inv_options = ["rasm", "sm4g", "top4g", "khov", "jspan", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_features = [(i, a) for i in inv_options for a in aggs]

# Pré-calcul TOUTES les colonnes possibles
print(f"Pré-calcul de {len(all_features)} colonnes…")
precomputed = {}
for inv, agg in all_features:
    precomputed[(inv, agg)] = feat_col(inv, agg)
b_mean_col = feat_col("b", "mean")


# Pré-calcul des indices de fold
folds_idx = list(kfold(n_data, k=5))


def cv_rms_quad(quad_features, alpha=0.1):
    """5-fold CV pour b_mean + 4 features additionnelles."""
    n_feat = n_knots + 1 + len(quad_features)
    X = np.zeros((n_data, n_feat))
    X[:, :n_knots] = X_single
    X[:, n_knots] = b_mean_col
    for i, key in enumerate(quad_features):
        X[:, n_knots + 1 + i] = precomputed[key]
    errs = []
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


# ───────── Recherche exhaustive sur quadruplets ─────────
print("\nRecherche exhaustive sur tous les quadruplets de features…")
t0 = time.time()

best_4 = None; best_4_cv = 0.591  # déjà connu pour triplet
n_tested = 0
prog_step = 2000

for combo in combinations(range(len(all_features)), 4):
    quad = tuple(all_features[k] for k in combo)
    cv = cv_rms_quad(quad)
    n_tested += 1
    if cv < best_4_cv:
        best_4_cv = cv
        best_4 = quad
        print(f"  NEW BEST: {' + '.join(f'{i}_{a}' for i,a in quad)} → {cv:.4f}%")
    if n_tested % prog_step == 0:
        elapsed = time.time() - t0
        rate = n_tested / elapsed
        print(f"  ... {n_tested} testés ({rate:.0f}/s)")

t_total = time.time() - t0
print(f"\n  Total : {n_tested} quadruplets testés en {t_total:.0f}s")
print(f"\n  ★ Meilleur quadruplet : {best_4}")
print(f"  RMS CV : {best_4_cv:.4f}%")


# ───────── Détail du meilleur modèle ─────────
print("\n" + "═"*100)
print(f"  Détail du modèle gagnant")
print("═"*100)

n_feat = n_knots + 1 + 4
X = np.zeros((n_data, n_feat))
X[:, :n_knots] = X_single
X[:, n_knots] = b_mean_col
for i, key in enumerate(best_4):
    X[:, n_knots + 1 + i] = precomputed[key]
r_full = Ridge(alpha=0.1).fit(X, y)
y_pred = X @ r_full.coef_
errors = (y - y_pred) * 100
rms_train = np.sqrt(np.mean(errors**2))

print(f"\n  Coefficients :")
print(f"    b_mean        : {r_full.coef_[n_knots]:+.6f}")
for i, (inv, agg) in enumerate(best_4):
    print(f"    {inv+'_'+agg:<14}: {r_full.coef_[n_knots+1+i]:+.6f}")

print(f"\n  RMS train : {rms_train:.4f}%")
print(f"  RMS CV    : {best_4_cv:.4f}%")

print(f"\n  Particule par particule :")
print(f"  {'nm':<5} {'Δ initial':>10} {'Δ corrigé':>10}")
print("  " + "-" * 30)
for i, d in enumerate(data):
    err_init = d["d_rel"] * 100
    err_after = errors[i]
    flag = "✓" if abs(err_after) < 0.3 else ("·" if abs(err_after) < 0.6 else "✗")
    print(f"  {d['nm']:<5} {err_init:>+9.2f}% {err_after:>+9.2f}% {flag}")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  Progression complète :

    Ω_ts strict seul                              : 1.218%
    + c_1(K) Ridge                                : 0.801%
    + b_mean (1 inv)                              : 0.653%
    + b_mean + 2 invariants                       : 0.607%
    + b_mean + 3 invariants                       : 0.591%
    + b_mean + 4 invariants                       : {best_4_cv:.3f}%

  Gain global vs Ω_ts strict : {(1.218 - best_4_cv)/1.218 * 100:+.1f}%

  Si gain marginal du 4ème terme < 1%, on est à la limite des
  invariants topologiques de ce dataset (23 particules).
""")
