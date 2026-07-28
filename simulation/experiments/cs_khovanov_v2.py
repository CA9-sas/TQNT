r"""
Suite de cs_khovanov_invariants.py : ajouter un TROISIÈME invariant
au modèle c_1 + b_mean + rasm_mean + top4g_mean (qui donne 0.607%).

Test toutes les combinaisons à 3 invariants additionnels et identifie
si le mur 0.6% peut être percé.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# Reload tables
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


def cv_rms(X, y, alpha=0.1, k_folds=5):
    errs = []
    for tr, te in kfold(n_data, k=k_folds):
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100


X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)


def feat(cfg, inv, agg='mean'):
    vals = [TOPO[K][inv] for K in cfg]
    if agg == 'mean': return np.mean(vals)
    if agg == 'max': return max(vals)
    if agg == 'sum': return sum(vals)


# Baseline : c_1 + b_mean + rasm_mean + top4g_mean = 0.607%
inv_options = ["rasm", "sm4g", "top4g", "khov", "jspan", "vol", "det", "U", "cr", "b"]
aggs = ['sum', 'mean', 'max']
all_inv = [(i, a) for i in inv_options for a in aggs]


# ───────── Test : combinaisons à 3 invariants en plus de b_mean ─────────
print(f"\n  Cherchons les TROIS meilleurs invariants additionnels à b_mean")
print(f"  Pour les {len(all_inv)} options agrégées au choix.\n")

best_3 = None; best_3_cv = float('inf')
n_tested = 0
for i1 in range(len(all_inv)):
    for i2 in range(i1+1, len(all_inv)):
        for i3 in range(i2+1, len(all_inv)):
            inv1, a1 = all_inv[i1]
            inv2, a2 = all_inv[i2]
            inv3, a3 = all_inv[i3]
            X = np.zeros((n_data, n_knots + 4))
            X[:, :n_knots] = X_single
            for k, d in enumerate(data):
                X[k, n_knots] = np.mean([TOPO[K]["b"] for K in d["cfg"]])
                X[k, n_knots+1] = feat(d["cfg"], inv1, a1)
                X[k, n_knots+2] = feat(d["cfg"], inv2, a2)
                X[k, n_knots+3] = feat(d["cfg"], inv3, a3)
            cv = cv_rms(X, y)
            n_tested += 1
            if cv < best_3_cv:
                best_3_cv = cv
                best_3 = ((inv1, a1), (inv2, a2), (inv3, a3))
                print(f"  NEW: b_mean + {inv1}_{a1} + {inv2}_{a2} + {inv3}_{a3} → {cv:.3f}%")

print(f"\n  Tested {n_tested} triplets")
print(f"\n  ★ Meilleur triplet : {best_3}")
print(f"  RMS CV : {best_3_cv:.3f}%")


# ───────── Vérification : prédictions du meilleur modèle ─────────
print("\n" + "═"*100)
print(f"  Vérification : prédictions particule par particule du meilleur modèle")
print("═"*100)

inv1, a1 = best_3[0]; inv2, a2 = best_3[1]; inv3, a3 = best_3[2]
X = np.zeros((n_data, n_knots + 4))
X[:, :n_knots] = X_single
for k, d in enumerate(data):
    X[k, n_knots] = np.mean([TOPO[K]["b"] for K in d["cfg"]])
    X[k, n_knots+1] = feat(d["cfg"], inv1, a1)
    X[k, n_knots+2] = feat(d["cfg"], inv2, a2)
    X[k, n_knots+3] = feat(d["cfg"], inv3, a3)

# Fit complet pour les c_1 et coefs
r_full = Ridge(alpha=0.1).fit(X, y)
y_pred = X @ r_full.coef_
errors = (y - y_pred) * 100

print(f"\n  Modèle : c_1(K) + b_mean + {inv1}_{a1} + {inv2}_{a2} + {inv3}_{a3}")
print(f"  Coef b_mean    : {r_full.coef_[n_knots]:+.6f}")
print(f"  Coef {inv1}_{a1:<5} : {r_full.coef_[n_knots+1]:+.6f}")
print(f"  Coef {inv2}_{a2:<5} : {r_full.coef_[n_knots+2]:+.6f}")
print(f"  Coef {inv3}_{a3:<5} : {r_full.coef_[n_knots+3]:+.6f}")

print(f"\n  {'nm':<5} {'Ω_ts':>9} {'m_exp':>9} {'Δ init':>9} {'Δ corr':>9}")
print("  " + "-" * 50)
for i, d in enumerate(data):
    err_init = d["d_rel"] * 100
    err_after = errors[i]
    flag = "✓" if abs(err_after) < 0.3 else ("·" if abs(err_after) < 0.6 else "✗")
    print(f"  {d['nm']:<5} {d['om']:>9.2f} {d['m_exp']:>9.2f} "
          f"{err_init:>+8.2f}% {err_after:>+8.2f}% {flag}")

rms_train = np.sqrt(np.mean(errors**2))
print(f"\n  RMS train : {rms_train:.3f}%")
print(f"  RMS CV (5-fold) : {best_3_cv:.3f}%")
