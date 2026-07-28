r"""
Direction D2 : invariants topologiques sophistiqués (Khovanov, Rasmussen,
slice genus différentiel, HOMFLY, Conway).

Tabulés depuis Knotinfo (knotinfo.math.indiana.edu) :
  - Rasmussen s-invariant
  - Smooth 4-genus (g_4^smooth)
  - Topological 4-genus (g_4^top)
  - Khovanov homology rank (somme des Betti)
  - Concordance order
  - Span du polynôme de Jones
  - q-degree minimum du Jones
  - q-degree maximum du Jones
  - Volume hyperbolique normalisé (vol/cr)
  - 2-bridge type p/q

Test : ajouter chaque invariant agrégé (sum/mean/max) au modèle
c_1 + b_mean (baseline 0.653% CV) et regarder lequel descend.

But : trouver UN invariant qui perce le mur 0.65%.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ───────── Tables d'invariants (Knotinfo) ─────────
# Rasmussen s-invariant
RASMUSSEN = {
    "3_1": -2, "4_1": 0, "5_1": -4, "5_2": -2, "6_1": 0,
    "6_2": -2, "6_3": 0, "7_1": -6, "7_2": -2, "7_3": -4,
    "7_4": -2, "7_5": -4, "7_6": -2, "7_7": 0,
    "8_1": 0, "8_2": -4, "8_3": 0, "9_1": -8, "10_1": 0,
}

# Smooth 4-genus
SM4G = {
    "3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1, "6_1": 1,
    "6_2": 2, "6_3": 2, "7_1": 3, "7_2": 1, "7_3": 2,
    "7_4": 1, "7_5": 2, "7_6": 2, "7_7": 2,
    "8_1": 1, "8_2": 3, "8_3": 1, "9_1": 4, "10_1": 1,
}

# Topological 4-genus
TOP4G = {
    "3_1": 1, "4_1": 1, "5_1": 2, "5_2": 1, "6_1": 0,
    "6_2": 2, "6_3": 2, "7_1": 3, "7_2": 1, "7_3": 2,
    "7_4": 1, "7_5": 2, "7_6": 2, "7_7": 2,
    "8_1": 0, "8_2": 3, "8_3": 1, "9_1": 4, "10_1": 1,
}

# Khovanov rank total (dim totale)
KHOV_RANK = {
    "3_1": 4, "4_1": 5, "5_1": 6, "5_2": 7, "6_1": 9,
    "6_2": 11, "6_3": 13, "7_1": 8, "7_2": 11, "7_3": 13,
    "7_4": 15, "7_5": 15, "7_6": 17, "7_7": 19,
    "8_1": 13, "8_2": 17, "8_3": 17, "9_1": 10, "10_1": 17,
}

# Span du polynôme de Jones (différence des degrés)
JONES_SPAN = {
    "3_1": 3, "4_1": 4, "5_1": 5, "5_2": 5, "6_1": 6,
    "6_2": 6, "6_3": 6, "7_1": 7, "7_2": 7, "7_3": 7,
    "7_4": 7, "7_5": 7, "7_6": 7, "7_7": 7,
    "8_1": 8, "8_2": 8, "8_3": 8, "9_1": 9, "10_1": 10,
}

# 2-bridge type p/q (just p/q ratio as float; all our knots are 2-bridge)
# Format : (p, q) with K = K(p/q)
BRIDGE_RATIONAL = {
    "3_1": (3, 1),  "4_1": (5, 2),  "5_1": (5, 1),  "5_2": (7, 3),
    "6_1": (9, 4),  "6_2": (11, 3), "6_3": (13, 5),
    "7_1": (7, 1),  "7_2": (11, 4), "7_3": (13, 4),
    "7_4": (15, 4), "7_5": (17, 5), "7_6": (19, 8), "7_7": (21, 8),
    "8_1": (13, 5), "8_2": (17, 4), "8_3": (17, 8),
    "9_1": (9, 1),  "10_1": (21, 8),
}

# Hyperbolic volume (0 pour torus)
HYP_VOL = {
    "3_1": 0.0, "4_1": 2.02988, "5_1": 0.0, "5_2": 2.82812,
    "6_1": 3.16396, "6_2": 4.40083, "6_3": 5.69302,
    "7_1": 0.0, "7_2": 3.33174, "7_3": 4.59213, "7_4": 5.13794,
    "7_5": 6.44354, "7_6": 7.08493, "7_7": 7.64338,
    "8_1": 3.42721, "8_2": 7.41862, "8_3": 8.13491,
    "9_1": 0.0, "10_1": 4.05766,
}


# ───────── Setup TQNT ─────────
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
    p, q = BRIDGE_RATIONAL[K]
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "U": U_KT[K][0]*U_KT[K][1],
               "rasm": RASMUSSEN[K], "sm4g": SM4G[K], "top4g": TOP4G[K],
               "khov": KHOV_RANK[K], "jspan": JONES_SPAN[K],
               "bridge_p": p, "bridge_q": q, "bridge_ratio": q/p,
               "hyp_vol": HYP_VOL[K], "is_torus": 1 if HYP_VOL[K] == 0.0 else 0}
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
    ("μ⁻",   ["3_1","6_1"],          105.66),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86),
    ("π",    ["3_1","3_1","3_1"],   137.27),
    ("η",    ["3_1","5_2","6_1"],   547.86),
    ("K",    ["6_1","6_1","6_1"],   495.65),
    ("D",    ["3_1","4_1","6_2"],  1867.0),
    ("B",    ["5_2","6_3","6_3"],  5279.0),
    ("ρ",    ["8_1","8_1"],         778.85),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9),
    ("Υ",    ["7_1","7_3","9_1"],  9460.0),
    ("p",    ["3_1","7_2","7_2"],   938.27),
    ("n",    ["3_1","7_2","7_2"],   939.57),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68),
    ("Σ",    ["5_1","7_7"],        1193.0),
    ("Ξ",    ["5_1","6_3"],        1318.0),
    ("Δ",    ["5_2","7_2","7_2"],  1232.0),
    ("Σ*",   ["10_1"],             1383.0),
    ("Ξ*",   ["7_1","8_1"],        1532.0),
    ("Ω-",   ["5_1","7_2","7_2"],  1672.0),
    ("D0",   ["3_1","4_1","6_2"],  1864.84),
    ("Λc",   ["5_1","5_1","7_2"],  2286.46),
    ("Σc",   ["4_1","5_1","7_1"],  2453.0),
    ("Ωc",   ["7_3","7_7"],        2695.0),
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
        n_feat = X.shape[1]
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(n_feat)
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


# Baseline single + b_mean
X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)

def b_mean(cfg): return np.mean([TOPO[K]["b"] for K in cfg])

X_ref = np.zeros((n_data, n_knots + 1))
X_ref[:, :n_knots] = X_single
for i, d in enumerate(data):
    X_ref[i, n_knots] = b_mean(d["cfg"])
rms_ref = cv_rms(X_ref, y)


# ───────── Test : ajouter chaque invariant Khovanov/Rasmussen ─────────
print(f"\n  Référence : c_1 + b_mean = {rms_ref:.3f}% RMS CV")
print("\n" + "═"*100)
print("  Ajout d'un seul invariant Khovanov/Rasmussen avancé")
print("═"*100)
print(f"\n  {'Invariant':<30} {'β':>12} {'RMS CV':>9}")
print("  " + "-" * 55)

inv_list = ["rasm", "sm4g", "top4g", "khov", "jspan",
            "bridge_p", "bridge_q", "bridge_ratio",
            "hyp_vol", "is_torus"]
aggs = ['sum', 'mean', 'max']

best_inv = None; best_inv_cv = rms_ref
results_inv = []
for inv in inv_list:
    for agg in aggs:
        X = np.zeros((n_data, n_knots + 2))
        X[:, :n_knots+1] = X_ref
        for i, d in enumerate(data):
            vals = [TOPO[K][inv] for K in d["cfg"]]
            if agg == 'sum': X[i, n_knots+1] = sum(vals)
            elif agg == 'mean': X[i, n_knots+1] = np.mean(vals)
            elif agg == 'max': X[i, n_knots+1] = max(vals)
        cv = cv_rms(X, y)
        r_full = Ridge(alpha=0.1).fit(X, y)
        beta = r_full.coef_[n_knots+1]
        results_inv.append((f"{inv}_{agg}", beta, cv))
        flag = "★★" if cv < 0.5 else ("★" if cv < rms_ref - 0.02 else " ")
        if cv < rms_ref - 0.01:
            print(f"  {inv+'_'+agg:<30} {beta:>+12.6f} {cv:>8.3f}% {flag}")
        if cv < best_inv_cv:
            best_inv_cv = cv; best_inv = f"{inv}_{agg}"

print(f"\n  ★ Meilleur invariant : {best_inv} (RMS CV = {best_inv_cv:.3f}%)")


# ───────── Combiner b_mean + 2 invariants ─────────
print("\n" + "═"*100)
print("  Tests : ajouter DEUX invariants simultanément à la baseline")
print("═"*100)

print(f"\n  Top combinaisons trouvées :")
best_combo = None; best_combo_cv = rms_ref
inv_aggs = [(i, a) for i in inv_list for a in aggs]
for k1, (i1, a1) in enumerate(inv_aggs):
    for (i2, a2) in inv_aggs[k1+1:]:
        X = np.zeros((n_data, n_knots + 3))
        X[:, :n_knots+1] = X_ref
        for i, d in enumerate(data):
            vals1 = [TOPO[K][i1] for K in d["cfg"]]
            vals2 = [TOPO[K][i2] for K in d["cfg"]]
            if a1 == 'sum': X[i, n_knots+1] = sum(vals1)
            elif a1 == 'mean': X[i, n_knots+1] = np.mean(vals1)
            elif a1 == 'max': X[i, n_knots+1] = max(vals1)
            if a2 == 'sum': X[i, n_knots+2] = sum(vals2)
            elif a2 == 'mean': X[i, n_knots+2] = np.mean(vals2)
            elif a2 == 'max': X[i, n_knots+2] = max(vals2)
        cv = cv_rms(X, y)
        if cv < best_combo_cv - 0.01:
            best_combo_cv = cv
            best_combo = (f"{i1}_{a1}", f"{i2}_{a2}")
            print(f"    {i1+'_'+a1:<22} + {i2+'_'+a2:<22} → {cv:>6.3f}% ★")


print(f"\n  Meilleure combinaison : {best_combo}")
print(f"  RMS CV : {best_combo_cv:.3f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  Hiérarchie des RMS CV obtenues :

    Ω_ts strict seul                              : 1.218%
    + c_1(K) Ridge                                : 0.801%
    + b_mean                                       : 0.653%
    + b_mean + meilleur invariant (1)             : {best_inv_cv:.3f}%  ({best_inv})
    + b_mean + meilleure paire d'invariants       : {best_combo_cv:.3f}%

  Gain net vs baseline (b_mean) : {(rms_ref - min(best_inv_cv, best_combo_cv))/rms_ref * 100:+.1f}%
  Gain net vs Ω_ts strict         : {(1.218 - min(best_inv_cv, best_combo_cv))/1.218 * 100:+.1f}%
""")
