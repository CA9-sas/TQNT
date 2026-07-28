r"""
Option D : ajouter les nombres quantiques non-topologiques (Q, I_3, S, C, B)
au meilleur modèle topologique (4 invariants 4D + b_mean + c_1).

Hypothèse à tester :
  - Si Q, I_3, S, C, B AMÉLIORENT le RMS CV au-delà de 0.566% :
    → il existe de la physique HORS topologie qui contribue à la masse
    → TQNT n'est pas complète sans ces nombres quantiques

  - Si elles n'améliorent PAS :
    → la topologie capture tout
    → confirmation forte de la thèse TQNT « masse = topologie »

Pour chaque particule, on tabule Q, I_3, S (étrangeté), C (charme),
B (beauté), Lk (linking number).
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# Tables topologiques avancées
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


# Particules avec nombres quantiques du PDG
# Format : (nm, config, m_exp, Q, |I_3|, |S|, C, B, |2J|)
# Pour les multiplets (Σ, Ξ, Δ, etc.), on prend valeurs isospin-moyennées
PARTICULES = [
    ("μ⁻",   ["3_1","6_1"],          105.66, -1, 0,    0, 0, 0, 1),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86, -1, 0,    0, 0, 0, 1),
    ("π",    ["3_1","3_1","3_1"],   137.27,   0, 0.5,  0, 0, 0, 0),  # iso avg
    ("η",    ["3_1","5_2","6_1"],   547.86,   0, 0,    0, 0, 0, 0),
    ("K",    ["6_1","6_1","6_1"],   495.65,   0, 0.5,  1, 0, 0, 0),  # iso avg
    ("D",    ["3_1","4_1","6_2"],  1867.0,    0, 0.5,  0, 1, 0, 0),
    ("B",    ["5_2","6_3","6_3"],  5279.0,    0, 0.5,  0, 0, 1, 0),
    ("ρ",    ["8_1","8_1"],         778.85,   0, 0.5,  0, 0, 0, 2),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9,    0, 0,    0, 0, 0, 2),  # cc̄
    ("Υ",    ["7_1","7_3","9_1"],  9460.0,    0, 0,    0, 0, 0, 2),  # bb̄
    ("p",    ["3_1","7_2","7_2"],   938.27,   1, 0.5,  0, 0, 0, 1),
    ("n",    ["3_1","7_2","7_2"],   939.57,   0, 0.5,  0, 0, 0, 1),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68,   0, 0,    1, 0, 0, 1),
    ("Σ",    ["5_1","7_7"],        1193.0,    0, 1,    1, 0, 0, 1),  # iso avg
    ("Ξ",    ["5_1","6_3"],        1318.0,    0, 0.5,  2, 0, 0, 1),  # iso avg
    ("Δ",    ["5_2","7_2","7_2"],  1232.0,    0, 1.5,  0, 0, 0, 3),  # iso avg
    ("Σ*",   ["10_1"],             1383.0,    0, 1,    1, 0, 0, 3),
    ("Ξ*",   ["7_1","8_1"],        1532.0,    0, 0.5,  2, 0, 0, 3),
    ("Ω-",   ["5_1","7_2","7_2"],  1672.0,   -1, 0,    3, 0, 0, 3),
    ("D0",   ["3_1","4_1","6_2"],  1864.84,   0, 0.5,  0, 1, 0, 0),
    ("Λc",   ["5_1","5_1","7_2"],  2286.46,   1, 0,    0, 1, 0, 1),
    ("Σc",   ["4_1","5_1","7_1"],  2453.0,    0, 1,    0, 1, 0, 1),  # iso avg
    ("Ωc",   ["7_3","7_7"],        2695.0,    0, 0,    2, 1, 0, 1),
]


data = []
for nm, cfg, m_exp, Q, I3, S, C, B, two_J in PARTICULES:
    om = omega_strict(cfg)
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                  "d_rel": (m_exp - om) / m_exp,
                  "Q": Q, "I3": I3, "S": S, "C": C, "B": B,
                  "two_J": two_J, "n_phys": two_J + 1})

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


# Pré-calcul du baseline (modèle à 4 invariants topologiques + b_mean + c_1)
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


# Baseline : meilleur quadruplet du test précédent
X_base = np.zeros((n_data, n_knots + 5))
X_base[:, :n_knots] = X_single
X_base[:, n_knots] = feat_col("b", "mean")        # b_mean
X_base[:, n_knots+1] = feat_col("sm4g", "max")     # sm4g_max
X_base[:, n_knots+2] = feat_col("top4g", "mean")   # top4g_mean
X_base[:, n_knots+3] = feat_col("top4g", "max")    # top4g_max
X_base[:, n_knots+4] = feat_col("vol", "mean")     # vol_mean

folds_idx = list(kfold(n_data, k=5))

def cv_rms(X, alpha=0.1):
    errs = []
    for tr, te in folds_idx:
        r = Ridge(alpha=alpha).fit(X[tr], y[tr])
        errs.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
    return np.mean(errs) * 100

rms_baseline = cv_rms(X_base)
print(f"\n  Baseline (c_1 + b_mean + 4 top inv) : RMS CV = {rms_baseline:.4f}%")


# ───────── Test : ajouter UN nombre quantique ─────────
print("\n" + "═"*100)
print("  Ajout d'UN nombre quantique non-topologique à la baseline")
print("═"*100)

qm_features = {
    "Q":     [d["Q"] for d in data],
    "|Q|":   [abs(d["Q"]) for d in data],
    "Q²":    [d["Q"]**2 for d in data],
    "I3":    [d["I3"] for d in data],
    "|I3|":  [abs(d["I3"]) for d in data],
    "I3²":   [d["I3"]**2 for d in data],
    "S":     [d["S"] for d in data],
    "|S|":   [abs(d["S"]) for d in data],
    "C":     [d["C"] for d in data],
    "|C|":   [abs(d["C"]) for d in data],
    "B_qm":  [d["B"] for d in data],
    "S+C+B": [d["S"] + d["C"] + d["B"] for d in data],
    "Q+I3":  [d["Q"] + d["I3"] for d in data],
    "Q·I3":  [d["Q"] * d["I3"] for d in data],
    "(2J+1)": [d["n_phys"] for d in data],
    "n_phys-1": [d["n_phys"] - 1 for d in data],
}

print(f"\n  {'QM number':<15} {'β extrait':>12} {'RMS CV':>9} {'gain':>7}")
print("  " + "-" * 55)

best_qm = None; best_qm_cv = rms_baseline
for label, vals in qm_features.items():
    X = np.zeros((n_data, X_base.shape[1] + 1))
    X[:, :X_base.shape[1]] = X_base
    X[:, -1] = vals
    cv = cv_rms(X)
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[-1]
    gain = (rms_baseline - cv) / rms_baseline * 100
    flag = "★★" if cv < 0.5 else ("★" if cv < rms_baseline - 0.02 else " ")
    print(f"  {label:<15} {beta:>+12.6f} {cv:>8.4f}% {gain:>+6.1f}% {flag}")
    if cv < best_qm_cv:
        best_qm_cv = cv; best_qm = label

print(f"\n  ★ Meilleur QM ajouté : {best_qm} → {best_qm_cv:.4f}%")


# ───────── Test : combinaison de 2 nombres quantiques ─────────
print("\n" + "═"*100)
print("  Combinaisons de DEUX nombres quantiques")
print("═"*100)

qm_keys = list(qm_features.keys())
best_pair = None; best_pair_cv = best_qm_cv
for i, k1 in enumerate(qm_keys):
    for k2 in qm_keys[i+1:]:
        X = np.zeros((n_data, X_base.shape[1] + 2))
        X[:, :X_base.shape[1]] = X_base
        X[:, -2] = qm_features[k1]
        X[:, -1] = qm_features[k2]
        cv = cv_rms(X)
        if cv < best_pair_cv - 0.005:
            best_pair_cv = cv
            best_pair = (k1, k2)
            print(f"  {k1:<10} + {k2:<10} → {cv:.4f}%")

print(f"\n  Meilleure paire QM : {best_pair} → {best_pair_cv:.4f}%")


# ───────── Verdict final ─────────
print("\n" + "═"*100)
print("  VERDICT")
print("═"*100)
print(f"""
  Baseline topologique pure (4 inv 4D + b_mean + c_1) : {rms_baseline:.4f}%

  Meilleur ajout d'1 nombre quantique : {best_qm_cv:.4f}% (gain {(rms_baseline-best_qm_cv)/rms_baseline*100:+.1f}%)
  Meilleur ajout de 2 nombres quantiques : {best_pair_cv:.4f}%

  Interprétation :
""")

if best_pair_cv < rms_baseline - 0.05:
    print("  → Les nombres quantiques apportent une INFORMATION SUPPLÉMENTAIRE")
    print("    significative au-delà de la topologie pure.")
    print("    TQNT topologique seule n'est PAS complète.")
elif best_pair_cv < rms_baseline - 0.02:
    print("  → Effet partiel : amélioration mais modeste.")
    print("    Topologie capture l'essentiel ; QM ajoute une nuance.")
else:
    print("  → Les nombres quantiques n'apportent PAS d'information")
    print("    significative au-delà de la topologie pure.")
    print("    Confirmation forte que TQNT capture toute la masse via topologie.")
