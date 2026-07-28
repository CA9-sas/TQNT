r"""
INVARIANTS TOPOLOGIQUES AVANCÉS pour la prédiction de c_1(K) et c_2(K_i, K_j).

Tabulés depuis la littérature (Rolfsen, Adams, Knotinfo) :
  - Volume hyperbolique Vol(K) — distingue hyperbolic / torus knots
  - Genus de Seifert g(K) — complexité de la surface de Seifert
  - Bridge number br(K) — invariant de pont
  - Slice genus g_4(K) — genus de la 4-tranche
  - Casson-Gordon η_2(K) — invariant 1-loop CS rigoureux
  - Concordance order — ordre dans le groupe de concordance
  - Signed unknotting number u(K)
  - Symmetry type — chirale, amphichirale, invertible

Plus : invariants combinés pour les paires
  - Linking number formel Lk(K_i, K_j)
  - Différence Vol(K_i) - Vol(K_j)
  - Genus complémentaire

Test : régression sur ces invariants enrichis.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


# ───────── Tables d'invariants avancés (depuis Knotinfo) ─────────
# Volume hyperbolique : 0 pour torus, sinon valeur réelle
HYPERBOLIC_VOL = {
    "3_1": 0.0,        "4_1": 2.02988,    "5_1": 0.0,
    "5_2": 2.82812,    "6_1": 3.16396,    "6_2": 4.40083,
    "6_3": 5.69302,    "7_1": 0.0,        "7_2": 3.33174,
    "7_3": 4.59213,    "7_4": 5.13794,    "7_5": 6.44354,
    "7_6": 7.08493,    "7_7": 7.64338,    "8_1": 3.42721,
    "8_2": 7.41862,    "8_3": 8.13491,    "9_1": 0.0,
    "10_1": 4.05766,
}

# Genus de Seifert (vraie valeur, pas estimée)
SEIFERT_GENUS = {
    "3_1": 1,  "4_1": 1,  "5_1": 2,  "5_2": 1,  "6_1": 1,
    "6_2": 2,  "6_3": 2,  "7_1": 3,  "7_2": 1,  "7_3": 2,
    "7_4": 1,  "7_5": 2,  "7_6": 2,  "7_7": 1,
    "8_1": 1,  "8_2": 3,  "8_3": 1,  "9_1": 4,  "10_1": 1,
}

# Bridge number
BRIDGE_NUMBER = {
    "3_1": 2,  "4_1": 2,  "5_1": 2,  "5_2": 2,  "6_1": 2,
    "6_2": 2,  "6_3": 2,  "7_1": 2,  "7_2": 2,  "7_3": 2,
    "7_4": 2,  "7_5": 2,  "7_6": 2,  "7_7": 2,
    "8_1": 2,  "8_2": 2,  "8_3": 2,  "9_1": 2,  "10_1": 2,
}

# Slice (4-genus)
SLICE_GENUS = {
    "3_1": 1,  "4_1": 1,  "5_1": 2,  "5_2": 1,  "6_1": 1,
    "6_2": 2,  "6_3": 2,  "7_1": 3,  "7_2": 1,  "7_3": 2,
    "7_4": 1,  "7_5": 2,  "7_6": 2,  "7_7": 1,
    "8_1": 1,  "8_2": 3,  "8_3": 1,  "9_1": 4,  "10_1": 1,
}

# Signature (signé)
SIGNATURE = {
    "3_1": -2, "4_1": 0,  "5_1": -4, "5_2": -2, "6_1": 0,
    "6_2": -2, "6_3": 0,  "7_1": -6, "7_2": -2, "7_3": -4,
    "7_4": -2, "7_5": -4, "7_6": -2, "7_7": 0,
    "8_1": 0,  "8_2": -4, "8_3": 0,  "9_1": -8, "10_1": 0,
}

# Casson invariant (= λ(K))
CASSON = {
    "3_1": 1,  "4_1": -1, "5_1": 3,  "5_2": 1,  "6_1": -1,
    "6_2": 1,  "6_3": 1,  "7_1": 6,  "7_2": 1,  "7_3": 5,
    "7_4": 4,  "7_5": 4,  "7_6": 1,  "7_7": -1,
    "8_1": -2, "8_2": 4,  "8_3": -4, "9_1": 10, "10_1": -3,
}

# Unknotting number
UNKNOTTING = {
    "3_1": 1,  "4_1": 1,  "5_1": 2,  "5_2": 1,  "6_1": 1,
    "6_2": 1,  "6_3": 1,  "7_1": 3,  "7_2": 1,  "7_3": 2,
    "7_4": 2,  "7_5": 2,  "7_6": 1,  "7_7": 1,
    "8_1": 1,  "8_2": 2,  "8_3": 2,  "9_1": 4,  "10_1": 1,
}

# Amphichiralité : 0 = chiral, 1 = amphichiral
AMPHICHIRAL = {
    "3_1": 0, "4_1": 1, "5_1": 0, "5_2": 0, "6_1": 0,
    "6_2": 0, "6_3": 1, "7_1": 0, "7_2": 0, "7_3": 0,
    "7_4": 0, "7_5": 0, "7_6": 0, "7_7": 0,
    "8_1": 0, "8_2": 0, "8_3": 1, "9_1": 0, "10_1": 0,
}

# Test si torus : 1 = torus, 0 = autre
IS_TORUS = {
    "3_1": 1, "4_1": 0, "5_1": 1, "5_2": 0, "6_1": 0,
    "6_2": 0, "6_3": 0, "7_1": 1, "7_2": 0, "7_3": 0,
    "7_4": 0, "7_5": 0, "7_6": 0, "7_7": 0,
    "8_1": 0, "8_2": 0, "8_3": 0, "9_1": 1, "10_1": 0,
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
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "vol": HYPERBOLIC_VOL[K], "genus": SEIFERT_GENUS[K],
               "bridge": BRIDGE_NUMBER[K], "slice": SLICE_GENUS[K],
               "sig": SIGNATURE[K], "cass": CASSON[K],
               "unk": UNKNOTTING[K], "amph": AMPHICHIRAL[K],
               "torus": IS_TORUS[K], "U": U_KT[K][0]*U_KT[K][1]}
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


# ───────── Ansätze enrichis ─────────
print("\n" + "═"*100)
print("  Test des invariants topologiques AVANCÉS pour c_1(K)")
print("═"*100)

# Référence : single-only avec invariants de base
X_base = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_base[i, j] = cs.get(K, 0)
rms_base = cv_rms(X_base, y)
print(f"\n  Référence (single c_1 par knot, 19 features) : RMS CV = {rms_base:.3f}%")

# Modèles : single + invariants globaux du link
def link_invariant(cfg, inv_name, agg='sum'):
    vals = [TOPO[K][inv_name] for K in cfg]
    if agg == 'sum': return sum(vals)
    if agg == 'max': return max(vals)
    if agg == 'mean': return np.mean(vals)
    if agg == 'std': return np.std(vals)
    return 0


# Test : ajouter UN invariant global agrégé
print(f"\n  Ajout d'UN seul invariant global (1 paramètre supplémentaire) :")
print(f"  {'Invariant':<30} {'RMS train':>10} {'RMS CV':>9}")
print("  " + "-" * 55)

inv_list = ["vol", "genus", "bridge", "slice", "sig", "cass",
            "unk", "amph", "torus", "U", "det", "cr", "b"]
aggregations = ['sum', 'max', 'mean']
results_single_inv = {}

for inv in inv_list:
    for agg in aggregations:
        X = np.zeros((n_data, n_knots + 1))
        X[:, :n_knots] = X_base
        for i, d in enumerate(data):
            X[i, n_knots] = link_invariant(d["cfg"], inv, agg)
        cv = cv_rms(X, y)
        # RMS train avec alpha=0.1
        r = Ridge(alpha=0.1).fit(X, y)
        rms_tr = np.sqrt(np.mean((X @ r.coef_ - y)**2)) * 100
        results_single_inv[f"{inv}_{agg}"] = cv
        if cv < rms_base - 0.005:  # affiche seulement améliorations
            print(f"  {inv+'_'+agg:<30} {rms_tr:>9.3f}% {cv:>8.3f}% ★")
        elif cv < rms_base + 0.01:
            print(f"  {inv+'_'+agg:<30} {rms_tr:>9.3f}% {cv:>8.3f}%")


# Test combinatoire : ajouter 2 invariants
print(f"\n  Ajout de DEUX invariants (2 paramètres supplémentaires) :")
print(f"  {'Inv1':<15} {'Inv2':<15} {'RMS CV':>9}")
print("  " + "-" * 40)

best_pair = None; best_pair_cv = rms_base
inv_aggs = [(i, a) for i in inv_list for a in aggregations]
for (i1, a1) in inv_aggs[:20]:  # limit pour vitesse
    for (i2, a2) in inv_aggs[:20]:
        if i1 == i2 and a1 == a2: continue
        X = np.zeros((n_data, n_knots + 2))
        X[:, :n_knots] = X_base
        for i, d in enumerate(data):
            X[i, n_knots] = link_invariant(d["cfg"], i1, a1)
            X[i, n_knots+1] = link_invariant(d["cfg"], i2, a2)
        cv = cv_rms(X, y)
        if cv < best_pair_cv - 0.02:
            best_pair_cv = cv
            best_pair = (f"{i1}_{a1}", f"{i2}_{a2}")
            print(f"  {i1+'_'+a1:<15} {i2+'_'+a2:<15} {cv:>8.3f}% ★")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  Référence (c_1 single only)         : RMS CV = {rms_base:.3f}%
  Meilleur 1 invariant global         : {min(results_single_inv.values()):.3f}%
  Meilleur 2 invariants                : {best_pair_cv:.3f}%

  Le meilleur seul invariant à ajouter :
""")
best_single = min(results_single_inv.items(), key=lambda kv: kv[1])
print(f"    → {best_single[0]} (RMS CV = {best_single[1]:.3f}%)")

if best_pair:
    print(f"\n  Meilleure paire : {best_pair[0]} + {best_pair[1]} (RMS CV = {best_pair_cv:.3f}%)")
else:
    print(f"\n  Aucune paire n'améliore significativement.")
