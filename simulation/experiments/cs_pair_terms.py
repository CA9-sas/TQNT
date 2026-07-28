r"""
EXTENSION du modèle c_1(K) avec TERMES CROISÉS c_2(K_i, K_j).

Modèle :
  Δ_L = Σ_i c_1(K_i)·count_i  +  Σ_{i<j} c_2(K_i, K_j)·count_{ij}  +  ε

où count_{ij} = nombre de paires non-ordonnées (K_i, K_j) dans le link
(pour K_i = K_j : combinaison C(n_i, 2) ; pour K_i ≠ K_j : n_i × n_j).

Ces termes croisés reflètent l'interaction Biot-Savart inter-composantes
mise en évidence par l'analyse de P23.

Test :
  1. Construire le design avec features c_1(K) + c_2(pair)
  2. Ridge regularization
  3. Validation 5-fold pour mesurer le gain RÉEL
  4. Identifier si certaines paires sont dominantes
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
    def fit(self, X, y):
        n_feat = X.shape[1]
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(n_feat)
        b = X.T @ y
        try:
            self.coef_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]
        return self
    def predict(self, X):
        return X @ self.coef_


def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1
    start = 0
    for size in sizes:
        test = idx[start:start+size]
        train = np.concatenate([idx[:start], idx[start+size:]])
        yield train, test
        start += size


# Paramètres Ω_ts
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
               "b": braid_index(K), "det": jones_determinant(pj)}
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


# Calcul des résidus
data = []
for nm, cfg, m_exp in PARTICULES:
    om = omega_strict(cfg)
    d_rel = (m_exp - om) / m_exp
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                  "d_rel": d_rel})

rms_init = float(np.sqrt(np.mean([d["d_rel"]**2 for d in data]))) * 100
print(f"\n  RMS Ω_ts strict initial : {rms_init:.3f}%")


# ───────── Construire le set des paires distinctes apparaissant ─────────
all_pairs = set()
for d in data:
    cfg = d["cfg"]
    # Paires inter-différentes
    for K1, K2 in combinations(set(cfg), 2):
        all_pairs.add(tuple(sorted([K1, K2])))
    # Paires identiques (K, K) si count >= 2
    counter = Counter(cfg)
    for K, ct in counter.items():
        if ct >= 2:
            all_pairs.add((K, K))

PAIRS = sorted(all_pairs)
print(f"\n  Paires distinctes dans le dataset : {len(PAIRS)}")
for p in PAIRS[:10]:
    print(f"    {p[0]} ↔ {p[1]}")
print(f"    ... ({len(PAIRS) - 10} autres)" if len(PAIRS) > 10 else "")


def count_pairs(cfg, pairs_list):
    """Compte des paires non-ordonnées dans cfg pour chaque paire de pairs_list."""
    counts = np.zeros(len(pairs_list))
    counter = Counter(cfg)
    for idx, (K1, K2) in enumerate(pairs_list):
        if K1 == K2:
            ct = counter.get(K1, 0)
            counts[idx] = ct * (ct - 1) / 2  # C(n, 2)
        else:
            counts[idx] = counter.get(K1, 0) * counter.get(K2, 0)
    return counts


# ───────── Matrice de design étendue ─────────
n_data = len(data)
n_knots = len(KNOTS)
n_pairs = len(PAIRS)
n_feat_total = n_knots + n_pairs
print(f"\n  Features : {n_knots} (single) + {n_pairs} (pairs) = {n_feat_total}")
print(f"  Échantillon : {n_data} particules")
print(f"  Sur-paramétrisation : {n_feat_total/n_data:.1f}× — régularisation essentielle")

X_full = np.zeros((n_data, n_feat_total))
y = np.zeros(n_data)
for i, d in enumerate(data):
    counts_single = Counter(d["cfg"])
    for j, K in enumerate(KNOTS):
        X_full[i, j] = counts_single.get(K, 0)
    # Termes croisés
    pair_counts = count_pairs(d["cfg"], PAIRS)
    X_full[i, n_knots:] = pair_counts
    y[i] = d["d_rel"]


# ───────── Comparaison : single-only vs single+pairs ─────────
print("\n" + "═"*100)
print("  Comparaison : single (c_1) vs single + pairs (c_1 + c_2)")
print("═"*100)

X_single = X_full[:, :n_knots]
X_both = X_full

for label, X in [("single (19 features)", X_single),
                  ("single + pairs ({} features)".format(n_feat_total), X_both)]:
    print(f"\n  {label} :")
    print(f"  {'alpha':<8} {'RMS train':>10} {'RMS CV':>10}")
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha).fit(X, y)
        rms_train = np.sqrt(np.mean((X @ ridge.coef_ - y)**2)) * 100
        # CV
        cv_errs = []
        for train_idx, test_idx in kfold_indices(n_data, k=5):
            r = Ridge(alpha=alpha).fit(X[train_idx], y[train_idx])
            err = np.sqrt(np.mean((X[test_idx] @ r.coef_ - y[test_idx])**2))
            cv_errs.append(err)
        rms_cv = np.mean(cv_errs) * 100
        print(f"  {alpha:<8.4f} {rms_train:>9.4f}% {rms_cv:>9.4f}%")


# ───────── Sélection meilleur alpha pour single+pairs ─────────
best_alpha = 0.1; best_cv = float('inf')
for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    cv_errs = []
    for train_idx, test_idx in kfold_indices(n_data, k=5):
        r = Ridge(alpha=alpha).fit(X_both[train_idx], y[train_idx])
        err = np.sqrt(np.mean((X_both[test_idx] @ r.coef_ - y[test_idx])**2))
        cv_errs.append(err)
    cv_rms = np.mean(cv_errs) * 100
    if cv_rms < best_cv:
        best_cv = cv_rms; best_alpha = alpha

print(f"\n  Meilleur alpha (single+pairs) : {best_alpha} (CV RMS = {best_cv:.3f}%)")


# Fit final
ridge_both = Ridge(alpha=best_alpha).fit(X_both, y)
coeffs = ridge_both.coef_
c1_final = dict(zip(KNOTS, coeffs[:n_knots]))
c2_final = dict(zip(PAIRS, coeffs[n_knots:]))


# ───────── Identification des paires dominantes ─────────
print("\n" + "═"*100)
print("  Paires c_2(K_i, K_j) les plus significatives")
print("═"*100)

pairs_sorted = sorted(c2_final.items(), key=lambda kv: abs(kv[1]), reverse=True)
print(f"\n  Top 10 paires par |c_2| :")
print(f"  {'paire':<18} {'c_2':>10} {'apparait dans':<40}")
for (K1, K2), c2 in pairs_sorted[:10]:
    # Identifier les particules contenant cette paire
    particles_with_pair = []
    for d in data:
        cc = Counter(d["cfg"])
        if K1 == K2:
            if cc.get(K1, 0) >= 2:
                particles_with_pair.append(d["nm"])
        else:
            if cc.get(K1, 0) >= 1 and cc.get(K2, 0) >= 1:
                particles_with_pair.append(d["nm"])
    pp_str = ", ".join(particles_with_pair[:5])
    if len(particles_with_pair) > 5: pp_str += f"... ({len(particles_with_pair)})"
    print(f"  {K1}↔{K2:<10} {c2:>+10.5f} {pp_str:<40}")


# ───────── Résultat final ─────────
print("\n" + "═"*100)
print(f"  Validation finale (alpha={best_alpha})")
print("═"*100)

y_pred_both = X_both @ coeffs
errors_pct = (y - y_pred_both) * 100

# RMS train et CV
ridge_single_final = Ridge(alpha=0.01).fit(X_single, y)
y_pred_single = X_single @ ridge_single_final.coef_
rms_single_train = np.sqrt(np.mean((y - y_pred_single)**2)) * 100

# CV pour single
cv_single = []
for train_idx, test_idx in kfold_indices(n_data, k=5):
    r = Ridge(alpha=0.01).fit(X_single[train_idx], y[train_idx])
    err = np.sqrt(np.mean((X_single[test_idx] @ r.coef_ - y[test_idx])**2))
    cv_single.append(err)
rms_single_cv = np.mean(cv_single) * 100

rms_both_train = np.sqrt(np.mean((y - y_pred_both)**2)) * 100

print(f"\n  Modèle             RMS train  RMS CV")
print(f"  Ω_ts strict        {rms_init:>8.3f}%  {rms_init:>6.3f}%")
print(f"  + c_1(K) (single)  {rms_single_train:>8.3f}%  {rms_single_cv:>6.3f}%")
print(f"  + c_2 (pairs)      {rms_both_train:>8.3f}%  {best_cv:>6.3f}%")

gain_cv = (rms_single_cv - best_cv) / rms_single_cv * 100
print(f"\n  Gain CV ajouté par paires : {gain_cv:+.1f}%")

print(f"\n  Détail particule par particule (single+pairs) :")
print(f"  {'nm':<5} {'Δ initial':>10} {'Δ corr':>10}")
for i, d in enumerate(data):
    err_init = d["d_rel"]*100
    err_after = errors_pct[i]
    flag = "✓" if abs(err_after) < 0.3 else ("·" if abs(err_after) < 0.5 else "")
    print(f"  {d['nm']:<5} {err_init:>+9.2f}% {err_after:>+9.2f}% {flag}")
