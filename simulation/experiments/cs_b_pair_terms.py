r"""
Direction 2 : combiner b_mean (qui marche) avec des termes croisés en b.

Modèle de base : Δ = Σ_i c_1(K_i)·n_i + α · b_mean(L)  →  RMS CV = 0.653%

On ajoute maintenant des termes croisés inspirés de Reshetikhin-Turaev :
  c_2(K_i, K_j) basé sur b(K_i), b(K_j)

Ansätze testés :
  G1 : β · Σ b_i · b_j · count_ij       (produit, naturel CS)
  G2 : β · Σ max(b_i, b_j) · count_ij   (rep dominante)
  G3 : β · Σ (b_i + b_j) · count_ij     (somme)
  G4 : β · Σ |b_i - b_j| · count_ij     (asymétrie)
  G5 : β · Σ √(b_i · b_j) · count_ij    (moyenne géométrique)
  G6 : β · Σ (b_i * b_j - 1) · count_ij  (centré sur produit-1)

But : descendre sous 0.65% en CV avec une structure motivée par
la représentation SU(2)_k.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


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
               "U": U_KT[K][0]*U_KT[K][1]}
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


# ───────── Construction des matrices ─────────
# Single c_1(K) seul
X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)


def b_mean_link(cfg):
    return np.mean([TOPO[K]["b"] for K in cfg])


def b_pair_sum(cfg, func):
    """Σ_{i<j} func(b_i, b_j) · count_ij"""
    cs = Counter(cfg)
    total = 0
    knots_in_cfg = list(cs.keys())
    for K1, K2 in combinations(knots_in_cfg, 2):
        ct = cs[K1] * cs[K2]
        total += ct * func(TOPO[K1]["b"], TOPO[K2]["b"])
    for K in knots_in_cfg:
        if cs[K] >= 2:
            ct = cs[K] * (cs[K] - 1) / 2
            total += ct * func(TOPO[K]["b"], TOPO[K]["b"])
    return total


# Définir les ansätze
ANSATZE = {
    "G1: β·b_i·b_j": lambda b1, b2: b1 * b2,
    "G2: β·max(b_i,b_j)": lambda b1, b2: max(b1, b2),
    "G3: β·(b_i+b_j)": lambda b1, b2: b1 + b2,
    "G4: β·|b_i-b_j|": lambda b1, b2: abs(b1 - b2),
    "G5: β·√(b_i·b_j)": lambda b1, b2: np.sqrt(b1 * b2),
    "G6: β·(b_i·b_j-1)": lambda b1, b2: b1 * b2 - 1,
    "G7: β·(b_i-2)·(b_j-2)": lambda b1, b2: (b1-2) * (b2-2),
    "G8: β·b_i·b_j/(b_i+b_j)": lambda b1, b2: b1*b2 / (b1+b2),
}


# ───────── Référence : single + b_mean ─────────
X_with_bmean = np.zeros((n_data, n_knots + 1))
X_with_bmean[:, :n_knots] = X_single
for i, d in enumerate(data):
    X_with_bmean[i, n_knots] = b_mean_link(d["cfg"])
rms_ref = cv_rms(X_with_bmean, y)

print("\n" + "═"*100)
print(f"  Référence : single c_1 + b_mean → RMS CV = {rms_ref:.3f}%")
print("═"*100)


# ───────── Tests : single + b_mean + paire G ─────────
print(f"\n  Ajout d'UN terme croisé β · G(b_i, b_j) :")
print(f"  {'Ansatz':<35} {'β extrait':>12} {'RMS CV':>9}")
print("  " + "-" * 60)

best_g = None; best_g_cv = rms_ref
for label, func in ANSATZE.items():
    X = np.zeros((n_data, n_knots + 2))
    X[:, :n_knots] = X_single
    for i, d in enumerate(data):
        X[i, n_knots] = b_mean_link(d["cfg"])
        X[i, n_knots+1] = b_pair_sum(d["cfg"], func)
    cv = cv_rms(X, y)
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[n_knots+1]
    flag = "★★" if cv < 0.55 else ("★" if cv < rms_ref - 0.02 else "")
    print(f"  {label:<35} {beta:>+12.6f} {cv:>8.3f}% {flag}")
    if cv < best_g_cv:
        best_g_cv = cv
        best_g = label


# ───────── Tests : sans b_mean, avec seulement paires G ─────────
print(f"\n  Sans b_mean, seulement β · G(b_i, b_j) :")
print(f"  {'Ansatz':<35} {'β extrait':>12} {'RMS CV':>9}")
print("  " + "-" * 60)
for label, func in ANSATZE.items():
    X = np.zeros((n_data, n_knots + 1))
    X[:, :n_knots] = X_single
    for i, d in enumerate(data):
        X[i, n_knots] = b_pair_sum(d["cfg"], func)
    cv = cv_rms(X, y)
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[n_knots]
    print(f"  {label:<35} {beta:>+12.6f} {cv:>8.3f}%")


# ───────── Combinaisons : b_mean + paire G + autre invariant ─────────
print(f"\n  Combinaison : b_mean + meilleure paire G + invariant additionnel :")

# Best paire trouvée ci-dessus
def link_invariant(cfg, key, agg='mean'):
    vals = [TOPO[K][key] for K in cfg]
    if agg == 'mean': return np.mean(vals)
    if agg == 'max': return max(vals)
    if agg == 'sum': return sum(vals)
    return 0

best_combo_cv = rms_ref
best_combo = None
for inv_key in ["det", "U", "cr"]:
    for agg in ['mean', 'max', 'sum']:
        for g_label, g_func in ANSATZE.items():
            X = np.zeros((n_data, n_knots + 3))
            X[:, :n_knots] = X_single
            for i, d in enumerate(data):
                X[i, n_knots] = b_mean_link(d["cfg"])
                X[i, n_knots+1] = b_pair_sum(d["cfg"], g_func)
                X[i, n_knots+2] = link_invariant(d["cfg"], inv_key, agg)
            cv = cv_rms(X, y)
            if cv < best_combo_cv:
                best_combo_cv = cv
                best_combo = f"b_mean + {g_label} + {inv_key}_{agg}"
                print(f"    NEW BEST: {best_combo} → {cv:.3f}%")

print(f"\n  Meilleure combinaison globale : {best_combo}")
print(f"  RMS CV : {best_combo_cv:.3f}%")


# ───────── Détail du meilleur modèle ─────────
print("\n" + "═"*100)
print(f"  Détail du meilleur modèle trouvé")
print("═"*100)

# Re-fit du meilleur pour récupérer les coefficients
# (On utilise la meilleure combo trouvée)
print(f"\n  Hiérarchie des performances :")
print(f"    Ω_ts strict seul              : 1.218% RMS")
print(f"    + c_1(K) Ridge (single)       : 0.801% CV")
print(f"    + b_mean                       : {rms_ref:.3f}% CV")
print(f"    + b_mean + meilleure paire G   : {best_g_cv:.3f}% CV  ({best_g})")
print(f"    + combinaison à 3 termes       : {best_combo_cv:.3f}% CV")
print(f"\n  Gain global vs c_1 seul : {(0.801 - best_combo_cv)/0.801 * 100:+.1f}%")
print(f"  Gain global vs Ω_ts brut : {(1.218 - best_combo_cv)/1.218 * 100:+.1f}%")
