r"""
DIRECTION 3 : forme analytique de c_2(K_i, K_j) via invariants combinés.

Au lieu d'ajuster c_2 librement pour chaque paire (29 paramètres), on
postule une forme analytique :
  c_2(K_i, K_j) = beta · f(K_i, K_j)
avec f construit à partir d'invariants des deux nœuds.

Plusieurs ansätze testés :
  F1 : f = log(det_i · det_j) — produit géométrique des déterminants
  F2 : f = σ(K_i) · σ(K_j)   — produit des signatures
  F3 : f = √(U_κU_τ_i · U_κU_τ_j) — moyenne géométrique d'énergies
  F4 : f = (cr_i + cr_j) / log(det_i + det_j)
  F5 : f = (det_i × det_j)^(1/2)
  F6 : f = b_i · b_j · sign(det_i - det_j)  — asymétrie d'écheveau
  F7 : f = (det_i + det_j) / max(cr_i, cr_j)
  F8 : f = log(b_i · b_j) · log(det_i / det_j)
  F9 : f = symmetrized écart de complexité

Si une forme avec 1-3 paramètres globaux atteint le RMS CV du modèle
à 48 params, on a une structure analytique validée.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.coef_ = None
    def fit(self, X, y):
        n_feat = X.shape[1]
        A = X.T @ X + max(self.alpha, 1e-10) * np.eye(n_feat)
        b = X.T @ y
        try: self.coef_ = np.linalg.solve(A, b)
        except: self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]
        return self
    def predict(self, X): return X @ self.coef_


def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed); idx = np.arange(n); rng.shuffle(idx)
    sizes = [n // k] * k
    for i in range(n % k): sizes[i] += 1
    start = 0
    for size in sizes:
        yield np.concatenate([idx[:start], idx[start+size:]]), idx[start:start+size]
        start += size


M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
SIGNATURE = {"3_1":-2,"4_1":0,"5_1":-4,"5_2":-2,"6_1":0,"6_2":-2,"6_3":0,
             "7_1":-6,"7_2":-2,"7_3":-4,"7_4":-2,"7_5":-4,"7_6":-2,"7_7":0,
             "8_1":0,"8_2":-4,"8_3":0,"9_1":-8,"10_1":0}
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
               "sig": SIGNATURE[K], "U": U_KT[K][0]*U_KT[K][1]}
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
    d_rel = (m_exp - om) / m_exp
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                  "d_rel": d_rel})

rms_init = float(np.sqrt(np.mean([d["d_rel"]**2 for d in data]))) * 100
print(f"\n  RMS Ω_ts strict initial : {rms_init:.3f}%")


# ───────── Définir les formes analytiques candidates ─────────
def F1(K1, K2):  # log(det_i · det_j)
    return np.log(max(TOPO[K1]["det"] * TOPO[K2]["det"], 1))

def F2(K1, K2):  # σ_i · σ_j
    return TOPO[K1]["sig"] * TOPO[K2]["sig"]

def F3(K1, K2):  # √(U_i · U_j)
    return np.sqrt(TOPO[K1]["U"] * TOPO[K2]["U"])

def F4(K1, K2):  # (cr_i + cr_j) / log(det_i + det_j)
    cr_sum = TOPO[K1]["cr"] + TOPO[K2]["cr"]
    return cr_sum / np.log(max(TOPO[K1]["det"] + TOPO[K2]["det"], 2))

def F5(K1, K2):  # √(det_i · det_j)
    return np.sqrt(TOPO[K1]["det"] * TOPO[K2]["det"])

def F6(K1, K2):  # b_i · b_j · sign(det_i - det_j)
    return TOPO[K1]["b"] * TOPO[K2]["b"] * np.sign(TOPO[K1]["det"] - TOPO[K2]["det"])

def F7(K1, K2):  # 1 / (det_i · det_j)
    return 1.0 / max(TOPO[K1]["det"] * TOPO[K2]["det"], 1)

def F8(K1, K2):  # log(U_i + U_j)
    return np.log(max(TOPO[K1]["U"] + TOPO[K2]["U"], 1))

def F9(K1, K2):  # sign(σ_i + σ_j) · |σ_i - σ_j|
    return np.sign(TOPO[K1]["sig"] + TOPO[K2]["sig"] + 0.001) \
           * abs(TOPO[K1]["sig"] - TOPO[K2]["sig"])

def F10(K1, K2):  # 1/(U_i · U_j)^(1/2)  — interaction "inverse" Biot-Savart
    return 1.0 / np.sqrt(max(TOPO[K1]["U"] * TOPO[K2]["U"], 1))

FORMS = {
    "F1: log(det·det)": F1,
    "F2: σ·σ": F2,
    "F3: √(U·U)": F3,
    "F4: (cr+cr)/log(det+det)": F4,
    "F5: √(det·det)": F5,
    "F6: b·b·sign(det-det)": F6,
    "F7: 1/(det·det)": F7,
    "F8: log(U+U)": F8,
    "F9: sign(σ+σ)·|σ-σ|": F9,
    "F10: 1/√(U·U)": F10,
}


# ───────── Construire les features ─────────
n_data = len(data); n_knots = len(KNOTS)
all_pairs = set()
for d in data:
    for K1, K2 in combinations(set(d["cfg"]), 2):
        all_pairs.add(tuple(sorted([K1, K2])))
    counter = Counter(d["cfg"])
    for K, ct in counter.items():
        if ct >= 2: all_pairs.add((K, K))
PAIRS = sorted(all_pairs)
n_pairs = len(PAIRS)


def pair_count(cfg, K1, K2):
    counter = Counter(cfg)
    if K1 == K2:
        ct = counter.get(K1, 0)
        return ct * (ct - 1) / 2
    return counter.get(K1, 0) * counter.get(K2, 0)


def count_pairs_vec(cfg, pairs_list):
    return np.array([pair_count(cfg, K1, K2) for K1, K2 in pairs_list])


# Construire la matrice de design : single + analytic c_2
def design_analytic(F_fn):
    """Matrice : 19 single + 1 colonne globale = sum_pairs F(K_i, K_j) · count."""
    X = np.zeros((n_data, n_knots + 1))
    for i, d in enumerate(data):
        cs = Counter(d["cfg"])
        for j, K in enumerate(KNOTS):
            X[i, j] = cs.get(K, 0)
        # Somme analytique
        total = 0.0
        for K1, K2 in PAIRS:
            ct = pair_count(d["cfg"], K1, K2)
            if ct > 0:
                total += ct * F_fn(K1, K2)
        X[i, n_knots] = total
    return X


# y target
y = np.array([d["d_rel"] for d in data])


# ───────── Comparaison des formes analytiques ─────────
print("\n" + "═"*100)
print("  Évaluation des formes c_2(K_i, K_j) = β · f(K_i, K_j)")
print("═"*100)
print(f"\n  Single seul (référence) :")
X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)
cv_single = []
for tr, te in kfold_indices(n_data, k=5):
    r = Ridge(alpha=0.1).fit(X_single[tr], y[tr])
    cv_single.append(np.sqrt(np.mean((X_single[te] @ r.coef_ - y[te])**2)))
rms_single_cv = np.mean(cv_single) * 100
print(f"    RMS CV (single) : {rms_single_cv:.3f}%")

print(f"\n  {'Forme f':<28} {'β extrait':>10} {'RMS train':>10} {'RMS CV':>9}")
print("  " + "-" * 65)

results_forms = {}
for label, F_fn in FORMS.items():
    X = design_analytic(F_fn)
    # Cross-validation
    cv = []
    train_rms_list = []
    for tr, te in kfold_indices(n_data, k=5):
        r = Ridge(alpha=0.1).fit(X[tr], y[tr])
        cv.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
        train_rms_list.append(np.sqrt(np.mean((X[tr] @ r.coef_ - y[tr])**2)))
    rms_cv = np.mean(cv) * 100
    rms_train = np.mean(train_rms_list) * 100
    # Fit complet pour extraire β
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[n_knots]
    flag = "★★" if rms_cv < 0.7 else ("★" if rms_cv < 0.78 else "")
    print(f"  {label:<28} {beta:>+10.6f} {rms_train:>9.3f}% {rms_cv:>8.3f}% {flag}")
    results_forms[label] = (beta, rms_train, rms_cv)


# ───────── Combinaisons : 2 formes simultanées ─────────
print("\n" + "═"*100)
print("  Test : combinaison de 2 formes (c_2 = β1·f1 + β2·f2)")
print("═"*100)

def design_two_forms(F1_fn, F2_fn):
    X = np.zeros((n_data, n_knots + 2))
    for i, d in enumerate(data):
        cs = Counter(d["cfg"])
        for j, K in enumerate(KNOTS): X[i, j] = cs.get(K, 0)
        t1 = sum(pair_count(d["cfg"], K1, K2) * F1_fn(K1, K2) for K1, K2 in PAIRS)
        t2 = sum(pair_count(d["cfg"], K1, K2) * F2_fn(K1, K2) for K1, K2 in PAIRS)
        X[i, n_knots] = t1; X[i, n_knots+1] = t2
    return X

print(f"\n  Tests sélectionnés (formes les plus prometteuses) :")
form_keys = list(FORMS.keys())
best_combo = None; best_combo_cv = float('inf')
for i, k1 in enumerate(form_keys):
    for k2 in form_keys[i+1:]:
        X = design_two_forms(FORMS[k1], FORMS[k2])
        cv = []
        for tr, te in kfold_indices(n_data, k=5):
            r = Ridge(alpha=0.1).fit(X[tr], y[tr])
            cv.append(np.sqrt(np.mean((X[te] @ r.coef_ - y[te])**2)))
        rms_cv = np.mean(cv) * 100
        if rms_cv < best_combo_cv:
            best_combo_cv = rms_cv
            best_combo = (k1, k2)

print(f"\n  Meilleure paire de formes : {best_combo[0]}  +  {best_combo[1]}")
print(f"  RMS CV : {best_combo_cv:.3f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  RMS Ω_ts strict initial      : {rms_init:.3f}%
  RMS CV avec c_1(K) seul      : {rms_single_cv:.3f}%
  RMS CV (meilleure forme F)   : {min(r[2] for r in results_forms.values()):.3f}%
  RMS CV (meilleure paire FF)  : {best_combo_cv:.3f}%
  RMS CV (modèle plein 48 par) : 0.813%  (référence précédente)

  Interprétation :
    Si une forme analytique f atteint ~ 0.7-0.8% en CV, on a identifié
    la STRUCTURE de l'interaction inter-composante.

    Le β extrait est le SEUL nouveau paramètre à ajouter à V5 — vs 29
    paramètres dans le modèle plein. Énorme réduction.

  Top 3 formes par RMS CV :
""")
sorted_results = sorted(results_forms.items(), key=lambda kv: kv[1][2])
for rank, (label, (beta, rt, rc)) in enumerate(sorted_results[:3], 1):
    print(f"    {rank}. {label:<25} β = {beta:+.6f}, RMS CV = {rc:.3f}%")
