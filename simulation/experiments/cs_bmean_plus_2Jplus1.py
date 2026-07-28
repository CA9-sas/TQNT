r"""
Direction Option C : combiner b_mean (topologique) + n_phys = 2J+1 (représentation
physique de la particule).

Hypothèse : la correction CS au-delà de Ω_ts strict combine
  - une partie TOPOLOGIQUE (b_mean → 0.65% RMS CV)
  - une partie PHYSIQUE (rep dim n = 2J+1)

Ansätze testés :
  H1 : β · (2J+1)                  — additif simple
  H2 : β · log(2J+1)               — additif log
  H3 : β · ((2J+1)-1)              — (2J+1) centré sur pseudo
  H4 : β · b_mean · (2J+1)         — multiplicatif (interaction)
  H5 : β1·b_mean + β2·(2J+1)       — combinaison linéaire
  H6 : β1·b_mean + β2·(2J+1) + β3·b_mean·(2J+1)  — quadratique
  H7 : β · 1/(2J+1)                — inverse (pseudo dominant)
  H8 : β · sqrt(2J+1)              — racine
"""
from __future__ import annotations
import numpy as np
from collections import Counter
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


# Particules avec 2J pour chaque
PARTICULES = [
    # (nm, config, m_exp, 2J)
    ("μ⁻",   ["3_1","6_1"],          105.66,  1),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86, 1),
    ("π",    ["3_1","3_1","3_1"],   137.27,  0),
    ("η",    ["3_1","5_2","6_1"],   547.86,  0),
    ("K",    ["6_1","6_1","6_1"],   495.65,  0),
    ("D",    ["3_1","4_1","6_2"],  1867.0,   0),
    ("B",    ["5_2","6_3","6_3"],  5279.0,   0),
    ("ρ",    ["8_1","8_1"],         778.85,  2),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9,   2),
    ("Υ",    ["7_1","7_3","9_1"],  9460.0,   2),
    ("p",    ["3_1","7_2","7_2"],   938.27,  1),
    ("n",    ["3_1","7_2","7_2"],   939.57,  1),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68,  1),
    ("Σ",    ["5_1","7_7"],        1193.0,   1),
    ("Ξ",    ["5_1","6_3"],        1318.0,   1),
    ("Δ",    ["5_2","7_2","7_2"],  1232.0,   3),
    ("Σ*",   ["10_1"],             1383.0,   3),
    ("Ξ*",   ["7_1","8_1"],        1532.0,   3),
    ("Ω-",   ["5_1","7_2","7_2"],  1672.0,   3),
    ("D0",   ["3_1","4_1","6_2"],  1864.84,  0),
    ("Λc",   ["5_1","5_1","7_2"],  2286.46,  1),
    ("Σc",   ["4_1","5_1","7_1"],  2453.0,   1),
    ("Ωc",   ["7_3","7_7"],        2695.0,   1),
]


data = []
for nm, cfg, m_exp, two_J in PARTICULES:
    om = omega_strict(cfg)
    n_phys = two_J + 1  # 2J + 1
    data.append({"nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
                  "d_rel": (m_exp - om) / m_exp,
                  "two_J": two_J, "n_phys": n_phys})

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


# Matrice de base : c_1 single
X_single = np.zeros((n_data, n_knots))
for i, d in enumerate(data):
    cs = Counter(d["cfg"])
    for j, K in enumerate(KNOTS): X_single[i, j] = cs.get(K, 0)


def b_mean(cfg):
    return np.mean([TOPO[K]["b"] for K in cfg])


# Référence : single + b_mean
X_ref = np.zeros((n_data, n_knots + 1))
X_ref[:, :n_knots] = X_single
for i, d in enumerate(data):
    X_ref[i, n_knots] = b_mean(d["cfg"])
rms_ref = cv_rms(X_ref, y)


# ───────── Test des ansätze ─────────
print("\n" + "═"*100)
print(f"  Référence : c_1 single + b_mean = {rms_ref:.3f}% RMS CV")
print("═"*100)

print(f"\n  Ajout d'un seul terme physique (2J+1) :")
print(f"  {'Ansatz':<40} {'β extrait':>12} {'RMS CV':>9}")
print("  " + "-" * 65)

ansatze = {
    "H1: β · (2J+1)":         lambda d: d["n_phys"],
    "H2: β · log(2J+1)":       lambda d: np.log(d["n_phys"]),
    "H3: β · ((2J+1)-1)":     lambda d: d["n_phys"] - 1,
    "H4: β · b_mean·(2J+1)":   lambda d: b_mean(d["cfg"]) * d["n_phys"],
    "H7: β · 1/(2J+1)":        lambda d: 1.0 / d["n_phys"],
    "H8: β · sqrt(2J+1)":     lambda d: np.sqrt(d["n_phys"]),
    "H9: β · (2J+1)²":         lambda d: d["n_phys"]**2,
    "H10: β · (-1)^(2J)":      lambda d: (-1)**d["two_J"],
    "H11: β · sigma(2J=0)":    lambda d: 1.0 if d["two_J"] == 0 else -1.0,
    "H12: β · J(J+1)":         lambda d: (d["two_J"]/2) * (d["two_J"]/2 + 1),
}

best_h = None; best_h_cv = rms_ref
for label, fn in ansatze.items():
    X = np.zeros((n_data, n_knots + 2))
    X[:, :n_knots+1] = X_ref
    for i, d in enumerate(data):
        X[i, n_knots+1] = fn(d)
    cv = cv_rms(X, y)
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[n_knots+1]
    flag = "★★" if cv < 0.5 else ("★" if cv < 0.62 else " ")
    print(f"  {label:<40} {beta:>+12.5f} {cv:>8.3f}% {flag}")
    if cv < best_h_cv:
        best_h_cv = cv; best_h = label

print(f"\n  ★ Meilleur ansatz : {best_h} → {best_h_cv:.3f}%")


# ───────── Test SANS b_mean, juste (2J+1) ─────────
print("\n" + "═"*100)
print(f"  Sans b_mean, juste les ansätze (2J+1)")
print("═"*100)
print(f"\n  {'Ansatz':<40} {'β extrait':>12} {'RMS CV':>9}")
print("  " + "-" * 65)
for label, fn in ansatze.items():
    X = np.zeros((n_data, n_knots + 1))
    X[:, :n_knots] = X_single
    for i, d in enumerate(data):
        X[i, n_knots] = fn(d)
    cv = cv_rms(X, y)
    r_full = Ridge(alpha=0.1).fit(X, y)
    beta = r_full.coef_[n_knots]
    print(f"  {label:<40} {beta:>+12.5f} {cv:>8.3f}%")


# ───────── Combiner b_mean + n_phys (deux paramètres) ─────────
print("\n" + "═"*100)
print(f"  Combinaisons : c_1 + b_mean + meilleur ansatz physique")
print("═"*100)

# Tester combinaison b_mean + chaque ansatz simultanément (déjà fait dans le H_ref)
# Ici tester sans b_mean + 2 ansätze
print(f"\n  Combinaisons sans b_mean :")
print(f"  {'A1':<25} {'A2':<25} {'RMS CV':>9}")
print("  " + "-" * 70)
labels = list(ansatze.keys())
best_combo = None; best_combo_cv = float('inf')
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        X = np.zeros((n_data, n_knots + 2))
        X[:, :n_knots] = X_single
        for ii, d in enumerate(data):
            X[ii, n_knots] = ansatze[l1](d)
            X[ii, n_knots+1] = ansatze[l2](d)
        cv = cv_rms(X, y)
        if cv < best_combo_cv:
            best_combo_cv = cv
            best_combo = (l1, l2)
            if cv < 0.6:
                print(f"  {l1:<25} {l2:<25} {cv:>8.3f}% ★")

print(f"\n  Meilleure combo sans b_mean : {best_combo[0]} + {best_combo[1]} → {best_combo_cv:.3f}%")


# ───────── Synthèse finale ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)

print(f"""
  Hiérarchie des RMS CV obtenues :

    Ω_ts strict seul                              : 1.218%
    + c_1(K) Ridge (single)                       : 0.801%
    + b_mean (topologique)                        : 0.653%
    + b_mean + meilleur ansatz physique           : {best_h_cv:.3f}%
    + meilleure combo sans b_mean (2 physiques)   : {best_combo_cv:.3f}%

  Gain global vs Ω_ts strict : {(1.218 - min(best_h_cv, best_combo_cv))/1.218 * 100:+.1f}%
""")
