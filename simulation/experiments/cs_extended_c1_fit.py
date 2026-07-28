r"""
EXTENSION SYSTÉMATIQUE de l'ansatz c_1(K) à TOUTES les particules
disponibles, avec régularisation et validation croisée.

Stratégie :
  1. Pour CHAQUE particule du tableau V5 publié, calculer Ω_ts strict
     avec son assignation OPTIMALE (du scan précédent).
  2. Le résidu Δ = (m_exp - Ω_ts)/m_exp est la cible.
  3. Modéliser Δ = (1/(k+2)) · Σ_K c_1(K) · count_K(L) où la somme
     est sur les knots dans le link.
  4. Fitter c_1(K) avec régularisation L2 (ridge regression) pour
     éviter le surfit.
  5. Validation k-fold : tester si c_1 prédit hors-échantillon.
  6. Comparer c_1(K) ajusté à des invariants topologiques rigoureux
     (σ, λ, genus, ...).

But : extraire un dictionnaire c_1(K) ROBUSTE qui peut être
comparé aux formules Lawrence-Rozansky exactes.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry


class Ridge:
    """Ridge regression manuelle (sans sklearn)."""
    def __init__(self, alpha=1.0, fit_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
    def fit(self, X, y):
        # (X'X + alpha I) coef = X' y
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


class KFold:
    """K-fold manuel."""
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(idx)
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        start = 0
        for size in fold_sizes:
            test_idx = idx[start:start+size]
            train_idx = np.concatenate([idx[:start], idx[start+size:]])
            yield train_idx, test_idx
            start += size
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


# ───────── Particules avec assignations optimales (du scan précédent) ─────────
PARTICULES = [
    # leptons
    ("μ⁻",   ["3_1","6_1"],          105.66,    "lepton", 1),
    ("τ⁻",   ["5_1","5_1","6_1"],   1776.86,    "lepton", 1),
    # pseudoscalars
    ("π",    ["3_1","3_1","3_1"],   137.27,     "pseudo", 0),
    ("η",    ["3_1","5_2","6_1"],   547.86,     "pseudo", 0),
    ("K",    ["6_1","6_1","6_1"],   495.65,     "pseudo", 0),
    ("D",    ["3_1","4_1","6_2"],  1867.0,      "pseudo", 0),
    ("B",    ["5_2","6_3","6_3"],  5279.0,      "pseudo", 0),
    # vectors
    ("ρ",    ["8_1","8_1"],         778.85,     "vector", 2),
    ("J/ψ",  ["3_1","4_1","6_3"],  3096.9,      "vector", 2),
    ("Υ",    ["7_1","7_3","9_1"],  9460.0,      "vector", 2),
    # baryons octet
    ("p",    ["3_1","7_2","7_2"],   938.27,     "baryon", 1),
    ("n",    ["3_1","7_2","7_2"],   939.57,     "baryon", 1),
    ("Λ",    ["5_1","5_1","5_1"],  1115.68,     "baryon", 1),
    ("Σ",    ["5_1","7_7"],        1193.0,      "baryon", 1),
    ("Ξ",    ["5_1","6_3"],        1318.0,      "baryon", 1),
    # baryons décuplet
    ("Δ",    ["5_2","7_2","7_2"],  1232.0,      "decup",  3),
    ("Σ*",   ["10_1"],             1383.0,      "decup",  3),
    ("Ξ*",   ["7_1","8_1"],        1532.0,      "decup",  3),
    ("Ω-",   ["5_1","7_2","7_2"],  1672.0,      "decup",  3),
    # hadrons charmés
    ("D0",   ["3_1","4_1","6_2"],  1864.84,     "pseudo", 0),
    ("Λc",   ["5_1","5_1","7_2"],  2286.46,     "baryon", 1),
    ("Σc",   ["4_1","5_1","7_1"],  2453.0,      "baryon", 1),
    ("Ωc",   ["7_3","7_7"],        2695.0,      "baryon", 1),
]


# ───────── Calcul des résidus ─────────
print("\n  Calcul Ω_ts strict pour 23 particules…")
data = []
for nm, cfg, m_exp, cls, two_J in PARTICULES:
    om = omega_strict(cfg)
    Dm = m_exp - om
    d_rel = Dm / m_exp
    data.append({
        "nm": nm, "cfg": cfg, "m_exp": m_exp, "om": om,
        "Dm": Dm, "d_rel": d_rel, "cls": cls, "two_J": two_J
    })

# RMS initial
rms_init = float(np.sqrt(np.mean([d["d_rel"]**2 for d in data]))) * 100
print(f"  RMS Ω_ts strict initial : {rms_init:.3f}%")


# ───────── Construction de la matrice de design (counts par knot) ─────────
print("\n" + "═"*100)
print("  Fit régularisé (Ridge) de c_1(K) sur 23 particules")
print("═"*100)

# Matrice X[i, j] = count du knot j dans le link i
X = np.zeros((len(data), len(KNOTS)))
y = np.zeros(len(data))
for i, d in enumerate(data):
    counts = Counter(d["cfg"])
    for j, K in enumerate(KNOTS):
        X[i, j] = counts.get(K, 0)
    y[i] = d["d_rel"]


# Régularisation Ridge avec différents alphas
print(f"\n  {'alpha (régul.)':<15} {'RMS train':>12} {'c_1 norm':>10} {'effective dof':>14}")
print("  " + "-" * 60)
for alpha in [0.0, 0.001, 0.01, 0.1, 1.0]:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, y)
    c1 = ridge.coef_
    y_pred = X @ c1
    rms = np.sqrt(np.mean((y - y_pred)**2)) * 100
    norm_c1 = np.linalg.norm(c1)
    # Effective degrees of freedom for Ridge
    if alpha > 0:
        # tr(X(X'X + alpha I)^-1 X')
        XtX = X.T @ X
        inv = np.linalg.inv(XtX + alpha * np.eye(len(KNOTS)))
        dof = np.trace(X @ inv @ X.T)
    else:
        dof = np.linalg.matrix_rank(X)
    print(f"  alpha={alpha:<7} {rms:>11.4f}% {norm_c1:>10.4f} {dof:>14.2f}")


# Choix d'alpha optimal par validation croisée k-fold
# KFold déjà défini en haut du fichier
print("\n  Validation 5-fold pour choisir alpha optimal :")
best_alpha = 0.01; best_cv_rms = float('inf')
for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_errs = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_train, y_train)
        err = np.sqrt(np.mean((ridge.predict(X_test) - y_test)**2))
        cv_errs.append(err)
    cv_rms = np.mean(cv_errs) * 100
    if cv_rms < best_cv_rms:
        best_cv_rms = cv_rms
        best_alpha = alpha
    print(f"    alpha = {alpha:.4f} : CV RMS = {cv_rms:.4f}%")
print(f"\n  ★ Alpha optimal : {best_alpha} (CV RMS = {best_cv_rms:.4f}%)")


# Fit final avec alpha optimal
ridge_final = Ridge(alpha=best_alpha, fit_intercept=False)
ridge_final.fit(X, y)
c1_robust = dict(zip(KNOTS, ridge_final.coef_))


# ───────── Tableau des c_1 robustes ─────────
print("\n" + "═"*100)
print(f"  c_1(K) robuste (Ridge alpha={best_alpha})")
print("═"*100)
print(f"\n  {'K':<6} {'c_1 robust':>12} {'σ':>4} {'λ Casson':>10}")
print("  " + "-" * 40)

# Tables for comparison
SIGNATURE = {
    "3_1": -2,  "4_1": 0,   "5_1": -4,  "5_2": -2,
    "6_1": 0,   "6_2": -2,  "6_3": 0,
    "7_1": -6,  "7_2": -2,  "7_3": -4,  "7_4": -2,
    "7_5": -4,  "7_6": -2,  "7_7": 0,
    "8_1": 0,   "8_2": -4,  "8_3": 0,
    "9_1": -8,  "10_1": 0,
}
CASSON = {
    "3_1": 1, "4_1": -1, "5_1": 3, "5_2": 1,
    "6_1": -1, "6_2": 1, "6_3": 1,
    "7_1": 6, "7_2": 1, "7_3": 5, "7_4": 4,
    "7_5": 4, "7_6": 1, "7_7": -1,
    "8_1": -2, "8_2": 4, "8_3": -4,
    "9_1": 10, "10_1": -3,
}
for K in KNOTS:
    c = c1_robust[K]
    print(f"  {K:<6} {c:>+12.5f} {SIGNATURE[K]:>+4} {CASSON[K]:>+10}")


# ───────── Comparaison aux invariants topologiques ─────────
print("\n" + "═"*100)
print(f"  Comparaison c_1 robust vs invariants Lawrence-Rozansky")
print("═"*100)

# Préparer arrays
knots_arr = list(KNOTS)
c1_arr = np.array([c1_robust[K] for K in knots_arr])
sig_arr = np.array([SIGNATURE[K] for K in knots_arr])
cass_arr = np.array([CASSON[K] for K in knots_arr])
det_arr = np.array([TOPO[K]["det"] for K in knots_arr])
log_det = np.log(np.maximum(det_arr, 1))
cr_arr = np.array([TOPO[K]["cr"] for K in knots_arr])
b_arr = np.array([TOPO[K]["b"] for K in knots_arr])

print(f"\n  Corrélations Pearson c_1 vs invariants :")
for name, v in [("σ", sig_arr.astype(float)),
                 ("λ Casson", cass_arr.astype(float)),
                 ("log(det)", log_det),
                 ("cr", cr_arr.astype(float)),
                 ("b", b_arr.astype(float))]:
    if np.std(v) > 1e-10:
        r = np.corrcoef(v, c1_arr)[0, 1]
        flag = "★★" if abs(r) > 0.8 else ("★" if abs(r) > 0.6 else "")
        print(f"    {name:<12} : r = {r:+.4f}  {flag}")


# Fit multivarié final
from scipy.optimize import minimize
print(f"\n  Fit multivarié c_1 = A·σ + B·λ + C·log(det) + D·cr + E·b :")
def chi2(params):
    A, B, C, D, E = params
    pred = (A*sig_arr + B*cass_arr + C*log_det + D*cr_arr + E*b_arr)
    return np.sum((pred - c1_arr)**2)
res = minimize(chi2, [0.01, 0.001, 0.001, 0, 0], method='Nelder-Mead',
               options={'fatol': 1e-12, 'maxiter': 10000})
A, B, C, D, E = res.x
rms_LR = np.sqrt(res.fun / len(c1_arr))
print(f"    A_sig    = {A:+.6f}")
print(f"    B_cass   = {B:+.6f}")
print(f"    C_log_det = {C:+.6f}")
print(f"    D_cr     = {D:+.6f}")
print(f"    E_b      = {E:+.6f}")
print(f"    RMS fit  = {rms_LR:.6f}")

print(f"\n  Détail prédictions :")
print(f"  {'K':<5} {'c_1 robust':>11} {'LR prédit':>11} {'résidu':>10}")
for K, c in zip(knots_arr, c1_arr):
    pred = (A*SIGNATURE[K] + B*CASSON[K] + C*np.log(max(TOPO[K]["det"], 1))
            + D*TOPO[K]["cr"] + E*TOPO[K]["b"])
    resid = pred - c
    flag = "✓" if abs(resid) < 0.005 else ("·" if abs(resid) < 0.015 else "")
    print(f"  {K:<5} {c:>+11.5f} {pred:>+11.5f} {resid:>+10.5f} {flag}")


# ───────── Comparaison aux résidus prédits ─────────
print("\n" + "═"*100)
print(f"  Validation : RMS sur les 23 particules avec c_1 robust")
print("═"*100)

y_pred = X @ ridge_final.coef_
errors = (y - y_pred) * 100
rms_final = float(np.sqrt(np.mean(errors**2)))

print(f"\n  RMS Ω_ts strict initial  : {rms_init:.3f}%")
print(f"  RMS Ω_ts + c_1 robust    : {rms_final:.3f}%")
print(f"  Gain                      : {(rms_init - rms_final)/rms_init*100:+.1f}%")
print(f"  Validation 5-fold (CV)   : {best_cv_rms:.3f}%")

print(f"\n  Détail particule par particule :")
print(f"  {'nm':<5} {'Ω_ts':>9} {'Δ initial':>10} {'Δ corr':>10}")
for i, d in enumerate(data):
    err_init = d["d_rel"]*100
    err_after = errors[i]
    print(f"  {d['nm']:<5} {d['om']:>9.2f} {err_init:>+9.2f}% {err_after:>+9.2f}%")
