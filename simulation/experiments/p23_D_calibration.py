r"""
CALIBRATION de la distance effective D entre composantes par particule.

Modèle physique : F^rad(L; D) ∝ 1/D^3 (loi magnétostatique dipolaire).
On a aussi : γ(L) = k · F^rad(L; D_eff) avec k constante universelle.

Méthode :
  1. Pour chaque particule, calculer F^rad(L; D) sur grille de D
  2. Vérifier la loi 1/D^3 (ajuster a, b tels que F = a/D^b)
  3. Calibrer k sur μ⁻ : k = γ_emp(μ⁻) / F^rad(μ⁻; D_ref)
  4. Pour les autres : trouver D_eff(L) tel que k·F^rad(L; D_eff) = γ_emp(L)
  5. Chercher pattern : D_eff(L) corrèle-t-il avec un invariant topologique ?

Particules testées (avec γ extrait des sessions précédentes) :
  μ⁻ (3_1+6_1)              : γ_emp = -0.148 → référence
  η  (3_1+3_1+6_1)          : γ_emp = -0.184
  τ⁻ (4_1+5_1+6_1)          : γ_emp = -0.033
  φ  (3_1+5_1+5_2)          : γ_emp = +0.293
  D⁰ (3_1+5_1+7_2)          : γ_emp = -0.172
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _CURVE_REGISTRY


def get_curve_normal(name, n_points=200):
    pts = _CURVE_REGISTRY[name](n_points)
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1, keepdims=True)
    T = r1 / (speed + 1e-12)
    dT_ds = np.gradient(T, dt, axis=0)
    N = dT_ds / (np.linalg.norm(dT_ds, axis=1, keepdims=True) + 1e-12)
    return pts, T, N, speed.ravel() * dt


def field_at(r_target, curve_src, T_src, ds_src):
    disp = r_target[None, :] - curve_src
    d = np.linalg.norm(disp, axis=1) + 1e-6
    B = np.cross(T_src, disp) / d[:, None]**3
    return np.sum(B * ds_src[:, None], axis=0)


def F_radial(curve_i, T_i, N_i, ds_i, curves_o, T_o, ds_o):
    F_list = []
    for r_i, n_i in zip(curve_i, N_i):
        B = np.zeros(3)
        for c_j, T_j, ds_j in zip(curves_o, T_o, ds_o):
            B += field_at(r_i, c_j, T_j, ds_j)
        F_list.append(np.abs(np.dot(B, n_i)))  # |radial|
    return float(np.mean(F_list))


def place_link(component_names, D, n_points=200):
    """Place les composantes à distance D dans des positions stables."""
    curves, Ts, Ns, dss = [], [], [], []
    offsets = [np.array([0.0, 0.0, 0.0]),
               np.array([D, 0.0, 0.0]),
               np.array([D*np.cos(2*np.pi/3), D*np.sin(2*np.pi/3), 0.0])]
    for i, name in enumerate(component_names):
        pts, T, N, ds = get_curve_normal(name, n_points)
        if i < len(offsets):
            pts = pts + offsets[i]
        curves.append(pts); Ts.append(T); Ns.append(N); dss.append(ds)
    return curves, Ts, Ns, dss


def compute_F_rad_global(component_names, D, n_points=200):
    """F^rad global moyenné sur composantes."""
    curves, Ts, Ns, dss = place_link(component_names, D, n_points)
    F_total = 0
    for i in range(len(component_names)):
        others = [j for j in range(len(component_names)) if j != i]
        F_i = F_radial(curves[i], Ts[i], Ns[i], dss[i],
                       [curves[j] for j in others],
                       [Ts[j] for j in others],
                       [dss[j] for j in others])
        F_total += F_i
    return F_total / len(component_names)


# ───────── Particules avec γ extrait empiriquement ─────────
particules = [
    # (nm, assignment, γ_emp)
    ("μ⁻",   ["3_1","6_1"],          0.148),
    ("η",    ["3_1","3_1","6_1"],    0.184),
    ("τ⁻",   ["4_1","5_1","6_1"],    0.033),
    ("φ",    ["3_1","5_1","5_2"],    0.293),
    ("D⁰",   ["3_1","5_1","7_2"],    0.172),
    ("Λ_c",  ["3_1","3_1","9_1"],    0.016),
    ("p",    ["3_1","7_2","7_2"],    0.005),
]


# ───────── Étape 1 : Scan F^rad(L; D) pour chaque L ─────────
print("═"*100)
print("  Étape 1 : Scan F^rad(L; D) sur grille de D")
print("═"*100)

D_grid = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0])

print(f"  {'particule':<8} {'config':<20} {'γ_emp':>8}  ", end="")
for D in D_grid:
    print(f"D={D:<4.1f}  ", end="")
print()
print("  " + "-" * 100)

F_data = {}
for nm, cfg, gamma in particules:
    F_list = []
    print(f"  {nm:<8} {'+'.join(cfg):<20} {gamma:>8.4f}  ", end="", flush=True)
    for D in D_grid:
        F = compute_F_rad_global(cfg, D, n_points=150)
        F_list.append(F)
        print(f"{F:7.4f}  ", end="", flush=True)
    print()
    F_data[nm] = (cfg, gamma, np.array(F_list))


# ───────── Étape 2 : Vérification de la loi 1/D^k par fit log ─────────
print("\n" + "═"*100)
print("  Étape 2 : Loi de décroissance F^rad ∝ 1/D^k")
print("═"*100)
print(f"  {'particule':<8} {'k (exposant)':>14} {'a (préfacteur)':>16}")
print("  " + "-" * 50)
log_D = np.log(D_grid)
for nm, (cfg, gamma, F_arr) in F_data.items():
    log_F = np.log(F_arr + 1e-10)
    # Fit log F = -k log D + log a
    slope, intercept = np.polyfit(log_D, log_F, 1)
    k_exp = -slope
    a_pref = np.exp(intercept)
    print(f"  {nm:<8} {k_exp:>14.3f} {a_pref:>16.4f}")


# ───────── Étape 3 : Calibration k sur μ⁻ et résolution D_eff par particule ─────────
print("\n" + "═"*100)
print("  Étape 3 : Calibration sur μ⁻ et résolution D_eff(L) par particule")
print("═"*100)

# Choisir une distance de référence pour μ⁻
D_ref_mu = 3.0
F_mu_at_ref = F_data["μ⁻"][2][np.argmin(np.abs(D_grid - D_ref_mu))]
k_cal = F_data["μ⁻"][1] / F_mu_at_ref
print(f"\n  Calibration : à D_ref(μ⁻) = {D_ref_mu}, F^rad(μ⁻) = {F_mu_at_ref:.4f}")
print(f"  Constante de conversion k = γ_emp / F^rad = {k_cal:.5f}")

print(f"\n  Inversion : pour chaque particule, D_eff tel que k·F^rad(L; D_eff) = γ_emp")
print(f"  {'particule':<8} {'γ_emp':>8} {'D_eff':>8} {'F_rad calc':>12}")
print("  " + "-" * 50)

D_eff_data = []
for nm, (cfg, gamma, F_arr) in F_data.items():
    # γ = k · a / D_eff^k_exp → D_eff = (k · a / γ)^(1/k_exp)
    log_F = np.log(F_arr + 1e-10)
    slope, intercept = np.polyfit(log_D, log_F, 1)
    k_exp = -slope
    a_pref = np.exp(intercept)
    F_target = abs(gamma) / k_cal
    if F_target > 0 and a_pref > 0:
        D_eff = (a_pref / F_target)**(1/k_exp)
        F_at_D_eff = a_pref / D_eff**k_exp
    else:
        D_eff = float("nan")
        F_at_D_eff = float("nan")
    print(f"  {nm:<8} {gamma:>+8.4f} {D_eff:>8.3f} {F_at_D_eff:>12.4f}")
    D_eff_data.append({"nm": nm, "cfg": cfg, "gamma": gamma,
                        "D_eff": D_eff, "k_exp": k_exp})


# ───────── Étape 4 : Recherche du pattern D_eff vs invariants ─────────
print("\n" + "═"*100)
print("  Étape 4 : Le D_eff par particule a-t-il un sens physique ?")
print("═"*100)

# Invariants candidats
from theory.abraham import frenet_geometry

U_KT = {}
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","9_1"]
for K in KNOTS:
    g = frenet_geometry(K, 20000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = float(np.sum(g["kappa"]**2*ds)) * float(np.sum(g["tau"]**2*ds))

print(f"\n  {'particule':<8} {'D_eff':>7} {'Σ U_κU_τ':>10} {'⟨cr⟩':>6} {'n':>3} "
      f"{'D_eff/n':>8} {'(ΣU)^(1/3)':>12}")
print("  " + "-" * 70)
for d in D_eff_data:
    sum_U = sum(U_KT.get(K, 100) for K in d["cfg"])
    mean_cr = np.mean([int(K.split("_")[0]) for K in d["cfg"]])
    n = len(d["cfg"])
    ratio_n = d["D_eff"] / n
    cube_root = sum_U**(1/3)
    print(f"  {d['nm']:<8} {d['D_eff']:>7.3f} {sum_U:>10.2f} {mean_cr:>6.2f} {n:>3} "
          f"{ratio_n:>8.3f} {cube_root:>12.3f}")

# Corrélations
D_arr = np.array([d["D_eff"] for d in D_eff_data
                   if not np.isnan(d["D_eff"]) and d["D_eff"] > 0])
sum_U_arr = np.array([sum(U_KT.get(K, 100) for K in d["cfg"])
                       for d in D_eff_data
                       if not np.isnan(d["D_eff"]) and d["D_eff"] > 0])
n_arr = np.array([len(d["cfg"]) for d in D_eff_data
                   if not np.isnan(d["D_eff"]) and d["D_eff"] > 0])
mean_cr_arr = np.array([np.mean([int(K.split("_")[0]) for K in d["cfg"]])
                         for d in D_eff_data
                         if not np.isnan(d["D_eff"]) and d["D_eff"] > 0])

print(f"\n  Corrélations Pearson D_eff vs invariants :")
if len(D_arr) > 2:
    for label, v in [("Σ U_κU_τ", sum_U_arr), ("(ΣU)^(1/3)", sum_U_arr**(1/3)),
                      ("log(ΣU)", np.log(sum_U_arr)),
                      ("n_components", n_arr.astype(float)),
                      ("⟨cr⟩", mean_cr_arr)]:
        if np.std(v) > 1e-10:
            r = np.corrcoef(v, D_arr)[0, 1]
            flag = "★★" if abs(r) > 0.85 else ("★" if abs(r) > 0.65 else "")
            print(f"    {label:<14} r = {r:+.3f}  {flag}")

print(f"""
  Interprétation :
    Si D_eff ∝ (Σ U_κU_τ)^(1/3), cela signifie que la « taille effective »
    de l'entrelacs scale comme la racine cubique de l'énergie totale
    de courbure-torsion — interprétation géométrique très naturelle
    (volume de localisation ∝ énergie).

    Si D_eff ∝ ⟨cr⟩, c'est lié à la complexité topologique moyenne.

    Si pas de corrélation > 0.7, la calibration D_eff reste ad hoc.
""")
