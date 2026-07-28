r"""
VALIDATION COMPLÈTE DE P23 : Ω_ts strict + correction plastique ξ_BS.

Modèle :
  m_pred(L) = Ω_ts(L)  +  σ(L) · k · ξ_BS(L)  (additif)
  m_pred(L) = Ω_ts(L) · (1 + σ(L) · k · ξ_BS(L))  (multiplicatif)

Test :
  Calibrer k sur μ⁻ (où Ω_ts strict = 103.26, m_exp = 105.66, ξ_BS = 20.7)
  Évaluer sur les autres particules et comparer RMS

Cas A : V5 assignations publiées (avec leurs gros résidus Ω_ts strict)
Cas B : assignations optimales Ω_ts strict (sub-percent RMS déjà)

L'idée : P23 doit corriger des résidus PETITS (cas B) pour montrer
sa valeur ajoutée, pas combler des gaps de 60% (cas A) qui relèvent
d'une mauvaise topologie.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index
from theory.knot_catalog import _CURVE_REGISTRY

# Constantes Ω_ts
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul des invariants…")
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


# ───────── Biot-Savart radial integral ─────────
def get_cn(name, n_points=180):
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


def F_rad(curve_i, N_i, curves_o, T_o, ds_o):
    F_list = []
    for r_i, n_i in zip(curve_i, N_i):
        B = np.zeros(3)
        for c_j, T_j, ds_j in zip(curves_o, T_o, ds_o):
            B += field_at(r_i, c_j, T_j, ds_j)
        F_list.append(np.abs(np.dot(B, n_i)))
    return float(np.mean(F_list))


def place(component_names, D, n_points=180):
    curves, Ts, Ns, dss = [], [], [], []
    if len(component_names) == 1:
        offsets = [np.zeros(3)]
    elif len(component_names) == 2:
        offsets = [np.zeros(3), np.array([D, 0, 0])]
    else:
        offsets = [np.zeros(3), np.array([D, 0, 0]),
                   np.array([D/2, D*np.sqrt(3)/2, 0])]
    for i, name in enumerate(component_names):
        pts, T, N, ds = get_cn(name, n_points)
        curves.append(pts + offsets[i]); Ts.append(T); Ns.append(N); dss.append(ds)
    return curves, Ts, Ns, dss


def compute_xi_BS(component_names, n_points=150):
    if len(component_names) == 1: return 0.0
    D_grid = np.array([3.0, 4.0, 5.0, 7.0])
    F_vals = []
    for D in D_grid:
        curves, Ts, Ns, dss = place(component_names, D, n_points)
        F_tot = 0
        for i in range(len(component_names)):
            others = [j for j in range(len(component_names)) if j != i]
            F_i = F_rad(curves[i], Ns[i],
                        [curves[j] for j in others],
                        [Ts[j] for j in others],
                        [dss[j] for j in others])
            F_tot += F_i
        F_vals.append(F_tot / len(component_names))
    F_arr = np.array(F_vals)
    valid = F_arr > 1e-10
    if valid.sum() < 2: return 0.0
    slope, intercept = np.polyfit(np.log(D_grid[valid]),
                                   np.log(F_arr[valid]), 1)
    return float(np.exp(intercept))


def sigma_S7(two_J):
    return +1 if two_J == 0 else -1


# ───────── Cas A : assignations V5 publiées ─────────
print("\n" + "═"*100)
print("  Cas A : Assignations V5 publiées + Ω_ts strict + correction P23")
print("═"*100)

CAS_A = [
    # (nm, cfg, 2J, m_exp)
    ("μ⁻",   ["3_1","6_1"],          1, 105.66),
    ("η",    ["3_1","4_1","5_1"],    0, 547.86),
    ("π⁰",   ["8_1"],                0, 134.98),
    ("π±",   ["8_1"],                0, 139.57),
    ("K⁰",   ["4_1","4_1","5_2"],    0, 497.61),
    ("K±",   ["4_1","4_1","5_2"],    0, 493.68),
    ("p",    ["3_1","7_2","7_2"],    1, 938.27),
    ("n",    ["3_1","7_2","7_2"],    1, 939.57),
]

print(f"\n  Calcul ξ_BS et Ω_ts strict pour chaque particule…")
records_A = []
for nm, cfg, two_J, m_exp in CAS_A:
    omega = omega_strict(cfg)
    xi = compute_xi_BS(cfg)
    sigma = sigma_S7(two_J)
    records_A.append({"nm": nm, "cfg": cfg, "omega": omega, "xi": xi,
                       "sigma": sigma, "m_exp": m_exp, "two_J": two_J})

# Calibration k_p sur μ⁻ (correction additive)
mu_rec = next(r for r in records_A if r["nm"] == "μ⁻")
delta_mu = mu_rec["m_exp"] - mu_rec["omega"]  # +2.40 MeV
k_add_A = delta_mu / (mu_rec["sigma"] * mu_rec["xi"])
print(f"  Calibration additive sur μ⁻ : k = {k_add_A:.5f}")

# Calibration multiplicative
ratio_mu = mu_rec["m_exp"] / mu_rec["omega"]
k_mult_A = (ratio_mu - 1) / (mu_rec["sigma"] * mu_rec["xi"])
print(f"  Calibration multiplicative sur μ⁻ : k = {k_mult_A:.6f}")

print(f"\n  {'nm':<5} {'Ω_ts':>10} {'ξ_BS':>8} {'σ':>3} {'m_exp':>10} "
      f"{'Δ Ω':>8} {'+ξBS add':>10} {'·(1+ξBS)':>10}")
print("  " + "-" * 80)
deltas_A_omega = []; deltas_A_add = []; deltas_A_mult = []
for r in records_A:
    d_o = (r["omega"] - r["m_exp"])/r["m_exp"]*100
    m_add = r["omega"] + r["sigma"]*k_add_A*r["xi"]
    m_mult = r["omega"] * (1 + r["sigma"]*k_mult_A*r["xi"])
    d_add = (m_add - r["m_exp"])/r["m_exp"]*100
    d_mult = (m_mult - r["m_exp"])/r["m_exp"]*100
    print(f"  {r['nm']:<5} {r['omega']:>10.2f} {r['xi']:>8.2f} {r['sigma']:>+3} "
          f"{r['m_exp']:>10.2f} {d_o:>+7.2f}% {d_add:>+9.2f}% {d_mult:>+9.2f}%")
    deltas_A_omega.append(d_o)
    deltas_A_add.append(d_add)
    deltas_A_mult.append(d_mult)

print(f"\n  RMS Ω_ts strict          : {np.sqrt(np.mean(np.array(deltas_A_omega)**2)):.2f}%")
print(f"  RMS Ω_ts + ξ_BS (additif): {np.sqrt(np.mean(np.array(deltas_A_add)**2)):.2f}%")
print(f"  RMS Ω_ts · (1+ξ_BS) (mult): {np.sqrt(np.mean(np.array(deltas_A_mult)**2)):.2f}%")


# ───────── Cas B : assignations optimales Ω_ts strict ─────────
print("\n" + "═"*100)
print("  Cas B : Assignations OPTIMALES Ω_ts strict + correction P23")
print("═"*100)
print("\n  (Issues du scan exhaustif précédent : RMS ~ 1% sans correction)")

CAS_B = [
    # (nm, cfg_optimal, 2J, m_exp)
    ("μ⁻",   ["3_1","6_1"],          1, 105.66),  # même que V5
    ("τ⁻",   ["5_1","5_1","6_1"],    1, 1776.86),
    ("η",    ["3_1","5_2","6_1"],    0, 547.86),  # different optimal
    ("π",    ["3_1","3_1","3_1"],    0, 137.27),  # iso-avg
    ("K",    ["6_1","6_1","6_1"],    0, 495.65),  # iso-avg, ma découverte
    ("p",    ["3_1","7_2","7_2"],    1, 938.27),
    ("n",    ["3_1","7_2","7_2"],    1, 939.57),
    ("J/ψ",  ["3_1","4_1","6_3"],    2, 3096.9),
    ("Λ_c",  ["5_1","5_1","7_2"],    1, 2286.46),
    ("D0",   ["3_1","4_1","6_2"],    0, 1864.84),
]

print(f"\n  Calcul ξ_BS et Ω_ts strict pour chaque assignation optimale…")
records_B = []
for nm, cfg, two_J, m_exp in CAS_B:
    omega = omega_strict(cfg)
    xi = compute_xi_BS(cfg)
    sigma = sigma_S7(two_J)
    records_B.append({"nm": nm, "cfg": cfg, "omega": omega, "xi": xi,
                       "sigma": sigma, "m_exp": m_exp, "two_J": two_J})

mu_rec_B = next(r for r in records_B if r["nm"] == "μ⁻")
delta_mu_B = mu_rec_B["m_exp"] - mu_rec_B["omega"]
k_add_B = delta_mu_B / (mu_rec_B["sigma"] * mu_rec_B["xi"]) if mu_rec_B["xi"] > 0 else 0
ratio_mu_B = mu_rec_B["m_exp"] / mu_rec_B["omega"]
k_mult_B = (ratio_mu_B - 1) / (mu_rec_B["sigma"] * mu_rec_B["xi"]) if mu_rec_B["xi"] > 0 else 0
print(f"  Calibration k_add = {k_add_B:.5f}, k_mult = {k_mult_B:.6f}")

print(f"\n  {'nm':<5} {'Ω_ts':>10} {'ξ_BS':>8} {'σ':>3} {'m_exp':>10} "
      f"{'Δ Ω':>8} {'+ξBS add':>10} {'·(1+ξBS)':>10}")
print("  " + "-" * 80)
deltas_B_omega = []; deltas_B_add = []; deltas_B_mult = []
for r in records_B:
    d_o = (r["omega"] - r["m_exp"])/r["m_exp"]*100
    m_add = r["omega"] + r["sigma"]*k_add_B*r["xi"]
    m_mult = r["omega"] * (1 + r["sigma"]*k_mult_B*r["xi"])
    d_add = (m_add - r["m_exp"])/r["m_exp"]*100
    d_mult = (m_mult - r["m_exp"])/r["m_exp"]*100
    print(f"  {r['nm']:<5} {r['omega']:>10.2f} {r['xi']:>8.2f} {r['sigma']:>+3} "
          f"{r['m_exp']:>10.2f} {d_o:>+7.2f}% {d_add:>+9.2f}% {d_mult:>+9.2f}%")
    deltas_B_omega.append(d_o)
    deltas_B_add.append(d_add)
    deltas_B_mult.append(d_mult)

print(f"\n  RMS Ω_ts strict             : {np.sqrt(np.mean(np.array(deltas_B_omega)**2)):.3f}%")
print(f"  RMS Ω_ts + ξ_BS (additif)   : {np.sqrt(np.mean(np.array(deltas_B_add)**2)):.3f}%")
print(f"  RMS Ω_ts · (1+ξ_BS) (mult)  : {np.sqrt(np.mean(np.array(deltas_B_mult)**2)):.3f}%")


# ───────── Synthèse ─────────
print("\n" + "═"*100)
print("  SYNTHÈSE")
print("═"*100)
print(f"""
  CAS A (V5 publié + Ω_ts strict + P23) :
    RMS Ω_ts seul     : {np.sqrt(np.mean(np.array(deltas_A_omega)**2)):.2f}%
    RMS + ξ_BS add    : {np.sqrt(np.mean(np.array(deltas_A_add)**2)):.2f}%
    RMS + ξ_BS mult   : {np.sqrt(np.mean(np.array(deltas_A_mult)**2)):.2f}%

  CAS B (assignations optimales + Ω_ts strict + P23) :
    RMS Ω_ts seul     : {np.sqrt(np.mean(np.array(deltas_B_omega)**2)):.3f}%
    RMS + ξ_BS add    : {np.sqrt(np.mean(np.array(deltas_B_add)**2)):.3f}%
    RMS + ξ_BS mult   : {np.sqrt(np.mean(np.array(deltas_B_mult)**2)):.3f}%

  INTERPRÉTATION :
    Si Cas A montre un gain faible mais Cas B descend sous 1% :
      P23 est un raffinement PERTINENT mais nécessite les bonnes
      assignations topologiques.
    Si gain dans les deux cas :
      P23 est validé universellement.
    Si pas de gain dans Cas B :
      P23 sous sa forme actuelle n'apporte rien quand les topologies
      sont déjà optimales — la correction est dans le bruit.
""")
