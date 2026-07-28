r"""
ÉLASTICITÉ SEULEMENT SUR LE 3_1 — modèle à 1 paramètre global k.

PRINCIPE : le 3_1 est un OSCILLATEUR ÉLASTIQUE autour de λ=1, avec
constante de raideur k. Dans un entrelacs avec d'autres composants,
ces derniers exercent une force qui étire/comprime le 3_1.

Modèle :
  Énergie totale : V(λ) = Ω_ts(L; λ_3_1=λ) + (k/2) (λ-1)²

  Équilibre : λ* = argmin_λ V(λ)
  Masse prédite : m_V6(L) = V(λ*)
                = Ω_ts(L; λ*) + (k/2)(λ*-1)²

  Pour e⁻ : Ω_ts(L=3_1) est INVARIANT par déformation (avec P_31 mis à
  jour). Donc λ* = 1 (force élastique nulle). Donc m_V6(e⁻) = m_V5(e⁻)
  EXACT (ancrage préservé) ✓

  Pour les autres entrelacs : λ* dépend de la pression des autres
  composantes sur le 3_1.

UN SEUL PARAMÈTRE : k (constante de raideur).
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin
from theory.knot_catalog import knot_curve

# ───────── précalcul ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 50000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)), float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]),
               "n_l": N_LOBES[K], "b": braid_index(K),
               "det": jones_determinant(pj), "s": topological_spin(K)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_curvature_torsion(pts):
    n = len(pts); dt = 2*np.pi / n
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    return kappa, tau, speed


def U_KT_3_1_at(lam_z, n_points=4000):
    """U_κ, U_τ du 3_1 à λ_z donné (normalisés à V5 à λ=1)."""
    pts = knot_curve("3_1", n_points)
    pts_def = pts.copy(); pts_def[:, 2] *= lam_z
    kappa, tau, speed = frenet_curvature_torsion(pts_def)
    ds = speed * 2*np.pi / n_points
    Uk = np.sum(kappa**2 * ds)
    Ut = np.sum(tau**2 * ds)
    return Uk * NORM_K, Ut * NORM_T

# Normalisation
_pts = knot_curve("3_1", 4000)
_k, _t, _s = frenet_curvature_torsion(_pts)
_ds = _s * 2*np.pi / 4000
NORM_K = U_KT["3_1"][0] / np.sum(_k**2 * _ds)
NORM_T = U_KT["3_1"][1] / np.sum(_t**2 * _ds)


def omega_at_lambda(comps, lam_3_1):
    """Ω_ts avec TOUS les 3_1 déformés à λ_z = lam_3_1.

    P_31 EST MIS À JOUR pour préserver l'ancrage e⁻ exact.
    """
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    Uk_31_def, Ut_31_def = U_KT_3_1_at(lam_3_1)
    P_31_eff = Uk_31_def * Ut_31_def
    G_sum = 0
    for K in comps:
        if K == "3_1":
            G_sum += Uk_31_def * Ut_31_def
        else:
            G_sum += U_KT[K][0] * U_KT[K][1]
    G = G_sum / P_31_eff
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


def m_elastic(comps, k_elastic, return_lam=False):
    """Masse V6 avec élasticité k sur le 3_1.

    V(λ) = Ω(L; λ) + (k/2)(λ-1)²
    λ* = argmin V
    m = V(λ*)
    """
    if "3_1" not in comps:
        # Pas de 3_1 dans l'entrelacs → pas de correction
        m_v5 = omega_at_lambda(comps, 1.0)
        return (m_v5, 1.0) if return_lam else m_v5

    def V(lam):
        return omega_at_lambda(comps, lam) + 0.5 * k_elastic * (lam - 1)**2

    res = minimize_scalar(V, bounds=(0.5, 2.5), method='bounded',
                          options={'xatol': 1e-5})
    lam_star = res.x
    m_pred = res.fun
    return (m_pred, lam_star) if return_lam else m_pred


# ───────── PDG ÉTENDU ─────────
PDG = [
    ("e⁻",        ["3_1"],              0.51099895),
    ("μ⁻",        ["3_1","6_1"],        105.66),
    ("τ⁻",       ["4_1","5_1","6_1"],    1776.86),
    ("π⁰",        ["3_1","3_1","5_2"],  134.98),
    ("π±",        ["3_1","3_1","5_2"],  139.57),
    ("K±",        ["4_1","4_1","4_1"],  493.68),
    ("η",         ["3_1","3_1","6_1"],  547.86),
    ("η'",        ["7_7","7_7"],        957.78),
    ("ρ",         ["8_1","8_1"],        775.26),
    ("ω",         ["8_1","8_1"],        782.65),
    ("φ",         ["3_1","5_1","5_2"],  1019.46),
    ("K*(892)",   ["6_1","6_2"],        891.66),
    ("J/ψ",       ["3_1","4_1","6_3"],  3096.9),
    ("ψ(2S)",     ["6_1","8_2"],        3686.1),
    ("Υ(1S)",     ["6_2","7_1","7_3"],  9460.30),
    ("D⁰",        ["3_1","5_1","7_2"],  1864.84),
    ("D±",        ["3_1","5_1","7_2"],  1869.66),
    ("D_s",       ["3_1","6_1","7_1"],  1968.34),
    ("B±",        ["3_1","6_2","7_7"],  5279.34),
    ("B⁰",        ["3_1","6_2","7_7"],  5279.66),
    ("B_s",       ["6_3","7_6"],        5366.92),
    ("p",         ["3_1","7_2","7_2"],  938.27),
    ("n",         ["3_1","7_2","7_2"],  939.57),
    ("Λ",         ["5_1","8_1"],        1115.68),
    ("Σ⁺",        ["4_1","4_1","5_1"],  1189.37),
    ("Σ⁰",        ["4_1","4_1","5_1"],  1192.64),
    ("Σ⁻",        ["4_1","4_1","5_1"],  1197.45),
    ("Ξ⁰",        ["4_1","5_1","5_2"],  1314.86),
    ("Ξ⁻",        ["4_1","5_1","5_2"],  1321.71),
    ("Λ_c",       ["3_1","3_1","9_1"],  2286.46),
    ("Ξ_c⁺",      ["3_1","4_1","8_1"],  2467.94),
    ("Ξ_c⁰",      ["3_1","4_1","8_1"],  2470.9),
    ("Λ_b",       ["6_3","7_4"],        5619.6),
    ("W±",        ["7_1","7_2","8_3"],  80369),
    ("Z⁰",        ["7_2","8_3","9_1"],  91188),
    ("H",         ["6_3","8_2","8_3"],  125250),
    ("t",         ["6_3","7_1","7_5"],  172690),
]

print("\n" + "═"*82)
print("  V6 ÉLASTIQUE — formule de masse avec 1 paramètre k sur le 3_1")
print("═"*82)

# Calibrer k sur le RMS global
def rms_global(k_elast):
    err2 = 0
    for nm, cfg, m_exp in PDG:
        m = m_elastic(cfg, k_elast)
        err2 += ((m - m_exp)/m_exp*100)**2
    return np.sqrt(err2/len(PDG))

# Scan
print(f"\n  Scan en k pour trouver l'optimum global :")
print(f"  {'k':>10} {'RMS %':>10}")
print("  " + "-" * 22)
for k in [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7]:
    rms = rms_global(k)
    print(f"  {k:>10.1g} {rms:>9.3f}%")

# Optimisation fine en échelle log
def rms_log_k(log_k):
    return rms_global(np.exp(log_k))

res = minimize_scalar(rms_log_k, bounds=(-2, 20), method='bounded',
                       options={'xatol': 1e-4})
k_opt = np.exp(res.x)
rms_opt = res.fun
print(f"\n  ★ k OPTIMAL = {k_opt:.4e}, RMS = {rms_opt:.3f}%")

# Évaluation détaillée
print("\n" + "═"*82)
print(f"  ÉVALUATION DÉTAILLÉE à k = {k_opt:.3e}")
print("═"*82)
print(f"\n  {'particule':<10} {'config':<22} {'m_exp':>9} {'m_V5':>9} {'m_V6':>9} "
      f"{'λ*':>6} {'Δ V5':>7} {'Δ V6':>7} {'gain':>7}")
print("  " + "-" * 95)

gains = []; v5_errs = []; v6_errs = []
for nm, cfg, m_exp in PDG:
    m_v5 = omega_at_lambda(cfg, 1.0)
    m_v6, lam_star = m_elastic(cfg, k_opt, return_lam=True)
    d_v5 = (m_v5 - m_exp)/m_exp*100
    d_v6 = (m_v6 - m_exp)/m_exp*100
    gain = abs(d_v5) - abs(d_v6)
    gains.append(gain)
    v5_errs.append(d_v5); v6_errs.append(d_v6)
    has_31 = "★" if "3_1" in cfg else " "
    flag = "✓" if gain > 0.01 else ("=" if abs(gain) < 0.01 else "✗")
    print(f"  {has_31}{nm:<9} {'+'.join(cfg):<22} {m_exp:>9.2f} {m_v5:>9.2f} {m_v6:>9.2f} "
          f"{lam_star:>6.3f} {d_v5:>+6.2f}% {d_v6:>+6.2f}% {gain:>+6.2f} {flag}")

# Bilan
rms_v5 = np.sqrt(np.mean(np.array(v5_errs)**2))
rms_v6 = np.sqrt(np.mean(np.array(v6_errs)**2))
n_v5_05 = sum(1 for d in v5_errs if abs(d) < 0.5)
n_v6_05 = sum(1 for d in v6_errs if abs(d) < 0.5)
n_v5_1 = sum(1 for d in v5_errs if abs(d) < 1.0)
n_v6_1 = sum(1 for d in v6_errs if abs(d) < 1.0)

print(f"\n  ─── BILAN ───")
print(f"  RMS V5 : {rms_v5:.3f}%")
print(f"  RMS V6 : {rms_v6:.3f}%")
print(f"  Gain   : {(1-rms_v6/rms_v5)*100:+.1f}%")
print(f"\n  Précision <0.5% : V5 {n_v5_05}/{len(PDG)}  →  V6 {n_v6_05}/{len(PDG)}")
print(f"  Précision <1.0% : V5 {n_v5_1}/{len(PDG)}   →  V6 {n_v6_1}/{len(PDG)}")

# Validation LOO sur particules contenant 3_1
print(f"\n  ─── VALIDATION LEAVE-ONE-OUT ───")
PDG_31 = [(nm, cfg, m_exp) for nm, cfg, m_exp in PDG if "3_1" in cfg]
loo_errs = []
for i in range(len(PDG_31)):
    train = PDG_31[:i] + PDG_31[i+1:] + [(nm, cfg, m_exp) for nm, cfg, m_exp in PDG if "3_1" not in cfg]
    # Optimiser sur train
    def rms_train(log_k):
        kt = np.exp(log_k)
        err2 = 0
        for nm, cfg, m_exp in train:
            m = m_elastic(cfg, kt)
            err2 += ((m - m_exp)/m_exp*100)**2
        return np.sqrt(err2/len(train))
    res_t = minimize_scalar(rms_train, bounds=(-2, 20), method='bounded',
                            options={'xatol': 1e-3})
    k_t = np.exp(res_t.x)
    # Évaluer sur test
    nm_t, cfg_t, m_exp_t = PDG_31[i]
    m_t = m_elastic(cfg_t, k_t)
    err_t = (m_t - m_exp_t)/m_exp_t*100
    loo_errs.append(err_t)

rms_loo = np.sqrt(np.mean(np.array(loo_errs)**2))
print(f"  RMS LOO V6 (sur particules avec 3_1) : {rms_loo:.3f}%")
# Comparer au V5 sur le même sous-ensemble
v5_31 = [(omega_at_lambda(cfg, 1.0) - m_exp)/m_exp*100 for nm, cfg, m_exp in PDG_31]
rms_v5_31 = np.sqrt(np.mean(np.array(v5_31)**2))
print(f"  RMS V5 (sur particules avec 3_1)     : {rms_v5_31:.3f}%")
if rms_loo < rms_v5_31:
    print(f"  ★ V6 PASSE la LOO : {rms_loo:.3f}% < {rms_v5_31:.3f}% (V5)")
else:
    print(f"  ⚠ V6 sur-ajuste : LOO {rms_loo:.3f}% > V5 {rms_v5_31:.3f}%")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Formule V6 ÉLASTIQUE :
    m_V6(L) = Ω_ts(L; λ*) + (k/2)(λ*-1)²
    où λ* minimise Ω_ts(L; λ) + (k/2)(λ-1)²

  UN SEUL paramètre : k = {k_opt:.3e}

  Performance :
    RMS V5 : {rms_v5:.3f}%
    RMS V6 : {rms_v6:.3f}%
    Gain   : {(1-rms_v6/rms_v5)*100:+.1f}%
    LOO    : {rms_loo:.3f}% (sur particules avec 3_1)

  PROPRIÉTÉS :
    ✓ e⁻ EXACT (par construction, P_31 mis à jour)
    ✓ Particules sans 3_1 INCHANGÉES
    ✓ UN seul paramètre global
""")
