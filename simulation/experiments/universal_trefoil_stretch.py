r"""
HYPOTHÈSE : le trefle 3_1 est UNIVERSELLEMENT étiré à λ_z ≈ 1.5
dans tous les entrelacs (sauf le lepton isolé e⁻ = 3_1 seul).

Test : remplacer U_κ(3_1) et U_τ(3_1) par leurs valeurs déformées
       pour λ_z dans [1.0, 2.0] et calculer Ω_ts sur tout le PDG.

Recherche du λ optimal qui minimise le RMS global.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin
from theory.knot_catalog import knot_curve

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul Frenet…")
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


def U_KT_deformed_3_1(lam_z, n_points=8000):
    pts = knot_curve("3_1", n_points)
    pts_def = pts.copy(); pts_def[:, 2] *= lam_z
    kappa, tau, speed = frenet_curvature_torsion(pts_def)
    ds = speed * 2*np.pi / n_points
    Uk = np.sum(kappa**2 * ds)
    Ut = np.sum(tau**2 * ds)
    # Normaliser pour matcher V5 à λ=1
    Uk_std, Ut_std = U_KT_deformed_3_1.std_cache if hasattr(U_KT_deformed_3_1, 'std_cache') else (None, None)
    return Uk, Ut

# Normalisation pour λ=1 ≡ V5
pts_std = knot_curve("3_1", 8000)
kappa_std, tau_std, speed_std = frenet_curvature_torsion(pts_std)
ds_std = speed_std * 2*np.pi / 8000
Uk_my_1 = np.sum(kappa_std**2 * ds_std)
Ut_my_1 = np.sum(tau_std**2 * ds_std)
NORM_K = U_KT["3_1"][0] / Uk_my_1
NORM_T = U_KT["3_1"][1] / Ut_my_1
print(f"  Normalisation 3_1 : NORM_K = {NORM_K:.4f}, NORM_T = {NORM_T:.4f}")


def U_KT_3_1_at(lam_z):
    Uk, Ut = U_KT_deformed_3_1(lam_z)
    return Uk * NORM_K, Ut * NORM_T


def omega_with_lambda(comps, lam_3_1=1.0):
    """Ω_ts avec TOUS les 3_1 déformés à λ_z = lam_3_1.

    Redéfinit P_31 = U_κ_def × U_τ_def pour PRÉSERVER l'ancrage e⁻.
    Cela garantit que omega(['3_1']) = m_e à λ quelconque.
    """
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    # Recalcul du P_31 pour préserver l'ancrage e⁻
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


# ───────── PDG ÉTENDU ─────────
PDG = [
    # Particules CONTENANT 3_1 (vont être affectées)
    ("e⁻",        ["3_1"],              0.51099895, "★ contient 3_1"),
    ("μ⁻",        ["3_1","6_1"],        105.66,     "★ contient 3_1"),
    ("π⁰",        ["3_1","3_1","5_2"],  134.98,     "★ contient 3_1×2"),
    ("π±",        ["3_1","3_1","5_2"],  139.57,     "★ contient 3_1×2"),
    ("η",         ["3_1","3_1","6_1"],  547.86,     "★ contient 3_1×2"),
    ("φ",         ["3_1","5_1","5_2"],  1019.46,    "★ contient 3_1"),
    ("D⁰",        ["3_1","5_1","7_2"],  1864.84,    "★ contient 3_1"),
    ("D±",        ["3_1","5_1","7_2"],  1869.66,    "★ contient 3_1"),
    ("D_s",       ["3_1","6_1","7_1"],  1968.34,    "★ contient 3_1"),
    ("J/ψ",       ["3_1","4_1","6_3"],  3096.9,     "★ contient 3_1"),
    ("ψ(3770)",   ["3_1","7_2","7_7"],  3773.7,     "★ contient 3_1"),
    ("B±",        ["3_1","6_2","7_7"],  5279.34,    "★ contient 3_1"),
    ("B⁰",        ["3_1","6_2","7_7"],  5279.66,    "★ contient 3_1"),
    ("B*",        ["3_1","6_2","7_7"],  5324.71,    "★ contient 3_1"),
    ("p",         ["3_1","7_2","7_2"],  938.27,     "★ contient 3_1"),
    ("n",         ["3_1","7_2","7_2"],  939.57,     "★ contient 3_1"),
    ("Λ_c",       ["3_1","3_1","9_1"],  2286.46,    "★ contient 3_1×2"),
    ("Ξ_c⁺",      ["3_1","4_1","8_1"],  2467.94,    "★ contient 3_1"),
    ("Ξ_c⁰",      ["3_1","4_1","8_1"],  2470.9,     "★ contient 3_1"),
    # Particules SANS 3_1 (devraient être INCHANGÉES)
    ("τ⁻",        ["4_1","5_1","6_1"],  1776.86,    "PAS de 3_1"),
    ("K±",        ["4_1","4_1","4_1"],  493.68,     "PAS de 3_1"),
    ("ρ",         ["8_1","8_1"],        775.26,     "PAS de 3_1"),
    ("ω",         ["8_1","8_1"],        782.65,     "PAS de 3_1"),
    ("K*(892)",   ["6_1","6_2"],        891.66,     "PAS de 3_1"),
    ("η'",        ["7_7","7_7"],        957.78,     "PAS de 3_1"),
    ("ψ(2S)",     ["6_1","8_2"],        3686.1,     "PAS de 3_1"),
    ("Λ",         ["5_1","8_1"],        1115.68,    "PAS de 3_1"),
    ("Σ⁺",        ["4_1","4_1","5_1"],  1189.37,    "PAS de 3_1"),
    ("Σ⁰",        ["4_1","4_1","5_1"],  1192.64,    "PAS de 3_1"),
    ("Σ⁻",        ["4_1","4_1","5_1"],  1197.45,    "PAS de 3_1"),
    ("Ξ⁰",        ["4_1","5_1","5_2"],  1314.86,    "PAS de 3_1"),
    ("Ξ⁻",        ["4_1","5_1","5_2"],  1321.71,    "PAS de 3_1"),
    ("Λ_b",       ["6_3","7_4"],        5619.6,     "PAS de 3_1"),
    ("W±",        ["7_1","7_2","8_3"],  80369,      "PAS de 3_1"),
    ("Z⁰",        ["7_2","8_3","9_1"],  91188,      "PAS de 3_1"),
    ("H",         ["6_3","8_2","8_3"],  125250,     "PAS de 3_1"),
    ("t",         ["6_3","7_1","7_5"],  172690,     "PAS de 3_1"),
]

# ───────── SCAN λ_3_1 ─────────
print("\n" + "═"*82)
print("  SCAN — λ optimal du trefle 3_1 universel")
print("═"*82)

# Particules avec 3_1
PDG_3_1 = [p for p in PDG if "3_1" in p[1]]
PDG_NO_3_1 = [p for p in PDG if "3_1" not in p[1]]

print(f"\n  Particules contenant 3_1   : {len(PDG_3_1)}")
print(f"  Particules SANS 3_1        : {len(PDG_NO_3_1)}")

# Référence V5
def compute_rms(subset, lam_3_1):
    err2 = 0
    for nm, cfg, m_exp, _ in subset:
        m_pred = omega_with_lambda(cfg, lam_3_1)
        err2 += ((m_pred - m_exp)/m_exp*100)**2
    return np.sqrt(err2/len(subset)) if subset else 0

print(f"\n  RMS V5 brut (λ=1.0) sur sous-ensembles :")
print(f"    Avec 3_1   : {compute_rms(PDG_3_1, 1.0):.3f}%")
print(f"    Sans 3_1   : {compute_rms(PDG_NO_3_1, 1.0):.3f}%")
print(f"    GLOBAL     : {compute_rms(PDG, 1.0):.3f}%")

# Scan en λ
print(f"\n  Scan en λ_3_1 ∈ [0.8, 2.0] :")
print(f"  {'λ_3_1':>7} {'RMS avec 3_1':>14} {'RMS sans 3_1':>14} {'RMS GLOBAL':>12}")
print("  " + "-" * 56)
best_lam = 1.0; best_rms = float('inf')
for lam in np.linspace(0.8, 2.0, 25):
    rms_with = compute_rms(PDG_3_1, lam)
    rms_without = compute_rms(PDG_NO_3_1, lam)
    rms_glob = compute_rms(PDG, lam)
    if rms_glob < best_rms:
        best_rms = rms_glob; best_lam = lam
    flag = "★" if rms_glob == best_rms else ""
    print(f"  {lam:>7.3f} {rms_with:>13.3f}% {rms_without:>13.3f}% {rms_glob:>11.3f}% {flag}")

# Affiner autour du meilleur
print(f"\n  Affinement autour de λ = {best_lam:.3f} :")
for lam in np.linspace(best_lam - 0.1, best_lam + 0.1, 11):
    rms_glob = compute_rms(PDG, lam)
    flag = "★" if abs(rms_glob - best_rms) < 0.01 else ""
    print(f"    λ = {lam:.4f}  RMS = {rms_glob:.3f}%  {flag}")

from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda lam: compute_rms(PDG, lam), bounds=(0.8, 2.5),
                      method='bounded', options={'xatol': 1e-4})
print(f"\n  → λ OPTIMAL = {res.x:.4f}, RMS = {res.fun:.3f}%")
LAM_OPT = res.x

# Évaluation détaillée
print("\n" + "═"*82)
print(f"  ÉVALUATION DÉTAILLÉE à λ(3_1) = {LAM_OPT:.4f}")
print("═"*82)
print(f"\n  {'particule':<12} {'config':<22} {'m_exp':>9} {'V5':>9} {'V6':>9} "
      f"{'Δ V5':>7} {'Δ V6':>7} {'gain':>7}")
print("  " + "-" * 88)

gains_31 = []; gains_no31 = []
v5_errs = []; v6_errs = []
for nm, cfg, m_exp, note in PDG:
    m_v5 = omega_with_lambda(cfg, 1.0)
    m_v6 = omega_with_lambda(cfg, LAM_OPT)
    d_v5 = (m_v5 - m_exp)/m_exp*100
    d_v6 = (m_v6 - m_exp)/m_exp*100
    gain = abs(d_v5) - abs(d_v6)
    v5_errs.append(abs(d_v5)); v6_errs.append(abs(d_v6))
    if "3_1" in cfg: gains_31.append(gain)
    else: gains_no31.append(gain)
    flag = "✓" if gain > 0.01 else ("=" if abs(gain) < 0.01 else "✗")
    has31 = "★" if "3_1" in cfg else " "
    print(f"  {has31}{nm:<11} {'+'.join(cfg):<22} {m_exp:>9.2f} {m_v5:>9.2f} {m_v6:>9.2f} "
          f"{d_v5:>+6.2f}% {d_v6:>+6.2f}% {gain:>+6.2f} {flag}")

# Bilan
print(f"\n  ─── BILAN ───")
arr_31 = np.array(gains_31); arr_no = np.array(gains_no31)
print(f"  Particules CONTENANT 3_1 ({len(arr_31)}) :")
print(f"    Améliorées  : {(arr_31 > 0.01).sum()}/{len(arr_31)}")
print(f"    Détériorées : {(arr_31 < -0.01).sum()}/{len(arr_31)}")
print(f"    Gain moyen  : {arr_31.mean():+.3f}%")
print(f"  Particules SANS 3_1 ({len(arr_no)}) :")
print(f"    Inchangées  : {(np.abs(arr_no) < 0.001).sum()}/{len(arr_no)} (attendu : tous)")

# RMS global
rms_v5_glob = np.sqrt(np.mean(np.array(v5_errs)**2))
rms_v6_glob = np.sqrt(np.mean(np.array(v6_errs)**2))
print(f"\n  RMS V5 global : {rms_v5_glob:.3f}%")
print(f"  RMS V6 global : {rms_v6_glob:.3f}%")
print(f"  Réduction     : {(1 - rms_v6_glob/rms_v5_glob)*100:.1f}%")

# Cas particuliers
print(f"\n  ─── CAS PARTICULIERS (e⁻ et grosses améliorations) ───")
m_e_v5 = omega_with_lambda(["3_1"], 1.0)
m_e_v6 = omega_with_lambda(["3_1"], LAM_OPT)
print(f"  e⁻ : V5 = {m_e_v5:.6f}, V6 = {m_e_v6:.6f}, exp = 0.511")
print(f"    Note : si l'e⁻ change, l'ancrage α_0 doit être ré-évalué")
