r"""
PRINCIPE D'ÉQUILIBRE — minimisation de l'énergie (Ω_ts) sur λ_z.

Hypothèse PRÉDICTIVE :
  Chaque entrelacs adopte l'embedding de chaque composante qui
  MINIMISE la masse-énergie Ω_ts(L) — analogue à l'équilibre
  mécanique d'un système en interaction.

  λ_z^* = argmin_{λ_z} Ω_ts(L; λ_z)

Si cette minimisation reproduit naturellement les masses
expérimentales, le principe d'équilibre est validé prédictivement.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar, minimize
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


def U_KT_deformed(knot_name, lam_z, n_points=6000):
    pts = knot_curve(knot_name, n_points)
    pts_def = pts.copy(); pts_def[:, 2] *= lam_z
    kappa, tau, speed = frenet_curvature_torsion(pts_def)
    ds = speed * 2*np.pi / n_points
    return np.sum(kappa**2 * ds), np.sum(tau**2 * ds)


def normalize_factors(knot_name):
    Uk, Ut = U_KT_deformed(knot_name, 1.0)
    return U_KT[knot_name][0]/Uk, U_KT[knot_name][1]/Ut

print("Normalisation des embeddings…")
NORM = {K: normalize_factors(K) for K in KNOTS}


def U_KT_def_normalized(knot_name, lam_z):
    Uk, Ut = U_KT_deformed(knot_name, lam_z)
    nK, nT = NORM[knot_name]
    return Uk * nK, Ut * nT


def omega_def(comps, deform_dict):
    """Ω_ts avec déformations spécifiées par deform_dict {idx: λ_z}."""
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G_sum = 0
    for i, K in enumerate(comps):
        if i in deform_dict:
            Uk, Ut = U_KT_def_normalized(K, deform_dict[i])
        else:
            Uk, Ut = U_KT[K]
        G_sum += Uk * Ut
    G = G_sum / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── ÉTAPE 1 : embedding d'équilibre INTRINSÈQUE (par knot isolé) ─────────
print("\n" + "═"*82)
print("  ÉTAPE 1 : λ_z* qui minimise U_κ·U_τ pour chaque nœud isolé")
print("═"*82)

print(f"\n  Pour chaque nœud, balayage λ_z ∈ [0.2, 3.0] et minimisation de")
print(f"  U_κ × U_τ (la quantité qui entre dans G de Ω_ts).")
print(f"\n  {'nœud':<5} {'λ_z*':>8} {'U_κ×U_τ min':>14} {'U_κ×U_τ ref':>14} {'ratio':>8}")
print("  " + "-" * 56)

natural_lams = {}
for K in KNOTS:
    def f(lz):
        Uk, Ut = U_KT_def_normalized(K, lz)
        return Uk * Ut
    res = minimize_scalar(f, bounds=(0.2, 3.0), method='bounded', options={'xatol': 1e-4})
    natural_lams[K] = res.x
    ref = U_KT[K][0] * U_KT[K][1]
    deform_pct = (res.x - 1) * 100
    print(f"  {K:<5} {res.x:>8.3f} {res.fun:>14.4f} {ref:>14.4f} {res.fun/ref:>7.3f}  ({deform_pct:+.0f}%)")


# ───────── ÉTAPE 2 : embedding d'équilibre EN CONTEXTE (par entrelacs) ─────────
print("\n" + "═"*82)
print("  ÉTAPE 2 : λ_z minimisant Ω_ts(L) pour chaque entrelacs")
print("═"*82)

# Test particules
PDG_TEST = [
    ("e⁻",        ["3_1"],            0.51099895),
    ("μ⁻",        ["3_1","6_1"],      105.66),
    ("τ⁻",        ["4_1","5_1","6_1"], 1776.86),
    ("π⁰",        ["3_1","3_1","5_2"], 134.98),
    ("π±",        ["3_1","3_1","5_2"], 139.57),
    ("K±",        ["4_1","4_1","4_1"], 493.68),
    ("η",         ["3_1","3_1","6_1"], 547.86),
    ("K*(892)",   ["6_1","6_2"],      891.66),
    ("φ",         ["3_1","5_1","5_2"], 1019.46),
    ("Λ",         ["5_1","8_1"],      1115.68),
    ("Σ⁺",        ["4_1","4_1","5_1"], 1189.37),
    ("Ξ⁰",        ["4_1","5_1","5_2"], 1314.86),
    ("D⁰",        ["3_1","5_1","7_2"], 1864.84),
    ("p",         ["3_1","7_2","7_2"], 938.27),
    ("Λ_c",       ["3_1","3_1","9_1"], 2286.46),
]

print(f"\n  Pour chaque particule, minimisation de Ω_ts(L) sur les")
print(f"  λ_z de chaque composante distincte (UN paramètre par nœud unique).\n")
print(f"  {'particule':<10} {'config':<22} {'m_exp':>8} {'V5':>8} {'V6 min':>8} "
      f"{'Δ V5':>7} {'Δ V6':>7} {'gain':>7}")
print("  " + "-" * 90)

gains = []
for nm, cfg, m_exp in PDG_TEST:
    m_v5 = omega_def(cfg, {})

    # Composantes uniques
    unique_K = list(set(cfg))
    if len(unique_K) == 1 and len(cfg) == 1:
        # Cas trivial : un seul nœud → minimiser sur λ_z seul
        def f1(lz):
            return omega_def(cfg, {0: lz})
        res = minimize_scalar(f1, bounds=(0.2, 3.0), method='bounded')
        m_v6 = res.fun
        lams_str = f"{res.x:.3f}"
    else:
        # Plusieurs composantes : minimisation multi-paramètre
        # Variable : λ_z par nœud unique
        K_to_param = {K: i for i, K in enumerate(unique_K)}
        def f_multi(params):
            d = {idx: params[K_to_param[cfg[idx]]] for idx in range(len(cfg))}
            return omega_def(cfg, d)
        x0 = [natural_lams[K] for K in unique_K]
        bounds = [(0.2, 3.0)] * len(unique_K)
        res = minimize(f_multi, x0, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-9, 'maxiter': 500})
        m_v6 = res.fun
        lams_str = ", ".join(f"{l:.2f}" for l in res.x)

    d_v5 = (m_v5 - m_exp)/m_exp*100
    d_v6 = (m_v6 - m_exp)/m_exp*100
    gain = abs(d_v5) - abs(d_v6)
    gains.append(gain)
    flag = "✓" if gain > 0.01 else ("=" if abs(gain) < 0.01 else "✗")

    print(f"  {nm:<10} {'+'.join(cfg):<22} {m_exp:>8.2f} {m_v5:>8.2f} {m_v6:>8.2f} "
          f"{d_v5:>+6.2f}% {d_v6:>+6.2f}% {gain:>+6.2f} {flag}")

# Statistiques
print(f"\n  ─── BILAN ───")
rms_v5 = np.sqrt(np.mean([((omega_def(cfg, {}) - m_exp)/m_exp*100)**2
                          for nm, cfg, m_exp in PDG_TEST]))
n_better = sum(1 for g in gains if g > 0.01)
n_worse = sum(1 for g in gains if g < -0.01)
print(f"  Particules améliorées par minimisation : {n_better}/{len(PDG_TEST)}")
print(f"  Particules détériorées                  : {n_worse}/{len(PDG_TEST)}")

# ───────── ÉTAPE 3 : diagnostic ─────────
print("\n" + "═"*82)
print("  DIAGNOSTIC : la minimisation est-elle PRÉDICTIVE ?")
print("═"*82)
print(f"""
  Pour chaque particule, V6 minimise Ω_ts → toujours plus petit que V5.
  Mais l'expérience donne des m bien spécifiques.

  • Si m_V6 < m_exp pour toutes : le minimum global est trop bas
    → besoin d'une PÉNALITÉ supplémentaire (rope-length, élasticité)

  • Si m_V6 ≈ m_exp pour la plupart : le principe d'équilibre marche !

  • Si m_V6 = m_V5 (pas de minimum non-trivial) : V5 utilise déjà
    l'embedding optimal.
""")
