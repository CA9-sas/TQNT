r"""
V5-μΛ : formule de masse anchorée sur μ⁻ et Λ pour les entrelacs n=2.

Constat : la V5 actuelle est anchorée principalement sur m_e (n=1, α₀)
et m_p, m_n (n=3, via β tension harmonique et k_b). Les particules n=2
ont des résidus systématiques (~1.5%) non-corrélés.

Proposition : utiliser m_μ et m_Λ comme ANCRES pour fixer DEUX
paramètres spécifiques à n=2 (analogue au double-anchor p/n pour n=3).

PARAMÈTRES INTRODUITS :
  a_2     : exposant Hölder pour n=2 (V5 utilise a = √(7/(8π)) ≈ 0.528)
  α_2     : facteur d'échelle multiplicatif pour n=2

ÉQUATIONS D'ANCRAGE :
  Ω_μΛ(3_1+6_1; a_2, α_2) = m_μ = 105.66 MeV
  Ω_μΛ(5_1+8_1; a_2, α_2) = m_Λ = 1115.68 MeV

→ 2 équations, 2 inconnues → solution unique.

ÉVALUATION : appliquer Ω_μΛ à toutes les autres particules n=2 et
comparer aux résidus V5.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# ───────── précalcul ─────────
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A_ANA = np.sqrt(7/(8*np.pi)); P_ANA = 3/4; Q_ANA = 3/4
X_ANA = -np.sqrt(6); B_ANA = 33/20; RHO_0 = 10 - 3/(8*np.pi)

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


def omega_v5(comps):
    """V5 standard (référence)."""
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_ANA for K in comps]
    skel = float(sum(w**A_ANA for w in ws)**(1/A_ANA))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_ANA)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B_ANA))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_ANA


def omega_muLambda(comps, a_2, alpha_2):
    """V5-μΛ : remplace a et Φ_n par valeurs anchorées sur μ⁻ et Λ.

    Pour n=2 : utilise a_2 (Hölder) et alpha_2 (facteur d'échelle).
    Pour n≠2 : identique à V5 (non touché).
    """
    n = len(comps)
    if n != 2:
        return omega_v5(comps)
    # n=2 : nouvelle formule
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_ANA for K in comps]
    # Hölder avec a_2 au lieu de a
    skel = float(sum(w**a_2 for w in ws)**(1/a_2))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B_ANA))**(ct-1)
    # alpha_2 absorbe Φ_n adapté à n=2
    return alpha_2 * Phi_p * skel * G**Q_ANA


# ───────── ancrage exact sur (μ, Λ) ─────────
print("\n" + "═"*82)
print("  V5-μΛ : ancrage exact sur m_μ et m_Λ")
print("═"*82)

M_MU = 105.6583755
M_LAMBDA = 1115.683

def chi2_anchor(params):
    a_2, alpha_2 = params
    m_mu_pred = omega_muLambda(["3_1","6_1"], a_2, alpha_2)
    m_la_pred = omega_muLambda(["5_1","8_1"], a_2, alpha_2)
    return (np.log(M_MU/m_mu_pred))**2 + (np.log(M_LAMBDA/m_la_pred))**2

# Optimisation : trouver a_2, alpha_2 qui rendent exact
res = minimize(chi2_anchor, [A_ANA, 1.5], method='Nelder-Mead',
               options={'xatol': 1e-9, 'fatol': 1e-15, 'maxiter': 5000})
a_2_opt, alpha_2_opt = res.x

print(f"\n  Paramètres anchorés (μ⁻, Λ) :")
print(f"    a_2 (Hölder n=2)    = {a_2_opt:.6f}")
print(f"    α_2 (échelle n=2)   = {alpha_2_opt:.6f}")
print(f"    (comparaison V5 : a = {A_ANA:.6f})")
print(f"\n  Vérification d'ancrage :")
m_mu_check = omega_muLambda(["3_1","6_1"], a_2_opt, alpha_2_opt)
m_la_check = omega_muLambda(["5_1","8_1"], a_2_opt, alpha_2_opt)
print(f"    μ⁻ : m_V5 = {omega_v5(['3_1','6_1']):>9.3f},  m_μΛ = {m_mu_check:>9.3f},  exp = {M_MU:.3f}")
print(f"    Λ  : m_V5 = {omega_v5(['5_1','8_1']):>9.3f},  m_μΛ = {m_la_check:>9.3f},  exp = {M_LAMBDA:.3f}")

# ───────── PDG n=2 ─────────
PDG_N2 = [
    ("μ⁻",        ["3_1","6_1"],      105.66,    "ANCRE"),
    ("Λ",         ["5_1","8_1"],      1115.68,   "ANCRE"),
    ("η'",        ["7_7","7_7"],      957.78,    "test"),
    ("ρ",         ["8_1","8_1"],      775.26,    "test"),
    ("ω",         ["8_1","8_1"],      782.65,    "test"),
    ("K*(892)",   ["6_1","6_2"],      891.66,    "test"),
    ("ψ(2S)",     ["6_1","8_2"],      3686.1,    "test"),
    ("χ_c2",      ["6_1","8_2"],      3556.17,   "test"),
    ("η_c(2S)",   ["6_1","8_2"],      3637.5,    "test"),
    ("Υ(3S)",     ["7_2","10_1"],     10355.2,   "test"),
    ("D*(2007)⁰", ["7_7","9_1"],      2006.85,   "test"),
    ("D*(2010)±", ["7_7","9_1"],      2010.26,   "test"),
    ("B_s",       ["6_3","7_6"],      5366.92,   "test"),
    ("Λ_b",       ["6_3","7_4"],      5619.6,    "test"),
    ("Σ_b",       ["7_3","7_4"],      5811.0,    "test"),
    ("Ξ_b⁰",      ["7_3","7_4"],      5791.9,    "test"),
    ("Ξ_b⁻",      ["7_3","7_4"],      5797.0,    "test"),
]

# ───────── évaluation comparative ─────────
print(f"\n  ─── COMPARAISON V5 vs V5-μΛ sur n=2 ───")
print(f"  {'particule':<12} {'config':<12} {'m_exp':>10} {'V5':>10} {'V5-μΛ':>10} "
      f"{'Δ V5':>8} {'Δ μΛ':>8} {'gain':>8} {'rôle':>6}")
print("  " + "-" * 96)
rms_v5 = 0; rms_ml = 0; gains = []
for nm, cfg, m_exp, role in PDG_N2:
    m_v5 = omega_v5(cfg)
    m_ml = omega_muLambda(cfg, a_2_opt, alpha_2_opt)
    d_v5 = (m_exp - m_v5)/m_exp*100
    d_ml = (m_exp - m_ml)/m_exp*100
    rms_v5 += d_v5**2; rms_ml += d_ml**2
    gain = abs(d_v5) - abs(d_ml)
    gains.append(gain)
    flag = "✓" if gain > 0.01 else ("=" if abs(gain) < 0.01 else "✗")
    print(f"  {nm:<12} {'+'.join(cfg):<12} {m_exp:>10.2f} {m_v5:>10.2f} {m_ml:>10.2f} "
          f"{d_v5:>+7.2f}% {d_ml:>+7.2f}% {gain:>+7.2f} {role:>6} {flag}")

rms_v5 = np.sqrt(rms_v5/len(PDG_N2))
rms_ml = np.sqrt(rms_ml/len(PDG_N2))

# Statistiques sur particules TEST seules (sans les ancres)
test_only = [(nm, cfg, m_exp) for nm, cfg, m_exp, r in PDG_N2 if r == "test"]
rms_v5_test = 0; rms_ml_test = 0; gain_test = []
for nm, cfg, m_exp in test_only:
    m_v5 = omega_v5(cfg); m_ml = omega_muLambda(cfg, a_2_opt, alpha_2_opt)
    d_v5 = (m_exp - m_v5)/m_exp*100; d_ml = (m_exp - m_ml)/m_exp*100
    rms_v5_test += d_v5**2; rms_ml_test += d_ml**2
    gain_test.append(abs(d_v5) - abs(d_ml))
rms_v5_test = np.sqrt(rms_v5_test/len(test_only))
rms_ml_test = np.sqrt(rms_ml_test/len(test_only))

print(f"\n  ─── BILAN ───")
print(f"    RMS V5 (toutes n=2)    : {rms_v5:.3f}%")
print(f"    RMS μΛ (toutes n=2)    : {rms_ml:.3f}%  (mais 2 sont ancres exactes)")
print(f"\n    RMS V5 (test seul)     : {rms_v5_test:.3f}%   ({len(test_only)} particules)")
print(f"    RMS μΛ (test seul)     : {rms_ml_test:.3f}%   ({len(test_only)} particules)")
print(f"    Gain μΛ sur test       : {(1-rms_ml_test/rms_v5_test)*100:+.1f}%")

n_better = sum(1 for g in gain_test if g > 0.05)
n_worse = sum(1 for g in gain_test if g < -0.05)
print(f"    Particules améliorées  : {n_better}/{len(test_only)}")
print(f"    Particules détériorées : {n_worse}/{len(test_only)}")

# ───────── non-régression sur n=1, n=3 ─────────
print(f"\n  ─── NON-RÉGRESSION sur n=1 et n=3 ───")
OTHER = [
    ("e⁻",       ["3_1"],              0.51099895),
    ("τ⁻",       ["4_1","5_1","6_1"],  1776.86),
    ("π⁰",       ["3_1","3_1","5_2"],  134.98),
    ("K±",       ["4_1","4_1","4_1"],  493.68),
    ("J/ψ",      ["3_1","4_1","6_3"],  3096.9),
    ("Υ(1S)",    ["6_2","7_1","7_3"],  9460.30),
    ("p",        ["3_1","7_2","7_2"],  938.27),
    ("n",        ["3_1","7_2","7_2"],  939.57),
    ("W±",       ["7_1","7_2","8_3"],  80369),
    ("Z⁰",       ["7_2","8_3","9_1"],  91188),
    ("H",        ["6_3","8_2","8_3"],  125250),
    ("t",        ["6_3","7_1","7_5"],  172690),
]
print(f"  {'particule':<10} {'n':>2} {'Δ V5':>9} {'Δ V5-μΛ':>10}")
print("  " + "-" * 36)
for nm, cfg, m_exp in OTHER:
    m_v5 = omega_v5(cfg); m_ml = omega_muLambda(cfg, a_2_opt, alpha_2_opt)
    d_v5 = (m_exp - m_v5)/m_exp*100
    d_ml = (m_exp - m_ml)/m_exp*100
    print(f"  {nm:<10} {len(cfg):>2} {d_v5:>+8.3f}% {d_ml:>+9.3f}%")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Paramètres V5-μΛ (anchorés sur μ⁻ et Λ) :
    a_2 = {a_2_opt:.4f}  (vs V5 : a = {A_ANA:.4f})
    α_2 = {alpha_2_opt:.4f}  (facteur d'échelle n=2)

  Performance comparée (sur 15 particules n=2 test, hors ancres) :
    RMS V5    : {rms_v5_test:.3f}%
    RMS V5-μΛ : {rms_ml_test:.3f}%
    Gain      : {(1-rms_ml_test/rms_v5_test)*100:+.1f}%

  Amélioration  : {n_better}/{len(test_only)} particules
  Détérioration : {n_worse}/{len(test_only)} particules

  ★ μ⁻ et Λ sont MAINTENANT EXACTS (par construction).
  ★ Les autres particules n=2 sont mieux ou moins bien.
  ★ Pour n=1 et n=3 : INCHANGÉES (formule identique).
""")
