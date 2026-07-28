r"""
RAFFINEMENT du facteur de comptage Φ_n pour n=2.

Constat (cas μ⁻ vs τ⁻) : V5 prédit le τ (n=3) à 0.04% mais le μ⁻
(n=2) à 2.27%. La formule Φ_n = ((n+1)/2)^(-x), x=-√6, semble bien
calibrée pour n=3 mais imparfaite pour n=2.

Démarche :
  1. Extraire toutes les particules n=2 du PDG
  2. Analyser les résidus en fonction de divers invariants (c_total,
     asymétrie, identité, etc.)
  3. Proposer une forme paramétrée Φ_2(K_i, K_j) à 1-2 paramètres
  4. Calibrer et valider par CV stratifiée
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

# Précalcul standard
M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P = 3/4; Q = 3/4
X_ANA = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)

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


def omega_v5(comps):
    """V5 baseline (Φ_n V5)."""
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_ANA)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q


def omega_v6_n2(comps, gamma, delta):
    """V6 avec Φ_2 raffiné par (1 + γ·asym + δ·log(c_tot/c_ref))."""
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_ANA)
    # Correction Φ_n2 SEULEMENT si n=2
    if n == 2:
        cs = [TOPO[K]["cr"] for K in comps]
        c_tot = sum(cs)
        asym = abs(cs[0] - cs[1]) / (cs[0] + cs[1])  # ∈ [0, 1]
        c_ref = 10  # référence empirique
        Phi_n = Phi_n * (1 + gamma * asym + delta * np.log(c_tot / c_ref))
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q


# ───────── PDG : particules n=2 essentiellement propres ─────────
PDG_N2 = [
    # (nom, config, m_exp, Γ_MeV, classe)
    ("μ⁻",        ["3_1","6_1"],      105.66,    3e-16, "lepton"),
    ("Λ",         ["5_1","8_1"],      1115.68,   2.5e-15, "hyperon"),
    ("η'",        ["7_7","7_7"],      957.78,    0.196, "meson"),
    ("ρ",         ["8_1","8_1"],      775.26,    147.4, "meson_large"),
    ("ω",         ["8_1","8_1"],      782.65,    8.49, "meson"),
    ("K*(892)",   ["6_1","6_2"],      891.66,    51.4, "meson_large"),
    ("ψ(2S)",     ["6_1","8_2"],      3686.1,    0.294, "charmonium"),
    ("χ_c2",      ["6_1","8_2"],      3556.17,   1.97, "charmonium"),
    ("η_c(2S)",   ["6_1","8_2"],      3637.5,    11.3, "charmonium"),
    ("Υ(3S)",     ["7_2","10_1"],     10355.2,   0.0202, "bottomonium"),
    ("D*(2007)⁰", ["7_7","9_1"],      2006.85,   2.1e-3, "meson"),
    ("D*(2010)±", ["7_7","9_1"],      2010.26,   8.34e-2, "meson"),
    ("B_s",       ["6_3","7_6"],      5366.92,   4.39e-10, "meson"),
    ("Λ_b",       ["6_3","7_4"],      5619.6,    4.6e-10, "baryon"),
    ("Σ_b",       ["7_3","7_4"],      5811.0,    5.0, "baryon_large"),
    ("Ξ_b⁰",      ["7_3","7_4"],      5791.9,    3e-10, "baryon"),
    ("Ξ_b⁻",      ["7_3","7_4"],      5797.0,    3e-10, "baryon"),
]

print("\n" + "═"*82)
print("  RAFFINEMENT Φ_n pour n=2 — analyse exploratoire")
print("═"*82)

print("\n  Particules n=2 avec leurs résidus V5 :")
print(f"  {'particule':<14} {'config':<14} {'m_exp':>10} {'Ω_ts':>10} {'Δ %':>8} {'c_tot':>6} {'asym':>6} {'Γ/m':>10}")
print("  " + "-" * 86)
data_n2 = []
for nm, cfg, m_exp, gamma, classe in PDG_N2:
    m_pred = omega_v5(cfg)
    delta = (m_exp - m_pred) / m_exp * 100
    cs = [TOPO[K]["cr"] for K in cfg]
    c_tot = sum(cs)
    asym = abs(cs[0] - cs[1]) / (cs[0] + cs[1])
    gm = gamma / m_exp
    data_n2.append((nm, cfg, m_exp, m_pred, delta, c_tot, asym, gm, classe))
    print(f"  {nm:<14} {'+'.join(cfg):<14} {m_exp:>10.2f} {m_pred:>10.2f} "
          f"{delta:>+7.2f}% {c_tot:>6} {asym:>6.3f} {gm:>10.2e}")

# Filtrer particules propres (Γ/m < 1%)
clean_n2 = [d for d in data_n2 if d[7] < 0.01]
print(f"\n  Particules propres n=2 (Γ/m < 1%) : {len(clean_n2)}/{len(data_n2)}")

# Statistiques V5 sur n=2 propres
deltas = np.array([d[4] for d in clean_n2])
print(f"\n  V5 sur n=2 propres :")
print(f"    Δ moyen  : {deltas.mean():+.3f}%")
print(f"    Δ médian : {np.median(deltas):+.3f}%")
print(f"    Δ RMS    : {np.sqrt(np.mean(deltas**2)):.3f}%")
print(f"    Δ max abs: {np.max(np.abs(deltas)):.3f}%")

# Corrélations avec invariants
print(f"\n  Corrélations résidu V5 ~ invariant n=2 :")
c_tots = np.array([d[5] for d in clean_n2])
asyms = np.array([d[6] for d in clean_n2])
masses = np.array([d[2] for d in clean_n2])
print(f"    r(Δ, c_tot)  = {np.corrcoef(deltas, c_tots)[0,1]:+.3f}")
print(f"    r(Δ, asym)   = {np.corrcoef(deltas, asyms)[0,1]:+.3f}")
print(f"    r(Δ, log m)  = {np.corrcoef(deltas, np.log(masses))[0,1]:+.3f}")
print(f"    r(Δ, log c)  = {np.corrcoef(deltas, np.log(c_tots))[0,1]:+.3f}")

# ───────── Optimisation : Φ_2 → Φ_2 × (1 + γ·asym + δ·log(c/c_ref)) ─────────
print("\n  ─── CALIBRATION (γ asymétrie, δ taille log) ───")
print("  Forme :  Φ_2 → Φ_2 × (1 + γ·asym + δ·log(c_tot/10))")

def chi2(params, dataset):
    gamma, delta = params
    total = 0.0
    for nm, cfg, m_exp, _, _, _, _, _, _ in dataset:
        m_pred = omega_v6_n2(cfg, gamma, delta)
        if m_pred <= 0: return 1e10
        total += np.log(m_exp/m_pred)**2
    return total / len(dataset)

# Optimisation sur n=2 propres
res = minimize(chi2, [0.0, 0.0], args=(clean_n2,), method='Nelder-Mead',
               options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 3000})
gamma_opt, delta_opt = res.x
print(f"\n  Paramètres optimisés :")
print(f"    γ (asymétrie)        = {gamma_opt:+.4f}")
print(f"    δ (taille log c_tot) = {delta_opt:+.4f}")

# Évaluation V5 vs V6 sur n=2 propres
print(f"\n  ─── COMPARAISON V5 vs V6 (Φ_n2 raffiné) ───")
print(f"  {'particule':<14} {'Δ V5 %':>10} {'Δ V6 %':>10} {'gain':>8}")
print("  " + "-" * 50)
rms5 = 0; rms6 = 0; gains_pos = 0
for nm, cfg, m_exp, _, d5, c_tot, asym, gm, _ in clean_n2:
    m_v6 = omega_v6_n2(cfg, gamma_opt, delta_opt)
    d6 = (m_exp - m_v6)/m_exp*100
    rms5 += d5**2; rms6 += d6**2
    gain = abs(d5) - abs(d6)
    if gain > 0.01: gains_pos += 1
    flag = "✓" if gain > 0.01 else ("=" if abs(gain) <= 0.01 else "✗")
    print(f"  {nm:<14} {d5:>+9.2f}% {d6:>+9.2f}% {gain:>+7.2f} {flag}")
rms5 = np.sqrt(rms5/len(clean_n2))
rms6 = np.sqrt(rms6/len(clean_n2))
print(f"\n  RMS V5 (n=2 propres)  : {rms5:.3f}%")
print(f"  RMS V6 (n=2 propres)  : {rms6:.3f}%")
print(f"  Réduction             : {(1 - rms6/rms5)*100:.1f}%")
print(f"  Particules améliorées : {gains_pos}/{len(clean_n2)}")

# ───────── Validation croisée Leave-One-Out (LOO) ─────────
print(f"\n  ─── VALIDATION CROISÉE LEAVE-ONE-OUT ───")
loo_errors = []
for i in range(len(clean_n2)):
    train = clean_n2[:i] + clean_n2[i+1:]
    test = clean_n2[i]
    res_cv = minimize(chi2, [gamma_opt, delta_opt], args=(train,), method='Nelder-Mead',
                      options={'xatol': 1e-5, 'fatol': 1e-9, 'maxiter': 1500})
    g_cv, d_cv = res_cv.x
    m_v6_test = omega_v6_n2(test[1], g_cv, d_cv)
    loo_errors.append(np.log(test[2] / m_v6_test)**2)

loo_rms_log = np.sqrt(np.mean(loo_errors))
loo_rms_pct = loo_rms_log * 100  # approximation log → %
print(f"\n  RMS LOO V6            : {loo_rms_pct:.3f}%")
print(f"  RMS V5 (référence)    : {rms5:.3f}%")
if loo_rms_pct < rms5:
    print(f"  ★★★ V6 RÉSISTE à la LOO : généralisation robuste !")
else:
    print(f"  ⚠ V6 surajuste : LOO {loo_rms_pct:.3f}% > V5 {rms5:.3f}%")

# ───────── Évaluation hors n=2 (n=1 et n=3) pour vérifier la non-régression ─────────
print(f"\n  ─── NON-RÉGRESSION HORS n=2 ───")
PDG_OTHER = [
    ("e⁻",       ["3_1"],              0.51099895),
    ("τ⁻",       ["4_1","5_1","6_1"],  1776.86),
    ("p",        ["3_1","7_2","7_2"],  938.27),
    ("n",        ["3_1","7_2","7_2"],  939.57),
    ("J/ψ",      ["3_1","4_1","6_3"],  3096.9),
    ("Υ(1S)",    ["6_2","7_1","7_3"],  9460.30),
    ("Υ(2S)",    ["6_2","6_2","8_2"],  10023.26),
    ("W±",       ["7_1","7_2","8_3"],  80369),
    ("Z⁰",       ["7_2","8_3","9_1"],  91188),
    ("H",        ["6_3","8_2","8_3"],  125250),
    ("t",        ["6_3","7_1","7_5"],  172690),
]
print(f"  {'particule':<10} {'n':>2} {'Δ V5 %':>10} {'Δ V6 %':>10} {'change':>8}")
print("  " + "-" * 46)
for nm, cfg, m_exp in PDG_OTHER:
    m_v5 = omega_v5(cfg)
    m_v6 = omega_v6_n2(cfg, gamma_opt, delta_opt)
    d5 = (m_exp - m_v5)/m_exp*100
    d6 = (m_exp - m_v6)/m_exp*100
    n = len(cfg)
    change = d6 - d5
    flag = "✓ inchangé" if abs(change) < 0.001 else "MODIFIÉ"
    print(f"  {nm:<10} {n:>2} {d5:>+9.2f}% {d6:>+9.2f}% {change:>+7.4f} {flag}")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Forme proposée :  Φ_2 → Φ_2 × (1 + {gamma_opt:+.4f}·asym + {delta_opt:+.4f}·log(c_tot/10))

  Résultats (n=2 propres) :
    RMS V5  : {rms5:.3f}%
    RMS V6  : {rms6:.3f}%
    RMS LOO : {loo_rms_pct:.3f}%
    Gain in-sample : {(1-rms6/rms5)*100:.1f}%

  La correction n'affecte QUE les configurations n=2.
  Les n=1 et n=3 (e⁻, τ⁻, p, n, J/ψ, Υ, W, Z, H, t) restent inchangés
  par construction.
""")
