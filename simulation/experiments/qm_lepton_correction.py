r"""
TEST de l'effet QM LEPTON-SPÉCIFIQUE.

Hypothèses testées :

(H1) Correction radiative de type Sommerfeld :
     δm = (α_EM / π) · m_V5 · log(M_scale / m_V5)
     où M_scale est une échelle effective dépendant de la structure.

(H2) Self-énergie liée au COUPLAGE INTERNE entre composantes :
     δm = c_lep · Σ_{i<j} √(c_J(K_i) · c_J(K_j))

(H3) Correction proportionnelle à n_components-1 (générations) :
     δm = c_lep · m_V5 · (n - 1)

(H4) Loi puissance sur masse "naive" des composantes hors-3_1 :
     δm = c_lep · (Σ M_naive de non-3_1)^β

Pour chaque hypothèse :
  - Calibrer c_lep sur le SOUS-ENSEMBLE des leptons (e, μ, τ)
  - Évaluer sur les HADRONS contenant 3_1
  - Mesurer si la correction marche SEULEMENT pour leptons (= spécifique)
    ou aussi pour hadrons (= correction universelle)
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar, minimize
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index, topological_spin

M_E = 0.51099895; F_KAPPA = 0.8682
M_C = M_E*F_KAPPA/3; M_L = M_E*(1-F_KAPPA)/3
KNOTS = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","7_3",
         "7_4","7_5","7_6","7_7","8_1","8_2","8_3","9_1","10_1"]
N_LOBES = {"3_1":3,"4_1":2,"5_1":5,"5_2":2,"6_1":2,"6_2":3,"6_3":3,
           "7_1":7,"7_2":2,"7_3":3,"7_4":2,"7_5":3,"7_6":2,"7_7":3,
           "8_1":2,"8_2":3,"8_3":3,"9_1":9,"10_1":2}
A = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); B = 33/20; RHO_0 = 10 - 3/(8*np.pi)
ALPHA_EM = 1/137.036

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


def omega(comps):
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G = sum(U_KT[K][0]*U_KT[K][1] for K in comps)/P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n*Phi_p*skel*G**Q_EXP


# ───────── PDG ─────────
# Leptons (sur lesquels on calibre) + hadrons contenant 3_1 (test prédictif)
LEPTONS = [
    ("e⁻",  ["3_1"],               0.51099895),
    ("μ⁻",  ["3_1","6_1"],         105.66),
    ("τ⁻", ["4_1","5_1","6_1"],   1776.86),  # τ ne contient PAS 3_1
]

# Hadrons contenant 3_1 (validation hors-échantillon de l'hypothèse "lepton-spécifique")
HADRONS_3_1 = [
    ("π⁰",        ["3_1","3_1","5_2"],  134.98),
    ("π±",        ["3_1","3_1","5_2"],  139.57),
    ("η",         ["3_1","3_1","6_1"],  547.86),
    ("φ",         ["3_1","5_1","5_2"],  1019.46),
    ("J/ψ",       ["3_1","4_1","6_3"],  3096.9),
    ("D⁰",        ["3_1","5_1","7_2"],  1864.84),
    ("D±",        ["3_1","5_1","7_2"],  1869.66),
    ("B±",        ["3_1","6_2","7_7"],  5279.34),
    ("B⁰",        ["3_1","6_2","7_7"],  5279.66),
    ("p",         ["3_1","7_2","7_2"],  938.27),
    ("n",         ["3_1","7_2","7_2"],  939.57),
    ("Λ_c",       ["3_1","3_1","9_1"],  2286.46),
]

print("\n" + "═"*82)
print("  TEST QM LEPTON-SPÉCIFIQUE — hypothèses (H1)-(H4)")
print("═"*82)

# ───────── Hypothèses ─────────
def H1_radiative(comps, c_lep):
    """δm = (α/π) · m_V5 · log(c_lep / m_V5) si c_lep > m_V5, sinon 0."""
    m_v5 = omega(comps)
    if c_lep <= m_v5: return 0.0
    return (ALPHA_EM / np.pi) * m_v5 * np.log(c_lep / m_v5)


def H2_self_coupling(comps, c_lep):
    """δm = c_lep · Σ_{i<j} √(c_J(K_i) · c_J(K_j))."""
    if len(comps) < 2: return 0.0
    cs = [TOPO[K]["det"] for K in comps]
    s = 0.0
    for i in range(len(comps)):
        for j in range(i+1, len(comps)):
            s += np.sqrt(cs[i] * cs[j])
    return c_lep * s


def H3_generation(comps, c_lep):
    """δm = c_lep · m_V5 · (n - 1)."""
    n = len(comps)
    m_v5 = omega(comps)
    return c_lep * m_v5 * (n - 1)


def H4_other_mass(comps, c_lep_pair):
    """δm = c_lep · (Σ M_naive de non-3_1)^β."""
    c_lep, beta = c_lep_pair
    other_sum = sum(M_NAIVE[K] for K in comps if K != "3_1")
    if other_sum <= 0: return 0.0
    return c_lep * other_sum**beta


# ───────── Calibration sur LEPTONS ─────────
def calibrate(hyp, n_params=1):
    """Calibrer hyp sur leptons (e, μ, τ)."""
    def chi2(params):
        if n_params == 1:
            p = params[0] if hasattr(params, '__len__') else params
        else:
            p = params
        total = 0.0
        for nm, cfg, m_exp in LEPTONS:
            m_v5 = omega(cfg)
            delta = hyp(cfg, p)
            m_pred = m_v5 + delta
            if m_pred <= 0: return 1e10
            total += np.log(m_exp/m_pred)**2
        return total/len(LEPTONS)
    if n_params == 1:
        res = minimize_scalar(lambda x: chi2([x]), bounds=(-100, 1e10),
                              method='bounded', options={'xatol': 1e-6})
        return res.x, chi2([res.x])
    else:
        res = minimize(chi2, [1.0]*n_params, method='Nelder-Mead',
                       options={'fatol': 1e-12, 'maxiter': 1000})
        return res.x, res.fun


# ───────── Tester chaque hypothèse ─────────
results = {}
print("\n  ─── (H1) Correction radiative α/π ─── ")
c_H1, chi_H1 = calibrate(H1_radiative)
print(f"  c_lep_opt = {c_H1:.3e}  (échelle UV effective)")
print(f"  χ² leptons = {chi_H1:.6e}")
results["H1"] = (c_H1, H1_radiative)

print("\n  ─── (H2) Self-coupling Σ √(c_J·c_J) ─── ")
c_H2, chi_H2 = calibrate(H2_self_coupling)
print(f"  c_lep_opt = {c_H2:.4e}")
print(f"  χ² leptons = {chi_H2:.6e}")
results["H2"] = (c_H2, H2_self_coupling)

print("\n  ─── (H3) Génération (n-1) ─── ")
c_H3, chi_H3 = calibrate(H3_generation)
print(f"  c_lep_opt = {c_H3:.4e}")
print(f"  χ² leptons = {chi_H3:.6e}")
results["H3"] = (c_H3, H3_generation)

print("\n  ─── (H4) Loi puissance M_naive_other^β ─── ")
c_H4, chi_H4 = calibrate(H4_other_mass, n_params=2)
print(f"  c_lep_opt = {c_H4[0]:.4e}, β = {c_H4[1]:.4f}")
print(f"  χ² leptons = {chi_H4:.6e}")
results["H4"] = (c_H4, H4_other_mass)

# ───────── Évaluation détaillée ─────────
print("\n" + "═"*82)
print("  ÉVALUATION : leptons (cible) + hadrons avec 3_1 (test prédictif)")
print("═"*82)

for label, (c_opt, hyp) in results.items():
    print(f"\n  ─── {label} ───")
    print(f"  {'particule':<10} {'config':<20} {'m_V5':>10} {'δm':>10} "
          f"{'m_pred':>10} {'m_exp':>10} {'Δ V5':>7} {'Δ V6':>7}")
    print("  " + "-" * 92)
    for nm, cfg, m_exp in LEPTONS:
        m_v5 = omega(cfg)
        d = hyp(cfg, c_opt)
        m_pred = m_v5 + d
        d_v5 = (m_v5 - m_exp)/m_exp*100
        d_v6 = (m_pred - m_exp)/m_exp*100
        print(f"  L {nm:<8} {'+'.join(cfg):<20} {m_v5:>10.3f} {d:>10.3f} "
              f"{m_pred:>10.3f} {m_exp:>10.3f} {d_v5:>+6.2f}% {d_v6:>+6.2f}%")
    print()
    rms_h5 = 0; rms_h6 = 0
    for nm, cfg, m_exp in HADRONS_3_1:
        m_v5 = omega(cfg)
        d = hyp(cfg, c_opt)
        m_pred = m_v5 + d
        d_v5 = (m_v5 - m_exp)/m_exp*100
        d_v6 = (m_pred - m_exp)/m_exp*100
        rms_h5 += d_v5**2; rms_h6 += d_v6**2
        flag = "✓" if abs(d_v6) < abs(d_v5) else "✗"
        print(f"  H {nm:<8} {'+'.join(cfg):<20} {m_v5:>10.3f} {d:>10.3f} "
              f"{m_pred:>10.3f} {m_exp:>10.3f} {d_v5:>+6.2f}% {d_v6:>+6.2f}% {flag}")
    rms_h5 = np.sqrt(rms_h5/len(HADRONS_3_1))
    rms_h6 = np.sqrt(rms_h6/len(HADRONS_3_1))
    print(f"\n    RMS hadrons V5 : {rms_h5:.3f}%")
    print(f"    RMS hadrons V6 : {rms_h6:.3f}%")
    if rms_h6 < rms_h5:
        print(f"    → V6 améliore les hadrons aussi → correction UNIVERSELLE, pas spécifiquement lepton")
    else:
        print(f"    → V6 détériore les hadrons → effet plausiblement spécifique aux leptons (mais détériore tout)")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Question : existe-t-il une correction QM qui :
    • résout μ⁻ (+2.27%) tout en gardant e⁻ et τ⁻ exacts
    • ne détériore pas les hadrons contenant 3_1

  Si OUI → effet QM lepton-spécifique confirmé
  Si NON → l'hypothèse 3 (effet lepton-spécifique) est rejetée

  Verdict : voir l'analyse détaillée ci-dessus.
""")
