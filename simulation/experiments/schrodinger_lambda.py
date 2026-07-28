r"""
SCHRÖDINGER EFFECTIVE sur λ_z — quantification de l'embedding.

Hypothèse : l'embedding de chaque composante est un état QUANTIQUE
stationnaire d'un Hamiltonien effectif sur λ_z.

  H(λ) = -α d²/dλ² + V(λ)
  V(λ) = Ω_ts(L; λ) + W(λ)   [W = pénalité élastique optionnelle]

  Ground state ψ_0(λ) → ⟨λ⟩ = embedding physique
  Énergie E_0 = masse prédite

CALIBRATION : α et W ajustés sur μ⁻ (ou m_e).
ÉVALUATION : test sur autres particules.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.linalg import eigh
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


def U_KT_deformed(knot_name, lam_z, n_points=4000):
    pts = knot_curve(knot_name, n_points)
    pts_def = pts.copy(); pts_def[:, 2] *= lam_z
    kappa, tau, speed = frenet_curvature_torsion(pts_def)
    ds = speed * 2*np.pi / n_points
    return np.sum(kappa**2 * ds), np.sum(tau**2 * ds)


def normalize_factors(knot_name):
    Uk, Ut = U_KT_deformed(knot_name, 1.0)
    return U_KT[knot_name][0]/Uk, U_KT[knot_name][1]/Ut

NORM = {K: normalize_factors(K) for K in KNOTS}


def U_KT_def_normalized(knot_name, lam_z):
    Uk, Ut = U_KT_deformed(knot_name, lam_z)
    nK, nT = NORM[knot_name]
    return Uk * nK, Ut * nT


def omega_def(comps, lam_z_deform_K, lam_z):
    """Ω_ts avec UNE composante (la première occurrence de lam_z_deform_K) déformée par lam_z."""
    n = len(comps)
    ws = [M_NAIVE[K]*(TOPO[K]["b"]*TOPO[K]["det"]/T_31)**P_EXP for K in comps]
    skel = float(sum(w**A for w in ws)**(1/A))
    G_sum = 0
    deformed = False
    for K in comps:
        if K == lam_z_deform_K and not deformed:
            Uk, Ut = U_KT_def_normalized(K, lam_z)
            deformed = True
        else:
            Uk, Ut = U_KT[K]
        G_sum += Uk * Ut
    G = G_sum / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(comps); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-B))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── SCHRÖDINGER 1D sur λ ─────────
def schrodinger_lambda(comps, deform_K, alpha, beta_elastic=0.0,
                       lam_range=(0.2, 4.0), n_grid=200):
    """Résout -α ψ'' + V(λ) ψ = E ψ sur λ ∈ lam_range.

    V(λ) = Ω_ts(L; λ) + beta_elastic · (λ - 1)²

    Retourne (lambdas, V_grid, E0, psi0_normalized, ⟨λ⟩, ⟨λ²⟩).
    """
    lambdas = np.linspace(lam_range[0], lam_range[1], n_grid)
    dlam = lambdas[1] - lambdas[0]

    # Construire V
    V = np.array([omega_def(comps, deform_K, lz) + beta_elastic*(lz-1)**2
                  for lz in lambdas])

    # Matrice cinétique (différences finies 2e ordre)
    T = np.zeros((n_grid, n_grid))
    coef = alpha / dlam**2
    for i in range(n_grid):
        T[i, i] = 2*coef
        if i > 0: T[i, i-1] = -coef
        if i < n_grid-1: T[i, i+1] = -coef

    # Hamiltonien
    H = T + np.diag(V)

    # Diagonaliser (état fondamental seulement)
    eigvals, eigvecs = eigh(H, subset_by_index=[0, 0])
    E0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    # Normaliser
    psi0 /= np.sqrt(np.sum(psi0**2) * dlam)
    # ⟨λ⟩ et ⟨λ²⟩
    lam_mean = np.sum(lambdas * psi0**2) * dlam
    lam2_mean = np.sum(lambdas**2 * psi0**2) * dlam
    lam_rms = np.sqrt(lam2_mean - lam_mean**2)

    return lambdas, V, E0, psi0, lam_mean, lam_rms


# ───────── ÉTAPE 1 : ÉTUDE DE V(λ) POUR LE MUON ─────────
print("\n" + "═"*82)
print("  ÉTUDE DE V(λ) = Ω_ts(3_1+6_1; λ_z(3_1)) POUR LE MUON")
print("═"*82)

# Sondage du potentiel
print(f"\n  Sondage de V(λ_z) pour μ⁻ = 3_1 + 6_1 :")
print(f"  {'λ_z':>6} {'V(λ) MeV':>12}")
print("  " + "-" * 22)
for lz in np.linspace(0.2, 3.0, 15):
    v = omega_def(["3_1","6_1"], "3_1", lz)
    flag = "← cible m_μ=105.66" if abs(v - 105.66) < 0.5 else ""
    print(f"  {lz:>6.3f} {v:>12.4f}  {flag}")

# ───────── ÉTAPE 2 : CALIBRATION de α sur μ⁻ ─────────
print("\n" + "═"*82)
print("  CALIBRATION de α pour que ⟨λ⟩(μ⁻) = 1.578 (cible μ⁻ exact)")
print("═"*82)

M_MU = 105.6584

# Test plusieurs α avec et sans β
print(f"\n  Test SANS pénalité élastique (β=0) :")
print(f"  {'α':>8} {'⟨λ⟩':>8} {'⟨Δλ⟩':>8} {'E_0 (V)':>10} {'m(⟨λ⟩)':>10}")
print("  " + "-" * 50)
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    lams, V, E0, psi, lam_m, lam_r = schrodinger_lambda(
        ["3_1","6_1"], "3_1", alpha, 0.0, (0.2, 4.0), 200)
    m_at_lam = omega_def(["3_1","6_1"], "3_1", lam_m)
    print(f"  {alpha:>8.2f} {lam_m:>8.3f} {lam_r:>8.3f} {E0:>10.3f} {m_at_lam:>10.3f}")

# Avec pénalité (force harmoniquement vers λ=1)
print(f"\n  Test AVEC pénalité élastique β·(λ-1)² :")
print(f"  {'α':>8} {'β':>8} {'⟨λ⟩':>8} {'E_0':>10} {'m(⟨λ⟩)':>10}")
print("  " + "-" * 56)
for alpha in [1.0, 10.0, 100.0]:
    for beta in [1.0, 10.0, 100.0]:
        lams, V, E0, psi, lam_m, lam_r = schrodinger_lambda(
            ["3_1","6_1"], "3_1", alpha, beta, (0.2, 4.0), 200)
        m_at_lam = omega_def(["3_1","6_1"], "3_1", lam_m)
        err = (m_at_lam - M_MU)/M_MU*100
        flag = "★" if abs(err) < 0.5 else ""
        print(f"  {alpha:>8.2f} {beta:>8.2f} {lam_m:>8.3f} {E0:>10.3f} "
              f"{m_at_lam:>10.3f} ({err:+.2f}%) {flag}")

# Optimisation : trouver (α, β) qui donnent ⟨λ⟩ correspondant à m_μ
print(f"\n  ─── OPTIMISATION de (α, β) pour m_μ exact ───")
from scipy.optimize import minimize as scipy_min
def chi2_mu(params):
    alpha, beta = params
    if alpha <= 0 or beta <= 0: return 1e10
    try:
        _,_,_,_, lam_m, _ = schrodinger_lambda(
            ["3_1","6_1"], "3_1", alpha, beta, (0.2, 4.0), 150)
        m_pred = omega_def(["3_1","6_1"], "3_1", lam_m)
        return ((m_pred - M_MU)/M_MU)**2
    except Exception:
        return 1e10

res = scipy_min(chi2_mu, [10.0, 10.0], method='Nelder-Mead',
                options={'xatol': 1e-3, 'fatol': 1e-9, 'maxiter': 200})
alpha_opt, beta_opt = res.x
print(f"  α_opt = {alpha_opt:.3f}, β_opt = {beta_opt:.3f}")
_,_,_,_, lam_mu, _ = schrodinger_lambda(["3_1","6_1"], "3_1", alpha_opt, beta_opt, (0.2, 4.0), 200)
m_mu_sch = omega_def(["3_1","6_1"], "3_1", lam_mu)
print(f"  ⟨λ⟩ pour μ⁻ : {lam_mu:.4f}  (cible : 1.578)")
print(f"  m_μ prédit : {m_mu_sch:.4f}  (cible : {M_MU})")

# ───────── ÉTAPE 3 : ÉVALUATION sur autres particules ─────────
print("\n" + "═"*82)
print("  ÉVALUATION : appliquer (α, β) optimisés à d'autres particules")
print("═"*82)

PDG_TEST = [
    ("μ⁻",        ["3_1","6_1"],         105.66, "3_1"),
    ("η",         ["3_1","3_1","6_1"],   547.86, "3_1"),
    ("π⁰",        ["3_1","3_1","5_2"],   134.98, "3_1"),
    ("φ",         ["3_1","5_1","5_2"],   1019.46, "5_1"),
    ("Λ",         ["5_1","8_1"],         1115.68, "8_1"),
    ("K*(892)",   ["6_1","6_2"],         891.66, "6_2"),
    ("D⁰",        ["3_1","5_1","7_2"],   1864.84, "5_1"),
    ("Σ⁺",        ["4_1","4_1","5_1"],   1189.37, "5_1"),
    ("p",         ["3_1","7_2","7_2"],   938.27, "3_1"),
    ("τ⁻",       ["4_1","5_1","6_1"],    1776.86, "6_1"),
]

print(f"\n  Avec α = {alpha_opt:.2f}, β = {beta_opt:.2f} :")
print(f"  {'particule':<10} {'config':<22} {'déforme':>8} {'⟨λ⟩':>7} {'m_V5':>9} "
      f"{'m_Sch':>9} {'m_exp':>9} {'Δ Sch':>7}")
print("  " + "-" * 84)
for nm, cfg, m_exp, deform_K in PDG_TEST:
    try:
        _,_,_,_, lam_m, _ = schrodinger_lambda(cfg, deform_K, alpha_opt, beta_opt,
                                               (0.2, 4.0), 150)
        m_sch = omega_def(cfg, deform_K, lam_m)
        m_v5 = omega_def(cfg, deform_K, 1.0)
        err_sch = (m_sch - m_exp)/m_exp*100
        flag = "✓" if abs(err_sch) < abs((m_v5 - m_exp)/m_exp*100) else "✗"
        print(f"  {nm:<10} {'+'.join(cfg):<22} {deform_K:>8} {lam_m:>7.3f} "
              f"{m_v5:>9.2f} {m_sch:>9.2f} {m_exp:>9.2f} {err_sch:>+6.2f}% {flag}")
    except Exception as e:
        print(f"  {nm:<10} erreur : {e}")

print("\n" + "═"*82)
print("  CONCLUSION")
print("═"*82)
print(f"""
  Le modèle Schrödinger sur λ_z avec V = Ω_ts + β(λ-1)² peut être
  CALIBRÉ pour reproduire μ⁻. Mais sa PRÉDICTIVITÉ sur les autres
  particules dépend de :
    • La cohérence des paramètres (α, β) à travers les particules
    • Le choix de la composante à déformer pour chaque entrelacs

  Si Δ_Sch < Δ_V5 pour la majorité : Schrödinger est PRÉDICTIF.
  Sinon : il faut un modèle quantique plus raffiné.
""")
