r"""
PRINCIPE VARIATIONNEL DE MÖBIUS APPROXIMÉ

Idée : pour un entrelacs L = {K_1, ..., K_n}, l'énergie de Möbius
totale combine self-énergies et termes croisés :

   E_M(L) = Σ_i E_self(K_i; déformations) + Σ_{i<j} E_cross(K_i, K_j; déformations)

L'équilibre est trouvé en minimisant E_M par rapport aux déformations
{s_i} de chaque composante.

Approximations :
   E_self(K_i; s_i)  ≈ U_κ(K_i; s_i)         (énergie de courbure)
   E_cross(K_i, K_j) ≈ -A · √(U_κU_τ_i · U_κU_τ_j)  (couplage attractif)

Total :
   E(s_1,...) = Σ_i U_κ(K_i; s_i) - A·Σ_{i<j} √(U_κU_τ_i·U_κU_τ_j)

Si A est UNIVERSEL (1 paramètre), c'est prédictif.
Si l'équilibre donne (s_1*, s_2*, ...), on calcule Ω_ts(L; s_1*, ...)
et on compare à m_exp.

Test :
  1. Calibrer A sur la μ⁻ pour que s_z* = 1.0436 (qui donne m_μ exact)
  2. Calculer (s_1*, ..., s_n*) pour 10+ particules
  3. Calculer m_pred et comparer à m_exp et m_V5

Si A unique → principe variationnel universel découvert.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import minimize, brentq, minimize_scalar
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, jones_determinant, braid_index
from theory.knot_catalog import _CURVE_REGISTRY

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
U_KT = {}; TOPO = {}; U_KAPPA = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    U_KAPPA[K] = float(np.sum(g["kappa"]**2*ds))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]
T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def frenet_deformed(K, s_z, n_points=20000):
    """U_κ, U_τ, U_κU_τ pour K avec scale_z = s_z."""
    curve_fn = _CURVE_REGISTRY[K]
    pts = curve_fn(n_points)
    pts = pts * np.array([1.0, 1.0, s_z])
    dt = 2*np.pi / n_points
    r1 = np.gradient(pts, dt, axis=0)
    r2 = np.gradient(r1, dt, axis=0)
    r3 = np.gradient(r2, dt, axis=0)
    speed = np.linalg.norm(r1, axis=1)
    cross = np.cross(r1, r2)
    kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
    triple = np.einsum('ij,ij->i', cross, r3)
    tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
    ds = speed * dt
    Uk = float(np.sum(kappa**2 * ds))
    Ut = float(np.sum(tau**2 * ds))
    return Uk, Ut


# Cache pour ne pas recalculer
_DEFORM_CACHE = {}
def deformed_invariants(K, s_z):
    key = (K, round(s_z, 5))
    if key not in _DEFORM_CACHE:
        _DEFORM_CACHE[key] = frenet_deformed(K, s_z)
    return _DEFORM_CACHE[key]


def omega_ts_link(components, deformations, return_uk_ut=False):
    """Ω_ts pour link avec déformations {s_i} par composante."""
    n = len(components)
    ws = []
    Us = []
    for K, s in zip(components, deformations):
        Uk_def, Ut_def = deformed_invariants(K, s)
        U_κτ_def = Uk_def * Ut_def
        Us.append(U_κτ_def)
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    m = Phi_n * Phi_p * skel * G**Q_EXP
    if return_uk_ut: return m, Us
    return m


def E_mobius(components, deformations, A_coupling):
    """Énergie de Möbius approximée :
       E_self = Σ U_κ(K_i; s_i)
       E_cross = -A · Σ_{i<j} √(U_κτ_i · U_κτ_j)
    """
    E_self = 0
    Us = []
    for K, s in zip(components, deformations):
        Uk_def, Ut_def = deformed_invariants(K, s)
        E_self += Uk_def  # énergie courbure
        Us.append(Uk_def * Ut_def)
    E_cross = 0
    n = len(Us)
    for i in range(n):
        for j in range(i+1, n):
            E_cross -= A_coupling * np.sqrt(Us[i] * Us[j])
    return E_self + E_cross


def find_equilibrium(components, A_coupling, s_init=None):
    """Trouve {s_i*} qui minimisent E_M."""
    n = len(components)
    if s_init is None:
        s_init = np.ones(n)
    res = minimize(
        lambda s: E_mobius(components, s, A_coupling),
        s_init,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 500}
    )
    return res.x, res.fun


# ───────── Test sur particules ─────────
PARTICULES = [
    ("μ⁻",  ["3_1","6_1"],          105.6584),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98),
    ("π±",  ["3_1","3_1","5_2"],    139.57),
    ("η",   ["3_1","3_1","6_1"],    547.86),
    ("φ",   ["3_1","5_1","5_2"],   1019.46),
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84),
    ("p",   ["3_1","7_2","7_2"],    938.27),
    ("n",   ["3_1","7_2","7_2"],    939.57),
]


print("\n" + "═"*82)
print("  CALIBRATION : trouver A_coupling tel que μ⁻ → s_z(6_1) ≈ 1.0436")
print("═"*82)

mu_components = ["3_1", "6_1"]
def err_mu(A_coupling):
    s_opt, _ = find_equilibrium(mu_components, A_coupling)
    m_pred = omega_ts_link(mu_components, s_opt)
    return m_pred - 105.6584

# Scan rapide
print(f"\n  Scan A : effet sur s* et m(μ⁻)")
print(f"  {'A':>9} {'s_3_1':>9} {'s_6_1':>9} {'m_pred':>10} {'Δ %':>9}")
for A_test in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    s_opt, _ = find_equilibrium(mu_components, A_test)
    m_p = omega_ts_link(mu_components, s_opt)
    err = (m_p - 105.6584)/105.6584*100
    print(f"  {A_test:>9.3f} {s_opt[0]:>9.4f} {s_opt[1]:>9.4f} {m_p:>10.4f} {err:>+8.3f}%")

# Calibration : trouver A_coupling exact via brentq
try:
    A_opt = brentq(err_mu, -1.0, 1.0, xtol=1e-5)
    print(f"\n  ★ Calibration A_opt = {A_opt:.5f}")
    s_mu, _ = find_equilibrium(mu_components, A_opt)
    print(f"     s_3_1 = {s_mu[0]:.4f}, s_6_1 = {s_mu[1]:.4f}")
    print(f"     m(μ⁻) = {omega_ts_link(mu_components, s_mu):.4f} (cible 105.6584)")
except Exception as e:
    print(f"  ! Calibration échouée : {e}")
    A_opt = 0.0


# ───────── Test sur toutes les particules ─────────
print("\n" + "═"*82)
print(f"  TEST avec A = {A_opt:.5f} : prédictions pour 10 particules")
print("═"*82)
print(f"  {'Particule':<8} {'config':<22} {'s_i optimaux':<30} {'m_pred':>10} "
      f"{'m_exp':>10} {'Δ V5':>7} {'Δ Möb':>7}")
print("  " + "-" * 100)

for nm, cfg, m_exp in PARTICULES:
    # m_V5 (sans déformation)
    m_v5 = omega_ts_link(cfg, [1.0]*len(cfg))
    d_v5 = (m_v5 - m_exp)/m_exp*100
    # Équilibre Möbius
    s_opt, E_opt = find_equilibrium(cfg, A_opt)
    m_mob = omega_ts_link(cfg, s_opt)
    d_mob = (m_mob - m_exp)/m_exp*100
    s_str = ",".join(f"{s:.3f}" for s in s_opt)
    print(f"  {nm:<8} {'+'.join(cfg):<22} {s_str:<30} {m_mob:>10.3f} "
          f"{m_exp:>10.3f} {d_v5:>+6.2f}% {d_mob:>+6.2f}%")

# Calcul RMS global
rms_v5 = 0; rms_mob = 0
for nm, cfg, m_exp in PARTICULES:
    m_v5 = omega_ts_link(cfg, [1.0]*len(cfg))
    s_opt, _ = find_equilibrium(cfg, A_opt)
    m_mob = omega_ts_link(cfg, s_opt)
    rms_v5 += ((m_v5-m_exp)/m_exp)**2
    rms_mob += ((m_mob-m_exp)/m_exp)**2
rms_v5 = float(np.sqrt(rms_v5/len(PARTICULES))*100)
rms_mob = float(np.sqrt(rms_mob/len(PARTICULES))*100)

print(f"\n  RMS global (10 particules) :")
print(f"    V5 (sans déformation)  : {rms_v5:.3f}%")
print(f"    Möbius (A = {A_opt:.4f}) : {rms_mob:.3f}%")
print(f"    Gain                    : {(rms_v5-rms_mob)/rms_v5*100:+.1f}%")

# ───────── Diagnostic ─────────
print("\n" + "═"*82)
print("  DIAGNOSTIC")
print("═"*82)
print(f"""
  Le principe de Möbius approximé :
    E(s_1,...,s_n) = Σ U_κ(K_i; s_i)  -  A · Σ_{{i<j}} √(U_κU_τ_i · U_κU_τ_j)

  donne un équilibre {{s_i*}} et un m_pred.

  Si gain RMS > 50% : principe utile (réduit le résidu structurel)
  Si gain RMS < 10% : principe non productif

  Résultat : RMS V5 = {rms_v5:.3f}%, RMS Möbius = {rms_mob:.3f}%

  Le terme de couplage cross ~ √(U_κU_τ_i U_κU_τ_j) est précisément
  celui qui avait rejeté l'hypothèse "lepton-spécifique H2" car il
  s'applique aussi aux hadrons. Ici, contrairement à H2, il est
  ÉQUILIBRÉ par un terme d'auto-énergie U_κ(K_i; s_i) qui pénalise
  les grandes déformations. La compétition pourrait restaurer la
  consistence sur tout le spectre.
""")
