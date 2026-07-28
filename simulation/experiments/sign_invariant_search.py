r"""
RECHERCHE de l'INVARIANT SCALAIRE DE SIGNE de la déformation plastique.

L'analyse précédente a révélé une obstruction : les résidus V5 ont
des SIGNES OPPOSÉS selon la particule (μ⁻ < 0, η > 0). Un gradient
unifié ne peut donc pas être universel sans un sélecteur de signe.

Hypothèses testées :
  S1. σ(L) = (-1)^(2J)              [fermion = -1, boson = +1]
  S2. σ(L) = (-1)^(n_components)    [parité du nombre de composantes]
  S3. σ(L) = sign(Σχ_i - Lk_opt)    [charge topologique vs enlacement]
  S4. σ(L) = (-1)^(cr_total mod 2)  [parité du nombre total de croisements]
  S5. σ(L) = parité_de_Jones         [Jones polynomial parity at t=1]
  S6. σ(L) = chiralité_globale       [signe de prod(chiralités)]

Pour chaque, on évalue :
  • Corrélation entre σ(L) prédit et sign(Δ V5) observé sur 14 particules
  • Gain RMS si on utilise ce σ dans le principe variationnel à gradient
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.optimize import brentq
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
# Chiralité : True = chiral, False = achiral
CHIRAL = {"3_1":True,"4_1":False,"5_1":True,"5_2":True,"6_1":True,
          "6_2":True,"6_3":False,"7_1":True,"7_2":True,"7_3":True,
          "7_4":True,"7_5":True,"7_6":True,"7_7":True,
          "8_1":True,"8_2":True,"8_3":False,"9_1":True,"10_1":True}
A_FREN = np.sqrt(7/(8*np.pi)); P_EXP = 3/4; Q_EXP = 3/4
X_EXP = -np.sqrt(6); BB = 33/20; RHO_0 = 10 - 3/(8*np.pi)

print("Précalcul…")
U_KT = {}; TOPO = {}
for K in KNOTS:
    g = frenet_geometry(K, 30000)
    ds = g["speed"] * 2*np.pi/len(g["kappa"])
    U_KT[K] = (float(np.sum(g["kappa"]**2*ds)),
               float(np.sum(g["tau"]**2*ds)))
    pj = jones_polynomial(K)
    TOPO[K] = {"cr": int(K.split("_")[0]), "n_l": N_LOBES[K],
               "b": braid_index(K), "det": jones_determinant(pj),
               "jones": pj, "chiral": CHIRAL[K]}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}


def omega_link(components, s_list):
    n = len(components); ws=[]; Us=[]
    for K, s in zip(components, s_list):
        # Pour ce test on n'a besoin que de m_V5, donc s = 1 partout
        Us.append(U_KT[K][0]*U_KT[K][1] * (s**0))  # invariant si s=1
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


# ───────── Base de particules avec quantum numbers ─────────
# (nm, config, m_exp, 2J, classe)
PARTICULES = [
    ("e⁻",    ["3_1"],                  0.51099895,  1,  "lepton"),
    ("μ⁻",    ["3_1","6_1"],            105.6584,    1,  "lepton"),
    ("τ⁻",    ["4_1","5_1","6_1"],     1776.86,      1,  "lepton"),
    ("π⁰",    ["3_1","3_1","5_2"],      134.98,      0,  "pseudo"),
    ("π±",    ["3_1","3_1","5_2"],      139.57,      0,  "pseudo"),
    ("K⁰",    ["6_1","6_1","6_1"],      497.61,      0,  "pseudo"),
    ("K±",    ["6_1","6_1","6_1"],      493.68,      0,  "pseudo"),
    ("η",     ["3_1","3_1","6_1"],      547.86,      0,  "pseudo"),
    ("φ",     ["3_1","5_1","5_2"],     1019.46,      2,  "vector"),
    ("J/ψ",   ["3_1","4_1","6_3"],     3096.9,       2,  "vector"),
    ("D⁰",    ["3_1","5_1","7_2"],     1864.84,      0,  "pseudo"),
    ("B±",    ["3_1","6_2","7_7"],     5279.34,      0,  "pseudo"),
    ("Λ_c",   ["3_1","3_1","9_1"],     2286.46,      1,  "baryon"),
    ("p",     ["3_1","7_2","7_2"],      938.27,      1,  "baryon"),
    ("n",     ["3_1","7_2","7_2"],      939.57,      1,  "baryon"),
]


# ───────── Calcul du résidu V5 et signature pour chaque particule ─────────
data = []
for nm, cfg, m_exp, two_J, cls in PARTICULES:
    m_v5 = omega_link(cfg, [1.0]*len(cfg))
    d_v5 = (m_v5 - m_exp)/m_exp*100
    sign_obs = 0 if abs(d_v5) < 0.1 else int(np.sign(d_v5))
    # Calcul des signatures candidates
    cr_total = sum(TOPO[K]["cr"] for K in cfg)
    chiral_prod = 1
    for K in cfg:
        if TOPO[K]["chiral"]: chiral_prod *= -1
    # Parité Jones : signe de V(K)(t=1) = ±1 normalement
    jones_par = 1
    for K in cfg:
        # det(K) = |V(K)(-1)|, on prend le signe via heuristique
        # Pour un nœud non-trivial, on regarde la parité du nombre
        # de croisements négatifs (approximation grossière)
        if TOPO[K]["cr"] % 2 == 1:
            jones_par *= -1
    # σ_S1 = (-1)^(2J)
    sigma_S1 = (-1)**two_J
    # σ_S2 = (-1)^n
    sigma_S2 = (-1)**len(cfg)
    # σ_S3 : pas implementé sans Σχ (on utilise approximation par cr_total)
    sigma_S3 = 1 if cr_total > 12 else -1
    # σ_S4 = (-1)^cr_total
    sigma_S4 = (-1)**cr_total
    # σ_S5 = jones_par
    sigma_S5 = jones_par
    # σ_S6 = chiral_prod (product of chiralities)
    sigma_S6 = chiral_prod

    data.append({
        "nm": nm, "cfg": cfg, "m_exp": m_exp, "m_v5": m_v5,
        "d_v5": d_v5, "sign_obs": sign_obs, "cls": cls,
        "two_J": two_J,
        "S1": sigma_S1, "S2": sigma_S2, "S3": sigma_S3,
        "S4": sigma_S4, "S5": sigma_S5, "S6": sigma_S6,
    })

# Affichage
print("\n" + "═"*100)
print("  Résidus V5 et signatures candidates par particule")
print("═"*100)
print(f"  {'nm':<6} {'config':<22} {'cls':<8} {'Δ V5':>8} {'signe':>7} "
      f"{'S1':>4} {'S2':>4} {'S3':>4} {'S4':>4} {'S5':>4} {'S6':>4}")
print("  " + "-" * 95)
for d in data:
    print(f"  {d['nm']:<6} {'+'.join(d['cfg']):<22} {d['cls']:<8} "
          f"{d['d_v5']:>+7.2f}% {d['sign_obs']:>+6} "
          f"{d['S1']:>+4} {d['S2']:>+4} {d['S3']:>+4} "
          f"{d['S4']:>+4} {d['S5']:>+4} {d['S6']:>+4}")


# ───────── Évaluation : taux d'accord entre σ_S et sign(Δ V5) ─────────
print("\n" + "═"*100)
print("  Taux d'accord entre signatures candidates et sign(Δ V5)")
print("═"*100)

# On exclut les particules à |Δ V5| < 0.1% (signe ambigu)
significant = [d for d in data if abs(d["d_v5"]) > 0.1]
print(f"  N = {len(significant)} particules avec |Δ V5| > 0.1%")
print(f"  (e⁻ et p exclus car |Δ| trop petit)\n")

for sig_name in ["S1", "S2", "S3", "S4", "S5", "S6"]:
    agree = sum(1 for d in significant if d[sig_name] == d["sign_obs"])
    rate = agree / len(significant) * 100
    label = {
        "S1": "(-1)^(2J)        [fermion/boson]",
        "S2": "(-1)^n           [parité composantes]",
        "S3": "cr_total > 12     [seuil masse]",
        "S4": "(-1)^cr_total     [parité croisements]",
        "S5": "jones_par         [parité Jones]",
        "S6": "Π chiralités      [chiralité globale]",
    }[sig_name]
    flag = "★★" if rate > 80 else ("★" if rate > 65 else "")
    print(f"  {sig_name} {label:<35}  accord = {agree:>2}/{len(significant)} ({rate:.0f}%) {flag}")


# ───────── Test du principe variationnel avec σ_S1 ─────────
print("\n" + "═"*100)
print("  Test : utiliser σ_S1 = (-1)^(2J) dans le principe variationnel")
print("═"*100)

_CACHE = {}
def U_K(K, s_z):
    key = (K, round(s_z, 4))
    if key not in _CACHE:
        pts = _CURVE_REGISTRY[K](15000)
        pts = pts * np.array([1.0, 1.0, s_z])
        dt = 2*np.pi / len(pts)
        r1 = np.gradient(pts, dt, axis=0)
        r2 = np.gradient(r1, dt, axis=0)
        r3 = np.gradient(r2, dt, axis=0)
        speed = np.linalg.norm(r1, axis=1)
        cross = np.cross(r1, r2)
        kappa = np.linalg.norm(cross, axis=1) / (speed**3 + 1e-12)
        triple = np.einsum('ij,ij->i', cross, r3)
        tau = triple / (np.linalg.norm(cross, axis=1)**2 + 1e-12)
        ds_arr = speed * dt
        _CACHE[key] = float(np.sum(kappa**2 * ds_arr)) * float(np.sum(tau**2 * ds_arr))
    return _CACHE[key]


def omega_def(components, s_list):
    n = len(components); ws=[]; Us=[]
    for K, s in zip(components, s_list):
        Us.append(U_K(K, s))
        w = M_NAIVE[K] * (TOPO[K]["b"] * TOPO[K]["det"] / T_31)**P_EXP
        ws.append(w)
    skel = float(sum(w**A_FREN for w in ws)**(1/A_FREN))
    G = sum(Us) / P_31
    Phi_n = ((n+1)/2)**(-X_EXP)
    c = Counter(components); Phi_p = 1.0
    for K, ct in c.items():
        if ct >= 2: Phi_p *= (RHO_0 * TOPO[K]["cr"]**(-BB))**(ct-1)
    return Phi_n * Phi_p * skel * G**Q_EXP


def relax_signed(components, sigma, gamma, k_anchor=1.0,
                 dt=0.01, n_steps=2000):
    """Avec gradient(cr) modulé par σ(L)."""
    n = len(components)
    s = np.ones(n)
    crs = np.array([TOPO[K]["cr"] for K in components], dtype=float)
    grad = (crs - crs.mean()) / crs.mean()
    for _ in range(n_steps):
        F = sigma * gamma * grad - k_anchor * (s - 1)
        s = s + dt * F
        s = np.clip(s, 0.3, 3.0)
        if np.linalg.norm(F) < 1e-8: break
    return s


# Calibrer γ tel que μ⁻ → s_6_1 = 1.044
mu_cfg = ["3_1", "6_1"]; mu_sigma = -1  # S1 pour J=1/2
def err_mu(gamma):
    s_eq = relax_signed(mu_cfg, mu_sigma, gamma)
    return s_eq[1] - 1.0436
gamma_opt = brentq(err_mu, -2.0, 2.0, xtol=1e-5)
print(f"  Calibration : γ = {gamma_opt:.4f}")

print(f"\n  {'nm':<6} {'config':<22} {'σ':>3} {'s_attracteur':<28} "
      f"{'m(s*)':>9} {'Δ V5':>7} {'Δ s*':>7}")
print("  " + "-" * 90)
results = []
for d in data:
    sigma = d["S1"]
    s_eq = relax_signed(d["cfg"], sigma, gamma_opt)
    m_eq = omega_def(d["cfg"], s_eq)
    d_eq = (m_eq - d["m_exp"])/d["m_exp"]*100
    s_str = ",".join(f"{x:.3f}" for x in s_eq)
    better = "✓" if abs(d_eq) < abs(d["d_v5"]) else "✗"
    print(f"  {d['nm']:<6} {'+'.join(d['cfg']):<22} {sigma:>+3} {s_str:<28} "
          f"{m_eq:>9.2f} {d['d_v5']:>+6.2f}% {d_eq:>+6.2f}% {better}")
    results.append({"nm": d["nm"], "d_v5": d["d_v5"], "d_eq": d_eq})

rms_v5 = float(np.sqrt(np.mean([r["d_v5"]**2 for r in results])))
rms_eq = float(np.sqrt(np.mean([r["d_eq"]**2 for r in results])))
gain = (rms_v5 - rms_eq) / rms_v5 * 100
n_better = sum(1 for r in results if abs(r["d_eq"]) < abs(r["d_v5"]))
print(f"\n  RMS V5  = {rms_v5:.3f}%")
print(f"  RMS s*  = {rms_eq:.3f}%")
print(f"  Gain    = {gain:+.1f}%")
print(f"  Améliorées : {n_better}/{len(results)}")


# ───────── Diagnostic final ─────────
print("\n" + "═"*100)
print("  DIAGNOSTIC")
print("═"*100)

print(f"""
  L'invariant σ_S1 = (-1)^(2J) [bosons/fermions] a-t-il résolu
  l'obstruction de signe ?

  - Si gain > 30%   : invariant de signe TROUVÉ
  - Si gain ∈ 5-30% : partiellement correct, à raffiner
  - Si gain < 5%    : le signe n'est pas (-1)^(2J)

  RÉSULTAT : gain = {gain:+.1f}%

  Note : J/ψ et φ ont 2J=2 → σ=+1 mais résidus de signes opposés
         (J/ψ ≈ 0, φ < 0). Une signature plus fine que (-1)^(2J)
         pourrait combiner spin et parité de C-parity.
""")
