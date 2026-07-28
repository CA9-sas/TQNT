r"""
RECHERCHE DE LA FONCTION D'AMPLITUDE γ(L).

Étape précédente : invariant de signe identifié à 100%
  σ(L) = +1 si pseudoscalaire (J=0), -1 sinon

Étape actuelle : calculer γ(L) requis par particule pour reproduire
m_exp, puis chercher le pattern dans γ(L) vs invariants.

Méthode :
  1. Pour chaque particule, à l'équilibre de la relaxation plastique :
       s_i = 1 + (σ·γ/k) · gradient_i
     où gradient_i = (cr_i - ⟨cr⟩) / ⟨cr⟩, k = 1, σ fixé par S7.

  2. Résoudre numériquement pour γ tel que m_pred(s*(γ)) = m_exp.

  3. Tabuler γ(L) pour chaque particule.

  4. Chercher corrélations avec :
       • m_exp (énergie caractéristique)
       • log(m_exp)
       • Σ U_κU_τ (énergie de courbure-torsion)
       • Σ M_naive (masse naïve totale)
       • n (nombre de composantes)
       • cr_max - cr_min (asymétrie)
       • Variance(cr)
       • |Δ V5| (résidu lui-même, sanity check)
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
               "b": braid_index(K), "det": jones_determinant(pj)}
P_31 = U_KT["3_1"][0]*U_KT["3_1"][1]; T_31 = TOPO["3_1"]["b"]*TOPO["3_1"]["det"]
M_NAIVE = {K: TOPO[K]["cr"]*M_C + TOPO[K]["n_l"]*M_L for K in KNOTS}

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


PARTICULES = [
    # nm, cfg, m_exp, 2J, classe (pseudo=>σ=+1, autre=>σ=-1)
    ("μ⁻",  ["3_1","6_1"],          105.6584,    1, "lepton"),
    ("τ⁻",  ["4_1","5_1","6_1"],   1776.86,      1, "lepton"),
    ("π⁰",  ["3_1","3_1","5_2"],    134.98,      0, "pseudo"),
    ("π±",  ["3_1","3_1","5_2"],    139.57,      0, "pseudo"),
    ("K⁰",  ["6_1","6_1","6_1"],    497.61,      0, "pseudo"),
    ("K±",  ["6_1","6_1","6_1"],    493.68,      0, "pseudo"),
    ("η",   ["3_1","3_1","6_1"],    547.86,      0, "pseudo"),
    ("φ",   ["3_1","5_1","5_2"],   1019.46,      2, "vector"),
    ("J/ψ", ["3_1","4_1","6_3"],   3096.9,       2, "vector"),
    ("D⁰",  ["3_1","5_1","7_2"],   1864.84,      0, "pseudo"),
    ("B±",  ["3_1","6_2","7_7"],   5279.34,      0, "pseudo"),
    ("Λ_c", ["3_1","3_1","9_1"],   2286.46,      1, "baryon"),
    ("p",   ["3_1","7_2","7_2"],    938.27,      1, "baryon"),
    ("n",   ["3_1","7_2","7_2"],    939.57,      1, "baryon"),
]


def sigma_S7(two_J):
    return +1 if two_J == 0 else -1


def gradient_cr(components):
    crs = np.array([TOPO[K]["cr"] for K in components], dtype=float)
    return (crs - crs.mean()) / crs.mean()


def m_pred_with_gamma(components, sigma, gamma, k_anchor=1.0):
    """s_i équilibre = 1 + (σ·γ/k) · grad_i ; calcule m_pred."""
    grad = gradient_cr(components)
    s_eq = 1 + (sigma * gamma / k_anchor) * grad
    s_eq = np.clip(s_eq, 0.3, 3.0)
    return omega_def(components, s_eq), s_eq


def find_gamma_required(components, sigma, m_exp, k_anchor=1.0):
    """Trouve γ tel que m_pred(s*(γ)) = m_exp."""
    def err(gamma):
        m_p, _ = m_pred_with_gamma(components, sigma, gamma, k_anchor)
        return m_p - m_exp
    try:
        return brentq(err, -5.0, 5.0, xtol=1e-5)
    except ValueError:
        return None


# ───────── Extraction de γ par particule ─────────
print("\n" + "═"*100)
print("  Extraction de γ(L) requis par particule")
print("═"*100)
print(f"  {'nm':<6} {'config':<22} {'σ':>3} {'γ requis':>10} "
      f"{'s_attracteur':<28} {'m(s*)':>9} {'Δ%':>7}")
print("  " + "-" * 95)

extracted = []
for nm, cfg, m_exp, two_J, cls in PARTICULES:
    sigma = sigma_S7(two_J)
    gamma = find_gamma_required(cfg, sigma, m_exp)
    if gamma is None:
        print(f"  {nm:<6} {'+'.join(cfg):<22} {sigma:>+3} {'no_sol':>10}")
        continue
    m_p, s_eq = m_pred_with_gamma(cfg, sigma, gamma)
    err = (m_p - m_exp)/m_exp*100
    s_str = ",".join(f"{x:.3f}" for x in s_eq)
    flag = "  ★" if abs(gamma) < 0.5 else ""
    print(f"  {nm:<6} {'+'.join(cfg):<22} {sigma:>+3} {gamma:>+10.5f} "
          f"{s_str:<28} {m_p:>9.2f} {err:>+6.2f}%{flag}")
    extracted.append({
        "nm": nm, "cfg": cfg, "m_exp": m_exp,
        "two_J": two_J, "cls": cls, "sigma": sigma,
        "gamma": gamma, "s_eq": s_eq,
        "log_m": np.log10(m_exp),
        "n_comp": len(cfg),
        "sum_U": sum(U_KT[K][0]*U_KT[K][1] for K in cfg),
        "sum_M_naive": sum(M_NAIVE[K] for K in cfg),
        "max_cr": max(TOPO[K]["cr"] for K in cfg),
        "var_cr": np.var([TOPO[K]["cr"] for K in cfg]),
        "asym_cr": (max(TOPO[K]["cr"] for K in cfg)
                    - min(TOPO[K]["cr"] for K in cfg)),
    })


# ───────── Recherche de pattern ─────────
print("\n" + "═"*100)
print("  Corrélations Pearson γ(L) vs invariants")
print("═"*100)

# Excluons les configurations symétriques (gradient=0, γ indéterminé)
def has_nonzero_grad(cfg):
    grad = gradient_cr(cfg)
    return np.std(grad) > 0.01


usable = [d for d in extracted if has_nonzero_grad(d["cfg"]) and d["gamma"] is not None]
print(f"\n  N = {len(usable)} particules avec gradient non-trivial")
gammas = np.array([d["gamma"] for d in usable])
print(f"  γ range : [{gammas.min():+.4f}, {gammas.max():+.4f}]")
print(f"  |γ| moyenne : {np.mean(np.abs(gammas)):.4f}")
print(f"  ⟨γ⟩ = {gammas.mean():+.4f}, σ(γ) = {gammas.std():.4f}")

print(f"\n  Corrélations |γ| vs invariants :")
abs_gamma = np.abs(gammas)
features = {
    "m_exp": np.array([d["m_exp"] for d in usable]),
    "log(m_exp)": np.array([d["log_m"] for d in usable]),
    "1/m_exp": np.array([1/d["m_exp"] for d in usable]),
    "Σ U_κU_τ": np.array([d["sum_U"] for d in usable]),
    "log(Σ U)": np.array([np.log10(d["sum_U"]) for d in usable]),
    "Σ M_naive": np.array([d["sum_M_naive"] for d in usable]),
    "n_components": np.array([d["n_comp"] for d in usable], dtype=float),
    "max_cr": np.array([d["max_cr"] for d in usable], dtype=float),
    "var(cr)": np.array([d["var_cr"] for d in usable]),
    "asym(cr)": np.array([d["asym_cr"] for d in usable], dtype=float),
}
print(f"  {'invariant':<14} {'r vs |γ|':>10} {'r vs γ':>10}")
print("  " + "-" * 40)
for label, v in features.items():
    if np.std(v) > 1e-10:
        r_abs = np.corrcoef(v, abs_gamma)[0, 1]
        r_signed = np.corrcoef(v, gammas)[0, 1]
        flag = "★★" if abs(r_abs) > 0.85 else ("★" if abs(r_abs) > 0.65 else "")
        print(f"  {label:<14} {r_abs:>+9.3f}  {r_signed:>+9.3f}  {flag}")


# ───────── Fit modèles candidats pour γ(L) ─────────
print("\n" + "═"*100)
print("  Fit de modèles candidats γ(L) = f(invariant)")
print("═"*100)

print(f"\n  Modèles à 1 paramètre (γ = C · proxy) :")
print(f"  {'modèle':<35} {'C':>10} {'RMS résid':>10}")
print("  " + "-" * 60)
for label, v in features.items():
    if np.std(v) > 1e-10:
        # Fit y = C·x par moindres carrés
        C = np.dot(abs_gamma, v) / np.dot(v, v)
        residuals = abs_gamma - C * v
        rms = np.sqrt(np.mean(residuals**2))
        flag = "★★" if rms < 0.05 else ("★" if rms < 0.1 else "")
        print(f"  γ = C · {label:<25} {C:>+10.5f} {rms:>10.5f} {flag}")


# ───────── Test du principe variationnel complet ─────────
print("\n" + "═"*100)
print("  Test : principe variationnel complet avec γ(L) = C / m_exp")
print("═"*100)

# Calibrer C tel que γ(μ⁻) reproduise s_6_1 = 1.0436
# γ_μ⁻ extrait : voir tableau
gamma_mu = next(d["gamma"] for d in usable if d["nm"] == "μ⁻")
m_mu = 105.6584
# Si on suppose γ = C/m_exp : C = γ_μ · m_μ
C_mass_inv = gamma_mu * m_mu
print(f"  Hypothèse γ(L) = C / m_exp avec C = {C_mass_inv:.4f}")

# Hypothèse γ ∝ Σ U_κU_τ
sum_U_mu = next(d["sum_U"] for d in usable if d["nm"] == "μ⁻")
C_sumU = gamma_mu / sum_U_mu
print(f"  Hypothèse γ(L) = C · Σ U_κU_τ avec C = {C_sumU:.6e}")

# Hypothèse γ ∝ Σ M_naive
sum_M_mu = next(d["sum_M_naive"] for d in usable if d["nm"] == "μ⁻")
C_sumM = gamma_mu / sum_M_mu
print(f"  Hypothèse γ(L) = C · Σ M_naive avec C = {C_sumM:.6f}")

print(f"\n  Évaluation des trois hypothèses sur le spectre complet :")
print(f"  {'nm':<6} {'γ requis':>9} {'γ=C/m':>9} {'γ=C·ΣU':>10} {'γ=C·ΣM':>10}")
print("  " + "-" * 55)
for d in usable:
    g_req = d["gamma"]
    g_inv_m = C_mass_inv / d["m_exp"]
    g_sumU = C_sumU * d["sum_U"]
    g_sumM = C_sumM * d["sum_M_naive"]
    print(f"  {d['nm']:<6} {g_req:>+9.4f} {g_inv_m:>+9.4f} {g_sumU:>+10.4f} {g_sumM:>+10.4f}")


# ───────── Évaluation finale ─────────
print("\n" + "═"*100)
print("  Application du principe variationnel complet")
print("═"*100)

print(f"\n  γ(L) = C/m_exp avec C = {C_mass_inv:.4f} ; σ(L) = sign rule")
print(f"  {'nm':<6} {'σ':>3} {'γ(L)':>9} {'s_eq':<28} {'m_pred':>9} {'Δ V5':>7} {'Δ s*':>7}")
print("  " + "-" * 90)
rms_v5 = 0; rms_p = 0; n_better = 0
for d in extracted:
    cfg = d["cfg"]; sigma = d["sigma"]
    gamma_pred = C_mass_inv / d["m_exp"]
    m_p, s_eq = m_pred_with_gamma(cfg, sigma, gamma_pred)
    m_v5 = omega_def(cfg, [1.0]*len(cfg))
    d_v5 = (m_v5 - d["m_exp"])/d["m_exp"]*100
    d_p = (m_p - d["m_exp"])/d["m_exp"]*100
    flag = "✓" if abs(d_p) < abs(d_v5) else "✗"
    s_str = ",".join(f"{x:.3f}" for x in s_eq)
    print(f"  {d['nm']:<6} {sigma:>+3} {gamma_pred:>+9.5f} {s_str:<28} "
          f"{m_p:>9.2f} {d_v5:>+6.2f}% {d_p:>+6.2f}% {flag}")
    rms_v5 += d_v5**2
    rms_p += d_p**2
    if abs(d_p) < abs(d_v5): n_better += 1

rms_v5 = float(np.sqrt(rms_v5/len(extracted)))
rms_p = float(np.sqrt(rms_p/len(extracted)))
gain = (rms_v5 - rms_p)/rms_v5 * 100
print(f"\n  RMS V5     = {rms_v5:.3f}%")
print(f"  RMS final  = {rms_p:.3f}%")
print(f"  Gain       = {gain:+.1f}%")
print(f"  Améliorées : {n_better}/{len(extracted)}")
